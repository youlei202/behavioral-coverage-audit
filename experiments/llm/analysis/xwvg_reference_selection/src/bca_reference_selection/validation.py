"""Completion gates that independently recompute saved results from their weights."""
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .select import select_policies, check_training_invariants


def _list(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return list(value)


def _close(actual, expected, label, tolerance=1e-11):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=tolerance):
        raise ValueError(f'Saved result does not reproduce {label}')
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def _index(frame, name, keys, required):
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')
    if frame.duplicated(keys).any():
        raise ValueError(f'Duplicate {name} result keys')
    return {tuple(record[k] for k in keys): record for record in frame.to_dict('records')}


def validate_result_tables(fits, evaluations, strategies, losses, inputs, config, split_map):
    """Reject missing/extra/tampered settings and verify all metrics from saved weights.

    No fitting or selection tuning occurs here. Candidate predictions are rebuilt
    in a batch for each target/family/split. Both policies are reselected from the
    saved, independently verified training errors only.
    """
    settings = list(itertools.product(config['model_roster'], config['families'], config['split_seeds']))
    npeer = len(config['model_roster']) - 1
    subsets = [s for k in config['budgets'] for s in itertools.combinations(range(npeer), k)]
    candidate_keys = {(target, family, seed, subset) for target, family, seed in settings for subset in subsets}
    strategy_keys = {(target, family, seed, k, method) for target, family, seed in settings
                     for k in config['budgets'] for method in config['methods']}
    fits, evaluations, strategies, losses = (frame.copy() for frame in (fits, evaluations, strategies, losses))
    for frame, name in [(fits, 'candidate fits'), (evaluations, 'candidate evaluations')]:
        if not {'subset', 'k'} <= set(frame):
            raise ValueError(f'{name} requires subset and k')
        converted = []
        for raw, k in zip(frame['subset'], frame.k, strict=True):
            value = _list(raw)
            if value is None or any(not isinstance(v, (int, np.integer)) for v in value):
                raise ValueError(f'{name} subset indices must be integers')
            subset = tuple(int(v) for v in value)
            if tuple(sorted(set(subset))) != subset or len(subset) != k or any(v < 0 or v >= npeer for v in subset):
                raise ValueError(f'Invalid {name} subset or budget')
            converted.append(subset)
        frame['subset'] = converted
    candidate_index = ['target', 'family', 'split_seed', 'subset']
    fit_index = _index(fits, 'candidate fits', candidate_index,
                      candidate_index + ['peers', 'weights', 'train_tv', 'status'])
    eval_index = _index(evaluations, 'candidate evaluations', candidate_index,
                       candidate_index + ['evaluation_tv', 'top_option_agreement', 'question_ids', 'per_question_tv', 'per_question_agreement'])
    strategy_index = _index(strategies, 'strategies', ['target', 'family', 'split_seed', 'k', 'method'],
                           ['target', 'family', 'split_seed', 'k', 'method', 'selected_peers', 'selected_weights',
                            'training_tv', 'evaluation_tv', 'top_option_agreement', 'fitting_questions', 'evaluation_questions', 'solver_complete'])
    if set(fit_index) != candidate_keys or set(eval_index) != candidate_keys:
        raise ValueError('Exact exhaustive candidate grid is incomplete or unexpected')
    if set(strategy_index) != strategy_keys:
        raise ValueError('Exact strategy grid is incomplete or unexpected')
    if set(split_map) != set(config['split_seeds']):
        raise ValueError('Split map differs from the locked split seeds')
    if list(inputs.model_ids) != config['model_roster']:
        raise ValueError('Input model ordering differs from the configured roster')
    expected_loss_keys = {(target, family, seed, k, method, str(q)) for target, family, seed, k, method in strategy_keys
                          for q in split_map[seed].evaluation_ids}
    losses.base_question_id = losses.base_question_id.astype(str)
    loss_keys = ['target', 'family', 'split_seed', 'k', 'method', 'base_question_id']
    loss_index = _index(losses, 'held-out questions', loss_keys, loss_keys + ['tv', 'top_option_agreement'])
    if set(loss_index) != expected_loss_keys:
        raise ValueError('Exact held-out question grid is incomplete or unexpected')
    fingerprints = set()
    for frame in (fits, evaluations, strategies):
        if 'input_fingerprint' in frame:
            fingerprints.update(frame.input_fingerprint)
    if len(fingerprints) > 1:
        raise ValueError('Saved results have inconsistent input fingerprints')
    qids = list(map(str, inputs.question_ids))
    q_index = {q: i for i, q in enumerate(qids)}
    if len(q_index) != len(qids):
        raise ValueError('Duplicate aligned input questions')
    worst_training = worst_evaluation = worst_policy = 0.0
    objective_tolerance = config['solver']['objective_recompute_tolerance']
    for target, family, seed in settings:
        target_i = inputs.model_ids.index(target)
        peer_ids = [m for m in inputs.model_ids if m != target]
        peer_indices = [inputs.model_ids.index(m) for m in peer_ids]
        split = split_map[seed]
        if split.fitting_ids & split.evaluation_ids or set(split.fitting_ids) | set(split.evaluation_ids) != set(qids):
            raise ValueError('Original split partitions are invalid')
        train_ids = [q for q in qids if q in split.fitting_ids]
        eval_ids = [q for q in qids if q in split.evaluation_ids]
        local = {}
        for column, subset in enumerate(subsets):
            record = fit_index[(target, family, seed, subset)]
            if _list(record['peers']) != [peer_ids[i] for i in subset]:
                raise ValueError('Candidate peer identities differ from subset indices')
            w = np.asarray(_list(record['weights']), dtype=float)
            if w.shape != (len(subset),) or not np.isfinite(w).all() or w.min() < -1e-12 or abs(w.sum() - 1) > 1e-10:
                raise ValueError('Saved candidate weights violate their simplex')
            if record['status'] != 0:
                raise ValueError('Saved candidate solver is incomplete')
            local[subset] = SimpleNamespace(train_tv=float(record['train_tv']), status=record['status'], weights=w)
        scores = {}
        values = inputs.probabilities[family]
        mass = np.asarray(inputs.masses[family], dtype=float)
        _close(mass, np.asarray([config['families'][family]['clean_mass']] + [config['families'][family]['stress_track_mass']] * 3), 'endpoint masses')
        if abs(mass.sum() - 1) > 1e-12:
            raise ValueError('Question endpoint masses must sum to one')
        for side, identifiers in [('train', train_ids), ('evaluation', eval_ids)]:
            ix = [q_index[q] for q in identifiers]
            truth = values[target_i, ix]
            peer = values[peer_indices][:, ix]
            valid = inputs.mask[ix]
            # Match the original subset coordinate order and einsum arithmetic
            # exactly: alternate BLAS reduction order can change an exact argmax
            # tie even when TV agrees within floating-point roundoff.
            prediction = np.stack([
                np.einsum('nap,p->na',
                    peer[list(subset)].transpose(1, 2, 3, 0).reshape(-1, truth.shape[-1], len(subset)),
                    local[subset].weights).reshape(truth.shape)
                for subset in subsets
            ], axis=-1)
            tv = .5 * np.abs(truth[..., None] - prediction)
            per_question_tv = np.einsum('qoc,o->qc', np.where(valid[..., None], tv, 0).sum(axis=2), mass)
            target_choice = np.argmax(np.where(valid, truth, -np.inf), axis=2)
            prediction_choice = np.argmax(np.where(valid[..., None], prediction, -np.inf), axis=2)
            agreement = np.einsum('qoc,o->qc', (prediction_choice == target_choice[..., None]).astype(float), mass)
            scores[side] = per_question_tv, agreement
        for column, subset in enumerate(subsets):
            record = eval_index[(target, family, seed, subset)]
            raw_ids = _list(record['question_ids'])
            observed_ids = list(map(str, raw_ids)) if raw_ids is not None else []
            if len(observed_ids) != len(eval_ids) or set(observed_ids) != set(eval_ids):
                raise ValueError('Candidate evaluation question identities are incomplete or unexpected')
            order = [observed_ids.index(q) for q in eval_ids]
            saved_tv = np.asarray(_list(record['per_question_tv']), dtype=float)
            saved_agreement = np.asarray(_list(record['per_question_agreement']), dtype=float)
            if saved_tv.shape != (len(eval_ids),) or saved_agreement.shape != (len(eval_ids),):
                raise ValueError('Candidate evaluation arrays do not match question identities')
            expected_tv, expected_agreement = scores['evaluation'][0][:, column], scores['evaluation'][1][:, column]
            worst_evaluation = max(worst_evaluation, _close(saved_tv[order], expected_tv, 'candidate question TV'),
                                   _close(saved_agreement[order], expected_agreement, 'candidate question agreement'),
                                   _close(record['evaluation_tv'], expected_tv.mean(), 'candidate mean TV'),
                                   _close(record['top_option_agreement'], expected_agreement.mean(), 'candidate mean agreement'))
            worst_training = max(worst_training, _close(local[subset].train_tv, scores['train'][0][:, column].mean(),
                                                        'candidate training TV', objective_tolerance))
        selections = select_policies(local, npeer, config['budgets'], config['selection_tie_tolerance'])
        check_training_invariants(local, selections, npeer, objective_tolerance)
        for k in config['budgets']:
            all_columns = [i for i, subset in enumerate(subsets) if len(subset) == k]
            for method in config['methods']:
                key = (target, family, seed, k, method)
                record = strategy_index[key]
                if not record['solver_complete'] or record['fitting_questions'] != len(train_ids) or record['evaluation_questions'] != len(eval_ids):
                    raise ValueError('Strategy completion or split counts disagree')
                if method == 'uniform_random_expected':
                    columns = all_columns
                    if _list(record['selected_peers']) is not None or _list(record['selected_weights']) is not None:
                        raise ValueError('Random expectation must have null selected peers and weights')
                else:
                    subset = selections[k][method]
                    columns = [subsets.index(subset)]
                    if _list(record['selected_peers']) != [peer_ids[i] for i in subset]:
                        raise ValueError('Saved selected peers do not reproduce training-only policy')
                    _close(_list(record['selected_weights']), local[subset].weights, 'selected weights')
                tv = scores['evaluation'][0][:, columns].mean(axis=1)
                agreement = scores['evaluation'][1][:, columns].mean(axis=1)
                train_tv = np.mean([local[subsets[i]].train_tv for i in columns])
                worst_policy = max(worst_policy, _close(record['training_tv'], train_tv, 'strategy training TV'),
                                   _close(record['evaluation_tv'], tv.mean(), 'strategy evaluation TV'),
                                   _close(record['top_option_agreement'], agreement.mean(), 'strategy agreement'))
                observed = [loss_index[(*key, q)] for q in eval_ids]
                _close([r['tv'] for r in observed], tv, 'held-out strategy question TV')
                _close([r['top_option_agreement'] for r in observed], agreement, 'held-out strategy question agreement')
    return dict(passed=True, exact_candidate_grid=True, exact_strategy_grid=True, exact_question_grid=True,
                candidate_count=len(fit_index), candidate_evaluation_count=len(eval_index),
                strategy_count=len(strategy_index), heldout_question_count=len(loss_index),
                candidate_training_recompute_max_abs_error=worst_training,
                candidate_evaluation_recompute_max_abs_error=worst_evaluation,
                strategy_recompute_max_abs_error=worst_policy,
                selections_recomputed_from_training_only=True,
                candidate_metrics_recomputed_from_saved_weights=True,
                random_expectation_recomputed_from_candidate_metrics=True)
