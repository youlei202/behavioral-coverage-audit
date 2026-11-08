"""Descriptive common-question resampling with every learned fit held fixed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ('collective', 'nearest_k', 'uniform_random_expected')
METRICS = ('tv', 'top_option_agreement')
INTERVAL_LABEL = 'frozen_fit_question_resampling_interval'
INTERVAL_SCOPE = (
    'Intervals describe common-question resampling of held-out losses with selected '
    'subsets and fitted weights fixed. They do not include uncertainty from relearning '
    'the selection rules. The ten overlapping splits are not treated as independent replications.'
)


def common_bootstrap_multiplicities(selected: pd.DataFrame, replicate: int, seed: int) -> dict[str, int]:
    """Exact port of V2.3 core.common_bootstrap_multiplicities and stable_seed."""
    unique = selected[['base_question_id', 'category']].drop_duplicates().copy()
    unique['base_question_id'] = unique['base_question_id'].astype(str)
    if unique.base_question_id.duplicated().any():
        raise ValueError('A question belongs to more than one category')
    payload = '\x1f'.join(map(str, (seed, 'common-global', replicate))).encode('utf-8')
    stable_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % 2**32
    rng = np.random.default_rng(stable_seed)
    result = {}
    for _, group in unique.groupby('category', sort=True):
        identifiers = sorted(group.base_question_id.tolist())
        draws = rng.choice(identifiers, size=len(identifiers), replace=True)
        values, counts = np.unique(draws, return_counts=True)
        result.update({identifier: 0 for identifier in identifiers})
        result.update({str(value): int(count) for value, count in zip(values, counts, strict=True)})
    return result


def _contrasts(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = dict(values)
    for baseline, method in [('nearest', 'nearest_k'), ('random', 'uniform_random_expected')]:
        values[f'{baseline}_minus_collective_tv'] = values[f'{method}_tv'] - values['collective_tv']
        values[f'collective_minus_{baseline}_agreement'] = (
            values['collective_top_option_agreement'] - values[f'{method}_top_option_agreement']
        )
    return values


def frozen_fit_resampling(
    heldout_losses: pd.DataFrame,
    questions: pd.DataFrame,
    config: dict,
    output_root: Path | None = None,
    *,
    multiplicities: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return macro replicate metrics, all-cell intervals, and reproducibility diagnostics.

    ``heldout_losses`` has target/family/split_seed/k/method/base_question_id,
    tv and top_option_agreement columns. Question losses must already include
    the four endpoint masses. Optional multiplicities are an explicit test hook.
    """
    losses = heldout_losses.copy()
    required = {'target', 'family', 'split_seed', 'k', 'method', 'base_question_id', *METRICS}
    if not required <= set(losses):
        raise ValueError(f'Missing held-out columns: {sorted(required - set(losses))}')
    questions = questions[['base_question_id', 'category']].copy()
    questions.base_question_id = questions.base_question_id.astype(str)
    if questions.base_question_id.duplicated().any():
        raise ValueError('Duplicate question identifiers')
    qids = sorted(questions.base_question_id)
    qindex = {q: index for index, q in enumerate(qids)}
    losses.base_question_id = losses.base_question_id.astype(str)
    if not set(losses.base_question_id) <= set(qids):
        raise ValueError('Held-out table contains unknown questions')
    key = ['target', 'family', 'split_seed', 'k', 'method', 'base_question_id']
    if losses.duplicated(key).any():
        raise ValueError('Duplicate held-out question losses')
    values = losses[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
        raise ValueError('Held-out metrics must be finite and lie in [0,1]')
    targets = list(config['model_roster'])
    families = list(config['families'])
    budgets = list(config['budgets'])
    splits = list(config['split_seeds'])
    if set(losses.split_seed) != set(splits):
        raise ValueError('Incomplete or unexpected split set')
    cells = pd.MultiIndex.from_product([targets, families, budgets], names=['target', 'family', 'k'])
    columns = pd.MultiIndex.from_product(
        [list(METRICS), targets, families, budgets, list(METHODS)],
        names=[None, 'target', 'family', 'k', 'method'],
    )
    settings = losses[['target', 'family', 'k', 'method']].drop_duplicates()
    if len(settings) != len(cells) * len(METHODS):
        raise ValueError('Incomplete or unexpected policy settings')
    nrep = int(config['uncertainty']['replicates'])
    seed = int(config['uncertainty']['seed'])
    if multiplicities is None:
        multiplicities = np.asarray([
            [counts[q] for q in qids]
            for counts in (common_bootstrap_multiplicities(questions, r, seed) for r in range(1, nrep + 1))
        ], dtype=np.int16)
    else:
        multiplicities = np.asarray(multiplicities)
        if multiplicities.shape != (nrep, len(qids)):
            raise ValueError('Multiplicity dimensions differ from configured replicate/question counts')
        if not np.isfinite(multiplicities).all() or (multiplicities < 0).any() or not np.equal(multiplicities, multiplicities.astype(int)).all():
            raise ValueError('Multiplicities must be finite nonnegative integers')
    for _, group in questions.groupby('category', sort=True):
        indices = [qindex[q] for q in group.base_question_id]
        if not np.equal(multiplicities[:, indices].sum(axis=1), len(indices)).all():
            raise ValueError('Subject multiplicities do not preserve their original count')

    # The leading all-ones row checks the mean-of-split-means point estimator.
    weights = np.vstack([np.ones((1, len(qids))), multiplicities]).astype(float)
    accumulated = np.zeros((nrep + 1, len(columns)), dtype=float)
    appearances = np.zeros(len(qids), dtype=int)
    point_direct = np.zeros(len(columns), dtype=float)
    for split in splits:
        block = losses.loc[losses.split_seed == split]
        matrix = block.pivot(index='base_question_id', columns=['target', 'family', 'k', 'method'], values=list(METRICS))
        if set(matrix.columns) != set(columns):
            raise ValueError(f'Incomplete policy settings in split {split}')
        matrix = matrix.reindex(columns=columns).sort_index()
        if matrix.isna().any().any():
            raise ValueError(f'Inconsistent held-out question identities in split {split}')
        indices = [qindex[q] for q in matrix.index]
        appearances[indices] += 1
        restricted = weights[:, indices]
        denominator = restricted.sum(axis=1)
        if (denominator <= 0).any():
            invalid = np.flatnonzero(denominator <= 0).tolist()
            raise ValueError(f'Zero evaluation denominator in split {split}, replicate IDs {invalid}')
        data = matrix.to_numpy(dtype=float)
        accumulated += (restricted @ data) / denominator[:, None] / len(splits)
        point_direct += data.mean(axis=0) / len(splits)
    point_error = float(np.max(np.abs(accumulated[0] - point_direct)))
    if point_error > 1e-12:
        raise ValueError(f'All-ones resampling failed point-estimate check: {point_error}')
    cube = accumulated.reshape(nrep + 1, len(METRICS), len(targets), len(families), len(budgets), len(METHODS))
    mapped = _contrasts({
        f'{method}_{metric}': cube[:, metric_i, :, :, :, method_i]
        for metric_i, metric in enumerate(METRICS) for method_i, method in enumerate(METHODS)
    })
    macro = {metric: array.mean(axis=(1, 2)) for metric, array in mapped.items()}
    replicate_rows = [
        {'replicate': r, 'k': k, **{metric: float(array[r, ki]) for metric, array in macro.items()}}
        for r in range(1, nrep + 1) for ki, k in enumerate(budgets)
    ]
    low, high = config['uncertainty']['percentiles']
    interval_rows = []
    for metric, array in mapped.items():
        limits = np.percentile(array[1:], [low, high], axis=0)
        for ti, target in enumerate(targets):
            for fi, family in enumerate(families):
                for ki, k in enumerate(budgets):
                    interval_rows.append(dict(scope='target_family', target=target, family=family, k=k,
                        metric=metric, estimate=float(array[0, ti, fi, ki]), lower=float(limits[0, ti, fi, ki]),
                        upper=float(limits[1, ti, fi, ki])))
        limits = np.percentile(macro[metric][1:], [low, high], axis=0)
        for ki, k in enumerate(budgets):
            interval_rows.append(dict(scope='macro', target=None, family=None, k=k, metric=metric,
                estimate=float(macro[metric][0, ki]), lower=float(limits[0, ki]), upper=float(limits[1, ki])))
    intervals = pd.DataFrame(interval_rows)
    intervals['interval_type'] = INTERVAL_LABEL
    intervals['includes_selection_and_fitting_uncertainty'] = False
    intervals['replicates'] = nrep
    diagnostics = dict(
        mode='frozen_fit_common_question_descriptive', seed=seed, replicate_ids=[1, nrep],
        algorithm='V2.3 common_bootstrap_multiplicities, SHA256 stable_seed, sorted category/question IDs, default_rng.choice',
        question_ids=qids, evaluation_appearance_counts=dict(zip(qids, appearances.tolist(), strict=True)),
        all_ones_point_max_abs_error=point_error, includes_selection_and_fitting_uncertainty=False,
        p_values=False, interval_scope=INTERVAL_SCOPE,
    )
    if output_root is not None:
        output_root = Path(output_root)
        (output_root / 'validation').mkdir(parents=True, exist_ok=True)
        path = output_root / 'validation' / 'bootstrap_multiplicities.npz'
        np.savez_compressed(path, question_ids=np.asarray(qids), multiplicities=multiplicities,
                            replicate_ids=np.arange(1, nrep + 1), seed=np.asarray(seed))
        diagnostics['multiplicities_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output_root / 'validation' / 'resampling.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    return pd.DataFrame(replicate_rows), intervals, diagnostics
