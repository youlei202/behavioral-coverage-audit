"""Result tampering/completeness tests using tiny constructed fitted data."""
import copy
from itertools import combinations
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from bca_reference_selection.fit import fit_tv_simplex
from bca_reference_selection.evaluate import evaluate_candidate
from bca_reference_selection.select import select_policies
from bca_reference_selection.splits import Split
from bca_reference_selection.validation import validate_result_tables


def fixture():
    models = ['a', 'b', 'c']
    config = dict(model_roster=models, families={'family': dict(clean_mass=.5, stress_track_mass=1 / 6)},
                  split_seeds=[1], budgets=[1, 2], methods=['collective', 'nearest_k', 'uniform_random_expected'],
                  selection_tie_tolerance=1e-9, solver={'objective_recompute_tolerance': 1e-7})
    rng = np.random.default_rng(1241)
    p = rng.dirichlet(np.ones(3), size=(3, 4, 4))
    qids = ['q0', 'q1', 'q2', 'q3']
    inputs = SimpleNamespace(model_ids=models, question_ids=qids, probabilities={'family': p},
                             mask=np.ones((4, 4, 3), bool), masses={'family': np.array([.5, 1 / 6, 1 / 6, 1 / 6])})
    split_map = {1: Split(1, frozenset(qids[:2]), frozenset(qids[2:]))}
    fits, evaluations, strategies, losses = [], [], [], []
    for target_i, target in enumerate(models):
        peers = [i for i in range(3) if i != target_i]
        peer_ids = [models[i] for i in peers]
        x = p[peers].transpose(1, 2, 3, 0)
        y = p[target_i]
        mask = np.ones((8, 3), bool)
        mass = np.tile(inputs.masses['family'], 2)
        local = {}
        ev = {}
        for k in config['budgets']:
            for subset in combinations(range(2), k):
                # Explicit take preserves NumPy advanced-index dimension order.
                fit = fit_tv_simplex(y[:2].reshape(8, 3), np.take(x[:2], subset, axis=-1).reshape(8, 3, k), mask, mass)
                local[subset] = fit
                key = dict(target=target, family='family', split_seed=1, subset=list(subset), k=k)
                fits.append({**key, **fit.to_record(), 'peers': [peer_ids[i] for i in subset]})
                result = evaluate_candidate(fit, subset, y[2:].reshape(8, 3), x[2:].reshape(8, 3, 2), mask, mass,
                                            question_ids=np.repeat(qids[2:], 4))
                ev[subset] = result
                evaluations.append({**key, 'evaluation_tv': result['evaluation_tv'], 'top_option_agreement': result['top_option_agreement'],
                                    'question_ids': result['question_ids'], 'per_question_tv': result['question_tv'].tolist(),
                                    'per_question_agreement': result['question_top_option_agreement'].tolist()})
        selections = select_policies(local, 2, [1, 2], 1e-9)
        for k in config['budgets']:
            for method in config['methods']:
                subset = None if method == 'uniform_random_expected' else selections[k][method]
                selected = [s for s in local if len(s) == k] if subset is None else [subset]
                tv = np.mean([ev[s]['question_tv'] for s in selected], axis=0)
                agreement = np.mean([ev[s]['question_top_option_agreement'] for s in selected], axis=0)
                key = dict(target=target, family='family', split_seed=1, k=k, method=method)
                strategies.append({**key, 'selected_peers': None if subset is None else [peer_ids[i] for i in subset],
                                   'selected_weights': None if subset is None else local[subset].weights.tolist(),
                                   'training_tv': np.mean([local[s].train_tv for s in selected]),
                                   'evaluation_tv': tv.mean(), 'top_option_agreement': agreement.mean(),
                                   'fitting_questions': 2, 'evaluation_questions': 2, 'solver_complete': True})
                for q, loss, agr in zip(qids[2:], tv, agreement, strict=True):
                    losses.append({**key, 'base_question_id': q, 'tv': loss, 'top_option_agreement': agr})
    return [pd.DataFrame(x) for x in [fits, evaluations, strategies, losses]], inputs, config, split_map


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables, cls.inputs, cls.config, cls.splits = fixture()

    def check(self, tables):
        return validate_result_tables(*tables, self.inputs, self.config, self.splits)

    def test_complete_results_reproduce_raw_weights_and_roundtrip_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            roundtripped = []
            for i, table in enumerate(self.tables):
                path = Path(tmp) / f'{i}.parquet'
                table.to_parquet(path)
                roundtripped.append(pd.read_parquet(path))
            result = self.check(roundtripped)
        self.assertTrue(result['passed'])
        self.assertTrue(result['candidate_metrics_recomputed_from_saved_weights'])
        self.assertEqual(result['candidate_count'], 9)
        self.assertEqual(result['candidate_evaluation_count'], 9)

    def test_duplicate_replacing_candidate_with_same_total_count_rejected(self):
        tables = [t.copy(deep=True) for t in self.tables]
        tables[1] = pd.concat([tables[1].iloc[:-1], tables[1].iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate evaluations'):
            self.check(tables)

    def test_missing_candidate_evaluation_rejected(self):
        tables = [t.copy(deep=True) for t in self.tables]
        tables[1] = tables[1].iloc[:-1]
        with self.assertRaisesRegex(ValueError, 'candidate grid'):
            self.check(tables)

    def test_self_consistent_candidate_metrics_still_must_match_saved_weights(self):
        tables = [copy.deepcopy(t) for t in self.tables]
        table = tables[1]
        table.at[0, 'per_question_tv'] = [float(v) + .01 for v in table.at[0, 'per_question_tv']]
        table.at[0, 'evaluation_tv'] += .01
        with self.assertRaisesRegex(ValueError, 'candidate question TV'):
            self.check(tables)

    def test_changed_policy_weights_rejected(self):
        tables = [t.copy(deep=True) for t in self.tables]
        row = tables[2].index[(tables[2].k == 2) & (tables[2].method == 'collective')][0]
        tables[2].at[row, 'selected_weights'] = [.41, .59]
        with self.assertRaisesRegex(ValueError, 'selected weights'):
            self.check(tables)

    def test_random_expected_metrics_recomputed_from_candidate_losses(self):
        tables = [t.copy(deep=True) for t in self.tables]
        row = tables[2].index[tables[2].method == 'uniform_random_expected'][0]
        tables[2].at[row, 'evaluation_tv'] += .01
        with self.assertRaisesRegex(ValueError, 'strategy evaluation TV'):
            self.check(tables)

    def test_missing_question_replaced_by_wrong_side_is_rejected(self):
        tables = [t.copy(deep=True) for t in self.tables]
        tables[3].at[0, 'base_question_id'] = 'q0'
        with self.assertRaisesRegex(ValueError, 'question grid'):
            self.check(tables)


if __name__ == '__main__':
    unittest.main()
