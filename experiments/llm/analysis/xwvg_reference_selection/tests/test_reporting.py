"""Constructed checks; these are not experimental results."""
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from bca_reference_selection.resample import common_bootstrap_multiplicities, frozen_fit_resampling
from bca_reference_selection.reporting import make_figure, summarize_strategy_results, write_results


def fixture():
    config = dict(model_roster=['target'], families={'family': {}}, budgets=[2], split_seeds=[11, 12],
                  primary_budget=2, secondary_budget=3, primary_protocol='synthetic_test_only',
                  uncertainty=dict(replicates=2, seed=20261107, percentiles=[2.5, 97.5]))
    questions = pd.DataFrame({'base_question_id': list('abcd'), 'category': ['subject'] * 4})
    rows = []
    for split, qds in [(11, [('a', .1), ('b', .3)]), (12, [('b', .5), ('c', .9)])]:
        for q, difference in qds:
            for method, loss in [('collective', .0), ('nearest_k', difference), ('uniform_random_expected', difference / 2)]:
                rows.append(dict(target='target', family='family', split_seed=split, k=2, method=method,
                                 base_question_id=q, tv=loss, top_option_agreement=1 - loss))
    heldout = pd.DataFrame(rows)
    strategy = heldout.groupby(['target', 'family', 'split_seed', 'k', 'method'])[['tv', 'top_option_agreement']].mean().reset_index().rename(columns={'tv': 'evaluation_tv'})
    strategy['selected_peers'] = [['p', 'q']] * len(strategy)
    return config, questions, heldout, strategy


class ResamplingTests(unittest.TestCase):
    def test_overlap_preserves_mean_of_split_means_and_restricts_one_common_draw(self):
        config, questions, losses, _ = fixture()
        mult = np.asarray([[1, 1, 1, 1], [2, 1, 0, 1]])
        replicates, intervals, diagnostics = frozen_fit_resampling(losses, questions, config, multiplicities=mult)
        self.assertAlmostEqual(replicates.iloc[0].nearest_minus_collective_tv, .45)
        self.assertAlmostEqual(replicates.iloc[1].nearest_minus_collective_tv, 1 / 3)
        self.assertAlmostEqual(replicates.iloc[1].random_minus_collective_tv, 1 / 6)
        point = intervals.loc[(intervals.scope == 'macro') & (intervals.metric == 'nearest_minus_collective_tv')].iloc[0]
        self.assertAlmostEqual(point.estimate, .45)
        self.assertEqual(diagnostics['evaluation_appearance_counts'], dict(a=1, b=2, c=1, d=0))
        self.assertFalse(intervals.includes_selection_and_fitting_uncertainty.any())

    def test_zero_split_denominator_is_rejected(self):
        config, questions, losses, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Zero evaluation denominator'):
            frozen_fit_resampling(losses, questions, config, multiplicities=np.asarray([[0, 0, 0, 4], [1, 1, 1, 1]]))

    def test_incomplete_policy_questions_are_rejected(self):
        config, questions, losses, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Inconsistent held-out question identities'):
            frozen_fit_resampling(losses.iloc[1:], questions, config, multiplicities=np.ones((2, 4), dtype=int))

    def test_exact_historical_algorithm(self):
        # Execute the two original small functions in isolation, without their
        # machine-specific imports or any historical pipeline orchestration.
        import ast
        root = Path(__file__).resolve().parents[5]
        source_root = root / 'experiments/llm/analysis/v23/src/pier_llm_v23'
        namespace = dict(np=np, pd=pd, hashlib=hashlib, Any=object)
        for filename, names in [('utils.py', {'stable_u64', 'stable_seed'}), ('core.py', {'common_bootstrap_multiplicities'})]:
            tree = ast.parse((source_root / filename).read_text())
            small = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
            exec(compile(small, filename, 'exec'), namespace)
        selected = pd.DataFrame({'base_question_id': [str(x) for x in range(80)], 'category': ['A'] * 40 + ['B'] * 40})
        for replicate in (1, 2, 2000):
            expected = namespace['common_bootstrap_multiplicities'](selected, replicate, 20261107)
            self.assertEqual(common_bootstrap_multiplicities(selected.sample(frac=1, random_state=3), replicate, 20261107), expected)
            self.assertEqual(sum(expected[str(i)] for i in range(40)), 40)
            self.assertEqual(sum(expected[str(i)] for i in range(40, 80)), 40)

    def test_multiplicity_artifact_and_checksum(self):
        config, questions, losses, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            _, _, diagnostics = frozen_fit_resampling(losses, questions, config, Path(tmp), multiplicities=np.ones((2, 4), dtype=int))
            artifact = Path(tmp) / 'validation/bootstrap_multiplicities.npz'
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), diagnostics['multiplicities_sha256'])
            self.assertEqual(np.load(artifact)['multiplicities'].shape, (2, 4))


class ReportingTests(unittest.TestCase):
    def test_macro_relative_change_is_ratio_of_macro_errors(self):
        config, _, _, strategy = fixture()
        extra = strategy.copy()
        extra.target = 'second'
        extra.loc[extra.method == 'collective', 'evaluation_tv'] = .1
        extra.loc[extra.method == 'nearest_k', 'evaluation_tv'] = .2
        strategy = pd.concat([strategy, extra], ignore_index=True)
        config['model_roster'].append('second')
        cells, macro = summarize_strategy_results(strategy, config)
        self.assertAlmostEqual(macro.iloc[0].nearest_relative_tv_reduction_percent, 100 * (.325 - .05) / .325)
        self.assertNotAlmostEqual(macro.iloc[0].nearest_relative_tv_reduction_percent, cells.nearest_relative_tv_reduction_percent.mean())

    def test_standalone_figure_writes_pdf_and_png(self):
        config, questions, losses, strategy = fixture()
        cells, macro = summarize_strategy_results(strategy, config)
        _, intervals, _ = frozen_fit_resampling(losses, questions, config, multiplicities=np.ones((2, 4), dtype=int))
        with tempfile.TemporaryDirectory() as tmp:
            make_figure(cells, macro, intervals, config, Path(tmp))
            self.assertTrue((Path(tmp) / 'figures/reference_selection.pdf').read_bytes().startswith(b'%PDF'))
            self.assertTrue((Path(tmp) / 'figures/reference_selection.png').read_bytes().startswith(b'\x89PNG'))

    def test_end_to_end_report_preserves_negative_primary_and_positive_secondary(self):
        # This constructed fixture is permitted only in an ephemeral test directory;
        # no report or empirical claim is exposed as an experimental artifact.
        config, questions, heldout, _ = fixture()
        config['budgets'] = [2, 3]
        secondary = heldout.copy()
        secondary['k'] = 3
        heldout = pd.concat([heldout, secondary], ignore_index=True)
        for k, errors in [(2, {'collective': .2, 'nearest_k': .1, 'uniform_random_expected': .15}),
                          (3, {'collective': .1, 'nearest_k': .2, 'uniform_random_expected': .3})]:
            for method, error in errors.items():
                select = (heldout.k == k) & (heldout.method == method)
                heldout.loc[select, 'tv'] = error
                heldout.loc[select, 'top_option_agreement'] = 1 - error
        strategy = heldout.groupby(['target', 'family', 'split_seed', 'k', 'method'])[['tv', 'top_option_agreement']].mean().reset_index().rename(columns={'tv': 'evaluation_tv'})
        strategy['selected_peers'] = [None if method == 'uniform_random_expected' else ['p', 'q'] for method in strategy.method]
        with tempfile.TemporaryDirectory() as tmp:
            # Deterministic all-ones samples isolate reporting from finite-fixture
            # zero-denominator draws; the resampling math is tested separately.
            with patch('bca_reference_selection.resample.common_bootstrap_multiplicities', return_value={q: 1 for q in 'abcd'}):
                result = write_results(strategy, heldout, questions, config, Path(tmp),
                                       run_metadata={'test_context': 'ephemeral constructed fixture only'},
                                       command='synthetic-test-only', complete=True)
            report = (Path(tmp) / 'REPORT.md').read_text()
            draft = (Path(tmp) / 'XWVG_RESPONSE_DRAFT.md').read_text()
            self.assertIn('collective selection increased', report)
            self.assertIn('difference -0.100000', report)
            self.assertIn('0 target/family cells, favors nearest-k in 1', report)
            self.assertIn('do not show a macro-average improvement', draft)
            self.assertIn('TV was 0.200000 for collective selection, 0.100000 for nearest-k', draft)
            self.assertIn('At k=3, the corresponding TV errors were 0.100000, 0.200000 and 0.300000', draft)
            self.assertEqual(result['main_rows'], 2)
            self.assertEqual(result['replicate_rows'], 4)
            self.assertFalse((Path(tmp) / 'status/complete.json').exists())
            macro = pd.read_csv(Path(tmp) / 'macro_results.csv').set_index('k')
            self.assertAlmostEqual(macro.loc[2, 'nearest_minus_collective_tv'], -.1)
            self.assertAlmostEqual(macro.loc[3, 'nearest_minus_collective_tv'], .1)
            for name in ['main_results.csv', 'all_budget_results.csv', 'descriptive_intervals.csv',
                         'selection_frequencies.csv', 'selection_differences.csv',
                         'figures/reference_selection.pdf', 'figures/reference_selection.png']:
                self.assertTrue((Path(tmp) / name).is_file(), name)

    def test_missing_setting_prevents_completed_tables(self):
        config, _, _, strategy = fixture()
        with self.assertRaisesRegex(ValueError, 'All target/family'):
            summarize_strategy_results(strategy.iloc[1:], config)

    def test_no_completed_prose_for_synthetic_or_partial_run(self):
        config, questions, losses, strategy = fixture()
        for flags in [dict(synthetic=True), dict(complete=False), dict(run_metadata={'synthetic': True})]:
            kwargs = dict(run_metadata={}, command='synthetic-test')
            kwargs.update(flags)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'complete real-data run'):
                    write_results(strategy, losses, questions, config, Path(tmp), **kwargs)
                self.assertFalse((Path(tmp) / 'REPORT.md').exists())
                self.assertFalse((Path(tmp) / 'XWVG_RESPONSE_DRAFT.md').exists())


if __name__ == '__main__':
    unittest.main()
