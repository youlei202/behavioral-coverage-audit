"""Final fresh-cache provenance, reproducible-command and figure audit checks."""
import copy
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bca_reference_selection.pipeline import reproduction_command, validate_probability_provenance
from bca_reference_selection.reporting import (
    make_figure, measurement_provenance_text, summarize_strategy_results, write_results,
)
from bca_reference_selection.resample import frozen_fit_resampling
from test_reporting import fixture


def shared_provenance():
    """Constructed execution metadata only; no model measurements are claimed."""
    change = {'model_id': 'fixture-model', 'status': 'complete', 'hostname': 'fixture-shared-node',
              'batch_size': 1, 'memory_budget_mib': 71680, 'reserve_mib': 16384,
              'new_committed_row_count': 10, 'new_committed_shard_count': 1}
    return {'measurement_origin': 'fresh_endpoint_scoring', 'precision': 'bfloat16',
            'candidate_scoring': 'summed_full_continuation_log_likelihood',
            'input_identity': {
                'shards': [{'data': {'path': path}, 'execution': {'effective_batch_size': batch}}
                           for path, batch in [('retained-pilot', 4), ('retained-local', 2), ('shared', 1)]],
                'distributed_execution_provenance': {
                    'shared_batch_changes': [change, {**change, 'model_id': 'second-fixture-model'}],
                    'shared_new_shard_attribution': {'shared': ['sealed-fixture-proof']},
                    'committed_batch_sizes_by_model': {'fixture-model': {'effective_batch_size': [1, 2, 4]}}}}}


class FinalReportingAuditChecks(unittest.TestCase):
    def test_pasteable_command_imports_without_inherited_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            command = reproduction_command(['--repo-root', str(Path(__file__).resolve().parents[5]),
                                            '--output-root', str(Path(temporary)/'space in path')])
            argv = shlex.split(command)
            self.assertEqual(argv[0], 'env')
            self.assertTrue(argv[1].startswith('PYTHONPATH='))
            self.assertEqual(argv.count('--resume'), 1)
            environment = dict(os.environ)
            environment.pop('PYTHONPATH', None)
            result = subprocess.run([*argv, '--help'], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--input-provenance', result.stdout)
            self.assertEqual(shlex.split(reproduction_command(['--resume'])).count('--resume'), 1)

    def test_probability_provenance_must_match_actual_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)/'input.parquet'
            cache.write_bytes(b'identity fixture, not actual probabilities')
            actual = hashlib.sha256(cache.read_bytes()).hexdigest()
            provenance = {'measurement_origin': 'fresh_endpoint_scoring', 'fresh_measurements': True,
                          'historical_cache_restoration': False,
                          'averaged_cache': {'path': 'old/relocated/path', 'sha256': actual, 'bytes': cache.stat().st_size}}
            result = validate_probability_provenance(provenance, {'cache': cache})
            self.assertTrue(result['verified'])
            self.assertEqual(result['sha256'], actual)
            wrong = copy.deepcopy(provenance)
            wrong['averaged_cache']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA-256 does not match'):
                validate_probability_provenance(wrong, {'cache': cache})
            wrong = copy.deepcopy(provenance)
            wrong['averaged_cache']['bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'size does not match'):
                validate_probability_provenance(wrong, {'cache': cache})
            with self.assertRaisesRegex(ValueError, 'must identify averaged_cache'):
                validate_probability_provenance({}, {'cache': cache})
            wrong = copy.deepcopy(provenance)
            wrong['historical_cache_restoration'] = True
            with self.assertRaisesRegex(ValueError, 'inconsistent origin flags'):
                validate_probability_provenance(wrong, {'cache': cache})

    def test_figure_places_budgets_at_their_actual_numeric_values(self):
        import matplotlib.axes
        config, questions, heldout, _ = fixture()
        config['budgets'] = [1, 2, 3, 7]
        frames = []
        for budget in config['budgets']:
            frame = heldout.copy()
            frame['k'] = budget
            frames.append(frame)
        heldout = pd.concat(frames, ignore_index=True)
        strategy = heldout.groupby(['target', 'family', 'split_seed', 'k', 'method'])[['tv', 'top_option_agreement']].mean().reset_index().rename(columns={'tv': 'evaluation_tv'})
        cells, macro = summarize_strategy_results(strategy, config)
        _, intervals, _ = frozen_fit_resampling(heldout, questions, config, multiplicities=np.ones((2, 4), dtype=int))
        original = matplotlib.axes.Axes.plot
        observed = []
        def record_plot(axis, *args, **kwargs):
            observed.append(np.asarray(args[0]).copy())
            return original(axis, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary, patch.object(matplotlib.axes.Axes, 'plot', record_plot):
            make_figure(cells, macro, intervals, config, Path(temporary))
        np.testing.assert_allclose(observed[1], [1, 2, 3, 7])
        self.assertAlmostEqual(observed[1][-1] - observed[1][-2], 4)

    def test_tiny_positive_difference_has_observational_wording_and_fresh_disclosure(self):
        config, questions, heldout, _ = fixture()
        config['budgets'] = [2, 3]
        secondary = heldout.copy(); secondary['k'] = 3
        heldout = pd.concat([heldout, secondary], ignore_index=True)
        for method, error in [('collective', .1), ('nearest_k', .1 + 1e-10), ('uniform_random_expected', .2)]:
            heldout.loc[heldout.method == method, 'tv'] = error
        strategy = heldout.groupby(['target', 'family', 'split_seed', 'k', 'method'])[['tv', 'top_option_agreement']].mean().reset_index().rename(columns={'tv': 'evaluation_tv'})
        strategy['selected_peers'] = [None if method == 'uniform_random_expected' else ['p', 'q'] for method in strategy.method]
        metadata = {'input_measurement_provenance': {'measurement_origin': 'fresh_endpoint_scoring'},
                    'input_provenance_validation': {'verified': True},
                    'test_context': 'ephemeral constructed fixture only'}
        with tempfile.TemporaryDirectory() as temporary:
            with patch('bca_reference_selection.resample.common_bootstrap_multiplicities', return_value={q: 1 for q in 'abcd'}):
                write_results(strategy, heldout, questions, config, Path(temporary), run_metadata=metadata, command='test-only')
            draft = (Path(temporary)/'XWVG_RESPONSE_DRAFT.md').read_text()
            report = (Path(temporary)/'REPORT.md').read_text()
            for text in [report, draft]:
                self.assertIn('fresh measurements', text)
                self.assertIn('not recovered historical score bytes', text)
                self.assertNotIn('support this concrete use', text)
                self.assertNotIn('shared-node', text)
            self.assertIn('lower observed macro-average TV', draft)
            self.assertIn('absolute magnitude', draft)
            with self.assertRaisesRegex(ValueError, 'cache-bound provenance'):
                write_results(strategy, heldout, questions, config, Path(temporary),
                              run_metadata={'input_measurement_provenance': metadata['input_measurement_provenance']}, command='test-only')

    def test_shared_execution_paragraph_appears_in_both_deliverables(self):
        config, questions, heldout, _ = fixture()
        config['budgets'] = [2, 3]
        heldout = pd.concat([heldout, heldout.assign(k=3)], ignore_index=True)
        strategy = heldout.groupby(['target', 'family', 'split_seed', 'k', 'method'])[['tv', 'top_option_agreement']].mean().reset_index().rename(columns={'tv': 'evaluation_tv'})
        strategy['selected_peers'] = [['p', 'q']] * len(strategy)
        metadata = {'input_measurement_provenance': shared_provenance(),
                    'input_provenance_validation': {'verified': True}}
        with tempfile.TemporaryDirectory() as temporary:
            with patch('bca_reference_selection.resample.common_bootstrap_multiplicities', return_value={q: 1 for q in 'abcd'}):
                write_results(strategy, heldout, questions, config, Path(temporary),
                              run_metadata=metadata, command='test-only')
            for filename in ('REPORT.md', 'XWVG_RESPONSE_DRAFT.md'):
                text = (Path(temporary) / filename).read_text()
                self.assertIn('same original full-continuation scorer was used in BF16', text)
                self.assertIn('retained pilot/local batch settings 2 and 4', text)
                self.assertIn('shared-node batch settings 1', text)
                self.assertIn('1 shared node; 70 GiB allocator cap; 16 GiB reserve', text)
                self.assertIn('No bitwise equivalence across these batch groupings is claimed.', text)
                self.assertEqual(text.count('same original full-continuation scorer'), 1)

    def test_disclosure_uses_actual_settings_and_excludes_failed_pilot_only_runs(self):
        provenance = shared_provenance()
        execution = provenance['input_identity']['distributed_execution_provenance']
        execution['shared_batch_changes'][1].update(hostname='second-shared-node', batch_size=3,
                                                   memory_budget_mib=98304, reserve_mib=8192)
        for row in provenance['input_identity']['shards'][:2]:
            row['execution']['effective_batch_size'] = 5
        text = measurement_provenance_text(provenance)
        self.assertIn('retained pilot/local batch settings 5', text)
        self.assertIn('shared-node batch settings 1 and 3', text)
        self.assertIn('2 shared nodes; 70 and 96 GiB allocator caps; 8 and 16 GiB reserves', text)
        for row in execution['shared_batch_changes']:
            row.update(status='failed', new_committed_row_count=0, new_committed_shard_count=0)
        serial = measurement_provenance_text({'measurement_origin': 'fresh_endpoint_scoring'})
        self.assertEqual(measurement_provenance_text(provenance), serial)
        self.assertNotIn('BF16', measurement_provenance_text(None))


if __name__ == '__main__':
    unittest.main()
