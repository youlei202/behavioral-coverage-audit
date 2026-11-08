"""CPU-only checks of shared scoring caps, feasibility gates, and runpy isolation."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'recovery/shared_scoring_runtime.py'
SPEC = importlib.util.spec_from_file_location('shared_runtime', PATH)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
FIXTURE_SPEC = importlib.util.spec_from_file_location('scoring_fixtures', Path(__file__).with_name('test_recovery_scoring.py'))
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


class SharedRuntimeChecks(unittest.TestCase):
    def test_cap_is_absolute70gib_and_reserve_is_at_least16gib(self):
        total = 180 * 1024 ** 3
        config = shared.resource_config(71680, 16384, total)
        self.assertEqual(config['allocator_limit_bytes'], 70 * 1024 ** 3)
        self.assertEqual(config['allocator_fraction'], 70 / 180)
        self.assertEqual(config['batch_size'], 1)
        for budget, reserve, memory in ((71681, 16384, total), (0, 16384, total),
            (71680, 16383, total), (71680, 16384, 80 * 1024 ** 3), (True, 16384, total)):
            with self.subTest(budget=budget, reserve=reserve), self.assertRaises(ValueError):
                shared.resource_config(budget, reserve, memory)

    def test_start_requires_budget_plus_reserve_then_retains_reserve(self):
        config = shared.resource_config(71680, 16384, 180 * 1024 ** 3)
        shared.require_headroom({'free_mib': 88064}, config, before_load=True)
        with self.assertRaises(RuntimeError):
            shared.require_headroom({'free_mib': 88063}, config, before_load=True)
        shared.require_headroom({'free_mib': 16384}, config)
        with self.assertRaises(RuntimeError):
            shared.require_headroom({'free_mib': 16383}, config)

    def test_stress_probe_accounts_for_option_count_and_continuation_length(self):
        prompts = [{'prompt_id': 'long', 'option_count': 2}, {'prompt_id': 'wide', 'option_count': 10}]
        rendered = {'long': {'prompt_token_count': 100}, 'wide': {'prompt_token_count': 20}}
        chosen = shared.largest_workload(prompts, rendered, {2: [1, 1], 10: [4] * 10})
        self.assertEqual(chosen['prompt_id'], 'wide')
        self.assertEqual(chosen['estimated_workload'], 240)
        self.assertTrue(chosen['outcome_blind'])
        for row in prompts:
            row['accuracy'] = 1 if row['prompt_id'] == 'long' else 0
        self.assertEqual(shared.largest_workload(prompts, rendered, {2: [1, 1], 10: [4] * 10}), chosen)

    def test_stress_ties_resolve_by_prompt_identity_not_input_order(self):
        prompts = [{'prompt_id': 'z', 'option_count': 2}, {'prompt_id': 'a', 'option_count': 2}]
        rendered = {pid: {'prompt_token_count': 10} for pid in ('a', 'z')}
        self.assertEqual(shared.largest_workload(prompts, rendered, {2: [3, 3]})['prompt_id'], 'a')

    def args(self):
        return SimpleNamespace(source_repo=Path('/repo'), run_root=Path('/run'), model_id='m',
            model_path=Path('/cache/revision'), device='cuda:0', pilot_batch_size=4)

    def test_original_script_runs_with_batch1_and_original_argv_restored(self):
        original_argv = sys.argv
        observed = []
        def run(path, run_name):
            observed.append((path, run_name, list(sys.argv)))
            raise SystemExit(0)
        with patch.object(shared.runpy, 'run_path', side_effect=run):
            shared.invoke_baseline(self.args())
        self.assertIs(sys.argv, original_argv)
        path, name, argv = observed[0]
        self.assertTrue(path.endswith('/score_endpoints.py'))
        self.assertEqual(name, '__main__')
        self.assertEqual(argv[argv.index('--batch-size') + 1], '1')
        self.assertEqual(argv[argv.index('--stage') + 1], 'score')
        self.assertIn('--resume', argv)

    def test_original_script_failure_cannot_be_promoted_to_complete(self):
        original_argv = sys.argv
        with patch.object(shared.runpy, 'run_path', side_effect=SystemExit(1)):
            with self.assertRaisesRegex(RuntimeError, 'status 1'):
                shared.invoke_baseline(self.args())
        self.assertIs(sys.argv, original_argv)

    def test_watchdog_exits_only_its_injected_own_process_on_breach(self):
        failures, exits = [], []
        watchdog = shared.ReserveWatchdog(lambda: {'free_mib': 16383}, {'reserve_mib': 16384},
            failures.append, terminate=exits.append)
        self.assertFalse(watchdog.check_once())
        self.assertEqual(exits, [86])
        self.assertEqual(failures[0]['resource_failure'], 'reserve_breach')
        self.assertNotIn('other_process_pid', failures[0])

    def test_watchdog_memory_query_error_fails_closed(self):
        failures, exits = [], []
        def broken():
            raise RuntimeError('nvidia-smi unavailable')
        watchdog = shared.ReserveWatchdog(broken, {'reserve_mib': 16384}, failures.append, terminate=exits.append)
        self.assertFalse(watchdog.check_once())
        self.assertEqual(failures[0]['resource_failure'], 'memory_query_failed')
        self.assertEqual(exits, [86])

    def test_watchdog_healthy_memory_never_exits(self):
        exits = []
        watchdog = shared.ReserveWatchdog(lambda: {'free_mib': 50000}, {'reserve_mib': 16384},
            lambda row: self.fail('healthy memory must not emit failure'), terminate=exits.append)
        self.assertTrue(watchdog.check_once())
        self.assertEqual(exits, [])

    def build_proof(self, directory):
        prompts, rendered_rows = fixtures.fixture()
        by_id = {r['prompt_id']: r for r in prompts}
        rendered = {r['prompt_id']: r for r in rendered_rows}
        ids = shared.fixed_pilot_ids(prompts, rendered)
        rows = fixtures.score_rows([by_id[pid] for pid in ids], rendered_rows)
        expected = {'model_id': 'test/model', 'model_revision': 'revision',
                    'memory_budget_mib': 71680, 'reserve_mib': 16384}
        proof = {**expected, 'status': 'passed', 'prompt_count': 42, 'finite_all_outputs': True,
            'worst_case_probe_passed': True, 'immutable_inputs_sources_unchanged': True,
            'exact_equality_to_old_batches_claimed': False,
            'worst_case_probe': {'prompt_id': ids[0]},
            'pilot_rows': shared.atomic_rows(directory / 'pilot.parquet', rows),
            'resource_probe_rows': shared.atomic_rows(directory / 'stress.parquet', [rows[0]])}
        return shared.seal_proof(proof), expected, by_id, rendered, ids

    def test_valid_fresh_pilot_and_stress_artifacts_allow_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            shared.verify_pilot(*self.build_proof(Path(directory)))

    def test_pilot_rejects_changed_cap_failed_stress_and_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            proof, expected, by_id, rendered, ids = self.build_proof(Path(directory))
            with self.assertRaisesRegex(ValueError, 'memory_budget_mib'):
                shared.verify_pilot(proof, dict(expected, memory_budget_mib=60000), by_id, rendered, ids)
            bad = shared.seal_proof(dict(proof, worst_case_probe_passed=False))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                shared.verify_pilot(bad, expected, by_id, rendered, ids)
            with Path(proof['pilot_rows']['path']).open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                shared.verify_pilot(proof, expected, by_id, rendered, ids)

    def test_resealed_nonfinite_output_still_fails_validation(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            proof, expected, by_id, rendered, ids = self.build_proof(Path(directory))
            path = Path(proof['pilot_rows']['path'])
            rows = pd.read_parquet(path).to_dict('records')
            rows[0]['candidate_sequence_log_likelihoods'] = list(rows[0]['candidate_sequence_log_likelihoods'])
            rows[0]['candidate_sequence_log_likelihoods'][0] = float('nan')
            proof['pilot_rows'] = shared.atomic_rows(path, rows)
            proof = shared.seal_proof(proof)
            with self.assertRaisesRegex(ValueError, 'finite'):
                shared.verify_pilot(proof, expected, by_id, rendered, ids)


if __name__ == '__main__':
    unittest.main()
