"""CPU-only strict replay gates; no GPU, SSH, or real model loading."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[1] / 'recovery/validate_remote_reference.py'
SPEC = importlib.util.spec_from_file_location('remote_reference', PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class RemoteReferenceChecks(unittest.TestCase):
    def pilot(self):
        ids = [f'p{i}' for i in range(42)]
        batches = [ids[i:i + 4] for i in range(0, 42, 4)]
        pilot = {'input_fingerprint': 'identity', 'status': 'complete',
                 'pilot_measured_prompt_count': 42, 'scoring_seconds': 2,
                 'batches': [{'prompt_ids': batch, 'batch_size': len(batch)} for batch in batches]}
        design = {'input_fingerprint': 'identity', 'row_count': 42,
                  'outcome_blind': True, 'prompt_ids': ids}
        return ids, batches, pilot, design

    def test_exact_saved_grouping_preserved(self):
        ids, batches, pilot, design = self.pilot()
        self.assertEqual(module.saved_pilot_batches(pilot, design, 'identity', ids), batches)

    def test_duplicate_missing_or_regrouped_metadata_fails(self):
        ids, batches, pilot, design = self.pilot()
        for mutation in ('duplicate', 'missing', 'batch_count', 'wrong_identity'):
            changed = copy.deepcopy(pilot)
            if mutation == 'duplicate':
                changed['batches'][-1]['prompt_ids'][-1] = ids[0]
            elif mutation == 'missing':
                changed['batches'].pop()
            elif mutation == 'batch_count':
                changed['batches'][0]['batch_size'] = 2
            else:
                changed['input_fingerprint'] = 'other'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.saved_pilot_batches(changed, design, 'identity', ids)

    def test_selection_is_recomputed_not_trusted_from_saved_design(self):
        ids, _, pilot, design = self.pilot()
        altered = ids.copy()
        altered[-1] = 'unselected'
        with self.assertRaises(ValueError):
            module.saved_pilot_batches(pilot, design, 'identity', altered)

    def test_runtime_hardware_source_and_weight_changes_rejected(self):
        current = {'versions': {'torch': '2.9.1'}, 'hardware': {'name': 'B200'},
                   'source_sha256': {'inference.py': 'source'},
                   'model_inventory': {'files': {'weights': 'sha'}}}
        baseline = dict(current, input_fingerprint=hashlib.sha256(module.canonical(current)).hexdigest())
        self.assertEqual(module.require_identity(baseline, current), baseline['input_fingerprint'])
        for key in current:
            altered = dict(current, **{key: {'changed': True}})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs'):
                module.require_identity(baseline, altered)
        corrupted = dict(baseline, input_fingerprint='wrong')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            module.require_identity(corrupted, current)

    def test_complete_rows_include_every_saved_field_and_reject_one_ulp(self):
        actual = {'prompt_id': 'p', 'track': 1, 'candidate_sequence_log_likelihoods': [-1.25]}
        stored = dict(actual, track=1.0, model_id='m', model_revision='r',
                      score_origin='fresh_endpoint_scoring', manifest_sha256='h')
        self.assertTrue(module.compare_complete_rows([stored], [actual], 'm', 'r', 'h')['exact'])
        changed = dict(actual, candidate_sequence_log_likelihoods=[-1.2500000000000002])
        self.assertFalse(module.compare_complete_rows([stored], [changed], 'm', 'r', 'h')['exact'])
        # Do not project saved rows onto actual keys: lost scientific fields fail.
        missing = {k: v for k, v in actual.items() if k != 'candidate_sequence_log_likelihoods'}
        self.assertFalse(module.compare_complete_rows([stored], [missing], 'm', 'r', 'h')['exact'])
        self.assertFalse(module.compare_complete_rows([dict(stored, nullable=None)], [actual], 'm', 'r', 'h')['exact'])

    def test_replay_uses_exact_batch_order_without_warmup(self):
        batches = [['p2', 'p0'], ['p1']]
        by_id = {pid: {'prompt_id': pid, 'track': None} for pid in ('p0', 'p1', 'p2')}
        stored = {pid: dict(row, model_id='m', model_revision='r',
                  score_origin='fresh_endpoint_scoring', manifest_sha256='h') for pid, row in by_id.items()}
        calls, recorded = [], []
        def score(rows):
            calls.append([row['prompt_id'] for row in rows])
            return rows, {'seconds': 0.1}
        report = module.replay_batches(batches, by_id, stored, score, 'm', 'r', 'h', recorded.append)
        self.assertEqual(calls, batches)
        self.assertEqual(recorded, report)
        self.assertTrue(all(row['comparison']['exact'] for row in report))

    def test_oom_aborts_without_resizing_or_retry(self):
        calls = []
        def score(rows):
            calls.append(rows)
            raise RuntimeError('CUDA OOM')
        with self.assertRaisesRegex(RuntimeError, 'OOM'):
            module.replay_batches([['p0', 'p1'], ['p2']],
                {pid: {'prompt_id': pid} for pid in ('p0', 'p1', 'p2')}, {},
                score, 'm', 'r', 'h', lambda _: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)

    def test_output_cannot_overwrite_baseline_sources_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root, snapshot = repo / 'outputs/run', repo / 'cache/model'
            for path in (root / 'data/a.json', root / 'outputs/raw_scores/a.json',
                         snapshot / 'a.json', repo / 'experiments/a.json',
                         root / 'status/fresh_score_model_pilot.json'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    module.checked_output(path, repo, root, snapshot)
            proof = root / 'status/distributed_remote_proofs/token/model.json'
            self.assertEqual(module.checked_output(proof, repo, root, snapshot), proof)
            proof.parent.mkdir(parents=True)
            proof.write_text('{}')
            with self.assertRaises(ValueError):
                module.checked_output(proof, repo, root, snapshot)

    def test_proof_digest_covers_failure_and_every_batch(self):
        report = {'status': 'failed', 'batches': [{'comparison': {'exact': False}}]}
        sealed = module.seal_proof(report)
        self.assertEqual(sealed['proof_sha256'], hashlib.sha256(module.canonical(report)).hexdigest())
        changed = dict(report, status='passed')
        self.assertNotEqual(module.seal_proof(changed)['proof_sha256'], sealed['proof_sha256'])

    def test_gpu_uuid_uses_actual_cuda_mapping_not_device_ordinal(self):
        uuid = 'f644e06d-4b94-edfb-1e3e-5287d4d7379d'
        props = SimpleNamespace(uuid=uuid, total_memory=191545606144)
        fake = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda device: props,
            get_device_name=lambda device: 'NVIDIA B200', get_device_capability=lambda device: (10, 0)),
            version=SimpleNamespace(cuda='12.8'))
        answer = SimpleNamespace(stdout=f'GPU-00000000-0000-0000-0000-000000000000, 615.1\nGPU-{uuid}, 615.71.10\n')
        with patch.object(module.subprocess, 'run', return_value=answer):
            hardware, observed, driver = module.gpu_details(fake, 'cuda:0')
        self.assertEqual(observed, 'GPU-' + uuid)
        self.assertEqual(driver, '615.71.10')
        self.assertEqual(hardware['capability'], [10, 0])


if __name__ == '__main__':
    unittest.main()
