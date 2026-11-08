"""Multi-host execution evidence is frozen and tied to actual committed bytes."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from test_recovery_assembly import COMPONENT, assembly, fixture, score


def load(path):
    return json.loads(path.read_text())


def seal(record):
    record = {k: v for k, v in record.items() if k != 'proof_sha256'}
    return {**record, 'proof_sha256': hashlib.sha256(score.canonical(record)).hexdigest()}


def distributed_fixture(root, config):
    """A fabricated CPU-only lineage with real source identities, never GPU scores."""
    model = config['model_roster'][0]
    name = score.slug(model)
    identity_path = root / f'status/fresh_score_{name}_identity.json'
    identity = load(identity_path)
    identity.pop('input_fingerprint')
    repo = COMPONENT.parents[3]
    identity['source_sha256'] = {p: score.file_sha256(repo / p) for p in score.SOURCE_FILES}
    identity['driver_sha256'] = score.file_sha256(COMPONENT / 'recovery/score_endpoints.py')
    identity['hardware']['total_memory_bytes'] = 180 * 1024**3
    identity['input_fingerprint'] = hashlib.sha256(score.canonical(identity)).hexdigest()
    score.atomic_json(identity_path, identity)
    data_path = next((root / 'outputs/raw_scores' / name).glob('*.parquet'))
    rows = pd.read_parquet(data_path)
    meta_path, commit_path = data_path.with_suffix('.meta.json'), data_path.with_suffix('.complete.json')
    meta, commit = load(meta_path), load(commit_path)
    meta.update(input_fingerprint=identity['input_fingerprint'], execution={
        'stage': 'score', 'requested_batch_size': 1, 'effective_batch_size': 1,
        'batches': [{'batch_size': 1, 'prompt_ids': [p]} for p in meta['prompt_ids']]})
    commit['input_fingerprint'] = identity['input_fingerprint']
    score.atomic_json(meta_path, meta)
    score.atomic_json(commit_path, commit)
    complete_path = root / f'status/fresh_score_{name}_complete.json'
    complete = load(complete_path)
    complete.update(input_fingerprint=identity['input_fingerprint'], requested_batch_size=1, final_batch_size=1)
    score.atomic_json(complete_path, complete)

    pilot_path = root / 'evidence' / f'{name}_pilot.json'
    pilot_path.parent.mkdir()
    artifacts = {}
    for key, frame in [('pilot_rows', rows.iloc[:42]), ('resource_probe_rows', rows.iloc[-1:])]:
        path = pilot_path.with_name(f'{name}_{key}.parquet')
        frame.to_parquet(path, index=False)
        artifacts[key] = {'path': str(path), 'sha256': score.file_sha256(path), 'row_count': len(frame)}
    resource = {'batch_size': 1, 'memory_budget_mib': 71680, 'reserve_mib': 16384,
                'allocator_limit_bytes': 71680 * 1024**2}
    common = {'schema_version': 'xwvg_shared_scoring_execution_v1',
        'model_id': model, 'model_revision': identity['model_revision'],
        'input_fingerprint': identity['input_fingerprint'], 'baseline_identity': identity,
        'versions': identity['versions'], 'hardware': identity['hardware'],
        'source_sha256': identity['source_sha256'], 'baseline_driver_sha256': identity['driver_sha256'],
        'wrapper_sha256': score.file_sha256(COMPONENT / 'recovery/shared_scoring_runtime.py'),
        'hostname': 'fabricated-cpu-test-host', 'gpu_uuid': 'fabricated-never-used-gpu',
        'driver_version': 'fabricated', 'batch_size': 1, 'memory_budget_mib': 71680,
        'reserve_mib': 16384, 'resource_config': resource,
        'exact_equality_to_old_batches_claimed': False,
        'batch_change_reason': 'Fabricated CPU fixture for explicit batch/resource provenance',
        'immutable_inputs_sources_unchanged': True, 'finite_all_outputs': True,
        'worst_case_probe_passed': True}
    execution_keys = ('wrapper_sha256', 'baseline_driver_sha256', 'input_fingerprint', 'hostname',
                      'gpu_uuid', 'driver_version', 'resource_config', 'versions', 'hardware')
    common['execution_fingerprint'] = assembly._content_hash({k: common[k] for k in execution_keys})
    pilot = seal({**common, 'stage': 'pilot', 'status': 'passed', 'prompt_count': 42,
                  'baseline_score_outputs_written': False, **artifacts})
    score.atomic_json(pilot_path, pilot)
    attributed = {'sha256': score.file_sha256(data_path), 'meta_sha256': score.file_sha256(meta_path),
        'commit_sha256': score.file_sha256(commit_path), 'row_count': len(rows),
        'input_fingerprint': identity['input_fingerprint'], 'execution': meta['execution']}
    score_path = root / 'status/distributed_shared/current_owner' / f'{name}_score.json'
    execution = seal({**common, 'stage': 'score', 'status': 'complete', 'prompt_count': len(rows),
        'pilot_proof': {'path': str(pilot_path), 'sha256': score.file_sha256(pilot_path)},
        'committed_shards_before': {}, 'new_committed_shards': {str(data_path): attributed},
        'existing_committed_shards_unchanged': True,
        'baseline_complete_status_path': str(complete_path),
        'baseline_complete_status_sha256': score.file_sha256(complete_path)})
    score.atomic_json(score_path, execution)
    old_path = root / 'status/distributed_shared/older_owner/failed_pilot.json'
    score.atomic_json(old_path, seal({**common, 'stage': 'pilot', 'status': 'failed',
        'prompt_count': 0, 'finite_all_outputs': False, 'error': 'Fabricated earlier failed attempt'}))
    controller = root / 'status/distributed_campaign.json'
    score.atomic_json(controller, {'status': 'finishing', 'hostname': 'fabricated-controller',
                                  'slots': [{'id': 'shared', 'shared_pilot_proofs': {model: str(pilot_path)}}]})
    worker = root / f'status/distributed_workers/{name}.json'
    score.atomic_json(worker, {'status': 'complete', 'model_id': model, 'shared_execution': {
        'batch_size': 1, 'cross_batch_bitwise_equality_claimed': False, 'proofs': [
            {'stage': 'pilot', 'path': str(pilot_path), 'sha256': score.file_sha256(pilot_path)},
            {'stage': 'score', 'path': str(score_path), 'sha256': score.file_sha256(score_path)}]}})
    return {'pilot': pilot_path, 'score': score_path, 'controller': controller, 'worker': worker,
            'old': old_path, 'data': data_path, 'pilot_rows': Path(artifacts['pilot_rows']['path'])}


def rewrite_execution(paths, record):
    score.atomic_json(paths['score'], seal(record))
    worker = load(paths['worker'])
    worker['shared_execution']['proofs'][-1]['sha256'] = score.file_sha256(paths['score'])
    score.atomic_json(paths['worker'], worker)


class DistributedAssemblyChecks(unittest.TestCase):
    def test_snapshot_survives_live_status_transition_and_retains_failed_attempts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, questions = fixture(root)
            paths = distributed_fixture(root, config)
            result = assembly.assemble(root, questions, config)
            provenance = load(root / 'status/measurement_provenance.json')
            evidence = provenance['input_identity']['distributed_execution_provenance']
            snapshot_path = Path(evidence['snapshot_file']['path'])
            original_bytes = snapshot_path.read_bytes()
            snapshot = load(snapshot_path)
            captured = {r['observed_source_path']: r['content'] for r in snapshot['records']}
            self.assertEqual(captured[str(paths['controller'])]['status'], 'finishing')
            self.assertEqual(captured[str(paths['old'])]['status'], 'failed')
            change = evidence['shared_batch_changes'][0]
            self.assertEqual(change['new_committed_row_count'], 63)
            self.assertEqual(change['memory_budget_mib'], 71680)
            self.assertEqual(change['reserve_mib'], 16384)
            self.assertFalse(evidence['cross_batch_bitwise_equality_claimed'])
            self.assertFalse(evidence['historical_score_bytes_restored'])
            self.assertEqual(evidence['committed_batch_sizes_by_model'][config['model_roster'][0]]['effective_batch_size'], [1])
            self.assertEqual(evidence['shared_new_shard_attribution'][str(paths['data'])], [str(paths['score'])])
            score.atomic_json(paths['controller'], {'status': 'complete', 'completed_utc': 'later'})
            score.atomic_json(paths['worker'], {'status': 'complete', 'updated_utc': 'later'})
            again = assembly.assemble(root, questions, config)
            self.assertEqual(again['invocation_status'], 'reused_verified_complete_cache')
            self.assertEqual(again['assembly_fingerprint'], result['assembly_fingerprint'])
            self.assertEqual(snapshot_path.read_bytes(), original_bytes)

    def test_tampered_proof_is_rejected_before_cache_publication(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, questions = fixture(root)
            paths = distributed_fixture(root, config)
            proof = load(paths['score']); proof['memory_budget_mib'] = 1
            score.atomic_json(paths['score'], proof)
            with self.assertRaisesRegex(ValueError, 'proof checksum mismatch'):
                assembly.assemble(root, questions, config)
            self.assertFalse((root / 'outputs/analysis/permutation_averaged_semantic_scores.parquet').exists())

    def test_resealed_scientific_source_and_shard_attribution_tampering_rejected(self):
        for kind in ('source', 'shard', 'batch'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                config, questions = fixture(root)
                paths = distributed_fixture(root, config)
                execution = load(paths['score'])
                if kind == 'source':
                    execution['source_sha256'][score.SOURCE_FILES[0]] = '0' * 64
                    error = 'scientific source identity mismatch'
                elif kind == 'shard':
                    execution['new_committed_shards'][str(paths['data'])]['sha256'] = '0' * 64
                    error = 'shard attribution mismatch'
                else:
                    execution['exact_equality_to_old_batches_claimed'] = True
                    error = 'batch/resource declaration mismatch'
                rewrite_execution(paths, execution)
                with self.assertRaisesRegex(ValueError, error):
                    assembly.assemble(root, questions, config)

    def test_pilot_artifact_tamper_rejected_even_with_untouched_proof(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, questions = fixture(root)
            paths = distributed_fixture(root, config)
            with paths['pilot_rows'].open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'supporting artifact checksum mismatch'):
                assembly.assemble(root, questions, config)

    def test_snapshot_and_supporting_artifacts_are_checked_on_resume(self):
        for kind in ('snapshot', 'artifact', 'scoring_inputs'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                config, questions = fixture(root)
                paths = distributed_fixture(root, config)
                assembly.assemble(root, questions, config)
                if kind == 'snapshot':
                    snapshot_path = root / 'outputs/provenance/execution_snapshot.json'
                    snapshot = load(snapshot_path)
                    snapshot['summary']['historical_score_bytes_restored'] = True
                    score.atomic_json(snapshot_path, snapshot)
                    error = 'snapshot checksum mismatch'
                elif kind == 'artifact':
                    with paths['pilot_rows'].open('ab') as handle:
                        handle.write(b'corruption')
                    error = 'supporting artifact changed'
                else:
                    metadata_path = paths['data'].with_suffix('.meta.json')
                    metadata = load(metadata_path)
                    metadata['execution']['requested_batch_size'] = 2
                    score.atomic_json(metadata_path, metadata)
                    error = 'snapshot no longer matches'
                with self.assertRaisesRegex(ValueError, error):
                    assembly.assemble(root, questions, config)


if __name__ == '__main__':
    unittest.main()
