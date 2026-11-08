#!/usr/bin/env python3
"""Validate all eight fresh scoring runs and assemble CPU-only balanced caches.

Only committed, identity-matched complete rotations enter the new cache. Original
run files are never modified. Fresh probabilities remain labelled new measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT / 'src'))
from bca_reference_selection.data import (
    aggregate_rotations, fingerprint, load_questions, SEMANTIC_MANIFEST_SHA256,
)
from score_endpoints import (
    SCHEMA, atomic_json, canonical, file_sha256, read_jsonl, slug,
    validate_design, validate_score_rows,
)

LOCKED_CONFIG = COMPONENT / 'protocol/config/experiment.json'
ROTATION_COLUMNS = [
    'schema_version', 'model_id', 'model_revision', 'prompt_id', 'base_question_id',
    'category', 'semantic_condition', 'condition', 'family', 'dose', 'track',
    'permutation_id', 'option_count', 'gold_semantic_option',
    'semantic_remapped_probabilities', 'score_origin', 'canonical_prompt_hash',
    'rendered_prompt_hash',
]


def _load_json(path):
    return json.loads(Path(path).read_text())


def _published_file(temporary: Path, destination: Path) -> None:
    """Atomic new file, or verify an identical interrupted publication."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(temporary):
            raise ValueError(f'Refusing to replace a different existing cache: {destination}')
        return
    os.link(temporary, destination)


def _replace_paths(value, mapping):
    if isinstance(value, dict):
        return {k: _replace_paths(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_paths(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def _expected_questions(root: Path, supplied: Path | None):
    if supplied is not None:
        return Path(supplied).expanduser().resolve()
    choices = [root.parent / 'prepared_v2/data/mmlu_pro_selected_560.jsonl',
               root.parent / 'inputs/mmlu_pro_selected_560.jsonl']
    config = root / 'configs/experiment_v23.json'
    if config.is_file():
        source = _load_json(config).get('source_roots', {}).get('v2')
        if source:
            choices.append(Path(source) / 'data/mmlu_pro_selected_560.jsonl')
    found = next((p for p in choices if p.is_file()), None)
    if found is None:
        raise FileNotFoundError('Original selected560 JSONL required; pass --questions. Checked: ' + ', '.join(map(str, choices)))
    return found.resolve()


def _identity_check(path, model, config, manifest_hash, rendered_hash):
    record = _load_json(path)
    identity = record.get('input_fingerprint')
    payload = {k: v for k, v in record.items() if k != 'input_fingerprint'}
    if identity != hashlib.sha256(canonical(payload)).hexdigest():
        raise ValueError(f'Scoring identity checksum mismatch: {path}')
    expected = {
        'model_id': model, 'model_revision': config['model_revisions'][model],
        'precision': 'bfloat16', 'candidate_definition': 'summed_full_continuation_log_likelihood',
        'semantic_manifest_sha256': manifest_hash, 'rendered_prompt_sha256': rendered_hash,
        'measurement_origin': 'fresh_endpoint_scoring', 'historical_cache_restoration': False,
        'all_rotations_scored': True,
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Pinned model/protocol identity mismatch: {path}')
    for required in ['hardware', 'versions', 'source_sha256', 'driver_sha256', 'model_inventory', 'model_adapter']:
        if not record.get(required):
            raise ValueError(f'Scoring identity lacks {required}: {path}')
    if record['model_inventory'].get('revision') != config['model_revisions'][model]:
        raise ValueError(f'Model inventory revision mismatch: {path}')
    return record


def _content_hash(value, omit=()):
    return hashlib.sha256(canonical({k: v for k, v in value.items() if k not in omit})).hexdigest()


def _execution_paths(root):
    """Keep every attempt, including failed attempts and archived controller files."""
    status = root / 'status'
    paths = set(status.glob('distributed_campaign*.json'))
    for name in ('distributed_workers', 'distributed_shared', 'distributed_remote_proofs',
                 'distributed_cancellations', 'distributed_archive'):
        paths.update((status / name).rglob('*.json'))
    for directory in (root.parent / 'validation').glob('shared_pilots*'):
        if directory.is_dir():
            paths.update(directory.rglob('*.json'))
    return paths


def _evidence_references(record):
    """JSON proofs explicitly referenced by the controller, worker, or wrapper."""
    references = list(record.get('shared_execution', {}).get('proofs', []))
    for key in ('pilot_proof', 'remote_reference_proof'):
        if record.get(key):
            references.append(record[key])
    for slot in record.get('slots', []):
        references.extend({'path': path} for path in slot.get('shared_pilot_proofs', {}).values())
    return references


def _execution_snapshot(root, model_records, shard_records):
    """Freeze live status content once; bind immutable proof, source and shard bytes.

    A controller changes from finishing to complete after assembly. Its live path
    is therefore an observation label, never a file identity required on resume.
    """
    destination = root / 'outputs/provenance/execution_snapshot.json'
    scoring_inputs_hash = _content_hash({'models': model_records, 'shards': shard_records})
    if destination.is_file():
        snapshot = _load_json(destination)
        if snapshot.get('snapshot_sha256') != _content_hash(snapshot, ('snapshot_sha256',)):
            raise ValueError('Execution snapshot checksum mismatch')
        if snapshot.get('scoring_inputs_sha256') != scoring_inputs_hash:
            raise ValueError('Execution snapshot no longer matches the completed scoring inputs')
        for entry in snapshot['records']:
            if entry['content_sha256'] != _content_hash(entry['content']):
                raise ValueError('Execution snapshot record checksum mismatch')
        for artifact in snapshot['supporting_artifacts']:
            if fingerprint(artifact['path']) != artifact:
                raise ValueError(f'Execution supporting artifact changed: {artifact["path"]}')
        return {'snapshot_file': fingerprint(destination), **snapshot['summary']}

    paths = _execution_paths(root)
    if not paths:
        return None  # Serial scoring and serial fixtures have no distributed dependency.
    identities = {r['model_id']: r['scoring_identity'] for r in model_records}
    shards = {r['data']['path']: r for r in shard_records}
    artifacts, captured = {}, {}
    repo = COMPONENT.parents[3]

    def bind_file(path, expected_sha, expected_rows=None):
        path = str(Path(path).expanduser().resolve(strict=True))
        if path not in artifacts:
            artifacts[path] = fingerprint(path)
        if artifacts[path]['sha256'] != expected_sha:
            raise ValueError(f'Execution supporting artifact checksum mismatch: {path}')
        if expected_rows is not None:
            import pyarrow.parquet as pq
            if pq.read_metadata(path).num_rows != expected_rows:
                raise ValueError(f'Execution supporting artifact row count mismatch: {path}')
        return artifacts[path]

    # Read each live JSON atomically published by its writer exactly once. A proof
    # reference is compared to these captured bytes, not a second live-file read.
    pending = list(sorted(paths))
    while pending:
        path = pending.pop(0).expanduser().resolve(strict=True)
        if str(path) in captured:
            continue
        payload = path.read_bytes()
        record = json.loads(payload)
        if not isinstance(record, dict):
            raise ValueError(f'Execution evidence must be a JSON object: {path}')
        captured[str(path)] = {'observed_source_path': str(path),
            'source_file_sha256_at_capture': hashlib.sha256(payload).hexdigest(),
            'content_sha256': _content_hash(record), 'content': record}
        pending.extend(Path(ref['path']) for ref in _evidence_references(record))

    changes, attributed = [], {}
    for path, entry in sorted(captured.items()):
        record = entry['content']
        shared = record.get('schema_version') == 'xwvg_shared_scoring_execution_v1'
        remote = record.get('schema_version') == 'xwvg_remote_reference_validation_v1'
        terminal = record.get('status') in ('complete', 'passed')
        new_shards = record.get('new_committed_shards', {})
        strict = (shared or remote) and (terminal or bool(new_shards))
        validation = {'self_checksum_verified': False, 'bound_to_scoring_identity': False,
                      'retained_attempt_status': record.get('status')}
        if 'proof_sha256' in record:
            if record['proof_sha256'] != _content_hash(record, ('proof_sha256',)):
                raise ValueError(f'Execution proof checksum mismatch: {path}')
            validation['self_checksum_verified'] = True
        elif strict:
            raise ValueError(f'Completed/contributing execution evidence is unsealed: {path}')
        if Path(path) == root / 'status/distributed_campaign.json':
            for key, filename in [('controller_sha256', 'run_distributed.py'),
                                  ('reference_driver_sha256', 'score_endpoints.py'),
                                  ('shared_runtime_sha256', 'shared_scoring_runtime.py')]:
                if record.get(key):
                    bind_file(Path(__file__).with_name(filename), record[key])
        for ref in _evidence_references(record):
            target = captured[str(Path(ref['path']).expanduser().resolve())]
            if ref.get('sha256') and target['source_file_sha256_at_capture'] != ref['sha256']:
                raise ValueError(f'Execution proof reference checksum mismatch: {ref["path"]}')
        if strict:
            identity = identities.get(record.get('model_id'))
            if identity is None:
                raise ValueError(f'Execution evidence model is outside scoring roster: {path}')
            for key in ('model_revision', 'input_fingerprint', 'versions', 'hardware'):
                if record.get(key) != identity[key]:
                    raise ValueError(f'Execution evidence differs from frozen {key}: {path}')
            if record.get('baseline_identity') != identity:
                raise ValueError(f'Execution baseline identity mismatch: {path}')
            source_map = record.get('source_sha256', {})
            for relative, expected_sha in identity['source_sha256'].items():
                if source_map.get(relative) != expected_sha:
                    raise ValueError(f'Execution scientific source identity mismatch: {path}')
                bind_file(repo / relative, expected_sha)
            for relative, expected_sha in source_map.items():
                bind_file(repo / relative if '/' in relative else Path(__file__).with_name(relative), expected_sha)
            driver_sha = record.get('baseline_driver_sha256') if shared else source_map.get('score_endpoints.py')
            if driver_sha != identity['driver_sha256']:
                raise ValueError(f'Execution baseline driver mismatch: {path}')
            bind_file(Path(__file__).with_name('score_endpoints.py'), driver_sha)
            for source, expected_sha in record.get('immutable_input_source_sha256', {}).items():
                bind_file(source, expected_sha)
            if record.get('immutable_inputs_sources_unchanged') is not True:
                raise ValueError(f'Execution immutable source/input verification failed: {path}')
            for key in ('hostname', 'gpu_uuid', 'driver_version'):
                if not record.get(key):
                    raise ValueError(f'Execution lacks physical runtime {key}: {path}')
            validation['bound_to_scoring_identity'] = True
            if shared:
                bind_file(Path(__file__).with_name('shared_scoring_runtime.py'), record.get('wrapper_sha256'))
                execution_keys = ('wrapper_sha256', 'baseline_driver_sha256', 'input_fingerprint',
                    'hostname', 'gpu_uuid', 'driver_version', 'resource_config', 'versions', 'hardware')
                if record.get('execution_fingerprint') != _content_hash({k: record[k] for k in execution_keys}):
                    raise ValueError(f'Shared execution fingerprint mismatch: {path}')
                resource = record.get('resource_config', {})
                if (record.get('batch_size') != 1 or resource.get('batch_size') != 1
                        or record.get('exact_equality_to_old_batches_claimed') is not False
                        or not record.get('batch_change_reason')
                        or any(record.get(k) != resource.get(k) for k in ('memory_budget_mib', 'reserve_mib'))
                        or resource.get('allocator_limit_bytes') != record['memory_budget_mib'] * 1024**2):
                    raise ValueError(f'Shared execution batch/resource declaration mismatch: {path}')
                if record.get('stage') == 'pilot' and terminal:
                    if (record.get('prompt_count') != 42 or record.get('finite_all_outputs') is not True
                            or record.get('worst_case_probe_passed') is not True
                            or record.get('baseline_score_outputs_written') is not False
                            or record.get('pilot_rows', {}).get('row_count') != 42
                            or record.get('resource_probe_rows', {}).get('row_count') != 1):
                        raise ValueError(f'Shared pilot does not establish feasibility: {path}')
                if record.get('stage') == 'score':
                    proof_ref = record.get('pilot_proof')
                    if not proof_ref:
                        raise ValueError(f'Shared scoring lacks a pilot proof: {path}')
                    proof = captured[str(Path(proof_ref['path']).resolve())]['content']
                    if (proof.get('status') != 'passed' or proof.get('stage') != 'pilot'
                            or any(proof.get(k) != record.get(k) for k in
                                   (*execution_keys, 'model_id', 'model_revision', 'execution_fingerprint'))):
                        raise ValueError(f'Shared score/pilot provenance mismatch: {path}')
                    if terminal and (record.get('finite_all_outputs') is not True
                            or record.get('worst_case_probe_passed') is not True):
                        raise ValueError(f'Shared scoring completion is invalid: {path}')
            elif terminal and (record.get('exact_reference_equality') is not True
                    or record.get('baseline_artifacts_unchanged') is not True or record.get('prompt_count') != 42):
                raise ValueError(f'Remote fixed-pilot equality proof is incomplete: {path}')

        for kind in ('pilot_rows', 'resource_probe_rows', 'partial_pilot_rows'):
            if record.get(kind):
                artifact = record[kind]
                bind_file(artifact['path'], artifact['sha256'], artifact.get('row_count'))
        if record.get('baseline_complete_status_path'):
            bind_file(record['baseline_complete_status_path'], record['baseline_complete_status_sha256'])
        before = record.get('committed_shards_before', {})
        if not isinstance(before, dict) or not isinstance(new_shards, dict):
            raise ValueError(f'Execution shard inventories must be mappings: {path}')
        if set(before) & set(new_shards):
            raise ValueError(f'Execution before/new shard inventories overlap: {path}')
        for shard_path, claimed in {**before, **new_shards}.items():
            actual = shards.get(str(Path(shard_path).resolve()))
            if actual is None or actual['model_id'] != record.get('model_id'):
                raise ValueError(f'Execution refers to an uncommitted or wrong-model shard: {shard_path}')
            expected = {'sha256': actual['data']['sha256'], 'meta_sha256': actual['metadata']['sha256'],
                'commit_sha256': actual['completion']['sha256'], 'row_count': actual['row_count'],
                'input_fingerprint': actual['input_fingerprint'], 'execution': actual['execution']}
            if claimed != expected:
                raise ValueError(f'Execution shard attribution mismatch: {shard_path}')
            if shard_path in new_shards:
                if shared and (claimed['execution'].get('requested_batch_size') != 1
                        or claimed['execution'].get('effective_batch_size') != 1
                        or any(b.get('batch_size') != 1 for b in claimed['execution'].get('batches', []))):
                    raise ValueError(f'Shared attributed shard was not scored at batch one: {shard_path}')
                attributed.setdefault(shard_path, []).append(path)
        if 'new_committed_shards' in record:
            if record.get('existing_committed_shards_unchanged') is not True:
                raise ValueError(f'Execution changed pre-existing committed shards: {path}')
            if terminal and set(before) | set(new_shards) != {
                    p for p, shard in shards.items() if shard['model_id'] == record.get('model_id')}:
                raise ValueError(f'Completed execution shard inventory is incomplete: {path}')
        if shared and record.get('stage') == 'score':
            changes.append({k: record.get(k) for k in ('model_id', 'status', 'hostname', 'gpu_uuid',
                'driver_version', 'batch_size', 'memory_budget_mib', 'reserve_mib', 'wrapper_sha256',
                'baseline_driver_sha256', 'batch_change_reason', 'exact_equality_to_old_batches_claimed')}
                | {'evidence_observed_path': path, 'evidence_content_sha256': entry['content_sha256'],
                   'new_committed_shard_count': len(new_shards),
                   'new_committed_row_count': sum(row['row_count'] for row in new_shards.values())})
        if record.get('shared_execution') and record.get('status') == 'complete':
            refs = record['shared_execution'].get('proofs', [])
            if not any(ref.get('stage') == 'score' and captured[str(Path(ref['path']).resolve())]['content'].get('status') == 'complete'
                       for ref in refs):
                raise ValueError(f'Complete shared worker lacks completed score evidence: {path}')
            for ref in refs:
                proof = captured[str(Path(ref['path']).resolve())]['content']
                if proof.get('stage') != ref.get('stage') or proof.get('model_id') != record.get('model_id'):
                    raise ValueError(f'Shared worker/proof model or stage differs: {path}')
                for key in ('wrapper_sha256', 'baseline_driver_sha256', 'memory_budget_mib', 'reserve_mib', 'batch_size'):
                    if key in record['shared_execution'] and record['shared_execution'][key] != proof.get(key):
                        raise ValueError(f'Shared worker/proof resource identity differs: {path}')
        entry['validation'] = validation

    batch_sizes = {}
    for model in identities:
        execution = [row['execution'] for row in shard_records if row['model_id'] == model]
        batch_sizes[model] = {
            key: sorted({row[key] for row in execution if key in row})
            for key in ('requested_batch_size', 'effective_batch_size')}
        batch_sizes[model]['observed_batch_sizes'] = sorted({b['batch_size']
            for row in execution for b in row.get('batches', []) if 'batch_size' in b})
    summary = {'records_captured': len(captured), 'shared_batch_changes': changes,
        'shared_new_shard_attribution': attributed, 'cross_batch_bitwise_equality_claimed': False,
        'committed_batch_sizes_by_model': batch_sizes,
        'historical_score_bytes_restored': False,
        'scope': 'Fixed snapshots of live statuses plus all discovered execution/proof attempts. Batch-one shared execution preserves the scientific scorer but does not establish byte equality with differently batched or historical scores.'}
    snapshot = {'schema_version': 'xwvg_execution_provenance_snapshot_v1',
        'scoring_inputs_sha256': scoring_inputs_hash,
        'captured_utc': datetime.now(timezone.utc).isoformat(), 'records': list(captured.values()),
        'supporting_artifacts': [artifacts[p] for p in sorted(artifacts)], 'summary': summary}
    snapshot['snapshot_sha256'] = _content_hash(snapshot)
    atomic_json(destination, snapshot, immutable=True)
    return {'snapshot_file': fingerprint(destination), **summary}


def assemble(root: Path, questions: Path, config: dict) -> dict:
    import pandas as pd

    root = Path(root).expanduser().resolve()
    if any(p.name.startswith('PIER_LLM_ECOSYSTEM_') for p in [root, *root.parents]):
        raise ValueError('Assembler must write into a new fresh-scoring workspace, not a historical run')
    selected, question_identity = load_questions(questions, config)
    semantic_path = root / 'data/permutations/score_semantic_manifest.jsonl'
    semantic = read_jsonl(semantic_path)
    semantic_hash = file_sha256(semantic_path)
    if not config.get('synthetic_fixture', False) and semantic_hash != SEMANTIC_MANIFEST_SHA256:
        raise ValueError('Canonical semantic manifest differs from historical protocol')
    execution_path = root / 'data/permutations/score_execution_manifest.jsonl'
    if execution_path.is_file():
        prompts = read_jsonl(execution_path)
        expected = [{**row, 'inference_required': True, 'rotation0_reuse': False} for row in semantic]
        if prompts != expected:
            raise ValueError('Fresh execution manifest changes more than rotation-zero execution metadata')
    else:
        execution_path = semantic_path
        prompts = semantic
    by_id = {row['prompt_id']: row for row in prompts}
    expected_count = int(selected['option_count'].sum()) * 7
    if len(prompts) != expected_count or len(by_id) != expected_count:
        raise ValueError('Execution manifest prompt count/identity is incomplete')
    if set(str(row['base_question_id']) for row in prompts) != set(selected.base_question_id):
        raise ValueError('Execution manifest does not use exactly the original questions')
    manifest_hash = file_sha256(execution_path)
    missing = []
    for model in config['model_roster']:
        name = slug(model)
        paths = [root/f'status/fresh_score_{name}_identity.json',
                 root/f'status/fresh_score_{name}_complete.json',
                 root/f'data/permutations/rendered/{name}.jsonl']
        missing.extend(str(p) for p in paths if not p.is_file())
        directory = root/'outputs/raw_scores'/name
        if not any(directory.glob('shard_*.meta.json')):
            missing.append(str(directory/'shard_*.meta.json'))
    if missing:
        report = {'status': 'incomplete_fresh_scoring_inputs', 'missing': missing,
                  'required_model_count': len(config['model_roster']),
                  'required_rows_per_model': expected_count, 'empirical_analysis_run': False}
        atomic_json(root/'status/fresh_assembly_blocked.json', report)
        raise FileNotFoundError('Complete eight-model scoring required; see status/fresh_assembly_blocked.json')
    records, model_records, shard_records, ignored_uncommitted = [], [], [], []
    for model in config['model_roster']:
        name = slug(model)
        rendered_path = root/f'data/permutations/rendered/{name}.jsonl'
        rendered = read_jsonl(rendered_path)
        validate_design(prompts, rendered, historical_counts=not config.get('synthetic_fixture', False))
        rendered_by_id = {row['prompt_id']: row for row in rendered}
        identity_path = root/f'status/fresh_score_{name}_identity.json'
        identity = _identity_check(identity_path, model, config, manifest_hash, file_sha256(rendered_path))
        fingerprint_value = identity['input_fingerprint']
        complete_path = root/f'status/fresh_score_{name}_complete.json'
        complete = _load_json(complete_path)
        if (complete.get('input_fingerprint') != fingerprint_value or complete.get('status') != 'complete'
                or complete.get('stage') != 'score' or complete.get('model_id') != model
                or complete.get('model_revision') != config['model_revisions'][model]
                or complete.get('completed_rows') != expected_count
                or complete.get('fresh_measurements') is not True
                or complete.get('historical_cache_restoration') is not False):
            raise ValueError(f'Model scoring completion record is invalid: {complete_path}')
        directory = root/'outputs/raw_scores'/name
        seen, committed_paths = set(), set()
        model_shards = []
        for sidecar in sorted(directory.glob('shard_*.meta.json')):
            data = sidecar.with_suffix('').with_suffix('.parquet')
            marker = data.with_suffix('.complete.json')
            if not marker.is_file():
                ignored_uncommitted.append(str(sidecar))
                continue
            meta, commit = _load_json(sidecar), _load_json(marker)
            expected = {'schema_version': SCHEMA, 'model_id': model,
                        'model_revision': config['model_revisions'][model],
                        'input_fingerprint': fingerprint_value, 'manifest_sha256': manifest_hash,
                        'measurement_origin': 'fresh_endpoint_scoring'}
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Shard metadata model/protocol mismatch: {sidecar}')
            if (commit.get('complete') is not True or commit.get('input_fingerprint') != fingerprint_value
                    or commit.get('row_count') != meta.get('row_count')
                    or commit.get('sha256') != meta.get('sha256')
                    or not data.is_file() or file_sha256(data) != meta.get('sha256')):
                raise ValueError(f'Shard commit/checksum mismatch: {data}')
            rows = pd.read_parquet(data).to_dict('records')
            ids = [str(row['prompt_id']) for row in rows]
            if len(rows) != meta['row_count'] or ids != meta.get('prompt_ids') or len(set(ids)) != len(ids):
                raise ValueError(f'Shard ordered prompt identity mismatch: {data}')
            if set(ids) & seen:
                raise ValueError(f'Duplicate prompts across committed shards: {model}')
            validate_score_rows(rows, by_id, rendered_by_id, model, config['model_revisions'][model])
            if any(row.get('score_origin') != 'fresh_endpoint_scoring' for row in rows):
                raise ValueError(f'Shard contains non-fresh measurement origin: {data}')
            if any('manifest_sha256' in row and row['manifest_sha256'] != manifest_hash for row in rows):
                raise ValueError(f'Row manifest hash mismatch: {data}')
            seen.update(ids)
            records.extend({column: row[column] for column in ROTATION_COLUMNS} for row in rows)
            source = {'data': fingerprint(data), 'metadata': fingerprint(sidecar),
                      'completion': fingerprint(marker), 'row_count':len(rows),
                      'input_fingerprint':fingerprint_value, 'model_id':model,
                      'execution':meta.get('execution', {})}
            shard_records.append(source)
            model_shards.append(source)
            committed_paths.update([data, sidecar, marker])
        if seen != set(by_id):
            missing_ids = sorted(set(by_id) - seen)
            raise ValueError(f'Incomplete all-rotation prompt set for {model}; missing={len(missing_ids)}, first={missing_ids[:5]}')
        for path in directory.glob('shard_*.*'):
            if path not in committed_paths and str(path) not in ignored_uncommitted:
                ignored_uncommitted.append(str(path))
        model_records.append({'model_id':model, 'model_revision':config['model_revisions'][model],
                              'row_count':len(seen), 'scoring_identity':identity,
                              'identity_file':fingerprint(identity_path),
                              'completion_file':fingerprint(complete_path),
                              'scoring_summary':complete, 'rendered_file':fingerprint(rendered_path),
                              'shard_count':len(model_shards)})
        print(json.dumps({'assembly_model_verified':model,'rotation_rows':len(seen)}),flush=True)
    if len(records) != expected_count * len(config['model_roster']):
        raise ValueError('Incomplete aggregate rotation inventory')
    execution_provenance = _execution_snapshot(root, model_records, shard_records)
    input_record = {'model_records':model_records, 'shards':shard_records,
                    'questions':question_identity, 'semantic_manifest':fingerprint(semantic_path),
                    'execution_manifest':fingerprint(execution_path), 'configuration':config,
                    'assembler_sha256':file_sha256(Path(__file__)),
                    'score_validator_sha256':file_sha256(Path(__file__).with_name('score_endpoints.py')),
                    'assembly_versions':{name:version(name) for name in ['numpy','pandas','pyarrow','scipy']}}
    if execution_provenance is not None:
        input_record['distributed_execution_provenance'] = execution_provenance
    assembly_identity = hashlib.sha256(canonical(input_record)).hexdigest()
    rotation_path = root/'outputs/validation/permutation_wise_semantic_scores.parquet'
    average_path = root/'outputs/analysis/permutation_averaged_semantic_scores.parquet'
    completion_path = root/'status/fresh_assembly_complete.json'
    provenance_path = root/'status/measurement_provenance.json'
    if completion_path.is_file():
        old = _load_json(completion_path)
        if (old.get('assembly_fingerprint') != assembly_identity or not provenance_path.is_file()
                or old.get('measurement_provenance_sha256') != file_sha256(provenance_path)):
            raise ValueError('Existing assembly completion identity/provenance mismatch')
        for key, path in [('rotation_cache',rotation_path),('averaged_cache',average_path)]:
            if not path.is_file() or old[key]['sha256'] != file_sha256(path):
                raise ValueError(f'Existing assembled cache checksum mismatch: {path}')
        return {**old, 'invocation_status':'reused_verified_complete_cache'}
    (root/'outputs').mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fresh-assembly-',dir=root/'outputs') as temporary:
        temp = Path(temporary)
        rotations, averaged = temp/'rotations.parquet', temp/'averaged.parquet'
        frame = pd.DataFrame(records)
        frame.to_parquet(rotations,index=False)
        aggregation = aggregate_rotations(rotations, averaged, questions, config)
        _published_file(rotations,rotation_path)
        _published_file(averaged,average_path)
        aggregation = _replace_paths(aggregation,{str(rotations):str(rotation_path),str(averaged):str(average_path)})
    # The complete provenance file is itself an identity-checked input to analysis.
    provenance = {'schema_version':'xwvg_fresh_measurement_provenance_v1',
                  'measurement_origin':'fresh_endpoint_scoring','historical_cache_restoration':False,
                  'fresh_measurements':True, 'assembly_completed_utc':datetime.now(timezone.utc).isoformat(),
                  'model_scoring_completed_utc':{r['model_id']:r['scoring_summary'].get('created_at') for r in model_records},
                  'assembly_fingerprint':assembly_identity,'model_count':len(model_records),
                  'question_count':len(selected),'rotation_rows_per_model':expected_count,
                  'rotation_rows':len(records),'balanced_probability_rows':config['expected_primary_cache_rows'],
                  'precision':'bfloat16','all_rotations_scored':True,'rotation_zero_reused':False,
                  'candidate_scoring':'summed_full_continuation_log_likelihood',
                  'input_identity':input_record,'rotation_cache':fingerprint(rotation_path),
                  'averaged_cache':fingerprint(average_path), 'aggregation_validation':aggregation,
                  'ignored_uncommitted_files':sorted(set(ignored_uncommitted)),
                  'empirical_reference_selection_run':False}
    atomic_json(provenance_path,provenance)
    completion = {'status':'fresh_balanced_cache_complete','assembly_fingerprint':assembly_identity,
                  'measurement_origin':'fresh_endpoint_scoring','historical_cache_restoration':False,
                  'rotation_cache':fingerprint(rotation_path),'averaged_cache':fingerprint(average_path),
                  'measurement_provenance_path':str(provenance_path),
                  'measurement_provenance_sha256':file_sha256(provenance_path),
                  'model_count':len(model_records),'rotation_rows':len(records),
                  'balanced_probability_rows':config['expected_primary_cache_rows']}
    atomic_json(completion_path,completion)
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--questions',type=Path)
    parser.add_argument('--config',type=Path,default=LOCKED_CONFIG)
    args=parser.parse_args(argv)
    try:
        config=_load_json(args.config)
        if config != _load_json(LOCKED_CONFIG):
            raise ValueError('CLI requires the locked eight-model analysis configuration')
        questions=_expected_questions(args.run_root.expanduser().resolve(),args.questions)
        result=assemble(args.run_root,questions,config)
        print(json.dumps(result,indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status':'assembly_incomplete','error':str(exc)}),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
