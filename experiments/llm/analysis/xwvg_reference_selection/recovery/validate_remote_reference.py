#!/usr/bin/env python3
"""Replay one model's fixed historical pilot before a new host resumes scoring.

Only the original scorer is used. The exact saved pilot batches are immutable;
an OOM or any numerical difference fails the gate. Baseline files are read-only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid as uuid_module

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_endpoints import (SOURCE_FILES, atomic_json, canonical, file_sha256,
    fixed_pilot_ids, import_historical_scorer, read_jsonl, runtime_versions,
    scan_completed, slug, snapshot_inventory, utc_now, validate_design,
    validate_score_rows)
from validate_score_math import compare_rows, normalize

PILOT_ROWS = 42


def load(path):
    return json.loads(Path(path).read_text())


def require_identity(baseline, current):
    """Compare every original identity field, including full model file hashes."""
    expected = {k: v for k, v in baseline.items() if k != 'input_fingerprint'}
    fingerprint = hashlib.sha256(canonical(expected)).hexdigest()
    if baseline.get('input_fingerprint') != fingerprint:
        raise ValueError('Baseline identity fingerprint is corrupt')
    if canonical(expected) != canonical(current):
        changed = sorted(k for k in set(expected) | set(current)
                         if canonical(expected.get(k)) != canonical(current.get(k)))
        raise ValueError(f'Remote identity differs from baseline: {changed}')
    return fingerprint


def saved_pilot_batches(pilot, design, fingerprint, deterministic_ids):
    if (pilot.get('input_fingerprint') != fingerprint
        or design.get('input_fingerprint') != fingerprint
        or pilot.get('status') != 'complete'
        or pilot.get('pilot_measured_prompt_count') != PILOT_ROWS
        or not pilot.get('scoring_seconds', 0) > 0
        or design.get('row_count') != PILOT_ROWS
        or design.get('outcome_blind') is not True
        or design.get('prompt_ids') != deterministic_ids
        or len(deterministic_ids) != PILOT_ROWS):
        raise ValueError('Saved pilot provenance/design differs')
    batches = [batch['prompt_ids'] for batch in pilot['batches']]
    if not batches or any(not isinstance(ids, list) or not ids for ids in batches):
        raise ValueError('Saved pilot contains empty/invalid batches')
    flat = [pid for batch in batches for pid in batch]
    if len(flat) != PILOT_ROWS or len(set(flat)) != PILOT_ROWS or set(flat) != set(deterministic_ids):
        raise ValueError('Saved pilot batches must cover exactly the fixed 42 prompts once')
    if any(batch.get('batch_size') != len(batch['prompt_ids']) for batch in pilot['batches']):
        raise ValueError('Saved pilot batch size metadata differs')
    return batches


def compare_complete_rows(stored, actual, model_id, revision, manifest_sha256):
    """Restore only the four fields added by the frozen driver, then compare all keys."""
    observed = []
    for row in actual:
        row = dict(row)
        row.update(model_id=model_id, model_revision=revision,
                   score_origin='fresh_endpoint_scoring', manifest_sha256=manifest_sha256)
        observed.append(row)
    # The historical comparator assumes the same schema. Check key sets first,
    # including absent-versus-present-null fields, before delegating values.
    if len(stored) == len(observed):
        errors = [{'prompt_id': left.get('prompt_id'), 'kind': 'row_schema',
                   'missing_fields': sorted(set(left) - set(right)),
                   'unexpected_fields': sorted(set(right) - set(left))}
                  for left, right in zip(stored, observed, strict=True)
                  if set(left) != set(right)]
        if errors:
            return {'exact': False, 'errors': errors,
                    'baseline_sha256': hashlib.sha256(canonical(normalize(stored))).hexdigest(),
                    'candidate_sha256': hashlib.sha256(canonical(normalize(observed))).hexdigest()}
    return compare_rows(stored, observed)


def replay_batches(batches, by_id, stored, score, model_id, revision, manifest_sha256,
                   record_batch):
    """The injected scoring callable permits CPU tests without changing GPU code."""
    reports = []
    for index, ids in enumerate(batches):
        # Deliberately no OOM retry, batch resizing, warmup, or alternate backend.
        actual, timing = score([by_id[pid] for pid in ids])
        comparison = compare_complete_rows([stored[pid] for pid in ids], actual,
                                          model_id, revision, manifest_sha256)
        result = {'batch_index': index, 'prompt_ids': ids, 'timing': timing,
                  'comparison': comparison}
        reports.append(result)
        record_batch(result)
    return reports


def baseline_artifacts(root, model_id):
    name = slug(model_id)
    paths = [p for p in (root / 'status').glob(f'fresh_score_{name}_*') if p.is_file()]
    paths += [p for p in (root / 'outputs/raw_scores' / name).glob('shard_*') if p.is_file()]
    return {str(p): file_sha256(p) for p in sorted(paths)}


def checked_output(path, repo, root, snapshot):
    path = path.expanduser().resolve()
    protected = [root / 'data', root / 'outputs', snapshot]
    protected += [repo / p for p in ('experiments', 'paper', 'data', 'manifests')]
    if path == root or any(path == p or path.is_relative_to(p) for p in protected):
        raise ValueError('Validation output must not overwrite source, inputs, weights, or score shards')
    if path.exists() or path.name.startswith('fresh_score_'):
        raise ValueError('Use a new separate validation JSON output file')
    if path.suffix != '.json':
        raise ValueError('--output must be a JSON file')
    return path


def seal_proof(report):
    result = {k: v for k, v in report.items() if k != 'proof_sha256'}
    result['proof_sha256'] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def gpu_details(torch, device):
    properties = torch.cuda.get_device_properties(device)
    raw_uuid = properties.uuid
    if isinstance(raw_uuid, bytes) and len(raw_uuid) == 16:
        uuid = 'GPU-' + str(uuid_module.UUID(bytes=raw_uuid))
    else:
        if isinstance(raw_uuid, bytes):
            raw_uuid = raw_uuid.decode('ascii')
        uuid = 'GPU-' + str(uuid_module.UUID(str(raw_uuid).removeprefix('GPU-')))
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True,
                           check=True, timeout=20)
    matches = [line.split(',')[1].strip() for line in query.stdout.splitlines()
               if line.split(',')[0].strip() == uuid]
    if len(matches) != 1:
        raise ValueError('CUDA device UUID does not resolve uniquely in nvidia-smi')
    hardware = {'name': torch.cuda.get_device_name(device),
                'capability': list(torch.cuda.get_device_capability(device)),
                'total_memory_bytes': properties.total_memory,
                'torch_cuda_version': torch.version.cuda}
    return hardware, uuid, matches[0]


def run(args):
    repo, root, snapshot = (p.expanduser().resolve() for p in
                            (args.source_repo, args.run_root, args.model_path))
    output = checked_output(args.output, repo, root, snapshot)
    report = {'schema_version': 'xwvg_remote_reference_validation_v1',
              'status': 'running', 'model_id': args.model_id, 'hostname': platform.node(),
              'pid': os.getpid(), 'device': args.device, 'python': sys.executable,
              'started_at': utc_now(), 'batches': [],
              'acceptance_rule': 'Exact complete-row equality for all 42 saved pilot prompts in their original batches; no tolerance.',
              'warmup_policy': 'No extra warmup inference. First saved batch timing is cold and reported separately.',
              'storage_metadata_normalization': 'Only nullable track integer metadata and null storage representations; no score rounding.'}
    before = baseline_artifacts(root, args.model_id)
    report['baseline_artifact_sha256'] = before
    immutable_hashes = {}
    model = tokenizer = torch = None
    try:
        name = slug(args.model_id)
        identity = load(root / 'status' / f'fresh_score_{name}_identity.json')
        pilot = load(root / 'status' / f'fresh_score_{name}_pilot.json')
        design = load(root / 'status' / f'fresh_score_{name}_pilot_design.json')
        records = load(repo / 'experiments/llm/analysis/v23/configs/resolved_models_v23.json')['models']
        records = [r for r in records if r['id'] == args.model_id]
        if len(records) != 1:
            raise ValueError('Model must occur exactly once in the original roster')
        record = dict(records[0])
        manifest = root / 'data/permutations/score_execution_manifest.jsonl'
        if not manifest.is_file():
            manifest = root / 'data/permutations/score_semantic_manifest.jsonl'
        rendered_path = root / 'data/permutations/rendered' / f'{name}.jsonl'
        manifest_sha = file_sha256(manifest)
        rendered_sha = file_sha256(rendered_path)
        sources = {relative: file_sha256(repo / relative) for relative in SOURCE_FILES}
        versions = runtime_versions()
        immutable_paths = [manifest, rendered_path, *(repo / p for p in SOURCE_FILES),
                           HERE / 'score_endpoints.py', HERE / 'validate_score_math.py', Path(__file__)]
        immutable_hashes = {str(path): file_sha256(path) for path in immutable_paths}
        report['immutable_input_source_sha256'] = immutable_hashes
        report.update(model_revision=record['revision'], input_fingerprint=identity.get('input_fingerprint'),
                      versions=versions, source_sha256={**sources,
                      'score_endpoints.py': file_sha256(HERE / 'score_endpoints.py'),
                      'validate_score_math.py': file_sha256(HERE / 'validate_score_math.py'),
                      'validate_remote_reference.py': file_sha256(Path(__file__))})
        # Fail on CPU-visible differences before initializing CUDA or loading weights.
        for key, value in {'versions': versions, 'source_sha256': sources,
                           'driver_sha256': file_sha256(HERE / 'score_endpoints.py'),
                           'rendered_prompt_sha256': rendered_sha,
                           'semantic_manifest_sha256': manifest_sha}.items():
            if identity.get(key) != value:
                raise ValueError(f'Remote {key} differs from baseline')
        prompts, rendered = read_jsonl(manifest), read_jsonl(rendered_path)
        validate_design(prompts, rendered)
        by_id = {r['prompt_id']: r for r in prompts}
        rendered_by_id = {r['prompt_id']: r for r in rendered}
        batches = saved_pilot_batches(pilot, design, identity['input_fingerprint'],
                                     fixed_pilot_ids(prompts, rendered_by_id))
        inventory = snapshot_inventory(snapshot, record['revision'])
        import torch as torch_module
        torch = torch_module
        if not args.device.startswith('cuda:') or not torch.cuda.is_available():
            raise ValueError('An explicitly selected available CUDA device is required')
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        hardware, gpu_uuid, driver_version = gpu_details(torch, device)
        report.update(hardware=hardware, gpu_uuid=gpu_uuid, driver_version=driver_version)
        current = {'schema_version': 'xwvg_fresh_scoring_identity_v1', 'model_id': args.model_id,
            'model_revision': record['revision'], 'precision': 'bfloat16',
            'candidate_definition': 'summed_full_continuation_log_likelihood',
            'model_adapter': record['model_adapter'], 'historical_model_adapter': record['model_adapter'],
            'historical_rendered_prompt_sha256': record['v23_rendered_prompt_sha256'],
            'historical_rendering_match': rendered_sha == record['v23_rendered_prompt_sha256'],
            'interface_change_reason': None, 'model_inventory': inventory,
            'semantic_manifest_sha256': manifest_sha, 'rendered_prompt_sha256': rendered_sha,
            'versions': versions, 'hardware': hardware, 'source_sha256': sources,
            'driver_sha256': file_sha256(HERE / 'score_endpoints.py'), 'all_rotations_scored': True,
            'measurement_origin': 'fresh_endpoint_scoring', 'historical_cache_restoration': False}
        fingerprint = require_identity(identity, current)
        completed, _ = scan_completed(root / 'outputs/raw_scores' / name, fingerprint,
            by_id, rendered_by_id, args.model_id, record['revision'], manifest_sha)
        desired = set(design['prompt_ids'])
        if not desired <= completed:
            raise ValueError('Committed baseline pilot rows are incomplete')
        import pandas as pd
        stored = {}
        for sidecar in sorted((root / 'outputs/raw_scores' / name).glob('shard_*.meta.json')):
            meta = load(sidecar)
            data = sidecar.with_suffix('').with_suffix('.parquet')
            if not data.with_suffix('.complete.json').is_file() or not desired.intersection(meta['prompt_ids']):
                continue
            for row in pd.read_parquet(data).to_dict('records'):
                if row['prompt_id'] in desired:
                    stored[row['prompt_id']] = row
        if set(stored) != desired:
            raise ValueError('Missing saved pilot rows')
        report.update(prompt_count=PILOT_ROWS, saved_batch_prompt_ids=batches,
                      baseline_identity=identity)
        atomic_json(output, report)
        original = import_historical_scorer(repo)
        record['local_cache_path'] = str(snapshot)
        started = time.perf_counter()
        model, tokenizer = original._load_model(record, args.device)
        report['model_load_seconds'] = time.perf_counter() - started

        def score(rows):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            actual = original.score_prompt_batch(model, tokenizer, rows, rendered_by_id, device=args.device)
            torch.cuda.synchronize(device)
            timing = {'seconds': time.perf_counter() - started,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            augmented = [dict(r, model_id=args.model_id, model_revision=record['revision']) for r in actual]
            validate_score_rows(augmented, by_id, rendered_by_id, args.model_id, record['revision'])
            return actual, timing

        def record_batch(result):
            report['batches'].append(result)
            atomic_json(output, report)

        results = replay_batches(batches, by_id, stored, score, args.model_id,
                                 record['revision'], manifest_sha, record_batch)
        report['exact_reference_equality'] = all(b['comparison']['exact'] for b in results)
        report['first_batch_seconds'] = results[0]['timing']['seconds']
        report['remaining_batches_seconds'] = sum(b['timing']['seconds'] for b in results[1:])
        report['status'] = 'passed' if report['exact_reference_equality'] else 'failed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      exact_reference_equality=False)
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch is not None:
            gc.collect()
            torch.cuda.empty_cache()
        after = baseline_artifacts(root, args.model_id)
        report['baseline_artifacts_unchanged'] = before == after
        report['changed_baseline_artifacts'] = sorted(p for p in set(before) | set(after)
                                                     if before.get(p) != after.get(p))
        if before != after:
            report['status'] = 'failed'
        changed_inputs = [path for path, sha in immutable_hashes.items()
                          if not Path(path).is_file() or file_sha256(Path(path)) != sha]
        report['immutable_inputs_sources_unchanged'] = not changed_inputs
        report['changed_immutable_inputs_sources'] = changed_inputs
        if changed_inputs:
            report['status'] = 'failed'
        report['completed_at'] = utc_now()
        report = seal_proof(report)
        atomic_json(output, report)
    print(json.dumps({k: report.get(k) for k in ('status', 'model_id', 'prompt_count',
        'exact_reference_equality', 'baseline_artifacts_unchanged', 'proof_sha256')}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-repo', 'run-root', 'model-path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args(argv)
    try:
        return int(run(args)['status'] != 'passed')
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': repr(exc)}), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
