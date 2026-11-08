#!/usr/bin/env python3
"""Score only complete balanced semantic endpoints using the pinned V2.3 scorer.

No generation, paper postprocessing, target fitting, or historical writes. All
rotations, including rotation zero, are fresh measurements. The scientific
scoring functions are imported unchanged from an explicitly supplied checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

SCHEMA = 'pier_v23_candidate_score_v1'
EXPECTED_PROMPTS = 36946
EXPECTED_GROUPS = 3920
EXPECTED_CONTINUATIONS = 357196
CONDITIONS = ('C0', 'I0', 'I1', 'I2', 'D0', 'D1', 'D2')
SOURCE_FILES = (
    'experiments/llm/analysis/v23/src/pier_llm_v23/inference.py',
    'experiments/llm/analysis/v23/src/pier_llm_v23/prepare.py',
    'experiments/llm/analysis/v23/src/pier_llm_v23/core.py',
    'experiments/llm/analysis/v23/src/pier_llm_v23/utils.py',
    'experiments/llm/analysis/v23/configs/resolved_models_v23.json',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '--', value).strip('-')


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(16 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: Any, *, immutable=False) -> None:
    path = Path(path)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Immutable scoring record differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_design(prompts: list[dict], rendered: list[dict], *, historical_counts=True) -> dict:
    """Identity validation never scores or filters by the old inference flag."""
    if not prompts or not rendered:
        raise ValueError('Both the semantic and rendered prompt tables are required')
    by_id = {}
    grouped = defaultdict(list)
    question_meta = {}
    question_conditions = defaultdict(set)
    continuations = 0
    for row in prompts:
        pid = str(row['prompt_id'])
        if pid in by_id:
            raise ValueError(f'Duplicate semantic prompt ID: {pid}')
        by_id[pid] = row
        qid, condition = str(row['base_question_id']), str(row['semantic_condition'])
        n, rotation = int(row['option_count']), int(row['permutation_id'])
        if n != row['option_count'] or rotation != row['permutation_id'] or not 2 <= n <= 10:
            raise ValueError(f'Invalid option count/rotation: {pid}')
        if condition not in CONDITIONS or not 0 <= rotation < n:
            raise ValueError(f'Invalid semantic condition/rotation: {pid}')
        if pid != f'{qid}__{condition}__cyclic_r{rotation:02d}':
            raise ValueError(f'Prompt ID disagrees with its semantic identity: {pid}')
        metadata = (str(row['category']), n, int(row['semantic_answer_index']))
        if not 0 <= metadata[2] < n or question_meta.setdefault(qid, metadata) != metadata:
            raise ValueError(f'Question option/category/gold metadata disagree: {qid}')
        family = 'clean' if condition == 'C0' else 'irrelevant_context' if condition[0] == 'I' else 'content_deletion'
        dose = 0.0 if condition == 'C0' else 512.0 if condition[0] == 'I' else 0.4
        track = None if condition == 'C0' else int(condition[1])
        if row['family'] != family or row['condition'] != family or float(row['dose']) != dose or row['track'] != track:
            raise ValueError(f'Invalid endpoint condition metadata: {pid}')
        labels = list('ABCDEFGHIJ'[:n])
        if row['option_labels'] != labels or row['candidate_continuations'] != [f' ({label})' for label in labels]:
            raise ValueError(f'Candidate continuation convention differs: {pid}')
        if row['semantic_to_visible'] != [(i + rotation) % n for i in range(n)] or row['visible_to_semantic'] != [(i - rotation) % n for i in range(n)]:
            raise ValueError(f'Invalid cyclic semantic mapping: {pid}')
        if hashlib.sha256(row['canonical_content'].encode('utf-8')).hexdigest() != row['canonical_content_sha256']:
            raise ValueError(f'Canonical content checksum mismatch: {pid}')
        grouped[(qid, condition)].append(rotation)
        question_conditions[qid].add(condition)
        continuations += n
    for key, rotations in grouped.items():
        if sorted(rotations) != list(range(question_meta[key[0]][1])):
            raise ValueError(f'Incomplete cyclic set: {key}')
    if any(conditions != set(CONDITIONS) for conditions in question_conditions.values()):
        raise ValueError('All seven semantic conditions are required for every question')
    rendered_by_id = {}
    for row in rendered:
        pid = str(row['prompt_id'])
        if pid in rendered_by_id or pid not in by_id:
            raise ValueError(f'Duplicate or unexpected rendered prompt: {pid}')
        rendered_by_id[pid] = row
        if row['canonical_content_sha256'] != by_id[pid]['canonical_content_sha256']:
            raise ValueError(f'Rendered canonical identity mismatch: {pid}')
        if hashlib.sha256(row['rendered_prompt'].encode('utf-8')).hexdigest() != row['rendered_prompt_sha256']:
            raise ValueError(f'Rendered prompt checksum mismatch: {pid}')
        if int(row['prompt_token_count']) <= 0 or int(row['prompt_token_count']) != row['prompt_token_count']:
            raise ValueError(f'Invalid prompt token count: {pid}')
    if set(rendered_by_id) != set(by_id):
        raise ValueError('Rendered prompt set is incomplete')
    if historical_counts:
        if len(prompts) != EXPECTED_PROMPTS or len(grouped) != EXPECTED_GROUPS or continuations != EXPECTED_CONTINUATIONS:
            raise ValueError(f'Historical design dimensions disagree: prompts={len(prompts)}, groups={len(grouped)}, continuations={continuations}')
        categories = Counter(meta[0] for meta in question_meta.values())
        if len(question_meta) != 560 or len(categories) != 14 or set(categories.values()) != {40}:
            raise ValueError('Historical selected question/category counts disagree')
    return {'passed': True, 'prompt_count': len(prompts), 'question_count': len(question_meta),
            'semantic_group_count': len(grouped), 'candidate_continuations': continuations,
            'rotation_zero_count': sum(int(r['permutation_id']) == 0 for r in prompts),
            'all_rotations_scored': True, 'historical_inference_required_flag_ignored': True}


def fixed_pilot_ids(prompts: list[dict], rendered_by_id: dict[str, dict]) -> list[str]:
    """Outcome-blind condition × zero/nonzero × shortest/median/longest pilot."""
    chosen = set()
    for condition in CONDITIONS:
        for is_zero in (True, False):
            rows = [r for r in prompts if r['semantic_condition'] == condition and
                    (int(r['permutation_id']) == 0) == is_zero]
            ordered = sorted(rows, key=lambda r: (int(rendered_by_id[r['prompt_id']]['prompt_token_count']), r['prompt_id']))
            if not ordered:
                raise ValueError(f'Missing pilot stratum: {condition}, rotation_zero={is_zero}')
            for index in (0, len(ordered) // 2, len(ordered) - 1):
                chosen.add(ordered[index]['prompt_id'])
    # Order in the original semantic manifest is fixed and reused on resume.
    return [row['prompt_id'] for row in prompts if row['prompt_id'] in chosen]


def runtime_versions() -> dict:
    result = {'python': platform.python_version()}
    for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'accelerate', 'numpy', 'pandas', 'pyarrow', 'scipy'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def snapshot_inventory(snapshot: Path, revision: str) -> dict:
    if not snapshot.is_dir() or snapshot.name != revision:
        raise ValueError('Model path must be the exact pinned HF snapshot directory ending in its revision')
    files = [p for p in snapshot.rglob('*') if p.is_file()]
    if not files or not (snapshot / 'config.json').is_file():
        raise ValueError('Model snapshot is missing its configuration')
    weight_files = [p for p in files if p.suffix == '.safetensors' or p.name.startswith('pytorch_model') and p.suffix == '.bin']
    if not weight_files:
        raise ValueError('Model snapshot contains no full-precision weight files')
    inventory = {str(p.relative_to(snapshot)): {'bytes': p.stat().st_size, 'sha256': file_sha256(p)}
                 for p in sorted(files)}
    # Sharded checkpoints must be complete, including each index-referenced file.
    for index_path in snapshot.glob('*.index.json'):
        record = json.loads(index_path.read_text())
        for name in set(record.get('weight_map', {}).values()):
            if name not in inventory:
                raise ValueError(f'Model checkpoint shard missing: {name}')
    return {'revision': revision, 'files': inventory,
            'weight_bytes': sum(p.stat().st_size for p in weight_files)}


def import_historical_scorer(repo: Path):
    source = repo / 'experiments/llm/analysis/v23/src'
    sys.path.insert(0, str(source))
    module = importlib.import_module('pier_llm_v23.inference')
    if Path(module.__file__).resolve() != (source / 'pier_llm_v23/inference.py').resolve():
        raise ValueError('Historical scorer imported from an unexpected checkout')
    return module


def validate_score_rows(rows: list[dict], by_id: dict[str, dict], rendered_by_id: dict[str, dict], model_id: str, revision: str) -> None:
    import numpy as np
    seen = set()
    for row in rows:
        pid = str(row['prompt_id'])
        if pid in seen or pid not in by_id:
            raise ValueError(f'Duplicate/unexpected scored prompt: {pid}')
        seen.add(pid)
        source = by_id[pid]
        if row['schema_version'] != SCHEMA or row['model_id'] != model_id or row['model_revision'] != revision:
            raise ValueError(f'Score schema/model/revision mismatch: {pid}')
        for name in ('base_question_id', 'category', 'semantic_condition', 'condition', 'family', 'dose', 'track', 'permutation_id', 'option_count', 'semantic_to_visible', 'visible_to_semantic', 'candidate_continuations'):
            observed = row[name].tolist() if hasattr(row[name], 'tolist') else row[name]
            expected = source[name]
            if name == 'track' and expected is None and (observed is None or isinstance(observed, float) and np.isnan(observed)):
                continue
            if observed != expected:
                raise ValueError(f'Score {name} differs from semantic manifest: {pid}')
        if row['canonical_prompt_hash'] != source['canonical_content_sha256'] or row['rendered_prompt_hash'] != rendered_by_id[pid]['rendered_prompt_sha256']:
            raise ValueError(f'Score prompt hash mismatch: {pid}')
        n = int(source['option_count'])
        visible = np.asarray(row['visible_label_probabilities'], dtype=float)
        semantic = np.asarray(row['semantic_remapped_probabilities'], dtype=float)
        likelihood = np.asarray(row['candidate_sequence_log_likelihoods'], dtype=float)
        if any(v.shape != (n,) or not np.isfinite(v).all() for v in (visible, semantic, likelihood)):
            raise ValueError(f'Score lacks a complete finite option vector: {pid}')
        if np.any(visible < 0) or np.any(visible > 1) or abs(visible.sum() - 1) > 1e-10:
            raise ValueError(f'Invalid option probabilities: {pid}')
        remapped = np.zeros(n)
        remapped[np.asarray(source['visible_to_semantic'])] = visible
        if not np.allclose(semantic, remapped, atol=1e-12, rtol=0):
            raise ValueError(f'Semantic score mapping mismatch: {pid}')
        logits = np.exp(likelihood - likelihood.max())
        if not np.allclose(visible, logits / logits.sum(), atol=1e-12, rtol=0):
            raise ValueError(f'Probabilities differ from full continuation likelihoods: {pid}')
        if row['gold_semantic_option'] != source['semantic_answer_index']:
            raise ValueError(f'Score gold identity mismatch: {pid}')
        if abs(float(row['gold_semantic_probability']) - semantic[int(source['semantic_answer_index'])]) > 1e-12:
            raise ValueError(f'Score gold probability mismatch: {pid}')


def scan_completed(directory: Path, identity: str, by_id: dict, rendered_by_id: dict, model_id: str, revision: str, manifest_hash: str | None = None) -> tuple[set[str], list[dict]]:
    """Only atomically committed, hash-valid shards can contribute completed rows."""
    import pandas as pd
    completed = set()
    metadata = []
    for sidecar in sorted(directory.glob('shard_*.meta.json')):
        meta = json.loads(sidecar.read_text())
        if meta.get('input_fingerprint') != identity:
            raise ValueError(f'Shard input/config/code/runtime identity mismatch: {sidecar}')
        expected_metadata = {'schema_version': SCHEMA, 'model_id': model_id,
                             'model_revision': revision, 'measurement_origin': 'fresh_endpoint_scoring'}
        if any(meta.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f'Shard schema/model/origin metadata mismatch: {sidecar}')
        if manifest_hash is not None and meta.get('manifest_sha256') != manifest_hash:
            raise ValueError(f'Shard semantic manifest identity mismatch: {sidecar}')
        data = sidecar.with_suffix('').with_suffix('.parquet')
        marker = data.with_suffix('.complete.json')
        if not marker.is_file():
            # A sidecar without its final commit marker is an interrupted write.
            continue
        committed = json.loads(marker.read_text())
        if committed.get('complete') is not True or committed.get('row_count') != meta['row_count'] or not data.is_file() or file_sha256(data) != meta['sha256'] or committed.get('sha256') != meta['sha256'] or committed.get('input_fingerprint') != identity:
            raise ValueError(f'Shard checksum/commit mismatch: {data}')
        rows = pd.read_parquet(data).to_dict('records')
        if len(rows) != meta['row_count'] or [str(r['prompt_id']) for r in rows] != meta['prompt_ids']:
            raise ValueError(f'Shard row coverage mismatch: {data}')
        validate_score_rows(rows, by_id, rendered_by_id, model_id, revision)
        ids = set(meta['prompt_ids'])
        if completed & ids:
            raise ValueError(f'Duplicate prompts across score shards: {data}')
        completed.update(ids)
        metadata.append(meta)
    return completed, metadata


def publish_shard(directory: Path, rows: list[dict], identity: str, manifest_hash: str,
                  *, model_id: str, revision: str, execution: dict) -> Path:
    import pandas as pd
    directory.mkdir(parents=True, exist_ok=True)
    indices = [int(p.name.split('_')[1].split('.')[0]) for p in directory.glob('shard_*.*')]
    index = max(indices, default=-1) + 1
    path = directory / f'shard_{index:05d}.parquet'
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sha = file_sha256(path)
    meta = {'schema_version': SCHEMA, 'input_fingerprint': identity, 'model_id': model_id,
            'model_revision': revision, 'manifest_sha256': manifest_hash, 'row_count': len(rows),
            'sha256': sha, 'prompt_ids': [str(r['prompt_id']) for r in rows],
            'measurement_origin': 'fresh_endpoint_scoring', 'created_at': utc_now(), 'execution': execution}
    atomic_json(path.with_suffix('.meta.json'), meta)
    atomic_json(path.with_suffix('.complete.json'), {'input_fingerprint': identity, 'sha256': sha,
                'row_count': len(rows), 'complete': True})
    return path


def _is_cuda_oom(exc: BaseException, torch) -> bool:
    return isinstance(exc, torch.OutOfMemoryError)


def run(args) -> dict:
    repo, root, snapshot = (p.expanduser().resolve() for p in (args.source_repo, args.run_root, args.model_path))
    if root == repo or any((repo / prefix) == root or (repo / prefix) in root.parents for prefix in ('data', 'paper', 'experiments', 'manifests')):
        raise ValueError('Fresh scoring outputs must be outside source/frozen/paper paths')
    if any(p.name.startswith('PIER_LLM_ECOSYSTEM_') for p in [root, *root.parents]):
        raise ValueError('Never write into a historical experiment directory')
    roster_path = repo / 'experiments/llm/analysis/v23/configs/resolved_models_v23.json'
    matches = [r for r in json.loads(roster_path.read_text())['models'] if r['id'] == args.model_id]
    if len(matches) != 1:
        raise ValueError('Model ID must occur exactly once in the fixed original roster')
    record = dict(matches[0])
    original_adapter = record['model_adapter']
    if args.model_adapter:
        record['model_adapter'] = args.model_adapter
    record['local_cache_path'] = str(snapshot)
    manifest_path = root / 'data/permutations/score_execution_manifest.jsonl'
    if not manifest_path.exists():
        manifest_path = root / 'data/permutations/score_semantic_manifest.jsonl'
    rendered_path = root / 'data/permutations/rendered' / f'{slug(args.model_id)}.jsonl'
    prompts, rendered = read_jsonl(manifest_path), read_jsonl(rendered_path)
    design = validate_design(prompts, rendered)
    by_id = {r['prompt_id']: r for r in prompts}
    rendered_by_id = {r['prompt_id']: r for r in rendered}
    # Check historical renderer identity before loading model weights. New
    # execution flags can change the semantic manifest hash, never rendering.
    rendering_match = file_sha256(rendered_path) == record['v23_rendered_prompt_sha256']
    if not rendering_match and not args.interface_change_reason:
        raise ValueError('Rendered prompt file differs from archived interface; explicit --interface-change-reason required')
    model_inventory = snapshot_inventory(snapshot, record['revision'])
    import torch
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Fresh model scoring requires an explicitly selected available CUDA device')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    hardware = {'name': torch.cuda.get_device_name(device),
                'capability': list(torch.cuda.get_device_capability(device)),
                'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
                'torch_cuda_version': torch.version.cuda}
    identity_record = {'schema_version': 'xwvg_fresh_scoring_identity_v1', 'model_id': args.model_id,
        'model_revision': record['revision'], 'precision': 'bfloat16', 'candidate_definition': 'summed_full_continuation_log_likelihood',
        'model_adapter': record['model_adapter'], 'historical_model_adapter': original_adapter,
        'historical_rendered_prompt_sha256': record['v23_rendered_prompt_sha256'],
        'historical_rendering_match': rendering_match, 'interface_change_reason': args.interface_change_reason,
        'model_inventory': model_inventory, 'semantic_manifest_sha256': file_sha256(manifest_path),
        'rendered_prompt_sha256': file_sha256(rendered_path), 'versions': runtime_versions(), 'hardware': hardware,
        'source_sha256': {relative: file_sha256(repo/relative) for relative in SOURCE_FILES},
        'driver_sha256': file_sha256(Path(__file__)), 'all_rotations_scored': True,
        'measurement_origin': 'fresh_endpoint_scoring', 'historical_cache_restoration': False}
    identity = hashlib.sha256(canonical(identity_record)).hexdigest()
    status = root / 'status'
    metadata_path = status / f'fresh_score_{slug(args.model_id)}_identity.json'
    if metadata_path.exists() and not args.resume:
        raise ValueError('A score run already exists; supply --resume with identical identities')
    atomic_json(metadata_path, {'input_fingerprint': identity, **identity_record}, immutable=True)
    directory = root / 'outputs/raw_scores' / slug(args.model_id)
    completed, metas = scan_completed(directory, identity, by_id, rendered_by_id, args.model_id, record['revision'], identity_record['semantic_manifest_sha256'])
    pilot_ids = fixed_pilot_ids(prompts, rendered_by_id)
    pilot_record = {'input_fingerprint': identity, 'policy': 'each condition x rotation-zero/nonzero x min/median/max token length',
                    'prompt_ids': pilot_ids, 'row_count': len(pilot_ids), 'outcome_blind': True}
    atomic_json(status / f'fresh_score_{slug(args.model_id)}_pilot_design.json', pilot_record, immutable=True)
    pilot_report_path = status / f'fresh_score_{slug(args.model_id)}_pilot.json'
    previous_pilot = json.loads(pilot_report_path.read_text()) if pilot_report_path.is_file() else None
    valid_pilot = (previous_pilot is not None and previous_pilot.get('input_fingerprint') == identity
                   and previous_pilot.get('status') == 'complete' and set(pilot_ids) <= completed
                   and previous_pilot.get('pilot_measured_prompt_count') == len(pilot_ids)
                   and previous_pilot.get('scoring_seconds', 0) > 0)
    if args.stage == 'score' and not valid_pilot:
        raise ValueError('A successful identity-matched measured pilot is required before the full score stage')
    if args.stage == 'pilot' and valid_pilot:
        print(json.dumps({'model_id':args.model_id,'stage':'pilot','status':'complete',
                          'reused_measured_pilot':True,'completed_rows':len(completed)}),flush=True)
        return previous_pilot
    selected_ids = set(pilot_ids) if args.stage == 'pilot' else set(by_id)
    todo = [r for r in prompts if r['prompt_id'] in selected_ids and r['prompt_id'] not in completed]
    all_complete_before = len(completed)
    start = time.perf_counter()
    load_seconds = 0.0
    batch_size = args.batch_size
    events, batches = [], []
    scored_prompt_count = scored_sequences = scored_candidate_tokens = 0
    scorer = None
    if todo:
        scorer = import_historical_scorer(repo)
        load_start = time.perf_counter()
        model, tokenizer = scorer._load_model(record, args.device)
        load_seconds = time.perf_counter() - load_start
        torch.cuda.reset_peak_memory_stats(device)
        offset = 0
        pending = []
        pending_batches = []
        while offset < len(todo):
            current = todo[offset: offset + batch_size]
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            try:
                rows = scorer.score_prompt_batch(model, tokenizer, current, rendered_by_id, device=args.device)
                torch.cuda.synchronize(device)
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    atomic_json(status / f'fresh_score_{slug(args.model_id)}_resource_failure.json',
                                {'model_id': args.model_id, 'input_fingerprint': identity,
                                 'prompt_id': current[0]['prompt_id'], 'batch_size': 1,
                                 'error': 'CUDA OOM at one prompt; no truncation or quantization performed',
                                 'completed_rows': all_complete_before + scored_prompt_count - len(pending)})
                    raise
                previous = batch_size
                batch_size = max(1, batch_size // 2)
                events.append({'event': 'cuda_oom_batch_reduction', 'from': previous, 'to': batch_size,
                               'prompt_id': current[0]['prompt_id'], 'time': utc_now()})
                gc.collect()
                torch.cuda.empty_cache()
                continue
            elapsed = time.perf_counter() - before
            for row in rows:
                row.update(model_id=args.model_id, model_revision=record['revision'], score_origin='fresh_endpoint_scoring', manifest_sha256=identity_record['semantic_manifest_sha256'])
            if [r['prompt_id'] for r in rows] != [r['prompt_id'] for r in current]:
                raise ValueError('Scorer changed batch prompt order or coverage')
            validate_score_rows(rows, by_id, rendered_by_id, args.model_id, record['revision'])
            sequence_count = sum(int(r['option_count']) for r in current)
            token_count = sum(int(r['option_count']) * int(r['prompt_token_count']) + sum(r['candidate_token_counts']) for r in rows)
            observation = {'prompt_ids': [r['prompt_id'] for r in current], 'seconds': elapsed,
                           'batch_size': len(current), 'candidate_continuations': sequence_count,
                           'candidate_tokens_including_prompt': token_count,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                           'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
            batches.append(observation)
            pending_batches.append(observation)
            pending.extend(rows)
            offset += len(current)
            scored_prompt_count += len(current)
            scored_sequences += sequence_count
            scored_candidate_tokens += token_count
            if len(pending) >= args.shard_size or offset == len(todo):
                shard = publish_shard(directory, pending, identity, identity_record['semantic_manifest_sha256'],
                   model_id=args.model_id, revision=record['revision'],
                   execution={'stage': args.stage, 'requested_batch_size': args.batch_size,
                              'effective_batch_size': batch_size, 'batches': pending_batches, 'oom_events': events})
                completed.update(r['prompt_id'] for r in pending)
                print(json.dumps({'model_id':args.model_id, 'stage':args.stage, 'completed_rows':len(completed),
                                  'required_rows':len(by_id), 'shard':str(shard), 'batch_size':batch_size}),flush=True)
                pending, pending_batches = [], []
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    # Readback is mandatory, including when the invocation reused all scores.
    completed, metas = scan_completed(directory, identity, by_id, rendered_by_id, args.model_id, record['revision'], identity_record['semantic_manifest_sha256'])
    if not selected_ids <= completed:
        raise ValueError(f'Incomplete scoring stage; {len(selected_ids-completed)} selected prompts remain')
    new_scoring_seconds = sum(r['seconds'] for r in batches)
    if args.stage == 'pilot':
        # Reconstruct measurements from committed pilot shards if a process was
        # interrupted between committing scores and writing the pilot report.
        batches = [batch for meta in metas if meta['execution']['stage'] == 'pilot'
                   for batch in meta['execution']['batches']]
        measured_ids = [pid for batch in batches for pid in batch['prompt_ids']]
        if len(measured_ids) != len(set(measured_ids)) or set(measured_ids) != set(pilot_ids):
            raise ValueError('Pilot timing records do not cover exactly its fixed prompts')
        scored_candidate_tokens = sum(batch['candidate_tokens_including_prompt'] for batch in batches)
    scoring_seconds = sum(r['seconds'] for r in batches)
    full_prompt_tokens = sum(int(r['option_count']) * int(rendered_by_id[r['prompt_id']]['prompt_token_count']) for r in prompts)
    projection = None if not scored_candidate_tokens else scoring_seconds * full_prompt_tokens / scored_candidate_tokens
    report = {'model_id':args.model_id, 'model_revision':record['revision'], 'input_fingerprint':identity,
              'stage':args.stage, 'status':'complete', 'design':design, 'completed_rows':len(completed),
              'newly_scored_rows':scored_prompt_count, 'newly_scored_candidate_continuations':scored_sequences,
              'requested_batch_size':args.batch_size, 'final_batch_size':batch_size, 'load_seconds':load_seconds,
              'scoring_seconds':scoring_seconds, 'new_scoring_seconds':new_scoring_seconds,
              'pilot_measured_prompt_count':len(pilot_ids) if args.stage == 'pilot' else None,
              'total_committed_scoring_seconds':sum(b['seconds'] for meta in metas for b in meta['execution']['batches']),
              'wall_seconds':time.perf_counter()-start,
              'projection_full_scoring_seconds':projection,
              'projection_note':'Rough token-normalized estimate from fixed pilot; ignores future padding/batch/OOM variation and loading.',
              'hardware':hardware, 'versions':identity_record['versions'], 'oom_events':events, 'batches':batches,
              'fresh_measurements':True, 'historical_cache_restoration':False,
              'shard_count':len(metas), 'created_at':utc_now()}
    atomic_json(status / f'fresh_score_{slug(args.model_id)}_{"pilot" if args.stage == "pilot" else "complete"}.json', report)
    print(json.dumps({k:report[k] for k in ['model_id','stage','status','completed_rows','newly_scored_rows','wall_seconds']}),flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--model-adapter', choices=['causal_lm', 'image_text_to_text'],
                        help='Explicit recorded correction if the historical adapter metadata cannot load the pinned class')
    parser.add_argument('--interface-change-reason',
                        help='Explicitly document a reviewed fresh-renderer difference if historical rendering cannot be restored')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--shard-size', type=int, default=128)
    parser.add_argument('--stage', choices=['pilot', 'score'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.shard_size < 1:
        parser.error('batch-size and shard-size must be positive')
    try:
        run(args)
        return 0
    except Exception as exc:
        print(json.dumps({'status':'failed','model_id':args.model_id,'error':repr(exc)}), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
