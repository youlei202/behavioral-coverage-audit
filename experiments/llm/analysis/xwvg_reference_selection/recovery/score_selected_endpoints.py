#!/usr/bin/env python3
"""Optional selected-position backend in a separate, fully identified score run.

The orchestration below is a direct copy of frozen score_endpoints.run/main,
with only acceptance validation, explicit backend identity, and the scorer call
changed. Shared validators/writers remain imported from that unchanged file.
There is no runtime source transformation or monkeypatching. All eight models
must pass the separate exact GPU comparison before either pilot or full scoring.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import score_endpoints as reference
from score_endpoints import (
    SOURCE_FILES, atomic_json, canonical, file_sha256, fixed_pilot_ids,
    import_historical_scorer, publish_shard, read_jsonl, runtime_versions,
    scan_completed, slug, snapshot_inventory, utc_now, validate_design,
    validate_score_rows,
)

HERE = Path(__file__).resolve().parent
BACKEND_NAME = 'selected_positions'


def _json(path):
    return json.loads(Path(path).read_text())


def _strict_comparison(value, label):
    if value.get('exact') is not True or value.get('errors') != []:
        raise ValueError(f'Exact GPU comparison failed: {label}')
    left, right = value.get('baseline_sha256'), value.get('candidate_sha256')
    if (not isinstance(left,str) or len(left) != 64 or left != right
            or any(c not in '0123456789abcdef' for c in left)):
        raise ValueError(f'Exact GPU comparison hashes disagree: {label}')


def validate_acceptance_report(path: Path, repo: Path, root: Path) -> dict:
    """Validate saved all-model GPU proof and unchanged baseline before any load."""
    path, repo, root = Path(path).resolve(), Path(repo).resolve(), Path(root).resolve()
    proof = _json(path)
    if (proof.get('status') != 'passed' or proof.get('strict_all_eight_models_passed') is not True
            or proof.get('baseline_artifacts_unchanged') is not True
            or proof.get('changed_artifacts') != []):
        raise ValueError('A passed exact all-eight-model GPU acceptance report is required')
    roster = _json(repo/'experiments/llm/analysis/v23/configs/resolved_models_v23.json')['models']
    expected = {r['id']:r['revision'] for r in roster}
    reports = proof.get('models',[])
    if (len(expected) != 8 or len(reports) != 8
            or {r.get('model_id') for r in reports} != set(expected)):
        raise ValueError('GPU acceptance must contain exactly the fixed eight-model roster')
    required_sources = [repo/'experiments/llm/analysis/v23/src/pier_llm_v23/inference.py',
                        HERE/'score_endpoints.py', HERE/'score_math.py', HERE/'validate_score_math.py']
    hashes = {str(Path(p).resolve()):h for p,h in proof.get('source_sha256',{}).items()}
    for source in required_sources:
        if hashes.get(str(source.resolve())) != file_sha256(source):
            raise ValueError(f'Accepted GPU comparison source changed or absent: {source}')
    artifacts = proof.get('baseline_artifact_sha256',{})
    if not artifacts:
        raise ValueError('GPU acceptance lacks baseline artifact identities')
    for item, expected_hash in artifacts.items():
        source = Path(item)
        if not source.is_file() or file_sha256(source) != expected_hash:
            raise ValueError(f'Baseline artifact changed after exact GPU validation: {source}')
    identities, model_proofs, baseline_roots = {}, {}, set()
    for summary in reports:
        model = summary['model_id']
        if (summary.get('model_revision') != expected[model] or summary.get('status') != 'passed'
                or summary.get('exact_candidate_equality') is not True
                or summary.get('exact_baseline_repeat') is not True or summary.get('prompt_count') != 42):
            raise ValueError(f'Incomplete exact GPU proof: {model}')
        detail_path = path.parent/f'{slug(model)}.json'
        detail = _json(detail_path)
        if {k:v for k,v in detail.items() if k not in ('batches','traceback')} != summary:
            raise ValueError(f'GPU acceptance summary/detail disagree: {model}')
        names = [Path(p) for p in artifacts if p.endswith(f'/status/fresh_score_{slug(model)}_identity.json')]
        if len(names) != 1:
            raise ValueError(f'GPU acceptance has no unique baseline identity: {model}')
        identity_path = names[0]
        baseline_root = identity_path.parent.parent.resolve()
        baseline_roots.add(baseline_root)
        identity = _json(identity_path)
        if (identity.get('model_id') != model or identity.get('model_revision') != expected[model]
                or identity.get('input_fingerprint') != hashlib.sha256(canonical(
                    {k:v for k,v in identity.items() if k != 'input_fingerprint'})).hexdigest()):
            raise ValueError(f'Baseline model identity is invalid: {model}')
        design_path = baseline_root/f'status/fresh_score_{slug(model)}_pilot_design.json'
        pilot_path = baseline_root/f'status/fresh_score_{slug(model)}_pilot.json'
        if str(design_path) not in artifacts or str(pilot_path) not in artifacts:
            raise ValueError(f'GPU comparison does not bind the fixed pilot records: {model}')
        design,pilot = _json(design_path),_json(pilot_path)
        if (design.get('input_fingerprint') != identity['input_fingerprint']
                or pilot.get('input_fingerprint') != identity['input_fingerprint']
                or design.get('outcome_blind') is not True or design.get('row_count') != 42
                or pilot.get('status') != 'complete' or pilot.get('pilot_measured_prompt_count') != 42):
            raise ValueError(f'Baseline fixed pilot is invalid: {model}')
        batches = detail.get('batches',[])
        tested_ids = [pid for batch in batches for pid in batch.get('prompt_ids',[])]
        if (len(tested_ids) != 42 or len(set(tested_ids)) != 42
                or set(tested_ids) != set(design['prompt_ids'])
                or [batch['prompt_ids'] for batch in batches] != [batch['prompt_ids'] for batch in pilot['batches']]):
            raise ValueError(f'GPU comparison did not replay every fixed pilot batch: {model}')
        for batch in batches:
            _strict_comparison(batch.get('baseline_vs_candidate',{}), model)
            _strict_comparison(batch.get('cached_vs_repeated_baseline',{}), model)
            if batch['baseline_vs_candidate']['baseline_sha256'] != batch['cached_vs_repeated_baseline']['candidate_sha256']:
                raise ValueError(f'GPU comparisons do not bind the same baseline rows: {model}')
            for backend in ['baseline','candidate']:
                timing=batch.get('timings',{}).get(backend,{})
                if timing.get('prompt_count') != len(batch['prompt_ids']) or timing.get('seconds',0) <= 0:
                    raise ValueError(f'GPU comparison lacks real timed batches: {model}')
        identities[model] = identity
        model_proofs[model] = {'detail_path':str(detail_path),'detail_sha256':file_sha256(detail_path),
                              'baseline_identity_path':str(identity_path),
                              'baseline_identity_sha256':file_sha256(identity_path),
                              'baseline_input_fingerprint':identity['input_fingerprint'],
                              'exact_candidate_equality':True,'exact_baseline_repeat':True,'pilot_prompt_count':42}
    if len(baseline_roots) != 1:
        raise ValueError('GPU proof mixes baseline run roots')
    baseline_root = next(iter(baseline_roots))
    if root == baseline_root or root in baseline_root.parents or baseline_root in root.parents:
        raise ValueError('Selected-position scores require a disjoint new run root')
    # Read-only data/model paths may be shared, but every mutable score/status
    # destination must remain in the new namespace even through symlinks.
    for destination in [root/'status',root/'outputs', *(
            root/'outputs/raw_scores'/slug(model) for model in expected)]:
        if not destination.resolve().is_relative_to(root):
            raise ValueError(f'Selected-position writable directory escapes its run root: {destination}')
    return {'report_path':str(path),'report_sha256':file_sha256(path),
            'source_sha256':hashes,'baseline_root':str(baseline_root),
            'baseline_identities':identities,'model_proofs':model_proofs,
            'versions':proof.get('versions'), 'gpu_name':proof.get('gpu_name')}


def selected_backend_identity(acceptance: dict, record: dict, args) -> dict:
    """Bind implementation, acceptance evidence and tested input/runtime identities."""
    baseline=acceptance['baseline_identities'][args.model_id]
    for key in ['model_revision','precision','candidate_definition','model_adapter',
                'model_inventory','semantic_manifest_sha256','rendered_prompt_sha256',
                'source_sha256','hardware']:
        if record[key] != baseline[key]:
            raise ValueError(f'Selected backend differs from GPU-tested {key}')
    if record['versions'] != acceptance['versions'] or record['hardware']['name'] != acceptance['gpu_name']:
        raise ValueError('Selected backend runtime/hardware differs from exact GPU acceptance')
    return {'scoring_backend':{
        'name':BACKEND_NAME, 'reference_driver_sha256':file_sha256(HERE/'score_endpoints.py'),
        'score_math_sha256':file_sha256(HERE/'score_math.py'),
        'selected_driver_sha256':file_sha256(Path(__file__)),
        'acceptance_report':acceptance['report_path'],
        'acceptance_report_sha256':acceptance['report_sha256'],
        'acceptance_source_sha256':acceptance['source_sha256'],
        'acceptance_model_proofs':acceptance['model_proofs'],
        'all_eight_models_exact_gpu_proof':True,
        'baseline_root':acceptance['baseline_root'],
        'orchestration':'direct copied baseline run; explicit acceptance and identity additions; scorer call replaced',
        'batch_policy':'separately measured fixed pilot and standard OOM batch reduction; batch choice recorded as execution metadata',
    }}


def load_selected_backend(repo: Path, acceptance: dict):
    source=HERE/'score_math.py'
    if file_sha256(source) != acceptance['source_sha256'][str(source.resolve())]:
        raise ValueError('Selected-position backend changed after acceptance validation')
    # The original scorer has already been imported and path-checked by run().
    spec=importlib.util.spec_from_file_location('xwvg_selected_positions_backend',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args) -> dict:
    repo, root, snapshot = (p.expanduser().resolve() for p in (args.source_repo, args.run_root, args.model_path))
    acceptance = validate_acceptance_report(args.acceptance_report, repo, root)
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
    identity_record.update(selected_backend_identity(acceptance, identity_record, args))
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
        selected_backend = load_selected_backend(repo, acceptance)
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
                rows = selected_backend.score_prompt_batch(model, tokenizer, current, rendered_by_id, device=args.device)
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
    parser.add_argument('--acceptance-report', type=Path, required=True,
                        help='Completed strict all-eight-model GPU equality summary JSON')
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
