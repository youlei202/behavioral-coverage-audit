#!/usr/bin/env python3
"""Resource-capped, batch-one execution of the unchanged historical scorer.

The isolated pilot measures the fixed 42 prompts and an outcome-blind maximum
workload probe. It makes no equality claim against differently batched scores.
The score stage invokes the frozen score_endpoints.py through runpy after setting
the PyTorch allocator cap; it does not patch model forward or scoring functions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score_endpoints import (SOURCE_FILES, atomic_json, canonical, file_sha256,
    fixed_pilot_ids, import_historical_scorer, read_jsonl, runtime_versions, slug,
    snapshot_inventory, utc_now, validate_design, validate_score_rows)
from validate_remote_reference import (checked_output, gpu_details, load,
    require_identity, saved_pilot_batches, seal_proof)

MIB = 1024 ** 2
MAX_BUDGET_MIB = 70 * 1024
MIN_RESERVE_MIB = 16 * 1024


def resource_config(budget_mib, reserve_mib, total_bytes):
    if isinstance(budget_mib, bool) or not isinstance(budget_mib, int) or not 0 < budget_mib <= MAX_BUDGET_MIB:
        raise ValueError('Shared allocator budget must be a positive integer at most 71680 MiB')
    if isinstance(reserve_mib, bool) or not isinstance(reserve_mib, int) or reserve_mib < MIN_RESERVE_MIB:
        raise ValueError('Shared free-memory reserve must be at least 16384 MiB')
    if (budget_mib + reserve_mib) * MIB >= total_bytes:
        raise ValueError('Budget plus reserve must fit below total GPU memory')
    return {'batch_size': 1, 'memory_budget_mib': budget_mib, 'reserve_mib': reserve_mib,
            'allocator_limit_bytes': budget_mib * MIB,
            'allocator_fraction': budget_mib * MIB / total_bytes,
            'allocator_cap_scope': 'PyTorch CUDA caching allocator; other jobs are unchanged',
            'required_external_monitor': 'Controller stops only its own process group if free memory falls below reserve'}


def gpu_memory(gpu_uuid):
    result = subprocess.run(['nvidia-smi', '--id=' + gpu_uuid,
        '--query-gpu=uuid,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    rows = [[s.strip() for s in line.split(',')] for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != gpu_uuid:
        raise ValueError('GPU memory query did not resolve the selected device uniquely')
    _, total, free, used, utilization = rows[0]
    return {'uuid': gpu_uuid, 'total_mib': int(total), 'free_mib': int(free),
            'used_mib': int(used), 'utilization_percent': int(utilization), 'observed_at': utc_now()}


def require_headroom(observation, config, *, before_load=False):
    required = config['reserve_mib'] + (config['memory_budget_mib'] if before_load else 0)
    if observation['free_mib'] < required:
        raise RuntimeError(f'Shared GPU free memory {observation["free_mib"]} MiB is below required {required} MiB')


class ReserveWatchdog:
    """Terminate this process only if its shared-resource contract is breached."""
    def __init__(self, observe, config, record_failure, terminate=os._exit, interval=2):
        self.observe, self.config = observe, config
        self.record_failure, self.terminate, self.interval = record_failure, terminate, interval
        self.finished = threading.Event()
        self.thread = None

    def check_once(self):
        observation = None
        try:
            observation = self.observe()
            require_headroom(observation, self.config)
            return True
        except Exception as exc:
            failure = {'status': 'failed', 'resource_failure': 'reserve_breach' if observation else 'memory_query_failed',
                       'error': repr(exc), 'observation': observation, 'resource_config': self.config,
                       'exit_code': 86, 'observed_at': utc_now()}
            try:
                self.record_failure(failure)
            finally:
                # A signal handled by Python can wait for a long C++ forward.
                # _exit immediately ends only our process and its CUDA context.
                self.terminate(86)
            return False

    def start(self):
        def monitor():
            while not self.finished.wait(self.interval):
                if not self.check_once():
                    return
        self.thread = threading.Thread(target=monitor, name='shared-memory-reserve', daemon=True)
        self.thread.start()

    def stop(self):
        self.finished.set()
        if self.thread is not None:
            self.thread.join()


def largest_workload(prompts, rendered, candidate_lengths):
    """Choose by padded candidate-sequence size only, never by score outcomes."""
    if not prompts:
        raise ValueError('No prompts for the resource probe')
    ranked = []
    for row in prompts:
        count = int(row['option_count'])
        maximum = max(candidate_lengths[count])
        workload = count * (int(rendered[row['prompt_id']]['prompt_token_count']) + maximum)
        ranked.append((-workload, row['prompt_id'], maximum))
    negative, pid, maximum = min(ranked)
    return {'prompt_id': pid, 'estimated_workload': -negative,
            'max_candidate_token_count': maximum,
            'formula': 'option_count * (prompt_token_count + max_candidate_token_count)',
            'tie_break': 'lexicographically smallest prompt_id', 'outcome_blind': True}


def atomic_rows(path, rows):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    return {'path': str(path), 'sha256': file_sha256(path), 'row_count': len(rows)}


def identity_record(record, inventory, manifest_sha, rendered_sha, sources, versions, hardware):
    return {'schema_version': 'xwvg_fresh_scoring_identity_v1', 'model_id': record['id'],
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


def baseline_command(args):
    return [str(HERE / 'score_endpoints.py'), '--source-repo', str(args.source_repo),
            '--run-root', str(args.run_root), '--model-id', args.model_id,
            '--model-path', str(args.model_path), '--device', args.device,
            '--batch-size', '1', '--stage', 'score', '--resume']


def invoke_baseline(args):
    command = baseline_command(args)
    old_argv = sys.argv
    try:
        sys.argv = command
        try:
            runpy.run_path(command[0], run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError(f'Frozen reference driver exited with status {exc.code}') from exc
    finally:
        sys.argv = old_argv


def committed_shards(root, model_id):
    result = {}
    for meta_path in sorted((root / 'outputs/raw_scores' / slug(model_id)).glob('shard_*.meta.json')):
        data = meta_path.with_suffix('').with_suffix('.parquet')
        marker = data.with_suffix('.complete.json')
        if not marker.is_file():
            continue
        meta, commit = load(meta_path), load(marker)
        sha = file_sha256(data)
        if commit.get('complete') is not True or meta.get('sha256') != sha or commit.get('sha256') != sha:
            raise ValueError('Invalid committed score shard during provenance capture')
        result[str(data)] = {'sha256': sha, 'meta_sha256': file_sha256(meta_path),
            'commit_sha256': file_sha256(marker), 'row_count': meta['row_count'],
            'input_fingerprint': meta['input_fingerprint'], 'execution': meta['execution']}
    return result


def verify_pilot(proof, expected, by_id, rendered, pilot_ids):
    if proof.get('proof_sha256') != seal_proof(proof)['proof_sha256']:
        raise ValueError('Shared pilot proof checksum differs')
    for key, value in expected.items():
        if proof.get(key) != value:
            raise ValueError(f'Shared pilot provenance mismatch: {key}')
    if (proof.get('status') != 'passed' or proof.get('prompt_count') != 42
        or proof.get('finite_all_outputs') is not True or proof.get('worst_case_probe_passed') is not True
        or proof.get('immutable_inputs_sources_unchanged') is not True
        or proof.get('exact_equality_to_old_batches_claimed') is not False):
        raise ValueError('Shared pilot is incomplete or makes an invalid cross-batch equality claim')
    import pandas as pd
    for kind in ('pilot_rows', 'resource_probe_rows'):
        artifact = proof[kind]
        path = Path(artifact['path'])
        if not path.is_file() or file_sha256(path) != artifact['sha256']:
            raise ValueError(f'Shared pilot {kind} artifact is missing or changed')
        rows = pd.read_parquet(path).to_dict('records')
        if len(rows) != artifact['row_count']:
            raise ValueError(f'Shared pilot {kind} row count differs')
        validate_score_rows(rows, by_id, rendered, expected['model_id'], expected['model_revision'])
        expected_ids = pilot_ids if kind == 'pilot_rows' else [proof['worst_case_probe']['prompt_id']]
        if [r['prompt_id'] for r in rows] != expected_ids:
            raise ValueError(f'Shared pilot {kind} coverage differs')


def run(args):
    args.source_repo, args.run_root, args.model_path = (p.expanduser().resolve() for p in
        (args.source_repo, args.run_root, args.model_path))
    repo, root, snapshot = args.source_repo, args.run_root, args.model_path
    output = checked_output(args.output, repo, root, snapshot)
    rows_path = output.with_suffix('.rows.parquet')
    stress_path = output.with_suffix('.resource.rows.parquet')
    failure_path = output.with_suffix('.resource_failure.json')
    if rows_path.exists() or stress_path.exists() or failure_path.exists():
        raise ValueError('Use fresh output paths for shared measurements')
    if args.stage == 'score' and args.pilot_proof is None:
        raise ValueError('--pilot-proof is required for shared score execution')
    report = {'schema_version': 'xwvg_shared_scoring_execution_v1', 'status': 'running',
        'stage': args.stage, 'model_id': args.model_id, 'hostname': platform.node(),
        'pid': os.getpid(), 'python': sys.executable, 'device': args.device,
        'batch_size': 1, 'memory_budget_mib': args.memory_budget_mib, 'reserve_mib': args.reserve_mib,
        'wrapper_sha256': file_sha256(Path(__file__)),
        'baseline_driver_sha256': file_sha256(HERE / 'score_endpoints.py'),
        'measurement_origin': 'fresh_shared_batch1_pilot' if args.stage == 'pilot' else 'fresh_endpoint_scoring',
        'exact_equality_to_old_batches_claimed': False,
        'batch_change_reason': 'Explicitly authorized shared GPU resource limit; original model/scoring/precision unchanged',
        'started_at': utc_now(), 'batches': [], 'prompt_count': 0,
        'finite_all_outputs': False, 'worst_case_probe_passed': False}
    immutable = {}
    model = tokenizer = torch = None
    before_shards = None
    collected = []
    watchdog = None
    try:
        name = slug(args.model_id)
        identity_path = root / 'status' / f'fresh_score_{name}_identity.json'
        pilot_path = root / 'status' / f'fresh_score_{name}_pilot.json'
        design_path = root / 'status' / f'fresh_score_{name}_pilot_design.json'
        identity, pilot, design = load(identity_path), load(pilot_path), load(design_path)
        matches = [r for r in load(repo / 'experiments/llm/analysis/v23/configs/resolved_models_v23.json')['models']
                   if r['id'] == args.model_id]
        if len(matches) != 1:
            raise ValueError('Shared model must be in the fixed original roster')
        record = dict(matches[0])
        manifest = root / 'data/permutations/score_execution_manifest.jsonl'
        if not manifest.is_file():
            manifest = root / 'data/permutations/score_semantic_manifest.jsonl'
        rendered_path = root / 'data/permutations/rendered' / f'{name}.jsonl'
        manifest_sha, rendered_sha = file_sha256(manifest), file_sha256(rendered_path)
        sources = {path: file_sha256(repo / path) for path in SOURCE_FILES}
        versions = runtime_versions()
        for key, value in {'versions': versions, 'source_sha256': sources,
            'driver_sha256': report['baseline_driver_sha256'],
            'semantic_manifest_sha256': manifest_sha, 'rendered_prompt_sha256': rendered_sha}.items():
            if identity.get(key) != value:
                raise ValueError(f'Shared {key} differs from frozen baseline identity')
        paths = [manifest, rendered_path, identity_path, pilot_path, design_path,
                 *(repo / p for p in SOURCE_FILES), HERE / 'score_endpoints.py',
                 HERE / 'validate_remote_reference.py', HERE / 'validate_score_math.py', Path(__file__)]
        immutable = {str(path): file_sha256(path) for path in paths}
        prompts, rendered_rows = read_jsonl(manifest), read_jsonl(rendered_path)
        validate_design(prompts, rendered_rows)
        by_id = {r['prompt_id']: r for r in prompts}
        rendered = {r['prompt_id']: r for r in rendered_rows}
        pilot_ids = fixed_pilot_ids(prompts, rendered)
        saved_pilot_batches(pilot, design, identity['input_fingerprint'], pilot_ids)
        inventory = snapshot_inventory(snapshot, record['revision'])
        import torch as torch_module
        torch = torch_module
        if not args.device.startswith('cuda:') or not torch.cuda.is_available():
            raise ValueError('Shared scoring requires an explicitly selected CUDA device')
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        hardware, gpu_uuid, driver_version = gpu_details(torch, device)
        current = identity_record(record, inventory, manifest_sha, rendered_sha, sources, versions, hardware)
        fingerprint = require_identity(identity, current)
        config = resource_config(args.memory_budget_mib, args.reserve_mib, hardware['total_memory_bytes'])
        initial_memory = gpu_memory(gpu_uuid)
        require_headroom(initial_memory, config, before_load=True)
        torch.cuda.set_per_process_memory_fraction(config['allocator_fraction'], device)
        applied_fraction = torch.cuda.get_per_process_memory_fraction(device)
        if abs(applied_fraction - config['allocator_fraction']) > 1e-12:
            raise ValueError('PyTorch allocator cap did not match the requested fraction')
        report.update(model_revision=record['revision'], input_fingerprint=fingerprint,
            hardware=hardware, versions=versions, gpu_uuid=gpu_uuid, driver_version=driver_version,
            source_sha256=sources, baseline_identity=identity, resource_config=config,
            initial_memory=initial_memory, applied_allocator_fraction=applied_fraction,
            pilot_prompt_ids=pilot_ids, immutable_input_source_sha256=immutable)
        execution_identity = {k: report[k] for k in ('wrapper_sha256', 'baseline_driver_sha256',
            'input_fingerprint', 'hostname', 'gpu_uuid', 'driver_version', 'resource_config', 'versions', 'hardware')}
        report['execution_fingerprint'] = hashlib.sha256(canonical(execution_identity)).hexdigest()
        report['memory_watchdog'] = {'interval_seconds': 2, 'failure_exit_code': 86,
            'failure_output': str(failure_path), 'scope': 'this process only'}
        atomic_json(output, report)

        def memory_failure(failure):
            evidence = {**execution_identity, **failure, 'model_id': args.model_id,
                        'model_revision': record['revision'], 'stage': args.stage,
                        'pid': os.getpid(), 'execution_fingerprint': report['execution_fingerprint'],
                        'execution_started_at': report['started_at'],
                        'execution_report_path': str(output),
                        'execution_report_sha256_at_failure': file_sha256(output),
                        'committed_shards_before_inventory_present': before_shards is not None,
                        'retained_shard_attribution': 'Compare current committed shards with committed_shards_before in the linked execution report; that inventory is published before the frozen driver starts'}
            atomic_json(failure_path, seal_proof(evidence))

        watchdog = ReserveWatchdog(lambda: gpu_memory(gpu_uuid), config, memory_failure)
        watchdog.start()
        if args.stage == 'pilot':
            original = import_historical_scorer(repo)
            record['local_cache_path'] = str(snapshot)
            started = time.perf_counter()
            model, tokenizer = original._load_model(record, args.device)
            report['model_load_seconds'] = time.perf_counter() - started
            counts = {int(r['option_count']): r['option_labels'] for r in prompts}
            lengths = {n: list(map(len, original._candidate_token_ids(tokenizer, labels))) for n, labels in counts.items()}
            stress = largest_workload(prompts, rendered, lengths)
            stress['was_in_fixed_pilot'] = stress['prompt_id'] in pilot_ids
            report['worst_case_probe'] = stress

            def measure(pid):
                before = gpu_memory(gpu_uuid)
                require_headroom(before, config)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                rows = original.score_prompt_batch(model, tokenizer, [by_id[pid]], rendered, device=args.device)
                torch.cuda.synchronize(device)
                timing = {'seconds': time.perf_counter() - started,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'free_mib_before': before['free_mib']}
                for row in rows:
                    row.update(model_id=args.model_id, model_revision=record['revision'],
                        score_origin='fresh_endpoint_scoring', manifest_sha256=manifest_sha)
                if len(rows) != 1 or rows[0]['prompt_id'] != pid:
                    raise ValueError('Original scorer changed single-prompt coverage')
                validate_score_rows(rows, by_id, rendered, args.model_id, record['revision'])
                after = gpu_memory(gpu_uuid)
                timing['free_mib_after'] = after['free_mib']
                require_headroom(after, config)
                return rows[0], timing

            observations = {}
            for pid in pilot_ids:
                row, timing = measure(pid)
                collected.append(row)
                observations[pid] = timing
                report['batches'].append({'prompt_ids': [pid], 'batch_size': 1, **timing})
                report['prompt_count'] = len(collected)
                atomic_json(output, report)
            report['pilot_rows'] = atomic_rows(rows_path, collected)
            if stress['was_in_fixed_pilot']:
                stress_row = next(row for row in collected if row['prompt_id'] == stress['prompt_id'])
                stress['timing'] = observations[stress['prompt_id']]
                stress['reused_fixed_pilot_observation'] = True
            else:
                stress_row, stress['timing'] = measure(stress['prompt_id'])
                stress['reused_fixed_pilot_observation'] = False
            report['resource_probe_rows'] = atomic_rows(stress_path, [stress_row])
            report.update(status='passed', finite_all_outputs=True, worst_case_probe_passed=True,
                first_batch_seconds=report['batches'][0]['seconds'],
                remaining_batches_seconds=sum(b['seconds'] for b in report['batches'][1:]),
                scoring_seconds=sum(b['seconds'] for b in report['batches']),
                peak_allocated_bytes=max(b['peak_allocated_bytes'] for b in report['batches'] + [stress['timing']]),
                peak_reserved_bytes=max(b['peak_reserved_bytes'] for b in report['batches'] + [stress['timing']]),
                baseline_score_outputs_written=False)
        else:
            proof_path = args.pilot_proof.expanduser().resolve()
            expected = {k: report[k] for k in ('model_id', 'model_revision', 'input_fingerprint', 'hostname',
                'gpu_uuid', 'driver_version', 'versions', 'hardware', 'batch_size', 'memory_budget_mib',
                'reserve_mib', 'wrapper_sha256', 'baseline_driver_sha256', 'execution_fingerprint')}
            verify_pilot(load(proof_path), expected, by_id, rendered, pilot_ids)
            report['pilot_proof'] = {'path': str(proof_path), 'sha256': file_sha256(proof_path)}
            report['baseline_command'] = baseline_command(args)
            before_shards = committed_shards(root, args.model_id)
            report['committed_shards_before'] = before_shards
            atomic_json(output, report)
            invoke_baseline(args)
            complete_path = root / 'status' / f'fresh_score_{name}_complete.json'
            complete = load(complete_path)
            if (complete.get('status') != 'complete' or complete.get('completed_rows') != len(prompts)
                or complete.get('input_fingerprint') != fingerprint or complete.get('final_batch_size') != 1):
                raise ValueError('Frozen baseline full score did not complete with batch size one')
            report.update(status='complete', prompt_count=complete['completed_rows'], finite_all_outputs=True,
                worst_case_probe_passed=True, baseline_complete_status_path=str(complete_path),
                baseline_complete_status_sha256=file_sha256(complete_path),
                baseline_complete_status=complete)
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        if torch is not None and isinstance(exc, torch.OutOfMemoryError):
            report['resource_failure'] = 'CUDA OOM at batch size one within the explicit allocator cap; no batch or precision relaxation'
        if collected and not rows_path.exists():
            report['partial_pilot_rows'] = atomic_rows(rows_path, collected)
    finally:
        if watchdog is not None:
            watchdog.stop()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch is not None:
            gc.collect()
            torch.cuda.empty_cache()
        if before_shards is not None:
            after_shards = committed_shards(root, args.model_id)
            report['new_committed_shards'] = {p: row for p, row in after_shards.items() if p not in before_shards}
            report['existing_committed_shards_unchanged'] = all(after_shards.get(p) == row for p, row in before_shards.items())
            if not report['existing_committed_shards_unchanged']:
                report['status'] = 'failed'
        changed = [path for path, sha in immutable.items() if not Path(path).is_file() or file_sha256(Path(path)) != sha]
        report['immutable_inputs_sources_unchanged'] = not changed
        report['changed_immutable_inputs_sources'] = changed
        if changed:
            report['status'] = 'failed'
        report['completed_at'] = utc_now()
        report = seal_proof(report)
        atomic_json(output, report)
    print(json.dumps({k: report.get(k) for k in ('model_id', 'stage', 'status', 'prompt_count',
        'finite_all_outputs', 'worst_case_probe_passed', 'proof_sha256')}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-repo', 'run-root', 'model-path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--memory-budget-mib', type=int, required=True)
    parser.add_argument('--reserve-mib', type=int, required=True)
    parser.add_argument('--stage', choices=('pilot', 'score'), required=True)
    parser.add_argument('--pilot-proof', type=Path)
    args = parser.parse_args(argv)
    try:
        return int(run(args)['status'] not in ('passed', 'complete'))
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': repr(exc)}), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
