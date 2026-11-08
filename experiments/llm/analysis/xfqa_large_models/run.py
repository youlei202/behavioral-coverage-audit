#!/usr/bin/env python3
"""Execute the isolated XfqA larger-model campaign with recorded commands."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

from preflight import run_preflight, sha, write_json

PHASES = ['preflight', 'prepare', 'resolve', 'pilot', 'score', 'assemble',
          'analyze', 'bootstrap', 'report', 'verify', 'package']
COMMAND_LOCK = threading.Lock()


def execute(args, command, name):
    started = datetime.now(timezone.utc).isoformat()
    log = args.output_root / 'logs' / f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with COMMAND_LOCK:
        commands = args.output_root / 'RUN_COMMANDS.sh'
        if not commands.exists():
            commands.write_text('#!/usr/bin/env bash\nset -euo pipefail\n# Actual commands; use the portable bundle replay for a relocated output.\n')
        with commands.open('a') as handle:
            handle.write(f'\n# {started}\n{shlex.join(command)}\n')
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    begin = time.monotonic()
    with log.open('a') as handle:
        result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record = {'command': command, 'started_utc': started, 'wall_seconds': time.monotonic() - begin,
              'returncode': result.returncode, 'log': str(log)}
    with COMMAND_LOCK:
        with (args.output_root / 'status/command_history.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
    if result.returncode:
        raise RuntimeError(f'{name} exited {result.returncode}; inspect {log}')
    return record


def slots(args, models):
    if args.placement:
        records = json.loads(args.placement.read_text())['slots']
        requested = {row['id'] for row in models}
        selected = [row for row in records if row['model_id'] in requested]
        if {r['model_id'] for r in selected} != requested or len(selected) != len(requested):
            raise ValueError('Placement must assign each requested model exactly once')
        identities = [(r.get('host', 'local'), r.get('port'), r['device']) for r in selected]
        uuids = [r.get('expected_gpu_uuid') for r in selected]
        if not all(isinstance(value, str) and value.strip() for value in uuids):
            raise ValueError('Explicit placement requires an expected physical GPU UUID for every slot')
        normalized = [value.lower().removeprefix('gpu-') for value in uuids]
        if len(set(identities)) != len(identities) or len(set(normalized)) != len(normalized):
            raise ValueError('Duplicate GPU assignment')
        return selected
    return [{'model_id': r['id'], 'device': args.device, 'scoring_python': args.scoring_python}
            for r in models]


def scoring_command(args, phase, model=None, slot=None):
    slot = slot or {}
    python = str(slot.get('scoring_python', args.scoring_python))
    command = [python, str(Path(__file__).with_name('scoring.py')),
               '--repo-root', str(args.repo_root), '--plan-root', str(args.plan_root),
               '--output-root', str(args.output_root), '--phase', phase,
               '--model-cache', str(args.model_cache)]
    if args.resume:
        command.append('--resume')
    if args.execution_manifest:
        command += ['--execution-manifest', str(args.execution_manifest)]
    if model:
        command += ['--model-id', model]
    if phase in ('pilot', 'score'):
        command += ['--device', slot.get('device', args.device), '--batch-size', str(args.batch_size),
                    '--gpu-reserve-gib', str(args.gpu_reserve_gib), '--cpu-reserve-gib', str(args.cpu_reserve_gib)]
        if slot.get('expected_gpu_uuid'):
            command += ['--expected-gpu-uuid', slot['expected_gpu_uuid']]
    if slot.get('host'):
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
               '-o', 'StrictHostKeyChecking=yes']
        if slot.get('identity_file'):
            ssh += ['-i', slot['identity_file'], '-o', 'IdentitiesOnly=yes']
        ssh += ['-p', str(slot.get('port', 22)), f"{slot['user']}@{slot['host']}"]
        remote = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1',
                  'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', *command]
        command = ssh + [shlex.join(remote)]
    return command


def run_scoring(args, phase, config):
    models = config['new_models']
    if args.model_id:
        models = [r for r in models if r['id'] == args.model_id]
        if not models:
            raise ValueError('Only the three declared new checkpoints may be scored')
    if phase in ('prepare', 'resolve', 'assemble'):
        return execute(args, scoring_command(args, phase), phase)
    if phase == 'pilot':
        # Input preparation is CPU-only. Completed immutable downloads/renders
        # are verified and reused, not regenerated under a different identity.
        for record in models:
            model = record['id']
            for step in ('download', 'render'):
                execute(args, scoring_command(args, step, model), f'{step}-{model.replace("/", "--")}')
    assignments = slots(args, models)
    def work(slot):
        model = slot['model_id']
        return execute(args, scoring_command(args, phase, model, slot),
                       f'{phase}-{model.replace("/", "--")}')
    # Without explicit placement, all models share the requested device and run
    # serially. Parallelism requires distinct explicitly supplied device slots.
    if not args.placement:
        return [work(slot) for slot in assignments]
    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        futures = [pool.submit(work, slot) for slot in assignments]
        results, errors = [], []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('; '.join(errors))
        return results


def run_phase(args, phase):
    config = json.loads((args.plan_root / 'config/experiment.json').read_text())
    if phase == 'preflight':
        return run_preflight(args.repo_root, args.plan_root, args.output_root, args.resume)
    if phase in ('prepare', 'resolve', 'pilot', 'score', 'assemble'):
        return run_scoring(args, phase, config)
    if phase in ('analyze', 'bootstrap'):
        import analysis
        function = getattr(analysis, phase)
        return function(args.repo_root, args.plan_root, args.output_root,
                        cache_path=args.cache or args.output_root / 'data/permutation_averaged_semantic_scores.parquet',
                        workers=args.workers, resume=args.resume)
    if phase == 'report':
        from resources import finalize_resources
        from reporting import run_report
        finalize_resources(args.output_root)
        return run_report(args.repo_root, args.plan_root, args.output_root, resume=args.resume)
    if phase == 'verify':
        from verify import run_verify
        return run_verify(args.repo_root, args.plan_root, args.output_root, resume=args.resume)
    if phase == 'package':
        from delivery import run_package
        return run_package(args.repo_root, args.plan_root, args.output_root, resume=args.resume)
    raise ValueError(phase)


def finalize(args):
    """Publish completion only after real inference, uncertainty and delivery."""
    root = args.output_root
    def checked(relative, expected_status='complete'):
        value = json.loads((root / relative).read_text())
        if value.get('status') != expected_status:
            raise ValueError(f'Completion prerequisite is not {expected_status}: {relative}')
        return value
    config = json.loads((args.plan_root / 'config/experiment.json').read_text())
    lock = checked('model_lock.json')
    if [r['id'] for r in lock['models']] != [r['id'] for r in config['new_models']]:
        raise ValueError('Complete run requires exactly the three declared new models')
    for record in lock['models']:
        name = record['id'].replace('/', '--')
        proof = checked(f'status/model_scoring/{name}_complete.json')
        if (proof['model_revision'] != record['revision'] or proof['completed_rows'] != 36946
                or proof['candidate_continuations'] != 357196):
            raise ValueError('Incomplete pinned-model scoring inventory')
    assembly = checked('status/assembly_complete.json')
    if (assembly.get('real_three_model_scoring_complete') is not True
            or assembly['new_rotation_rows'] != 110838
            or assembly['new_averaged_vectors'] != 11760
            or assembly['merged_averaged_vectors'] != 43120):
        raise ValueError('Complete eleven-model input is required')
    point = checked('analysis/point_complete.json')
    manifest = json.loads((root / 'analysis/analysis_manifest.json').read_text())
    if (manifest.get('identity', {}).get('synthetic_fixture') is not False
            or manifest.get('identity', {}).get('scope') != 'complete_expanded_experiment'):
        raise ValueError('Synthetic or partial-scope analysis cannot be marked empirically complete')
    primary = checked('analysis/primary_refit_complete.json')
    secondary = checked('analysis/secondary_complete.json')
    bootstrap = checked('analysis/bootstrap_complete.json')
    fingerprint = point['input_fingerprint']
    if manifest.get('input_fingerprint') != fingerprint:
        raise ValueError('Completed point results differ from the original analysis identity')
    if any(p['input_fingerprint'] != fingerprint for p in (primary, secondary, bootstrap)):
        raise ValueError('Analysis and uncertainty identities differ')
    for stage in (point, primary, secondary):
        if not stage.get('artifact_hashes'):
            raise ValueError('Completed analysis lacks an artifact inventory')
        for name, expected_hash in stage['artifact_hashes'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Analysis artifact path is not portable')
            if sha(root / 'analysis' / relative) != expected_hash:
                raise ValueError(f'Completed analysis artifact changed: {name}')
    if (point['point_fit_count'] != 5840 or primary['replicates'] != 1000
            or primary['fits'] != 100000 or secondary['replicates'] != 2000):
        raise ValueError('Full prescribed point/refit/descriptive work is incomplete')
    replay = checked('validation/numerical_replay.json', 'passed')
    if replay['input_fingerprint'] != fingerprint:
        raise ValueError('Numerical replay belongs to different inputs')
    expected = {'new_rotation_rows': 110838, 'merged_vectors': 43120,
                'point_fits': 5840, 'per_question_rows': 6540800,
                'primary_refit_replicates': 1000, 'primary_refit_fits': 100000,
                'secondary_replicates': 2000}
    if any(replay['counts'].get(k) != v for k, v in expected.items()):
        raise ValueError('Numerical replay inventory is incomplete')
    report = checked('status/report_complete.json')
    if report['input_fingerprint'] != fingerprint or not report.get('artifacts'):
        raise ValueError('Completed report has a different identity or no artifact inventory')
    for artifact in report['artifacts']:
        relative = Path(artifact['path'])
        if relative.is_absolute() or '..' in relative.parts or sha(root / relative) != artifact['sha256']:
            raise ValueError(f'Previously validated report artifact changed: {relative}')
    delivery = checked('status/package_complete.json', 'verified')
    if (delivery.get('input_fingerprint') != fingerprint
            or delivery.get('numerical_replay_sha256') != sha(root / 'validation/numerical_replay.json')
            or delivery.get('report_complete_sha256') != sha(root / 'status/report_complete.json')):
        raise ValueError('Delivery proof is not bound to the validated analysis/report snapshot')
    if not all(delivery.get(k) is True for k in ('zip_crc_verified', 'bundle_hashes_verified', 'portable_replay_verified')):
        raise ValueError('Portable result archive has not passed delivery verification')
    archive = Path(delivery['archive_path'])
    if not archive.is_file() or sha(archive) != delivery['archive_sha256']:
        raise ValueError('Verified delivery archive is missing or changed')
    required = ['REPORT.md', 'XfqA_RESPONSE_DRAFT.md', 'METHODS_INSERT.md',
                'RESULTS_INSERT.md', 'LIMITATIONS_AND_DEVIATIONS.md', 'headline_evidence.json']
    required += [f'figures/{stem}.{extension}' for stem in (
        'large_target_coverage', 'large_pair_directionality', 'roster_expansion')
                 for extension in ('pdf', 'png')]
    artifacts = {}
    for relative in required:
        path = root / 'analysis' / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Required empirical deliverable missing: {path}')
        artifacts[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    result = {'schema': 'xfqa_empirical_completion_v1', 'status': 'complete',
              'input_fingerprint': fingerprint, 'counts': expected,
              'model_lock_sha256': sha(root / 'model_lock.json'),
              'numerical_replay_sha256': sha(root / 'validation/numerical_replay.json'),
              'archive': str(archive), 'archive_sha256': delivery['archive_sha256'],
              'artifacts': artifacts}
    path = root / 'status/complete.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Existing empirical completion record differs')
    else:
        write_json(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('repo-root', 'plan-root', 'output-root'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=[*PHASES, 'all'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--scoring-python', default=sys.executable)
    parser.add_argument('--analysis-python', default=sys.executable)
    parser.add_argument('--model-cache', type=Path)
    parser.add_argument('--execution-manifest', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--placement', type=Path)
    parser.add_argument('--model-id')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu-reserve-gib', type=float, default=16)
    parser.add_argument('--cpu-reserve-gib', type=float, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    for name in ('repo_root', 'plan_root', 'output_root'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.output_root.is_relative_to(args.repo_root / 'outputs/xfqa_large_models') or args.output_root == args.repo_root / 'outputs/xfqa_large_models':
        parser.error('Use outputs/xfqa_large_models/<run_id>')
    if args.phase == 'all' and args.model_id:
        parser.error('all requires the complete three-model experiment')
    args.model_cache = args.model_cache or args.output_root / 'cache/models'
    (args.output_root / 'status').mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / 'status/driver.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phases = PHASES if args.phase == 'all' else [args.phase]
        for phase in phases:
            print(json.dumps({'phase': phase, 'status': 'starting'}), flush=True)
            started = time.monotonic()
            try:
                if phase in ('analyze', 'bootstrap', 'report', 'verify', 'package') and Path(args.analysis_python).absolute() != Path(sys.executable).absolute():
                    original = list(sys.argv[1:] if argv is None else argv)
                    for index, value in enumerate(original):
                        if value == '--phase':
                            original[index + 1] = phase
                        elif value.startswith('--phase='):
                            original[index] = '--phase=' + phase
                    # The child owns the phase lock; release our lock while waiting.
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    try:
                        result = execute(args, [args.analysis_python, str(Path(__file__).resolve()), *original], f'cpu-{phase}')
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    result = run_phase(args, phase)
                if phase == 'package':
                    finalize(args)
                write_json(args.output_root / f'status/driver_{phase}.json',
                           {'phase': phase, 'status': 'complete', 'wall_seconds': time.monotonic() - started})
            except Exception as error:
                write_json(args.output_root / f'status/driver_{phase}.json',
                           {'phase': phase, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
                raise
            print(json.dumps({'phase': phase, 'status': 'complete'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
