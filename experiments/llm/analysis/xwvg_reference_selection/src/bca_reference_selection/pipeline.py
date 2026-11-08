"""Run the locked cached-data analysis; never launches model inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from .data import load_inputs, aggregate_rotations, reconstruct_from_shards
from .splits import fixed_splits, split_manifest
from .fit import TVFit, fit_tv_simplex, weighted_tv, verify_highs_runtime
from .select import select_policies, check_training_invariants
from .evaluate import evaluate_candidate

COMPONENT = Path(__file__).resolve().parents[2]
V2 = 'PIER_LLM_ECOSYSTEM_GOLDMINE_V2'
V23 = 'PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION'
AVERAGE = Path(V23) / 'outputs/analysis/permutation_averaged_semantic_scores.parquet'
ROTATIONS = Path(V23) / 'outputs/validation/permutation_wise_semantic_scores.parquet'
QUESTIONS = Path(V2) / 'data/mmlu_pro_selected_560.jsonl'
_STATE = None


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def write_json(path, value, immutable=False):
    path = Path(path)
    payload = encoded(value)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Immutable record differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes(payload + b'\n')
    temporary.replace(path)


def reproduction_command(argv=None):
    """Return a pasteable resume command, including the source import path."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--resume' not in arguments:
        arguments.append('--resume')
    return shlex.join(['env', f'PYTHONPATH={COMPONENT / "src"}', sys.executable,
                       '-m', 'bca_reference_selection.pipeline', *arguments])


def validate_probability_provenance(provenance, input_paths):
    """Bind a supplied origin statement to the actual probability bytes read."""
    if not isinstance(provenance, dict):
        raise ValueError('Input provenance must be a JSON object')
    for input_key, record_key in [('cache', 'averaged_cache'), ('rotations', 'rotation_cache')]:
        path = input_paths.get(input_key)
        if path is None:
            continue
        recorded = provenance.get(record_key)
        if not isinstance(recorded, dict) or not isinstance(recorded.get('sha256'), str):
            raise ValueError(f'Input provenance must identify {record_key}.sha256')
        actual_hash = digest(path)
        if recorded['sha256'] != actual_hash:
            raise ValueError(f'Input provenance {record_key} SHA-256 does not match actual input: {path}')
        recorded_size = recorded.get('bytes', recorded.get('size_bytes'))
        actual_size = Path(path).stat().st_size
        if recorded_size is not None and recorded_size != actual_size:
            raise ValueError(f'Input provenance {record_key} size does not match actual input: {path}')
        if provenance.get('measurement_origin') == 'fresh_endpoint_scoring':
            if provenance.get('fresh_measurements') is not True or provenance.get('historical_cache_restoration') is not False:
                raise ValueError('Fresh measurement provenance contains inconsistent origin flags')
        return {'verified': True, 'source_record': record_key, 'path': str(path),
                'sha256': actual_hash, 'size_bytes': actual_size}
    # A missing-input run will stop before fitting; it cannot claim verification.
    return {'verified': False, 'reason': 'No probability input is available to bind to provenance'}


def source_identity(repo):
    files = {str(p.relative_to(COMPONENT)): digest(p)
             for p in sorted((COMPONENT / 'src').rglob('*.py'))}
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    return {'git_revision': revision, 'component_source_hashes': files,
            'component_sha256': hashlib.sha256(encoded(files)).hexdigest()}


def resolve_inputs(args):
    roots = [Path(p).expanduser().resolve() for p in
             [args.raw_root, os.environ.get('BEHAVIORAL_COVERAGE_RAW_DATA'),
              args.repo_root, '/work/Lei', '/work/Users/leiyo'] if p]
    roots = list(dict.fromkeys(roots))
    result = {}
    searched = {}
    for role, relative, override in [('cache', AVERAGE, args.cache),
                                    ('rotations', ROTATIONS, args.rotations_cache),
                                    ('questions', QUESTIONS, args.questions)]:
        paths = [Path(override).expanduser().resolve()] if override else []
        if not override:
            for root in roots:
                paths.append(root / relative)
                if root.name in (V2, V23):
                    paths.extend([root.parent / relative, root / Path(*relative.parts[1:])])
        paths = list(dict.fromkeys(paths))
        searched[role] = [str(p) for p in paths]
        result[role] = next((p for p in paths if p.is_file()), None)
    for name in ['v2_root', 'v23_root']:
        override = getattr(args, name, None)
        run_name = V2 if name == 'v2_root' else V23
        candidates = [Path(override).expanduser().resolve()] if override else [
            root if root.name == run_name else root / run_name for root in roots]
        searched[name] = [str(p) for p in candidates]
        result[name] = next((p for p in candidates if (p/'outputs/raw_scores').is_dir()), None)
    return result, searched


def initialize(inputs, config, output, fingerprint):
    global _STATE
    _STATE = inputs, config, Path(output), fingerprint


def arrays(inputs, family, target_index, question_set):
    index = np.array([i for i, q in enumerate(inputs.question_ids) if q in question_set])
    peers = [i for i in range(len(inputs.model_ids)) if i != target_index]
    values = inputs.probabilities[family]
    y = values[target_index, index].reshape(-1, values.shape[-1])
    x = values[peers][:, index].transpose(1, 2, 3, 0).reshape(-1, values.shape[-1], len(peers))
    mask = inputs.mask[index].reshape(y.shape)
    mass = np.tile(inputs.masses[family], len(index))
    ids = np.repeat(np.asarray(inputs.question_ids)[index], 4)
    return y, x, mask, mass, ids, peers


def setting_path(target_index, family, seed):
    return Path(f'target_{target_index:02d}') / family / str(seed)


def fit_setting(task):
    """Persist fitting-only choices before accessing held-out target arrays."""
    target_index, family, seed, fitting_ids, evaluation_ids, pilot = task
    inputs, config, output, fingerprint = _STATE
    target = inputs.model_ids[target_index]
    relative = setting_path(target_index, family, seed)
    directory = output / 'checkpoints' / fingerprint / relative
    y, x, mask, mass, _, peer_indices = arrays(inputs, family, target_index, fitting_ids)
    peer_ids = [inputs.model_ids[i] for i in peer_indices]
    fits = {}
    fit_rows = []
    for k in config['budgets']:
        candidates = list(itertools.combinations(range(len(peer_ids)), k))
        if pilot and k != 1:
            candidates = [candidates[i] for i in config['compute']['pilot_subset_indices'][str(k)]]
        for subset in candidates:
            path = directory / ('candidate_' + '_'.join(map(str, subset)) + '.json')
            expected = {'input_fingerprint': fingerprint, 'target': target, 'family': family,
                        'split_seed': seed, 'subset': list(subset), 'peers': [peer_ids[i] for i in subset]}
            if path.exists():
                record = json.loads(path.read_text())
                if record['identity'] != expected or record['record_sha256'] != hashlib.sha256(encoded(record['fit'])).hexdigest():
                    raise ValueError(f'Checkpoint identity or checksum mismatch: {path}')
                fit = TVFit.from_record(record['fit'])
                recomputed = weighted_tv(y, np.einsum('nap,p->na', x[:, :, subset], fit.weights), mask, mass)
                if not np.isfinite(recomputed) or abs(recomputed - fit.train_tv) > config['solver']['objective_recompute_tolerance']:
                    raise ValueError(f'Checkpoint training objective mismatch: {path}')
            else:
                fit = fit_tv_simplex(y, x[:, :, subset], mask, mass, solver_config=config['solver'])
                fit_record = fit.to_record()
                write_json(path, {'identity': expected, 'fit': fit_record,
                                  'record_sha256': hashlib.sha256(encoded(fit_record)).hexdigest()}, immutable=True)
            fits[subset] = fit
            fit_rows.append({**expected, 'k': k, **fit.to_record()})
    if pilot:
        selection = None
        invariants = None
    else:
        selection = select_policies(fits, len(peer_ids), config['budgets'], config['selection_tie_tolerance'])
        invariants = check_training_invariants(fits, selection, peer_count=len(peer_ids), tolerance=config['solver']['objective_recompute_tolerance'])
        selections = []
        for k, methods in selection.items():
            for method, subset in methods.items():
                selections.append({'k': k, 'method': method, 'subset': list(subset),
                                   'peers': [peer_ids[i] for i in subset],
                                   'weights': fits[subset].weights.tolist(), 'training_tv': fits[subset].train_tv})
        write_json(directory / 'selection.json', {'input_fingerprint': fingerprint, 'target': target,
                   'family': family, 'split_seed': seed, 'selections': selections}, immutable=True)
    # No evaluation target values have been passed into fitting/selection above.
    evaluation_started = time.perf_counter()
    ey, ex, emask, emass, eids, _ = arrays(inputs, family, target_index, evaluation_ids)
    evaluations = {subset: evaluate_candidate(fit, subset, ey, ex, emask, emass, question_ids=eids)
                   for subset, fit in fits.items()}
    eval_seconds = time.perf_counter() - evaluation_started
    if pilot:
        return {'target': target, 'family': family, 'split_seed': seed, 'fits': fit_rows,
                'evaluation_seconds': eval_seconds,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    rows, losses, candidate_evaluations = [], [], []
    for subset, ev in evaluations.items():
        candidate_evaluations.append({'target': target, 'family': family, 'split_seed': seed,
          'k': len(subset), 'subset': list(subset), 'input_fingerprint': fingerprint,
          'evaluation_tv': ev['evaluation_tv'], 'top_option_agreement': ev['top_option_agreement'],
          'question_ids': list(ev['question_ids']), 'per_question_tv': list(ev['question_tv']),
          'per_question_agreement': list(ev['question_top_option_agreement']),
          'repair_evaluation_prediction_max_abs_bound': ev['repair_evaluation_prediction_max_abs_bound'],
          'repair_evaluation_prediction_max_abs': ev.get('repair_evaluation_prediction_max_abs', 0.0),
          'repair_evaluation_tv_change': ev.get('repair_evaluation_tv_change', 0.0)})
    for k in config['budgets']:
        choices = [s for s in fits if len(s) == k]
        for method in config['methods']:
            subset = None if method == 'uniform_random_expected' else selection[k][method]
            chosen = choices if subset is None else [subset]
            tv = np.mean([evaluations[s]['question_tv'] for s in chosen], axis=0)
            agreement = np.mean([evaluations[s]['question_top_option_agreement'] for s in chosen], axis=0)
            qids = evaluations[chosen[0]]['question_ids']
            row = {'protocol': config['primary_protocol'], 'target': target, 'family': family,
                   'split_seed': seed, 'k': k, 'method': method,
                   'selected_peers': None if subset is None else json.dumps([peer_ids[i] for i in subset]),
                   'selected_weights': None if subset is None else json.dumps(fits[subset].weights.tolist()),
                   'training_tv': float(np.mean([fits[s].train_tv for s in chosen])),
                   'evaluation_tv': float(np.mean(tv)), 'top_option_agreement': float(np.mean(agreement)),
                   'fitting_questions': len(fitting_ids), 'evaluation_questions': len(evaluation_ids),
                   'solver_complete': True, 'input_fingerprint': fingerprint}
            rows.append(row)
            for q, loss, agr in zip(qids, tv, agreement, strict=True):
                losses.append({key: row[key] for key in ('target','family','split_seed','k','method')} |
                              {'base_question_id': q, 'tv': float(loss), 'top_option_agreement': float(agr)})
    # Both controls must match on held-out values as well as training values.
    frame = pd.DataFrame(rows)
    for k, methods in [(1, ['collective','nearest_k']), (7, config['methods'])]:
        control = frame[(frame.k == k) & frame.method.isin(methods)]
        for metric in ['evaluation_tv','top_option_agreement']:
            if np.ptp(control[metric].to_numpy()) > 1e-12:
                raise ValueError(f'Control disagreement: k={k}, {metric}')
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit_rows).to_parquet(directory / 'fits.parquet', index=False)
    pd.DataFrame(candidate_evaluations).to_parquet(directory / 'evaluations.parquet', index=False)
    frame.to_parquet(directory / 'strategies.parquet', index=False)
    pd.DataFrame(losses).to_parquet(directory / 'losses.parquet', index=False)
    write_json(directory / 'complete.json', {'input_fingerprint': fingerprint, 'target': target,
        'family': family, 'split_seed': seed, 'candidate_count': len(fits), 'invariants': invariants,
        'evaluation_seconds': eval_seconds,
        'artifact_sha256': {name:digest(directory/name) for name in
            ['fits.parquet','evaluations.parquet','strategies.parquet','losses.parquet','selection.json']}})
    return {'path': str(directory), 'candidate_count': len(fits)}


def resources(cgroup_root=Path('/sys/fs/cgroup')):
    """Estimate allocatable memory without counting inactive file cache as RSS.

    cgroup memory.current includes charged filesystem cache. Inactive file pages
    are normally reclaimable under allocation pressure; active file cache and
    kernel/slab memory are deliberately not added back. The pilot still limits
    aggregate worker peaks to half this estimate and records the raw counters.
    """
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    memory = psutil.virtual_memory().available
    cgroup_memory = None
    for parent in [Path(cgroup_root)]:
        quota = parent / 'cpu.max'
        if quota.exists():
            q, period = quota.read_text().split()
            if q != 'max':
                cpus = min(cpus, max(1, int(q) // int(period)))
        limit = parent / 'memory.max'
        current = parent / 'memory.current'
        if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
            maximum = int(limit.read_text())
            usage = int(current.read_text())
            counters = {}
            stats = parent / 'memory.stat'
            if stats.exists():
                counters = {key: int(value) for key, value in
                            (line.split() for line in stats.read_text().splitlines())}
            inactive = max(0, min(usage, counters.get('inactive_file', 0)))
            headroom = max(0, min(maximum, maximum - usage + inactive))
            cgroup_memory = {
                'limit_bytes': maximum, 'current_bytes': usage,
                'inactive_file_bytes': counters.get('inactive_file', 0),
                'immediate_headroom_bytes': max(0, maximum - usage),
                'estimated_available_bytes': headroom,
                'estimate_scope': 'limit-current+inactive_file, capped at limit; assumes inactive file reclamation',
            }
            memory = min(memory, headroom)
    return {'available_cpus': cpus, 'available_memory_bytes': memory,
            'cgroup_memory': cgroup_memory,
            'thread_environment': {n: os.environ.get(n) for n in
              ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}


def pilot(inputs, config, tasks, output, fingerprint, requested_workers, load_seconds):
    records = []
    for task in tasks:
        if inputs.model_ids[task[0]] in config['compute']['pilot_targets'] and task[2] == config['compute']['pilot_split_seed']:
            records.append(fit_setting((*task[:-1], True)))
    all_fits = [fit for r in records for fit in r['fits']]
    if sum(r['k'] > 1 for r in all_fits) != config['compute']['pilot_nontrivial_fits']:
        raise ValueError('Incomplete fixed pilot')
    allocation = resources()
    allocation['highs_runtime'] = verify_highs_runtime()
    peak = max(r['peak_rss_bytes'] for r in records)
    safe_by_memory = int(allocation['available_memory_bytes'] * 0.5 // peak)
    if safe_by_memory < 1:
        raise MemoryError('Pilot worker exceeds half available memory; candidates saved for continuation')
    workers = min(requested_workers, config['compute']['max_workers'], allocation['available_cpus'], safe_by_memory)
    timings = {}
    total = 0.0
    for k in [2,3,7]:
        times = [r['solve_seconds'] for r in all_fits if r['k'] == k]
        remaining = len(tasks) * config['candidate_counts'][str(k)] - len(times)
        timings[str(k)] = {'count': len(times), 'mean_seconds': float(np.mean(times)),
                           'min_seconds': min(times), 'max_seconds': max(times), 'remaining_fits': remaining}
        total += float(np.mean(times)) * remaining
    report = {'cases': records, 'load_seconds': load_seconds, 'resources': allocation,
              'pilot_workers': 1, 'chosen_workers': workers, 'peak_rss_bytes': peak,
              'timing_by_budget': timings, 'projected_remaining_seconds': total / workers,
              'projection_scope': 'Timing estimate only; solver variation, I/O and parallel overhead excluded.',
              'input_fingerprint': fingerprint}
    write_json(output / 'validation/timing_pilot.json', report)
    return workers


def run(args):
    repo = args.repo_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    config = json.loads(args.config.read_text())
    locked = json.loads((COMPONENT / 'protocol/config/experiment.json').read_text())
    if config != locked:
        raise ValueError('CLI requires the locked real-data protocol; synthetic fixtures use the Python API only')
    if args.workers < 1 or args.workers > config['compute']['max_workers']:
        raise ValueError('workers must be between 1 and 4')
    # Prevent accidental output into historical inputs or published paths.
    if output == repo or any(output == repo / name or (repo / name) in output.parents
                             for name in ['data', 'paper', 'experiments', 'manifests']):
        raise ValueError('Use a fresh output directory outside source and frozen evidence')
    for parent in [output, *output.parents]:
        if parent.name.startswith('PIER_LLM_ECOSYSTEM_'):
            raise ValueError('Outputs must not be placed inside a historical run')
    output.mkdir(parents=True, exist_ok=True)
    inputs_paths, searched = resolve_inputs(args)
    write_json(output / 'validation/input_inventory.json', {'searched_paths': searched,
        'environment_raw_root': os.environ.get('BEHAVIORAL_COVERAGE_RAW_DATA'),
        'resolved': {k: str(v) if v else None for k,v in inputs_paths.items()}})
    preflight = [sys.executable, str(COMPONENT/'protocol/tools/preflight.py'), '--repo-root', str(repo),
                 '--out', str(output/'validation/preflight.json')]
    for key, flag in [('cache','--balanced-cache'),('rotations','--rotations-cache'),('questions','--questions')]:
        if inputs_paths[key]:
            preflight.extend([flag, str(inputs_paths[key])])
    if args.raw_root:
        preflight.extend(['--raw-root', str(args.raw_root)])
    result = subprocess.run(preflight, check=False)
    if result.returncode == 3:
        raise ValueError('Preflight found changed inspected source or historical question identity')
    identities = {k: {'path': str(v), 'sha256': digest(v), 'size_bytes': v.stat().st_size}
                  for k,v in inputs_paths.items() if v and v.is_file()}
    shard_route = (not inputs_paths['cache'] and not inputs_paths['rotations']
                   and inputs_paths['v2_root'] and inputs_paths['v23_root'])
    if shard_route:
        source_files = []
        for role in ['v2_root','v23_root']:
            for shard in sorted((inputs_paths[role]/'outputs/raw_scores').glob('*/shard_*.parquet')):
                source_files.extend([shard, shard.with_suffix('.meta.json')])
        source_files.append(inputs_paths['v23_root']/'data/permutations/score_semantic_manifest.jsonl')
        absent = [str(p) for p in source_files if not p.is_file()]
        if absent:
            write_json(output/'status/blocked_inputs.json', {'status':'incomplete_shard_inputs', 'missing':absent})
            raise FileNotFoundError(f'Missing shard reconstruction inputs: {absent}')
        identities.update({f'shard_source_{i}': {'path':str(p),'sha256':digest(p),'size_bytes':p.stat().st_size}
                           for i,p in enumerate(source_files)})
    input_provenance = None
    provenance_validation = None
    if args.input_provenance is not None:
        provenance_path = args.input_provenance.expanduser().resolve(strict=True)
        input_provenance = json.loads(provenance_path.read_text())
        provenance_validation = validate_probability_provenance(input_provenance, inputs_paths)
        identities['input_provenance'] = {'path':str(provenance_path), 'sha256':digest(provenance_path), 'size_bytes':provenance_path.stat().st_size}
    versions = {n: version(n) for n in ['numpy','scipy','pandas','pyarrow','matplotlib','psutil']}
    highs_runtime = verify_highs_runtime()
    identity = {'source': source_identity(repo), 'inputs': identities, 'config': config, 'highs_runtime': highs_runtime,
                'versions': versions, 'python': platform.python_version(), 'input_measurement_provenance':input_provenance,
                'input_provenance_validation':provenance_validation}
    fingerprint = hashlib.sha256(encoded(identity)).hexdigest()
    manifest_file = output/'run_manifest.json'
    if manifest_file.exists():
        old = json.loads(manifest_file.read_text())
        if not args.resume:
            raise ValueError('Run exists; use --resume with identical inputs/config/code or a fresh directory')
        if old['input_fingerprint'] != fingerprint:
            raise ValueError('Input/config/code/version changed; use a fresh output directory')
        manifest = old
    else:
        manifest = {**identity, 'input_fingerprint': fingerprint, 'start_utc': now(),
                    'execution_scope': 'CPU only; balanced complete-option endpoints; no model inference',
                    'cache_historical_digest_available': False,
                    'command': reproduction_command()}
        write_json(manifest_file, manifest)
    missing = []
    if not inputs_paths['questions']:
        missing.append({'role': 'selected_questions', 'required_path': str(QUESTIONS),
                        'sha256': '9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6',
                        'size_bytes': 458957})
    if not inputs_paths['cache'] and not inputs_paths['rotations'] and not shard_route:
        missing.append({'role': 'balanced_probabilities', 'required_path': str(AVERAGE),
          'required_column': 'permutation_averaged_semantic_probabilities',
          'cpu_reconstruction_alternative': str(ROTATIONS),
          'alternative_column': 'semantic_remapped_probabilities',
          'shard_alternative': [str(Path(V2)/'outputs/raw_scores/<model_slug>/shard_*.parquet'),
             str(Path(V23)/'outputs/raw_scores/<model_slug>/shard_*.parquet'),
             str(Path(V23)/'data/permutations/score_semantic_manifest.jsonl')],
          'note': 'All shard sidecars and complete original rotations required; fitted summary tables are insufficient.'})
    if missing:
        write_json(output/'status/blocked_inputs.json', {'status': 'blocked_missing_primary_inputs',
          'empirical_experiment_run': False, 'candidate_fits_completed': 0,
          'missing': missing, 'searched_paths': searched, 'input_fingerprint': fingerprint})
        (output/'REPORT.md').write_text('# XWVG experiment: blocked inputs\n\n'
          'The locked real-data analysis could not start. No empirical results, main figure, '
          'or completed-results reviewer response have been produced.\n\n'
          + '\n'.join('- Required: `' + r['required_path'] + '`' for r in missing)
          + '\n\nSee `status/blocked_inputs.json` for exact fields, alternative inputs and question checksum. '
          'See `validation/preflight.json` for source identity checks. Implementation tests are recorded separately.\n')
        manifest.update(status='blocked_missing_primary_inputs', end_utc=now())
        write_json(manifest_file, manifest)
        print(json.dumps({'status': manifest['status'], 'report': str(output/'REPORT.md')}), flush=True)
        return 2
    start = time.perf_counter()
    cache = inputs_paths['cache']
    if cache is None:
        cache = output/'derived/permutation_averaged_semantic_scores.parquet'
        aggregation_path = output/'validation/aggregation.json'
        if cache.exists() and aggregation_path.exists():
            aggregation = json.loads(aggregation_path.read_text())
            if aggregation.get('input_fingerprint') != fingerprint or aggregation['output']['sha256'] != digest(cache):
                raise ValueError('CPU-derived cache provenance mismatch')
        else:
            if shard_route:
                aggregation = reconstruct_from_shards(inputs_paths['v2_root'], inputs_paths['v23_root'],
                                                       cache, inputs_paths['questions'], config)
            else:
                aggregation = aggregate_rotations(inputs_paths['rotations'], cache, inputs_paths['questions'], config)
            aggregation['input_fingerprint'] = fingerprint
            write_json(aggregation_path, aggregation)
    inputs = load_inputs(cache, inputs_paths['questions'], config)
    write_json(output/'validation/input_checks.json', inputs.report)
    split_map = fixed_splits(inputs.questions, config['split_seeds'])
    split_records = split_manifest(inputs.questions, config['split_seeds'])
    write_json(output/'validation/splits.json', {'splits': split_records,
       'questions': dict(zip(inputs.question_ids, inputs.categories, strict=True))})
    tasks = [(t,f,seed,s.fitting_ids,s.evaluation_ids,False)
             for t in range(len(inputs.model_ids)) for f in config['families'] for seed,s in split_map.items()]
    initialize(inputs, config, output, fingerprint)
    if args.stage == 'check':
        print(json.dumps({'status': 'inputs_validated', 'settings': len(tasks)}), flush=True)
        return 0
    if args.stage == 'report':
        timing = json.loads((output/'validation/timing_pilot.json').read_text())
        if timing['input_fingerprint'] != fingerprint:
            raise ValueError('Pilot fingerprint mismatch')
        workers = timing['chosen_workers']
    else:
        workers = pilot(inputs, config, tasks, output, fingerprint, args.workers, time.perf_counter()-start)
    if args.stage == 'pilot':
        print(json.dumps({'status': 'pilot_complete', 'workers': workers}), flush=True)
        return 0
    directories = []
    failures = []
    if args.stage == 'report':
        directories = [output/'checkpoints'/fingerprint/setting_path(t,f,seed) for t,f,seed,*_ in tasks]
    elif workers == 1:
        for task in tasks:
            try:
                directories.append(Path(fit_setting(task)['path']))
                print(json.dumps({'completed_settings':len(directories),'total_settings':len(tasks)}),flush=True)
            except Exception as exc:
                failures.append({'target':inputs.model_ids[task[0]], 'family':task[1], 'split_seed':task[2], 'error':repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
             initializer=initialize, initargs=(inputs,config,str(output),fingerprint)) as pool:
            futures = {pool.submit(fit_setting, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    directories.append(Path(future.result()['path']))
                    print(json.dumps({'completed_settings':len(directories),'total_settings':len(tasks)}),flush=True)
                except Exception as exc:
                    failures.append({'target':inputs.model_ids[task[0]],'family':task[1],'split_seed':task[2],'error':repr(exc)})
    if failures:
        write_json(output/'status/incomplete.json', {'failed_settings':failures, 'completed_settings':len(directories)})
        raise RuntimeError(f'{len(failures)} failed settings; see status/incomplete.json; resume preserves valid fits')
    for directory in directories:
        record=json.loads((directory/'complete.json').read_text())
        if record['input_fingerprint'] != fingerprint or record['candidate_count'] != sum(config['candidate_counts'].values()):
            raise ValueError(f'Incomplete or mismatched setting: {directory}')
        for name, expected_hash in record['artifact_sha256'].items():
            if digest(directory/name) != expected_hash:
                raise ValueError(f'Setting artifact checksum mismatch: {directory/name}')
    frames = {}
    for kind,destination in [('fits','candidate_fits.parquet'),('evaluations','candidate_evaluations.parquet'),
                             ('strategies','strategy_split_results.csv'),('losses','heldout_question_losses.parquet')]:
        frame = pd.concat([pd.read_parquet(d/f'{kind}.parquet') for d in sorted(directories)], ignore_index=True)
        if kind == 'strategies': frame.to_csv(output/destination,index=False)
        else: frame.to_parquet(output/destination,index=False)
        frames[kind]=frame
    counts = {'candidate_fits':len(frames['fits']), 'strategy_rows':len(frames['strategies']), 'heldout_rows':len(frames['losses'])}
    if counts != {'candidate_fits':config['expected_candidate_fits'], 'strategy_rows':config['expected_strategy_split_rows'],
                  'heldout_rows':config['expected_strategy_split_rows']*config['question_count']//2}:
        raise ValueError(f'Incomplete output counts: {counts}')
    write_json(output/'validation/numerical_checks.json', {'passed':True, 'counts':counts,
       'settings':[json.loads((d/'complete.json').read_text()) for d in sorted(directories)]})
    from .validation import validate_result_tables
    table_checks = validate_result_tables(frames['fits'], frames['evaluations'], frames['strategies'],
                                          frames['losses'], inputs, config, split_map)
    write_json(output/'validation/output_checks.json', table_checks)
    # Integrity is checked before any completed-results prose is generated.
    for info in identities.values():
        if digest(info['path']) != info['sha256']:
            raise ValueError(f'Original input changed: {info["path"]}')
    if source_identity(repo) != identity['source']:
        raise ValueError('Analysis source changed during run')
    if args.stage == 'point':
        print(json.dumps({'status':'point_complete',**counts}),flush=True)
        return 0
    from .reporting import write_results
    timing = json.loads((output/'validation/timing_pilot.json').read_text())
    manifest.update(status='complete', end_utc=now(), workers=workers,
                    compute={'pilot': {k:v for k,v in timing.items() if k != 'cases'},
                             'total_candidate_solve_seconds':float(frames['fits'].solve_seconds.sum()),
                             'nontrivial_fits':int((frames['fits'].k > 1).sum())})
    write_results(frames['strategies'], frames['losses'], inputs.questions, config, output,
                  run_metadata=manifest, command=manifest['command'], complete=True)
    # Confirm original input bytes were not altered during computation.
    for info in identities.values():
        if digest(info['path']) != info['sha256']:
            raise ValueError(f'Original input changed: {info["path"]}')
    if source_identity(repo) != identity['source']:
        raise ValueError('Analysis source changed during run')
    write_json(manifest_file,manifest)
    write_json(output/'status/complete.json', {'status':'complete', 'input_fingerprint':fingerprint, **counts,
      'artifacts':{str(p.relative_to(output)):digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status':'complete','output_root':str(output)}),flush=True)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--questions',type=Path)
    parser.add_argument('--input-provenance',type=Path,help='JSON provenance for recovered or freshly scored input probabilities')
    parser.add_argument('--rotations-cache',type=Path)
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--v2-root',type=Path,help='Original V2 root for CPU reconstruction from shards')
    parser.add_argument('--v23-root',type=Path,help='Original V2.3 root for CPU reconstruction from shards')
    parser.add_argument('--config',type=Path,default=COMPONENT/'protocol/config/experiment.json')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','check','pilot','point','report'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({'status':'failed','error':repr(exc)}),file=sys.stderr,flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
