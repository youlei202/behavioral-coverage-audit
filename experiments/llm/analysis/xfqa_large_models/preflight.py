"""Read-only baseline checks and portable input/job inventory for XfqA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preserve(repo, output):
    """Record old tracked source/data and completed scientific outputs once."""
    dest = output / 'validation/preservation_baseline.json'
    if dest.exists():
        result = json.loads(dest.read_text())
        for row in result['files']:
            path = repo / row['path']
            if not path.is_file() or sha(path) != row['sha256']:
                raise ValueError(f'Preserved original changed: {path}')
        return result
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0')
    paths = {repo / name for name in names if name and not name.startswith('experiments/llm/analysis/xfqa_large_models/')}
    old = repo / 'outputs/xwvg_reference_selection'
    roots = [old / '20261107_fresh_analysis',
             old / '20261107_recovery/fresh_endpoints/outputs',
             old / '20261107_recovery/fresh_endpoints/status',
             old / '20261107_recovery/fresh_endpoints/data/permutations',
             old / '20261107_recovery/prepared_v2/data/mmlu_pro_selected_560.jsonl',
             old / 'XWVG_Utility_Results_20261107.zip',
             old / 'XWVG_Utility_Results_20261107.zip.sha256']
    for root in roots:
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(p for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        else:
            raise FileNotFoundError(f'Preservation input missing: {root}')
    records = [{'path': str(p.relative_to(repo)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(paths) if p.is_file()]
    result = {'schema': 'xfqa_preservation_v1', 'status': 'recorded',
              'git_revision': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'All pre-existing tracked files; completed XWVG analysis, score outputs, status, semantic/rendered manifests, selected questions and final ZIP. Model weights, environments, transient logs and download caches excluded.',
              'files': records, 'file_count': len(records), 'total_bytes': sum(r['bytes'] for r in records)}
    write_json(dest, result)
    return result


def run_preflight(repo_root, plan_root, output_root, resume=False):
    repo, plan, output = [Path(p).expanduser().resolve() for p in (repo_root, plan_root, output_root)]
    allowed = repo / 'outputs/xfqa_large_models'
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('Use an isolated outputs/xfqa_large_models/<run_id> directory')
    for line in (plan / 'MANIFEST.sha256').read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        relative = Path(name.lstrip('*'))
        if relative.is_absolute() or '..' in relative.parts or sha(plan / relative) != expected:
            raise ValueError(f'Plan checksum/path mismatch: {relative}')
    config = json.loads((plan / 'config/experiment.json').read_text())
    # The supplied verifier imports its archived XWVG copy. Isolate that import
    # so all-in-one and separately resumed CPU phases use the same live module
    # paths in their source identity, rather than inheriting sys.modules here.
    seed_path = output / 'validation/seed_check.json'
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(plan / 'tools/verify_seed.py'),
                    '--package-root', str(plan), '--deep', '--report', str(seed_path)],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seed = json.loads(seed_path.read_text())
    write_json(output / 'inputs/splits.json', seed['splits'])
    source_manifest = json.loads((plan / 'evidence/inspected_sources.json').read_text())
    comparisons = []
    for row in source_manifest['sources']:
        actual = sha(repo / row['repository_path'])
        comparisons.append({'path': row['repository_path'], 'expected_sha256': row['sha256'],
                            'actual_sha256': actual, 'matches': actual == row['sha256']})
    for name, expected in {
        'experiments/llm/analysis/v23/src/pier_llm_v23/inference.py': '0bcd17ba4b3939f257c16e35b78d7ce03239b10442ca761fa3bc225fa2271a09',
        'experiments/llm/analysis/v23/src/pier_llm_v23/prepare.py': '0f5b8eabcdf9e19f86beeae22c1cdb04f0e35f47ae24e9fb283796688f265803',
        'experiments/llm/inference/v2/src/pier_llm/data_prep.py': '1f8f86b978a1009f350251901005b47b4ff92b741b629932f6197adae28d7bac',
    }.items():
        actual = sha(repo / name)
        comparisons.append({'path': name, 'expected_sha256': expected,
                            'actual_sha256': actual, 'matches': actual == expected})
    write_json(output / 'validation/source_comparison.json', {'files': comparisons})
    if not all(r['matches'] for r in comparisons):
        raise ValueError('Historical source mismatch: inspect validation/source_comparison.json before proceeding')
    jobs_module = import_file('_xfqa_plan_jobs', plan / 'tools/build_analysis_jobs.py')
    jobs = jobs_module.build_jobs(config)
    payload = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in jobs)
    job_path = output / 'inputs/analysis_jobs.jsonl'
    if job_path.exists() and job_path.read_text() != payload:
        raise ValueError('Saved job inventory changed')
    job_path.parent.mkdir(parents=True, exist_ok=True)
    if not job_path.exists():
        job_path.write_text(payload)
    write_json(output / 'validation/job_counts.json', jobs_module.summary(jobs))
    baseline = preserve(repo, output)
    result = {'status': 'complete', 'protocol_sha256': sha(plan / 'config/experiment.json'),
              'plan_manifest_sha256': sha(plan / 'MANIFEST.sha256'),
              'seed_validation': seed['status'], 'splits': len(seed['splits']),
              'point_jobs': len(jobs), 'source_matches': len(comparisons),
              'preserved_files': baseline['file_count'],
              'preservation_baseline_sha256': sha(output / 'validation/preservation_baseline.json')}
    write_json(output / 'status/preflight.json', result)
    return result
