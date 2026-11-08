#!/usr/bin/env python3
"""Rebuild original V23 semantic manifests and a separate all-rotation score plan.

Only deterministic CPU prompt metadata preparation runs. Tokenizer rendering,
model downloads, candidate scoring and generation are deliberately absent.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import time

from prepare_inputs import digest, verify_design

SOURCE_HASHES = {
    'experiments/llm/analysis/v23/src/pier_llm_v23/prepare.py':
        '0f5b8eabcdf9e19f86beeae22c1cdb04f0e35f47ae24e9fb283796688f265803',
    'experiments/llm/analysis/v23/src/pier_llm_v23/core.py':
        '9d71da8efd9ea88bb0d9da3b7ccfacf8fdf148750dd24a32623004e3f8719f8a',
    'experiments/llm/analysis/v23/src/pier_llm_v23/utils.py':
        '36ebf9390c1340148ccb41533b337cd4c19971d5911af6b828021db5347ab95c',
    'experiments/llm/analysis/v23/configs/experiment_v23.json':
        '4c1b68f3ce5cb8ecc7ab17973cc3722d85a9b46b87b0e59d9d8e8f51bdb6197e',
    'experiments/llm/manifests/v23/status/prompt_manifest_status.json':
        'ef149c6982970448f4393563723d0bfa4cdda9beec8ff12510a4fe0ef13478e6',
}
PATHS = {
    'score_manifest_sha256': 'data/permutations/score_semantic_manifest.jsonl',
    'score_inference_manifest_sha256': 'data/permutations/score_inference_manifest.jsonl',
    'generation_manifest_sha256': 'data/generation/long_generation_manifest.jsonl',
}
EXECUTION = 'data/permutations/score_execution_manifest.jsonl'


def verify_v23(output, historical):
    files = []
    for key, relative in PATHS.items():
        path = output / relative
        actual = digest(path)
        if actual != historical[key]:
            raise RuntimeError(f'Historical V23 manifest mismatch: {relative}; {actual}')
        files.append({'relative_path': relative, 'size_bytes': path.stat().st_size,
                      'sha256': actual, 'historical_sha256': historical[key], 'match': True})
    with (output / PATHS['score_manifest_sha256']).open() as source, (output / EXECUTION).open() as executed:
        rows = rotations0 = candidates = 0
        conditions = Counter()
        for old_line, new_line in zip(source, executed, strict=True):
            old, new = json.loads(old_line), json.loads(new_line)
            expected = {**old, 'inference_required': True, 'rotation0_reuse': False}
            if new != expected:
                raise RuntimeError('Fresh execution manifest changed prompt semantics beyond required execution flags')
            rows += 1
            candidates += new['option_count']
            rotations0 += new['permutation_id'] == 0
            conditions[new['semantic_condition']] += 1
    if (rows, rotations0, candidates) != (36946, 3920, 357196) or set(conditions) != {'C0','I0','I1','I2','D0','D1','D2'}:
        raise RuntimeError(f'Incomplete fresh scoring design: {rows}, {rotations0}, {candidates}, {conditions}')
    return files, {'rows_per_model': rows, 'rotation0_scored_per_model': rotations0,
                   'candidate_sequences_per_model': candidates, 'rows_all_models': rows * 8,
                   'candidate_sequences_all_models': candidates * 8,
                   'condition_counts_per_model': dict(conditions), 'path': str(output / EXECUTION),
                   'sha256': digest(output / EXECUTION), 'size_bytes': (output / EXECUTION).stat().st_size}


def prepare_v23(repo, v2, output, resume=False):
    started = time.monotonic()
    repo, v2, output = Path(repo).resolve(), Path(v2).resolve(), Path(output).resolve()
    source_checks = []
    for relative, expected in SOURCE_HASHES.items():
        actual = digest(repo / relative)
        if actual != expected:
            raise RuntimeError(f'Audited V23 source changed: {relative}')
        source_checks.append({'path': relative, 'sha256': actual})
    historical_v2 = json.loads((repo/'experiments/llm/manifests/v2/data/manifests/dataset_source.json').read_text())
    verify_design(v2, historical_v2)
    historical = json.loads((repo/'experiments/llm/manifests/v23/status/prompt_manifest_status.json').read_text())
    owned = [output/relative for relative in [*PATHS.values(), EXECUTION,
        'configs/experiment_v23.json', 'status/prompt_manifest_status.json', 'status/input_preparation.json']]
    if resume and all(p.is_file() for p in owned):
        files, execution = verify_v23(output, historical)
        return {'status': 'reused_verified_v23_design', 'historical_manifests': files, 'execution': execution}
    if any(p.exists() for p in owned):
        raise FileExistsError('Refusing to overwrite V23 preparation files; use --resume for a complete verified design')
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    sys.path.insert(0, str(repo/'experiments/llm/analysis/v23/src'))
    from pier_llm_v23.prepare import build_manifests
    from pier_llm_v23.utils import atomic_write_jsonl, read_jsonl
    original_config = repo/'experiments/llm/analysis/v23/configs/experiment_v23.json'
    config = json.loads(original_config.read_text())
    config['source_roots']['v2'] = str(v2)
    (output/'configs').mkdir(parents=True,exist_ok=True)
    (output/'configs/experiment_v23.json').write_text(json.dumps(config,indent=2)+'\n')
    result = build_manifests(output)
    # Preserve unchanged historical semantic and inference manifests as provenance.
    # Only a distinct execution file disables reuse and schedules every rotation.
    atomic_write_jsonl(output/EXECUTION,
        ({**row, 'inference_required': True, 'rotation0_reuse': False}
         for row in read_jsonl(output/PATHS['score_manifest_sha256'])))
    files, execution = verify_v23(output, historical)
    report = {'status': 'semantic_design_historical_identity_verified_fresh_scoring_plan_ready',
              'source_checks': source_checks, 'historical_manifests': files, 'execution': execution,
              'config_changes': {'source_roots.v2': {'from':json.loads(original_config.read_text())['source_roots']['v2'], 'to':str(v2)}},
              'config_sha256':digest(output/'configs/experiment_v23.json'),
              'model_inference_run': False, 'model_download_run': False, 'tokenizer_rendering_run': False,
              'generation_run': False, 'planned_rotation0_reuse': False,
              'historical_reuse_metadata_preserved_in_semantic_manifest': True,
              'elapsed_seconds': time.monotonic()-started,
              'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'completed_utc':datetime.now(timezone.utc).isoformat(), 'adapter_sha256':digest(__file__)}
    (output/'status/input_preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--v2-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(prepare_v23(args.repo_root,args.v2_root,args.output_root,args.resume),indent=2))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
