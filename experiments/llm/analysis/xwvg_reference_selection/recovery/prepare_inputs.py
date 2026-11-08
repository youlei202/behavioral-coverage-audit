#!/usr/bin/env python3
"""Rebuild the archived V2 prompt design from the pinned dataset, CPU only.

This adapter calls the checksum-verified historical build_from_rows with an
explicit fresh output root. It does not render tokenizers, download checkpoints,
score models, or call historical orchestration. Every design output must equal
its archived digest before the operation is marked verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

DATASET = 'TIGER-Lab/MMLU-Pro'
REVISION = 'b189ec765aa7ed75c8acfea42df31fdae71f97be'
SOURCE_HASHES = {
    'experiments/llm/inference/v2/src/pier_llm/data_prep.py':
        '1f8f86b978a1009f350251901005b47b4ff92b741b629932f6197adae28d7bac',
    'experiments/llm/inference/v2/src/pier_llm/utils.py':
        '5e033ca7715044879b4e3d1c36b9ae14a47520bf884f12930d9a6ea6462035fc',
    'experiments/llm/inference/v2/configs/experiment.json':
        'a0777e2f08b6e75f083531acf7bf397f6328ccc8e5e8d5f378e2e4ea04272eac',
    'experiments/llm/manifests/v2/data/manifests/dataset_source.json':
        'ea1023b4d9f56ac39d1c6c38643ba6b8080659e2bdf2847013ddd0fe75bbc6c3',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_design(root, historical):
    records = []
    for name, expected in historical['hashes'].items():
        path = root / name
        actual = digest(path)
        with path.open('rb') as handle:
            rows = sum(1 for line in handle if line.strip())
        records.append({'path': str(path), 'relative_path': name,
                        'size_bytes': path.stat().st_size, 'rows': rows,
                        'expected_rows': historical['counts'][name],
                        'sha256': actual, 'historical_sha256': expected,
                        'match': actual == expected and rows == historical['counts'][name]})
    if not all(r['match'] for r in records):
        raise RuntimeError('Historical design mismatch: ' + json.dumps([r for r in records if not r['match']]))
    manifest = json.loads((root / 'data/manifests/dataset_source.json').read_text())
    if manifest != historical:
        raise RuntimeError('Rebuilt source manifest differs from archived source manifest')
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--dataset-cache', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--v23-output', type=Path, help='Also rebuild historical semantic manifests and all-rotation score plan here')
    parser.add_argument('--offline', action='store_true', help='Require previously cached pinned dataset')
    parser.add_argument('--resume', action='store_true', help='Verify an existing complete design without rebuilding')
    args = parser.parse_args(argv)
    started = time.monotonic()
    repo = args.repo_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    source_checks = []
    for name, expected in SOURCE_HASHES.items():
        actual = digest(repo / name)
        if actual != expected:
            raise ValueError(f'Audited source changed: {name}; expected {expected}, found {actual}')
        source_checks.append({'path': name, 'sha256': actual})
    historical = json.loads((repo / 'experiments/llm/manifests/v2/data/manifests/dataset_source.json').read_text())
    if historical['dataset_id'] != DATASET or historical['dataset_revision'] != REVISION or historical['raw_row_count'] != 12032:
        raise ValueError('Historical dataset identity differs from the locked recovery contract')
    if output.exists():
        if not args.resume:
            raise FileExistsError('Refusing to overwrite an existing output root; use a fresh path or --resume to verify')
        records = verify_design(output, historical)
        print(json.dumps({'status': 'reused_verified_design', 'files': records}, indent=2))
        if args.v23_output:
            from prepare_v23 import prepare_v23
            print(json.dumps(prepare_v23(repo, output, args.v23_output, args.resume), indent=2))
        return 0
    if args.resume:
        raise FileNotFoundError('--resume requires an existing complete design')
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    # Only the explicit builder is called; prepare() would resolve a floating
    # dataset revision and is deliberately not part of this adapter.
    sys.path.insert(0, str(repo / 'experiments/llm/inference/v2/src'))
    from pier_llm import data_prep
    if Path(data_prep.__file__).resolve() != repo / 'experiments/llm/inference/v2/src/pier_llm/data_prep.py':
        raise RuntimeError('Imported preparation source differs from the verified repository module')
    from datasets import load_dataset
    dataset = load_dataset(DATASET, split='test', revision=REVISION,
                           cache_dir=str(args.dataset_cache.expanduser().resolve()))
    rows = [dict(row) for row in dataset]
    if len(rows) != 12032:
        raise ValueError(f'Wrong source row count: {len(rows)}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configs').mkdir()
    config_checks = []
    for filename in ['experiment.json', 'stopwords_en.txt']:
        source = repo / 'experiments/llm/inference/v2/configs' / filename
        destination = output / 'configs' / filename
        shutil.copyfile(source, destination)
        config_checks.append({'filename': filename, 'sha256': digest(source)})
    manifest = data_prep.build_from_rows(rows, output, REVISION)
    records = verify_design(output, historical)
    report = {'status': 'regenerated_design_historical_identity_verified',
              'dataset_id': DATASET, 'dataset_revision': REVISION, 'source_rows': len(rows),
              'offline': args.offline, 'source_checks': source_checks, 'copied_configs': config_checks,
              'output_files': records, 'manifest_sha256': digest(output / 'data/manifests/dataset_source.json'),
              'passed': True, 'model_inference_run': False, 'tokenizer_rendering_run': False,
              'model_download_run': False,
              'elapsed_seconds': time.monotonic() - started,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              'completed_utc': datetime.now(timezone.utc).isoformat(),
              'versions': {name: version(name) for name in ['numpy','scipy','scikit-learn','datasets','pyarrow']},
              'adapter_sha256': digest(__file__)}
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if args.v23_output:
        from prepare_v23 import prepare_v23
        print(json.dumps(prepare_v23(repo, output, args.v23_output, args.resume), indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
