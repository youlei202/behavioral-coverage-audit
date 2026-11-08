#!/usr/bin/env python3
"""Inspect committed XfqA progress without loading model weights or scores."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def progress(root):
    root = Path(root)
    config = read(root / 'plan/config/experiment.json')
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'models': [],
              'complete': (read(root / 'status/complete.json') or {}).get('status') == 'complete'}
    for model in config['new_models']:
        name = model['id'].replace('/', '--')
        directory = root / 'outputs/raw_scores' / name
        rows = shards = sequences = 0
        seconds = 0.
        for path in sorted(directory.glob('shard_*.complete.json')):
            marker = read(path)
            metadata = read(path.with_name(path.name.replace('.complete.json', '.meta.json')))
            if not marker.get('complete') or metadata is None:
                continue
            rows += int(marker['row_count'])
            shards += 1
            for batch in metadata.get('execution', {}).get('batches', []):
                sequences += int(batch.get('candidate_continuations', 0))
                seconds += float(batch['seconds'])
        proof = read(root / f'status/model_scoring/{name}_complete.json')
        failure = read(root / f'status/model_scoring/{name}_last_failure.json')
        resource_failure = read(root / f'status/model_scoring/{name}_resource_failure.json')
        result['models'].append({'model_id': model['id'], 'committed_prompts': rows,
            'required_prompts': config['scoring']['required_prompts_per_model'],
            'committed_shards': shards, 'candidate_sequences': sequences,
            'committed_inference_seconds': seconds,
            'status': 'complete' if proof and proof.get('status') == 'complete' else 'in_progress',
            'last_failure': failure, 'resource_failure': resource_failure})
    result['total_committed_prompts'] = sum(m['committed_prompts'] for m in result['models'])
    result['total_required_prompts'] = config['expected_outputs']['new_rotation_rows']
    result['analysis'] = {name: read(root / 'analysis' / (name + '.json'))
        for name in ('point_complete', 'primary_refit_complete', 'secondary_complete', 'bootstrap_complete')}
    result['delivery'] = read(root / 'status/package_complete.json')
    # Compact summaries avoid dumping large artifact inventories into monitors.
    for name, value in result['analysis'].items():
        if value:
            result['analysis'][name] = {k: value[k] for k in ('status', 'point_fit_count', 'replicates', 'fits') if k in value}
    result['timing_scope'] = 'Committed scorer batch wall time only; excludes loading, repeats, download and unfinished batches; not dedicated GPU-hours.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(progress(args.output_root), indent=2))
