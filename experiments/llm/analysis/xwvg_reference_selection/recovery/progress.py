"""Read committed scoring progress without loading model weights or scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def progress(root: Path) -> dict:
    config = Path(__file__).resolve().parents[1] / 'protocol/config/experiment.json'
    models = json.loads(config.read_text())['model_roster']
    rows = []
    for model in models:
        name = model.replace('/', '--')
        markers = list((root / 'outputs/raw_scores' / name).glob('shard_*.complete.json'))
        completed = sum(json.loads(path.read_text())['row_count'] for path in markers)
        report = root / 'status' / f'fresh_score_{name}_complete.json'
        done = report.is_file() and json.loads(report.read_text()).get('status') == 'complete'
        rows.append({'model_id': model, 'committed_rows': completed,
                     'required_rows': 36946, 'complete': done})
    campaigns = [root / 'status' / name for name in ('campaign.json', 'distributed_campaign.json')]
    campaigns = [path for path in campaigns if path.is_file()]
    campaign = max(campaigns, key=lambda path: path.stat().st_mtime) if campaigns else None
    state = json.loads(campaign.read_text()) if campaign else {}
    return {'campaign_status': state.get('status'), 'current_step': state.get('current_step'),
            'controller_record': str(campaign) if campaign else None,
            'active_models': [model for model, record in state.get('jobs', {}).items()
                              if record.get('status') == 'running'],
            'committed_rows': sum(row['committed_rows'] for row in rows),
            'required_rows': 295568, 'models': rows,
            'note': 'Progress from atomic commit markers; scientific completion additionally requires assembly and all analysis checks.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(progress(args.run_root.resolve()), indent=2))


if __name__ == '__main__':
    main()
