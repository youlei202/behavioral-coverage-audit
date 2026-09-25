import ast
import json
import pytest
from common import ROOT
from stage_experiments import stage, STAGES

def test_fresh_staging_pins_revisions_and_preserves_design(tmp_path):
    destination=stage(tmp_path/'fresh-run')
    v2=destination/STAGES['v2'][1]
    config=json.loads((v2/'configs/experiment.json').read_text())
    old=json.loads((ROOT/'experiments/llm/inference/v2/configs/experiment.json').read_text())
    for key in old:
        if key not in ['project_root','persistent_project_root']:
            assert config[key]==old[key]
    assert config['dataset_revision']=='b189ec765aa7ed75c8acfea42df31fdae71f97be'
    roster=json.loads((v2/'configs/models.json').read_text())
    expected={r['id']:r['revision'] for r in json.loads((v2/'configs/resolved_models.json').read_text())['models']}
    assert roster['fallback']==[]
    assert {r['id']:r['revision'] for r in roster['preferred']}==expected
    for source in destination.rglob('*.py'):
        ast.parse(source.read_text(),filename=str(source))
    assert json.loads((destination/'PATH_MIGRATION.json').read_text())['experiments_executed'] is False
    with pytest.raises(ValueError,match='already exists'):
        stage(destination)
