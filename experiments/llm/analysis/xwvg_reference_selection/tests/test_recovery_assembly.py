"""CPU assembly accepts only complete, committed and semantically valid scores."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

COMPONENT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COMPONENT/'recovery'))
import assemble_scores as assembly
import score_endpoints as score

spec = importlib.util.spec_from_file_location('scoring_fixture', Path(__file__).with_name('test_recovery_scoring.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))


def fixture(root):
    config=json.loads((COMPONENT/'protocol/config/experiment.json').read_text())
    config.update(synthetic_fixture=True,question_count=3,category_count=1,questions_per_category=3,
                  expected_primary_cache_rows=8*3*7)
    prompts,rendered=helpers.fixture()
    semantic=root/'data/permutations/score_semantic_manifest.jsonl'
    executed=root/'data/permutations/score_execution_manifest.jsonl'
    jsonl(semantic,prompts)
    jsonl(executed,[{**row,'inference_required':True,'rotation0_reuse':False} for row in prompts])
    qpath=root/'questions.jsonl'
    jsonl(qpath,[{'base_question_id':f'q{i}','category':'subject','options':['a','b','c'],'answer_index':1}
                 for i in range(3)])
    for model in config['model_roster']:
        name=score.slug(model)
        rendered_path=root/f'data/permutations/rendered/{name}.jsonl'
        jsonl(rendered_path,rendered)
        identity={'model_id':model,'model_revision':config['model_revisions'][model],
            'precision':'bfloat16','candidate_definition':'summed_full_continuation_log_likelihood',
            'semantic_manifest_sha256':score.file_sha256(executed),
            'rendered_prompt_sha256':score.file_sha256(rendered_path),
            'measurement_origin':'fresh_endpoint_scoring','historical_cache_restoration':False,
            'all_rotations_scored':True,'hardware':{'name':'synthetic fixture'},
            'versions':{'python':'synthetic fixture'},'source_sha256':{'fixture':'fixture'},
            'driver_sha256':'fixture','model_inventory':{'revision':config['model_revisions'][model]},
            'model_adapter':'causal_lm'}
        fingerprint=hashlib.sha256(score.canonical(identity)).hexdigest()
        score.atomic_json(root/f'status/fresh_score_{name}_identity.json',
                          {'input_fingerprint':fingerprint,**identity})
        rows=helpers.score_rows(prompts,rendered)
        for row in rows:
            row.update(model_id=model,model_revision=config['model_revisions'][model],
                       score_origin='fresh_endpoint_scoring')
        score.publish_shard(root/'outputs/raw_scores'/name,rows,fingerprint,score.file_sha256(executed),
                            model_id=model,revision=config['model_revisions'][model],execution={'stage':'score'})
        score.atomic_json(root/f'status/fresh_score_{name}_complete.json',
            {'input_fingerprint':fingerprint,'status':'complete','stage':'score','model_id':model,
             'model_revision':config['model_revisions'][model],'completed_rows':len(prompts),
             'fresh_measurements':True,'historical_cache_restoration':False})
    return config,qpath


def replace_rows(path, rows):
    pd.DataFrame(rows).to_parquet(path,index=False)
    meta_path,commit_path=path.with_suffix('.meta.json'),path.with_suffix('.complete.json')
    meta=json.loads(meta_path.read_text());commit=json.loads(commit_path.read_text())
    meta.update(row_count=len(rows),prompt_ids=[r['prompt_id'] for r in rows],sha256=score.file_sha256(path))
    commit.update(row_count=len(rows),sha256=meta['sha256'])
    score.atomic_json(meta_path,meta);score.atomic_json(commit_path,commit)


class AssemblyChecks(unittest.TestCase):
    def test_complete_eight_models_aggregate_and_resume_with_provenance(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            result=assembly.assemble(root,questions,config)
            self.assertEqual(result['model_count'],8)
            self.assertEqual(result['rotation_rows'],504)
            self.assertEqual(result['balanced_probability_rows'],168)
            provenance=json.loads((root/'status/measurement_provenance.json').read_text())
            self.assertEqual(provenance['measurement_origin'],'fresh_endpoint_scoring')
            self.assertFalse(provenance['historical_cache_restoration'])
            self.assertEqual(len(provenance['input_identity']['model_records']),8)
            averaged=pd.read_parquet(result['averaged_cache']['path'])
            self.assertEqual(len(averaged),168)
            self.assertEqual(set(averaged.model_id),set(config['model_roster']))
            again=assembly.assemble(root,questions,config)
            self.assertEqual(again['invocation_status'],'reused_verified_complete_cache')
            self.assertEqual(again['averaged_cache'],result['averaged_cache'])
            # No staging-directory path survives in published provenance.
            self.assertNotIn('.fresh-assembly-',json.dumps(provenance))

    def test_partial_roster_cannot_produce_balanced_cache(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            model=config['model_roster'][-1]
            (root/f'status/fresh_score_{score.slug(model)}_complete.json').rename(root/'incomplete_model.json')
            with self.assertRaisesRegex(FileNotFoundError,'Complete eight-model'):
                assembly.assemble(root,questions,config)
            self.assertFalse((root/'outputs/analysis/permutation_averaged_semantic_scores.parquet').exists())
            self.assertIn(score.slug(model),(root/'status/fresh_assembly_blocked.json').read_text())

    def test_false_commit_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            path=next((root/'outputs/raw_scores').glob('*/shard_*.complete.json'))
            record=json.loads(path.read_text());record['complete']=False;score.atomic_json(path,record)
            with self.assertRaisesRegex(ValueError,'commit/checksum'):
                assembly.assemble(root,questions,config)

    def test_missing_rotation_cannot_hide_behind_recomputed_hashes(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            path=next((root/'outputs/raw_scores').glob('*/shard_*.parquet'))
            replace_rows(path,pd.read_parquet(path).to_dict('records')[1:])
            with self.assertRaisesRegex(ValueError,'Incomplete all-rotation'):
                assembly.assemble(root,questions,config)

    def test_corrupt_semantic_probability_rejected_after_hash_reseal(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            path=next((root/'outputs/raw_scores').glob('*/shard_*.parquet'))
            rows=pd.read_parquet(path).to_dict('records')
            rows[0]['semantic_remapped_probabilities']=[.1,.5,.4]
            replace_rows(path,rows)
            with self.assertRaisesRegex(ValueError,'Semantic score mapping'):
                assembly.assemble(root,questions,config)

    def test_identity_tamper_and_existing_cache_tamper_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            path=next((root/'status').glob('fresh_score_*_identity.json'))
            record=json.loads(path.read_text());record['precision']='float16';score.atomic_json(path,record)
            with self.assertRaisesRegex(ValueError,'identity checksum'):
                assembly.assemble(root,questions,config)
        with TemporaryDirectory() as directory:
            root=Path(directory);config,questions=fixture(root)
            result=assembly.assemble(root,questions,config)
            path=Path(result['averaged_cache']['path'])
            with path.open('ab') as handle:handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'cache checksum'):
                assembly.assemble(root,questions,config)


if __name__=='__main__':
    unittest.main()
