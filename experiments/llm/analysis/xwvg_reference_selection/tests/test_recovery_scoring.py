"""CPU-only tests for the fresh scoring runner's provenance and resume gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'recovery/score_endpoints.py'
spec = importlib.util.spec_from_file_location('score_endpoints', SOURCE)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def fixture():
    prompts, rendered = [], []
    for q in range(3):
        qid = f'q{q}'
        for condition in score.CONDITIONS:
            family = 'clean' if condition == 'C0' else 'irrelevant_context' if condition[0] == 'I' else 'content_deletion'
            for rotation in range(3):
                content = f'{qid}:{condition}:{rotation}'
                pid = f'{qid}__{condition}__cyclic_r{rotation:02d}'
                sha = score.hashlib.sha256(content.encode()).hexdigest()
                prompts.append(dict(prompt_id=pid, source_v2_prompt_id=f'{qid}__{condition}', base_question_id=qid,
                    category='subject', semantic_condition=condition, condition=family, family=family,
                    dose=0. if condition=='C0' else 512. if condition[0]=='I' else .4,
                    track=None if condition=='C0' else int(condition[1]), permutation_id=rotation, option_count=3,
                    option_labels=list('ABC'), candidate_continuations=[' (A)',' (B)',' (C)'],
                    semantic_to_visible=[(i+rotation)%3 for i in range(3)],
                    visible_to_semantic=[(i-rotation)%3 for i in range(3)], semantic_answer_index=1,
                    canonical_content=content, canonical_content_sha256=sha,
                    inference_required=rotation!=0, rotation0_reuse=rotation==0))
                rendered.append(dict(prompt_id=pid, canonical_content_sha256=sha, rendered_prompt=content,
                                     rendered_prompt_sha256=sha, prompt_token_count=10+q*50+rotation))
    return prompts, rendered


def score_rows(prompts, rendered):
    rmap={r['prompt_id']:r for r in rendered}
    rows=[]
    for prompt in prompts:
        visible=np.array([.2,.5,.3])
        semantic=np.empty(3)
        semantic[prompt['visible_to_semantic']]=visible
        row={key:prompt[key] for key in ['prompt_id','source_v2_prompt_id','base_question_id','category',
             'semantic_condition','condition','family','dose','track','permutation_id','option_count',
             'semantic_to_visible','visible_to_semantic','candidate_continuations']}
        row.update(schema_version=score.SCHEMA, model_id='test/model', model_revision='revision',
            visible_label_probabilities=visible.tolist(), semantic_remapped_probabilities=semantic.tolist(),
            candidate_sequence_log_likelihoods=np.log(visible).tolist(), gold_semantic_option=1,
            gold_semantic_probability=float(semantic[1]), canonical_prompt_hash=prompt['canonical_content_sha256'],
            rendered_prompt_hash=rmap[prompt['prompt_id']]['rendered_prompt_sha256'])
        rows.append(row)
    return rows


class RecoveryScoringChecks(unittest.TestCase):
    def test_all_rotations_are_validated_despite_historical_reuse_flag(self):
        prompts,rendered=fixture()
        report=score.validate_design(prompts,rendered,historical_counts=False)
        self.assertEqual(report['rotation_zero_count'],21)
        self.assertEqual(report['prompt_count'],63)
        with self.assertRaisesRegex(ValueError,'Historical design'):
            score.validate_design(prompts,rendered)
        with self.assertRaisesRegex(ValueError,'Incomplete cyclic'):
            score.validate_design(prompts[1:],rendered,historical_counts=False)

    def test_strict_semantic_and_rendered_identity(self):
        prompts,rendered=fixture()
        bad=deepcopy(prompts)
        bad[0]['semantic_to_visible']=[2,1,0]
        with self.assertRaisesRegex(ValueError,'cyclic semantic'):
            score.validate_design(bad,rendered,historical_counts=False)
        bad=deepcopy(rendered)
        bad[0]['rendered_prompt']+=' altered'
        with self.assertRaisesRegex(ValueError,'Rendered prompt checksum'):
            score.validate_design(prompts,bad,historical_counts=False)

    def test_pilot_covers_all_conditions_rotation_strata_and_lengths(self):
        prompts,rendered=fixture()
        rmap={r['prompt_id']:r for r in rendered}
        ids=score.fixed_pilot_ids(prompts,rmap)
        self.assertEqual(len(ids),42)
        selected=[r for r in prompts if r['prompt_id'] in ids]
        for condition in score.CONDITIONS:
            for zero in (True,False):
                rows=[r for r in selected if r['semantic_condition']==condition and (r['permutation_id']==0)==zero]
                self.assertEqual(len(rows),3)
                self.assertEqual({r['base_question_id'] for r in rows},{'q0','q1','q2'})
        changed=deepcopy(prompts)
        for row in changed:
            row['fake_outcome']=999
        self.assertEqual(ids,score.fixed_pilot_ids(changed,rmap))

    def test_shards_roundtrip_resume_integrity_and_missing_commit(self):
        prompts,rendered=fixture()
        by_id={r['prompt_id']:r for r in prompts}
        rmap={r['prompt_id']:r for r in rendered}
        rows=score_rows(prompts[:7],rendered)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            shard=score.publish_shard(root,rows,'identity','manifest',model_id='test/model',revision='revision',execution={})
            done,meta=score.scan_completed(root,'identity',by_id,rmap,'test/model','revision')
            self.assertEqual(done,{r['prompt_id'] for r in rows})
            self.assertEqual(len(meta),1)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                score.scan_completed(root,'other',by_id,rmap,'test/model','revision')
            marker=shard.with_suffix('.complete.json')
            marker.rename(root/'interrupted_marker.json')
            self.assertEqual(score.scan_completed(root,'identity',by_id,rmap,'test/model','revision')[0],set())
            (root/'interrupted_marker.json').rename(marker)
            with shard.open('ab') as f: f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                score.scan_completed(root,'identity',by_id,rmap,'test/model','revision')

    def test_commit_and_sidecar_metadata_are_mandatory(self):
        prompts,rendered=fixture()
        by_id={r['prompt_id']:r for r in prompts}
        rmap={r['prompt_id']:r for r in rendered}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            path=score.publish_shard(root,score_rows(prompts[:2],rendered),'identity','manifest',
                                     model_id='test/model',revision='revision',execution={})
            marker=path.with_suffix('.complete.json')
            original=json.loads(marker.read_text())
            score.atomic_json(marker,{**original,'complete':False})
            with self.assertRaisesRegex(ValueError,'commit mismatch'):
                score.scan_completed(root,'identity',by_id,rmap,'test/model','revision','manifest')
            score.atomic_json(marker,original)
            with self.assertRaisesRegex(ValueError,'manifest identity'):
                score.scan_completed(root,'identity',by_id,rmap,'test/model','revision','another_manifest')
            meta_path=path.with_suffix('.meta.json')
            meta=json.loads(meta_path.read_text())
            score.atomic_json(meta_path,{**meta,'measurement_origin':'historical'})
            with self.assertRaisesRegex(ValueError,'origin metadata'):
                score.scan_completed(root,'identity',by_id,rmap,'test/model','revision','manifest')

    def test_full_score_cannot_skip_pilot_and_repilot_preserves_measurement(self):
        prompts,rendered=fixture()
        repo=Path(__file__).resolve().parents[5]
        model_id='Qwen/Qwen2.5-7B-Instruct'
        revision='a09a35458c702b33eeacc393d103063234e8bc28'
        cuda=SimpleNamespace(is_available=lambda:True,set_device=lambda _:None,
            get_device_name=lambda _:'constructed GPU',get_device_capability=lambda _:(10,0),
            get_device_properties=lambda _:SimpleNamespace(total_memory=100))
        torch=SimpleNamespace(cuda=cuda,device=lambda x:x,version=SimpleNamespace(cuda='test'))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest=root/'data/permutations/score_execution_manifest.jsonl'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in prompts))
            rendered_path=root/'data/permutations/rendered'/f'{score.slug(model_id)}.jsonl'
            rendered_path.parent.mkdir()
            rendered_path.write_text(''.join(json.dumps(r)+'\n' for r in rendered))
            args=SimpleNamespace(source_repo=repo,run_root=root,model_path=root/'snapshot',model_id=model_id,
                model_adapter=None,interface_change_reason='constructed fixture',device='cuda:0',resume=False,
                stage='score',batch_size=1,shard_size=128)
            actual_validate=score.validate_design
            with patch.dict('sys.modules',{'torch':torch}), patch.object(score,'runtime_versions',return_value={'test':True}), \
                 patch.object(score,'snapshot_inventory',return_value={}), \
                 patch.object(score,'validate_design',side_effect=lambda p,r:actual_validate(p,r,historical_counts=False)), \
                 patch.object(score,'import_historical_scorer',side_effect=AssertionError('must not load a model')):
                with self.assertRaisesRegex(ValueError,'measured pilot is required'):
                    score.run(args)
                identity=json.loads((root/'status'/f'fresh_score_{score.slug(model_id)}_identity.json').read_text())['input_fingerprint']
                pilot_ids=score.fixed_pilot_ids(prompts,{r['prompt_id']:r for r in rendered})
                rows=score_rows([r for r in prompts if r['prompt_id'] in pilot_ids],rendered)
                for row in rows:
                    row.update(model_id=model_id,model_revision=revision)
                score.publish_shard(root/'outputs/raw_scores'/score.slug(model_id),rows,identity,score.file_sha256(manifest),
                    model_id=model_id,revision=revision,execution={'stage':'pilot','batches':[]})
                previous={'input_fingerprint':identity,'status':'complete','pilot_measured_prompt_count':len(pilot_ids),
                          'scoring_seconds':123.456,'original_marker':'keep measured timing'}
                pilot_path=root/'status'/f'fresh_score_{score.slug(model_id)}_pilot.json'
                score.atomic_json(pilot_path,previous)
                args.stage,args.resume='pilot',True
                self.assertEqual(score.run(args),previous)
                self.assertEqual(json.loads(pilot_path.read_text()),previous)

    def test_duplicate_across_shards_rejected(self):
        prompts,rendered=fixture()
        by_id={r['prompt_id']:r for r in prompts}
        rmap={r['prompt_id']:r for r in rendered}
        rows=score_rows(prompts[:2],rendered)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for _ in range(2):
                score.publish_shard(root,rows,'identity','manifest',model_id='test/model',revision='revision',execution={})
            with self.assertRaisesRegex(ValueError,'Duplicate prompts across'):
                score.scan_completed(root,'identity',by_id,rmap,'test/model','revision')

    def test_semantic_probabilities_must_reproduce_full_likelihoods(self):
        prompts,rendered=fixture()
        by_id={r['prompt_id']:r for r in prompts}
        rmap={r['prompt_id']:r for r in rendered}
        rows=score_rows(prompts[:2],rendered)
        rows[0]['candidate_sequence_log_likelihoods']=[-1.,-1.,-1.]
        with self.assertRaisesRegex(ValueError,'continuation likelihoods'):
            score.validate_score_rows(rows,by_id,rmap,'test/model','revision')

    def test_snapshot_requires_exact_revision_and_all_index_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot=Path(temp)/'revision'
            snapshot.mkdir()
            (snapshot/'config.json').write_text('{}')
            (snapshot/'model-1.safetensors').write_bytes(b'constructed weight')
            (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'missing.safetensors'}}))
            with self.assertRaisesRegex(ValueError,'shard missing'):
                score.snapshot_inventory(snapshot,'revision')
            with self.assertRaisesRegex(ValueError,'exact pinned'):
                score.snapshot_inventory(snapshot,'different')


if __name__=='__main__':
    unittest.main()
