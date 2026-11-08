"""Constructed controls only: these tests do not constitute empirical evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from bca_reference_selection.data import load_inputs, aggregate_rotations, PROBABILITY_COLUMN
from bca_reference_selection.splits import fixed_splits, stable_u64

ROOT = Path(__file__).resolve().parents[5]
COMPONENT = Path(__file__).resolve().parents[1]


def fixture(directory):
    config = json.loads((COMPONENT / 'protocol/config/experiment.json').read_text())
    config.update(synthetic_fixture=True, question_count=4, category_count=2,
                  questions_per_category=2, expected_primary_cache_rows=4*2*7,
                  model_roster=['m0', 'm1'], model_revisions={'m0':'r0','m1':'r1'})
    questions = [{'base_question_id': f'q{i}', 'category': f'c{i//2}',
                  'options': ['one','two'] + (['three'] if i%2 else []),
                  'answer_index': i%2} for i in range(4)]
    qp = directory / 'questions.jsonl'
    qp.write_text(''.join(json.dumps(q)+'\n' for q in questions))
    rows = []
    for mi, model in enumerate(config['model_roster']):
        for q in questions:
            n = len(q['options'])
            probabilities = np.arange(1, n+1, dtype=float) + mi
            probabilities /= probabilities.sum()
            for family, code, dose, tracks in [('clean','C',0,[-1]), ('irrelevant_context','I',512,range(3)), ('content_deletion','D',.4,range(3))]:
                for track in tracks:
                    rows.append({'model_id':model, 'model_revision':config['model_revisions'][model],
                                 'base_question_id':q['base_question_id'], 'category':q['category'],
                                 'condition':family, 'family':family, 'dose':dose,
                                 'track':None if track==-1 else track,
                                 'semantic_condition':'C0' if track==-1 else f'{code}{track}',
                                 'option_count':n, 'gold_semantic_option':q['answer_index'],
                                 PROBABILITY_COLUMN:probabilities.tolist()})
    frame = pd.DataFrame(rows)
    cp = directory / 'average.parquet'
    frame.sample(frac=1, random_state=3).to_parquet(cp, index=False)
    return config, qp, cp, frame


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config, self.qp, self.cp, self.frame = fixture(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def bad(self, mutate, message):
        frame = self.frame.copy(deep=True)
        mutate(frame)
        frame.to_parquet(self.cp, index=False)
        with self.assertRaisesRegex(ValueError, message):
            load_inputs(self.cp, self.qp, self.config)

    def test_identity_alignment_masks_and_masses(self):
        data = load_inputs(self.cp, self.qp, self.config)
        self.assertEqual(data.probabilities['content_deletion'].shape, (2,4,4,3))
        self.assertTrue(np.array_equal(data.mask[:,0].sum(axis=1), [2,3,2,3]))
        for family in data.probabilities:
            self.assertTrue(np.all(data.probabilities[family][...,2][:,[0,2],:] == 0))
            np.testing.assert_allclose(data.masses[family], [.5,1/6,1/6,1/6])
        self.assertTrue(data.report['passed'])

    def test_original_question_digest_required(self):
        config = deepcopy(self.config)
        config.pop('synthetic_fixture')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            load_inputs(self.cp, self.qp, config)

    def test_bad_dose(self):
        self.bad(lambda f: f.__setitem__('dose',0), 'Incorrect stress dose')

    def test_bad_clean_identity(self):
        self.bad(lambda f: f.loc.__setitem__((0,'track'),0), 'Malformed clean')

    def test_bad_gold_identity(self):
        self.bad(lambda f: f.loc.__setitem__((0,'gold_semantic_option'),1), 'reference answer mismatch')

    def test_bad_semantic_identity(self):
        self.bad(lambda f: f.loc.__setitem__((1,'semantic_condition'),'I2'), 'duplicate semantic keys')

    def test_bad_revision(self):
        self.bad(lambda f: f.loc.__setitem__((0,'model_revision'),'other'), 'revision mismatch')

    def test_nonzero_padding(self):
        self.bad(lambda f: f.at.__setitem__((0,PROBABILITY_COLUMN),[.3,.7,.1]), 'Nonzero invalid')

    def test_unormalized_vector(self):
        self.bad(lambda f: f.at.__setitem__((0,PROBABILITY_COLUMN),[.3,.6]), 'normalization violated')

    def test_nonfinite_vector(self):
        self.bad(lambda f: f.at.__setitem__((0,PROBABILITY_COLUMN),[np.nan,1]), 'nonfinite')

    def test_count_and_duplicate(self):
        self.bad(lambda f: f.loc.__setitem__((1,slice(None)),f.loc[0]), 'duplicate semantic keys')

    def test_category_mismatch(self):
        self.bad(lambda f: f.loc.__setitem__((0,'category'),'c1'), 'Category mismatch')

    def test_roundoff_repair_logged(self):
        self.frame.at[0,PROBABILITY_COLUMN] = [.3,.7+1e-12]
        self.frame.to_parquet(self.cp,index=False)
        data = load_inputs(self.cp,self.qp,self.config)
        self.assertGreater(data.report['max_probability_repair_l1'], 0)
        self.assertGreater(data.report['roundoff_corrected_vectors'], 0)

    def rotations(self):
        rows=[]
        for row in self.frame.to_dict('records'):
            vector=np.asarray(row.pop(PROBABILITY_COLUMN))
            for rotation in range(row['option_count']):
                # Rotation-specific changes average away; inputs already semantic.
                shift=(rotation-(row['option_count']-1)/2)*.01
                changed=vector.copy(); changed[0]+=shift; changed[1]-=shift
                rows.append({**row,'permutation_id':rotation,
                             'schema_version':'pier_v23_candidate_score_v1',
                             'prompt_id':f'{row["base_question_id"]}__{row["semantic_condition"]}__cyclic_r{rotation:02d}',
                             'semantic_remapped_probabilities':changed.tolist()})
        return pd.DataFrame(rows)

    def test_complete_rotation_aggregation_matches_vector_average(self):
        rotations=self.rotations()
        path=self.directory/'rotations.parquet'; rotations.to_parquet(path,index=False)
        output=self.directory/'rebuilt.parquet'
        report=aggregate_rotations(path,output,self.qp,self.config)
        self.assertTrue(report['passed'])
        direct=load_inputs(self.cp,self.qp,self.config)
        rebuilt=load_inputs(output,self.qp,self.config)
        for family in direct.probabilities:
            np.testing.assert_allclose(direct.probabilities[family],rebuilt.probabilities[family],atol=1e-15)

    def test_missing_rotation_rejected(self):
        rotations=self.rotations().iloc[1:]
        path=self.directory/'rotations.parquet'; rotations.to_parquet(path,index=False)
        with self.assertRaisesRegex(ValueError,'Rotation row count'):
            aggregate_rotations(path,self.directory/'rebuilt.parquet',self.qp,self.config)

    def test_duplicated_rotation_rejected(self):
        rotations=self.rotations()
        rotations.loc[1,'permutation_id']=0
        path=self.directory/'rotations.parquet'; rotations.to_parquet(path,index=False)
        with self.assertRaisesRegex(ValueError,'cyclic rotation set'):
            aggregate_rotations(path,self.directory/'rebuilt.parquet',self.qp,self.config)

    def test_conflicting_rotation_metadata_rejected(self):
        rotations=self.rotations()
        rotations.loc[1,'dose']=1
        path=self.directory/'rotations.parquet'; rotations.to_parquet(path,index=False)
        with self.assertRaisesRegex(ValueError,'Conflicting rotation metadata'):
            aggregate_rotations(path,self.directory/'rebuilt.parquet',self.qp,self.config)

    def test_historical_split_equivalence(self):
        # Import the historical implementation as an independent oracle.
        import sys
        historical=ROOT/'experiments/llm/analysis/v23/src'
        sys.path.insert(0,str(historical))
        from pier_llm_v23.core import fixed_splits as original
        questions=pd.DataFrame([{'base_question_id':f'{c}-{i}','category':c} for c in ['a','b'] for i in range(40)])
        seeds=self.config['split_seeds']
        expected=original(questions,seeds)
        actual=fixed_splits(questions,seeds)
        for seed in seeds:
            self.assertEqual(actual[seed].fitting_ids,expected[seed].fitting_ids)
            self.assertEqual(actual[seed].evaluation_ids,expected[seed].evaluation_ids)
            self.assertFalse(actual[seed].fitting_ids & actual[seed].evaluation_ids)


class ShardTests(unittest.TestCase):
    setUp = DataTests.setUp
    tearDown = DataTests.tearDown
    def shard_fixture(self):
        from bca_reference_selection.data import fingerprint
        config, qp, cp, frame = self.config,self.qp,self.cp,self.frame
        v2=self.directory/'v2'; v23=self.directory/'v23'
        prompts=[]
        selected={q['base_question_id']:q for q in map(json.loads,qp.read_text().splitlines())}
        for row in frame[frame['model_id']=='m0'].to_dict('records'):
            n=row['option_count']; q=selected[row['base_question_id']]
            for rotation in range(n):
                v2s=[(i-rotation)%n for i in range(n)]
                prompts.append({**{k:row[k] for k in ['base_question_id','category','condition','family','dose','track','semantic_condition','option_count']},
                    'prompt_id':f'{row["base_question_id"]}__{row["semantic_condition"]}__cyclic_r{rotation:02d}',
                    'source_v2_prompt_id':f'{row["base_question_id"]}__{row["semantic_condition"]}',
                    'permutation_id':rotation,'semantic_answer_index':q['answer_index'],
                    'semantic_to_visible':[(i+rotation)%n for i in range(n)],
                    'visible_to_semantic':v2s,'options':[q['options'][i] for i in v2s],
                    'canonical_content_sha256':'test-canonical-hash'})
        manifest=v23/'data/permutations/score_semantic_manifest.jsonl'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(''.join(json.dumps(p)+'\n' for p in prompts))
        for mi, model in enumerate(config['model_roster']):
            for is_v2,root in [(True,v2),(False,v23)]:
                rows=[]
                schema='pier_raw_label_scores_v2' if is_v2 else 'pier_v23_candidate_score_v1'
                for p in prompts:
                    if (p['permutation_id']==0) != is_v2: continue
                    n=p['option_count']
                    sem=np.arange(1,n+1,dtype=float)+mi; sem/=sem.sum()
                    vis=sem[p['visible_to_semantic']]
                    row={**p,'schema_version':schema,'model_id':model,'model_revision':config['model_revisions'][model]}
                    if is_v2:
                        row.update(prompt_id=p['source_v2_prompt_id'],semantic_probabilities=sem.tolist(),probabilities=vis.tolist(),presented_to_original=p['visible_to_semantic'])
                    else:
                        row.update(semantic_remapped_probabilities=sem.tolist(),visible_label_probabilities=vis.tolist(),gold_semantic_option=p['semantic_answer_index'],canonical_prompt_hash=p['canonical_content_sha256'])
                    rows.append(row)
                path=root/f'outputs/raw_scores/{model}/shard_00000.parquet'
                path.parent.mkdir(parents=True); pd.DataFrame(rows).to_parquet(path,index=False)
                path.with_suffix('.meta.json').write_text(json.dumps({'schema_version':schema,'model_id':model,'model_revision':config['model_revisions'][model],'row_count':len(rows),'sha256':fingerprint(path)['sha256']}))
        return v2,v23

    def test_reconstruction_from_shards(self):
        from bca_reference_selection.data import reconstruct_from_shards
        v2,v23=self.shard_fixture()
        output=self.directory/'rebuilt_shards.parquet'
        result=reconstruct_from_shards(v2,v23,output,self.qp,self.config)
        self.assertTrue(result['passed'])
        direct=load_inputs(self.cp,self.qp,self.config)
        rebuilt=load_inputs(output,self.qp,self.config)
        for family in direct.probabilities:
            np.testing.assert_allclose(direct.probabilities[family],rebuilt.probabilities[family],atol=1e-15)

    def test_shard_sidecar_tamper_rejected(self):
        from bca_reference_selection.data import reconstruct_from_shards
        v2,v23=self.shard_fixture()
        sidecar=v2/'outputs/raw_scores/m0/shard_00000.meta.json'
        meta=json.loads(sidecar.read_text());meta['sha256']='bad';sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'checksum/schema mismatch'):
            reconstruct_from_shards(v2,v23,self.directory/'rebuilt_shards.parquet',self.qp,self.config)


if __name__=='__main__':
    unittest.main()
