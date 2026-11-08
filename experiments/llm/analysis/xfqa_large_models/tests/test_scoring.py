"""XfqA scoring identity, resource, native-boundary and exact likelihood checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scoring.py'
SPEC = importlib.util.spec_from_file_location('xfqa_scoring_tested', PATH)
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)
REPO = Path(__file__).resolve().parents[5]


class ScoringChecks(unittest.TestCase):
    def test_staging_is_byte_identical_and_refuses_different_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root/'source', root/'input'
            source.write_bytes(b'original source bytes\n')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            scoring.copy_exact(source, destination, expected)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(destination.stat().st_mode & 0o222, 0)
            scoring.copy_exact(source, destination, expected)
            destination.chmod(0o600)
            destination.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                scoring.copy_exact(source, destination, expected)
            self.assertEqual(destination.read_bytes(), b'changed')

    def test_snapshot_asset_selection_loads_only_indexed_safetensors(self):
        names = ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json',
                 'chat_template.jinja', 'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                 'model-00002-of-00002.safetensors', 'pytorch_model.bin', 'original/consolidated.safetensors',
                 'README.md']
        selected = scoring.selected_asset_names(names, {'weight_map': {'a': names[6], 'b': names[7]}})
        self.assertIn('chat_template.jinja', selected)
        self.assertIn(names[6], selected)
        self.assertIn(names[7], selected)
        self.assertNotIn('pytorch_model.bin', selected)
        self.assertNotIn('original/consolidated.safetensors', selected)
        with self.assertRaises(ValueError):
            scoring.selected_asset_names(names, {'weight_map': {'missing': 'missing.safetensors'}})

    def test_maximum_probe_uses_candidate_expansion_and_deterministic_ties(self):
        prompts = [{'prompt_id': 'long-few', 'option_count': 2},
                   {'prompt_id': 'short-many-z', 'option_count': 10},
                   {'prompt_id': 'short-many-a', 'option_count': 10}]
        rendered = {'long-few': {'prompt_token_count': 100}, 'short-many-z': {'prompt_token_count': 50},
                    'short-many-a': {'prompt_token_count': 50}}
        candidates = {'2': [{'token_count': 3}], '10': [{'token_count': 3}]}
        result = scoring.maximum_workload(prompts, rendered, candidates)
        self.assertEqual(result['prompt_id'], 'short-many-a')
        self.assertEqual(result['candidate_expanded_tokens'], 530)
        self.assertTrue(result['outcome_blind'])

    def test_reserve_monitor_checks_both_gpu_and_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            value = {'gpu_free_bytes': 100, 'cpu_available_bytes': 200}
            monitor = scoring.ResourceWatchdog(lambda: dict(value), 50, 100, Path(temp)/'failure.json')
            monitor.check()
            value['cpu_available_bytes'] = 99
            with self.assertRaises(RuntimeError):
                monitor.check()
            value['cpu_available_bytes'] = 101
            value['gpu_free_bytes'] = 49
            with self.assertRaises(RuntimeError):
                monitor.check()
            self.assertEqual(monitor.minimum_gpu_free, 49)

    def test_model_worker_lock_prevents_concurrent_model_writers(self):
        with tempfile.TemporaryDirectory() as temp:
            with scoring.exclusive_model_lock(temp, 'Qwen/QwQ-32B'):
                with self.assertRaises(RuntimeError):
                    with scoring.exclusive_model_lock(temp, 'Qwen/QwQ-32B'):
                        pass
                with scoring.exclusive_model_lock(temp, 'google/gemma-3-27b-it'):
                    pass
            with scoring.exclusive_model_lock(temp, 'Qwen/QwQ-32B'):
                pass

    def test_repeated_model_outputs_are_measured_against_fixed_tolerances(self):
        row = {'prompt_id': 'p', 'candidate_sequence_log_likelihoods': [-1., -2.],
               'semantic_remapped_probabilities': [.75, .25]}
        thresholds = {'max_abs_continuation_loglikelihood_difference': 1e-4,
                      'max_abs_candidate_probability_difference': 1e-5,
                      'mean_pilot_tv_difference': 1e-6, 'top_option_mismatches': 0}
        exact = scoring.compare_repeat([row], [dict(row)], thresholds)
        self.assertTrue(exact['passed'])
        self.assertTrue(exact['bitwise_equal'])
        changed = {**row, 'semantic_remapped_probabilities': [.25, .75]}
        self.assertFalse(scoring.compare_repeat([row], [changed], thresholds)['passed'])
        with self.assertRaises(ValueError):
            scoring.compare_repeat([row], [{**row, 'prompt_id': 'other'}], thresholds)

    def test_gpu_uuid_accepts_cuda_and_nvidia_smi_spellings(self):
        self.assertEqual(scoring.normalized_gpu_uuid('abc-def'), 'GPU-abc-def')
        self.assertEqual(scoring.normalized_gpu_uuid('GPU-ABC-DEF'), 'GPU-abc-def')

    def test_new_model_shards_resume_only_committed_hash_valid_rows(self):
        import numpy as np
        helper = scoring.original(REPO)
        pid = 'qid__C0__cyclic_r00'
        prompt = {'prompt_id':pid, 'base_question_id':'qid', 'category':'subject', 'semantic_condition':'C0',
                  'condition':'clean', 'family':'clean', 'dose':0., 'track':None, 'permutation_id':0,
                  'option_count':2, 'semantic_to_visible':[0,1], 'visible_to_semantic':[0,1],
                  'candidate_continuations':[' (A)',' (B)'], 'semantic_answer_index':1,
                  'canonical_content_sha256':'canonical'}
        row = {**prompt, 'schema_version':helper.SCHEMA, 'model_id':'Qwen/QwQ-32B', 'model_revision':'a'*40,
               'canonical_prompt_hash':'canonical', 'rendered_prompt_hash':'rendered',
               'candidate_sequence_log_likelihoods':np.log([.25,.75]).tolist(),
               'visible_label_probabilities':[.25,.75], 'semantic_remapped_probabilities':[.25,.75],
               'gold_semantic_option':1, 'gold_semantic_probability':.75}
        rendered = {pid:{'rendered_prompt_sha256':'rendered'}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = helper.publish_shard(root, [row], 'identity', 'manifest', model_id=row['model_id'],
                                        revision=row['model_revision'], execution={'stage':'pilot','batches':[]})
            scan = lambda: helper.scan_completed(root,'identity',{pid:prompt},rendered,row['model_id'],row['model_revision'],'manifest')
            self.assertEqual(scan()[0], {pid})
            marker = path.with_suffix('.complete.json')
            contents = marker.read_bytes()
            marker.unlink()
            self.assertEqual(scan()[0], set())
            marker.write_bytes(contents)
            path.write_bytes(b'corrupt parquet')
            with self.assertRaisesRegex(ValueError,'checksum/commit'):
                scan()

    def test_old_roster_cannot_be_sent_to_fresh_model_driver(self):
        args = SimpleNamespace(repo_root=REPO, plan_root=REPO/'outputs/xfqa_large_models/20261108/plan',
                               output_root=REPO/'outputs/xfqa_large_models/20261108', model_id='microsoft/phi-4')
        if not (args.output_root/'model_lock.json').exists():
            self.skipTest('Real frozen model lock not present')
        with self.assertRaisesRegex(ValueError, 'Only the three'):
            scoring.model_record(args)

    def test_expanded_input_config_preserves_old_revisions_without_editing_old_config(self):
        args = SimpleNamespace(plan_root=REPO/'outputs/xfqa_large_models/20261108/plan',
                               output_root=REPO/'outputs/xfqa_large_models/20261108')
        if not (args.output_root/'model_lock.json').exists():
            self.skipTest('Real frozen model lock not present')
        original = args.plan_root/'config/original_xwvg_input_config.json'
        before = scoring.sha(original)
        new = scoring.derived_input_config(args)
        old = scoring.load(original)
        self.assertEqual(len(new['model_roster']), 11)
        self.assertEqual(new['expected_primary_cache_rows'], 43120)
        for model in old['model_roster']:
            self.assertEqual(new['model_revisions'][model], old['model_revisions'][model])
        self.assertEqual(scoring.sha(original), before)

    def test_original_native_renderer_preserves_reasoning_prefix_verbatim(self):
        helper = scoring.original(REPO)
        helper.import_historical_scorer(REPO)
        from pier_llm_v23.prepare import _render_prompt
        class Tokenizer:
            chat_template = 'native'
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.called = (messages, tokenize, add_generation_prompt)
                return '<user>' + messages[0]['content'] + '</user><assistant><think>\n'
            def __call__(self, text, add_special_tokens):
                return {'input_ids': list(text.encode())}
        tokenizer = Tokenizer()
        text, ids = _render_prompt(tokenizer, 'canonical user content')
        self.assertTrue(text.endswith('<assistant><think>\n'))
        self.assertNotIn('</think>', text)
        self.assertEqual(tokenizer.called, ([{'role':'user', 'content':'canonical user content'}], False, True))
        self.assertEqual(ids, list(text.encode()))

    def test_original_full_continuation_sums_all_tokens_and_excludes_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch required: run this test in the scoring environment')
        import numpy as np
        helper = scoring.original(REPO)
        scorer = helper.import_historical_scorer(REPO)
        class Tokenizer:
            pad_token_id = 0
            all_special_ids = [0]
            token_ids = {'prompt': [1, 2], ' (A)': [3, 4], ' (B)': [5]}
            def __call__(self, text, add_special_tokens):
                self.assert_no_special = not add_special_tokens
                return {'input_ids': self.token_ids[text]}
            def decode(self, ids, **kwargs):
                return next(k for k, v in self.token_ids.items() if v == ids)
        class Model:
            def __call__(self, input_ids, attention_mask, use_cache):
                self.ids, self.mask = input_ids.clone(), attention_mask.clone()
                n, length = input_ids.shape
                logits = torch.arange(n*length*8, dtype=torch.float32).reshape(n,length,8)*.031
                logits += torch.arange(length, dtype=torch.float32)[None,:,None] * torch.arange(8)[None,None,:]*.07
                self.logits = logits
                return SimpleNamespace(logits=logits)
        prompt = {'prompt_id':'qid__C0__cyclic_r00', 'source_v2_prompt_id':'source', 'base_question_id':'qid',
                  'category':'subject', 'semantic_condition':'C0', 'condition':'clean', 'family':'clean',
                  'dose':0., 'track':None, 'permutation_id':0, 'option_count':2, 'semantic_to_visible':[0,1],
                  'visible_to_semantic':[0,1], 'option_labels':['A','B'], 'candidate_continuations':[' (A)',' (B)'],
                  'semantic_answer_index':1, 'canonical_content_sha256':'canonical'}
        rendered = {prompt['prompt_id']:{'rendered_prompt':'prompt','prompt_token_count':2,
                                       'canonical_content_sha256':'canonical','rendered_prompt_sha256':'rendered'}}
        model = Model()
        row = scorer.score_prompt_batch(model, Tokenizer(), [prompt], rendered, device='cpu')[0]
        logp = torch.log_softmax(model.logits.float(),dim=-1)
        expected = [float(logp[0,1,3])+float(logp[0,2,4]),float(logp[1,1,5])]
        self.assertEqual(row['candidate_sequence_log_likelihoods'], expected)
        self.assertEqual(row['candidate_token_counts'], [2,1])
        self.assertEqual(model.mask.tolist(), [[1,1,1,1],[1,1,1,0]])
        prob = np.exp(np.asarray(expected)-max(expected))
        np.testing.assert_array_equal(row['visible_label_probabilities'],prob/prob.sum())


if __name__ == '__main__':
    unittest.main()
