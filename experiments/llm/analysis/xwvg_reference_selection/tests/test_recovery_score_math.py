"""CPU exact-equivalence checks; these do not qualify GPU rollout by themselves."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'experiments/llm/analysis/v23/src'))
from pier_llm_v23.inference import score_prompt_batch as original_score

path = Path(__file__).resolve().parents[1] / 'recovery/score_math.py'
spec = importlib.util.spec_from_file_location('score_math', path)
math_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_module)


class Tokenizer:
    pad_token_id = 0
    all_special_ids = [0]
    candidates = {' (A)': [1, 4, 2], ' (B)': [3, 5], ' (C)': [6], ' (D)': [8, 2, 11, 7]}
    prefixes = {'short': [12, 13], 'medium': [12, 13, 14, 15, 16],
                'long': [12, 13, 14, 15, 16, 17, 18, 19, 20]}

    def __call__(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError('Historical scorer never adds special tokens here')
        return {'input_ids': self.candidates[text] if text.startswith(' (') else self.prefixes[text]}

    def decode(self, ids, **kwargs):
        for text, values in self.candidates.items():
            if ids == values:
                return text
        raise AssertionError(ids)


class Model:
    """Full padded forward with deterministic nontrivial sequence/position logits."""
    def __init__(self, dtype, scale=1):
        self.dtype, self.scale, self.calls = dtype, scale, []

    def __call__(self, **kwargs):
        self.calls.append({k: v.clone() if hasattr(v, 'clone') else v for k, v in kwargs.items()})
        ids = kwargs['input_ids']
        batch, length = ids.shape
        vocab = 47
        row = torch.arange(batch, dtype=torch.float32)[:, None, None]
        position = torch.arange(length, dtype=torch.float32)[None, :, None]
        option = torch.arange(vocab, dtype=torch.float32)[None, None, :]
        logits = torch.sin(row * .179 + position * .367 + option * .133) * self.scale
        logits += torch.cos(ids.float()[:, :, None] * .719 + option * .09) * self.scale
        return SimpleNamespace(logits=logits.to(self.dtype))


def inputs():
    tokenizer = Tokenizer()
    prompts, rendered = [], {}
    for index, (text, count, rotation) in enumerate([('long', 4, 1), ('short', 2, 0), ('medium', 3, 2)]):
        pid=f'question{index}'
        labels=list('ABCD'[:count])
        row=dict(prompt_id=pid, source_v2_prompt_id=pid+'__source', base_question_id=pid,
            category='subject', semantic_condition='I0', condition='irrelevant_context',
            family='irrelevant_context', dose=512, track=0, permutation_id=rotation,
            option_count=count, semantic_to_visible=[(i+rotation)%count for i in range(count)],
            visible_to_semantic=[(i-rotation)%count for i in range(count)],
            option_labels=labels, candidate_continuations=[f' ({label})' for label in labels],
            semantic_answer_index=1, canonical_content_sha256='canonical-'+pid)
        prompts.append(row)
        rendered[pid]=dict(rendered_prompt=text, prompt_token_count=len(tokenizer.prefixes[text]),
                           canonical_content_sha256=row['canonical_content_sha256'], rendered_prompt_sha256='rendered-'+pid)
    return tokenizer,prompts,rendered


@unittest.skipIf(torch is None, 'CPU Torch required; run in the scoring environment')
class ScoringMathChecks(unittest.TestCase):
    def setUp(self):
        # Small mocks should not use many host threads while GPU pilot is active.
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_complete_output_rows_bitwise_equal_fp32_and_bfloat16(self):
        tokenizer,prompts,rendered=inputs()
        for dtype in (torch.float32,torch.bfloat16):
            for scale in (1.0,700.0):
                with self.subTest(dtype=str(dtype), scale=scale):
                    left_model,right_model=Model(dtype,scale),Model(dtype,scale)
                    left=original_score(left_model,tokenizer,prompts,rendered,device='cpu')
                    right=math_module.score_prompt_batch(right_model,tokenizer,prompts,rendered,device='cpu')
                    self.assertEqual(left,right)
                    self.assertEqual(len(left_model.calls),1)
                    self.assertEqual(len(right_model.calls),1)
                    self.assertEqual(set(left_model.calls[0]),{'input_ids','attention_mask','use_cache'})
                    self.assertEqual(set(right_model.calls[0]),set(left_model.calls[0]))
                    for key in ('input_ids','attention_mask'):
                        self.assertTrue(torch.equal(left_model.calls[0][key],right_model.calls[0][key]))
                    self.assertFalse(right_model.calls[0]['use_cache'])

    def test_softmax_is_fp32_only_on_required_prediction_rows(self):
        tokenizer,prompts,rendered=inputs()
        observed=[]
        actual=torch.log_softmax
        def check(tensor,*args,**kwargs):
            observed.append((tuple(tensor.shape),tensor.dtype))
            return actual(tensor,*args,**kwargs)
        with patch.object(torch,'log_softmax',side_effect=check):
            math_module.score_prompt_batch(Model(torch.bfloat16),tokenizer,prompts,rendered,device='cpu')
        terms=sum(sum(len(tokenizer.candidates[s]) for s in p['candidate_continuations']) for p in prompts)
        self.assertEqual(observed,[((terms,47),torch.float32)])
        full_rows=sum(p['option_count'] for p in prompts)*(max(len(tokenizer.prefixes[r['rendered_prompt']]) for r in rendered.values())+4)
        self.assertLess(terms,full_rows)

    def test_validation_failures_match_baseline_before_forward(self):
        tokenizer,prompts,rendered=inputs()
        for issue in ('prompt_token_count','canonical_content_sha256'):
            bad=deepcopy(rendered)
            bad[prompts[0]['prompt_id']][issue]=999 if issue=='prompt_token_count' else 'different'
            for scorer in (original_score,math_module.score_prompt_batch):
                model=Model(torch.bfloat16)
                with self.assertRaises(ValueError):
                    scorer(model,tokenizer,prompts,bad,device='cpu')
                self.assertEqual(model.calls,[])

    def test_reordered_ragged_batches_preserve_exact_single_token_and_multitoken_sums(self):
        tokenizer,prompts,rendered=inputs()
        for batch in ([prompts[2]],list(reversed(prompts)),prompts[:1]):
            left=original_score(Model(torch.bfloat16),tokenizer,batch,rendered,device='cpu')
            right=math_module.score_prompt_batch(Model(torch.bfloat16),tokenizer,batch,rendered,device='cpu')
            self.assertEqual(left,right)
            for row in right:
                self.assertEqual(len(row['candidate_sequence_log_likelihoods']),row['option_count'])
                self.assertEqual(row['candidate_token_counts'],[len(tokenizer.candidates[s]) for s in row['candidate_continuations']])


if __name__=='__main__':
    unittest.main()
