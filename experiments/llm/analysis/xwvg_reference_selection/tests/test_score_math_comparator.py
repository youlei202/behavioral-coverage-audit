"""The GPU comparator normalizes nullable track metadata, never score values."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'recovery/validate_score_math.py'
spec=importlib.util.spec_from_file_location('validate_score_math',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparatorChecks(unittest.TestCase):
    def test_nullable_track_restoration_preserves_exact_row_identity(self):
        left=[{'prompt_id':'p','track':1.0,'candidate_sequence_log_likelihoods':[-1.25,-2.5]}]
        right=[{'prompt_id':'p','track':1,'candidate_sequence_log_likelihoods':[-1.25,-2.5]}]
        result=module.compare_rows(left,right)
        self.assertTrue(result['exact'])
        self.assertEqual(result['baseline_sha256'],result['candidate_sha256'])

    def test_one_ulp_score_difference_is_rejected(self):
        left=[{'prompt_id':'p','track':1.0,'candidate_sequence_log_likelihoods':[-1.25,-2.5]}]
        right=[{'prompt_id':'p','track':1,'candidate_sequence_log_likelihoods':[-1.2500000000000002,-2.5]}]
        result=module.compare_rows(left,right)
        self.assertFalse(result['exact'])
        self.assertEqual(result['errors'][0]['field'],'candidate_sequence_log_likelihoods')
        self.assertGreater(result['errors'][0]['max_absolute_difference'],0)

    def test_different_track_or_null_remains_a_real_mismatch(self):
        self.assertFalse(module.compare_rows([{'track':1.0}],[{'track':2}])['exact'])
        self.assertFalse(module.compare_rows([{'track':None}],[{'track':0}])['exact'])


if __name__=='__main__':unittest.main()
