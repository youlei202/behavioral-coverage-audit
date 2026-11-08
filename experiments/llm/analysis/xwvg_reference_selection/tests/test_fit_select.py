"""Constructed objective, selection, masking and leakage checks, not data results."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog as real_linprog

from bca_reference_selection.fit import TVFit, fit_tv_simplex, weighted_tv, verify_highs_runtime
from bca_reference_selection.select import (candidate_subsets, fit_candidate_subsets,
    choose_collective, choose_nearest, select_policies, check_training_invariants)
from bca_reference_selection.evaluate import evaluate_candidate, random_expected


class FitSelectionChecks(unittest.TestCase):
    def fixture(self, peers=7):
        rng = np.random.default_rng(20261107)
        x = np.stack([rng.dirichlet([1, 1, 1], size=12) for _ in range(peers)], axis=-1)
        y = .35 * x[:, :, 0] + .65 * x[:, :, 2]
        return y, x, np.ones(y.shape, dtype=bool), np.ones(len(y))

    def test_exact_fixed_mixture_and_sparse_constraints(self):
        y, x, valid, mass = self.fixture()
        with patch("bca_reference_selection.fit.linprog", wraps=real_linprog) as lp:
            fit = fit_tv_simplex(y, x, valid, mass)
        self.assertLess(fit.train_tv, 1e-8)
        self.assertLess(fit.constraint_nnz, np.prod(fit.constraint_shape))
        self.assertAlmostEqual(fit.weights.sum(), 1, places=10)
        self.assertEqual(lp.call_args.kwargs["options"]["threads"], 1)
        self.assertFalse(lp.call_args.kwargs["options"]["parallel"])
        self.assertEqual(lp.call_args.kwargs["A_ub"].format, "csr")
        restored = TVFit.from_record(fit.to_record())
        np.testing.assert_array_equal(restored.weights, fit.weights)
        corrupt = fit.to_record()
        corrupt["weights"] = [float("nan")] * len(fit.weights)
        with self.assertRaises(ValueError):
            TVFit.from_record(corrupt)

    def test_duplicates_boundary_and_singleton(self):
        y, x, valid, mass = self.fixture()
        duplicate = np.stack([y, y, x[:, :, 3]], axis=-1)
        self.assertLess(fit_tv_simplex(y, duplicate, valid, mass).train_tv, 1e-8)
        boundary = fit_tv_simplex(y, np.stack([y, x[:, :, 3]], axis=-1), valid, mass)
        self.assertLess(boundary.train_tv, 1e-8)
        with patch("bca_reference_selection.fit.linprog", side_effect=AssertionError("singleton LP")):
            self.assertLess(fit_tv_simplex(y, y[:, :, None], valid, mass).train_tv, 1e-15)

    def test_ragged_vectors_and_zero_mass(self):
        y = np.array([[1., 0., 0.], [1., 0., 0.]])
        x = np.array([[.6, .4, 0.], [.8, .1, .1]])[:, :, None]
        valid = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
        for peers in (x, np.repeat(x, 2, axis=2)):
            fit = fit_tv_simplex(y, peers, valid, np.ones(2))
            self.assertAlmostEqual(fit.train_tv, .3, places=9)
            fit = fit_tv_simplex(y, peers, valid, np.array([0., 1.]))
            self.assertAlmostEqual(fit.train_tv, .2, places=9)

    def test_clean_stress_mass_and_question_aggregation(self):
        y = np.tile([1., 0.], (8, 1))
        p = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]] * 2)
        mass = np.tile([.5, 1/6, 1/6, 1/6], 2)
        valid = np.ones_like(y, bool)
        fit = fit_tv_simplex(y, p[:, :, None], valid, mass)
        self.assertAlmostEqual(fit.train_tv, .5)
        out = evaluate_candidate(fit, (0,), y, p[:, :, None], valid, mass,
                                 ["q1"] * 4 + ["q2"] * 4)
        np.testing.assert_allclose(out["question_tv"], [.5, .5])
        np.testing.assert_allclose(out["question_top_option_agreement"], [.5, .5])
        np.testing.assert_allclose(out["question_mass"], [1., 1.])

    def test_complete_enumeration_and_training_invariants(self):
        y, x, valid, mass = self.fixture()
        subsets = candidate_subsets()
        self.assertEqual({k: len(v) for k, v in subsets.items()}, {1: 7, 2: 21, 3: 35, 7: 1})
        fits = {}
        for k in subsets:
            fits.update(fit_candidate_subsets(y, x, valid, mass, k))
        self.assertEqual(len(fits), 64)
        selections = select_policies(fits)
        diagnostics = check_training_invariants(fits, selections)
        self.assertTrue(diagnostics["passed"])
        self.assertLessEqual(diagnostics["collective_minus_random_training_max"], 1e-7)
        self.assertEqual(selections[1]["collective"], selections[1]["nearest_k"])
        self.assertEqual(selections[7]["collective"], tuple(range(7)))
        with self.assertRaises(ValueError):
            select_policies({s: v for s, v in fits.items() if s != (0, 1)})

    def test_training_only_selection_unchanged_when_evaluation_changes(self):
        y, x, valid, mass = self.fixture()
        fits = {s: fit_tv_simplex(y, x[:, :, s], valid, mass)
                for rows in candidate_subsets().values() for s in rows}
        before = select_policies(fits)
        saved_weights = {s: f.weights.copy() for s, f in fits.items()}
        heldout_target = y[:, ::-1].copy()
        for s, fit in fits.items():
            evaluate_candidate(fit, s, heldout_target, x, valid, mass)
        self.assertEqual(before, select_policies(fits))
        for s, fit in fits.items():
            np.testing.assert_array_equal(saved_weights[s], fit.weights)

    def test_random_mean_of_errors_not_error_of_mean_prediction(self):
        y = np.array([[.5, .5]])
        x = np.array([[[1., 0.], [0., 1.]]])
        valid, mass = np.ones_like(y, bool), np.ones(1)
        fits = fit_candidate_subsets(y, x, valid, mass, 1)
        evaluations = {s: evaluate_candidate(f, s, y, x, valid, mass, ["q"])
                       for s, f in fits.items()}
        expected = random_expected(evaluations)
        self.assertAlmostEqual(expected["evaluation_tv"], .5)
        self.assertAlmostEqual(expected["top_option_agreement"], .5)
        self.assertAlmostEqual(expected["question_tv"][0], .5)
        self.assertAlmostEqual(weighted_tv(y, x.mean(axis=2), valid, mass), 0.)

    def test_stable_training_ties(self):
        y = np.array([[.5, .5]])
        x = np.repeat(y[:, :, None], 7, axis=2)
        fits = {s: fit_tv_simplex(y, x[:, :, s], np.ones_like(y, bool), np.ones(1))
                for rows in candidate_subsets().values() for s in rows}
        choices = select_policies(fits)
        for k in choices:
            self.assertEqual(choices[k]["collective"], tuple(range(k)))
            self.assertEqual(choices[k]["nearest_k"], tuple(range(k)))

    def test_invalid_probability_and_padding_rejected(self):
        y = np.array([[.5, .5, 0.]])
        valid = np.array([[1, 1, 0]], bool)
        for vector in ([.5, .4, .1], [.5, .4, 0], [np.nan, .5, 0], [-.1, 1.1, 0]):
            with self.assertRaises(ValueError):
                fit_tv_simplex(y, np.asarray(vector)[None, :, None], valid, np.ones(1))

    def test_runtime_highs_thread_limit(self):
        record = verify_highs_runtime()
        self.assertTrue(record["verified"])
        self.assertEqual(record["threads"], 1)
        self.assertEqual(record["parallel"], "off")


if __name__ == "__main__":
    unittest.main()
