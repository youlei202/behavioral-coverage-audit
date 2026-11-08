"""Constructed numerical checks only; no experimental findings are generated."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from tv_core import (fit_tv_simplex, fit_candidate_subsets, choose_collective,
                     choose_nearest, evaluate_candidate, weighted_tv)


class TVCoreChecks(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(20261107)
        peers = np.stack([rng.dirichlet([1, 1, 1], size=12) for _ in range(4)], axis=-1)
        target = .35 * peers[:, :, 0] + .65 * peers[:, :, 2]
        return target, peers, np.ones(target.shape, dtype=bool), np.ones(12)

    def test_exact_fixed_mixture_and_sparse_constraints(self):
        y, x, mask, mass = self.fixture()
        result = fit_tv_simplex(y, x, mask, mass)
        self.assertLess(result.train_tv, 1e-8)
        self.assertLess(result.constraint_nnz, np.prod(result.constraint_shape))
        self.assertAlmostEqual(result.weights.sum(), 1, places=10)

    def test_duplicate_peers_do_not_require_unique_weights(self):
        y, x, mask, mass = self.fixture()
        duplicated = np.stack([y, y, x[:, :, 3]], axis=-1)
        self.assertLess(fit_tv_simplex(y, duplicated, mask, mass).train_tv, 1e-8)

    def test_ragged_vectors_have_equal_observation_mass(self):
        y = np.array([[1., 0., 0.], [1., 0., 0.]])
        x = np.array([[.6, .4, 0.], [.8, .1, .1]])[:, :, None]
        mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
        fit = fit_tv_simplex(y, x, mask, np.ones(2))
        self.assertAlmostEqual(fit.train_tv, .3, places=12)
        # Same observation weight remains .3 with two identical peer columns.
        fit2 = fit_tv_simplex(y, np.repeat(x, 2, axis=2), mask, np.ones(2))
        self.assertAlmostEqual(fit2.train_tv, .3, places=9)

    def test_clean_and_stress_design_mass(self):
        # One clean row plus three stress realizations: equal mass per dose.
        y = np.tile([1., 0.], (4, 1))
        p = np.array([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
        self.assertAlmostEqual(weighted_tv(y, p, np.ones_like(y, bool),
                                           np.array([1., 1/3, 1/3, 1/3])), .5)

    def test_budget_endpoints_and_training_dominance(self):
        y, x, mask, mass = self.fixture()
        previous = float("inf")
        for k in [1, 2, 3, 4]:
            fits = fit_candidate_subsets(y, x, mask, mass, k)
            collective = choose_collective(fits)
            nearest = choose_nearest(y, x, mask, mass, k)
            self.assertLessEqual(fits[collective].train_tv, fits[nearest].train_tv + 1e-8)
            self.assertLessEqual(fits[collective].train_tv, previous + 1e-8)
            previous = fits[collective].train_tv
            if k == 1:
                self.assertEqual(collective, nearest)
            if k == 4:
                self.assertEqual(len(fits), 1)

    def test_random_mean_is_mean_of_errors_not_error_of_mean(self):
        y = np.array([[.5, .5]])
        x = np.array([[[1., 0.], [0., 1.]]])
        mask, mass = np.ones_like(y, bool), np.ones(1)
        fits = fit_candidate_subsets(y, x, mask, mass, 1)
        expected = np.mean([evaluate_candidate(f, s, y, x, mask, mass)["test_tv"]
                            for s, f in fits.items()])
        self.assertAlmostEqual(expected, .5)
        self.assertAlmostEqual(weighted_tv(y, x.mean(axis=2), mask, mass), 0.)

    def test_invalid_probability_and_padding_fail(self):
        y = np.array([[.5, .5, 0.]])
        x = np.array([[[.5], [.4], [.1]]])
        with self.assertRaises(ValueError):
            fit_tv_simplex(y, x, np.array([[1, 1, 0]], bool), np.ones(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
