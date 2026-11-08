"""Reference implementation for the proposed XWVG experiment.

This is a tested numerical starting point, not a completed real-data experiment.
Arrays: target[n, a], peers[n, a, p], mask[n, a], observation_mass[n].
TV is a SUM over valid options and an average over weighted observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass
class TVFit:
    weights: np.ndarray
    train_tv: float
    solver_objective: float
    raw_min_weight: float
    raw_sum_error: float
    repair_l1: float
    iterations: int
    constraint_shape: tuple[int, int]
    constraint_nnz: int


def validate_inputs(target, peers, mask, observation_mass):
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(peers, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    mass = np.asarray(observation_mass, dtype=np.float64)
    if y.ndim != 2 or x.ndim != 3 or x.shape[:2] != y.shape:
        raise ValueError("Expected target[n,a] and peers[n,a,p]")
    if valid.shape != y.shape or mass.shape != (len(y),) or x.shape[2] < 1:
        raise ValueError("Invalid mask, mass, or peer count")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("Probabilities must be finite, including zero padding")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError("Observation masses must be finite and nonnegative")
    if np.any(valid.sum(axis=1) < 2):
        raise ValueError("Each observation needs at least two valid options")
    if np.any(y < -1e-12) or np.any(x < -1e-12):
        raise ValueError("Probabilities cannot be negative")
    if np.any(y > 1 + 1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("Probabilities cannot exceed one")
    if np.any(np.abs(y[~valid]) > 1e-12) or np.any(np.abs(x[~valid]) > 1e-12):
        raise ValueError("Padded coordinates must be zero")
    if not np.allclose(y.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("Target probabilities do not sum to one")
    if not np.allclose(x.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("Peer probabilities do not sum to one")
    return y, x, valid, mass


def per_observation_tv(target, prediction, mask):
    return 0.5 * np.sum(np.where(mask, np.abs(target - prediction), 0.0), axis=1)


def weighted_tv(target, prediction, mask, observation_mass):
    return float(np.average(per_observation_tv(target, prediction, mask), weights=observation_mass))


def top_option_agreement(target, prediction, mask, observation_mass):
    """Deterministic ties: first valid option in the fixed stored semantic order."""
    left = np.argmax(np.where(mask, target, -np.inf), axis=1)
    right = np.argmax(np.where(mask, prediction, -np.inf), axis=1)
    return float(np.average(left == right, weights=observation_mass))


def fit_tv_simplex(target, peers, mask, observation_mass) -> TVFit:
    """Minimize exact weighted vector TV with a sparse HiGHS linear program.

    Never form a dense identity matrix: full-vector observations can have tens
    of thousands of residual coordinates. No calibration or test data enter.
    """
    y, x, valid, mass = validate_inputs(target, peers, mask, observation_mass)
    positive = mass > 0
    y, x, valid, mass = y[positive], x[positive], valid[positive], mass[positive]
    peer_count = x.shape[-1]
    if peer_count == 1:
        error = weighted_tv(y, x[:, :, 0], valid, mass)
        return TVFit(np.ones(1), error, error, 1.0, 0.0, 0.0, 0, (0, 1), 0)

    # Every option of observation r has coefficient mass[r]/(2*sum(mass)).
    # Do NOT renormalize the flattened coefficients: it rescales reported TV.
    # Do NOT divide each observation's weights by its option count: that
    # would change observation weighting when option counts differ.
    flat_y = y[valid]
    flat_x = sparse.csr_matrix(x[valid])
    flat_mass = np.broadcast_to((0.5 * mass / mass.sum())[:, None], y.shape)[valid]
    coordinates = len(flat_y)
    slack = sparse.eye(coordinates, format="csr", dtype=np.float64)
    inequalities = sparse.vstack([
        sparse.hstack([flat_x, -slack], format="csr"),
        sparse.hstack([-flat_x, -slack], format="csr"),
    ], format="csr")
    equality = sparse.csr_matrix((np.ones(peer_count),
        (np.zeros(peer_count, dtype=np.int64), np.arange(peer_count))),
        shape=(1, peer_count + coordinates))
    objective = np.concatenate([np.zeros(peer_count), flat_mass])
    result = linprog(objective, A_ub=inequalities,
        b_ub=np.concatenate([flat_y, -flat_y]), A_eq=equality, b_eq=[1.0],
        bounds=(0.0, None), method="highs",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9})
    if not result.success or result.x is None:
        raise RuntimeError(f"TV LP failed: status={result.status}; {result.message}")
    raw = np.asarray(result.x[:peer_count], dtype=np.float64)
    if not np.all(np.isfinite(raw)) or not np.isfinite(result.fun):
        raise RuntimeError("Non-finite solver coefficients or objective")
    minimum = float(raw.min())
    sum_error = abs(float(raw.sum()) - 1.0)
    if minimum < -1e-8 or sum_error > 1e-8:
        raise RuntimeError(f"Infeasible simplex result: min={minimum}; sum_error={sum_error}")
    # Explicitly record any rounding-scale feasibility correction.
    weights = np.maximum(raw, 0.0)
    weights /= weights.sum()
    repair_l1 = float(np.abs(weights - raw).sum())
    if repair_l1 > 1e-7:
        raise RuntimeError("Numerical correction was larger than allowed")
    prediction = np.einsum("nap,p->na", x, weights)
    error = weighted_tv(y, prediction, valid, mass)
    if abs(error - float(result.fun)) > 1e-7:
        raise RuntimeError("Recomputed TV differs from the LP objective")
    return TVFit(weights, error, float(result.fun), minimum, sum_error,
                 repair_l1, int(result.nit), inequalities.shape, inequalities.nnz)


def fit_candidate_subsets(target, peers, mask, observation_mass, k):
    """Training-only reference enumerator. Peer columns must have fixed order."""
    if not 1 <= k <= peers.shape[-1]:
        raise ValueError("Invalid peer budget")
    return {subset: fit_tv_simplex(target, peers[:, :, subset], mask, observation_mass)
            for subset in combinations(range(peers.shape[-1]), k)}


def choose_collective(candidate_fits, tie_tolerance=1e-9):
    best = min(fit.train_tv for fit in candidate_fits.values())
    return min(subset for subset, fit in candidate_fits.items()
               if fit.train_tv <= best + tie_tolerance)


def choose_nearest(target, peers, mask, observation_mass, k, tie_tolerance=1e-9):
    if not 1 <= k <= peers.shape[-1]:
        raise ValueError("Invalid peer budget")
    errors = [weighted_tv(target, peers[:, :, j], mask, observation_mass)
              for j in range(peers.shape[-1])]
    available = set(range(len(errors)))
    chosen = []
    for _ in range(k):
        best = min(errors[j] for j in available)
        selected = min(j for j in available if errors[j] <= best + tie_tolerance)
        chosen.append(selected)
        available.remove(selected)
    return tuple(sorted(chosen))


def evaluate_candidate(fit, subset, target, peers, mask, observation_mass):
    prediction = np.einsum("nap,p->na", peers[:, :, subset], fit.weights)
    return {
        "test_tv": weighted_tv(target, prediction, mask, observation_mass),
        "top_option_agreement": top_option_agreement(target, prediction, mask, observation_mass),
        "per_observation_tv": per_observation_tv(target, prediction, mask),
    }
