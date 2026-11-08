"""Sparse, fixed-convex total-variation fitting on complete option vectors.

Arrays are target[n,a], peers[n,a,p], valid mask[n,a], and observation mass[n].
The entry point must set BLAS/OpenMP environment limits before importing NumPy.
HiGHS is independently restricted to one solver thread in every LP call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from types import SimpleNamespace
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog


def verify_highs_runtime():
    """Observe effective options on a real linprog solver at solve time.

    Call once in the serial pilot before creating worker processes. SciPy does
    not expose this solver object publicly; this explicit compatibility probe
    uses its bundled binding and restores the wrapper immediately afterward.
    Unsupported SciPy internals fail closed rather than reporting requested
    options as verified runtime settings. No experiment data enter this probe.
    """
    try:
        from scipy.optimize._highspy import _highs_wrapper as wrapper
    except ImportError as exc:
        raise RuntimeError("Cannot verify effective HiGHS threads with this SciPy build") from exc
    original = wrapper._h
    observed = []

    class ObservedHighs(original._Highs):
        def run(self):
            threads_status, threads = self.getOptionValue("threads")
            parallel_status, parallel = self.getOptionValue("parallel")
            if threads_status != original.HighsStatus.kOk or parallel_status != original.HighsStatus.kOk:
                raise RuntimeError("HiGHS did not expose runtime thread settings")
            observed.append({"highs_version": self.version(), "threads": int(threads),
                             "parallel": str(parallel), "observed_at": "HiGHS.run"})
            return super().run()

    # Replace only the wrapper's namespace, not the extension's global class.
    proxy = SimpleNamespace(**{name: getattr(original, name) for name in dir(original)})
    proxy._Highs = ObservedHighs
    wrapper._h = proxy
    try:
        fit_tv_simplex(np.array([[.25, .75]]),
            np.array([[[1., 0.], [0., 1.]]]), np.ones((1, 2), bool), np.ones(1))
    finally:
        wrapper._h = original
    if len(observed) != 1 or observed[0]["threads"] != 1 or observed[0]["parallel"] != "off":
        raise RuntimeError(f"Unexpected effective HiGHS settings: {observed}")
    return {"verified": True, "scope": "serial constructed-data linprog runtime probe", **observed[0]}


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
    status: int = 0
    message: str = "direct singleton score"
    solve_seconds: float = 0.0
    raw_train_tv: float = 0.0
    repair_prediction_max_abs: float = 0.0
    repair_training_tv_change: float = 0.0
    solver_threads: int = 1
    raw_weights: list[float] | None = None

    def to_record(self):
        result = asdict(self)
        result["weights"] = self.weights.tolist()
        result["constraint_shape"] = list(self.constraint_shape)
        return result

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        values["weights"] = np.asarray(values["weights"], dtype=float)
        values["constraint_shape"] = tuple(values["constraint_shape"])
        fit = cls(**values)
        if (fit.weights.ndim != 1 or not len(fit.weights) or
            not np.all(np.isfinite(fit.weights)) or fit.weights.min() < -1e-12 or
            abs(float(fit.weights.sum()) - 1.0) > 1e-10 or fit.status != 0):
            raise ValueError("Invalid or unfinished cached fit")
        diagnostics = (fit.train_tv, fit.solver_objective, fit.raw_min_weight,
                       fit.raw_sum_error, fit.repair_l1, fit.solve_seconds)
        if not all(np.isfinite(v) for v in diagnostics):
            raise ValueError("Nonfinite cached fit diagnostics")
        if fit.train_tv < -1e-10 or fit.train_tv > 1 + 1e-10:
            raise ValueError("Cached TV outside probability-distance bounds")
        if fit.raw_weights is not None:
            raw = np.asarray(fit.raw_weights, dtype=float)
            if raw.shape != fit.weights.shape or not np.all(np.isfinite(raw)):
                raise ValueError("Invalid cached raw weights")
        return fit


def validate_inputs(target, peers, mask, observation_mass):
    y = np.asarray(target, dtype=np.float64)
    x = np.asarray(peers, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    mass = np.asarray(observation_mass, dtype=np.float64)
    if y.ndim != 2 or x.ndim != 3 or x.shape[:2] != y.shape:
        raise ValueError("Expected target[n,a] and peers[n,a,p]")
    if valid.shape != y.shape or mass.shape != (len(y),) or x.shape[-1] < 1:
        raise ValueError("Invalid option mask, observation masses, or peer count")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("Probabilities must be finite, including padding")
    if not np.all(np.isfinite(mass)) or np.any(mass < 0) or mass.sum() <= 0:
        raise ValueError("Masses must be finite, nonnegative, and have positive sum")
    if np.any(valid.sum(axis=1) < 2):
        raise ValueError("Each observation requires at least two valid options")
    if np.any(y < -1e-12) or np.any(x < -1e-12):
        raise ValueError("Negative probability")
    if np.any(y > 1 + 1e-12) or np.any(x > 1 + 1e-12):
        raise ValueError("Probability exceeds one")
    if np.any(np.abs(y[~valid]) > 1e-12) or np.any(np.abs(x[~valid]) > 1e-12):
        raise ValueError("Invalid option coordinates must be zero padded")
    if not np.allclose(np.where(valid, y, 0).sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("Target probabilities do not sum to one")
    if not np.allclose(np.where(valid[:, :, None], x, 0).sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("Peer probabilities do not sum to one")
    return y, x, valid, mass


def per_observation_tv(target, prediction, mask):
    return 0.5 * np.sum(np.where(mask, np.abs(target - prediction), 0.0), axis=1)


def weighted_tv(target, prediction, mask, observation_mass):
    return float(np.average(per_observation_tv(target, prediction, mask), weights=observation_mass))


def fit_tv_simplex(target, peers, mask, observation_mass, solver_config=None) -> TVFit:
    """Minimize weighted TV with sparse residual constraints; no held-out inputs."""
    started = perf_counter()
    cfg = {} if solver_config is None else solver_config
    y, x, valid, mass = validate_inputs(target, peers, mask, observation_mass)
    positive = mass > 0
    y, x, valid, mass = y[positive], x[positive], valid[positive], mass[positive]
    count = x.shape[-1]
    if count == 1:
        error = weighted_tv(y, x[:, :, 0], valid, mass)
        return TVFit(np.ones(1), error, error, 1.0, 0.0, 0.0, 0, (0, 1), 0,
                     solve_seconds=perf_counter() - started, raw_train_tv=error,
                     raw_weights=[1.0])

    flat_y = y[valid]
    flat_x = sparse.csr_matrix(x[valid])
    # Every valid coordinate receives its observation's coefficient. TV sums
    # over options; neither option-count division nor flattening normalization.
    coefficients = np.broadcast_to((0.5 * mass / mass.sum())[:, None], y.shape)[valid]
    coordinates = len(flat_y)
    identity = sparse.eye(coordinates, format="csr", dtype=np.float64)
    inequalities = sparse.vstack([
        sparse.hstack([flat_x, -identity], format="csr"),
        sparse.hstack([-flat_x, -identity], format="csr"),
    ], format="csr")
    equality = sparse.csr_matrix((np.ones(count),
        (np.zeros(count, dtype=np.int64), np.arange(count))),
        shape=(1, count + coordinates))
    options = {
        "primal_feasibility_tolerance": cfg.get("primal_feasibility_tolerance", 1e-9),
        "dual_feasibility_tolerance": cfg.get("dual_feasibility_tolerance", 1e-9),
        # SciPy forwards these to HiGHS although not part of its public wrapper
        # option list. Unlike BLAS limits, these govern HiGHS itself.
        "threads": 1,
        "parallel": False,
    }
    if cfg.get("method", "highs") != "highs":
        raise ValueError("The locked solver method is highs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning,
            message=r"Unrecognized options detected:.*(?:threads|parallel).*")
        result = linprog(np.concatenate([np.zeros(count), coefficients]),
            A_ub=inequalities, b_ub=np.concatenate([flat_y, -flat_y]),
            A_eq=equality, b_eq=[1.0], bounds=(0.0, None),
            method="highs", options=options)
    if not result.success or result.x is None:
        raise RuntimeError(f"TV LP failed: status={result.status}; {result.message}")
    raw = np.asarray(result.x[:count], dtype=float)
    if not np.all(np.isfinite(raw)) or not np.isfinite(result.fun):
        raise RuntimeError("Non-finite solver coefficients or objective")
    minimum = float(raw.min())
    sum_error = abs(float(raw.sum()) - 1)
    acceptance = cfg.get("simplex_acceptance_tolerance", 1e-8)
    if minimum < -acceptance or sum_error > acceptance:
        raise RuntimeError(f"Infeasible simplex: min={minimum}; sum_error={sum_error}")
    raw_prediction = np.einsum("nap,p->na", x, raw)
    raw_error = weighted_tv(y, raw_prediction, valid, mass)
    tolerance = cfg.get("objective_recompute_tolerance", 1e-7)
    if abs(raw_error - float(result.fun)) > tolerance:
        raise RuntimeError("Unrepaired prediction TV differs from LP objective")
    weights = np.maximum(raw, 0)
    weights /= weights.sum()
    repair = float(np.abs(weights - raw).sum())
    if repair > cfg.get("max_repair_l1", 1e-7):
        raise RuntimeError("Simplex repair exceeds the locked numerical tolerance")
    prediction = np.einsum("nap,p->na", x, weights)
    error = weighted_tv(y, prediction, valid, mass)
    if abs(error - float(result.fun)) > tolerance:
        raise RuntimeError("Recomputed TV differs from LP objective")
    return TVFit(weights, error, float(result.fun), minimum, sum_error, repair,
        int(result.nit), inequalities.shape, int(inequalities.nnz),
        status=int(result.status), message=str(result.message),
        solve_seconds=perf_counter() - started, raw_train_tv=raw_error,
        repair_prediction_max_abs=float(np.max(np.abs(prediction - raw_prediction))),
        repair_training_tv_change=error - raw_error, raw_weights=raw.tolist())
