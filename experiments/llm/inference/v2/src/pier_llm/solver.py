from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


def _solve(problem: cp.Problem, solver: str) -> None:
    options: dict[str, Any] = {}
    if solver == "CLARABEL":
        options = {
            "tol_gap_abs": 1e-12,
            "tol_gap_rel": 1e-12,
            "tol_feas": 1e-12,
            "tol_infeas_abs": 1e-12,
            "tol_infeas_rel": 1e-12,
            "max_iter": 1_000,
        }
    elif solver == "OSQP":
        options = {
            "eps_abs": 1e-12,
            "eps_rel": 1e-12,
            "max_iter": 1_000_000,
            "polishing": True,
        }
    problem.solve(solver=solver, verbose=False, **options)


@dataclass(frozen=True)
class ProjectionResult:
    weights: np.ndarray
    stage1_objective: float
    stage2_objective: float
    tolerance: float
    solver_stage1: str
    solver_stage2: str
    status_stage1: str
    status_stage2: str
    minimum_weight: float
    weight_sum_error: float
    stage1_iterations: int | None
    stage2_iterations: int | None
    projection_preservation_tolerance: float
    maximum_projection_change: float

    def diagnostics(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("weights")
        return result


def _as_design(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if design.ndim == 3:
        design = design.reshape(-1, design.shape[-1])
    if target.ndim > 1:
        target = target.reshape(-1)
    if design.ndim != 2 or target.ndim != 1:
        raise ValueError("Expected a two-dimensional design and one-dimensional target")
    if design.shape[0] != target.shape[0] or design.shape[0] == 0:
        raise ValueError(f"Incompatible design {design.shape} and target {target.shape}")
    if design.shape[1] == 0:
        raise ValueError("At least one peer is required")
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("Projection input contains non-finite values")
    return np.ascontiguousarray(design), np.ascontiguousarray(target)


def squared_loss(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    design, target = _as_design(x, y)
    residual = design @ np.asarray(weights, dtype=np.float64) - target
    return float(np.mean(np.square(residual)))


def _solver_iterations(problem: cp.Problem) -> int | None:
    value = getattr(problem.solver_stats, "num_iters", None)
    return int(value) if value is not None else None


def _enumerated_active_set_qp(
    design: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float, int]:
    """Solve the small simplex QP exactly by enumerating all active faces."""
    peers = design.shape[1]
    if peers > 15:
        raise ValueError("Enumerated simplex QP is intended for at most 15 peers")
    best_weights: np.ndarray | None = None
    best_loss = math.inf
    evaluated = 0
    for mask in range(1, 1 << peers):
        active = [index for index in range(peers) if mask & (1 << index)]
        subdesign = design[:, active]
        gram = subdesign.T @ subdesign
        right = subdesign.T @ target
        size = len(active)
        kkt = np.block(
            [
                [gram, np.ones((size, 1))],
                [np.ones((1, size)), np.zeros((1, 1))],
            ]
        )
        solution = np.linalg.lstsq(kkt, np.concatenate([right, [1.0]]), rcond=None)[0][
            :size
        ]
        evaluated += 1
        if solution.min() < -1e-10 or abs(float(solution.sum()) - 1.0) > 1e-8:
            continue
        candidate = np.zeros(peers, dtype=np.float64)
        candidate[active] = np.maximum(solution, 0.0)
        candidate /= candidate.sum()
        loss = squared_loss(design, target, candidate)
        if loss < best_loss:
            best_loss = loss
            best_weights = candidate
    if best_weights is None:
        raise RuntimeError("Enumerated active-set QP found no feasible simplex face")
    return best_weights, best_loss, evaluated


def _minimum_norm_exact_representative(
    design: np.ndarray,
    reference_weights: np.ndarray,
    *,
    projection_tolerance: float,
) -> np.ndarray:
    """Find the minimum-norm simplex weights with the reference projection exactly."""
    peers = design.shape[1]
    if peers > 15:
        raise ValueError("Enumerated minimum-norm refinement is intended for at most 15 peers")
    reference = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
    if len(reference) != peers:
        raise ValueError("Reference weight count does not match the design")
    prediction = design @ reference
    _left, singular_values, right = np.linalg.svd(design, full_matrices=False)
    if singular_values.size and singular_values[0] > 0:
        rank_tolerance = (
            np.finfo(np.float64).eps * max(design.shape) * float(singular_values[0])
        )
        rank = int(np.sum(singular_values > rank_tolerance))
    else:
        rank = 0
    row_basis = right[:rank]
    constraints = np.vstack([row_basis, np.ones((1, peers), dtype=np.float64)])
    right_hand_side = constraints @ reference
    acceptance = max(1e-12, 0.1 * float(projection_tolerance))

    best = reference.copy()
    best_norm = float(best @ best)
    for mask in range(1, 1 << peers):
        active = [index for index in range(peers) if mask & (1 << index)]
        solution = np.linalg.lstsq(
            constraints[:, active], right_hand_side, rcond=None
        )[0]
        if solution.min() < -1e-10:
            continue
        candidate = np.zeros(peers, dtype=np.float64)
        candidate[active] = np.maximum(solution, 0.0)
        total = float(candidate.sum())
        if total <= 0:
            continue
        candidate /= total
        projection_change = float(np.max(np.abs(design @ candidate - prediction)))
        if projection_change > acceptance:
            continue
        norm = float(candidate @ candidate)
        if norm < best_norm:
            best = candidate
            best_norm = norm
    return best


def fit_simplex_projection(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tau_absolute: float = 1e-12,
    tau_relative: float = 1e-10,
    solver: str = "CLARABEL",
) -> ProjectionResult:
    """Fit the preregistered two-stage deterministic convex projection."""
    design, target = _as_design(x, y)
    count, peers = design.shape
    stage1_solver = "enumerated_active_set_qp"
    stage1_weights, optimum, stage1_iterations = _enumerated_active_set_qp(design, target)
    tolerance = max(float(tau_absolute), float(tau_relative) * max(1.0, optimum))
    stage1_prediction = design @ stage1_weights
    preservation_tolerance = 1e-10 * max(1.0, float(np.max(np.abs(stage1_prediction))))

    representative = cp.Variable(peers)
    representative_loss = cp.sum_squares(design @ representative - target) / count
    stage2 = cp.Problem(
        cp.Minimize(cp.sum_squares(representative)),
        [
            representative >= 0,
            cp.sum(representative) == 1,
            representative_loss <= optimum + tolerance,
            cp.norm_inf(design @ representative - stage1_prediction)
            <= preservation_tolerance,
        ],
    )
    allowed = optimum + tolerance
    refinement_reason: str | None = None
    fitted: np.ndarray | None = None
    solver_error: str | None = None
    try:
        _solve(stage2, solver)
    except cp.error.SolverError as exc:
        solver_error = f"{type(exc).__name__}: {exc}"
    if solver_error is not None:
        refinement_reason = solver_error
    elif stage2.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or representative.value is None:
        refinement_reason = f"status={stage2.status}"
    else:
        raw = np.asarray(representative.value, dtype=np.float64).reshape(-1)
        raw[np.abs(raw) < 1e-12] = 0.0
        if raw.min() < -1e-8 or float(raw.sum()) <= 0:
            refinement_reason = f"simplex_raw_min={raw.min()}"
        else:
            fitted = np.maximum(raw, 0.0)
            fitted /= fitted.sum()
            fitted_loss = squared_loss(design, target, fitted)
            maximum_projection_change = float(
                np.max(np.abs(design @ fitted - stage1_prediction))
            )
            if fitted_loss > allowed + max(1e-11, 10 * tolerance):
                refinement_reason = f"loss={fitted_loss}"
            elif maximum_projection_change > 10 * preservation_tolerance + 1e-11:
                refinement_reason = f"projection_change={maximum_projection_change}"

    stage2_solver = solver
    stage2_status = str(stage2.status) if solver_error is None else "solver_error"
    if refinement_reason is not None:
        fitted = _minimum_norm_exact_representative(
            design,
            stage1_weights,
            projection_tolerance=preservation_tolerance,
        )
        stage2_solver = f"{solver}+enumerated_exact_refinement"
        stage2_status = f"{stage2_status}:refined"
        print(
            "[solver] applied exact stage-2 refinement "
            f"for design={design.shape}; reason={refinement_reason}",
            flush=True,
        )

    if fitted is None:
        raise RuntimeError("Stage-2 simplex projection produced no representative")
    fitted_loss = squared_loss(design, target, fitted)
    maximum_projection_change = float(np.max(np.abs(design @ fitted - stage1_prediction)))
    if fitted_loss > allowed + max(1e-11, 10 * tolerance):
        raise RuntimeError(
            f"Stage-2 representative exceeds loss tolerance: {fitted_loss} > {allowed}"
        )
    sum_error = abs(float(fitted.sum()) - 1.0)
    if fitted.min() < -1e-9 or sum_error > 1e-8:
        raise RuntimeError("Simplex feasibility check failed")
    if maximum_projection_change > 10 * preservation_tolerance + 1e-11:
        raise RuntimeError(
            "Stage-2 representative did not preserve the stage-1 projected response: "
            f"{maximum_projection_change}"
        )
    return ProjectionResult(
        weights=fitted,
        stage1_objective=optimum,
        stage2_objective=fitted_loss,
        tolerance=tolerance,
        solver_stage1=stage1_solver,
        solver_stage2=stage2_solver,
        status_stage1="optimal",
        status_stage2=stage2_status,
        minimum_weight=float(fitted.min()),
        weight_sum_error=sum_error,
        stage1_iterations=stage1_iterations,
        stage2_iterations=_solver_iterations(stage2),
        projection_preservation_tolerance=preservation_tolerance,
        maximum_projection_change=maximum_projection_change,
    )


def fit_simplex_slsqp(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Independent numerical implementation used only for solver validation."""
    design, target = _as_design(x, y)
    peers = design.shape[1]

    def objective(weights: np.ndarray) -> float:
        return squared_loss(design, target, weights)

    result = minimize(
        objective,
        np.full(peers, 1.0 / peers),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * peers,
        constraints=[{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}],
        options={"ftol": 1e-13, "maxiter": 10_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Independent SLSQP projection failed: {result.message}")
    fitted = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    fitted /= fitted.sum()
    return fitted, objective(fitted), {
        "solver": "SLSQP",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "minimum_weight": float(fitted.min()),
        "weight_sum_error": abs(float(fitted.sum()) - 1.0),
    }


def feasible_weight_intervals(
    x: np.ndarray,
    y: np.ndarray,
    projection: ProjectionResult | None = None,
    *,
    solver: str = "CLARABEL",
) -> list[dict[str, Any]]:
    design, target = _as_design(x, y)
    fitted = projection or fit_simplex_projection(design, target, solver=solver)
    allowed = fitted.stage1_objective + fitted.tolerance
    count, peers = design.shape
    result: list[dict[str, Any]] = []
    for index in range(peers):
        bounds: list[float] = []
        extreme_weights: list[list[float]] = []
        statuses: list[str] = []
        for direction in (1.0, -1.0):
            weights = cp.Variable(peers)
            loss = cp.sum_squares(design @ weights - target) / count
            problem = cp.Problem(
                cp.Minimize(direction * weights[index]),
                [weights >= 0, cp.sum(weights) == 1, loss <= allowed],
            )
            _solve(problem, solver)
            statuses.append(str(problem.status))
            if weights.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                raise RuntimeError(f"Weight interval solve failed for peer {index}: {problem.status}")
            solution = np.asarray(weights.value, dtype=np.float64).reshape(-1)
            bounds.append(float(solution[index]))
            extreme_weights.append(solution.tolist())
        minimum, maximum = max(0.0, bounds[0]), min(1.0, bounds[1])
        result.append(
            {
                "peer_index": index,
                "minimum": minimum,
                "maximum": maximum,
                "width": maximum - minimum,
                "status_min": statuses[0],
                "status_max": statuses[1],
                "minimum_solution_weights": extreme_weights[0],
                "maximum_solution_weights": extreme_weights[1],
            }
        )
    return result


def project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (test/reference utility)."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1)
    active = np.flatnonzero(ordered - cumulative / indices > 0)
    if active.size == 0:
        return np.full(vector.size, 1.0 / vector.size)
    rho = active[-1]
    theta = cumulative[rho] / (rho + 1)
    return np.maximum(vector - theta, 0.0)
