from __future__ import annotations

import cvxpy as cp
import numpy as np

import pier_llm.solver as solver_module
from pier_llm.analysis import run_synthetic_controls
from pier_llm.solver import (
    _minimum_norm_exact_representative,
    feasible_weight_intervals,
    fit_simplex_projection,
    fit_simplex_slsqp,
    squared_loss,
)


def test_two_stage_projection_matches_independent_solver() -> None:
    rng = np.random.default_rng(17)
    design = rng.normal(size=(240, 6))
    target = design @ np.array([0.4, 0.0, 0.1, 0.2, 0.3, 0.0])
    target += rng.normal(scale=0.01, size=len(target))
    primary = fit_simplex_projection(design, target)
    independent_weights, independent_loss, _ = fit_simplex_slsqp(design, target)
    assert primary.weights.min() >= -1e-9
    assert abs(primary.weights.sum() - 1) <= 1e-8
    assert abs(primary.stage1_objective - independent_loss) <= 5e-8
    assert squared_loss(design, target, independent_weights) == independent_loss


def test_exact_sparse_boundary_and_vector_controls() -> None:
    controls, summary = run_synthetic_controls(seed=23)
    assert len(controls) >= 6
    assert summary["passed"]
    assert controls["passed"].all()


def test_duplicate_peer_intervals_reveal_ambiguity() -> None:
    rng = np.random.default_rng(99)
    source = rng.normal(size=(180, 3))
    design = np.column_stack([source, source[:, 0]])
    target = source[:, 0]
    projection = fit_simplex_projection(design, target)
    intervals = feasible_weight_intervals(design, target, projection)
    assert squared_loss(design, target, projection.weights) <= 1e-8
    assert intervals[0]["width"] > 0.9
    assert intervals[3]["width"] > 0.9


def test_exact_stage2_refinement_uses_minimum_norm_duplicate_representation() -> None:
    source = np.linspace(-1.0, 1.0, 51)
    design = np.column_stack([source, source, np.square(source)])
    reference = np.array([1.0, 0.0, 0.0])

    refined = _minimum_norm_exact_representative(
        design,
        reference,
        projection_tolerance=1e-10,
    )

    assert np.allclose(refined, [0.5, 0.5, 0.0], atol=1e-12)
    assert np.max(np.abs(design @ refined - design @ reference)) <= 1e-12
    assert abs(float(refined.sum()) - 1.0) <= 1e-12


def test_stage2_solver_error_uses_exact_refinement(monkeypatch) -> None:
    rng = np.random.default_rng(123)
    design = rng.normal(size=(80, 4))
    target = design @ np.array([0.1, 0.2, 0.3, 0.4])

    def fail_solver(problem, solver):
        del problem, solver
        raise cp.error.SolverError("synthetic failure")

    monkeypatch.setattr(solver_module, "_solve", fail_solver)
    result = fit_simplex_projection(design, target)

    assert result.solver_stage2.endswith("+enumerated_exact_refinement")
    assert result.status_stage2 == "solver_error:refined"
    assert result.maximum_projection_change <= 1e-12
    assert result.minimum_weight >= 0
    assert result.weight_sum_error <= 1e-12
