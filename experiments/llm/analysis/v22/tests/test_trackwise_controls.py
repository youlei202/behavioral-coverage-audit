from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pier_llm_reanalysis_v22.controls import (
    declared_trackwise_estimand,
    run_synthetic_controls,
)
from pier_llm_reanalysis_v22.data import (
    _slsqp_feasible_weight_intervals,
    design_row_weights,
    fit_projection,
)
from pier_llm_reanalysis_v22.utils import PHYSICAL_ROOT, sha256_file


def test_every_required_synthetic_control_passes() -> None:
    frame, summary = run_synthetic_controls(seed=20260828)
    assert summary["all_passed"] is True
    assert frame["passed"].all()
    assert {
        "synthetic_cancellation",
        "no_cancellation",
        "exact_clone",
        "known_convex_mixture",
        "boundary_mixture",
        "duplicate_peer_ambiguity",
        "equal_design_weighting",
        "bootstrap_cluster_integrity",
        "honest_single_peer_selection",
        "old_versus_new_regression",
    } == set(frame["control"])


def test_declared_track_duplicate_cannot_change_estimand() -> None:
    base = pd.DataFrame(
        {
            "base_question_id": ["q", "q", "q"],
            "dose": [1.0, 1.0, 1.0],
            "track": [0, 1, 2],
            "residual": [0.1, -0.4, 0.2],
        }
    )
    duplicated = pd.concat([base, base.iloc[[1]]], ignore_index=True)
    assert declared_trackwise_estimand(base) == declared_trackwise_estimand(duplicated)


def test_clean_and_each_nonzero_dose_have_equal_mass() -> None:
    metadata = pd.DataFrame(
        [{"base_question_id": "q", "dose": 0.0, "track_key": -1}]
        + [
            {"base_question_id": "q", "dose": dose, "track_key": track}
            for dose in (1.0, 2.0, 3.0, 4.0)
            for track in (0, 1, 2)
        ]
    )
    weights = design_row_weights(metadata)
    totals = metadata.assign(weight=weights).groupby("dose")["weight"].sum()
    assert np.allclose(totals, 1.0)
    assert np.allclose(weights[metadata["dose"].eq(0.0)], 1.0)
    assert np.allclose(weights[metadata["dose"].ne(0.0)], 1.0 / 3.0)


def test_v2_solver_reference_copy_is_byte_identical() -> None:
    source = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/src/pier_llm/solver.py")
    copied = PHYSICAL_ROOT / "src/pier_llm_reanalysis_v22/v2_solver.py"
    assert sha256_file(source) == sha256_file(copied)


def test_interval_fallback_exposes_duplicate_peer_ambiguity() -> None:
    generator = np.random.default_rng(20260828)
    base = generator.normal(size=(120, 2))
    design = np.column_stack([base[:, 0], base[:, 1], base[:, 0]])
    target = base[:, 0]
    fit = fit_projection(design, target)
    intervals = _slsqp_feasible_weight_intervals(
        design,
        target,
        fit.projection,
    )
    assert intervals[0]["minimum"] <= 1e-6
    assert intervals[0]["maximum"] >= 1.0 - 1e-6
    assert intervals[2]["minimum"] <= 1e-6
    assert intervals[2]["maximum"] >= 1.0 - 1e-6
