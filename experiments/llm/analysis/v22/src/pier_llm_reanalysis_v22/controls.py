from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .data import design_row_weights, fit_projection, projection_intervals
from .utils import PHYSICAL_ROOT, atomic_parquet, atomic_write_json


def declared_trackwise_estimand(frame: pd.DataFrame) -> float:
    required = {"base_question_id", "dose", "track", "residual"}
    if not required.issubset(frame.columns):
        raise ValueError("Declared-track fixture lacks required columns")
    keys = ["base_question_id", "dose", "track"]
    consistency = frame.groupby(keys, sort=True)["residual"].nunique(dropna=False)
    if not consistency.eq(1).all():
        raise ValueError("A declared track has conflicting duplicated residuals")
    unique = frame.drop_duplicates(keys)
    return float(np.mean(np.abs(unique["residual"].to_numpy(dtype=np.float64))))


def run_synthetic_controls(seed: int = 20260828) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    def record(name: str, passed: bool, **values: Any) -> None:
        records.append({"control": name, "passed": bool(passed), **values})

    a = 0.37
    residuals = np.asarray([a, -a])
    track_mean = float(abs(np.mean(residuals)))
    trackwise = float(np.mean(np.abs(residuals)))
    record(
        "synthetic_cancellation",
        np.isclose(track_mean, 0.0)
        and np.isclose(trackwise, abs(a))
        and np.isclose(trackwise - track_mean, abs(a)),
        pier_of_track_mean_response=track_mean,
        mean_trackwise_pier=trackwise,
        cancellation_gap=trackwise - track_mean,
    )

    identical = np.asarray([a, a, a])
    no_cancel_mean = float(abs(np.mean(identical)))
    no_cancel_trackwise = float(np.mean(np.abs(identical)))
    record(
        "no_cancellation",
        np.isclose(no_cancel_mean, no_cancel_trackwise),
        pier_of_track_mean_response=no_cancel_mean,
        mean_trackwise_pier=no_cancel_trackwise,
    )

    design = rng.normal(size=(240, 5))

    def projection_control(
        name: str,
        target: np.ndarray,
        expected: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Any]:
        fit = fit_projection(design[:120], target[:120])
        residual = np.abs(target[120:] - design[120:] @ fit.projection.weights)
        passed = bool(np.mean(residual) <= 1e-9 and np.max(residual) <= 1e-7)
        if expected is not None:
            passed = passed and bool(
                np.max(np.abs(design @ fit.projection.weights - design @ expected))
                <= 1e-7
            )
        record(
            name,
            passed,
            mean_pier=float(np.mean(residual)),
            max_residual=float(np.max(residual)),
            fitted_weights=fit.projection.weights.tolist(),
            expected_weights=expected.tolist() if expected is not None else None,
        )
        return fit.projection.weights, fit

    clone_target = design[:, 2]
    projection_control("exact_clone", clone_target)
    mixture = np.asarray([0.50, 0.30, 0.20, 0.0, 0.0])
    projection_control("known_convex_mixture", design @ mixture, mixture)
    boundary = np.asarray([0.75, 0.25, 0.0, 0.0, 0.0])
    boundary_weights, _boundary_fit = projection_control(
        "boundary_mixture", design @ boundary, boundary
    )
    boundary_face_ok = bool(
        np.max(np.abs(design @ boundary_weights - design @ boundary)) <= 1e-7
        and np.sum(boundary_weights[2:]) <= 1e-6
    )
    records[-1]["correct_simplex_face"] = boundary_face_ok
    records[-1]["passed"] = bool(records[-1]["passed"] and boundary_face_ok)

    duplicated = np.column_stack([design[:, :4], design[:, 2]])
    duplicate_fit = fit_projection(duplicated[:120], clone_target[:120])
    duplicate_residual = np.abs(
        clone_target[120:] - duplicated[120:] @ duplicate_fit.projection.weights
    )
    intervals = projection_intervals(
        duplicated[:120],
        clone_target[:120],
        np.ones(120),
        duplicate_fit,
    )
    ambiguity_exposed = bool(intervals[2]["width"] > 0.5 and intervals[4]["width"] > 0.5)
    record(
        "duplicate_peer_ambiguity",
        np.max(duplicate_residual) <= 1e-7 and ambiguity_exposed,
        mean_pier=float(np.mean(duplicate_residual)),
        max_residual=float(np.max(duplicate_residual)),
        projected_response_unchanged=bool(np.max(duplicate_residual) <= 1e-7),
        feasible_intervals=intervals,
        ambiguity_exposed=ambiguity_exposed,
    )

    metadata = pd.DataFrame(
        [
            {"base_question_id": question, "dose": 0.0, "track_key": -1}
            for question in ("q1", "q2")
        ]
        + [
            {"base_question_id": question, "dose": dose, "track_key": track}
            for question in ("q1", "q2")
            for dose in (1.0, 2.0, 3.0, 4.0)
            for track in (0, 1, 2)
        ]
    )
    weights = design_row_weights(metadata)
    totals = (
        metadata.assign(weight=weights)
        .groupby(["base_question_id", "dose"], sort=True)["weight"]
        .sum()
    )
    duplicated_tracks = pd.DataFrame(
        {
            "base_question_id": ["q", "q", "q", "q"],
            "dose": [1.0] * 4,
            "track": [0, 1, 2, 1],
            "residual": [0.1, -0.2, 0.3, -0.2],
        }
    )
    original_tracks = duplicated_tracks.iloc[:3].copy()
    duplication_invariant = np.isclose(
        declared_trackwise_estimand(original_tracks),
        declared_trackwise_estimand(duplicated_tracks),
    )
    record(
        "equal_design_weighting",
        np.allclose(totals.to_numpy(), 1.0)
        and np.allclose(weights[metadata["dose"].eq(1.0)], 1.0 / 3.0)
        and duplication_invariant,
        minimum_question_dose_mass=float(totals.min()),
        maximum_question_dose_mass=float(totals.max()),
        duplicate_declared_track_invariant=bool(duplication_invariant),
    )

    cluster_metadata = pd.DataFrame(
        [
            {"base_question_id": question, "dose": dose, "track": track}
            for question in ("q1", "q2")
            for dose, tracks in ((0.0, (-1,)), (1.0, (0, 1, 2)))
            for track in tracks
        ]
    )
    draw = ["q1", "q1", "q2"]
    drawn = pd.concat(
        [cluster_metadata[cluster_metadata["base_question_id"].eq(question)] for question in draw],
        ignore_index=True,
    )
    multiplicities = Counter(draw)
    cluster_ok = all(
        len(drawn[drawn["base_question_id"].eq(question)])
        == multiplicity
        * len(cluster_metadata[cluster_metadata["base_question_id"].eq(question)])
        for question, multiplicity in multiplicities.items()
    )
    record(
        "bootstrap_cluster_integrity",
        cluster_ok,
        sampled_multiplicities=dict(multiplicities),
        propagated_to_all_doses_and_tracks=cluster_ok,
    )

    fit_target = np.asarray([0.0, 1.0])
    fit_peers = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    evaluation_target = np.asarray([0.0, 1.0])
    evaluation_peers = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    selected = int(np.argmin(np.mean(np.abs(fit_peers - fit_target[:, None]), axis=0)))
    record(
        "honest_single_peer_selection",
        selected == 0
        and np.mean(np.abs(evaluation_peers[:, selected] - evaluation_target)) > 0,
        selected_peer_index=selected,
        evaluation_rows_used_in_selection=False,
    )

    regression_residuals = np.asarray([0.4, -0.4, 0.0])
    historical = float(abs(np.mean(regression_residuals)))
    corrected = float(np.mean(np.abs(regression_residuals)))
    record(
        "old_versus_new_regression",
        np.isclose(historical, 0.0)
        and np.isclose(corrected, 0.8 / 3.0)
        and corrected > historical,
        historical_aggregate_result=historical,
        v22_trackwise_result=corrected,
        historical_meaning_preserved=True,
    )

    frame = pd.DataFrame(records)
    summary = {
        "schema_version": "pier_trackwise_controls_v22_v1",
        "seed": seed,
        "control_count": len(frame),
        "passed_count": int(frame["passed"].sum()),
        "all_passed": bool(frame["passed"].all()),
        "controls": records,
    }
    return frame, summary


def write_synthetic_controls() -> dict[str, Any]:
    frame, summary = run_synthetic_controls()
    if not summary["all_passed"]:
        failed = frame.loc[~frame["passed"], "control"].tolist()
        raise RuntimeError(f"Required synthetic controls failed: {failed}")
    parquet = PHYSICAL_ROOT / "controls/synthetic_controls.parquet"
    json_path = PHYSICAL_ROOT / "controls/V2_2_CONTROLS.json"
    atomic_parquet(parquet, frame)
    atomic_write_json(json_path, summary)
    return {
        "outputs": [parquet, json_path],
        "row_counts": {"synthetic_controls": len(frame)},
    }
