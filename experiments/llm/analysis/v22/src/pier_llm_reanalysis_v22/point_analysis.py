from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import (
    FAMILIES,
    AlignedData,
    aggregate_aligned,
    aligned_track_arrays,
    design_row_weights,
    fit_projection,
    fixed_splits,
    high_dose,
    load_validated_inputs,
    projection_intervals,
    scalar_dose_metrics,
    track_response_frame,
    weighted_peer_mae,
)
from .utils import PHYSICAL_ROOT, atomic_parquet, load_config, weight_geometry


def sibling_for(target: str) -> str | None:
    for left, right in load_config()["sibling_pairs"]:
        if target == left:
            return str(right)
        if target == right:
            return str(left)
    return None


def _weight_vector(
    saved: pd.DataFrame, target: str, family: str, split_seed: int, peers: list[str]
) -> np.ndarray:
    subset = saved[
        saved["target"].eq(target)
        & saved["family"].eq(family)
        & saved["split_seed"].eq(split_seed)
    ].set_index("peer")
    return subset.loc[peers, "weight"].to_numpy(dtype=np.float64)


def _fit_trackwise(data: AlignedData, fitting_ids: set[str] | frozenset[str]):
    fitting = data.metadata["base_question_id"].astype(str).isin(fitting_ids).to_numpy()
    row_weights = design_row_weights(data.metadata)
    return fit_projection(data.design[fitting], data.target[fitting], row_weights[fitting]), fitting, row_weights


def run_primary_fit() -> dict[str, Any]:
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    dose_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        frame = track_response_frame(scores, family)
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            data = aligned_track_arrays(frame, target, peers)
            for split in splits.values():
                fit, fitting, all_row_weights = _fit_trackwise(data, split.fitting_ids)
                weights = fit.projection.weights
                intervals = projection_intervals(
                    data.design[fitting],
                    data.target[fitting],
                    all_row_weights[fitting],
                    fit,
                )
                peer_errors = weighted_peer_mae(data, split.fitting_ids)
                selected_index = int(np.argmin(peer_errors))
                single_weights = np.zeros(len(peers), dtype=np.float64)
                single_weights[selected_index] = 1.0
                primary_doses = scalar_dose_metrics(data, weights, split.evaluation_ids)
                single_doses = scalar_dose_metrics(
                    data, single_weights, split.evaluation_ids
                ).set_index("dose")
                geometry = weight_geometry(weights)
                for _, row in primary_doses.iterrows():
                    dose = float(row["dose"])
                    single_error = float(single_doses.loc[dose, "trackwise_pier"])
                    dose_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            **row.to_dict(),
                            "fit_selected_single_peer": peers[selected_index],
                            "single_error_trackwise": single_error,
                            "absolute_improvement": single_error
                            - float(row["trackwise_pier"]),
                            "gap_ratio": single_error
                            / (float(row["trackwise_pier"]) + load_config()["epsilon"]),
                            **geometry,
                        }
                    )
                overall = float(primary_doses["trackwise_pier"].mean())
                track_mean_overall = float(
                    primary_doses["track_mean_response_pier"].mean()
                )
                single_overall = float(single_doses["trackwise_pier"].mean())
                overall_rows.append(
                    {
                        "target": target,
                        "family": family,
                        "split_seed": split.seed,
                        "trackwise_overall_pier": overall,
                        "track_mean_response_overall_pier": track_mean_overall,
                        "overall_cancellation_gap": overall - track_mean_overall,
                        "fit_selected_single_peer": peers[selected_index],
                        "single_error_trackwise": single_overall,
                        "absolute_improvement": single_overall - overall,
                        "gap_ratio": single_overall
                        / (overall + load_config()["epsilon"]),
                        **geometry,
                    }
                )
                diagnostics = fit.projection.diagnostics()
                fallback = (
                    "enumerated_exact_refinement"
                    if "enumerated_exact_refinement" in str(diagnostics["solver_stage2"])
                    else "none"
                )
                solver_rows.append(
                    {
                        "target": target,
                        "family": family,
                        "split_seed": split.seed,
                        "normalized_weighted_objective": fit.normalized_weighted_objective,
                        "row_weight_total": fit.row_weight_total,
                        "fitted_row_count": fit.fitted_row_count,
                        "fallback_path": fallback,
                        "simplex_feasible": bool(
                            diagnostics["minimum_weight"] >= -1e-9
                            and diagnostics["weight_sum_error"] <= 1e-8
                        ),
                        "projected_response_preserved": bool(
                            diagnostics["maximum_projection_change"]
                            <= 10 * diagnostics["projection_preservation_tolerance"]
                            + 1e-11
                        ),
                        **diagnostics,
                    }
                )
                for peer, weight, interval in zip(
                    peers, weights, intervals, strict=True
                ):
                    weight_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "peer": peer,
                            "weight": float(weight),
                            "feasible_minimum": float(interval["minimum"]),
                            "feasible_maximum": float(interval["maximum"]),
                            "feasible_width": float(interval["width"]),
                            "weight_representation_ambiguous": bool(
                                interval["width"] > 1e-4
                            ),
                            "interval_solver": interval["interval_solver"],
                            "interval_fallback_path": interval[
                                "interval_fallback_path"
                            ],
                            "normalized_weighted_objective": fit.normalized_weighted_objective,
                            **diagnostics,
                        }
                    )

    dose = pd.DataFrame(dose_rows).sort_values(
        ["target", "family", "split_seed", "dose"]
    )
    overall = pd.DataFrame(overall_rows).sort_values(
        ["target", "family", "split_seed"]
    )
    weights = pd.DataFrame(weight_rows).sort_values(
        ["target", "family", "split_seed", "peer"]
    )
    diagnostics = pd.DataFrame(solver_rows).sort_values(
        ["target", "family", "split_seed"]
    )
    if len(dose) != 800 or len(overall) != 160 or len(weights) != 1120:
        raise ValueError(
            f"Unexpected primary output sizes: dose={len(dose)}, "
            f"overall={len(overall)}, weights={len(weights)}"
        )
    dose_path = PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    weight_path = PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_weights.parquet"
    overall_path = PHYSICAL_ROOT / "outputs/analysis/trackwise_overall_results.parquet"
    cancellation_path = PHYSICAL_ROOT / "outputs/analysis/cancellation_gap_results.parquet"
    solver_path = PHYSICAL_ROOT / "outputs/analysis/solver_diagnostics.parquet"
    cancellation_columns = [
        "target",
        "family",
        "split_seed",
        "dose",
        "trackwise_pier",
        "track_mean_response_pier",
        "cancellation_gap",
        "relative_cancellation_gap",
        "relative_hidden_fraction",
        "between_track_std",
        "between_track_range",
        "mean_within_question_signed_track_std",
        "mean_within_question_signed_track_range",
        "track_piers",
        "jensen_consistent",
    ]
    atomic_parquet(dose_path, dose)
    atomic_parquet(weight_path, weights)
    atomic_parquet(overall_path, overall)
    atomic_parquet(cancellation_path, dose[cancellation_columns])
    atomic_parquet(solver_path, diagnostics)
    return {
        "outputs": [dose_path, weight_path, overall_path, cancellation_path, solver_path],
        "row_counts": {
            "dose": len(dose),
            "weights": len(weights),
            "overall": len(overall),
            "cancellation": len(dose),
            "solver_diagnostics": len(diagnostics),
        },
    }


def run_estimand_decomposition() -> dict[str, Any]:
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    primary_weights = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_weights.parquet"
    )
    v21_root = Path(load_config()["source_roots"]["v21"]["physical_alias"])
    v21_primary = pd.read_parquet(
        v21_root / "outputs/analysis/primary_dose_results.parquet"
    )
    rows: list[dict[str, Any]] = []
    maximum_regression_error = 0.0

    for family in FAMILIES:
        frame = track_response_frame(scores, family)
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            track = aligned_track_arrays(frame, target, peers)
            aggregate = aggregate_aligned(track)
            for split in splits.values():
                aggregate_fit_mask = aggregate.metadata["base_question_id"].astype(str).isin(
                    split.fitting_ids
                ).to_numpy()
                aggregate_fit = fit_projection(
                    aggregate.design[aggregate_fit_mask],
                    aggregate.target[aggregate_fit_mask],
                )
                aggregate_weights = aggregate_fit.projection.weights
                trackwise_weights = _weight_vector(
                    primary_weights, target, family, split.seed, peers
                )
                aa = scalar_dose_metrics(
                    aggregate, aggregate_weights, split.evaluation_ids
                ).set_index("dose")
                at = scalar_dose_metrics(
                    track, aggregate_weights, split.evaluation_ids
                ).set_index("dose")
                ta = scalar_dose_metrics(
                    aggregate, trackwise_weights, split.evaluation_ids
                ).set_index("dose")
                tt = scalar_dose_metrics(
                    track, trackwise_weights, split.evaluation_ids
                ).set_index("dose")
                aggregate_eval = aggregate.metadata[
                    "base_question_id"
                ].astype(str).isin(split.evaluation_ids).to_numpy()
                for dose in aa.index:
                    v21 = v21_primary[
                        v21_primary["target"].eq(target)
                        & v21_primary["family"].eq(family)
                        & v21_primary["split_seed"].eq(split.seed)
                        & v21_primary["dose"].eq(float(dose))
                    ]
                    if len(v21) != 1:
                        raise ValueError("Missing unique V2.1 regression row")
                    historical = float(v21.iloc[0]["convex_error"])
                    baseline = float(aa.loc[dose, "trackwise_pier"])
                    regression_error = abs(historical - baseline)
                    maximum_regression_error = max(
                        maximum_regression_error, regression_error
                    )
                    dose_mask = aggregate_eval & aggregate.metadata["dose"].astype(float).eq(
                        float(dose)
                    ).to_numpy()
                    aggregate_prediction = (
                        aggregate.design[dose_mask] @ aggregate_weights
                    )
                    trackwise_prediction = (
                        aggregate.design[dose_mask] @ trackwise_weights
                    )
                    aggregate_fit_track_eval = float(at.loc[dose, "trackwise_pier"])
                    track_fit_aggregate_eval = float(ta.loc[dose, "trackwise_pier"])
                    primary = float(tt.loc[dose, "trackwise_pier"])
                    evaluation_correction = aggregate_fit_track_eval - baseline
                    fitting_correction = track_fit_aggregate_eval - baseline
                    interaction = (
                        primary
                        - aggregate_fit_track_eval
                        - track_fit_aggregate_eval
                        + baseline
                    )
                    rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "dose": float(dose),
                            "aggregate_fit_aggregate_eval": baseline,
                            "aggregate_fit_trackwise_eval": aggregate_fit_track_eval,
                            "trackwise_fit_aggregate_eval": track_fit_aggregate_eval,
                            "trackwise_fit_trackwise_eval": primary,
                            "evaluation_correction": evaluation_correction,
                            "fitting_correction": fitting_correction,
                            "interaction": interaction,
                            "total_correction": primary - baseline,
                            "cancellation_gap": float(
                                tt.loc[dose, "cancellation_gap"]
                            ),
                            "weight_l1_change": float(
                                np.abs(trackwise_weights - aggregate_weights).sum()
                            ),
                            "weight_l2_change": float(
                                np.linalg.norm(trackwise_weights - aggregate_weights)
                            ),
                            "projected_response_change": float(
                                np.mean(
                                    np.abs(
                                        trackwise_prediction - aggregate_prediction
                                    )
                                )
                            ),
                            "v21_historical_value": historical,
                            "v21_regression_absolute_error": regression_error,
                            "aggregate_weights": aggregate_weights.tolist(),
                            "trackwise_weights": trackwise_weights.tolist(),
                        }
                    )

    frame = pd.DataFrame(rows)
    if maximum_regression_error > 1e-10:
        raise AssertionError(
            f"V2.1 aggregate regression changed by {maximum_regression_error}"
        )
    endpoint_records: list[dict[str, Any]] = []
    for (target, family, split_seed), group in frame.groupby(
        ["target", "family", "split_seed"], sort=True
    ):
        clean = group[group["dose"].eq(0.0)].iloc[0]
        high = group[group["dose"].eq(high_dose(str(family)))].iloc[0]
        record: dict[str, Any] = {
            "target": target,
            "family": family,
            "split_seed": split_seed,
        }
        for column in (
            "aggregate_fit_aggregate_eval",
            "aggregate_fit_trackwise_eval",
            "trackwise_fit_aggregate_eval",
            "trackwise_fit_trackwise_eval",
        ):
            record[f"{column}_endpoint_delta"] = float(high[column] - clean[column])
        endpoint_records.append(record)
    endpoint = pd.DataFrame(endpoint_records)
    endpoint["endpoint_sign_agreement"] = (
        np.sign(endpoint["aggregate_fit_aggregate_eval_endpoint_delta"])
        == np.sign(endpoint["trackwise_fit_trackwise_eval_endpoint_delta"])
    )
    frame = frame.merge(
        endpoint,
        on=["target", "family", "split_seed"],
        validate="many_to_one",
    )
    rank_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["family", "split_seed", "dose"], sort=True):
        correlation = spearmanr(
            group["aggregate_fit_aggregate_eval"],
            group["trackwise_fit_trackwise_eval"],
        ).statistic
        rank_rows.append(
            {
                "family": keys[0],
                "split_seed": keys[1],
                "dose": keys[2],
                "rank_correlation_aggregate_to_v22": float(correlation),
            }
        )
    frame = frame.merge(
        pd.DataFrame(rank_rows),
        on=["family", "split_seed", "dose"],
        validate="many_to_one",
    )
    if len(frame) != 800:
        raise ValueError(f"Expected 800 decomposition rows, found {len(frame)}")
    path = PHYSICAL_ROOT / "outputs/analysis/estimand_decomposition.parquet"
    atomic_parquet(path, frame.sort_values(["target", "family", "split_seed", "dose"]))
    return {
        "outputs": [path],
        "row_counts": {"estimand_decomposition": len(frame)},
        "maximum_v21_regression_absolute_error": maximum_regression_error,
    }


def run_endpoint_analysis() -> dict[str, Any]:
    dose = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    )
    rows: list[dict[str, Any]] = []
    for (target, family, split_seed), group in dose.groupby(
        ["target", "family", "split_seed"], sort=True
    ):
        clean = group[group["dose"].eq(0.0)].iloc[0]
        high = group[group["dose"].eq(high_dose(str(family)))].iloc[0]
        delta = float(high["trackwise_pier"] - clean["trackwise_pier"])
        rows.append(
            {
                "target": target,
                "family": family,
                "split_seed": int(split_seed),
                "clean_dose": 0.0,
                "high_dose": high_dose(str(family)),
                "trackwise_clean_pier": float(clean["trackwise_pier"]),
                "trackwise_high_pier": float(high["trackwise_pier"]),
                "trackwise_delta": delta,
                "trackwise_relative_delta": delta
                / (float(clean["trackwise_pier"]) + load_config()["epsilon"]),
                "track_mean_response_clean_pier": float(
                    clean["track_mean_response_pier"]
                ),
                "track_mean_response_high_pier": float(
                    high["track_mean_response_pier"]
                ),
                "track_mean_response_delta": float(
                    high["track_mean_response_pier"]
                    - clean["track_mean_response_pier"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["target", "family"], sort=True):
        values = group["trackwise_delta"].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "target": keys[0],
                "family": keys[1],
                "mean_trackwise_delta": float(np.mean(values)),
                "median_trackwise_delta": float(np.median(values)),
                "trackwise_delta_standard_deviation": float(np.std(values, ddof=1)),
                "minimum_trackwise_delta": float(np.min(values)),
                "maximum_trackwise_delta": float(np.max(values)),
                "positive_split_count": int(np.sum(values > 0)),
                "negative_split_count": int(np.sum(values < 0)),
                "tie_split_count": int(np.sum(values == 0)),
            }
        )
    frame = frame.merge(
        pd.DataFrame(summaries), on=["target", "family"], validate="many_to_one"
    )
    if len(frame) != 160:
        raise ValueError(f"Expected 160 endpoint rows, found {len(frame)}")
    path = PHYSICAL_ROOT / "outputs/analysis/trackwise_endpoint_effects.parquet"
    atomic_parquet(path, frame.sort_values(["target", "family", "split_seed"]))
    return {"outputs": [path], "row_counts": {"endpoint_effects": len(frame)}}


def run_convexity_analysis() -> dict[str, Any]:
    overall = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_overall_results.parquet"
    )
    columns = [
        "target",
        "family",
        "split_seed",
        "fit_selected_single_peer",
        "single_error_trackwise",
        "trackwise_overall_pier",
        "absolute_improvement",
        "gap_ratio",
        "effective_peer_count",
        "top1_weight_mass",
        "top3_weight_mass",
    ]
    frame = overall[columns].rename(
        columns={"trackwise_overall_pier": "convex_error_trackwise"}
    )
    if len(frame) != 160:
        raise ValueError(f"Expected 160 convexity rows, found {len(frame)}")
    path = PHYSICAL_ROOT / "outputs/analysis/trackwise_convex_vs_single.parquet"
    atomic_parquet(path, frame.sort_values(["target", "family", "split_seed"]))
    return {"outputs": [path], "row_counts": {"convexity": len(frame)}}


def _fit_and_metrics(
    data: AlignedData,
    retained_indices: list[int],
    split: Any,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    fitting = data.metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    row_weights = design_row_weights(data.metadata)
    design = data.design[:, retained_indices]
    fit = fit_projection(
        design[fitting], data.target[fitting], row_weights[fitting]
    )
    reduced = AlignedData(data.metadata, data.target, design, tuple(data.peers[index] for index in retained_indices))
    dose = scalar_dose_metrics(reduced, fit.projection.weights, split.evaluation_ids)
    return fit.projection.weights, dose, float(dose["trackwise_pier"].mean())


def run_removal_analyses() -> dict[str, Any]:
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    sibling_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        frame = track_response_frame(scores, family)
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            data = aligned_track_arrays(frame, target, peers)
            sibling = sibling_for(target)
            for split in splits.values():
                all_indices = list(range(len(peers)))
                _all_weights, baseline_dose, baseline_overall = _fit_and_metrics(
                    data, all_indices, split
                )
                baseline_dose = baseline_dose.set_index("dose")
                removal_cache: dict[str, tuple[pd.DataFrame, float]] = {}
                split_influence: list[dict[str, Any]] = []
                for removed_index, removed_peer in enumerate(peers):
                    retained = [index for index in all_indices if index != removed_index]
                    _weights, removed_dose, removed_overall = _fit_and_metrics(
                        data, retained, split
                    )
                    removal_cache[removed_peer] = (
                        removed_dose.set_index("dose"),
                        removed_overall,
                    )
                    split_influence.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "removed_peer": removed_peer,
                            "all_peer_trackwise_pier": baseline_overall,
                            "removed_trackwise_pier": removed_overall,
                            "inflation": removed_overall - baseline_overall,
                            "is_declared_sibling": removed_peer == sibling,
                            "tie_handling": "minimum rank for exactly equal inflation",
                        }
                    )
                influence = pd.DataFrame(split_influence)
                influence["influence_rank"] = influence["inflation"].rank(
                    method="min", ascending=False
                ).astype(int)
                influence["exact_tie_count"] = influence.groupby("inflation")[
                    "removed_peer"
                ].transform("size").astype(int)
                influence_rows.extend(influence.to_dict(orient="records"))

                if sibling is None:
                    continue
                if sibling not in removal_cache:
                    raise AssertionError(f"Declared sibling {sibling} is not a peer of {target}")
                sibling_dose, sibling_overall = removal_cache[sibling]
                baseline_endpoint = float(
                    baseline_dose.loc[high_dose(family), "trackwise_pier"]
                    - baseline_dose.loc[0.0, "trackwise_pier"]
                )
                removed_endpoint = float(
                    sibling_dose.loc[high_dose(family), "trackwise_pier"]
                    - sibling_dose.loc[0.0, "trackwise_pier"]
                )
                for dose in baseline_dose.index:
                    sibling_rows.append(
                        {
                            "target": target,
                            "removed_sibling": sibling,
                            "family": family,
                            "split_seed": split.seed,
                            "dose": float(dose),
                            "evaluation_scope": "dose",
                            "all_peer_trackwise_pier": float(
                                baseline_dose.loc[dose, "trackwise_pier"]
                            ),
                            "sibling_removed_trackwise_pier": float(
                                sibling_dose.loc[dose, "trackwise_pier"]
                            ),
                            "inflation": float(
                                sibling_dose.loc[dose, "trackwise_pier"]
                                - baseline_dose.loc[dose, "trackwise_pier"]
                            ),
                            "all_peer_endpoint_delta": baseline_endpoint,
                            "sibling_removed_endpoint_delta": removed_endpoint,
                            "endpoint_inflation": removed_endpoint - baseline_endpoint,
                        }
                    )
                sibling_rows.append(
                    {
                        "target": target,
                        "removed_sibling": sibling,
                        "family": family,
                        "split_seed": split.seed,
                        "dose": math.nan,
                        "evaluation_scope": "overall_equal_dose",
                        "all_peer_trackwise_pier": baseline_overall,
                        "sibling_removed_trackwise_pier": sibling_overall,
                        "inflation": sibling_overall - baseline_overall,
                        "all_peer_endpoint_delta": baseline_endpoint,
                        "sibling_removed_endpoint_delta": removed_endpoint,
                        "endpoint_inflation": removed_endpoint - baseline_endpoint,
                    }
                )
                sibling_inflation = sibling_overall - baseline_overall
                non_siblings = [peer for peer in peers if peer != sibling]
                if len(non_siblings) != 6:
                    raise AssertionError("Exact non-sibling null must contain six removals")
                null_inflations = np.asarray(
                    [removal_cache[peer][1] - baseline_overall for peer in non_siblings],
                    dtype=np.float64,
                )
                sibling_rank = 1 + int(np.sum(null_inflations > sibling_inflation))
                percentile = float(np.mean(null_inflations <= sibling_inflation))
                for peer, inflation in zip(non_siblings, null_inflations, strict=True):
                    null_rows.append(
                        {
                            "target": target,
                            "removed_sibling": sibling,
                            "family": family,
                            "split_seed": split.seed,
                            "removed_non_sibling": peer,
                            "all_peer_trackwise_pier": baseline_overall,
                            "non_sibling_removed_trackwise_pier": removal_cache[peer][1],
                            "non_sibling_inflation": float(inflation),
                            "sibling_inflation": sibling_inflation,
                            "median_non_sibling_inflation": float(
                                np.median(null_inflations)
                            ),
                            "largest_non_sibling_inflation": float(
                                np.max(null_inflations)
                            ),
                            "sibling_rank": sibling_rank,
                            "sibling_percentile": percentile,
                            "sibling_is_largest": bool(
                                sibling_inflation > np.max(null_inflations)
                            ),
                            "null_method": "exact_enumeration",
                            "tie_handling": "minimum rank for exactly equal inflation",
                        }
                    )

    sibling_frame = pd.DataFrame(sibling_rows)
    null_frame = pd.DataFrame(null_rows)
    influence_frame = pd.DataFrame(influence_rows)
    largest = (
        null_frame.groupby(["target", "family", "split_seed"], as_index=False)[
            "sibling_is_largest"
        ]
        .first()
        .groupby(["target", "family"], as_index=False)["sibling_is_largest"]
        .sum()
        .rename(columns={"sibling_is_largest": "splits_sibling_largest"})
    )
    sibling_frame = sibling_frame.merge(
        largest, on=["target", "family"], validate="many_to_one"
    )
    null_frame = null_frame.merge(
        largest, on=["target", "family"], validate="many_to_one"
    )
    if len(sibling_frame) != 480 or len(null_frame) != 480 or len(influence_frame) != 1120:
        raise ValueError(
            f"Unexpected removal sizes: sibling={len(sibling_frame)}, "
            f"null={len(null_frame)}, influence={len(influence_frame)}"
        )
    sibling_path = PHYSICAL_ROOT / "outputs/analysis/trackwise_sibling_removal.parquet"
    null_path = (
        PHYSICAL_ROOT
        / "outputs/analysis/trackwise_non_sibling_removal_null.parquet"
    )
    influence_path = (
        PHYSICAL_ROOT / "outputs/analysis/trackwise_single_peer_influence.parquet"
    )
    atomic_parquet(
        sibling_path,
        sibling_frame.sort_values(
            ["target", "family", "split_seed", "evaluation_scope", "dose"],
            na_position="last",
        ),
    )
    atomic_parquet(
        null_path,
        null_frame.sort_values(
            ["target", "family", "split_seed", "removed_non_sibling"]
        ),
    )
    atomic_parquet(
        influence_path,
        influence_frame.sort_values(
            ["target", "family", "split_seed", "influence_rank", "removed_peer"]
        ),
    )
    return {
        "outputs": [sibling_path, null_path, influence_path],
        "row_counts": {
            "sibling_removal": len(sibling_frame),
            "non_sibling_null": len(null_frame),
            "single_peer_influence": len(influence_frame),
        },
    }
