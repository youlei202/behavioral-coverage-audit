from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pier_llm.analysis import aggregate_family_responses, aligned_response_arrays

from .core import (
    FAMILIES,
    fit_projection,
    fixed_splits,
    high_dose,
    track_aligned_arrays,
    track_response_frame,
    weighted_mean,
)
from .input_validation import load_validated_inputs
from .utils import PHYSICAL_ROOT, atomic_parquet


def cancellation_metrics(track_pier: np.ndarray, pier_of_mean_response: float) -> dict[str, float]:
    mean_trackwise = float(np.mean(track_pier))
    gap = mean_trackwise - float(pier_of_mean_response)
    if gap < -1e-12:
        raise AssertionError(
            "Mean trackwise absolute residual fell below the absolute residual of the mean"
        )
    return {
        "pier_of_mean_response": float(pier_of_mean_response),
        "mean_of_trackwise_pier": mean_trackwise,
        "between_track_standard_deviation": float(np.std(track_pier, ddof=0)),
        "between_track_range": float(np.ptp(track_pier)),
        "track_cancellation_gap": max(0.0, gap),
    }


def _weight_vector(weights: pd.DataFrame, target: str, family: str, split_seed: int, peers: list[str]) -> np.ndarray:
    subset = weights[
        weights["target"].eq(target)
        & weights["family"].eq(family)
        & weights["split_seed"].eq(split_seed)
    ].set_index("peer")
    return subset.loc[peers, "weight"].to_numpy(dtype=np.float64)


def run_track_analysis() -> dict[str, Any]:
    scores, _generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    primary = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/primary_dose_results.parquet")
    saved_weights = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/primary_weights.parquet")
    track_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        track_frame = track_response_frame(scores, family)
        aggregated = aggregate_family_responses(scores, family, "gold_probability")
        for split in splits.values():
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                weights = _weight_vector(saved_weights, target, family, split.seed, peers)
                metadata, target_values, design = track_aligned_arrays(track_frame, target, peers)
                identifiers = metadata["base_question_id"].astype(str)
                fitting = identifiers.isin(split.fitting_ids).to_numpy()
                evaluation = identifiers.isin(split.evaluation_ids).to_numpy()

                clean_mask = evaluation & metadata["dose"].astype(float).eq(0.0).to_numpy()
                clean_pier = float(
                    np.mean(np.abs(target_values[clean_mask] - design[clean_mask] @ weights))
                )
                primary_group = primary[
                    primary["target"].eq(target)
                    & primary["family"].eq(family)
                    & primary["split_seed"].eq(split.seed)
                ]
                aggregate_delta = float(
                    primary_group[primary_group["dose"].eq(high_dose(family))]["convex_error"].iloc[0]
                    - primary_group[primary_group["dose"].eq(0.0)]["convex_error"].iloc[0]
                )
                for dose in sorted(float(value) for value in metadata["dose"].unique()):
                    aggregate_pier = float(
                        primary_group[primary_group["dose"].eq(dose)]["convex_error"].iloc[0]
                    )
                    if math.isclose(dose, 0.0):
                        metrics = cancellation_metrics(np.asarray([clean_pier]), aggregate_pier)
                        track_rows.append(
                            {
                                "target": target,
                                "family": family,
                                "split_seed": split.seed,
                                "dose": dose,
                                "track": -1,
                                "track_pier": clean_pier,
                                "track_endpoint_delta": 0.0,
                                "aggregate_endpoint_delta": aggregate_delta,
                                "same_sign_as_aggregate": True,
                                "same_sign_track_count": 3,
                                **metrics,
                            }
                        )
                        continue
                    dose_rows: list[tuple[int, float]] = []
                    for track in sorted(metadata.loc[metadata["dose"].eq(dose), "track_key"].unique()):
                        mask = evaluation & metadata["dose"].eq(dose).to_numpy() & metadata[
                            "track_key"
                        ].eq(track).to_numpy()
                        pier = float(
                            np.mean(np.abs(target_values[mask] - design[mask] @ weights))
                        )
                        dose_rows.append((int(track), pier))
                    if len(dose_rows) != 3:
                        raise ValueError(f"Expected three tracks for {target}, {family}, {dose}")
                    track_piers = np.asarray([value for _track, value in dose_rows])
                    metrics = cancellation_metrics(track_piers, aggregate_pier)
                    endpoint_deltas = track_piers - clean_pier
                    same_sign_count = int(
                        np.sum(np.sign(endpoint_deltas) == np.sign(aggregate_delta))
                    ) if not math.isclose(aggregate_delta, 0.0, abs_tol=1e-15) else int(
                        np.sum(np.isclose(endpoint_deltas, 0.0, atol=1e-15))
                    )
                    for (track, pier), track_delta in zip(dose_rows, endpoint_deltas, strict=True):
                        track_rows.append(
                            {
                                "target": target,
                                "family": family,
                                "split_seed": split.seed,
                                "dose": dose,
                                "track": track,
                                "track_pier": pier,
                                "track_endpoint_delta": float(track_delta),
                                "aggregate_endpoint_delta": aggregate_delta,
                                "same_sign_as_aggregate": bool(
                                    np.sign(track_delta) == np.sign(aggregate_delta)
                                ),
                                "same_sign_track_count": same_sign_count,
                                **metrics,
                            }
                        )

                row_weights = np.where(metadata["dose"].astype(float).eq(0.0), 1.0, 1.0 / 3.0)
                sqrt_weights = np.sqrt(row_weights[fitting])
                trackwise_projection = fit_projection(
                    design[fitting] * sqrt_weights[:, None],
                    target_values[fitting] * sqrt_weights,
                )

                aggregate_meta, aggregate_y, aggregate_x = aligned_response_arrays(
                    aggregated, target, peers
                )
                aggregate_ids = aggregate_meta["base_question_id"].astype(str)
                aggregate_eval = aggregate_ids.isin(split.evaluation_ids).to_numpy()
                primary_prediction = aggregate_x[aggregate_eval] @ weights
                trackwise_prediction = aggregate_x[aggregate_eval] @ trackwise_projection.weights
                primary_overall = float(
                    np.mean(np.abs(aggregate_y[aggregate_eval] - primary_prediction))
                )
                trackwise_overall = float(
                    np.mean(np.abs(aggregate_y[aggregate_eval] - trackwise_prediction))
                )

                aggregate_clean = aggregate_eval & aggregate_meta["dose"].eq(0.0).to_numpy()
                aggregate_high = aggregate_eval & aggregate_meta["dose"].eq(
                    high_dose(family)
                ).to_numpy()

                def endpoint_delta(
                    candidate_weights: np.ndarray,
                    clean: np.ndarray = aggregate_clean,
                    high: np.ndarray = aggregate_high,
                    response: np.ndarray = aggregate_y,
                    response_design: np.ndarray = aggregate_x,
                ) -> float:
                    clean_error = float(
                        np.mean(
                            np.abs(
                                response[clean]
                                - response_design[clean] @ candidate_weights
                            )
                        )
                    )
                    high_error = float(
                        np.mean(
                            np.abs(
                                response[high]
                                - response_design[high] @ candidate_weights
                            )
                        )
                    )
                    return high_error - clean_error

                fit_weighted_abs = np.average(
                    np.abs(design[fitting] - target_values[fitting, None]),
                    axis=0,
                    weights=row_weights[fitting],
                )
                trackwise_single_index = int(np.argmin(fit_weighted_abs))
                trackwise_single_error = float(
                    np.mean(
                        np.abs(
                            aggregate_y[aggregate_eval]
                            - aggregate_x[aggregate_eval, trackwise_single_index]
                        )
                    )
                )
                primary_overall_row = primary_group[primary_group["dose"].isna()].iloc[0]
                individual_eval_residual = np.abs(
                    target_values[evaluation] - design[evaluation] @ trackwise_projection.weights
                )
                sensitivity_rows.append(
                    {
                        "target": target,
                        "family": family,
                        "split_seed": split.seed,
                        "primary_overall_pier": primary_overall,
                        "trackwise_fit_overall_pier": trackwise_overall,
                        "trackwise_loss_evaluation_pier": weighted_mean(
                            individual_eval_residual, row_weights[evaluation]
                        ),
                        "primary_endpoint_delta": endpoint_delta(weights),
                        "trackwise_fit_endpoint_delta": endpoint_delta(
                            trackwise_projection.weights
                        ),
                        "primary_convexity_improvement": float(
                            primary_overall_row["absolute_improvement"]
                        ),
                        "trackwise_fit_convexity_improvement": trackwise_single_error
                        - trackwise_overall,
                        "primary_fit_selected_single_peer": str(
                            primary_overall_row["fit_selected_single_peer"]
                        ),
                        "trackwise_fit_selected_single_peer": peers[trackwise_single_index],
                        "weight_l1_difference": float(
                            np.sum(np.abs(weights - trackwise_projection.weights))
                        ),
                        "weight_l2_difference": float(
                            np.linalg.norm(weights - trackwise_projection.weights)
                        ),
                        "mean_absolute_projected_response_difference": float(
                            np.mean(np.abs(primary_prediction - trackwise_prediction))
                        ),
                        "primary_weights": weights.tolist(),
                        "trackwise_weights": trackwise_projection.weights.tolist(),
                    }
                )

    track_path = PHYSICAL_ROOT / "outputs/analysis/track_specific_pier.parquet"
    sensitivity_path = PHYSICAL_ROOT / "outputs/analysis/track_aggregation_sensitivity.parquet"
    atomic_parquet(track_path, pd.DataFrame(track_rows))
    atomic_parquet(sensitivity_path, pd.DataFrame(sensitivity_rows))
    return {
        "outputs": [track_path, sensitivity_path],
        "track_rows": len(track_rows),
        "sensitivity_rows": len(sensitivity_rows),
    }
