from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .data import (
    FAMILIES,
    AlignedData,
    aligned_track_arrays,
    design_row_weights,
    fit_clean_temperature,
    fit_projection,
    fixed_splits,
    high_dose,
    load_validated_inputs,
    scalar_dose_metrics,
    semantic_probabilities,
    track_response_frame,
    vector_dose_metrics,
    weighted_peer_mae,
)
from .point_analysis import sibling_for
from .utils import PHYSICAL_ROOT, atomic_parquet


def _fit_scalar_point(data: AlignedData, split: Any) -> dict[str, Any]:
    fitting = data.metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    row_weights = design_row_weights(data.metadata)
    fitted = fit_projection(
        data.design[fitting], data.target[fitting], row_weights[fitting]
    )
    weights = fitted.projection.weights
    peer_errors = weighted_peer_mae(data, split.fitting_ids)
    selected_index = int(np.argmin(peer_errors))
    single_weights = np.zeros(len(data.peers), dtype=np.float64)
    single_weights[selected_index] = 1.0
    dose = scalar_dose_metrics(data, weights, split.evaluation_ids)
    single = scalar_dose_metrics(data, single_weights, split.evaluation_ids)
    return {
        "weights": weights,
        "dose": dose,
        "overall": float(dose["trackwise_pier"].mean()),
        "single_dose": single,
        "single_overall": float(single["trackwise_pier"].mean()),
        "selected_index": selected_index,
        "selected_peer": data.peers[selected_index],
    }


def _removed_scalar_overall(data: AlignedData, split: Any, removed_peer: str) -> float:
    retained = [index for index, peer in enumerate(data.peers) if peer != removed_peer]
    design = data.design[:, retained]
    reduced = AlignedData(
        data.metadata,
        data.target,
        design,
        tuple(data.peers[index] for index in retained),
    )
    fitting = reduced.metadata["base_question_id"].astype(str).isin(
        split.fitting_ids
    ).to_numpy()
    row_weights = design_row_weights(reduced.metadata)
    fit = fit_projection(
        reduced.design[fitting], reduced.target[fitting], row_weights[fitting]
    )
    dose = scalar_dose_metrics(reduced, fit.projection.weights, split.evaluation_ids)
    return float(dose["trackwise_pier"].mean())


def run_calibration_sensitivity() -> dict[str, Any]:
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    raw_dose = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    )
    raw_overall = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_overall_results.parquet"
    )
    raw_sibling = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_sibling_removal.parquet"
    )
    rows: list[dict[str, Any]] = []

    for split in splits.values():
        temperatures = {
            model: fit_clean_temperature(
                scores[scores["model_id"].eq(model)], split.fitting_ids
            )
            for model in model_ids
        }
        for family in FAMILIES:
            frame = track_response_frame(scores, family, temperatures=temperatures)
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                data = aligned_track_arrays(frame, target, peers)
                calibrated = _fit_scalar_point(data, split)
                calibrated_dose = calibrated["dose"].set_index("dose")
                calibrated_single = calibrated["single_dose"].set_index("dose")
                calibrated_clean = float(
                    calibrated_dose.loc[0.0, "trackwise_pier"]
                )
                calibrated_high = float(
                    calibrated_dose.loc[high_dose(family), "trackwise_pier"]
                )
                calibrated_endpoint = calibrated_high - calibrated_clean
                sibling = sibling_for(target)
                calibrated_sibling_inflation = math.nan
                if sibling is not None:
                    removed = _removed_scalar_overall(data, split, sibling)
                    calibrated_sibling_inflation = removed - calibrated["overall"]
                raw_target = raw_dose[
                    raw_dose["target"].eq(target)
                    & raw_dose["family"].eq(family)
                    & raw_dose["split_seed"].eq(split.seed)
                ].set_index("dose")
                raw_endpoint = float(
                    raw_target.loc[high_dose(family), "trackwise_pier"]
                    - raw_target.loc[0.0, "trackwise_pier"]
                )
                raw_overall_row = raw_overall[
                    raw_overall["target"].eq(target)
                    & raw_overall["family"].eq(family)
                    & raw_overall["split_seed"].eq(split.seed)
                ].iloc[0]
                raw_sibling_inflation = math.nan
                if sibling is not None:
                    raw_sibling_inflation = float(
                        raw_sibling[
                            raw_sibling["target"].eq(target)
                            & raw_sibling["family"].eq(family)
                            & raw_sibling["split_seed"].eq(split.seed)
                            & raw_sibling["evaluation_scope"].eq(
                                "overall_equal_dose"
                            )
                        ].iloc[0]["inflation"]
                    )
                calibrated_convexity = (
                    calibrated["single_overall"] - calibrated["overall"]
                )
                for dose in calibrated_dose.index:
                    rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "dose": float(dose),
                            "raw_trackwise_pier": float(
                                raw_target.loc[dose, "trackwise_pier"]
                            ),
                            "calibrated_trackwise_pier": float(
                                calibrated_dose.loc[dose, "trackwise_pier"]
                            ),
                            "raw_endpoint_delta": raw_endpoint,
                            "calibrated_endpoint_delta": calibrated_endpoint,
                            "endpoint_sign_agreement": bool(
                                np.sign(raw_endpoint)
                                == np.sign(calibrated_endpoint)
                            ),
                            "raw_convexity_improvement": float(
                                raw_overall_row["absolute_improvement"]
                            ),
                            "calibrated_convexity_improvement": calibrated_convexity,
                            "convexity_survives": bool(calibrated_convexity > 0),
                            "raw_sibling_removal_inflation": raw_sibling_inflation,
                            "calibrated_sibling_removal_inflation": calibrated_sibling_inflation,
                            "sibling_removal_survives": bool(
                                calibrated_sibling_inflation > 0
                            )
                            if sibling is not None
                            else False,
                            "fit_selected_single_peer": calibrated["selected_peer"],
                            "calibrated_single_error_trackwise": float(
                                calibrated_single.loc[dose, "trackwise_pier"]
                            ),
                            "target_temperature": temperatures[target],
                            "peer_ids": peers,
                            "peer_temperatures": [temperatures[peer] for peer in peers],
                            "calibrated_weights": calibrated["weights"].tolist(),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame["raw_rank"] = frame.groupby(["family", "split_seed", "dose"])[
        "raw_trackwise_pier"
    ].rank(method="average")
    frame["calibrated_rank"] = frame.groupby(["family", "split_seed", "dose"])[
        "calibrated_trackwise_pier"
    ].rank(method="average")
    frame["rank_shift"] = frame["calibrated_rank"] - frame["raw_rank"]
    if len(frame) != 800:
        raise ValueError(f"Expected 800 calibrated rows, found {len(frame)}")
    path = (
        PHYSICAL_ROOT / "outputs/analysis/trackwise_calibrated_results.parquet"
    )
    atomic_parquet(path, frame.sort_values(["target", "family", "split_seed", "dose"]))
    return {"outputs": [path], "row_counts": {"calibrated_results": len(frame)}}


@dataclass(frozen=True)
class LabelBiasFit:
    model_id: str
    split_seed: int
    biases: dict[int, np.ndarray]
    diagnostics: pd.DataFrame
    fitting_question_ids: frozenset[str]


def _permutation_design(scores: pd.DataFrame, fitting_ids: Iterable[str]) -> pd.DataFrame:
    fitting = {str(identifier) for identifier in fitting_ids}
    subset = scores[
        scores["base_question_id"].astype(str).isin(fitting)
        & (
            scores["condition"].eq("clean")
            | scores["family"].eq("option_permutation")
        )
    ].copy()
    if subset.empty:
        raise ValueError("No clean/permutation rows for label-bias fitting")
    subset["option_count"] = subset["candidate_log_likelihoods"].map(len)
    if not subset.groupby("base_question_id").size().eq(4).all():
        raise ValueError("Every label-bias fitting question must have four presentations")
    return subset


def _fit_bias_for_option_count(
    group: pd.DataFrame, option_count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    questions = sorted(group["base_question_id"].astype(str).unique())
    question_index = {identifier: index for index, identifier in enumerate(questions)}
    semantic_count = len(questions) * (option_count - 1)
    bias_offset = semantic_count
    parameter_count = semantic_count + option_count - 1
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    observations: list[float] = []
    observation_index = 0
    ordered = group.sort_values(
        ["base_question_id", "condition", "permutation_index"]
    )
    for _, row in ordered.iterrows():
        likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
        centered = likelihoods - float(np.mean(likelihoods))
        mapping = [int(value) for value in row["presented_to_original"]]
        question_offset = question_index[str(row["base_question_id"])] * (
            option_count - 1
        )
        for presented_position, semantic_position in enumerate(mapping):
            if semantic_position < option_count - 1:
                row_indices.append(observation_index)
                column_indices.append(question_offset + semantic_position)
                values.append(1.0)
            else:
                for index in range(option_count - 1):
                    row_indices.append(observation_index)
                    column_indices.append(question_offset + index)
                    values.append(-1.0)
            if presented_position < option_count - 1:
                row_indices.append(observation_index)
                column_indices.append(bias_offset + presented_position)
                values.append(1.0)
            else:
                for index in range(option_count - 1):
                    row_indices.append(observation_index)
                    column_indices.append(bias_offset + index)
                    values.append(-1.0)
            observations.append(float(centered[presented_position]))
            observation_index += 1
    design = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(observations), parameter_count),
        dtype=np.float64,
    ).tocsr()
    solution = lsqr(
        design,
        np.asarray(observations, dtype=np.float64),
        atol=1e-12,
        btol=1e-12,
        iter_lim=1000,
    )
    coefficients = np.asarray(solution[0], dtype=np.float64)
    bias = np.empty(option_count, dtype=np.float64)
    bias[:-1] = coefficients[bias_offset:]
    bias[-1] = -float(np.sum(bias[:-1]))
    bias -= float(np.mean(bias))
    diagnostics = {
        "option_count": option_count,
        "fitting_question_count": len(questions),
        "observation_count": len(observations),
        "parameter_count": parameter_count,
        "lsqr_stop_code": int(solution[1]),
        "lsqr_iterations": int(solution[2]),
        "residual_norm": float(solution[3]),
        "condition_estimate": float(solution[6]),
        "converged": bool(int(solution[1]) in {1, 2}),
        "estimable": True,
        "sum_to_zero_error": abs(float(np.sum(bias))),
    }
    if diagnostics["sum_to_zero_error"] > 1e-10:
        raise RuntimeError("Additive label-bias fit violated its sum-to-zero constraint")
    return bias, diagnostics


def fit_label_bias(
    scores: pd.DataFrame, fitting_ids: Iterable[str], *, split_seed: int
) -> LabelBiasFit:
    if scores["model_id"].nunique() != 1:
        raise ValueError("fit_label_bias expects exactly one model")
    model_id = str(scores["model_id"].iloc[0])
    fitting_set = frozenset(str(identifier) for identifier in fitting_ids)
    subset = _permutation_design(scores, fitting_set)
    biases: dict[int, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    for option_count, group in subset.groupby("option_count", sort=True):
        bias, record = _fit_bias_for_option_count(group, int(option_count))
        biases[int(option_count)] = bias
        diagnostics.append(
            {"model_id": model_id, "split_seed": split_seed, **record}
        )
    all_counts = sorted(
        {len(value) for value in scores["candidate_log_likelihoods"]}
    )
    for option_count in all_counts:
        if option_count in biases:
            continue
        biases[option_count] = np.zeros(option_count, dtype=np.float64)
        diagnostics.append(
            {
                "model_id": model_id,
                "split_seed": split_seed,
                "option_count": option_count,
                "fitting_question_count": 0,
                "observation_count": 0,
                "parameter_count": 0,
                "lsqr_stop_code": 0,
                "lsqr_iterations": 0,
                "residual_norm": math.nan,
                "condition_estimate": math.nan,
                "converged": False,
                "estimable": False,
                "sum_to_zero_error": 0.0,
            }
        )
    return LabelBiasFit(
        model_id,
        split_seed,
        biases,
        pd.DataFrame(diagnostics),
        fitting_set,
    )


def _heldout_bias_validation(
    model_scores: pd.DataFrame, evaluation_ids: Iterable[str], fit: LabelBiasFit
) -> dict[str, Any]:
    evaluation = {str(identifier) for identifier in evaluation_ids}
    if fit.fitting_question_ids.intersection(evaluation):
        raise ValueError("Held-out label-bias IDs entered fitting")
    subset = model_scores[model_scores["base_question_id"].astype(str).isin(evaluation)]
    canonical = subset[subset["condition"].eq("clean")].set_index(
        "base_question_id"
    )
    permuted = subset[subset["family"].eq("option_permutation")]
    before: list[float] = []
    after: list[float] = []
    for _, row in permuted.iterrows():
        baseline = canonical.loc[str(row["base_question_id"])]
        count = len(row["candidate_log_likelihoods"])
        raw_baseline = semantic_probabilities(baseline)
        raw_permuted = semantic_probabilities(row)
        corrected_baseline = semantic_probabilities(
            baseline, label_bias=fit.biases[count]
        )
        corrected_permuted = semantic_probabilities(
            row, label_bias=fit.biases[count]
        )
        before.append(float(0.5 * np.abs(raw_baseline - raw_permuted).sum()))
        after.append(
            float(0.5 * np.abs(corrected_baseline - corrected_permuted).sum())
        )
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    return {
        "model_id": fit.model_id,
        "split_seed": fit.split_seed,
        "evaluation_question_count": len(evaluation),
        "permutation_comparison_count": len(before),
        "before_mean_tv": float(np.mean(before_array)),
        "after_mean_tv": float(np.mean(after_array)),
        "before_median_tv": float(np.median(before_array)),
        "after_median_tv": float(np.median(after_array)),
        "before_95th_percentile_tv": float(np.quantile(before_array, 0.95)),
        "after_95th_percentile_tv": float(np.quantile(after_array, 0.95)),
        "mean_tv_reduction": float(np.mean(before_array) - np.mean(after_array)),
        "heldout_invariance_improved": bool(
            np.mean(after_array) < np.mean(before_array)
        ),
        "evaluation_ids_used_in_fitting": False,
    }


def _bias_estimate_rows(fits: dict[str, LabelBiasFit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = list("ABCDEFGHIJ")
    for fit in fits.values():
        diagnostics = fit.diagnostics.set_index("option_count")
        for option_count, bias in sorted(fit.biases.items()):
            diagnostic = diagnostics.loc[option_count].to_dict()
            for position, value in enumerate(bias):
                rows.append(
                    {
                        "model_id": fit.model_id,
                        "split_seed": fit.split_seed,
                        "option_count": option_count,
                        "label_position": position,
                        "label": labels[position],
                        "bias_estimate": float(value),
                        **diagnostic,
                    }
                )
    return rows


def run_interface_sensitivity() -> dict[str, Any]:
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    raw_dose = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    )
    raw_overall = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_overall_results.parquet"
    )
    raw_sibling = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_sibling_removal.parquet"
    )
    estimate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for split in splits.values():
        fits = {
            model: fit_label_bias(
                scores[scores["model_id"].eq(model)],
                split.fitting_ids,
                split_seed=split.seed,
            )
            for model in model_ids
        }
        estimate_rows.extend(_bias_estimate_rows(fits))
        validations = {
            model: _heldout_bias_validation(
                scores[scores["model_id"].eq(model)], split.evaluation_ids, fits[model]
            )
            for model in model_ids
        }
        validation_rows.extend(validations.values())
        label_biases = {model: fits[model].biases for model in model_ids}
        for family in FAMILIES:
            frame = track_response_frame(
                scores, family, label_biases=label_biases
            )
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                data = aligned_track_arrays(frame, target, peers)
                corrected = _fit_scalar_point(data, split)
                corrected_dose = corrected["dose"].set_index("dose")
                corrected_clean = float(
                    corrected_dose.loc[0.0, "trackwise_pier"]
                )
                corrected_high = float(
                    corrected_dose.loc[high_dose(family), "trackwise_pier"]
                )
                corrected_endpoint = corrected_high - corrected_clean
                corrected_convexity = corrected["single_overall"] - corrected["overall"]
                sibling = sibling_for(target)
                corrected_sibling = math.nan
                if sibling is not None:
                    corrected_sibling = (
                        _removed_scalar_overall(data, split, sibling)
                        - corrected["overall"]
                    )
                raw_target = raw_dose[
                    raw_dose["target"].eq(target)
                    & raw_dose["family"].eq(family)
                    & raw_dose["split_seed"].eq(split.seed)
                ].set_index("dose")
                raw_endpoint = float(
                    raw_target.loc[high_dose(family), "trackwise_pier"]
                    - raw_target.loc[0.0, "trackwise_pier"]
                )
                raw_overall_row = raw_overall[
                    raw_overall["target"].eq(target)
                    & raw_overall["family"].eq(family)
                    & raw_overall["split_seed"].eq(split.seed)
                ].iloc[0]
                raw_sibling_value = math.nan
                if sibling is not None:
                    raw_sibling_value = float(
                        raw_sibling[
                            raw_sibling["target"].eq(target)
                            & raw_sibling["family"].eq(family)
                            & raw_sibling["split_seed"].eq(split.seed)
                            & raw_sibling["evaluation_scope"].eq(
                                "overall_equal_dose"
                            )
                        ].iloc[0]["inflation"]
                    )
                for dose in corrected_dose.index:
                    result_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "dose": float(dose),
                            "raw_trackwise_pier": float(
                                raw_target.loc[dose, "trackwise_pier"]
                            ),
                            "interface_corrected_trackwise_pier": float(
                                corrected_dose.loc[dose, "trackwise_pier"]
                            ),
                            "raw_endpoint_delta": raw_endpoint,
                            "interface_corrected_endpoint_delta": corrected_endpoint,
                            "endpoint_sign_agreement": bool(
                                np.sign(raw_endpoint)
                                == np.sign(corrected_endpoint)
                            ),
                            "raw_convexity_improvement": float(
                                raw_overall_row["absolute_improvement"]
                            ),
                            "interface_corrected_convexity_improvement": corrected_convexity,
                            "convexity_survives": bool(corrected_convexity > 0),
                            "raw_sibling_removal_inflation": raw_sibling_value,
                            "interface_corrected_sibling_removal_inflation": corrected_sibling,
                            "sibling_removal_survives": bool(corrected_sibling > 0)
                            if sibling is not None
                            else False,
                            "fit_selected_single_peer": corrected["selected_peer"],
                            "interface_corrected_weights": corrected["weights"].tolist(),
                            "heldout_clean_permutation_tv_before": validations[target][
                                "before_mean_tv"
                            ],
                            "heldout_clean_permutation_tv_after": validations[target][
                                "after_mean_tv"
                            ],
                            "heldout_invariance_improved": validations[target][
                                "heldout_invariance_improved"
                            ],
                        }
                    )

    results = pd.DataFrame(result_rows)
    results["raw_rank"] = results.groupby(["family", "split_seed", "dose"])[
        "raw_trackwise_pier"
    ].rank(method="average")
    results["interface_corrected_rank"] = results.groupby(
        ["family", "split_seed", "dose"]
    )["interface_corrected_trackwise_pier"].rank(method="average")
    results["rank_shift"] = (
        results["interface_corrected_rank"] - results["raw_rank"]
    )
    if len(results) != 800:
        raise ValueError(f"Expected 800 interface-sensitivity rows, found {len(results)}")
    estimates = pd.DataFrame(estimate_rows)
    validation = pd.DataFrame(validation_rows)
    result_path = (
        PHYSICAL_ROOT
        / "outputs/analysis/trackwise_interface_sensitivity.parquet"
    )
    estimate_path = PHYSICAL_ROOT / "controls/label_bias_estimates.parquet"
    validation_path = (
        PHYSICAL_ROOT / "controls/label_bias_heldout_validation.parquet"
    )
    atomic_parquet(result_path, results.sort_values(["target", "family", "split_seed", "dose"]))
    atomic_parquet(estimate_path, estimates)
    atomic_parquet(validation_path, validation)
    return {
        "outputs": [result_path, estimate_path, validation_path],
        "row_counts": {
            "interface_results": len(results),
            "label_bias_estimates": len(estimates),
            "heldout_bias_validations": len(validation),
        },
        "warnings": [
            "The additive label-bias correction is exploratory and was not measured at every stress condition."
        ],
    }


def _vector_selected_peer(data: AlignedData, fitting_ids: Iterable[str]) -> int:
    fitting = data.metadata["base_question_id"].astype(str).isin(fitting_ids).to_numpy()
    row_weights = design_row_weights(data.metadata)[fitting]
    per_row_peer_tv = 0.5 * np.abs(
        data.design[fitting] - data.target[fitting, :, None]
    ).sum(axis=1)
    errors = np.average(per_row_peer_tv, axis=0, weights=row_weights)
    return int(np.argmin(errors))


def _vector_fit(data: AlignedData, split: Any) -> np.ndarray:
    fitting = data.metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    row_weights = design_row_weights(data.metadata)
    return fit_projection(
        data.design[fitting], data.target[fitting], row_weights[fitting]
    ).projection.weights


def _vector_overall(dose: pd.DataFrame) -> float:
    return float(dose["mean_total_variation"].mean())


def _vector_removed_overall(
    data: AlignedData, split: Any, removed_peer: str
) -> float:
    retained = [index for index, peer in enumerate(data.peers) if peer != removed_peer]
    reduced = AlignedData(
        data.metadata,
        data.target,
        data.design[:, :, retained],
        tuple(data.peers[index] for index in retained),
    )
    weights = _vector_fit(reduced, split)
    return _vector_overall(
        vector_dose_metrics(reduced, weights, split.evaluation_ids)
    )


def run_vector_sensitivity() -> dict[str, Any]:
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    scalar_saved = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_weights.parquet"
    )
    scalar_overall = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_overall_results.parquet"
    )
    scalar_endpoint = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_endpoint_effects.parquet"
    )
    scalar_sibling = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_sibling_removal.parquet"
    )
    rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        vector_frame = track_response_frame(
            scores, family, representation="probability_vector"
        )
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            vector_data = aligned_track_arrays(vector_frame, target, peers)
            for split in splits.values():
                scalar_subset = scalar_saved[
                    scalar_saved["target"].eq(target)
                    & scalar_saved["family"].eq(family)
                    & scalar_saved["split_seed"].eq(split.seed)
                ].set_index("peer")
                scalar_weights = scalar_subset.loc[peers, "weight"].to_numpy(
                    dtype=np.float64
                )
                vector_weights = _vector_fit(vector_data, split)
                selected_index = _vector_selected_peer(
                    vector_data, split.fitting_ids
                )
                single_weights = np.zeros(len(peers), dtype=np.float64)
                single_weights[selected_index] = 1.0
                single_dose = vector_dose_metrics(
                    vector_data, single_weights, split.evaluation_ids
                )
                single_overall = _vector_overall(single_dose)
                sibling = sibling_for(target)
                scalar_sibling_value = math.nan
                vector_sibling_value = math.nan
                if sibling is not None:
                    scalar_sibling_value = float(
                        scalar_sibling[
                            scalar_sibling["target"].eq(target)
                            & scalar_sibling["family"].eq(family)
                            & scalar_sibling["split_seed"].eq(split.seed)
                            & scalar_sibling["evaluation_scope"].eq(
                                "overall_equal_dose"
                            )
                        ].iloc[0]["inflation"]
                    )
                    all_vector_overall = _vector_overall(
                        vector_dose_metrics(
                            vector_data, vector_weights, split.evaluation_ids
                        )
                    )
                    vector_sibling_value = (
                        _vector_removed_overall(vector_data, split, sibling)
                        - all_vector_overall
                    )
                scalar_point = scalar_overall[
                    scalar_overall["target"].eq(target)
                    & scalar_overall["family"].eq(family)
                    & scalar_overall["split_seed"].eq(split.seed)
                ].iloc[0]
                scalar_delta = float(
                    scalar_endpoint[
                        scalar_endpoint["target"].eq(target)
                        & scalar_endpoint["family"].eq(family)
                        & scalar_endpoint["split_seed"].eq(split.seed)
                    ].iloc[0]["trackwise_delta"]
                )
                for weight_source, weights in (
                    ("scalar_fitted", scalar_weights),
                    ("vector_fitted", vector_weights),
                ):
                    dose = vector_dose_metrics(
                        vector_data, weights, split.evaluation_ids
                    )
                    overall_tv = _vector_overall(dose)
                    vector_delta = float(
                        dose.loc[
                            dose["dose"].eq(high_dose(family)),
                            "mean_total_variation",
                        ].iloc[0]
                        - dose.loc[
                            dose["dose"].eq(0.0), "mean_total_variation"
                        ].iloc[0]
                    )
                    convexity_improvement = single_overall - overall_tv
                    for _, metric in dose.iterrows():
                        rows.append(
                            {
                                "target": target,
                                "family": family,
                                "split_seed": split.seed,
                                "dose": float(metric["dose"]),
                                "weight_source": weight_source,
                                "mean_total_variation": float(
                                    metric["mean_total_variation"]
                                ),
                                "mean_jensen_shannon_divergence": float(
                                    metric["mean_jensen_shannon_divergence"]
                                ),
                                "top1_semantic_answer_agreement": float(
                                    metric["top1_semantic_answer_agreement"]
                                ),
                                "correctness_agreement": float(
                                    metric["correctness_agreement"]
                                ),
                                "vector_overall_total_variation": overall_tv,
                                "vector_endpoint_delta": vector_delta,
                                "scalar_endpoint_delta": scalar_delta,
                                "endpoint_direction_support": bool(
                                    np.sign(vector_delta) == np.sign(scalar_delta)
                                ),
                                "fit_selected_single_peer": peers[selected_index],
                                "single_peer_overall_total_variation": single_overall,
                                "vector_convexity_improvement": convexity_improvement,
                                "scalar_convexity_improvement": float(
                                    scalar_point["absolute_improvement"]
                                ),
                                "convexity_direction_support": bool(
                                    np.sign(convexity_improvement)
                                    == np.sign(
                                        float(scalar_point["absolute_improvement"])
                                    )
                                ),
                                "vector_sibling_removal_inflation": vector_sibling_value,
                                "scalar_sibling_removal_inflation": scalar_sibling_value,
                                "sibling_direction_support": bool(
                                    np.sign(vector_sibling_value)
                                    == np.sign(scalar_sibling_value)
                                )
                                if sibling is not None
                                else False,
                                "weights": weights.tolist(),
                                "scalar_weights": scalar_weights.tolist(),
                                "vector_weights": vector_weights.tolist(),
                                "scalar_vector_weight_l1_change": float(
                                    np.abs(scalar_weights - vector_weights).sum()
                                ),
                                "scalar_vector_weight_l2_change": float(
                                    np.linalg.norm(scalar_weights - vector_weights)
                                ),
                            }
                        )

    frame = pd.DataFrame(rows)
    if len(frame) != 1600:
        raise ValueError(f"Expected 1600 vector result rows, found {len(frame)}")
    path = PHYSICAL_ROOT / "outputs/analysis/trackwise_vector_results.parquet"
    atomic_parquet(
        path,
        frame.sort_values(
            ["target", "family", "split_seed", "weight_source", "dose"]
        ),
    )
    return {"outputs": [path], "row_counts": {"vector_results": len(frame)}}
