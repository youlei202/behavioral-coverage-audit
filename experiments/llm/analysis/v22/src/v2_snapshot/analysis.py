from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .solver import ProjectionResult, fit_simplex_projection
from .utils import stable_u64


@dataclass(frozen=True)
class Split:
    seed: int
    fitting_ids: frozenset[str]
    evaluation_ids: frozenset[str]


def stratified_question_split(selected: pd.DataFrame, seed: int) -> Split:
    required = {"base_question_id", "category"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Missing split columns: {sorted(required.difference(selected.columns))}")
    fitting: set[str] = set()
    evaluation: set[str] = set()
    unique = selected[list(required)].drop_duplicates()
    for category, group in unique.groupby("category", sort=True):
        ids = sorted(
            group["base_question_id"].astype(str),
            key=lambda value: (stable_u64(seed, category, value), value),
        )
        if len(ids) % 2:
            raise ValueError(f"Category {category!r} has odd question count {len(ids)}")
        midpoint = len(ids) // 2
        fitting.update(ids[:midpoint])
        evaluation.update(ids[midpoint:])
    if fitting.intersection(evaluation) or fitting.union(evaluation) != set(
        unique["base_question_id"].astype(str)
    ):
        raise RuntimeError("Base-question split leakage or loss")
    return Split(seed, frozenset(fitting), frozenset(evaluation))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def fit_clean_temperature(clean_scores: pd.DataFrame, fitting_ids: Iterable[str]) -> float:
    fitting_set = set(fitting_ids)
    subset = clean_scores[
        clean_scores["base_question_id"].astype(str).isin(fitting_set)
        & clean_scores["condition"].eq("clean")
    ]
    if subset.empty:
        raise ValueError("No clean fitting rows for temperature calibration")
    logits = [np.asarray(value, dtype=np.float64) for value in subset["candidate_log_likelihoods"]]
    gold = subset["answer_index"].astype(int).tolist()

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        losses = []
        for values, answer_index in zip(logits, gold, strict=True):
            probabilities = _softmax(values / temperature)
            losses.append(-math.log(max(float(probabilities[answer_index]), 1e-300)))
        return float(np.mean(losses))

    result = minimize_scalar(
        nll,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"Temperature fitting failed: {result.message}")
    return float(math.exp(float(result.x)))


def response_from_row(
    row: pd.Series,
    representation: str,
    temperature: float = 1.0,
) -> float | np.ndarray:
    log_likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
    probabilities = _softmax(log_likelihoods / temperature)
    mapping = [int(value) for value in row["presented_to_original"]]
    semantic = np.empty_like(probabilities)
    for presented_index, original_index in enumerate(mapping):
        semantic[original_index] = probabilities[presented_index]
    semantic_gold = int(row["semantic_answer_index"])
    if representation == "gold_probability":
        return float(semantic[semantic_gold])
    if representation == "margin":
        presented_gold = int(row["answer_index"])
        wrong = np.delete(log_likelihoods / temperature, presented_gold)
        return float(log_likelihoods[presented_gold] / temperature - wrong.max())
    if representation == "probability_vector":
        return semantic
    raise ValueError(f"Unknown response representation {representation!r}")


def aggregate_family_responses(
    scores: pd.DataFrame,
    family: str,
    representation: str,
    temperatures: dict[str, float] | None = None,
) -> pd.DataFrame:
    subset = scores[scores["condition"].eq("clean") | scores["family"].eq(family)].copy()
    if subset.empty:
        raise ValueError(f"No rows for family {family!r}")
    temperatures = temperatures or {}
    subset["response"] = [
        response_from_row(row, representation, temperatures.get(str(row["model_id"]), 1.0))
        for _, row in subset.iterrows()
    ]
    keys = ["model_id", "base_question_id", "category", "dose"]

    def average(values: pd.Series) -> float | np.ndarray:
        array = np.stack([np.asarray(value, dtype=np.float64) for value in values])
        result = np.mean(array, axis=0)
        return float(result) if result.ndim == 0 else result

    aggregated = subset.groupby(keys, sort=True, as_index=False).agg(response=("response", average))
    counts = subset.groupby(keys, sort=True).size()
    nonzero_counts = counts[counts.index.get_level_values("dose").astype(float) != 0.0]
    if not nonzero_counts.empty and not nonzero_counts.eq(3).all():
        raise ValueError("Non-zero dose rows do not have exactly three intervention tracks")
    clean_counts = counts[counts.index.get_level_values("dose").astype(float) == 0.0]
    if not clean_counts.eq(1).all():
        raise ValueError("Clean design contains duplicate/missing rows")
    return aggregated


def aligned_response_arrays(
    aggregated: pd.DataFrame,
    target: str,
    peers: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    target_rows = (
        aggregated[aggregated["model_id"].eq(target)]
        .sort_values(["base_question_id", "dose"])
        .reset_index(drop=True)
    )
    if target_rows.empty:
        raise ValueError(f"Missing target rows for {target}")
    keys = pd.MultiIndex.from_frame(target_rows[["base_question_id", "dose"]])
    target_arrays = [
        np.asarray(value, dtype=np.float64) for value in target_rows["response"]
    ]
    response_ranks = {value.ndim for value in target_arrays}
    if response_ranks == {0}:
        target_values = np.stack(target_arrays)
        target_lengths: np.ndarray | None = None
    elif response_ranks == {1}:
        target_lengths = np.asarray([value.size for value in target_arrays], dtype=np.int64)
        if np.any(target_lengths == 0):
            raise ValueError(f"Target {target} has an empty response vector")
        target_values = np.zeros((len(target_arrays), int(target_lengths.max())))
        for index, value in enumerate(target_arrays):
            target_values[index, : value.size] = value
    else:
        raise ValueError(
            f"Target {target} has inconsistent response ranks {sorted(response_ranks)}"
        )
    peer_values: list[np.ndarray] = []
    for peer in peers:
        peer_rows = aggregated[aggregated["model_id"].eq(peer)].set_index(
            ["base_question_id", "dose"]
        )
        if not keys.isin(peer_rows.index).all():
            raise ValueError(f"Peer {peer} lacks aligned rows for target {target}")
        values = peer_rows.loc[keys, "response"]
        arrays = [np.asarray(value, dtype=np.float64) for value in values]
        if target_lengths is None:
            if any(value.ndim != 0 for value in arrays):
                raise ValueError(
                    f"Peer {peer} response rank differs from scalar target {target}"
                )
            peer_values.append(np.stack(arrays))
            continue
        lengths = np.asarray([value.size for value in arrays], dtype=np.int64)
        if any(value.ndim != 1 for value in arrays) or not np.array_equal(
            lengths, target_lengths
        ):
            raise ValueError(
                f"Peer {peer} response-vector lengths differ from target {target}"
            )
        padded = np.zeros_like(target_values)
        for index, value in enumerate(arrays):
            padded[index, : value.size] = value
        peer_values.append(padded)
    if target_values.ndim == 1:
        design = np.column_stack(peer_values)
    else:
        design = np.stack(peer_values, axis=-1)
    return target_rows[["base_question_id", "category", "dose"]], target_values, design


def _fit_projection(design: np.ndarray, target: np.ndarray) -> ProjectionResult:
    if design.ndim == 3:
        return fit_simplex_projection(design.reshape(-1, design.shape[-1]), target.reshape(-1))
    return fit_simplex_projection(design, target)


def _weight_geometry(weights: np.ndarray) -> dict[str, float | int]:
    positive = weights[weights > 0]
    entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
    ordered = np.sort(weights)[::-1]
    return {
        "top1_weight_mass": float(ordered[:1].sum()),
        "top3_weight_mass": float(ordered[:3].sum()),
        "effective_peer_count": float(1.0 / np.square(weights).sum()),
        "weight_entropy": entropy,
        "support_size_1e-3": int(np.sum(weights >= 1e-3)),
        "support_size_1e-2": int(np.sum(weights >= 1e-2)),
    }


def _vector_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    target = np.clip(target, 1e-15, 1.0)
    predicted = np.clip(predicted, 1e-15, 1.0)
    target /= target.sum(axis=1, keepdims=True)
    predicted /= predicted.sum(axis=1, keepdims=True)
    midpoint = 0.5 * (target + predicted)
    js = 0.5 * np.sum(target * np.log(target / midpoint), axis=1)
    js += 0.5 * np.sum(predicted * np.log(predicted / midpoint), axis=1)
    target_top = np.argmax(target, axis=1)
    predicted_top = np.argmax(predicted, axis=1)
    return {
        "mean_total_variation": float(np.mean(0.5 * np.abs(target - predicted).sum(axis=1))),
        "mean_js_divergence": float(np.mean(js)),
        "top1_agreement": float(np.mean(target_top == predicted_top)),
    }


def fit_target_family(
    aggregated: pd.DataFrame,
    target: str,
    peers: list[str],
    split: Split,
    *,
    family: str,
    representation: str,
    peer_set_condition: str = "all_peers",
    epsilon: float = 1e-10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ProjectionResult]:
    metadata, target_values, design = aligned_response_arrays(aggregated, target, peers)
    fitting_mask = metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    evaluation_mask = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
    if not fitting_mask.any() or not evaluation_mask.any():
        raise ValueError("Empty fitting/evaluation design")
    projection = _fit_projection(design[fitting_mask], target_values[fitting_mask])
    weights = projection.weights
    fitted_prediction = np.tensordot(design[fitting_mask], weights, axes=([-1], [0]))
    geometry = _weight_geometry(weights)
    rows: list[dict[str, Any]] = []

    if target_values.ndim == 1:
        fit_peer_mae = np.mean(
            np.abs(design[fitting_mask] - target_values[fitting_mask, None]), axis=0
        )
        selected_peer_index = int(np.argmin(fit_peer_mae))
        ridge = Ridge(alpha=1e-6, fit_intercept=True)
        ridge.fit(design[fitting_mask], target_values[fitting_mask])
        dose_values = sorted(float(value) for value in metadata.loc[evaluation_mask, "dose"].unique())
        for dose in dose_values + [math.nan]:
            if math.isnan(dose):
                selected = evaluation_mask
            else:
                selected = evaluation_mask & metadata["dose"].eq(dose).to_numpy()
            truth = target_values[selected]
            prediction = np.tensordot(design[selected], weights, axes=([-1], [0]))
            pier = float(np.mean(np.abs(truth - prediction)))
            selected_error = float(
                np.mean(np.abs(truth - design[selected, selected_peer_index]))
            )
            uniform_error = float(np.mean(np.abs(truth - np.mean(design[selected], axis=1))))
            oracle_error = float(np.mean(np.min(np.abs(design[selected] - truth[:, None]), axis=1)))
            ridge_error = float(np.mean(np.abs(truth - ridge.predict(design[selected]))))
            if math.isnan(dose):
                dose_specific_error = math.nan
            else:
                fit_dose = fitting_mask & metadata["dose"].eq(dose).to_numpy()
                dose_projection = _fit_projection(design[fit_dose], target_values[fit_dose])
                dose_prediction = design[selected] @ dose_projection.weights
                dose_specific_error = float(np.mean(np.abs(truth - dose_prediction)))
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "dose": dose,
                    "representation": representation,
                    "peer_set_condition": peer_set_condition,
                    "split_seed": split.seed,
                    "evaluation_question_count": len(set(metadata.loc[selected, "base_question_id"])),
                    "evaluation_design_row_count": int(selected.sum()),
                    "pier": pier,
                    "fit_selected_single_peer": peers[selected_peer_index],
                    "fit_selected_single_error": selected_error,
                    "relative_coverage_ratio": pier / (selected_error + epsilon),
                    "honest_convexity_gap": selected_error / (pier + epsilon),
                    "uniform_peer_error": uniform_error,
                    "dose_specific_convex_error": dose_specific_error,
                    "affine_ridge_error": ridge_error,
                    "oracle_single_peer_error": oracle_error,
                    "fitting_pier": float(np.mean(np.abs(target_values[fitting_mask] - fitted_prediction))),
                    **geometry,
                    **projection.diagnostics(),
                }
            )
    else:
        dose_values = sorted(float(value) for value in metadata.loc[evaluation_mask, "dose"].unique())
        for dose in dose_values + [math.nan]:
            selected = (
                evaluation_mask
                if math.isnan(dose)
                else evaluation_mask & metadata["dose"].eq(dose).to_numpy()
            )
            truth = target_values[selected]
            prediction = np.tensordot(design[selected], weights, axes=([-1], [0]))
            metrics = _vector_metrics(truth, prediction)
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "dose": dose,
                    "representation": representation,
                    "peer_set_condition": peer_set_condition,
                    "split_seed": split.seed,
                    "evaluation_question_count": len(set(metadata.loc[selected, "base_question_id"])),
                    "evaluation_design_row_count": int(selected.sum()),
                    **metrics,
                    **geometry,
                    **projection.diagnostics(),
                }
            )
    weight_rows = [
        {
            "target": target,
            "peer": peer,
            "family": family,
            "representation": representation,
            "peer_set_condition": peer_set_condition,
            "split_seed": split.seed,
            "weight": float(weight),
            **projection.diagnostics(),
        }
        for peer, weight in zip(peers, weights, strict=True)
    ]
    return rows, weight_rows, projection


def evaluate_fixed_vector_weights(
    aggregated_vectors: pd.DataFrame,
    target: str,
    peers: list[str],
    split: Split,
    weights: np.ndarray,
) -> dict[str, float]:
    metadata, target_values, design = aligned_response_arrays(aggregated_vectors, target, peers)
    evaluation = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
    predicted = np.tensordot(design[evaluation], weights, axes=([-1], [0]))
    return _vector_metrics(target_values[evaluation], predicted)


def design_transfer(
    source: pd.DataFrame,
    destination: pd.DataFrame,
    target: str,
    peers: list[str],
    split: Split,
    *,
    source_selector: Any | None = None,
    destination_selector: Any | None = None,
) -> dict[str, Any]:
    source_meta, source_y, source_x = aligned_response_arrays(source, target, peers)
    destination_meta, destination_y, destination_x = aligned_response_arrays(
        destination, target, peers
    )
    source_fit = source_meta["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    destination_fit = destination_meta["base_question_id"].astype(str).isin(
        split.fitting_ids
    ).to_numpy()
    destination_eval = destination_meta["base_question_id"].astype(str).isin(
        split.evaluation_ids
    ).to_numpy()
    if source_selector is not None:
        source_fit &= np.asarray(source_selector(source_meta), dtype=bool)
    if destination_selector is not None:
        selector = np.asarray(destination_selector(destination_meta), dtype=bool)
        destination_fit &= selector
        destination_eval &= selector
    source_projection = _fit_projection(source_x[source_fit], source_y[source_fit])
    destination_projection = _fit_projection(
        destination_x[destination_fit], destination_y[destination_fit]
    )
    source_prediction = np.tensordot(
        destination_x[destination_eval], source_projection.weights, axes=([-1], [0])
    )
    same_prediction = np.tensordot(
        destination_x[destination_eval], destination_projection.weights, axes=([-1], [0])
    )
    if destination_y.ndim == 1:
        transferred = float(np.mean(np.abs(destination_y[destination_eval] - source_prediction)))
        same = float(np.mean(np.abs(destination_y[destination_eval] - same_prediction)))
    else:
        transferred = _vector_metrics(destination_y[destination_eval], source_prediction)[
            "mean_total_variation"
        ]
        same = _vector_metrics(destination_y[destination_eval], same_prediction)[
            "mean_total_variation"
        ]
    return {
        "target": target,
        "split_seed": split.seed,
        "transferred_error": transferred,
        "same_design_error": same,
        "transfer_penalty": transferred - same,
        "source_weights": source_projection.weights,
        "destination_weights": destination_projection.weights,
    }


def clustered_bootstrap_with_refitting(
    metadata_fit: pd.DataFrame,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    metadata_evaluation: pd.DataFrame,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    fitting_ids = sorted(metadata_fit["base_question_id"].astype(str).unique())
    evaluation_ids = sorted(metadata_evaluation["base_question_id"].astype(str).unique())
    fit_indices = defaultdict(list)
    evaluation_indices = defaultdict(list)
    for index, value in enumerate(metadata_fit["base_question_id"].astype(str)):
        fit_indices[value].append(index)
    for index, value in enumerate(metadata_evaluation["base_question_id"].astype(str)):
        evaluation_indices[value].append(index)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled_fit = rng.choice(fitting_ids, size=len(fitting_ids), replace=True)
        sampled_evaluation = rng.choice(
            evaluation_ids, size=len(evaluation_ids), replace=True
        )
        fit_rows = np.concatenate([fit_indices[value] for value in sampled_fit])
        evaluation_rows = np.concatenate(
            [evaluation_indices[value] for value in sampled_evaluation]
        )
        projection = _fit_projection(x_fit[fit_rows], y_fit[fit_rows])
        prediction = np.tensordot(
            x_evaluation[evaluation_rows], projection.weights, axes=([-1], [0])
        )
        if y_evaluation.ndim == 1:
            metric = float(np.mean(np.abs(y_evaluation[evaluation_rows] - prediction)))
        else:
            metric = _vector_metrics(y_evaluation[evaluation_rows], prediction)[
                "mean_total_variation"
            ]
        rows.append(
            {
                "replicate": replicate,
                "seed": seed,
                "bootstrap_unit": "base_question_id",
                "fitting_draws": len(sampled_fit),
                "evaluation_draws": len(sampled_evaluation),
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)


def run_synthetic_controls(seed: int = 20260828) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    peer_vectors = rng.dirichlet(np.ones(6), size=(240, 5))
    peers = np.transpose(peer_vectors, (0, 2, 1))
    scalar = peers[:, 0, :]
    controls: list[dict[str, Any]] = []

    def record(name: str, x: np.ndarray, y: np.ndarray, threshold_mean: float, threshold_max: float) -> None:
        projection = _fit_projection(x[:120], y[:120])
        prediction = np.tensordot(x[120:], projection.weights, axes=([-1], [0]))
        if y.ndim == 1:
            residuals = np.abs(y[120:] - prediction)
        else:
            residuals = 0.5 * np.abs(y[120:] - prediction).sum(axis=1)
        mean_residual = float(np.mean(residuals))
        max_residual = float(np.max(residuals))
        controls.append(
            {
                "control": name,
                "observed_mean_residual": mean_residual,
                "observed_max_residual": max_residual,
                "threshold_mean": threshold_mean,
                "threshold_max": threshold_max,
                "passed": mean_residual <= threshold_mean and max_residual <= threshold_max,
                "weights": projection.weights.tolist(),
            }
        )

    record("exact_clone_scalar", scalar, scalar[:, 2], 1e-9, 1e-7)
    mixture = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    record("known_sparse_mixture_scalar", scalar, scalar @ mixture, 1e-9, 1e-7)
    boundary = np.array([0.75, 0.25, 0.0, 0.0, 0.0])
    record("boundary_mixture_scalar", scalar, scalar @ boundary, 1e-9, 1e-7)
    record("exact_clone_vector", peers, peers[:, :, 2], 1e-9, 1e-7)
    vector_mixture = np.tensordot(peers, mixture, axes=([-1], [0]))
    record("known_sparse_mixture_vector", peers, vector_mixture, 1e-9, 1e-7)
    duplicated = np.column_stack([scalar[:, :4], scalar[:, 2]])
    projection = _fit_projection(duplicated[:120], scalar[:120, 2])
    prediction = duplicated[120:] @ projection.weights
    controls.append(
        {
            "control": "duplicated_peer_ambiguity",
            "observed_mean_residual": float(np.mean(np.abs(scalar[120:, 2] - prediction))),
            "observed_max_residual": float(np.max(np.abs(scalar[120:, 2] - prediction))),
            "threshold_mean": 1e-9,
            "threshold_max": 1e-7,
            "passed": bool(np.max(np.abs(scalar[120:, 2] - prediction)) <= 1e-7),
            "weights": projection.weights.tolist(),
        }
    )
    frame = pd.DataFrame(controls)
    summary = {
        "passed": bool(frame["passed"].all()),
        "control_count": len(frame),
        "controls": frame.to_dict(orient="records"),
    }
    return frame, summary


def option_permutation_audit(scores: pd.DataFrame) -> pd.DataFrame:
    clean = scores[scores["condition"].eq("clean")].set_index(
        ["model_id", "base_question_id"]
    )
    rows: list[dict[str, Any]] = []
    permuted = scores[scores["family"].eq("option_permutation")]
    for (model_id, base_id), group in permuted.groupby(["model_id", "base_question_id"]):
        baseline = clean.loc[(model_id, base_id)]
        clean_probs = np.asarray(baseline["semantic_probabilities"], dtype=np.float64)
        clean_top = int(np.argmax(clean_probs))
        gold = int(baseline["semantic_answer_index"])
        for _, row in group.iterrows():
            probabilities = np.asarray(row["semantic_probabilities"], dtype=np.float64)
            rows.append(
                {
                    "model_id": model_id,
                    "base_question_id": base_id,
                    "permutation_index": int(row["permutation_index"]),
                    "probability_vector_tv": float(0.5 * np.abs(clean_probs - probabilities).sum()),
                    "top1_semantic_agreement": int(np.argmax(probabilities)) == clean_top,
                    "gold_probability_change": float(probabilities[gold] - clean_probs[gold]),
                }
            )
    return pd.DataFrame(rows)


def split_rank_stability(results: pd.DataFrame, metric: str = "pier") -> pd.DataFrame:
    overall = results[results["dose"].isna() & results["peer_set_condition"].eq("all_peers")]
    rows: list[dict[str, Any]] = []
    group_columns = ["family", "representation"]
    for keys, group in overall.groupby(group_columns):
        pivot = group.pivot(index="target", columns="split_seed", values=metric)
        correlations = []
        for left, right in itertools.combinations(pivot.columns, 2):
            correlation = spearmanr(pivot[left], pivot[right], nan_policy="omit").statistic
            correlations.append(float(correlation))
        rows.append(
            {
                "family": keys[0],
                "representation": keys[1],
                "pairwise_correlation_count": len(correlations),
                "median_rank_correlation": float(np.nanmedian(correlations)),
                "minimum_rank_correlation": float(np.nanmin(correlations)),
                "mean_absolute_pier_change": float(
                    np.nanmean(np.abs(pivot.to_numpy() - pivot.mean(axis=1).to_numpy()[:, None]))
                ),
            }
        )
    return pd.DataFrame(rows)
