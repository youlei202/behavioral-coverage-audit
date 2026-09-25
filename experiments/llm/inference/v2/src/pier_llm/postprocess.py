from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    Split,
    aggregate_family_responses,
    aligned_response_arrays,
    clustered_bootstrap_with_refitting,
    design_transfer,
    evaluate_fixed_vector_weights,
    fit_clean_temperature,
    fit_target_family,
    option_permutation_audit,
    split_rank_stability,
    stratified_question_split,
)
from .inference import GENERATION_SCHEMA, SCORE_SCHEMA
from .packaging import build_results_package
from .plotting import make_all_figures
from .solver import feasible_weight_intervals, fit_simplex_projection
from .utils import (
    atomic_write_json,
    atomic_write_text,
    environment_basics,
    project_root,
    read_jsonl,
    sha256_file,
    stable_u64,
    tree_sha256,
    utc_now,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_and_verify_shards(root: Path, relative: str, schema: str) -> pd.DataFrame:
    paths = sorted((root / relative).glob("*/shard_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No inference shards under {root / relative}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        metadata_path = path.with_suffix(".meta.json")
        metadata = _load_json(metadata_path)
        if metadata["schema_version"] != schema or metadata["sha256"] != sha256_file(path):
            raise ValueError(f"Invalid shard metadata/checksum: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != metadata["row_count"] or not frame["schema_version"].eq(schema).all():
            raise ValueError(f"Invalid shard schema/count: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_validated_inference(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    complete_path = root / "status" / "B200_INFERENCE_COMPLETE.json"
    if not complete_path.exists():
        raise RuntimeError("B200_INFERENCE_COMPLETE.json is absent; post-processing is forbidden")
    complete = _load_json(complete_path)
    if not complete.get("complete"):
        raise RuntimeError("B200 inference marker is not complete")
    scores = _read_and_verify_shards(root, "outputs/raw_scores", SCORE_SCHEMA)
    generation = _read_and_verify_shards(
        root, "outputs/generation_validation", GENERATION_SCHEMA
    )
    resolved = _load_json(root / "configs" / "resolved_models.json")
    model_ids = [row["id"] for row in resolved["models"]]
    config = _load_json(root / "configs" / "experiment.json")
    expected_scores = len(model_ids) * int(config["scoring_prompts_per_model"])
    expected_generation = len(model_ids) * int(config["generation_prompts_per_model"])
    if len(scores) != expected_scores or len(generation) != expected_generation:
        raise ValueError(
            f"Validated row counts differ from expected: scores={len(scores)}/{expected_scores}, "
            f"generation={len(generation)}/{expected_generation}"
        )
    if scores.duplicated(["model_id", "prompt_id"]).any():
        raise ValueError("Duplicate raw score keys")
    if generation.duplicated(["model_id", "generation_prompt_id"]).any():
        raise ValueError("Duplicate generation keys")
    if set(scores["model_id"]) != set(model_ids) or set(generation["model_id"]) != set(model_ids):
        raise ValueError("Inference model roster differs from resolved roster")
    return scores, generation, resolved


def _primary_analyses(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, str, str], np.ndarray]]:
    scalar_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    scalar_weight_cache: dict[tuple[int, str, str], np.ndarray] = {}
    splits = {seed: stratified_question_split(selected, seed) for seed in split_seeds}
    scalar_aggregated = {
        (family, representation): aggregate_family_responses(scores, family, representation)
        for family in ("irrelevant_context", "content_deletion")
        for representation in ("gold_probability", "margin")
    }
    vector_aggregated = {
        family: aggregate_family_responses(scores, family, "probability_vector")
        for family in ("irrelevant_context", "content_deletion")
    }
    for representation in ("gold_probability", "margin"):
        for family in ("irrelevant_context", "content_deletion"):
            aggregated = scalar_aggregated[(family, representation)]
            for seed, split in splits.items():
                for target in model_ids:
                    peers = [model for model in model_ids if model != target]
                    rows, weights, projection = fit_target_family(
                        aggregated,
                        target,
                        peers,
                        split,
                        family=family,
                        representation=representation,
                    )
                    scalar_rows.extend(rows)
                    weight_rows.extend(weights)
                    if representation == "gold_probability":
                        scalar_weight_cache[(seed, family, target)] = projection.weights
    for family in ("irrelevant_context", "content_deletion"):
        aggregated = vector_aggregated[family]
        for seed, split in splits.items():
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                rows, weights, _projection = fit_target_family(
                    aggregated,
                    target,
                    peers,
                    split,
                    family=family,
                    representation="probability_vector",
                )
                for row in rows:
                    row["weight_fit"] = "vector_fitted"
                for row in weights:
                    row["weight_fit"] = "vector_fitted"
                vector_rows.extend(rows)
                weight_rows.extend(weights)
                scalar_weights = scalar_weight_cache[(seed, family, target)]
                metrics = evaluate_fixed_vector_weights(
                    aggregated, target, peers, split, scalar_weights
                )
                vector_rows.append(
                    {
                        "target": target,
                        "family": family,
                        "dose": math.nan,
                        "representation": "probability_vector",
                        "peer_set_condition": "all_peers",
                        "split_seed": seed,
                        "weight_fit": "gold_scalar_fitted",
                        **metrics,
                    }
                )
    return (
        pd.DataFrame(scalar_rows),
        pd.DataFrame(vector_rows),
        pd.DataFrame(weight_rows),
        scalar_weight_cache,
    )


def _scalar_error(
    aggregated: pd.DataFrame,
    target: str,
    peers: list[str],
    split: Split,
) -> tuple[float, np.ndarray]:
    metadata, target_values, design = aligned_response_arrays(aggregated, target, peers)
    fitting = metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    evaluation = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
    projection = fit_simplex_projection(design[fitting], target_values[fitting])
    predicted = design[evaluation] @ projection.weights
    return float(np.mean(np.abs(target_values[evaluation] - predicted))), projection.weights


def _peer_removal_controls(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_records: list[dict[str, Any]],
    split_seeds: list[int],
    replicates: int,
) -> pd.DataFrame:
    model_ids = [row["id"] for row in model_records]
    record_by_id = {row["id"]: row for row in model_records}
    aggregated_by_family = {
        family: aggregate_family_responses(scores, family, "gold_probability")
        for family in ("irrelevant_context", "content_deletion")
    }
    rows: list[dict[str, Any]] = []
    for seed in split_seeds:
        split = stratified_question_split(selected, seed)
        for family, aggregated in aggregated_by_family.items():
            for target in model_ids:
                all_peers = [model for model in model_ids if model != target]
                baseline, _ = _scalar_error(aggregated, target, all_peers, split)
                target_record = record_by_id[target]
                removals = {
                    "exact_sibling": [
                        peer
                        for peer in all_peers
                        if target_record.get("exact_sibling_group") is not None
                        and record_by_id[peer].get("exact_sibling_group")
                        == target_record.get("exact_sibling_group")
                    ],
                    "broad_lineage": [
                        peer
                        for peer in all_peers
                        if record_by_id[peer]["broad_lineage"]
                        == target_record["broad_lineage"]
                    ],
                }
                for removal_type, removed in removals.items():
                    if not removed or len(all_peers) - len(removed) < 1:
                        continue
                    retained = [peer for peer in all_peers if peer not in removed]
                    observed, _ = _scalar_error(aggregated, target, retained, split)
                    random_inflations: list[float] = []
                    for replicate in range(replicates):
                        ordered = sorted(
                            all_peers,
                            key=lambda peer: (
                                stable_u64(
                                    seed, family, target, removal_type, replicate, peer
                                ),
                                peer,
                            ),
                        )
                        random_removed = set(ordered[: len(removed)])
                        random_retained = [peer for peer in all_peers if peer not in random_removed]
                        error, _ = _scalar_error(aggregated, target, random_retained, split)
                        random_inflations.append(error - baseline)
                    inflation = observed - baseline
                    rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": seed,
                            "removal_type": removal_type,
                            "removed_peers": removed,
                            "removed_peer_count": len(removed),
                            "remaining_peer_count": len(retained),
                            "exclude_from_aggregate_rank": len(retained) < 3,
                            "baseline_pier": baseline,
                            "removed_pier": observed,
                            "observed_inflation": inflation,
                            "random_replicates": replicates,
                            "random_median_inflation": float(np.median(random_inflations)),
                            "random_lower_95": float(np.quantile(random_inflations, 0.025)),
                            "random_upper_95": float(np.quantile(random_inflations, 0.975)),
                            "observed_percentile": float(
                                np.mean(np.asarray(random_inflations) <= inflation)
                            ),
                            "random_inflations": random_inflations,
                        }
                    )
    return pd.DataFrame(rows)


def _design_transfer_results(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
) -> pd.DataFrame:
    irrelevant = aggregate_family_responses(scores, "irrelevant_context", "gold_probability")
    deletion = aggregate_family_responses(scores, "content_deletion", "gold_probability")
    rows: list[dict[str, Any]] = []
    for seed in split_seeds:
        split = stratified_question_split(selected, seed)
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            low_high_specs = [
                (
                    "irrelevant_low_to_high",
                    irrelevant,
                    lambda frame: frame["dose"].astype(float).isin([0.0, 64.0, 128.0]),
                    lambda frame: frame["dose"].astype(float).isin([256.0, 512.0]),
                ),
                (
                    "deletion_low_to_high",
                    deletion,
                    lambda frame: frame["dose"].astype(float).isin([0.0, 0.1, 0.2]),
                    lambda frame: frame["dose"].astype(float).isin([0.3, 0.4]),
                ),
            ]
            for name, aggregated, source_selector, destination_selector in low_high_specs:
                result = design_transfer(
                    aggregated,
                    aggregated,
                    target,
                    peers,
                    split,
                    source_selector=source_selector,
                    destination_selector=destination_selector,
                )
                result.update(
                    {
                        "transfer_type": name,
                        "source_family": name.split("_low_to_high")[0],
                        "destination_family": name.split("_low_to_high")[0],
                    }
                )
                rows.append(result)
            for source_name, source, destination_name, destination in (
                ("irrelevant_context", irrelevant, "content_deletion", deletion),
                ("content_deletion", deletion, "irrelevant_context", irrelevant),
            ):
                result = design_transfer(source, destination, target, peers, split)
                result.update(
                    {
                        "transfer_type": f"{source_name}_to_{destination_name}",
                        "source_family": source_name,
                        "destination_family": destination_name,
                    }
                )
                rows.append(result)
    return pd.DataFrame(rows)


def _calibration_sensitivity(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
    raw_scalar: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_overall = raw_scalar[
        raw_scalar["dose"].isna()
        & raw_scalar["representation"].eq("gold_probability")
        & raw_scalar["peer_set_condition"].eq("all_peers")
    ]
    for seed in split_seeds:
        split = stratified_question_split(selected, seed)
        temperatures = {
            model: fit_clean_temperature(scores[scores["model_id"].eq(model)], split.fitting_ids)
            for model in model_ids
        }
        for family in ("irrelevant_context", "content_deletion"):
            calibrated = aggregate_family_responses(
                scores, family, "gold_probability", temperatures
            )
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                result_rows, _weights, _projection = fit_target_family(
                    calibrated,
                    target,
                    peers,
                    split,
                    family=family,
                    representation="calibrated_gold_probability",
                )
                calibrated_overall = next(row for row in result_rows if math.isnan(row["dose"]))
                raw_match = raw_overall[
                    raw_overall["target"].eq(target)
                    & raw_overall["family"].eq(family)
                    & raw_overall["split_seed"].eq(seed)
                ].iloc[0]
                rows.append(
                    {
                        "target": target,
                        "family": family,
                        "split_seed": seed,
                        "temperature": temperatures[target],
                        "peer_temperatures": [temperatures[peer] for peer in peers],
                        "raw_pier": float(raw_match["pier"]),
                        "calibrated_pier": float(calibrated_overall["pier"]),
                        "pier_change": float(calibrated_overall["pier"] - raw_match["pier"]),
                        "raw_honest_convexity_gap": float(raw_match["honest_convexity_gap"]),
                        "calibrated_honest_convexity_gap": float(
                            calibrated_overall["honest_convexity_gap"]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["raw_rank"] = frame.groupby(["family", "split_seed"])["raw_pier"].rank(
        method="average"
    )
    frame["calibrated_rank"] = frame.groupby(["family", "split_seed"])[
        "calibrated_pier"
    ].rank(method="average")
    frame["rank_shift"] = frame["calibrated_rank"] - frame["raw_rank"]
    return frame


def _generation_validation(scores: pd.DataFrame, generation: pd.DataFrame) -> pd.DataFrame:
    lookup = scores[
        ["model_id", "prompt_id", "argmax_label", "semantic_argmax_index"]
    ].rename(columns={"prompt_id": "scoring_prompt_id", "argmax_label": "score_argmax_label"})
    merged = generation.merge(lookup, on=["model_id", "scoring_prompt_id"], validate="one_to_one")
    merged["score_generation_agreement"] = (
        merged["extracted_label"].notna()
        & merged["extracted_label"].eq(merged["score_argmax_label"])
    )
    merged["agreement_below_70_percent_model"] = merged["model_id"].map(
        merged.groupby("model_id")["score_generation_agreement"].mean().lt(0.70)
    )
    return merged


def _control_record(
    name: str,
    expected: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_evaluation: np.ndarray,
    y_evaluation: np.ndarray,
    *,
    vector: bool = False,
) -> dict[str, Any]:
    fit_x = x_fit.reshape(-1, x_fit.shape[-1]) if x_fit.ndim == 3 else x_fit
    fit_y = y_fit.reshape(-1) if y_fit.ndim > 1 else y_fit
    projection = fit_simplex_projection(fit_x, fit_y)
    predicted = np.tensordot(x_evaluation, projection.weights, axes=([-1], [0]))
    residuals = (
        0.5 * np.abs(y_evaluation - predicted).sum(axis=1)
        if vector
        else np.abs(y_evaluation - predicted)
    )
    threshold_mean = 1e-9
    threshold_max = 1e-7
    return {
        "control": name,
        "expected_behavior": expected,
        "observed_mean_residual": float(np.mean(residuals)),
        "observed_max_residual": float(np.max(residuals)),
        "threshold_mean": threshold_mean,
        "threshold_max": threshold_max,
        "passed": bool(
            float(np.mean(residuals)) <= threshold_mean
            and float(np.max(residuals)) <= threshold_max
        ),
        "fitted_weights": projection.weights.tolist(),
        **projection.diagnostics(),
    }


def _real_score_controls(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    canonical_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split = stratified_question_split(selected, canonical_seed)
    records: list[dict[str, Any]] = []
    for representation in ("gold_probability", "probability_vector"):
        aggregated = aggregate_family_responses(scores, "irrelevant_context", representation)
        metadata, _unused_target, all_design = aligned_response_arrays(
            aggregated, model_ids[0], model_ids
        )
        fitting = metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
        evaluation = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
        vector = representation == "probability_vector"
        clone = all_design[..., 0]
        records.append(
            _control_record(
                f"exact_clone_{representation}",
                "exact source recovery at numerical precision",
                all_design[fitting],
                clone[fitting],
                all_design[evaluation],
                clone[evaluation],
                vector=vector,
            )
        )
        sparse_weights = np.zeros(len(model_ids))
        sparse_weights[:3] = [0.5, 0.3, 0.2]
        sparse = np.tensordot(all_design, sparse_weights, axes=([-1], [0]))
        records.append(
            _control_record(
                f"known_sparse_mixture_{representation}",
                "0.50/0.30/0.20 mixture recovery at numerical precision",
                all_design[fitting],
                sparse[fitting],
                all_design[evaluation],
                sparse[evaluation],
                vector=vector,
            )
        )
        boundary_weights = np.zeros(len(model_ids))
        boundary_weights[:2] = [0.75, 0.25]
        boundary = np.tensordot(all_design, boundary_weights, axes=([-1], [0]))
        records.append(
            _control_record(
                f"boundary_mixture_{representation}",
                "simplex-face mixture recovery at numerical precision",
                all_design[fitting],
                boundary[fitting],
                all_design[evaluation],
                boundary[evaluation],
                vector=vector,
            )
        )
    scalar = aggregate_family_responses(scores, "irrelevant_context", "gold_probability")
    metadata, _target, design = aligned_response_arrays(scalar, model_ids[0], model_ids)
    fitting = metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
    evaluation = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
    duplicated = np.column_stack([design, design[:, 0]])
    records.append(
        _control_record(
            "duplicated_peer_ambiguity",
            "unchanged projected response despite non-unique weights",
            duplicated[fitting],
            design[fitting, 0],
            duplicated[evaluation],
            design[evaluation, 0],
        )
    )
    frame = pd.DataFrame(records)
    summary = {
        "schema_version": "pier_control_results_v2",
        "timestamp": utc_now(),
        "passed": bool(frame["passed"].all()),
        "control_count": len(frame),
        "controls": frame.to_dict(orient="records"),
    }
    return frame, summary


def _weight_ambiguity_audit(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    scalar_results: pd.DataFrame,
    canonical_seed: int,
) -> pd.DataFrame:
    candidates = scalar_results[
        scalar_results["dose"].isna()
        & scalar_results["representation"].eq("gold_probability")
        & scalar_results["peer_set_condition"].eq("all_peers")
        & scalar_results["split_seed"].eq(canonical_seed)
    ].nsmallest(6, "pier")
    split = stratified_question_split(selected, canonical_seed)
    rows: list[dict[str, Any]] = []
    aggregated_cache: dict[str, pd.DataFrame] = {}
    for _, candidate in candidates.iterrows():
        target = str(candidate["target"])
        family = str(candidate["family"])
        aggregated = aggregated_cache.setdefault(
            family, aggregate_family_responses(scores, family, "gold_probability")
        )
        peers = [model for model in model_ids if model != target]
        metadata, target_values, design = aligned_response_arrays(aggregated, target, peers)
        fitting = metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
        evaluation = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
        projection = fit_simplex_projection(design[fitting], target_values[fitting])
        intervals = feasible_weight_intervals(
            design[fitting], target_values[fitting], projection
        )
        for peer, interval in zip(peers, intervals, strict=True):
            minimum_weights = np.asarray(interval.pop("minimum_solution_weights"))
            maximum_weights = np.asarray(interval.pop("maximum_solution_weights"))
            minimum_prediction = design[evaluation] @ minimum_weights
            maximum_prediction = design[evaluation] @ maximum_weights
            rows.append(
                {
                    "target": target,
                    "family": family,
                    "split_seed": canonical_seed,
                    "peer": peer,
                    "representative_weight": float(
                        projection.weights[peers.index(peer)]
                    ),
                    "interval_minimum": interval["minimum"],
                    "interval_maximum": interval["maximum"],
                    "interval_width": interval["width"],
                    "evaluation_prediction_mean_absolute_variation": float(
                        np.mean(np.abs(minimum_prediction - maximum_prediction))
                    ),
                    "evaluation_prediction_max_absolute_variation": float(
                        np.max(np.abs(minimum_prediction - maximum_prediction))
                    ),
                    "fitting_loss_optimum": projection.stage1_objective,
                    "loss_tolerance": projection.tolerance,
                    "status_min": interval["status_min"],
                    "status_max": interval["status_max"],
                }
            )
    return pd.DataFrame(rows)


def _ecosystem_coverage(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_records: list[dict[str, Any]],
    canonical_seed: int,
) -> pd.DataFrame:
    model_ids = [row["id"] for row in model_records]
    lineage = {row["id"]: row["broad_lineage"] for row in model_records}
    irrelevant = aggregate_family_responses(scores, "irrelevant_context", "gold_probability")
    deletion = aggregate_family_responses(scores, "content_deletion", "gold_probability")
    split = stratified_question_split(selected, canonical_seed)
    designs = [
        ("clean", irrelevant, lambda frame: frame["dose"].astype(float).eq(0.0)),
        (
            "maximum_irrelevant_context",
            irrelevant,
            lambda frame: frame["dose"].astype(float).eq(512.0),
        ),
        (
            "maximum_content_deletion",
            deletion,
            lambda frame: frame["dose"].astype(float).eq(0.4),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for target in model_ids:
        all_peers = [model for model in model_ids if model != target]
        for design_name, aggregated, selector in designs:
            for subset_size in range(1, len(all_peers) + 1):
                for subset in itertools.combinations(all_peers, subset_size):
                    metadata, target_values, matrix = aligned_response_arrays(
                        aggregated, target, list(subset)
                    )
                    selected_rows = np.asarray(selector(metadata), dtype=bool)
                    fitting = (
                        metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
                        & selected_rows
                    )
                    evaluation = (
                        metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
                        & selected_rows
                    )
                    projection = fit_simplex_projection(matrix[fitting], target_values[fitting])
                    prediction = matrix[evaluation] @ projection.weights
                    peer_lineages = {lineage[peer] for peer in subset}
                    rows.append(
                        {
                            "target": target,
                            "design": design_name,
                            "split_seed": canonical_seed,
                            "peer_subset": list(subset),
                            "peer_subset_size": subset_size,
                            "pier": float(np.mean(np.abs(target_values[evaluation] - prediction))),
                            "lineage_count": len(peer_lineages),
                            "lineage_diverse": len(peer_lineages) == subset_size,
                            "contains_target_lineage": lineage[target] in peer_lineages,
                            "same_lineage_peer_count": sum(
                                lineage[peer] == lineage[target] for peer in subset
                            ),
                            "weights": projection.weights.tolist(),
                        }
                    )
    return pd.DataFrame(rows)


def _headline_bootstrap(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    scalar_results: pd.DataFrame,
    canonical_seed: int,
    replicates: int,
) -> pd.DataFrame:
    candidates = scalar_results[
        scalar_results["dose"].isna()
        & scalar_results["representation"].eq("gold_probability")
        & scalar_results["peer_set_condition"].eq("all_peers")
        & scalar_results["split_seed"].eq(canonical_seed)
    ].nlargest(3, "honest_convexity_gap")
    split = stratified_question_split(selected, canonical_seed)
    frames: list[pd.DataFrame] = []
    for _, candidate in candidates.iterrows():
        target = str(candidate["target"])
        family = str(candidate["family"])
        peers = [model for model in model_ids if model != target]
        aggregated = aggregate_family_responses(scores, family, "gold_probability")
        metadata, target_values, design = aligned_response_arrays(aggregated, target, peers)
        fitting = metadata["base_question_id"].astype(str).isin(split.fitting_ids).to_numpy()
        evaluation = metadata["base_question_id"].astype(str).isin(split.evaluation_ids).to_numpy()
        bootstrap = clustered_bootstrap_with_refitting(
            metadata.loc[fitting].reset_index(drop=True),
            design[fitting],
            target_values[fitting],
            metadata.loc[evaluation].reset_index(drop=True),
            design[evaluation],
            target_values[evaluation],
            replicates=replicates,
            seed=stable_u64(canonical_seed, target, family) % (2**32),
        )
        bootstrap["target"] = target
        bootstrap["family"] = family
        bootstrap["canonical_split_seed"] = canonical_seed
        frames.append(bootstrap)
    return pd.concat(frames, ignore_index=True)


def _table_outputs(
    root: Path,
    scores: pd.DataFrame,
    resolved: dict[str, Any],
    scalar: pd.DataFrame,
    vector: pd.DataFrame,
    removal: pd.DataFrame,
    transfer: pd.DataFrame,
    calibration: pd.DataFrame,
    stability: pd.DataFrame,
    permutation: pd.DataFrame,
    generation: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    table_dir = root / "outputs" / "tables"
    records = resolved["models"]
    clean_accuracy = (
        scores[scores["condition"].eq("clean")]
        .groupby("model_id")["correct"]
        .mean()
        .to_dict()
    )
    table1 = pd.DataFrame(
        [
            {
                "model": row["id"],
                "revision": row["revision"],
                "parameter_count": row.get("parameter_count"),
                "broad_lineage": row["broad_lineage"],
                "exact_sibling_group": row.get("exact_sibling_group"),
                "post_training_type": row["post_training_type"],
                "tokenizer": row["tokenizer_class"],
                "chat_template": row["chat_template_available"],
                "prompt_count": int(scores["model_id"].eq(row["id"]).sum()),
                "generation_count": int(generation["model_id"].eq(row["id"]).sum()),
                "clean_direct_answer_accuracy": clean_accuracy.get(row["id"], math.nan),
            }
            for row in records
        ]
    )
    primary = scalar[
        scalar["representation"].eq("gold_probability")
        & scalar["peer_set_condition"].eq("all_peers")
    ]
    overall = primary[primary["dose"].isna()]
    maximum = primary[primary["dose"].notna()].sort_values("pier").groupby("target").tail(1)
    clean = primary[primary["dose"].eq(0.0)].groupby("target")["pier"].mean()
    removal_summary = removal.groupby(["target", "removal_type"])[
        ["observed_inflation", "observed_percentile"]
    ].mean()
    vector_summary = vector[
        vector["dose"].isna() & vector["weight_fit"].eq("vector_fitted")
    ].groupby("target")[["mean_total_variation", "top1_agreement"]].mean()
    split_summary = overall.groupby("target")["pier"].std()
    generation_summary = generation.groupby("model_id")["score_generation_agreement"].mean()
    table2_rows = []
    for target in [row["id"] for row in records]:
        target_overall = overall[overall["target"].eq(target)]
        target_max = maximum[maximum["target"].eq(target)]
        max_row = target_max.sort_values("pier").tail(1)
        table2_rows.append(
            {
                "target": target,
                "clean_pier": clean.get(target, math.nan),
                "maximum_pier": float(max_row["pier"].iloc[0]) if not max_row.empty else math.nan,
                "condition_of_maximum": (
                    f"{max_row['family'].iloc[0]}:{max_row['dose'].iloc[0]}"
                    if not max_row.empty
                    else ""
                ),
                "honest_convexity_gain": float(
                    target_overall["honest_convexity_gap"].mean()
                ),
                "exact_sibling_removal_inflation": removal_summary.get(
                    "observed_inflation", pd.Series(dtype=float)
                ).get((target, "exact_sibling"), math.nan),
                "broad_lineage_removal_inflation": removal_summary.get(
                    "observed_inflation", pd.Series(dtype=float)
                ).get((target, "broad_lineage"), math.nan),
                "random_removal_percentile": removal_summary.get(
                    "observed_percentile", pd.Series(dtype=float)
                ).get((target, "broad_lineage"), math.nan),
                "vector_tv": vector_summary.get(
                    "mean_total_variation", pd.Series(dtype=float)
                ).get(target, math.nan),
                "top1_surrogate_agreement": vector_summary.get(
                    "top1_agreement", pd.Series(dtype=float)
                ).get(target, math.nan),
                "split_stability": split_summary.get(target, math.nan),
                "generation_score_agreement": generation_summary.get(target, math.nan),
            }
        )
    table2 = pd.DataFrame(table2_rows)

    ranks = (
        scalar[scalar["dose"].isna()]
        .groupby(["target", "representation"])["pier"]
        .mean()
        .unstack("representation")
        .rank()
    )
    vector_ranks = vector[
        vector["dose"].isna() & vector["weight_fit"].eq("vector_fitted")
    ].groupby("target")["mean_total_variation"].mean().rank()
    calibration_rank = calibration.groupby("target")["calibrated_pier"].mean().rank()
    table3 = pd.DataFrame(index=[row["id"] for row in records])
    table3.index.name = "target"
    table3["gold_probability_rank"] = ranks.get("gold_probability")
    table3["margin_rank"] = ranks.get("margin")
    table3["vector_response_rank"] = vector_ranks
    table3["calibrated_probability_rank"] = calibration_rank
    family_ranks = overall.groupby(["target", "family"])["pier"].mean().groupby(level=1).rank()
    family_pivot = family_ranks.unstack("family")
    table3["intervention_family_rank_shift"] = (
        family_pivot.get("irrelevant_context") - family_pivot.get("content_deletion")
    )
    table3["low_to_high_transfer_penalty"] = transfer[
        transfer["transfer_type"].str.contains("low_to_high")
    ].groupby("target")["transfer_penalty"].mean()
    table3["cross_family_transfer_penalty"] = transfer[
        ~transfer["transfer_type"].str.contains("low_to_high")
    ].groupby("target")["transfer_penalty"].mean()
    table3 = table3.reset_index()

    table4 = controls[
        [
            "control",
            "expected_behavior",
            "observed_mean_residual",
            "observed_max_residual",
            "passed",
        ]
    ].copy()
    candidate_rows: list[dict[str, Any]] = []

    def candidate(category: str, frame: pd.DataFrame, metric: str, largest: bool = True) -> None:
        usable = frame.dropna(subset=[metric])
        if usable.empty:
            return
        row = usable.nlargest(1, metric).iloc[0] if largest else usable.nsmallest(1, metric).iloc[0]
        candidate_rows.append(
            {
                "criterion": category,
                "target": row.get("target", row.get("model_id", "")),
                "value": float(row[metric]),
                "metric": metric,
            }
        )

    intervention_change = primary[primary["dose"].notna()].groupby("target")["pier"].agg(
        lambda values: float(values.max() - values.min())
    ).reset_index(name="increase")
    candidate("largest stable intervention-activated PIER increase", intervention_change, "increase")
    candidate("largest honest convexity gap", overall, "honest_convexity_gap")
    candidate("largest lineage effect beyond random removal", removal, "observed_inflation")
    candidate("lowest stable real-model PIER", overall, "pier", largest=False)
    candidate("highest stable real-model PIER", overall, "pier")
    vector_mismatch = vector_summary.reset_index().rename(
        columns={"mean_total_variation": "mismatch"}
    )
    candidate("largest scalar-to-vector mismatch", vector_mismatch, "mismatch")
    candidate("largest design-transfer penalty", transfer, "transfer_penalty")
    permutation_summary = permutation.groupby("model_id")["probability_vector_tv"].mean().reset_index()
    candidate("largest option-permutation artifact", permutation_summary, "probability_vector_tv")
    generation_mismatch = (
        1 - generation.groupby("model_id")["score_generation_agreement"].mean()
    ).reset_index(name="mismatch")
    candidate("largest score-vs-generation mismatch", generation_mismatch, "mismatch")
    table5 = pd.DataFrame(candidate_rows)
    tables = {
        "table1_ecosystem_manifest.csv": table1,
        "table2_main_target_audit_summary.csv": table2,
        "table3_design_representation_sensitivity.csv": table3,
        "table4_controls_quality_checks.csv": table4,
        "table5_gold_candidates.csv": table5,
    }
    for name, frame in tables.items():
        _atomic_csv(table_dir / name, frame)
    return tables


def _environment_manifest(root: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface-hub",
        "safetensors",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "cvxpy",
        "osqp",
        "clarabel",
        "matplotlib",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    import torch

    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        driver = []
    config = _load_json(root / "configs" / "experiment.json")
    source = _load_json(root / "data" / "manifests" / "dataset_source.json")
    launcher = root / "scripts" / "run_b200_inference.sh"
    return {
        "schema_version": "pier_environment_manifest_v2",
        **environment_basics(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "driver_versions": sorted(set(driver)),
        "dependency_versions": versions,
        "random_seeds": {
            "sample_seed": config["sample_seed"],
            "generation_seed": config["generation_seed"],
            "split_seeds": config["split_seeds"],
        },
        "dataset_revision": source["dataset_revision"],
        "dataset_selected_id_sha256": source["selected_id_sha256"],
        "intervention_manifest_hashes": source["hashes"],
        "model_revisions": {row["id"]: row["revision"] for row in resolved["models"]},
        "tokenizer_revisions": {row["id"]: row["revision"] for row in resolved["models"]},
        "chat_template_hashes": {
            row["id"]: _load_json(Path(row["compatibility_manifest_path"]))[
                "chat_template_sha256"
            ]
            for row in resolved["models"]
        },
        "source_tree_sha256": tree_sha256(root / "src"),
        "formal_launcher_sha256": sha256_file(launcher),
    }


def _report(
    tables: dict[str, pd.DataFrame],
    scalar: pd.DataFrame,
    vector: pd.DataFrame,
    removal: pd.DataFrame,
    transfer: pd.DataFrame,
    calibration: pd.DataFrame,
    stability: pd.DataFrame,
    permutation: pd.DataFrame,
    generation: pd.DataFrame,
    controls: pd.DataFrame,
    coverage: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    overall = scalar[
        scalar["dose"].isna()
        & scalar["representation"].eq("gold_probability")
        & scalar["peer_set_condition"].eq("all_peers")
    ]
    target_summary = overall.groupby("target")["pier"].mean().sort_values()
    family_summary = overall.groupby("family")["pier"].mean().sort_values(ascending=False)
    gap = overall.groupby("target")["honest_convexity_gap"].mean().sort_values(ascending=False)
    vector_tv = vector[vector["dose"].isna()].groupby("target")["mean_total_variation"].mean()
    generation_agreement = generation.groupby("model_id")["score_generation_agreement"].mean()
    malformed = generation.groupby("model_id")["malformed"].mean()
    perm_tv = permutation.groupby("model_id")["probability_vector_tv"].mean()
    bootstrap_intervals = bootstrap.groupby(["target", "family"])["metric"].quantile(
        [0.025, 0.975]
    )
    top_candidates = tables["table5_gold_candidates.csv"]
    findings = top_candidates.head(3)
    lines = [
        "# PIER Modern-LLM Ecosystem Gold-Mining Report V2",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Scope and interpretation",
        "",
        "This report treats held-out surrogate predictions and PIER as primary. Individual convex "
        "weights are descriptive unless their feasible intervals are narrow. MMLU-Pro is used as a "
        "fixed direct-answer probe, not as an official chain-of-thought leaderboard evaluation.",
        "",
        "## Numerical audit questions",
        "",
        f"1. Non-trivial structure: mean held-out gold-probability PIER was "
        f"{overall['pier'].mean():.6g}, spanning target means "
        f"{target_summary.min():.6g}–{target_summary.max():.6g}.",
        f"2. Most peer-expressible: `{target_summary.index[0]}` ({target_summary.iloc[0]:.6g}); "
        f"least peer-expressible: `{target_summary.index[-1]}` ({target_summary.iloc[-1]:.6g}).",
        f"3. Strongest mean intervention family: `{family_summary.index[0]}` "
        f"({family_summary.iloc[0]:.6g}).",
        "4. Dose trends are summarized with three-track aggregation in the main figure; "
        "track-specific dispersion remains a descriptive diagnostic.",
        f"5. The mean honest convexity gap (fit-selected single error / convex PIER) was "
        f"{overall['honest_convexity_gap'].mean():.4g}.",
        f"6. Strongest mean honest convexity gap: `{gap.index[0]}` ({gap.iloc[0]:.4g}).",
        f"7. Exact-sibling removal mean inflation was "
        f"{removal[removal['removal_type'].eq('exact_sibling')]['observed_inflation'].mean():.6g}; "
        "interpret only relative to its size-matched random percentile.",
        f"8. Broad-lineage removal mean inflation was "
        f"{removal[removal['removal_type'].eq('broad_lineage')]['observed_inflation'].mean():.6g}.",
        f"9. Median repeated-split rank correlation was "
        f"{stability['median_rank_correlation'].median():.4g}.",
        "10. Weight non-identifiability is reported through feasible per-peer intervals and "
        "projected-response variation in `weight_ambiguity_audit.parquet`.",
        f"11. Mean absolute calibrated-vs-raw PIER change was "
        f"{calibration['pier_change'].abs().mean():.6g}.",
        f"12. Margin and vector sensitivities are reported separately; mean vector TV was "
        f"{vector_tv.mean():.6g}, without equating response scales.",
        "13. Scalar-to-vector fidelity is shown in Panel E and the representation-sensitivity table.",
        f"14. Mean candidate-score/direct-generation agreement was "
        f"{generation_agreement.mean():.3%}; mean malformed rate was {malformed.mean():.3%}.",
        f"15. Mean low-to-high transfer penalty was "
        f"{transfer[transfer['transfer_type'].str.contains('low_to_high')]['transfer_penalty'].mean():.6g}.",
        f"16. Mean cross-family transfer penalty was "
        f"{transfer[~transfer['transfer_type'].str.contains('low_to_high')]['transfer_penalty'].mean():.6g}.",
        f"17. Exact-clone/known-mixture controls passed: {bool(controls['passed'].all())}.",
        f"18. Mean clean-vs-permuted probability-vector TV was {perm_tv.mean():.6g}; "
        "large model-specific values are retained as limitations.",
        f"19. The exploratory coverage analysis contains {len(coverage):,} real-model subset fits; "
        "it is not interpreted as a universal phase transition.",
        "20. The three strongest automatically ranked candidates are listed below.",
    ]
    for _, row in findings.iterrows():
        lines.append(f"    - {row['criterion']}: `{row['target']}` ({row['value']:.6g})")
    low_generation = generation_agreement.idxmin()
    high_permutation = perm_tv.idxmax()
    lines.extend(
        [
            "21. Important negative/limiting findings:",
            f"    - Lowest score-generation agreement: `{low_generation}` "
            f"({generation_agreement[low_generation]:.3%}).",
            f"    - Largest option-permutation TV: `{high_permutation}` "
            f"({perm_tv[high_permutation]:.6g}).",
            "    - Convex weights can be non-identifiable even when projected responses are stable.",
            "22. Replacement of a BERT/SST-2 flagship experiment should be decided from effect "
            "sizes, controls, repeated-split stability, and bootstrap intervals—not from one ranking. "
            "This pipeline does not make that editorial decision automatically.",
            "23. If evidence is insufficient, the single best next experiment is a preregistered "
            "replication on a second task family with the same roster and matched intervention design.",
            "",
            "## Bootstrap note",
            "",
            "Headline intervals use base-question cluster resampling with refitting. The formal run "
            f"contains {int(bootstrap['replicate'].max() + 1)} replicates per selected case. "
            f"Recorded interval groups: {len(bootstrap_intervals) // 2}.",
            "",
            "## Evidence, interpretation, limitation, next step",
            "",
            "- Observed evidence: all numerical statements above are derived from held-out rows and "
            "the fixed manifests.",
            "- Interpretation: low PIER supports response-space peer expressibility under the tested "
            "design, not causal equivalence or model interchangeability.",
            "- Limitation: candidate-label likelihood, direct generation, and prompt interventions "
            "probe a narrow task/interface slice and may reflect tokenizer or label artifacts.",
            "- Next step: replicate the strongest preregistered effects on a distinct benchmark and "
            "larger model ecosystem before generalizing.",
        ]
    )
    return "\n".join(lines) + "\n"


def _required_postprocess_paths(root: Path) -> list[Path]:
    relative = [
        "outputs/analysis/disco_scalar_results.parquet",
        "outputs/analysis/disco_vector_results.parquet",
        "outputs/analysis/disco_weights.parquet",
        "outputs/analysis/weight_ambiguity_audit.parquet",
        "outputs/analysis/peer_removal_controls.parquet",
        "outputs/analysis/design_transfer_results.parquet",
        "outputs/analysis/calibration_sensitivity.parquet",
        "outputs/analysis/split_stability.parquet",
        "outputs/analysis/option_permutation_results.parquet",
        "outputs/analysis/generation_validation.parquet",
        "outputs/analysis/ecosystem_coverage_results.parquet",
        "outputs/analysis/bootstrap_headline_results.parquet",
        "outputs/analysis/GOLDMINE_REPORT.md",
        "outputs/analysis/environment_manifest.json",
        "outputs/controls/control_results.json",
        "outputs/figures/fig_modern_llm_ecosystem_main.pdf",
        "outputs/figures/fig_modern_llm_ecosystem_main.png",
        "outputs/figures/fig_response_representation_sensitivity.pdf",
        "outputs/figures/fig_split_stability.pdf",
        "outputs/figures/fig_option_permutation_control.pdf",
        "outputs/figures/fig_generation_validation.pdf",
        "outputs/figures/fig_weight_ambiguity.pdf",
        "outputs/figures/fig_ecosystem_coverage_curve.pdf",
        "outputs/figures/fig_estimator_class_comparison.pdf",
        "outputs/tables/table1_ecosystem_manifest.csv",
        "outputs/tables/table2_main_target_audit_summary.csv",
        "outputs/tables/table3_design_representation_sensitivity.csv",
        "outputs/tables/table4_controls_quality_checks.csv",
        "outputs/tables/table5_gold_candidates.csv",
        "REPRODUCE.md",
    ]
    return [root / value for value in relative]


def run_postprocess(root: Path, resume: bool = True) -> dict[str, Any]:
    del resume  # valid inputs are always revalidated; completed shards are never modified here
    status_path = root / "status" / "postprocess_status.json"
    status: dict[str, Any] = {
        "schema_version": "pier_postprocess_status_v2",
        "timestamp_started": utc_now(),
        "complete": False,
        "stage": "loading_validated_inference",
        "errors": [],
    }
    atomic_write_json(status_path, status)
    try:
        scores, generation_raw, resolved = load_validated_inference(root)
        config = _load_json(root / "configs" / "experiment.json")
        selected = pd.DataFrame(read_jsonl(root / "data" / "mmlu_pro_selected_560.jsonl"))
        model_records = list(resolved["models"])
        model_ids = [row["id"] for row in model_records]
        split_seeds = [int(value) for value in config["split_seeds"]]
        canonical_seed = split_seeds[0]

        status["stage"] = "primary_scalar_and_vector"
        atomic_write_json(status_path, status)
        scalar, vector, weights, _weight_cache = _primary_analyses(
            scores, selected, model_ids, split_seeds
        )
        _atomic_parquet(root / "outputs/analysis/disco_scalar_results.parquet", scalar)
        _atomic_parquet(root / "outputs/analysis/disco_vector_results.parquet", vector)
        _atomic_parquet(root / "outputs/analysis/disco_weights.parquet", weights)

        status["stage"] = "peer_removal_and_transfer"
        atomic_write_json(status_path, status)
        removal = _peer_removal_controls(
            scores,
            selected,
            model_records,
            split_seeds,
            int(config["random_peer_removal_replicates"]),
        )
        transfer = _design_transfer_results(scores, selected, model_ids, split_seeds)
        _atomic_parquet(root / "outputs/analysis/peer_removal_controls.parquet", removal)
        _atomic_parquet(root / "outputs/analysis/design_transfer_results.parquet", transfer)

        status["stage"] = "calibration_and_quality_audits"
        atomic_write_json(status_path, status)
        calibration = _calibration_sensitivity(
            scores, selected, model_ids, split_seeds, scalar
        )
        stability = split_rank_stability(scalar)
        permutation = option_permutation_audit(scores)
        generation = _generation_validation(scores, generation_raw)
        _atomic_parquet(root / "outputs/analysis/calibration_sensitivity.parquet", calibration)
        _atomic_parquet(root / "outputs/analysis/split_stability.parquet", stability)
        _atomic_parquet(
            root / "outputs/analysis/option_permutation_results.parquet", permutation
        )
        _atomic_parquet(
            root / "outputs/analysis/generation_validation.parquet", generation
        )

        status["stage"] = "controls_ambiguity_coverage_bootstrap"
        atomic_write_json(status_path, status)
        controls, controls_summary = _real_score_controls(
            scores, selected, model_ids, canonical_seed
        )
        root.joinpath("outputs", "controls").mkdir(parents=True, exist_ok=True)
        atomic_write_json(root / "outputs/controls/control_results.json", controls_summary)
        if not controls_summary["passed"]:
            raise RuntimeError("One or more exact/mixture analysis controls failed")
        ambiguity = _weight_ambiguity_audit(
            scores, selected, model_ids, scalar, canonical_seed
        )
        coverage = _ecosystem_coverage(scores, selected, model_records, canonical_seed)
        bootstrap = _headline_bootstrap(
            scores,
            selected,
            model_ids,
            scalar,
            canonical_seed,
            int(config["bootstrap_replicates"]),
        )
        _atomic_parquet(
            root / "outputs/analysis/weight_ambiguity_audit.parquet", ambiguity
        )
        _atomic_parquet(
            root / "outputs/analysis/ecosystem_coverage_results.parquet", coverage
        )
        _atomic_parquet(
            root / "outputs/analysis/bootstrap_headline_results.parquet", bootstrap
        )

        status["stage"] = "figures_tables_report"
        atomic_write_json(status_path, status)
        make_all_figures(
            root / "outputs/figures",
            scalar=scalar,
            vector=vector,
            peer_removal=removal,
            transfer=transfer,
            stability=stability,
            permutation=permutation,
            generation=generation,
            ambiguity=ambiguity,
            coverage=coverage,
        )
        tables = _table_outputs(
            root,
            scores,
            resolved,
            scalar,
            vector,
            removal,
            transfer,
            calibration,
            stability,
            permutation,
            generation,
            controls,
        )
        report = _report(
            tables,
            scalar,
            vector,
            removal,
            transfer,
            calibration,
            stability,
            permutation,
            generation,
            controls,
            coverage,
            bootstrap,
        )
        atomic_write_text(root / "outputs/analysis/GOLDMINE_REPORT.md", report)
        atomic_write_json(
            root / "outputs/analysis/environment_manifest.json",
            _environment_manifest(root, resolved),
        )

        missing = [str(path) for path in _required_postprocess_paths(root) if not path.is_file()]
        if missing:
            raise RuntimeError(f"Postprocess output completeness failed: {missing}")
        status["stage"] = "packaging"
        atomic_write_json(status_path, status)
        package = build_results_package(root)
        packaged_status = {
            **status,
            "timestamp_completed": utc_now(),
            "complete": True,
            "stage": "complete",
            "package_path": str(package),
            "package_sha256": None,
            "package_sha256_note": (
                "The final archive SHA256 is recorded in the external completion marker "
                "to avoid a self-referential archive hash."
            ),
            "output_count": len(_required_postprocess_paths(root)),
        }
        atomic_write_json(status_path, packaged_status)
        atomic_write_json(root / "status" / "EXPERIMENT_COMPLETE.json", packaged_status)
        package = build_results_package(root, strict_validation=True)
        status = {
            **packaged_status,
            "package_sha256": sha256_file(package),
        }
        status.pop("package_sha256_note")
        atomic_write_json(status_path, status)
        atomic_write_json(root / "status" / "EXPERIMENT_COMPLETE.json", status)
        return status
    except Exception as exc:
        status["timestamp_failed"] = utc_now()
        status["complete"] = False
        status["errors"].append(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        atomic_write_json(root / "status" / "EXPERIMENT_COMPLETE.json", status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_postprocess(args.root.resolve(), resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
