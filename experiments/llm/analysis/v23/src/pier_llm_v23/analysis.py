from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .core import (
    categorical_entropy,
    endpoint_row_weights,
    fit_mae_simplex,
    fit_mse_simplex,
    fixed_splits,
    js_divergence,
    select_single_peer,
    total_variation,
    weighted_mae,
)
from .inference import GENERATION_SCHEMA, SCORE_SCHEMA, validate_all_outputs
from .utils import load_json, read_jsonl, sha256_file, slug, utc_now

FAMILIES = ("irrelevant_context", "content_deletion")


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_stage2_scores(root: Path, models: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in models:
        paths = sorted((root / "outputs/raw_scores" / slug(model["id"])).glob("shard_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No V2.3 score shards for {model['id']}")
        frames.extend(pd.read_parquet(path) for path in paths)
    output = pd.concat(frames, ignore_index=True)
    if not output["schema_version"].eq(SCORE_SCHEMA).all():
        raise ValueError("V2.3 score schema mismatch")
    return output


def load_rotation0_scores(root: Path, models: list[dict[str, Any]]) -> pd.DataFrame:
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    manifest = list(read_jsonl(root / "data/permutations/score_semantic_manifest.jsonl"))
    rotation0 = {row["source_v2_prompt_id"]: row for row in manifest if row["permutation_id"] == 0}
    frames: list[pd.DataFrame] = []
    for model in models:
        for path in sorted((v2 / "outputs/raw_scores" / slug(model["id"])).glob("shard_*.parquet")):
            frame = pd.read_parquet(path)
            subset = frame[frame["prompt_id"].isin(rotation0)]
            if not subset.empty:
                frames.append(subset)
    raw = pd.concat(frames, ignore_index=True)
    if len(raw) != len(rotation0) * len(models):
        raise ValueError(f"Rotation-0 row count mismatch: {len(raw)}")
    rows: list[dict[str, Any]] = []
    for source in raw.itertuples(index=False):
        prompt = rotation0[str(source.prompt_id)]
        semantic = np.asarray(source.semantic_probabilities, dtype=np.float64)
        rows.append(
            {
                "schema_version": SCORE_SCHEMA,
                "prompt_id": prompt["prompt_id"],
                "source_v2_prompt_id": source.prompt_id,
                "base_question_id": str(source.base_question_id),
                "category": source.category,
                "semantic_condition": prompt["semantic_condition"],
                "condition": prompt["condition"],
                "family": prompt["family"],
                "dose": float(prompt["dose"]),
                "track": prompt["track"],
                "permutation_id": 0,
                "option_count": int(prompt["option_count"]),
                "semantic_to_visible": prompt["semantic_to_visible"],
                "visible_to_semantic": prompt["visible_to_semantic"],
                "candidate_continuations": list(source.candidate_continuations),
                "candidate_sequence_log_likelihoods": list(source.candidate_log_likelihoods),
                "visible_label_probabilities": list(source.probabilities),
                "semantic_remapped_probabilities": semantic.tolist(),
                "argmax_visible_index": int(source.argmax_index),
                "argmax_visible_label": source.argmax_label,
                "argmax_semantic_option": int(source.semantic_argmax_index),
                "gold_semantic_option": int(source.semantic_answer_index),
                "gold_semantic_probability": float(semantic[int(source.semantic_answer_index)]),
                "prompt_token_count": int(source.prompt_token_count),
                "candidate_token_counts": list(source.candidate_token_counts),
                "canonical_prompt_hash": prompt["canonical_content_sha256"],
                "rendered_prompt_hash": None,
                "model_id": source.model_id,
                "model_revision": source.model_revision,
                "manifest_sha256": "reused_validated_v2_rotation0",
                "score_origin": "V2_reused_rotation0",
            }
        )
    return pd.DataFrame(rows)


def assemble_permutation_scores(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved = load_json(root / "configs/resolved_models_v23.json")
    models = resolved["models"]
    validation = validate_all_outputs(root, models)
    if not validation["passed"]:
        raise RuntimeError("Stage-2 shards did not pass validation")
    new = _load_stage2_scores(root, models)
    new["score_origin"] = "V2.3_B200_new_rotation"
    rotation0 = load_rotation0_scores(root, models)
    all_rotations = pd.concat([rotation0, new], ignore_index=True)
    expected_per_model = int(load_json(root / "status/prompt_manifest_status.json")[
        "score_all_rotation_rows_per_model"
    ])
    counts = all_rotations.groupby("model_id").size().to_dict()
    expected = {record["id"]: expected_per_model for record in models}
    if counts != expected or all_rotations.duplicated(["model_id", "prompt_id"]).any():
        raise ValueError("Combined rotation score counts/keys are invalid")

    averaged: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    group_columns = ["model_id", "base_question_id", "semantic_condition"]
    for keys, group in all_rotations.groupby(group_columns, sort=True):
        group = group.sort_values("permutation_id")
        option_count = int(group["option_count"].iloc[0])
        rotations = group["permutation_id"].astype(int).tolist()
        if rotations != list(range(option_count)) or len(group) != option_count:
            raise ValueError(f"Incomplete cyclic set: {keys}")
        vectors = np.stack(
            [np.asarray(value, dtype=np.float64) for value in group["semantic_remapped_probabilities"]]
        )
        average = vectors.mean(axis=0)
        if not np.isclose(average.sum(), 1.0, atol=1e-12):
            raise ValueError(f"Permutation average is not normalized: {keys}")
        tvs = np.asarray([total_variation(vector, average) for vector in vectors])
        argmax = np.argmax(vectors, axis=1)
        modal_count = Counter(argmax.tolist()).most_common(1)[0][1]
        first = group.iloc[0]
        gold = int(first["gold_semantic_option"])
        base = {
            "model": keys[0],
            "model_id": keys[0],
            "base_question_id": keys[1],
            "category": first["category"],
            "semantic_condition": keys[2],
            "condition": first["condition"],
            "family": first["family"],
            "dose": float(first["dose"]),
            "track": first["track"],
            "option_count": option_count,
            "gold_semantic_option": gold,
            "permutation_averaged_semantic_probabilities": average.tolist(),
            "permutation_averaged_gold_probability": float(average[gold]),
            "permutation_averaged_semantic_argmax": int(np.argmax(average)),
        }
        averaged.append(base)
        sensitivity.append(
            {
                **{key: base[key] for key in base if not key.startswith("permutation_averaged")},
                "permutation_wise_gold_probability_variance": float(np.var(vectors[:, gold])),
                "permutation_wise_gold_probability_sd": float(np.std(vectors[:, gold])),
                "mean_semantic_vector_TV_to_average": float(np.mean(tvs)),
                "maximum_semantic_vector_TV_to_average": float(np.max(tvs)),
                "semantic_top1_agreement_across_rotations": modal_count / option_count,
                "semantic_entropy_across_rotation_argmax_answers": categorical_entropy(argmax),
                "canonical_vs_permutation_average_TV": total_variation(vectors[0], average),
                "canonical_gold_probability": float(vectors[0, gold]),
            }
        )
    averaged_frame = pd.DataFrame(averaged)
    sensitivity_frame = pd.DataFrame(sensitivity)
    expected_averaged = 8 * 560 * 7
    if len(averaged_frame) != expected_averaged or len(sensitivity_frame) != expected_averaged:
        raise ValueError("Permutation aggregation row count mismatch")
    atomic_parquet(
        root / "outputs/analysis/permutation_averaged_semantic_scores.parquet", averaged_frame
    )
    atomic_parquet(root / "outputs/analysis/permutation_sensitivity.parquet", sensitivity_frame)
    atomic_parquet(root / "outputs/validation/permutation_wise_semantic_scores.parquet", all_rotations)
    return averaged_frame, sensitivity_frame, all_rotations


def response_frame_from_average(averaged: pd.DataFrame) -> pd.DataFrame:
    return averaged.rename(columns={"permutation_averaged_gold_probability": "response"})[
        [
            "model_id",
            "base_question_id",
            "category",
            "semantic_condition",
            "condition",
            "family",
            "dose",
            "track",
            "response",
        ]
    ].copy()


def response_frame_from_rotation0(rotation0: pd.DataFrame) -> pd.DataFrame:
    return rotation0.rename(columns={"gold_semantic_probability": "response"})[
        [
            "model_id",
            "base_question_id",
            "category",
            "semantic_condition",
            "condition",
            "family",
            "dose",
            "track",
            "response",
        ]
    ].copy()


def _aligned(frame: pd.DataFrame, family: str, model_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    subset = frame[frame["condition"].eq("clean") | frame["family"].eq(family)].copy()
    subset["track"] = subset["track"].fillna(-1).astype(int)
    key = ["base_question_id", "category", "dose", "track"]
    pivot = subset.pivot(index=key, columns="model_id", values="response").sort_index()
    if pivot[model_ids].isna().any().any():
        raise ValueError("Endpoint response alignment contains missing model rows")
    metadata = pivot.reset_index()[key]
    values = pivot[model_ids].to_numpy(dtype=np.float64)
    clean_counts = metadata[metadata["track"].eq(-1)].groupby("base_question_id").size()
    high_counts = metadata[metadata["track"].ne(-1)].groupby("base_question_id").size()
    if not clean_counts.eq(1).all() or not high_counts.eq(3).all():
        raise ValueError("Endpoint track counts are not 1 clean + 3 high per question")
    return metadata, values


def _evaluate_endpoint(
    metadata: pd.DataFrame,
    target: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
    evaluation_ids: Iterable[str],
    multiplicities: dict[str, int] | None = None,
) -> dict[str, float]:
    identifiers = metadata["base_question_id"].astype(str)
    mask = identifiers.isin(set(evaluation_ids)).to_numpy()
    residual = np.abs(target - design @ weights)
    multiplier = (
        np.ones(len(metadata), dtype=np.float64)
        if multiplicities is None
        else identifiers.map(multiplicities).to_numpy(dtype=np.float64)
    )
    clean = mask & metadata["track"].eq(-1).to_numpy()
    high = mask & metadata["track"].ne(-1).to_numpy()
    if multiplier[clean].sum() <= 0 or multiplier[high].sum() <= 0:
        raise ValueError("Bootstrap replicate has zero endpoint evaluation mass")
    clean_pier = float(np.sum(multiplier[clean] * residual[clean]) / multiplier[clean].sum())
    high_pier = float(np.sum(multiplier[high] * residual[high]) / multiplier[high].sum())
    return {
        "clean_pier": clean_pier,
        "high_pier": high_pier,
        "endpoint_delta": high_pier - clean_pier,
        "endpoint_design_pier": 0.5 * (clean_pier + high_pier),
    }


def fit_endpoint_design(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
    interface: str,
) -> pd.DataFrame:
    splits = fixed_splits(selected, split_seeds)
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        metadata, values = _aligned(frame, family, model_ids)
        identifiers = metadata["base_question_id"].astype(str)
        for target_index, target in enumerate(model_ids):
            peers = [model for model in model_ids if model != target]
            peer_indices = [model_ids.index(peer) for peer in peers]
            target_values = values[:, target_index]
            design = values[:, peer_indices]
            for split in splits.values():
                fitting = identifiers.isin(split.fitting_ids).to_numpy()
                fit_mass = endpoint_row_weights(metadata)[fitting]
                fitted = fit_mse_simplex(design[fitting], target_values[fitting], fit_mass)
                metrics = _evaluate_endpoint(
                    metadata, target_values, design, fitted, split.evaluation_ids
                )
                rows.append(
                    {
                        "interface": interface,
                        "target": target,
                        "family": family,
                        "split_seed": split.seed,
                        "peers": peers,
                        "weights": fitted.tolist(),
                        **metrics,
                    }
                )
    output = pd.DataFrame(rows)
    summaries = []
    for keys, group in output.groupby(["target", "family"], sort=True):
        deltas = group["endpoint_delta"].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "target": keys[0],
                "family": keys[1],
                "mean_endpoint_delta": float(np.mean(deltas)),
                "positive_split_count": int(np.sum(deltas > 0)),
                "negative_split_count": int(np.sum(deltas < 0)),
                "tie_split_count": int(np.sum(deltas == 0)),
            }
        )
    output = output.merge(pd.DataFrame(summaries), on=["target", "family"], validate="many_to_one")
    if len(output) != 160:
        raise ValueError(f"Endpoint-design row count is {len(output)}, expected 160")
    return output.sort_values(["target", "family", "split_seed"])


def matched_objective_analysis(
    raw_frame: pd.DataFrame,
    permutation_frame: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
) -> pd.DataFrame:
    splits = fixed_splits(selected, split_seeds)
    rows: list[dict[str, Any]] = []
    for interface, frame in (("raw_endpoint", raw_frame), ("permutation_averaged", permutation_frame)):
        for family in FAMILIES:
            metadata, values = _aligned(frame, family, model_ids)
            identifiers = metadata["base_question_id"].astype(str)
            for target_index, target in enumerate(model_ids):
                peers = [model for model in model_ids if model != target]
                peer_indices = [model_ids.index(peer) for peer in peers]
                target_values = values[:, target_index]
                design = values[:, peer_indices]
                for split in splits.values():
                    fitting = identifiers.isin(split.fitting_ids).to_numpy()
                    evaluation = identifiers.isin(split.evaluation_ids).to_numpy()
                    fit_mass = endpoint_row_weights(metadata)[fitting]
                    eval_mass = endpoint_row_weights(metadata)[evaluation]
                    mse_single_index = select_single_peer(
                        design[fitting], target_values[fitting], fit_mass, "mse"
                    )
                    mae_single_index = select_single_peer(
                        design[fitting], target_values[fitting], fit_mass, "mae"
                    )
                    mse_weights = fit_mse_simplex(
                        design[fitting], target_values[fitting], fit_mass
                    )
                    mae_weights = fit_mae_simplex(
                        design[fitting], target_values[fitting], fit_mass
                    )
                    y_eval, x_eval = target_values[evaluation], design[evaluation]
                    rows.append(
                        {
                            "interface": interface,
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "MSE_selected_single_peer": peers[mse_single_index],
                            "MSE_single_error": weighted_mae(
                                y_eval, x_eval[:, mse_single_index], eval_mass
                            ),
                            "MSE_convex_error": weighted_mae(y_eval, x_eval @ mse_weights, eval_mass),
                            "MSE_convex_weights": mse_weights.tolist(),
                            "MAE_selected_single_peer": peers[mae_single_index],
                            "MAE_single_error": weighted_mae(
                                y_eval, x_eval[:, mae_single_index], eval_mass
                            ),
                            "MAE_convex_error": weighted_mae(y_eval, x_eval @ mae_weights, eval_mass),
                            "MAE_convex_weights": mae_weights.tolist(),
                        }
                    )
    output = pd.DataFrame(rows)
    output["MSE_absolute_improvement"] = output["MSE_single_error"] - output["MSE_convex_error"]
    output["MAE_absolute_improvement"] = output["MAE_single_error"] - output["MAE_convex_error"]
    if len(output) != 320:
        raise ValueError("Matched-objective output row count mismatch")
    return output.sort_values(["interface", "target", "family", "split_seed"])


def sibling_removal_analysis(
    raw_frame: pd.DataFrame,
    permutation_frame: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
    sibling_directions: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = fixed_splits(selected, split_seeds)
    summaries: dict[tuple[str, str, int], dict[str, Any]] = {}
    null_rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for interface, frame, prefix in (
        ("raw_endpoint", raw_frame, "raw_endpoint"),
        ("permutation_averaged", permutation_frame, "permutation_averaged"),
    ):
        for family in FAMILIES:
            metadata, values = _aligned(frame, family, model_ids)
            identifiers = metadata["base_question_id"].astype(str)
            for target, sibling in sibling_directions.items():
                target_index = model_ids.index(target)
                peers = [model for model in model_ids if model != target]
                peer_indices = [model_ids.index(peer) for peer in peers]
                y = values[:, target_index]
                x = values[:, peer_indices]
                for split in splits.values():
                    fitting = identifiers.isin(split.fitting_ids).to_numpy()
                    fit_mass = endpoint_row_weights(metadata)[fitting]
                    baseline_weights = fit_mse_simplex(x[fitting], y[fitting], fit_mass)
                    baseline = _evaluate_endpoint(
                        metadata, y, x, baseline_weights, split.evaluation_ids
                    )["endpoint_design_pier"]
                    removals: dict[str, float] = {}
                    for removed_index, removed in enumerate(peers):
                        retained = [index for index in range(len(peers)) if index != removed_index]
                        fitted = fit_mse_simplex(x[fitting][:, retained], y[fitting], fit_mass)
                        value = _evaluate_endpoint(
                            metadata, y, x[:, retained], fitted, split.evaluation_ids
                        )["endpoint_design_pier"]
                        removals[removed] = value - baseline
                    sibling_inflation = removals[sibling]
                    non_siblings = [peer for peer in peers if peer != sibling]
                    null_values = np.asarray([removals[peer] for peer in non_siblings])
                    key = (target, family, split.seed)
                    summary = summaries.setdefault(
                        key,
                        {
                            "target": target,
                            "removed_sibling": sibling,
                            "family": family,
                            "split_seed": split.seed,
                        },
                    )
                    summary[f"{prefix}_all_peer_pier"] = baseline
                    summary[f"{prefix}_sibling_inflation"] = sibling_inflation
                    summary[f"{prefix}_largest_non_sibling_inflation"] = float(null_values.max())
                    summary[f"{prefix}_sibling_rank"] = 1 + int(np.sum(null_values > sibling_inflation))
                    summary[f"{prefix}_sibling_is_largest"] = bool(sibling_inflation > null_values.max())
                    for peer in non_siblings:
                        null_key = (*key, peer)
                        record = null_rows.setdefault(
                            null_key,
                            {
                                "target": target,
                                "removed_sibling": sibling,
                                "family": family,
                                "split_seed": split.seed,
                                "removed_non_sibling": peer,
                                "null_method": "exact_enumeration",
                            },
                        )
                        record[f"{prefix}_non_sibling_inflation"] = removals[peer]
                        record[f"{prefix}_sibling_inflation"] = sibling_inflation
    sibling = pd.DataFrame(summaries.values())
    null = pd.DataFrame(null_rows.values())
    largest = (
        sibling.groupby(["target", "family"], as_index=False)[
            "permutation_averaged_sibling_is_largest"
        ]
        .sum()
        .rename(
            columns={
                "permutation_averaged_sibling_is_largest": "splits_sibling_largest"
            }
        )
    )
    sibling = sibling.merge(largest, on=["target", "family"], validate="many_to_one")
    null = null.merge(largest, on=["target", "family"], validate="many_to_one")
    if len(sibling) != 80 or len(null) != 480:
        raise ValueError(f"Sibling/null row counts mismatch: {len(sibling)}/{len(null)}")
    return (
        sibling.sort_values(["target", "family", "split_seed"]),
        null.sort_values(["target", "family", "split_seed", "removed_non_sibling"]),
    )


def frozen_weight_transfer(
    root: Path,
    permutation_frame: pd.DataFrame,
    selected: pd.DataFrame,
    model_ids: list[str],
    split_seeds: list[int],
) -> pd.DataFrame:
    config = load_json(root / "configs/experiment_v23.json")
    v22 = Path(config["source_roots"]["v22"]).resolve(strict=True)
    saved = pd.read_parquet(v22 / "outputs/analysis/trackwise_primary_weights.parquet")
    v22_endpoint = pd.read_parquet(v22 / "outputs/analysis/trackwise_endpoint_effects.parquet")
    splits = fixed_splits(selected, split_seeds)
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        metadata, values = _aligned(permutation_frame, family, model_ids)
        for target_index, target in enumerate(model_ids):
            peers = [model for model in model_ids if model != target]
            peer_indices = [model_ids.index(peer) for peer in peers]
            y, x = values[:, target_index], values[:, peer_indices]
            for split in splits.values():
                subset = saved[
                    saved["target"].eq(target)
                    & saved["family"].eq(family)
                    & saved["split_seed"].eq(split.seed)
                ].set_index("peer")
                weights = subset.loc[peers, "weight"].to_numpy(dtype=np.float64)
                metrics = _evaluate_endpoint(metadata, y, x, weights, split.evaluation_ids)
                old = v22_endpoint[
                    v22_endpoint["target"].eq(target)
                    & v22_endpoint["family"].eq(family)
                    & v22_endpoint["split_seed"].eq(split.seed)
                ].iloc[0]
                rows.append(
                    {
                        "target": target,
                        "family": family,
                        "split_seed": split.seed,
                        "frozen_v22_weights": weights.tolist(),
                        "v22_raw_full_design_delta": float(old["trackwise_delta"]),
                        "permutation_endpoint_frozen_clean_pier": metrics["clean_pier"],
                        "permutation_endpoint_frozen_high_pier": metrics["high_pier"],
                        "permutation_endpoint_frozen_delta": metrics["endpoint_delta"],
                        "permutation_endpoint_frozen_pier": metrics["endpoint_design_pier"],
                    }
                )
    output = pd.DataFrame(rows)
    if len(output) != 160:
        raise ValueError("Frozen-weight transfer row count mismatch")
    return output.sort_values(["target", "family", "split_seed"])


def generation_validation(
    root: Path, all_rotations: pd.DataFrame, rotation0: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_json(root / "configs/experiment_v23.json")
    models = load_json(root / "configs/resolved_models_v23.json")["models"]
    generation_frames: list[pd.DataFrame] = []
    for model in models:
        paths = sorted((root / "outputs/generation" / slug(model["id"])).glob("shard_*.parquet"))
        generation_frames.extend(pd.read_parquet(path) for path in paths)
    generated = pd.concat(generation_frames, ignore_index=True)
    if not generated["schema_version"].eq(GENERATION_SCHEMA).all():
        raise ValueError("Long-generation schema mismatch")
    score = all_rotations[
        ["model_id", "prompt_id", "argmax_semantic_option", "semantic_remapped_probabilities"]
    ].rename(columns={"prompt_id": "scoring_prompt_id"})
    merged = generated.merge(
        score, on=["model_id", "scoring_prompt_id"], validate="one_to_one"
    )
    merged["score_generation_agreement"] = (
        merged["argmax_semantic_option"].astype("Int64")
        == merged["semantic_final_option"].astype("Int64")
    ).fillna(False)
    frequency_rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(
        ["model_id", "base_question_id", "semantic_condition"], sort=True
    ):
        group = group.sort_values("permutation_id")
        option_count = int(group["option_count"].iloc[0])
        if len(group) != option_count:
            raise ValueError(f"Generation cyclic set incomplete: {keys}")
        counts = np.zeros(option_count, dtype=np.float64)
        answers: list[int] = []
        for value in group["semantic_final_option"]:
            if pd.notna(value):
                counts[int(value)] += 1
                answers.append(int(value))
        frequency = counts / option_count
        malformed_frequency = 1.0 - float(frequency.sum())
        score_average = np.mean(
            np.stack(
                [np.asarray(value, dtype=np.float64) for value in group["semantic_remapped_probabilities"]]
            ),
            axis=0,
        )
        canonical_score = np.asarray(
            group.iloc[0]["semantic_remapped_probabilities"], dtype=np.float64
        )
        extended_generated = np.append(frequency, malformed_frequency)
        extended_score = np.append(score_average, 0.0)
        extended_canonical = np.append(canonical_score, 0.0)
        modal = max(Counter(answers).values()) / option_count if answers else 0.0
        first = group.iloc[0]
        gold = int(first["gold_semantic_option"])
        frequency_rows.append(
            {
                "model": keys[0],
                "model_id": keys[0],
                "base_question_id": keys[1],
                "semantic_condition": keys[2],
                "condition": first["condition"],
                "option_count": option_count,
                "generated_semantic_frequency": frequency.tolist(),
                "malformed_frequency": malformed_frequency,
                "score_permutation_average": score_average.tolist(),
                "canonical_score": canonical_score.tolist(),
                "TV_distance": total_variation(extended_generated, extended_score),
                "canonical_score_vs_generated_frequency_TV": total_variation(
                    extended_generated, extended_canonical
                ),
                "JS_divergence": js_divergence(extended_generated, extended_score),
                "top1_agreement": bool(
                    answers and int(np.argmax(frequency)) == int(np.argmax(score_average))
                ),
                "gold_answer_frequency": float(frequency[gold]),
                "gold_answer_score_probability": float(score_average[gold]),
                "gold_answer_probability_frequency_gap": float(score_average[gold] - frequency[gold]),
                "generated_answer_rotation_stability": modal,
                "number_distinct_semantic_generated_answers": len(set(answers)),
            }
        )
    frequency_frame = pd.DataFrame(frequency_rows)

    old_frames: list[pd.DataFrame] = []
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    for model in models:
        old_frames.extend(
            pd.read_parquet(path)
            for path in sorted(
                (v2 / "outputs/generation_validation" / slug(model["id"])).glob("shard_*.parquet")
            )
        )
    old = pd.concat(old_frames, ignore_index=True)
    source_argmax = rotation0[
        ["model_id", "source_v2_prompt_id", "argmax_semantic_option"]
    ].rename(columns={"source_v2_prompt_id": "scoring_prompt_id"})
    old = old.merge(source_argmax, on=["model_id", "scoring_prompt_id"], validate="one_to_one")
    old["old_agreement"] = old["argmax_semantic_option"].map(lambda value: LABEL_FROM_INDEX(value)) == old[
        "extracted_label"
    ]
    old_summary = (
        old.groupby(["model_id", "condition"], as_index=False)
        .agg(old_16_token_malformed_rate=("malformed", "mean"), old_score_generation_agreement=("old_agreement", "mean"))
    )
    summary_rows: list[dict[str, Any]] = []
    frequency_lookup = frequency_frame.groupby(["model_id", "condition"])
    for keys, group in merged.groupby(["model_id", "condition"], sort=True):
        canonical = group[group["permutation_id"].eq(0)]
        freq = frequency_lookup.get_group(keys)
        old_row = old_summary[
            old_summary["model_id"].eq(keys[0]) & old_summary["condition"].eq(keys[1])
        ]
        summary_rows.append(
            {
                "model": keys[0],
                "model_id": keys[0],
                "condition": keys[1],
                "canonical_score_generation_agreement": float(
                    canonical["score_generation_agreement"].mean()
                ),
                "all_rotation_score_generation_agreement": float(
                    group["score_generation_agreement"].mean()
                ),
                "malformed_rate": float(group["malformed"].mean()),
                "median_generation_tokens": float(group["generation_token_count"].median()),
                "EOS_reached_rate": float(group["eos_reached"].mean()),
                "generated_answer_rotation_stability": float(
                    freq["generated_answer_rotation_stability"].mean()
                ),
                "score_vs_generated_frequency_TV": float(freq["TV_distance"].mean()),
                "score_vs_generated_frequency_JS": float(freq["JS_divergence"].mean()),
                "old_16_token_malformed_rate": (
                    float(old_row["old_16_token_malformed_rate"].iloc[0]) if len(old_row) else math.nan
                ),
                "old_score_generation_agreement": (
                    float(old_row["old_score_generation_agreement"].iloc[0]) if len(old_row) else math.nan
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if len(summary) != 24 or len(frequency_frame) != 8 * 140 * 3:
        raise ValueError("Generation validation row counts mismatch")
    return summary.sort_values(["model", "condition"]), frequency_frame


def LABEL_FROM_INDEX(value: Any) -> str:
    labels = "ABCDEFGHIJ"
    return labels[int(value)]


def run_point_analyses(root: Path) -> dict[str, Any]:
    config = load_json(root / "configs/experiment_v23.json")
    models = load_json(root / "configs/resolved_models_v23.json")["models"]
    model_ids = [record["id"] for record in models]
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    selected = pd.read_json(
        v2 / "data/mmlu_pro_selected_560.jsonl", lines=True, dtype={"base_question_id": str}
    )
    averaged, sensitivity, all_rotations = assemble_permutation_scores(root)
    rotation0 = load_rotation0_scores(root, models)
    raw_frame = response_frame_from_rotation0(rotation0)
    permutation_frame = response_frame_from_average(averaged)
    atomic_parquet(root / "outputs/validation/raw_endpoint_response.parquet", raw_frame)
    atomic_parquet(
        root / "outputs/validation/permutation_endpoint_response.parquet", permutation_frame
    )
    raw_endpoint = fit_endpoint_design(
        raw_frame, selected, model_ids, config["split_seeds"], "raw_endpoint"
    )
    permutation_endpoint = fit_endpoint_design(
        permutation_frame,
        selected,
        model_ids,
        config["split_seeds"],
        "permutation_averaged",
    )
    matched = matched_objective_analysis(
        raw_frame, permutation_frame, selected, model_ids, config["split_seeds"]
    )
    sibling, null = sibling_removal_analysis(
        raw_frame,
        permutation_frame,
        selected,
        model_ids,
        config["split_seeds"],
        config["sibling_directions"],
    )
    frozen = frozen_weight_transfer(
        root, permutation_frame, selected, model_ids, config["split_seeds"]
    )
    generation, frequencies = generation_validation(root, all_rotations, rotation0)
    outputs = {
        "raw_endpoint_design_results.parquet": raw_endpoint,
        "permutation_endpoint_design_results.parquet": permutation_endpoint,
        "frozen_weight_interface_transfer.parquet": frozen,
        "matched_objective_convexity.parquet": matched,
        "permutation_sibling_removal.parquet": sibling,
        "non_sibling_removal_null.parquet": null,
        "long_generation_validation.parquet": generation,
        "generated_semantic_frequency.parquet": frequencies,
    }
    for name, frame in outputs.items():
        atomic_parquet(root / "outputs/analysis" / name, frame)
    status = {
        "schema_version": "pier_v23_point_analysis_v1",
        "completed_at": utc_now(),
        "passed": True,
        "outputs": {
            name: {"row_count": len(frame), "sha256": sha256_file(root / "outputs/analysis" / name)}
            for name, frame in outputs.items()
        },
        "permutation_sensitivity_rows": len(sensitivity),
    }
    return status
