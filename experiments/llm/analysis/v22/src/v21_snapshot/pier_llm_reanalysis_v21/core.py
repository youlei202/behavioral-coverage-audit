from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pier_llm.analysis import (
    Split,
    aligned_response_arrays,
    response_from_row,
    stratified_question_split,
)
from pier_llm.solver import ProjectionResult
from pier_llm.solver import fit_simplex_projection as v2_fit_simplex_projection

from .utils import load_config, stable_seed

FAMILIES = ("irrelevant_context", "content_deletion")


class _NullWriter:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


_NULL_WRITER = _NullWriter()


def fit_projection(design: np.ndarray, target: np.ndarray) -> ProjectionResult:
    """Run the unchanged V2 projection while suppressing handled solver chatter."""
    with warnings.catch_warnings(), redirect_stdout(_NULL_WRITER):
        warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
        )
        return v2_fit_simplex_projection(design, target)


@dataclass(frozen=True)
class PrimaryFit:
    dose_results: pd.DataFrame
    weights: np.ndarray
    selected_peer_index: int
    projection: ProjectionResult
    metadata: pd.DataFrame
    target_values: np.ndarray
    design: np.ndarray


def fixed_splits(selected: pd.DataFrame) -> dict[int, Split]:
    return {
        int(seed): stratified_question_split(selected, int(seed))
        for seed in load_config()["split_seeds"]
    }


def high_dose(family: str) -> float:
    return float(load_config()["families"][family]["high_dose"])


def primary_fit(
    aggregated: pd.DataFrame,
    target: str,
    peers: list[str],
    split: Split,
) -> PrimaryFit:
    metadata, target_values, design = aligned_response_arrays(aggregated, target, peers)
    if target_values.ndim != 1 or design.ndim != 2:
        raise ValueError("The corrected scalar pipeline requires scalar response arrays")
    identifiers = metadata["base_question_id"].astype(str)
    fitting = identifiers.isin(split.fitting_ids).to_numpy()
    evaluation = identifiers.isin(split.evaluation_ids).to_numpy()
    projection = fit_projection(design[fitting], target_values[fitting])
    fit_peer_mae = np.mean(
        np.abs(design[fitting] - target_values[fitting, None]), axis=0
    )
    selected_peer_index = int(np.argmin(fit_peer_mae))
    rows: list[dict[str, Any]] = []
    for dose in sorted(float(value) for value in metadata.loc[evaluation, "dose"].unique()):
        mask = evaluation & metadata["dose"].astype(float).eq(dose).to_numpy()
        truth = target_values[mask]
        prediction = design[mask] @ projection.weights
        single = design[mask, selected_peer_index]
        convex_error = float(np.mean(np.abs(truth - prediction)))
        single_error = float(np.mean(np.abs(truth - single)))
        rows.append(
            {
                "target": target,
                "family": "",
                "split_seed": split.seed,
                "dose": dose,
                "evaluation_question_count": int(metadata.loc[mask, "base_question_id"].nunique()),
                "convex_error": convex_error,
                "single_error": single_error,
                "absolute_improvement": single_error - convex_error,
                "relative_improvement": (single_error - convex_error) / (single_error + 1e-10),
                "gap_ratio": single_error / (convex_error + 1e-10),
                "fit_selected_single_peer": peers[selected_peer_index],
            }
        )
    truth = target_values[evaluation]
    prediction = design[evaluation] @ projection.weights
    single = design[evaluation, selected_peer_index]
    convex_error = float(np.mean(np.abs(truth - prediction)))
    single_error = float(np.mean(np.abs(truth - single)))
    rows.append(
        {
            "target": target,
            "family": "",
            "split_seed": split.seed,
            "dose": math.nan,
            "evaluation_question_count": int(metadata.loc[evaluation, "base_question_id"].nunique()),
            "convex_error": convex_error,
            "single_error": single_error,
            "absolute_improvement": single_error - convex_error,
            "relative_improvement": (single_error - convex_error) / (single_error + 1e-10),
            "gap_ratio": single_error / (convex_error + 1e-10),
            "fit_selected_single_peer": peers[selected_peer_index],
        }
    )
    return PrimaryFit(
        dose_results=pd.DataFrame(rows),
        weights=projection.weights,
        selected_peer_index=selected_peer_index,
        projection=projection,
        metadata=metadata,
        target_values=target_values,
        design=design,
    )


def paired_endpoint_rows(primary_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, family, split_seed), group in primary_results.groupby(
        ["target", "family", "split_seed"], sort=True
    ):
        clean = group[group["dose"].eq(0.0)]
        high = group[group["dose"].eq(high_dose(str(family)))]
        if len(clean) != 1 or len(high) != 1:
            raise ValueError(f"Missing paired endpoint for {target}, {family}, {split_seed}")
        clean_pier = float(clean.iloc[0]["convex_error"])
        high_pier = float(high.iloc[0]["convex_error"])
        delta = high_pier - clean_pier
        rows.append(
            {
                "target": target,
                "family": family,
                "split_seed": int(split_seed),
                "clean_dose": 0.0,
                "high_dose": high_dose(str(family)),
                "pier_clean": clean_pier,
                "pier_high": high_pier,
                "delta_absolute": delta,
                "delta_relative": delta / (clean_pier + 1e-10),
            }
        )
    return pd.DataFrame(rows)


def summarize_split_effects(
    effects: pd.DataFrame,
    *,
    metric: str,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in effects.groupby(group_columns, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = group[metric].astype(float).to_numpy()
        record = dict(zip(group_columns, key_tuple, strict=True))
        record.update(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "positive_split_count": int(np.sum(values > 0)),
                "negative_split_count": int(np.sum(values < 0)),
                "tie_split_count": int(np.sum(values == 0)),
                "split_count": len(values),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def stratified_cluster_draw(
    selected: pd.DataFrame,
    source_ids: Iterable[str],
    *,
    seed: int,
) -> list[str]:
    source = {str(value) for value in source_ids}
    available = selected[selected["base_question_id"].astype(str).isin(source)][
        ["base_question_id", "category"]
    ].drop_duplicates()
    rng = np.random.default_rng(seed)
    draws: list[str] = []
    for _category, group in available.groupby("category", sort=True):
        identifiers = sorted(group["base_question_id"].astype(str))
        sampled = rng.choice(identifiers, size=len(identifiers), replace=True)
        draws.extend(str(value) for value in sampled)
    if len(draws) != len(source):
        raise ValueError("Stratified draw lost source identities")
    return draws


def row_indices_for_draw(
    metadata: pd.DataFrame,
    draw: Iterable[str],
    *,
    dose: float | None = None,
) -> np.ndarray:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in metadata.reset_index(drop=True).iterrows():
        if dose is None or math.isclose(float(row["dose"]), float(dose), abs_tol=1e-12):
            indices[str(row["base_question_id"])].append(int(index))
    selected_rows: list[int] = []
    for identifier in draw:
        rows = indices.get(str(identifier), [])
        if not rows:
            raise ValueError(f"No aligned response rows for sampled question {identifier}")
        selected_rows.extend(rows)
    return np.asarray(selected_rows, dtype=np.int64)


def bootstrap_draws(
    selected: pd.DataFrame,
    split: Split,
    *,
    analysis_type: str,
    target: str,
    family: str,
    replicate: int,
) -> tuple[list[str], list[str], int]:
    seed = stable_seed(20260828, analysis_type, target, family, split.seed, replicate)
    fit_draw = stratified_cluster_draw(
        selected, split.fitting_ids, seed=stable_seed(seed, "fit")
    )
    evaluation_draw = stratified_cluster_draw(
        selected, split.evaluation_ids, seed=stable_seed(seed, "evaluation")
    )
    if set(fit_draw).intersection(evaluation_draw):
        raise RuntimeError("Bootstrap fitting/evaluation source identities overlap")
    return fit_draw, evaluation_draw, seed


def fit_from_draw(
    metadata: pd.DataFrame,
    design: np.ndarray,
    target_values: np.ndarray,
    fit_draw: Iterable[str],
) -> ProjectionResult:
    rows = row_indices_for_draw(metadata, fit_draw)
    return fit_projection(design[rows], target_values[rows])


def endpoint_delta_for_draw(
    metadata: pd.DataFrame,
    design: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
    evaluation_draw: Iterable[str],
    family: str,
) -> tuple[float, float, float]:
    clean_rows = row_indices_for_draw(metadata, evaluation_draw, dose=0.0)
    high_rows = row_indices_for_draw(metadata, evaluation_draw, dose=high_dose(family))
    clean = float(np.mean(np.abs(target_values[clean_rows] - design[clean_rows] @ weights)))
    high = float(np.mean(np.abs(target_values[high_rows] - design[high_rows] @ weights)))
    return clean, high, high - clean


def overall_error_for_draw(
    metadata: pd.DataFrame,
    design: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
    evaluation_draw: Iterable[str],
) -> float:
    rows = row_indices_for_draw(metadata, evaluation_draw)
    return float(np.mean(np.abs(target_values[rows] - design[rows] @ weights)))


def best_single_for_draw(
    metadata: pd.DataFrame,
    design: np.ndarray,
    target_values: np.ndarray,
    fit_draw: Iterable[str],
) -> int:
    rows = row_indices_for_draw(metadata, fit_draw)
    errors = np.mean(np.abs(design[rows] - target_values[rows, None]), axis=0)
    return int(np.argmin(errors))


def multiplicities(draw: Iterable[str]) -> Counter[str]:
    return Counter(str(value) for value in draw)


def track_response_frame(scores: pd.DataFrame, family: str) -> pd.DataFrame:
    subset = scores[scores["condition"].eq("clean") | scores["family"].eq(family)].copy()
    subset["response"] = [
        response_from_row(row, "gold_probability") for _, row in subset.iterrows()
    ]
    subset["track_key"] = subset["track"].fillna(-1).astype(int)
    columns = [
        "model_id",
        "base_question_id",
        "category",
        "dose",
        "track_key",
        "response",
    ]
    frame = subset[columns].copy()
    if frame.duplicated(columns[:-1]).any():
        raise ValueError(f"Duplicate track response rows for {family}")
    return frame


def track_aligned_arrays(
    track_frame: pd.DataFrame,
    target: str,
    peers: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    index_columns = ["base_question_id", "category", "dose", "track_key"]
    pivot = track_frame[track_frame["model_id"].isin([target, *peers])].pivot(
        index=index_columns, columns="model_id", values="response"
    )
    required = [target, *peers]
    if pivot[required].isna().any().any():
        raise ValueError("Track response alignment contains missing model rows")
    pivot = pivot.sort_index().reset_index()
    return (
        pivot[index_columns],
        pivot[target].to_numpy(dtype=np.float64),
        pivot[peers].to_numpy(dtype=np.float64),
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))
