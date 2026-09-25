from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .utils import (
    PHYSICAL_ROOT,
    atomic_write_json,
    atomic_write_text,
    environment_manifest,
    load_config,
    load_json,
    sha256_file,
    stable_u64,
)
from .v2_solver import ProjectionResult, feasible_weight_intervals, fit_simplex_projection

SCORE_SCHEMA = "pier_raw_label_scores_v2"
FAMILIES = ("irrelevant_context", "content_deletion")


class _NullWriter:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None


_NULL_WRITER = _NullWriter()


@dataclass(frozen=True)
class Split:
    seed: int
    fitting_ids: frozenset[str]
    evaluation_ids: frozenset[str]


@dataclass(frozen=True)
class RootResolution:
    source: str
    declared_logical: Path
    declared_physical_alias: Path
    selected_path: Path
    resolved_physical: Path
    attempted_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AlignedData:
    metadata: pd.DataFrame
    target: np.ndarray
    design: np.ndarray
    peers: tuple[str, ...]


@dataclass(frozen=True)
class WeightedFit:
    projection: ProjectionResult
    normalized_weighted_objective: float
    row_weight_total: float
    fitted_row_count: int


def resolve_source_root(source: str) -> RootResolution:
    specification = load_config()["source_roots"][source]
    logical = Path(specification["logical"])
    physical_alias = Path(specification["physical_alias"])
    attempted = (logical, physical_alias)
    selected: Path | None = None
    for candidate in attempted:
        try:
            candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        selected = candidate
        break
    if selected is None:
        raise FileNotFoundError(
            f"No available {source} source root; attempted: "
            + ", ".join(str(path) for path in attempted)
        )
    resolved = Path(os.path.realpath(selected)).resolve(strict=True)
    return RootResolution(
        source=source,
        declared_logical=logical,
        declared_physical_alias=physical_alias,
        selected_path=selected,
        resolved_physical=resolved,
        attempted_paths=attempted,
    )


def source_resolutions() -> dict[str, RootResolution]:
    return {source: resolve_source_root(source) for source in ("v2", "v21")}


def _v2_relative_inputs(root: Path) -> list[Path]:
    fixed = [
        "configs/experiment.json",
        "configs/resolved_models.json",
        "data/mmlu_pro_selected_560.jsonl",
        "data/interventions/irrelevant_context_v2.jsonl",
        "data/interventions/content_deletion_v2.jsonl",
        "data/interventions/option_permutation_control.jsonl",
        "data/manifests/mmlu_pro_selected_560.jsonl.sha256",
        "data/manifests/irrelevant_context_v2.jsonl.sha256",
        "data/manifests/content_deletion_v2.jsonl.sha256",
        "data/manifests/option_permutation_control.jsonl.sha256",
        "env/requirements.lock.txt",
        "src/pier_llm/solver.py",
        "src/pier_llm/analysis.py",
        "src/pier_llm/utils.py",
        "status/B200_INFERENCE_COMPLETE.json",
        "status/EXPERIMENT_COMPLETE.json",
    ]
    paths = [root / relative for relative in fixed]
    paths.extend(sorted((root / "outputs/raw_scores").glob("*/shard_*.parquet")))
    paths.extend(sorted((root / "outputs/raw_scores").glob("*/shard_*.meta.json")))
    return sorted(set(paths), key=lambda path: str(path.relative_to(root)))


def _v21_relative_inputs(root: Path) -> list[Path]:
    fixed = [
        "configs/reanalysis_v21.json",
        "outputs/analysis/V2_1_CORRECTED_REANALYSIS_REPORT.md",
        "outputs/analysis/primary_dose_results.parquet",
        "outputs/analysis/primary_weights.parquet",
        "outputs/analysis/paired_endpoint_effects.parquet",
        "outputs/analysis/convex_vs_single_corrected.parquet",
        "outputs/analysis/peer_removal_null_corrected.parquet",
        "outputs/manifests/INPUTS_BEFORE.sha256",
        "outputs/manifests/INPUTS_AFTER.sha256",
        "outputs/manifests/input_inventory_before.json",
        "outputs/manifests/input_inventory_after.json",
        "outputs/manifests/reanalysis_environment.json",
        "outputs/tables/table_v21_endpoint_effects.csv",
        "outputs/tables/table_v21_convexity_gap.csv",
        "outputs/tables/table_v21_sibling_removal.csv",
        "status/REANALYSIS_COMPLETE.json",
    ]
    paths = [root / relative for relative in fixed]
    paths.extend(sorted((root / "src/pier_llm_reanalysis_v21").glob("*.py")))
    return sorted(set(paths), key=lambda path: str(path.relative_to(root)))


def input_source_paths() -> dict[str, list[Path]]:
    resolutions = source_resolutions()
    return {
        "v2": _v2_relative_inputs(resolutions["v2"].resolved_physical),
        "v21": _v21_relative_inputs(resolutions["v21"].resolved_physical),
    }


def _file_record(path: Path, resolution: RootResolution) -> dict[str, Any]:
    relative = path.relative_to(resolution.resolved_physical)
    logical = resolution.declared_logical / relative
    stat = path.stat()
    return {
        "source": resolution.source,
        "relative_path": str(relative),
        "absolute_logical_path": str(logical),
        "resolved_physical_path": str(path.resolve(strict=True)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def build_input_inventory() -> list[dict[str, Any]]:
    resolutions = source_resolutions()
    paths = input_source_paths()
    records: list[dict[str, Any]] = []
    for source in ("v2", "v21"):
        for path in paths[source]:
            if not path.is_file():
                raise FileNotFoundError(f"Required {source} input is unavailable: {path}")
            records.append(_file_record(path, resolutions[source]))
    return sorted(records, key=lambda record: (record["source"], record["relative_path"]))


def inventory_sha256_lines(records: list[dict[str, Any]]) -> str:
    return "".join(
        f"{record['sha256']}  {record['source']}/{record['relative_path']}\n"
        for record in records
    )


def compare_inventories(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    left = {(record["source"], record["relative_path"]): record for record in before}
    right = {(record["source"], record["relative_path"]): record for record in after}
    differences: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences.append(
                {
                    "source": key[0],
                    "relative_path": key[1],
                    "before": left.get(key),
                    "after": right.get(key),
                }
            )
    return differences


def _verify_v21_raw_inventory(v2_records: list[dict[str, Any]], v21_root: Path) -> int:
    before = load_json(v21_root / "outputs/manifests/input_inventory_before.json")
    after = load_json(v21_root / "outputs/manifests/input_inventory_after.json")
    before_map = {record["relative_path"]: record for record in before}
    after_map = {record["relative_path"]: record for record in after}
    verified = 0
    for record in v2_records:
        relative = record["relative_path"]
        if not relative.startswith("outputs/raw_scores/"):
            continue
        for label, inventory in (("before", before_map), ("after", after_map)):
            expected = inventory.get(relative)
            if expected is None:
                raise ValueError(f"V2.1 {label} inventory lacks raw score {relative}")
            for field in ("size", "mtime_ns", "sha256"):
                if expected[field] != record[field]:
                    raise ValueError(
                        f"V2 raw score differs from V2.1 {label} inventory: "
                        f"{relative} field={field} expected={expected[field]!r} "
                        f"observed={record[field]!r}"
                    )
        verified += 1
    if verified == 0:
        raise RuntimeError("No raw V2 score files were verified against V2.1 inventories")
    return verified


def _metadata_path(parquet: Path) -> Path:
    return parquet.with_suffix(".meta.json")


def load_validated_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    roots = source_resolutions()
    v2 = roots["v2"].resolved_physical
    config = load_config()
    for status_name in ("B200_INFERENCE_COMPLETE.json", "EXPERIMENT_COMPLETE.json"):
        status = load_json(v2 / "status" / status_name)
        if status.get("complete") is not True:
            raise RuntimeError(f"Original V2 completion marker is incomplete: {status_name}")
    resolved = load_json(v2 / "configs/resolved_models.json")
    roster = [record["id"] for record in resolved["models"]]
    if roster != config["model_roster"]:
        raise ValueError("The V2 model roster differs from the fixed V2.2 roster")
    shard_paths = sorted((v2 / "outputs/raw_scores").glob("*/shard_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"Raw V2 score files are unavailable under {v2}")
    frames: list[pd.DataFrame] = []
    for parquet in shard_paths:
        metadata = load_json(_metadata_path(parquet))
        if metadata.get("schema_version") != SCORE_SCHEMA:
            raise ValueError(f"Raw-score schema mismatch: {parquet}")
        if metadata.get("sha256") != sha256_file(parquet):
            raise ValueError(f"Raw-score checksum mismatch: {parquet}")
        frame = pd.read_parquet(parquet)
        if len(frame) != int(metadata["row_count"]):
            raise ValueError(f"Raw-score row-count mismatch: {parquet}")
        if not frame["schema_version"].eq(SCORE_SCHEMA).all():
            raise ValueError(f"Raw-score row schema mismatch: {parquet}")
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    if len(scores) != 125_440:
        raise ValueError(f"Expected 125440 raw-score rows, found {len(scores)}")
    if scores.duplicated(["model_id", "prompt_id"]).any():
        raise ValueError("Duplicate (model_id, prompt_id) rows in raw scores")
    expected_counts = {model: 15_680 for model in roster}
    if scores.groupby("model_id").size().to_dict() != expected_counts:
        raise ValueError("Per-model raw-score counts do not match the fixed design")
    selected = pd.read_json(
        v2 / "data/mmlu_pro_selected_560.jsonl",
        lines=True,
        dtype={"base_question_id": str},
    )
    counts = selected.groupby("category")["base_question_id"].nunique()
    if len(selected) != 560 or len(counts) != 14 or not counts.eq(40).all():
        raise ValueError("The fixed 560-question stratified design is malformed")
    return scores, selected, resolved


def validate_and_record_before() -> dict[str, Any]:
    roots = source_resolutions()
    inventory = build_input_inventory()
    v2_records = [record for record in inventory if record["source"] == "v2"]
    verified_raw_records = _verify_v21_raw_inventory(
        v2_records, roots["v21"].resolved_physical
    )
    scores, selected, resolved = load_validated_inputs()
    manifests = PHYSICAL_ROOT / "manifests"
    atomic_write_json(manifests / "input_inventory_before.json", inventory)
    atomic_write_text(manifests / "INPUTS_BEFORE.sha256", inventory_sha256_lines(inventory))
    resolutions_payload = {
        source: {
            "declared_logical": str(record.declared_logical),
            "declared_physical_alias": str(record.declared_physical_alias),
            "attempted_paths": [str(path) for path in record.attempted_paths],
            "selected_path": str(record.selected_path),
            "resolved_physical": str(record.resolved_physical),
        }
        for source, record in roots.items()
    }
    atomic_write_json(manifests / "resolved_input_roots.json", resolutions_payload)
    atomic_write_json(manifests / "reanalysis_environment.json", environment_manifest())
    return {
        "raw_score_rows": len(scores),
        "selected_question_rows": len(selected),
        "models": len(resolved["models"]),
        "inventory_file_count": len(inventory),
        "v21_verified_raw_inventory_records": verified_raw_records,
    }


def recheck_and_record_after() -> dict[str, Any]:
    before = load_json(PHYSICAL_ROOT / "manifests/input_inventory_before.json")
    after = build_input_inventory()
    atomic_write_json(PHYSICAL_ROOT / "manifests/input_inventory_after.json", after)
    atomic_write_text(
        PHYSICAL_ROOT / "manifests/INPUTS_AFTER.sha256", inventory_sha256_lines(after)
    )
    differences = compare_inventories(before, after)
    if differences:
        raise RuntimeError(f"Immutable V2/V2.1 inputs changed: {differences[:3]}")
    return {"byte_identical": True, "file_count": len(after), "differences": []}


def stratified_question_split(selected: pd.DataFrame, seed: int) -> Split:
    unique = selected[["base_question_id", "category"]].drop_duplicates()
    fitting: set[str] = set()
    evaluation: set[str] = set()
    for category, group in unique.groupby("category", sort=True):
        identifiers = sorted(
            group["base_question_id"].astype(str),
            key=lambda value: (stable_u64(seed, category, value), value),
        )
        if len(identifiers) % 2:
            raise ValueError(f"Category {category!r} has an odd question count")
        midpoint = len(identifiers) // 2
        fitting.update(identifiers[:midpoint])
        evaluation.update(identifiers[midpoint:])
    all_ids = set(unique["base_question_id"].astype(str))
    if fitting & evaluation or fitting | evaluation != all_ids:
        raise RuntimeError("Fixed split leaked or lost base-question IDs")
    return Split(seed, frozenset(fitting), frozenset(evaluation))


def fixed_splits(selected: pd.DataFrame) -> dict[int, Split]:
    return {
        int(seed): stratified_question_split(selected, int(seed))
        for seed in load_config()["split_seeds"]
    }


def high_dose(family: str) -> float:
    return float(load_config()["families"][family]["high_dose"])


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def semantic_probabilities(
    row: pd.Series,
    *,
    temperature: float = 1.0,
    label_bias: np.ndarray | None = None,
) -> np.ndarray:
    likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
    if label_bias is not None:
        if len(label_bias) != len(likelihoods):
            raise ValueError("Label-bias vector length differs from candidate count")
        likelihoods = likelihoods - label_bias
    presented = _softmax(likelihoods / float(temperature))
    semantic = np.empty_like(presented)
    for presented_index, original_index in enumerate(row["presented_to_original"]):
        semantic[int(original_index)] = presented[presented_index]
    return semantic


def fit_clean_temperature(model_scores: pd.DataFrame, fitting_ids: Iterable[str]) -> float:
    fitting = {str(identifier) for identifier in fitting_ids}
    clean = model_scores[
        model_scores["base_question_id"].astype(str).isin(fitting)
        & model_scores["condition"].eq("clean")
    ]
    if clean.empty:
        raise ValueError("No clean fitting rows for temperature calibration")
    logits = [
        np.asarray(value, dtype=np.float64)
        for value in clean["candidate_log_likelihoods"]
    ]
    gold = clean["answer_index"].astype(int).tolist()

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        losses = [
            -math.log(max(float(_softmax(values / temperature)[index]), 1e-300))
            for values, index in zip(logits, gold, strict=True)
        ]
        return float(np.mean(losses))

    result = minimize_scalar(
        nll,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Temperature fit failed: {result.message}")
    return float(math.exp(float(result.x)))


def track_response_frame(
    scores: pd.DataFrame,
    family: str,
    *,
    representation: str = "gold_probability",
    temperatures: dict[str, float] | None = None,
    label_biases: dict[str, dict[int, np.ndarray]] | None = None,
) -> pd.DataFrame:
    subset = scores[
        scores["condition"].eq("clean") | scores["family"].eq(family)
    ].copy()
    if subset.empty:
        raise ValueError(f"No raw score rows for {family}")
    temperatures = temperatures or {}
    responses: list[float | np.ndarray] = []
    option_counts: list[int] = []
    for _, row in subset.iterrows():
        model = str(row["model_id"])
        likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
        bias = None
        if label_biases is not None:
            bias = label_biases[model].get(len(likelihoods))
            if bias is None:
                bias = np.zeros(len(likelihoods), dtype=np.float64)
        probabilities = semantic_probabilities(
            row,
            temperature=float(temperatures.get(model, 1.0)),
            label_bias=bias,
        )
        option_counts.append(len(probabilities))
        if representation == "gold_probability":
            responses.append(float(probabilities[int(row["semantic_answer_index"])]))
        elif representation == "probability_vector":
            responses.append(probabilities)
        elif representation == "margin":
            presented_gold = int(row["answer_index"])
            scaled = likelihoods / float(temperatures.get(model, 1.0))
            responses.append(float(scaled[presented_gold] - np.delete(scaled, presented_gold).max()))
        else:
            raise ValueError(f"Unknown response representation {representation!r}")
    subset["response"] = responses
    subset["option_count"] = option_counts
    subset["track_key"] = subset["track"].fillna(-1).astype(int)
    columns = [
        "model_id",
        "base_question_id",
        "category",
        "dose",
        "track_key",
        "semantic_answer_index",
        "option_count",
        "response",
    ]
    frame = subset[columns].copy()
    keys = ["model_id", "base_question_id", "dose", "track_key"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate track response rows for {family}")
    counts = frame.groupby(["model_id", "base_question_id", "dose"]).size()
    doses = counts.index.get_level_values("dose").astype(float)
    if not counts[doses == 0.0].eq(1).all() or not counts[doses != 0.0].eq(3).all():
        raise ValueError(f"Track counts violate the fixed design for {family}")
    return frame


def aligned_track_arrays(
    frame: pd.DataFrame, target: str, peers: list[str]
) -> AlignedData:
    key_columns = ["base_question_id", "category", "dose", "track_key"]
    target_rows = (
        frame[frame["model_id"].eq(target)]
        .sort_values(["base_question_id", "dose", "track_key"])
        .reset_index(drop=True)
    )
    if target_rows.empty:
        raise ValueError(f"No aligned rows for target {target}")
    keys = pd.MultiIndex.from_frame(target_rows[key_columns])
    first = np.asarray(target_rows.iloc[0]["response"])
    if first.ndim == 0:
        pivot = frame[frame["model_id"].isin([target, *peers])].pivot(
            index=key_columns, columns="model_id", values="response"
        )
        required = [target, *peers]
        if pivot[required].isna().any().any():
            raise ValueError("Scalar track response alignment contains missing model rows")
        pivot = pivot.sort_index().reset_index()
        metadata = pivot[key_columns].copy()
        target_values = pivot[target].to_numpy(dtype=np.float64)
        design = pivot[peers].to_numpy(dtype=np.float64)
        return AlignedData(metadata, target_values, design, tuple(peers))

    target_arrays = [np.asarray(value, dtype=np.float64) for value in target_rows["response"]]
    option_counts = np.asarray([len(value) for value in target_arrays], dtype=np.int64)
    maximum = int(option_counts.max())
    target_values = np.zeros((len(target_rows), maximum), dtype=np.float64)
    for index, value in enumerate(target_arrays):
        target_values[index, : len(value)] = value
    peer_values: list[np.ndarray] = []
    for peer in peers:
        peer_rows = frame[frame["model_id"].eq(peer)].set_index(key_columns)
        if not keys.isin(peer_rows.index).all():
            raise ValueError(f"Peer {peer} lacks vector rows aligned to {target}")
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in peer_rows.loc[keys, "response"]
        ]
        lengths = np.asarray([len(value) for value in arrays], dtype=np.int64)
        if not np.array_equal(lengths, option_counts):
            raise ValueError(f"Peer {peer} option counts differ from target {target}")
        padded = np.zeros_like(target_values)
        for index, value in enumerate(arrays):
            padded[index, : len(value)] = value
        peer_values.append(padded)
    design = np.stack(peer_values, axis=-1)
    metadata = target_rows[key_columns].copy()
    metadata["option_count"] = option_counts
    metadata["semantic_answer_index"] = target_rows[
        "semantic_answer_index"
    ].astype(int)
    return AlignedData(metadata, target_values, design, tuple(peers))


def aggregate_aligned(data: AlignedData) -> AlignedData:
    group_columns = ["base_question_id", "category", "dose"]
    groups = data.metadata.groupby(group_columns, sort=True).indices
    metadata_rows: list[dict[str, Any]] = []
    targets: list[np.ndarray | float] = []
    designs: list[np.ndarray] = []
    for key, indices in groups.items():
        positions = np.asarray(indices, dtype=np.int64)
        metadata_rows.append(dict(zip(group_columns, key, strict=True)))
        targets.append(np.mean(data.target[positions], axis=0))
        designs.append(np.mean(data.design[positions], axis=0))
    target = np.stack(targets) if data.target.ndim > 1 else np.asarray(targets, dtype=np.float64)
    design = np.stack(designs)
    metadata = pd.DataFrame(metadata_rows)
    if data.target.ndim > 1:
        counts = (
            data.metadata.groupby(group_columns, sort=True)["option_count"].first().to_numpy()
        )
        gold = (
            data.metadata.groupby(group_columns, sort=True)["semantic_answer_index"]
            .first()
            .to_numpy()
        )
        metadata["option_count"] = counts
        metadata["semantic_answer_index"] = gold
    return AlignedData(metadata, target, design, data.peers)


def design_row_weights(metadata: pd.DataFrame) -> np.ndarray:
    weights = np.where(metadata["dose"].astype(float).eq(0.0), 1.0, 1.0 / 3.0)
    check = metadata[["base_question_id", "dose"]].copy()
    check["weight"] = weights
    totals = check.groupby(["base_question_id", "dose"], sort=True)["weight"].sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-15, rtol=0.0):
        raise AssertionError("Each question-dose must carry exactly one unit of design mass")
    dose_totals = check.groupby("dose", sort=True)["weight"].sum().to_numpy()
    if not np.allclose(dose_totals, dose_totals[0], atol=1e-12, rtol=0.0):
        raise AssertionError("The five doses do not carry equal design mass")
    return weights.astype(np.float64)


def fit_projection(
    design: np.ndarray,
    target: np.ndarray,
    row_weights: np.ndarray | None = None,
) -> WeightedFit:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    weights = (
        np.ones(len(y), dtype=np.float64)
        if row_weights is None
        else np.asarray(row_weights, dtype=np.float64)
    )
    if len(weights) != len(y) or np.any(weights <= 0):
        raise ValueError("Projection row weights must be positive and aligned")
    square_root = np.sqrt(weights)
    if x.ndim == 2:
        weighted_x = x * square_root[:, None]
        weighted_y = y * square_root
    elif x.ndim == 3:
        weighted_x = x * square_root[:, None, None]
        weighted_y = y * square_root[:, None]
    else:
        raise ValueError(f"Unsupported projection design rank {x.ndim}")
    with warnings.catch_warnings(), redirect_stdout(_NULL_WRITER):
        warnings.filterwarnings(
            "ignore", message="Solution may be inaccurate.*", category=UserWarning
        )
        projection = fit_simplex_projection(weighted_x, weighted_y)
    prediction = np.tensordot(x, projection.weights, axes=([-1], [0]))
    residual = y - prediction
    row_squared = np.square(residual) if residual.ndim == 1 else np.square(residual).sum(axis=1)
    objective = float(np.sum(weights * row_squared) / np.sum(weights))
    return WeightedFit(projection, objective, float(weights.sum()), len(weights))


def projection_intervals(
    design: np.ndarray,
    target: np.ndarray,
    row_weights: np.ndarray,
    fit: WeightedFit,
) -> list[dict[str, Any]]:
    square_root = np.sqrt(row_weights)
    if design.ndim == 2:
        weighted_x = design * square_root[:, None]
        weighted_y = target * square_root
    else:
        weighted_x = design * square_root[:, None, None]
        weighted_y = target * square_root[:, None]
    with warnings.catch_warnings(), redirect_stdout(_NULL_WRITER):
        warnings.filterwarnings(
            "ignore", message="Solution may be inaccurate.*", category=UserWarning
        )
        try:
            intervals = feasible_weight_intervals(
                weighted_x, weighted_y, fit.projection, solver="CLARABEL"
            )
            interval_solver = "CLARABEL"
            fallback_path = "none"
        except Exception as clarabel_error:
            intervals = _slsqp_feasible_weight_intervals(
                weighted_x,
                weighted_y,
                fit.projection,
            )
            interval_solver = "SLSQP"
            fallback_path = (
                f"CLARABEL_to_SLSQP:{type(clarabel_error).__name__}"
            )
    return [
        {
            **interval,
            "interval_solver": interval_solver,
            "interval_fallback_path": fallback_path,
        }
        for interval in intervals
    ]


def _slsqp_feasible_weight_intervals(
    design: np.ndarray,
    target: np.ndarray,
    projection: ProjectionResult,
) -> list[dict[str, Any]]:
    """Bound simplex weight intervals with a deterministic convex fallback.

    This is only used when the byte-identical V2 CLARABEL interval diagnostic
    fails numerically.  It does not participate in the primary projection fit.
    """
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    if y.ndim > 1:
        y = y.reshape(-1)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Incompatible interval design {x.shape} and target {y.shape}")

    row_count, peer_count = x.shape
    allowed = float(projection.stage1_objective + projection.tolerance)
    start = np.asarray(projection.weights, dtype=np.float64).reshape(-1)
    identity = np.eye(peer_count, dtype=np.float64)
    ones = np.ones(peer_count, dtype=np.float64)

    def loss(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.dot(residual, residual) / row_count)

    def loss_gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * (x.T @ (x @ weights - y)) / row_count

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda _weights: ones,
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(allowed - loss(weights)),
            "jac": lambda weights: -loss_gradient(weights),
        },
    ]
    results: list[dict[str, Any]] = []
    for peer_index in range(peer_count):
        bounds: list[float] = []
        extreme_weights: list[list[float]] = []
        statuses: list[str] = []
        for direction in (1.0, -1.0):
            objective_gradient = direction * identity[peer_index]
            result = minimize(
                lambda weights, index=peer_index, sign=direction: float(
                    sign * weights[index]
                ),
                start,
                jac=lambda _weights, gradient=objective_gradient: gradient,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * peer_count,
                constraints=constraints,
                options={"ftol": 1e-13, "maxiter": 2_000, "disp": False},
            )
            solution = np.asarray(result.x, dtype=np.float64).reshape(-1)
            loss_excess = loss(solution) - allowed
            feasible = bool(
                result.success
                and solution.min() >= -1e-9
                and abs(float(solution.sum()) - 1.0) <= 1e-8
                and loss_excess <= 1e-9
            )
            if not feasible:
                raise RuntimeError(
                    "SLSQP weight interval solve failed for peer "
                    f"{peer_index}: status={result.status}, message={result.message}, "
                    f"minimum_weight={solution.min():.3e}, "
                    f"sum_error={abs(float(solution.sum()) - 1.0):.3e}, "
                    f"loss_excess={loss_excess:.3e}"
                )
            bounds.append(float(solution[peer_index]))
            extreme_weights.append(solution.tolist())
            statuses.append(f"optimal_slsqp_{result.status}")

        minimum, maximum = max(0.0, bounds[0]), min(1.0, bounds[1])
        if minimum > maximum + 1e-8:
            raise RuntimeError(
                f"SLSQP returned reversed interval for peer {peer_index}: "
                f"[{minimum}, {maximum}]"
            )
        results.append(
            {
                "peer_index": peer_index,
                "minimum": minimum,
                "maximum": maximum,
                "width": max(0.0, maximum - minimum),
                "status_min": statuses[0],
                "status_max": statuses[1],
                "minimum_solution_weights": extreme_weights[0],
                "maximum_solution_weights": extreme_weights[1],
            }
        )
    return results


def scalar_dose_metrics(
    data: AlignedData,
    weights: np.ndarray,
    evaluation_ids: Iterable[str],
) -> pd.DataFrame:
    evaluation = data.metadata["base_question_id"].astype(str).isin(evaluation_ids).to_numpy()
    prediction = data.design @ np.asarray(weights, dtype=np.float64)
    signed = data.target - prediction
    rows: list[dict[str, Any]] = []
    epsilon = float(load_config()["epsilon"])
    for dose in sorted(float(value) for value in data.metadata.loc[evaluation, "dose"].unique()):
        mask = evaluation & data.metadata["dose"].astype(float).eq(dose).to_numpy()
        meta = data.metadata.loc[mask].reset_index(drop=True)
        residual = signed[mask]
        absolute = np.abs(residual)
        row_weights = np.where(math.isclose(dose, 0.0), 1.0, 1.0 / 3.0)
        trackwise = float(np.sum(absolute * row_weights) / np.sum(np.ones(len(absolute)) * row_weights))
        has_declared_tracks = "track_key" in meta.columns
        if has_declared_tracks:
            temporary = meta[["base_question_id", "track_key"]].copy()
        else:
            temporary = meta[["base_question_id"]].copy()
            temporary["track_key"] = -1
        temporary["signed_residual"] = residual
        temporary["absolute_residual"] = absolute
        per_question = temporary.groupby("base_question_id", sort=True)
        track_mean = float(per_question["signed_residual"].mean().abs().mean())
        gap = trackwise - track_mean
        if gap < -float(load_config()["jensen_tolerance"]):
            raise AssertionError(
                f"Jensen inconsistency at dose {dose}: trackwise={trackwise}, track_mean={track_mean}"
            )
        if math.isclose(dose, 0.0) or not has_declared_tracks:
            track_piers = np.asarray([trackwise])
            within_std = 0.0
            within_range = 0.0
        else:
            track_piers = (
                temporary.groupby("track_key", sort=True)["absolute_residual"]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            if len(track_piers) != 3:
                raise ValueError(f"Expected exactly three evaluation tracks at dose {dose}")
            within_std = float(per_question["signed_residual"].std(ddof=0).mean())
            within_range = float(
                per_question["signed_residual"].agg(lambda values: np.ptp(values)).mean()
            )
        rows.append(
            {
                "dose": dose,
                "trackwise_pier": trackwise,
                "track_mean_response_pier": track_mean,
                "cancellation_gap": max(0.0, gap),
                "relative_cancellation_gap": max(0.0, gap) / (track_mean + epsilon),
                "relative_hidden_fraction": max(0.0, gap) / (trackwise + epsilon),
                "between_track_std": float(np.std(track_piers, ddof=0)),
                "between_track_range": float(np.ptp(track_piers)),
                "mean_within_question_signed_track_std": within_std,
                "mean_within_question_signed_track_range": within_range,
                "track_piers": track_piers.tolist(),
                "evaluation_question_count": int(meta["base_question_id"].nunique()),
                "evaluation_track_row_count": int(len(meta)),
                "jensen_consistent": True,
            }
        )
    return pd.DataFrame(rows)


def scalar_overall_error(
    data: AlignedData,
    weights: np.ndarray,
    evaluation_ids: Iterable[str],
) -> float:
    dose = scalar_dose_metrics(data, weights, evaluation_ids)
    return float(dose["trackwise_pier"].mean())


def weighted_peer_mae(
    data: AlignedData,
    fitting_ids: Iterable[str],
) -> np.ndarray:
    fitting = data.metadata["base_question_id"].astype(str).isin(fitting_ids).to_numpy()
    row_weights = design_row_weights(data.metadata)[fitting]
    errors = np.abs(data.design[fitting] - data.target[fitting, None])
    return np.average(errors, axis=0, weights=row_weights)


def _vector_instance_metrics(
    metadata: pd.DataFrame, target: np.ndarray, prediction: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, meta in metadata.reset_index(drop=True).iterrows():
        count = int(meta["option_count"])
        truth = np.clip(target[index, :count], 1e-15, 1.0)
        fitted = np.clip(prediction[index, :count], 1e-15, 1.0)
        truth /= truth.sum()
        fitted /= fitted.sum()
        midpoint = 0.5 * (truth + fitted)
        js = 0.5 * float(np.sum(truth * np.log(truth / midpoint)))
        js += 0.5 * float(np.sum(fitted * np.log(fitted / midpoint)))
        truth_top = int(np.argmax(truth))
        fitted_top = int(np.argmax(fitted))
        gold = int(meta["semantic_answer_index"])
        rows.append(
            {
                "mean_total_variation": 0.5 * float(np.abs(truth - fitted).sum()),
                "mean_jensen_shannon_divergence": js,
                "top1_semantic_answer_agreement": float(truth_top == fitted_top),
                "correctness_agreement": float((truth_top == gold) == (fitted_top == gold)),
            }
        )
    return pd.DataFrame(rows)


def vector_dose_metrics(
    data: AlignedData,
    weights: np.ndarray,
    evaluation_ids: Iterable[str],
) -> pd.DataFrame:
    evaluation = data.metadata["base_question_id"].astype(str).isin(evaluation_ids).to_numpy()
    prediction = np.tensordot(data.design, weights, axes=([-1], [0]))
    rows: list[dict[str, Any]] = []
    metric_columns = [
        "mean_total_variation",
        "mean_jensen_shannon_divergence",
        "top1_semantic_answer_agreement",
        "correctness_agreement",
    ]
    for dose in sorted(float(value) for value in data.metadata.loc[evaluation, "dose"].unique()):
        mask = evaluation & data.metadata["dose"].astype(float).eq(dose).to_numpy()
        metrics = _vector_instance_metrics(
            data.metadata.loc[mask], data.target[mask], prediction[mask]
        )
        row_weights = np.where(math.isclose(dose, 0.0), 1.0, 1.0 / 3.0)
        record = {column: float(np.average(metrics[column], weights=np.full(len(metrics), row_weights))) for column in metric_columns}
        record.update(
            {
                "dose": dose,
                "evaluation_question_count": int(
                    data.metadata.loc[mask, "base_question_id"].nunique()
                ),
                "evaluation_track_row_count": int(mask.sum()),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def vector_overall_tv(
    data: AlignedData,
    weights: np.ndarray,
    evaluation_ids: Iterable[str],
) -> float:
    return float(vector_dose_metrics(data, weights, evaluation_ids)["mean_total_variation"].mean())
