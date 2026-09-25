from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pier_llm.analysis import aggregate_family_responses, aligned_response_arrays

from .core import FAMILIES, fit_projection, fixed_splits, high_dose
from .input_validation import load_validated_inputs
from .peer_removal import designated_removals
from .utils import (
    PHYSICAL_ROOT,
    atomic_parquet,
    atomic_write_json,
    load_config,
    load_json,
    sha256_file,
    slug,
    stable_seed,
    utc_now,
)

BOOTSTRAP_SCHEMA = "pier_cluster_bootstrap_v21_v1"
_BOOT_CONTEXT: dict[str, Any] = {}


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def valid_bootstrap_shard(
    path: Path,
    *,
    analysis_type: str,
    target: str,
    family: str,
    replicate_start: int,
    replicate_end: int,
) -> bool:
    metadata_path = _metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = load_json(metadata_path)
        expected_rows = replicate_end - replicate_start + 1
        if (
            metadata.get("schema_version") != BOOTSTRAP_SCHEMA
            or metadata.get("analysis_type") != analysis_type
            or metadata.get("target") != target
            or metadata.get("family") != family
            or metadata.get("replicate_start") != replicate_start
            or metadata.get("replicate_end") != replicate_end
            or metadata.get("row_count") != expected_rows
            or metadata.get("sha256") != sha256_file(path)
        ):
            return False
        frame = pd.read_parquet(path)
        if len(frame) != expected_rows:
            return False
        if frame["replicate"].tolist() != list(range(replicate_start, replicate_end + 1)):
            return False
        required = {
            "analysis_type": analysis_type,
            "target": target,
            "family": family,
            "schema_version": BOOTSTRAP_SCHEMA,
        }
        return all(frame[column].eq(value).all() for column, value in required.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def shard_needs_recompute(path: Path, **expected: Any) -> bool:
    return not valid_bootstrap_shard(path, **expected)


def write_bootstrap_shard(
    path: Path,
    frame: pd.DataFrame,
    *,
    analysis_type: str,
    target: str,
    family: str,
    replicate_start: int,
    replicate_end: int,
) -> None:
    frame = frame.copy()
    frame["analysis_type"] = analysis_type
    frame["target"] = target
    frame["family"] = family
    frame["replicate_start"] = replicate_start
    frame["replicate_end"] = replicate_end
    frame["base_seed"] = 20260828
    frame["schema_version"] = BOOTSTRAP_SCHEMA
    atomic_parquet(path, frame)
    digest = sha256_file(path)
    metadata = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "analysis_type": analysis_type,
        "target": target,
        "family": family,
        "replicate_start": replicate_start,
        "replicate_end": replicate_end,
        "seed": 20260828,
        "row_count": len(frame),
        "sha256": digest,
        "SHA256": digest,
        "timestamp": utc_now(),
    }
    atomic_write_json(_metadata_path(path), metadata)


def _index_maps(metadata: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[tuple[str, float], np.ndarray]]:
    all_rows: dict[str, list[int]] = {}
    dose_rows: dict[tuple[str, float], list[int]] = {}
    for index, row in metadata.reset_index(drop=True).iterrows():
        identifier = str(row["base_question_id"])
        dose = float(row["dose"])
        all_rows.setdefault(identifier, []).append(int(index))
        dose_rows.setdefault((identifier, dose), []).append(int(index))
    return (
        {key: np.asarray(value, dtype=np.int64) for key, value in all_rows.items()},
        {key: np.asarray(value, dtype=np.int64) for key, value in dose_rows.items()},
    )


def _ids_by_category(
    selected: pd.DataFrame, identifiers: set[str] | frozenset[str]
) -> dict[str, list[str]]:
    subset = selected[selected["base_question_id"].astype(str).isin(identifiers)][
        ["base_question_id", "category"]
    ].drop_duplicates()
    return {
        str(category): sorted(group["base_question_id"].astype(str))
        for category, group in subset.groupby("category", sort=True)
    }


def prepare_bootstrap_context() -> tuple[list[str], list[str]]:
    global _BOOT_CONTEXT
    scores, _generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    for family in FAMILIES:
        aggregated = aggregate_family_responses(scores, family, "gold_probability")
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            metadata, target_values, design = aligned_response_arrays(
                aggregated, target, peers
            )
            all_map, dose_map = _index_maps(metadata)
            datasets[(family, target)] = {
                "metadata": metadata,
                "target_values": target_values,
                "design": design,
                "peers": peers,
                "all_map": all_map,
                "dose_map": dose_map,
            }
    split_draws = {
        seed: {
            "fit": _ids_by_category(selected, split.fitting_ids),
            "evaluation": _ids_by_category(selected, split.evaluation_ids),
        }
        for seed, split in splits.items()
    }
    records = {record["id"]: record for record in resolved["models"]}
    sibling_targets = [
        target
        for target in model_ids
        if designated_removals(
            target,
            [model for model in model_ids if model != target],
            records,
        )["exact_sibling"]
    ]
    _BOOT_CONTEXT = {
        "model_ids": model_ids,
        "datasets": datasets,
        "split_draws": split_draws,
        "split_seeds": list(splits),
        "records": records,
    }
    return model_ids, sibling_targets


def _draw(by_category: dict[str, list[str]], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    result: list[str] = []
    for category in sorted(by_category):
        identifiers = by_category[category]
        result.extend(
            str(value)
            for value in rng.choice(identifiers, size=len(identifiers), replace=True)
        )
    return result


def _draws(
    split_seed: int,
    *,
    analysis_type: str,
    target: str,
    family: str,
    replicate: int,
    synchronize_families: bool = False,
) -> tuple[list[str], list[str], int]:
    family_component = "synchronized_families" if synchronize_families else family
    seed = stable_seed(
        20260828, analysis_type, target, family_component, split_seed, replicate
    )
    groups = _BOOT_CONTEXT["split_draws"][split_seed]
    fit = _draw(groups["fit"], stable_seed(seed, "fit"))
    evaluation = _draw(groups["evaluation"], stable_seed(seed, "evaluation"))
    if set(fit).intersection(evaluation):
        raise RuntimeError("Bootstrap source-identity leakage")
    return fit, evaluation, seed


def _rows(mapping: dict[Any, np.ndarray], draw: list[str], dose: float | None = None) -> np.ndarray:
    if dose is None:
        chunks = [mapping[identifier] for identifier in draw]
    else:
        chunks = [mapping[(identifier, float(dose))] for identifier in draw]
    return np.concatenate(chunks)


def _endpoint_worker(task: tuple[str, str, int, int, str]) -> str:
    target, family, start, end, output_name = task
    dataset = _BOOT_CONTEXT["datasets"][(family, target)]
    x = dataset["design"]
    y = dataset["target_values"]
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        split_clean: list[float] = []
        split_high: list[float] = []
        split_delta: list[float] = []
        split_seeds: list[int] = []
        for split_seed in _BOOT_CONTEXT["split_seeds"]:
            fit_draw, evaluation_draw, _seed = _draws(
                split_seed,
                analysis_type="endpoint_effects",
                target=target,
                family=family,
                replicate=replicate,
                synchronize_families=True,
            )
            fit_rows = _rows(dataset["all_map"], fit_draw)
            projection = fit_projection(x[fit_rows], y[fit_rows])
            clean_rows = _rows(dataset["dose_map"], evaluation_draw, 0.0)
            high_rows = _rows(dataset["dose_map"], evaluation_draw, high_dose(family))
            clean = float(np.mean(np.abs(y[clean_rows] - x[clean_rows] @ projection.weights)))
            high = float(np.mean(np.abs(y[high_rows] - x[high_rows] @ projection.weights)))
            split_clean.append(clean)
            split_high.append(high)
            split_delta.append(high - clean)
            split_seeds.append(split_seed)
        mean_clean = float(np.mean(split_clean))
        mean_high = float(np.mean(split_high))
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(20260828, "endpoint_effects", target, replicate),
                "pier_clean": mean_clean,
                "pier_high": mean_high,
                "delta_absolute": float(np.mean(split_delta)),
                "delta_relative": float(np.mean(split_delta)) / (mean_clean + 1e-10),
                "median_split_delta": float(np.median(split_delta)),
                "positive_split_count": int(np.sum(np.asarray(split_delta) > 0)),
                "negative_split_count": int(np.sum(np.asarray(split_delta) < 0)),
                "split_seeds": split_seeds,
                "split_deltas": split_delta,
            }
        )
    output = Path(output_name)
    write_bootstrap_shard(
        output,
        pd.DataFrame(rows),
        analysis_type="endpoint_effects",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
    )
    return str(output)


def _sibling_worker(task: tuple[str, str, int, int, str]) -> str:
    target, family, start, end, output_name = task
    dataset = _BOOT_CONTEXT["datasets"][(family, target)]
    peers = dataset["peers"]
    sibling = designated_removals(
        target, peers, _BOOT_CONTEXT["records"]
    )["exact_sibling"]
    if len(sibling) != 1:
        raise ValueError(f"Expected one exact sibling for {target}, found {sibling}")
    retained_indices = [index for index, peer in enumerate(peers) if peer not in sibling]
    x_all = dataset["design"]
    x_removed = x_all[:, retained_indices]
    y = dataset["target_values"]
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        all_errors: list[float] = []
        removed_errors: list[float] = []
        effects: list[float] = []
        for split_seed in _BOOT_CONTEXT["split_seeds"]:
            fit_draw, evaluation_draw, _seed = _draws(
                split_seed,
                analysis_type="sibling_removal",
                target=target,
                family=family,
                replicate=replicate,
            )
            fit_rows = _rows(dataset["all_map"], fit_draw)
            evaluation_rows = _rows(dataset["all_map"], evaluation_draw)
            all_projection = fit_projection(x_all[fit_rows], y[fit_rows])
            removed_projection = fit_projection(x_removed[fit_rows], y[fit_rows])
            all_error = float(
                np.mean(
                    np.abs(
                        y[evaluation_rows]
                        - x_all[evaluation_rows] @ all_projection.weights
                    )
                )
            )
            removed_error = float(
                np.mean(
                    np.abs(
                        y[evaluation_rows]
                        - x_removed[evaluation_rows] @ removed_projection.weights
                    )
                )
            )
            all_errors.append(all_error)
            removed_errors.append(removed_error)
            effects.append(removed_error - all_error)
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(20260828, "sibling_removal", target, family, replicate),
                "sibling": sibling[0],
                "all_peer_pier": float(np.mean(all_errors)),
                "sibling_removed_pier": float(np.mean(removed_errors)),
                "sibling_removal_inflation": float(np.mean(effects)),
                "median_split_inflation": float(np.median(effects)),
                "positive_split_count": int(np.sum(np.asarray(effects) > 0)),
                "split_effects": effects,
            }
        )
    output = Path(output_name)
    write_bootstrap_shard(
        output,
        pd.DataFrame(rows),
        analysis_type="sibling_removal",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
    )
    return str(output)


def _convex_worker(task: tuple[str, str, int, int, str]) -> str:
    target, family, start, end, output_name = task
    dataset = _BOOT_CONTEXT["datasets"][(family, target)]
    x = dataset["design"]
    y = dataset["target_values"]
    peers = dataset["peers"]
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        single_errors: list[float] = []
        convex_errors: list[float] = []
        improvements: list[float] = []
        selected_peers: list[str] = []
        for split_seed in _BOOT_CONTEXT["split_seeds"]:
            fit_draw, evaluation_draw, _seed = _draws(
                split_seed,
                analysis_type="convex_vs_single",
                target=target,
                family=family,
                replicate=replicate,
            )
            fit_rows = _rows(dataset["all_map"], fit_draw)
            evaluation_rows = _rows(dataset["all_map"], evaluation_draw)
            projection = fit_projection(x[fit_rows], y[fit_rows])
            fit_peer_errors = np.mean(
                np.abs(x[fit_rows] - y[fit_rows, None]), axis=0
            )
            selected_index = int(np.argmin(fit_peer_errors))
            convex_error = float(
                np.mean(
                    np.abs(y[evaluation_rows] - x[evaluation_rows] @ projection.weights)
                )
            )
            single_error = float(
                np.mean(
                    np.abs(y[evaluation_rows] - x[evaluation_rows, selected_index])
                )
            )
            convex_errors.append(convex_error)
            single_errors.append(single_error)
            improvements.append(single_error - convex_error)
            selected_peers.append(peers[selected_index])
        mean_single = float(np.mean(single_errors))
        mean_convex = float(np.mean(convex_errors))
        mean_improvement = float(np.mean(improvements))
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(20260828, "convex_vs_single", target, family, replicate),
                "single_error": mean_single,
                "convex_error": mean_convex,
                "absolute_improvement": mean_improvement,
                "relative_improvement": mean_improvement / (mean_single + 1e-10),
                "gap_ratio": mean_single / (mean_convex + 1e-10),
                "convex_wins_split_count": int(np.sum(np.asarray(improvements) > 0)),
                "fit_selected_single_peers": selected_peers,
                "split_improvements": improvements,
            }
        )
    output = Path(output_name)
    write_bootstrap_shard(
        output,
        pd.DataFrame(rows),
        analysis_type="convex_vs_single",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
    )
    return str(output)


def _shard_path(
    analysis_type: str,
    target: str,
    family: str,
    start: int,
    end: int,
) -> Path:
    return (
        PHYSICAL_ROOT
        / "outputs/bootstrap"
        / analysis_type
        / f"{slug(target)}__{family}__{start:04d}_{end:04d}.parquet"
    )


def _tasks_for(
    analysis_type: str,
    targets: list[str],
    *,
    resume: bool,
) -> list[tuple[str, str, int, int, str]]:
    config = load_config()
    replicates = int(config["bootstrap_replicates"])
    shard_size = int(config["bootstrap_shard_size"])
    tasks: list[tuple[str, str, int, int, str]] = []
    for target in targets:
        for family in FAMILIES:
            for start in range(0, replicates, shard_size):
                end = min(replicates - 1, start + shard_size - 1)
                path = _shard_path(analysis_type, target, family, start, end)
                if resume and valid_bootstrap_shard(
                    path,
                    analysis_type=analysis_type,
                    target=target,
                    family=family,
                    replicate_start=start,
                    replicate_end=end,
                ):
                    continue
                tasks.append((target, family, start, end, str(path)))
    return tasks


def _run_tasks(
    tasks: list[tuple[str, str, int, int, str]],
    worker: Any,
    *,
    workers: int,
) -> None:
    if not tasks:
        return
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            output = future.result()
            print(f"[bootstrap] {completed}/{len(tasks)} complete: {output}", flush=True)


def read_bootstrap_analysis(analysis_type: str) -> pd.DataFrame:
    paths = sorted((PHYSICAL_ROOT / "outputs/bootstrap" / analysis_type).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {analysis_type} bootstrap shards")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _validate_complete(
    analysis_type: str,
    targets: list[str],
) -> list[Path]:
    config = load_config()
    replicates = int(config["bootstrap_replicates"])
    shard_size = int(config["bootstrap_shard_size"])
    paths: list[Path] = []
    for target in targets:
        for family in FAMILIES:
            for start in range(0, replicates, shard_size):
                end = min(replicates - 1, start + shard_size - 1)
                path = _shard_path(analysis_type, target, family, start, end)
                if not valid_bootstrap_shard(
                    path,
                    analysis_type=analysis_type,
                    target=target,
                    family=family,
                    replicate_start=start,
                    replicate_end=end,
                ):
                    raise RuntimeError(f"Missing or corrupt bootstrap shard: {path}")
                paths.extend([path, _metadata_path(path)])
    return paths


def run_bootstraps(*, workers: int, resume: bool) -> dict[str, Any]:
    model_ids, sibling_targets = prepare_bootstrap_context()
    task_specs = [
        ("endpoint_effects", model_ids, _endpoint_worker),
        ("sibling_removal", sibling_targets, _sibling_worker),
        ("convex_vs_single", model_ids, _convex_worker),
    ]
    for analysis_type, targets, worker in task_specs:
        tasks = _tasks_for(analysis_type, targets, resume=resume)
        print(
            f"[bootstrap] {analysis_type}: {len(tasks)} shard(s) to compute with {workers} workers",
            flush=True,
        )
        _run_tasks(tasks, worker, workers=workers)

    outputs: list[Path] = []
    outputs.extend(_validate_complete("endpoint_effects", model_ids))
    outputs.extend(_validate_complete("sibling_removal", sibling_targets))
    outputs.extend(_validate_complete("convex_vs_single", model_ids))

    endpoint = read_bootstrap_analysis("endpoint_effects")
    irrelevant = endpoint[endpoint["family"].eq("irrelevant_context")][
        ["target", "replicate", "delta_absolute"]
    ].rename(columns={"delta_absolute": "irrelevant_endpoint_delta"})
    deletion = endpoint[endpoint["family"].eq("content_deletion")][
        ["target", "replicate", "delta_absolute"]
    ].rename(columns={"delta_absolute": "deletion_endpoint_delta"})
    contrast = irrelevant.merge(
        deletion, on=["target", "replicate"], validate="one_to_one"
    )
    contrast["stress_specific_contrast"] = (
        contrast["irrelevant_endpoint_delta"] - contrast["deletion_endpoint_delta"]
    )
    contrast_path = PHYSICAL_ROOT / "outputs/analysis/endpoint_stress_contrasts.parquet"
    atomic_parquet(contrast_path, contrast)
    outputs.append(contrast_path)
    return {
        "outputs": outputs,
        "endpoint_rows": len(endpoint),
        "sibling_rows": len(read_bootstrap_analysis("sibling_removal")),
        "convexity_rows": len(read_bootstrap_analysis("convex_vs_single")),
        "stress_contrast_rows": len(contrast),
    }
