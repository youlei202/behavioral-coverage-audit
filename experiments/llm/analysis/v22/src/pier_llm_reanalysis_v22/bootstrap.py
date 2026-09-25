from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    FAMILIES,
    aligned_track_arrays,
    design_row_weights,
    fit_projection,
    fixed_splits,
    high_dose,
    load_validated_inputs,
    track_response_frame,
)
from .point_analysis import sibling_for
from .utils import (
    PHYSICAL_ROOT,
    atomic_csv,
    atomic_parquet,
    atomic_write_json,
    load_config,
    load_json,
    sha256_file,
    slug,
    stable_seed,
    utc_now,
)

BOOTSTRAP_SCHEMA = "pier_cluster_bootstrap_v22_v1"
BOOTSTRAP_WORDING = (
    "base-question clustered-bootstrap interval for the ten-split bagged estimator"
)
_BOOT_CONTEXT: dict[str, Any] = {}


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _completion_path(path: Path) -> Path:
    return path.with_suffix(".complete.json")


def valid_bootstrap_shard(
    path: Path,
    *,
    analysis_type: str,
    target: str,
    family: str,
    replicate_start: int,
    replicate_end: int,
    schema_version: str = BOOTSTRAP_SCHEMA,
) -> bool:
    metadata_path = _metadata_path(path)
    completion_path = _completion_path(path)
    if not path.is_file() or not metadata_path.is_file() or not completion_path.is_file():
        return False
    try:
        metadata = load_json(metadata_path)
        completion = load_json(completion_path)
        expected_rows = replicate_end - replicate_start + 1
        expected = {
            "schema_version": schema_version,
            "analysis_type": analysis_type,
            "target": target,
            "family": family,
            "replicate_start": replicate_start,
            "replicate_end": replicate_end,
            "row_count": expected_rows,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if metadata.get("sha256") != sha256_file(path):
            return False
        if completion.get("complete") is not True:
            return False
        if completion.get("sha256") != metadata.get("sha256"):
            return False
        frame = pd.read_parquet(path)
        if len(frame) != expected_rows:
            return False
        if frame["replicate"].tolist() != list(
            range(replicate_start, replicate_end + 1)
        ):
            return False
        checks = {
            "analysis_type": analysis_type,
            "target": target,
            "family": family,
            "schema_version": schema_version,
        }
        return all(frame[column].eq(value).all() for column, value in checks.items())
    except (OSError, ValueError, TypeError, KeyError):
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
    schema_version: str = BOOTSTRAP_SCHEMA,
) -> None:
    output = frame.copy()
    output["analysis_type"] = analysis_type
    output["target"] = target
    output["family"] = family
    output["replicate_start"] = replicate_start
    output["replicate_end"] = replicate_end
    output["base_seed"] = int(load_config()["bootstrap_seed"])
    output["schema_version"] = schema_version
    output["bootstrap_unit"] = "base_question"
    output["interval_estimand"] = BOOTSTRAP_WORDING
    atomic_parquet(path, output)
    digest = sha256_file(path)
    metadata = {
        "schema_version": schema_version,
        "analysis_type": analysis_type,
        "target": target,
        "family": family,
        "replicate_start": replicate_start,
        "replicate_end": replicate_end,
        "seed": int(load_config()["bootstrap_seed"]),
        "row_count": len(output),
        "sha256": digest,
        "SHA256": digest,
        "completion_marker": str(_completion_path(path)),
        "timestamp": utc_now(),
    }
    atomic_write_json(_metadata_path(path), metadata)
    atomic_write_json(
        _completion_path(path),
        {
            "schema_version": schema_version,
            "complete": True,
            "analysis_type": analysis_type,
            "target": target,
            "family": family,
            "replicate_start": replicate_start,
            "replicate_end": replicate_end,
            "row_count": len(output),
            "sha256": digest,
            "completed_at": utc_now(),
        },
    )


def _index_maps(
    metadata: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, float], np.ndarray]]:
    all_rows: dict[str, list[int]] = {}
    dose_rows: dict[tuple[str, float], list[int]] = {}
    for index, row in metadata.reset_index(drop=True).iterrows():
        identifier = str(row["base_question_id"])
        dose = float(row["dose"])
        all_rows.setdefault(identifier, []).append(index)
        dose_rows.setdefault((identifier, dose), []).append(index)
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


def prepare_primary_bootstrap_context() -> tuple[list[str], list[str]]:
    global _BOOT_CONTEXT
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    datasets: dict[tuple[str, str], dict[str, Any]] = {}
    for family in FAMILIES:
        frame = track_response_frame(scores, family)
        for target in model_ids:
            peers = [model for model in model_ids if model != target]
            data = aligned_track_arrays(frame, target, peers)
            all_map, dose_map = _index_maps(data.metadata)
            datasets[(family, target)] = {
                "metadata": data.metadata,
                "target_values": data.target,
                "design": data.design,
                "peers": peers,
                "row_weights": design_row_weights(data.metadata),
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
    sibling_targets = [target for target in model_ids if sibling_for(target) is not None]
    _BOOT_CONTEXT = {
        "datasets": datasets,
        "split_draws": split_draws,
        "split_seeds": list(splits),
        "doses": {
            family: [float(value) for value in load_config()["families"][family]["doses"]]
            for family in FAMILIES
        },
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


def bootstrap_draws(
    split_seed: int,
    *,
    analysis_type: str,
    target: str,
    family: str,
    replicate: int,
) -> tuple[list[str], list[str], int]:
    seed = stable_seed(
        load_config()["bootstrap_seed"],
        analysis_type,
        target,
        family,
        split_seed,
        replicate,
    )
    groups = _BOOT_CONTEXT["split_draws"][split_seed]
    fitting = _draw(groups["fit"], stable_seed(seed, "fit"))
    evaluation = _draw(groups["evaluation"], stable_seed(seed, "evaluation"))
    if set(fitting).intersection(evaluation):
        raise RuntimeError("Bootstrap fitting/evaluation source identities overlap")
    return fitting, evaluation, seed


def _rows(
    mapping: dict[Any, np.ndarray], draw: list[str], dose: float | None = None
) -> np.ndarray:
    chunks = (
        [mapping[identifier] for identifier in draw]
        if dose is None
        else [mapping[(identifier, float(dose))] for identifier in draw]
    )
    return np.concatenate(chunks)


def _fit(dataset: dict[str, Any], fit_draw: list[str], design: np.ndarray) -> np.ndarray:
    rows = _rows(dataset["all_map"], fit_draw)
    return fit_projection(
        design[rows],
        dataset["target_values"][rows],
        dataset["row_weights"][rows],
    ).projection.weights


def _dose_error(
    dataset: dict[str, Any],
    evaluation_draw: list[str],
    dose: float,
    design: np.ndarray,
    weights: np.ndarray,
) -> float:
    rows = _rows(dataset["dose_map"], evaluation_draw, dose)
    residual = dataset["target_values"][rows] - design[rows] @ weights
    return float(np.mean(np.abs(residual)))


def _overall_error(
    dataset: dict[str, Any],
    evaluation_draw: list[str],
    family: str,
    design: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        np.mean(
            [
                _dose_error(dataset, evaluation_draw, dose, design, weights)
                for dose in _BOOT_CONTEXT["doses"][family]
            ]
        )
    )


def _endpoint_worker(task: tuple[str, str, int, int, str]) -> str:
    target, family, start, end, output_name = task
    dataset = _BOOT_CONTEXT["datasets"][(family, target)]
    design = dataset["design"]
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        split_clean: list[float] = []
        split_high: list[float] = []
        split_delta: list[float] = []
        split_seeds: list[int] = []
        for split_seed in _BOOT_CONTEXT["split_seeds"]:
            fit_draw, evaluation_draw, _seed = bootstrap_draws(
                split_seed,
                analysis_type="endpoint",
                target=target,
                family=family,
                replicate=replicate,
            )
            weights = _fit(dataset, fit_draw, design)
            clean = _dose_error(dataset, evaluation_draw, 0.0, design, weights)
            high = _dose_error(
                dataset, evaluation_draw, high_dose(family), design, weights
            )
            split_clean.append(clean)
            split_high.append(high)
            split_delta.append(high - clean)
            split_seeds.append(split_seed)
        mean_clean = float(np.mean(split_clean))
        delta = float(np.mean(split_delta))
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(
                    load_config()["bootstrap_seed"],
                    "endpoint",
                    target,
                    family,
                    replicate,
                ),
                "trackwise_clean_pier": mean_clean,
                "trackwise_high_pier": float(np.mean(split_high)),
                "trackwise_delta": delta,
                "trackwise_relative_delta": delta
                / (mean_clean + load_config()["epsilon"]),
                "median_split_delta": float(np.median(split_delta)),
                "positive_split_count": int(np.sum(np.asarray(split_delta) > 0)),
                "negative_split_count": int(np.sum(np.asarray(split_delta) < 0)),
                "split_seeds": split_seeds,
                "split_estimates": split_delta,
            }
        )
    path = Path(output_name)
    write_bootstrap_shard(
        path,
        pd.DataFrame(rows),
        analysis_type="endpoint",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
    )
    return str(path)


def _convexity_worker(task: tuple[str, str, int, int, str]) -> str:
    target, family, start, end, output_name = task
    dataset = _BOOT_CONTEXT["datasets"][(family, target)]
    design = dataset["design"]
    target_values = dataset["target_values"]
    peers = dataset["peers"]
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        single_errors: list[float] = []
        convex_errors: list[float] = []
        improvements: list[float] = []
        selected_peers: list[str] = []
        for split_seed in _BOOT_CONTEXT["split_seeds"]:
            fit_draw, evaluation_draw, _seed = bootstrap_draws(
                split_seed,
                analysis_type="convexity",
                target=target,
                family=family,
                replicate=replicate,
            )
            fit_rows = _rows(dataset["all_map"], fit_draw)
            weights = _fit(dataset, fit_draw, design)
            peer_errors = np.average(
                np.abs(design[fit_rows] - target_values[fit_rows, None]),
                axis=0,
                weights=dataset["row_weights"][fit_rows],
            )
            selected_index = int(np.argmin(peer_errors))
            single_weights = np.zeros(len(peers), dtype=np.float64)
            single_weights[selected_index] = 1.0
            convex = _overall_error(
                dataset, evaluation_draw, family, design, weights
            )
            single = _overall_error(
                dataset, evaluation_draw, family, design, single_weights
            )
            convex_errors.append(convex)
            single_errors.append(single)
            improvements.append(single - convex)
            selected_peers.append(peers[selected_index])
        mean_single = float(np.mean(single_errors))
        mean_convex = float(np.mean(convex_errors))
        improvement = float(np.mean(improvements))
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(
                    load_config()["bootstrap_seed"],
                    "convexity",
                    target,
                    family,
                    replicate,
                ),
                "single_error_trackwise": mean_single,
                "convex_error_trackwise": mean_convex,
                "absolute_improvement": improvement,
                "gap_ratio": mean_single
                / (mean_convex + load_config()["epsilon"]),
                "positive_split_count": int(np.sum(np.asarray(improvements) > 0)),
                "negative_split_count": int(np.sum(np.asarray(improvements) < 0)),
                "fit_selected_single_peers": selected_peers,
                "split_estimates": improvements,
            }
        )
    path = Path(output_name)
    write_bootstrap_shard(
        path,
        pd.DataFrame(rows),
        analysis_type="convexity",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
    )
    return str(path)


def _sibling_worker(task: tuple[str, str, int, int, str]) -> str:
    target, family, start, end, output_name = task
    dataset = _BOOT_CONTEXT["datasets"][(family, target)]
    peers = dataset["peers"]
    sibling = sibling_for(target)
    if sibling is None or sibling not in peers:
        raise ValueError(f"No unique declared sibling for {target}")
    retained = [index for index, peer in enumerate(peers) if peer != sibling]
    all_design = dataset["design"]
    removed_design = all_design[:, retained]
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        all_errors: list[float] = []
        removed_errors: list[float] = []
        effects: list[float] = []
        for split_seed in _BOOT_CONTEXT["split_seeds"]:
            fit_draw, evaluation_draw, _seed = bootstrap_draws(
                split_seed,
                analysis_type="sibling_removal",
                target=target,
                family=family,
                replicate=replicate,
            )
            all_weights = _fit(dataset, fit_draw, all_design)
            removed_weights = _fit(dataset, fit_draw, removed_design)
            all_error = _overall_error(
                dataset, evaluation_draw, family, all_design, all_weights
            )
            removed_error = _overall_error(
                dataset, evaluation_draw, family, removed_design, removed_weights
            )
            all_errors.append(all_error)
            removed_errors.append(removed_error)
            effects.append(removed_error - all_error)
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(
                    load_config()["bootstrap_seed"],
                    "sibling_removal",
                    target,
                    family,
                    replicate,
                ),
                "removed_sibling": sibling,
                "all_peer_trackwise_pier": float(np.mean(all_errors)),
                "sibling_removed_trackwise_pier": float(np.mean(removed_errors)),
                "inflation": float(np.mean(effects)),
                "median_split_inflation": float(np.median(effects)),
                "positive_split_count": int(np.sum(np.asarray(effects) > 0)),
                "negative_split_count": int(np.sum(np.asarray(effects) < 0)),
                "split_estimates": effects,
            }
        )
    path = Path(output_name)
    write_bootstrap_shard(
        path,
        pd.DataFrame(rows),
        analysis_type="sibling_removal",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
    )
    return str(path)


def _directory_for(analysis_type: str) -> Path:
    mapping = {
        "endpoint": "endpoint",
        "convexity": "convexity",
        "sibling_removal": "sibling_removal",
    }
    return PHYSICAL_ROOT / "bootstrap_shards" / mapping[analysis_type]


def _shard_path(
    analysis_type: str, target: str, family: str, start: int, end: int
) -> Path:
    return (
        _directory_for(analysis_type)
        / f"{slug(target)}__{family}__{start:04d}_{end:04d}.parquet"
    )


def _tasks_for(
    analysis_type: str, targets: list[str], *, resume: bool
) -> list[tuple[str, str, int, int, str]]:
    replicates = int(load_config()["bootstrap_replicates"])
    shard_size = int(load_config()["bootstrap_shard_size"])
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
    tasks: list[tuple[str, str, int, int, str]], worker: Any, *, workers: int
) -> None:
    if not tasks:
        return
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            output = future.result()
            print(
                f"[bootstrap] {completed}/{len(tasks)} complete: {output}",
                flush=True,
            )


def _expected_paths(analysis_type: str, targets: list[str]) -> list[Path]:
    replicates = int(load_config()["bootstrap_replicates"])
    shard_size = int(load_config()["bootstrap_shard_size"])
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
                paths.extend([path, _metadata_path(path), _completion_path(path)])
    return paths


def read_bootstrap_analysis(analysis_type: str, targets: list[str]) -> pd.DataFrame:
    paths = [
        path
        for path in _expected_paths(analysis_type, targets)
        if path.suffix == ".parquet"
    ]
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _summary(
    frame: pd.DataFrame, metric: str, analysis_type: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target, family), group in frame.groupby(["target", "family"], sort=True):
        values = group.sort_values("replicate")[metric].to_numpy(dtype=np.float64)
        if len(values) != int(load_config()["bootstrap_replicates"]):
            raise ValueError(
                f"Incomplete {analysis_type} bootstrap for {target}/{family}"
            )
        rows.append(
            {
                "analysis_type": analysis_type,
                "target": target,
                "family": family,
                "metric": metric,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_lower": float(np.quantile(values, 0.025)),
                "bootstrap_upper": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0)),
                "probability_negative": float(np.mean(values < 0)),
                "replicates": len(values),
                "bootstrap_unit": "base_question",
                "interval_estimand": BOOTSTRAP_WORDING,
            }
        )
    return pd.DataFrame(rows)


def _write_inventory(shard_outputs: list[Path]) -> Path:
    parquet_paths = sorted(path for path in shard_outputs if path.suffix == ".parquet")
    records: list[dict[str, Any]] = []
    for path in parquet_paths:
        metadata = load_json(_metadata_path(path))
        completion = load_json(_completion_path(path))
        records.append(
            {
                **metadata,
                "path": str(path.relative_to(PHYSICAL_ROOT)),
                "metadata_path": str(_metadata_path(path).relative_to(PHYSICAL_ROOT)),
                "metadata_sha256": sha256_file(_metadata_path(path)),
                "completion_path": str(
                    _completion_path(path).relative_to(PHYSICAL_ROOT)
                ),
                "completion_sha256": sha256_file(_completion_path(path)),
                "complete": bool(completion.get("complete")),
            }
        )
    path = (
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/BOOTSTRAP_SHARD_INVENTORY.json"
    )
    atomic_write_json(
        path,
        {
            "schema_version": "pier_bootstrap_shard_inventory_v22_v1",
            "shard_count": len(records),
            "shards": records,
        },
    )
    return path


def run_primary_bootstraps(*, workers: int, resume: bool) -> dict[str, Any]:
    model_ids, sibling_targets = prepare_primary_bootstrap_context()
    specifications = [
        ("endpoint", model_ids, _endpoint_worker),
        ("convexity", model_ids, _convexity_worker),
        ("sibling_removal", sibling_targets, _sibling_worker),
    ]
    for analysis_type, targets, worker in specifications:
        tasks = _tasks_for(analysis_type, targets, resume=resume)
        print(
            f"[bootstrap] {analysis_type}: {len(tasks)} shard(s), {workers} workers",
            flush=True,
        )
        _run_tasks(tasks, worker, workers=workers)

    shard_outputs: list[Path] = []
    shard_outputs.extend(_expected_paths("endpoint", model_ids))
    shard_outputs.extend(_expected_paths("convexity", model_ids))
    shard_outputs.extend(_expected_paths("sibling_removal", sibling_targets))
    endpoint = read_bootstrap_analysis("endpoint", model_ids)
    convexity = read_bootstrap_analysis("convexity", model_ids)
    sibling = read_bootstrap_analysis("sibling_removal", sibling_targets)
    endpoint_summary = _summary(endpoint, "trackwise_delta", "endpoint")
    convexity_summary = _summary(
        convexity, "absolute_improvement", "convexity"
    )
    sibling_summary = _summary(sibling, "inflation", "sibling_removal")
    endpoint_path = (
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/endpoint_bootstrap_summary.csv"
    )
    convexity_path = (
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/convexity_bootstrap_summary.csv"
    )
    sibling_path = (
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/sibling_removal_bootstrap_summary.csv"
    )
    atomic_csv(endpoint_path, endpoint_summary)
    atomic_csv(convexity_path, convexity_summary)
    atomic_csv(sibling_path, sibling_summary)
    inventory_path = _write_inventory(shard_outputs)
    return {
        "outputs": [
            *shard_outputs,
            endpoint_path,
            convexity_path,
            sibling_path,
            inventory_path,
        ],
        "row_counts": {
            "endpoint_bootstrap": len(endpoint),
            "convexity_bootstrap": len(convexity),
            "sibling_bootstrap": len(sibling),
            "bootstrap_shards": len(
                [path for path in shard_outputs if path.suffix == ".parquet"]
            ),
        },
    }
