from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import (
    BOOTSTRAP_WORDING,
    _completion_path,
    _draw,
    _ids_by_category,
    _index_maps,
    _metadata_path,
    _rows,
    valid_bootstrap_shard,
    write_bootstrap_shard,
)
from .data import (
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
    load_config,
    slug,
    stable_seed,
)

CALIBRATED_BOOTSTRAP_SCHEMA = "pier_selected_calibrated_bootstrap_v22_v1"
_CAL_CONTEXT: dict[str, Any] = {}


def _prepare_context() -> list[tuple[str, str, str]]:
    global _CAL_CONTEXT
    scores, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    specifications = [tuple(value) for value in load_config()["selected_calibrated_bootstraps"]]
    target_families = sorted({(target, family) for _analysis, target, family in specifications})
    calibrated = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_calibrated_results.parquet"
    )
    datasets: dict[tuple[int, str, str], dict[str, Any]] = {}
    for split_seed, _split in splits.items():
        temperatures = {
            model: float(
                calibrated[
                    calibrated["target"].eq(model)
                    & calibrated["split_seed"].eq(split_seed)
                    & calibrated["dose"].eq(0.0)
                ].iloc[0]["target_temperature"]
            )
            for model in model_ids
        }
        frames: dict[str, pd.DataFrame] = {}
        for target, family in target_families:
            if family not in frames:
                frames[family] = track_response_frame(
                    scores, family, temperatures=temperatures
                )
            peers = [model for model in model_ids if model != target]
            data = aligned_track_arrays(frames[family], target, peers)
            all_map, dose_map = _index_maps(data.metadata)
            datasets[(split_seed, family, target)] = {
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
    _CAL_CONTEXT = {
        "datasets": datasets,
        "split_draws": split_draws,
        "split_seeds": list(splits),
        "doses": {
            family: [float(value) for value in specification["doses"]]
            for family, specification in load_config()["families"].items()
        },
    }
    return specifications


def _draws(
    split_seed: int,
    analysis_type: str,
    target: str,
    family: str,
    replicate: int,
) -> tuple[list[str], list[str]]:
    seed = stable_seed(
        load_config()["bootstrap_seed"],
        f"calibrated_{analysis_type}",
        target,
        family,
        split_seed,
        replicate,
    )
    groups = _CAL_CONTEXT["split_draws"][split_seed]
    fitting = _draw(groups["fit"], stable_seed(seed, "fit"))
    evaluation = _draw(groups["evaluation"], stable_seed(seed, "evaluation"))
    if set(fitting).intersection(evaluation):
        raise RuntimeError("Calibrated bootstrap source-side leakage")
    return fitting, evaluation


def _fit(dataset: dict[str, Any], draw: list[str], design: np.ndarray) -> np.ndarray:
    rows = _rows(dataset["all_map"], draw)
    return fit_projection(
        design[rows],
        dataset["target_values"][rows],
        dataset["row_weights"][rows],
    ).projection.weights


def _dose_error(
    dataset: dict[str, Any],
    draw: list[str],
    dose: float,
    design: np.ndarray,
    weights: np.ndarray,
) -> float:
    rows = _rows(dataset["dose_map"], draw, dose)
    return float(
        np.mean(
            np.abs(
                dataset["target_values"][rows] - design[rows] @ weights
            )
        )
    )


def _overall(
    dataset: dict[str, Any],
    draw: list[str],
    family: str,
    design: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        np.mean(
            [
                _dose_error(dataset, draw, dose, design, weights)
                for dose in _CAL_CONTEXT["doses"][family]
            ]
        )
    )


def _worker(task: tuple[str, str, str, int, int, str]) -> str:
    analysis_type, target, family, start, end, output_name = task
    rows: list[dict[str, Any]] = []
    for replicate in range(start, end + 1):
        split_estimates: list[float] = []
        split_primary: list[float] = []
        split_comparator: list[float] = []
        selected_peers: list[str] = []
        for split_seed in _CAL_CONTEXT["split_seeds"]:
            dataset = _CAL_CONTEXT["datasets"][(split_seed, family, target)]
            design = dataset["design"]
            fitting_draw, evaluation_draw = _draws(
                split_seed, analysis_type, target, family, replicate
            )
            if analysis_type == "endpoint":
                weights = _fit(dataset, fitting_draw, design)
                clean = _dose_error(
                    dataset, evaluation_draw, 0.0, design, weights
                )
                high = _dose_error(
                    dataset,
                    evaluation_draw,
                    high_dose(family),
                    design,
                    weights,
                )
                split_primary.append(clean)
                split_comparator.append(high)
                split_estimates.append(high - clean)
            elif analysis_type == "convexity":
                fit_rows = _rows(dataset["all_map"], fitting_draw)
                weights = _fit(dataset, fitting_draw, design)
                peer_errors = np.average(
                    np.abs(
                        design[fit_rows]
                        - dataset["target_values"][fit_rows, None]
                    ),
                    axis=0,
                    weights=dataset["row_weights"][fit_rows],
                )
                selected_index = int(np.argmin(peer_errors))
                single_weights = np.zeros(len(dataset["peers"]), dtype=np.float64)
                single_weights[selected_index] = 1.0
                convex = _overall(
                    dataset, evaluation_draw, family, design, weights
                )
                single = _overall(
                    dataset,
                    evaluation_draw,
                    family,
                    design,
                    single_weights,
                )
                split_primary.append(convex)
                split_comparator.append(single)
                split_estimates.append(single - convex)
                selected_peers.append(dataset["peers"][selected_index])
            elif analysis_type == "sibling_removal":
                sibling = sibling_for(target)
                if sibling is None:
                    raise ValueError(f"No sibling for selected target {target}")
                retained = [
                    index
                    for index, peer in enumerate(dataset["peers"])
                    if peer != sibling
                ]
                removed_design = design[:, retained]
                all_weights = _fit(dataset, fitting_draw, design)
                removed_weights = _fit(dataset, fitting_draw, removed_design)
                all_error = _overall(
                    dataset, evaluation_draw, family, design, all_weights
                )
                removed_error = _overall(
                    dataset,
                    evaluation_draw,
                    family,
                    removed_design,
                    removed_weights,
                )
                split_primary.append(all_error)
                split_comparator.append(removed_error)
                split_estimates.append(removed_error - all_error)
            else:
                raise ValueError(f"Unknown calibrated bootstrap type {analysis_type}")
        rows.append(
            {
                "replicate": replicate,
                "seed": stable_seed(
                    load_config()["bootstrap_seed"],
                    f"calibrated_{analysis_type}",
                    target,
                    family,
                    replicate,
                ),
                "metric": float(np.mean(split_estimates)),
                "primary_value": float(np.mean(split_primary)),
                "comparator_value": float(np.mean(split_comparator)),
                "positive_split_count": int(
                    np.sum(np.asarray(split_estimates) > 0)
                ),
                "negative_split_count": int(
                    np.sum(np.asarray(split_estimates) < 0)
                ),
                "split_estimates": split_estimates,
                "fit_selected_single_peers": selected_peers,
                "temperature_refit_within_bootstrap": False,
                "temperature_conditioning": "fixed clean-fitting split estimate",
            }
        )
    path = Path(output_name)
    write_bootstrap_shard(
        path,
        pd.DataFrame(rows),
        analysis_type=f"calibrated_{analysis_type}",
        target=target,
        family=family,
        replicate_start=start,
        replicate_end=end,
        schema_version=CALIBRATED_BOOTSTRAP_SCHEMA,
    )
    return str(path)


def _shard_path(
    analysis_type: str, target: str, family: str, start: int, end: int
) -> Path:
    return (
        PHYSICAL_ROOT
        / "bootstrap_shards/selected_sensitivities"
        / f"calibrated-{analysis_type}__{slug(target)}__{family}__{start:04d}_{end:04d}.parquet"
    )


def _expected(
    specifications: list[tuple[str, str, str]], *, resume: bool
) -> tuple[list[tuple[str, str, str, int, int, str]], list[Path]]:
    replicates = int(load_config()["bootstrap_replicates"])
    shard_size = int(load_config()["bootstrap_shard_size"])
    tasks: list[tuple[str, str, str, int, int, str]] = []
    outputs: list[Path] = []
    for analysis_type, target, family in specifications:
        for start in range(0, replicates, shard_size):
            end = min(replicates - 1, start + shard_size - 1)
            path = _shard_path(analysis_type, target, family, start, end)
            expected = {
                "analysis_type": f"calibrated_{analysis_type}",
                "target": target,
                "family": family,
                "replicate_start": start,
                "replicate_end": end,
                "schema_version": CALIBRATED_BOOTSTRAP_SCHEMA,
            }
            if not (resume and valid_bootstrap_shard(path, **expected)):
                tasks.append((analysis_type, target, family, start, end, str(path)))
            outputs.extend([path, _metadata_path(path), _completion_path(path)])
    return tasks, outputs


def run_selected_calibrated_bootstraps(
    *, workers: int, resume: bool
) -> dict[str, Any]:
    specifications = _prepare_context()
    tasks, expected_outputs = _expected(specifications, resume=resume)
    print(
        f"[calibrated-bootstrap] {len(tasks)} shard(s), {workers} workers",
        flush=True,
    )
    if tasks:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(_worker, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                print(
                    f"[calibrated-bootstrap] {completed}/{len(tasks)} complete: {future.result()}",
                    flush=True,
                )
    frames: list[pd.DataFrame] = []
    for analysis_type, target, family in specifications:
        paths = [
            path
            for path in expected_outputs
            if path.suffix == ".parquet"
            and f"calibrated-{analysis_type}__{slug(target)}__{family}__" in path.name
        ]
        for path in paths:
            start_end = path.stem.rsplit("__", 1)[-1].split("_")
            start, end = int(start_end[0]), int(start_end[1])
            if not valid_bootstrap_shard(
                path,
                analysis_type=f"calibrated_{analysis_type}",
                target=target,
                family=family,
                replicate_start=start,
                replicate_end=end,
                schema_version=CALIBRATED_BOOTSTRAP_SCHEMA,
            ):
                raise RuntimeError(f"Missing/corrupt calibrated shard: {path}")
            frames.append(pd.read_parquet(path))
    all_results = pd.concat(frames, ignore_index=True)
    summaries: list[dict[str, Any]] = []
    for keys, group in all_results.groupby(
        ["analysis_type", "target", "family"], sort=True
    ):
        values = group["metric"].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "analysis_type": keys[0],
                "target": keys[1],
                "family": keys[2],
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_lower": float(np.quantile(values, 0.025)),
                "bootstrap_upper": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0)),
                "probability_negative": float(np.mean(values < 0)),
                "replicates": len(values),
                "interval_estimand": BOOTSTRAP_WORDING,
                "temperature_refit_within_bootstrap": False,
                "temperature_conditioning": "fixed clean-fitting split estimate",
            }
        )
    summary = pd.DataFrame(summaries)
    if len(summary) != len(specifications) or not summary["replicates"].eq(1000).all():
        raise ValueError("Selected calibrated bootstrap is incomplete")
    summary_path = (
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/selected_calibrated_bootstrap_summary.csv"
    )
    atomic_csv(summary_path, summary)
    return {
        "outputs": [*expected_outputs, summary_path],
        "row_counts": {
            "selected_calibrated_bootstrap": len(all_results),
            "selected_calibrated_summary": len(summary),
            "selected_calibrated_shards": len(
                [path for path in expected_outputs if path.suffix == ".parquet"]
            ),
        },
        "warnings": [
            "Selected calibrated intervals refit convex projections and fit-only single peers, but condition on each split's clean-fit temperature estimate."
        ],
    }
