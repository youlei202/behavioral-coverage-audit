from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import _aligned, _evaluate_endpoint, atomic_parquet
from .core import (
    REPAIR_DIAGNOSTIC_COLUMNS,
    common_bootstrap_multiplicities,
    endpoint_row_weights,
    fit_mae_simplex,
    fit_mse_simplex,
    fixed_splits,
    select_single_peer,
    weighted_mae,
)
from .utils import atomic_write_json, load_json, sha256_file, slug, utc_now

BOOTSTRAP_SCHEMA = "pier_v23_common_resample_bootstrap_v1"
BOOTSTRAP_WORDING = (
    "common-base-question clustered-bootstrap interval for the ten-split bagged estimator"
)


def _paths(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(".meta.json"), path.with_suffix(".complete.json")


def valid_shard(path: Path, expected: dict[str, Any]) -> bool:
    metadata_path, completion_path = _paths(path)
    try:
        metadata = load_json(metadata_path)
        completion = load_json(completion_path)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if metadata.get("sha256") != sha256_file(path):
            return False
        if completion.get("complete") is not True or completion.get("sha256") != metadata["sha256"]:
            return False
        frame = pd.read_parquet(path)
        return bool(
            len(frame) == expected["replicate_end"] - expected["replicate_start"] + 1
            and frame["replicate"].tolist()
            == list(range(expected["replicate_start"], expected["replicate_end"] + 1))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_shard(path: Path, frame: pd.DataFrame, expected: dict[str, Any]) -> None:
    metadata_path, completion_path = _paths(path)
    atomic_parquet(path, frame)
    digest = sha256_file(path)
    metadata = {
        **expected,
        "row_count": len(frame),
        "sha256": digest,
        "SHA256": digest,
        "completion_marker": str(completion_path),
        "interval_estimand": BOOTSTRAP_WORDING,
        "created_at": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        completion_path,
        {
            **expected,
            "complete": True,
            "row_count": len(frame),
            "sha256": digest,
            "completed_at": utc_now(),
        },
    )
    if not valid_shard(path, expected):
        raise RuntimeError(f"New bootstrap shard failed validation: {path}")


def _response_path(root: Path, interface: str) -> Path:
    name = (
        "raw_endpoint_response.parquet"
        if interface == "raw_endpoint"
        else "permutation_endpoint_response.parquet"
    )
    return root / "outputs/validation" / name


def _multiplicity_vector(metadata: pd.DataFrame, mapping: dict[str, int]) -> np.ndarray:
    return metadata["base_question_id"].astype(str).map(mapping).to_numpy(dtype=np.float64)


def _repair_sidecar(root: Path, shard: Path) -> Path:
    return (
        root
        / "bootstrap_shards/repair_diagnostics"
        / shard.parent.name
        / shard.name
    )


def _job(
    root_text: str,
    analysis_type: str,
    interface: str,
    target: str,
    family: str,
    replicate_start: int,
    replicate_end: int,
    output_text: str,
) -> str:
    root = Path(root_text)
    output = Path(output_text)
    expected = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "analysis_type": analysis_type,
        "interface": interface,
        "target": target,
        "family": family,
        "replicate_start": replicate_start,
        "replicate_end": replicate_end,
    }
    if valid_shard(output, expected):
        return str(output)
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    selected = pd.read_json(
        v2 / "data/mmlu_pro_selected_560.jsonl", lines=True, dtype={"base_question_id": str}
    )
    model_ids = load_json(root / "configs/resolved_models_v23.json")
    model_ids = [record["id"] for record in model_ids["models"]]
    frame = pd.read_parquet(_response_path(root, interface))
    metadata, values = _aligned(frame, family, model_ids)
    identifiers = metadata["base_question_id"].astype(str)
    splits = fixed_splits(selected, config["split_seeds"])
    target_index = model_ids.index(target)
    peers = [model for model in model_ids if model != target]
    peer_indices = [model_ids.index(peer) for peer in peers]
    y = values[:, target_index]
    x = values[:, peer_indices]
    sibling = config["sibling_directions"].get(target)
    rows: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    for replicate in range(replicate_start, replicate_end + 1):
        mapping = common_bootstrap_multiplicities(
            selected, replicate, int(config["bootstrap_seed"])
        )
        multiplicity = _multiplicity_vector(metadata, mapping)
        estimates: list[dict[str, float]] = []
        for split in splits.values():
            fitting = identifiers.isin(split.fitting_ids).to_numpy()
            evaluation = identifiers.isin(split.evaluation_ids).to_numpy()
            fit_mass = endpoint_row_weights(metadata, multiplicity)[fitting]
            eval_mass = endpoint_row_weights(metadata, multiplicity)[evaluation]
            base_context = {
                "analysis_type": analysis_type,
                "interface": interface,
                "model": target,
                "target": target,
                "family": family,
                "split": int(split.seed),
                "bootstrap_replicate": int(replicate),
                "shard_id": output.name,
            }
            if analysis_type == "endpoint":
                fitted = fit_mse_simplex(
                    x[fitting],
                    y[fitting],
                    fit_mass,
                    evaluation_design=x[evaluation],
                    repair_context={**base_context, "fit_role": "endpoint"},
                    repair_diagnostics=repair_records,
                )
                estimates.append(
                    _evaluate_endpoint(metadata, y, x, fitted, split.evaluation_ids, mapping)
                )
            elif analysis_type == "convexity":
                mse_single = select_single_peer(x[fitting], y[fitting], fit_mass, "mse")
                mae_single = select_single_peer(x[fitting], y[fitting], fit_mass, "mae")
                mse_convex = fit_mse_simplex(
                    x[fitting],
                    y[fitting],
                    fit_mass,
                    evaluation_design=x[evaluation],
                    repair_context={**base_context, "fit_role": "convex"},
                    repair_diagnostics=repair_records,
                )
                mae_convex = fit_mae_simplex(
                    x[fitting],
                    y[fitting],
                    fit_mass,
                    evaluation_design=x[evaluation],
                    repair_context={**base_context, "fit_role": "convex"},
                    repair_diagnostics=repair_records,
                )
                x_eval, y_eval = x[evaluation], y[evaluation]
                estimates.append(
                    {
                        "MSE_improvement": weighted_mae(
                            y_eval, x_eval[:, mse_single], eval_mass
                        )
                        - weighted_mae(y_eval, x_eval @ mse_convex, eval_mass),
                        "MAE_improvement": weighted_mae(
                            y_eval, x_eval[:, mae_single], eval_mass
                        )
                        - weighted_mae(y_eval, x_eval @ mae_convex, eval_mass),
                    }
                )
            elif analysis_type == "sibling_removal":
                if sibling is None:
                    raise ValueError(f"No declared sibling for {target}")
                all_weights = fit_mse_simplex(
                    x[fitting],
                    y[fitting],
                    fit_mass,
                    evaluation_design=x[evaluation],
                    repair_context={**base_context, "fit_role": "all_peers"},
                    repair_diagnostics=repair_records,
                )
                baseline = _evaluate_endpoint(
                    metadata, y, x, all_weights, split.evaluation_ids, mapping
                )["endpoint_design_pier"]
                retained = [index for index, peer in enumerate(peers) if peer != sibling]
                removed_weights = fit_mse_simplex(
                    x[fitting][:, retained],
                    y[fitting],
                    fit_mass,
                    evaluation_design=x[evaluation][:, retained],
                    repair_context={**base_context, "fit_role": "sibling_removed"},
                    repair_diagnostics=repair_records,
                )
                removed = _evaluate_endpoint(
                    metadata,
                    y,
                    x[:, retained],
                    removed_weights,
                    split.evaluation_ids,
                    mapping,
                )["endpoint_design_pier"]
                estimates.append({"sibling_inflation": removed - baseline})
            else:
                raise ValueError(analysis_type)
        keys = estimates[0]
        rows.append(
            {
                "schema_version": BOOTSTRAP_SCHEMA,
                "analysis_type": analysis_type,
                "interface": interface,
                "target": target,
                "family": family,
                "replicate": replicate,
                "global_multiplicity_seed": int(config["bootstrap_seed"]),
                "shared_across_all_ten_splits": True,
                "interval_estimand": BOOTSTRAP_WORDING,
                **{
                    key: float(np.mean([estimate[key] for estimate in estimates]))
                    for key in keys
                },
            }
        )
    atomic_parquet(
        _repair_sidecar(root, output),
        pd.DataFrame(repair_records, columns=REPAIR_DIAGNOSTIC_COLUMNS),
    )
    write_shard(output, pd.DataFrame(rows), expected)
    return str(output)


def _jobs(root: Path) -> list[tuple[Any, ...]]:
    config = load_json(root / "configs/experiment_v23.json")
    model_ids = [
        record["id"] for record in load_json(root / "configs/resolved_models_v23.json")["models"]
    ]
    replicates = int(config["bootstrap_replicates"])
    shard_size = int(config["bootstrap_shard_size"])
    jobs: list[tuple[Any, ...]] = []
    for analysis_type in ("endpoint", "convexity", "sibling_removal"):
        targets = (
            list(config["sibling_directions"])
            if analysis_type == "sibling_removal"
            else model_ids
        )
        for interface in ("raw_endpoint", "permutation_averaged"):
            for target in targets:
                for family in ("irrelevant_context", "content_deletion"):
                    directory = root / "bootstrap_shards" / analysis_type
                    for start in range(0, replicates, shard_size):
                        end = min(replicates - 1, start + shard_size - 1)
                        filename = (
                            f"{slug(interface)}__{slug(target)}__{family}__{start:04d}_{end:04d}.parquet"
                        )
                        jobs.append(
                            (
                                str(root),
                                analysis_type,
                                interface,
                                target,
                                family,
                                start,
                                end,
                                str(directory / filename),
                            )
                        )
    return jobs


def _job_expected(job: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "schema_version": BOOTSTRAP_SCHEMA,
        "analysis_type": job[1],
        "interface": job[2],
        "target": job[3],
        "family": job[4],
        "replicate_start": job[5],
        "replicate_end": job[6],
    }


def _valid_job_counts(jobs: list[tuple[Any, ...]]) -> dict[str, int]:
    counts = {"endpoint": 0, "convexity": 0, "sibling_removal": 0}
    for job in jobs:
        if valid_shard(Path(job[7]), _job_expected(job)):
            counts[str(job[1])] += 1
    return counts


def _pending_jobs(
    jobs: list[tuple[Any, ...]], *, resume: bool
) -> list[tuple[Any, ...]]:
    return [
        job
        for job in jobs
        if not (resume and valid_shard(Path(job[7]), _job_expected(job)))
    ]


def _audit_existing_weight_solutions(root: Path) -> dict[str, Any]:
    specifications = (
        ("frozen_weight_interface_transfer.parquet", "frozen_v22_weights"),
        ("matched_objective_convexity.parquet", "MSE_convex_weights"),
        ("matched_objective_convexity.parquet", "MAE_convex_weights"),
        ("permutation_endpoint_design_results.parquet", "weights"),
        ("raw_endpoint_design_results.parquet", "weights"),
    )
    records: list[dict[str, Any]] = []
    all_finite = True
    all_strict = True
    total_solutions = 0
    for filename, column in specifications:
        path = root / "outputs/analysis" / filename
        frame = pd.read_parquet(path, columns=[column])
        minimum = float("inf")
        maximum_sum_error = 0.0
        finite = True
        strict = True
        for value in frame[column]:
            weights = np.asarray(value, dtype=np.float64).reshape(-1)
            value_finite = bool(weights.size and np.isfinite(weights).all())
            finite = finite and value_finite
            if not value_finite:
                strict = False
                continue
            minimum = min(minimum, float(weights.min()))
            maximum_sum_error = max(
                maximum_sum_error,
                abs(float(weights.sum()) - 1.0),
            )
            strict = strict and float(weights.min()) >= -1e-14
            strict = strict and abs(float(weights.sum()) - 1.0) <= 1e-12
        record = {
            "path": str(path),
            "column": column,
            "solution_count": int(len(frame)),
            "all_finite": finite,
            "minimum_weight": minimum if np.isfinite(minimum) else None,
            "maximum_sum_to_one_error": maximum_sum_error,
            "strict_simplex_feasible": strict,
        }
        records.append(record)
        total_solutions += len(frame)
        all_finite = all_finite and finite
        all_strict = all_strict and strict
    if not all_finite or not all_strict:
        raise RuntimeError("Existing stored weight-solution audit failed")
    return {
        "solution_count": int(total_solutions),
        "all_finite": all_finite,
        "all_strict_simplex_feasible": all_strict,
        "records": records,
    }


def _consolidate_repair_diagnostics(
    root: Path,
    jobs: list[tuple[Any, ...]],
    shard_counts: dict[str, int],
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for job in jobs:
        output = Path(job[7])
        if not valid_shard(output, _job_expected(job)):
            raise RuntimeError(f"Cannot consolidate diagnostics for invalid shard: {output}")
        sidecar = _repair_sidecar(root, output)
        if sidecar.is_file():
            frame = pd.read_parquet(sidecar)
            if list(frame.columns) != list(REPAIR_DIAGNOSTIC_COLUMNS):
                raise RuntimeError(f"Repair diagnostic schema mismatch: {sidecar}")
            frames.append(frame)
    repairs = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=REPAIR_DIAGNOSTIC_COLUMNS)
    )
    numeric_columns = (
        "raw_minimum_weight",
        "raw_sum_to_one_error",
        "repaired_minimum_weight",
        "repaired_sum_to_one_error",
        "l1_weight_change",
        "l2_weight_change",
        "linf_weight_change",
        "raw_objective",
        "repaired_objective",
        "absolute_objective_change",
        "relative_objective_change",
        "maximum_fitted_prediction_change",
        "maximum_evaluation_prediction_change",
    )
    if len(repairs):
        values = repairs[list(numeric_columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("Repair diagnostics contain non-finite values")
        gates = (
            repairs["raw_minimum_weight"].ge(-1e-7).all()
            and repairs["raw_sum_to_one_error"].le(1e-7).all()
            and repairs["repaired_minimum_weight"].ge(-1e-14).all()
            and repairs["repaired_sum_to_one_error"].le(1e-12).all()
            and repairs["linf_weight_change"].le(1e-7).all()
            and repairs["maximum_fitted_prediction_change"].le(1e-6).all()
            and repairs["maximum_evaluation_prediction_change"].le(1e-6).all()
        )
        objective_tolerance = np.maximum(
            1e-10,
            1e-7 * np.maximum(1.0, repairs["raw_objective"].abs()),
        )
        gates = bool(gates and repairs["absolute_objective_change"].le(objective_tolerance).all())
        if not gates:
            raise RuntimeError("Consolidated repair diagnostics failed strict acceptance gates")
    output = root / "outputs/validation/numerical_feasibility_repairs.parquet"
    atomic_parquet(output, repairs)
    audit = _audit_existing_weight_solutions(root)

    def maximum(column: str) -> float:
        return float(repairs[column].max()) if len(repairs) else 0.0

    summary = {
        "schema_version": "pier_v231_numerical_feasibility_repair_summary_v1",
        "created_at": utc_now(),
        "passed": True,
        "authorization_marker": str(root / "status/V2_3_1_AUTHORIZATION.json"),
        "diagnostics_path": str(output),
        "diagnostics_sha256": sha256_file(output),
        "repaired_solution_count": int(len(repairs)),
        "maximum_raw_negativity": max(0.0, -float(repairs["raw_minimum_weight"].min())) if len(repairs) else 0.0,
        "maximum_l1_repair": maximum("l1_weight_change"),
        "maximum_l2_repair": maximum("l2_weight_change"),
        "maximum_linf_repair": maximum("linf_weight_change"),
        "maximum_absolute_objective_change": maximum("absolute_objective_change"),
        "maximum_relative_objective_change": maximum("relative_objective_change"),
        "maximum_fitted_prediction_change": maximum("maximum_fitted_prediction_change"),
        "maximum_evaluation_prediction_change": maximum("maximum_evaluation_prediction_change"),
        "existing_stored_solution_audit": audit,
        "final_shard_counts": {
            "endpoint": shard_counts["endpoint"],
            "convexity": shard_counts["convexity"],
            "sibling_removal": shard_counts["sibling_removal"],
            "total": int(sum(shard_counts.values())),
            "pending": 0,
            "failed": 0,
        },
    }
    atomic_write_json(root / "status/numerical_feasibility_repair_summary.json", summary)
    return summary


def run_common_bootstraps(root: Path, *, resume: bool = True) -> dict[str, Any]:
    config = load_json(root / "configs/experiment_v23.json")
    jobs = _jobs(root)
    pending = _pending_jobs(jobs, resume=resume)
    print(f"[bootstrap] valid={len(jobs) - len(pending)} pending={len(pending)}", flush=True)
    failures: list[dict[str, str]] = []
    if pending:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=int(config["bootstrap_workers"]), mp_context=context
        ) as executor:
            futures = {executor.submit(_job, *job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "analysis_type": job[1],
                            "interface": job[2],
                            "target": job[3],
                            "family": job[4],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    if failures:
        atomic_write_json(root / "status/bootstrap_failures.json", failures)
        raise RuntimeError(f"Common-resample bootstrap failures: {failures[:3]}")
    shard_counts = _valid_job_counts(jobs)
    required_counts = {"endpoint": 320, "convexity": 320, "sibling_removal": 160}
    if shard_counts != required_counts:
        raise RuntimeError(
            f"Final bootstrap shard gate failed: {shard_counts} != {required_counts}"
        )
    atomic_write_json(root / "status/bootstrap_failures.json", [])
    repair_summary = _consolidate_repair_diagnostics(root, jobs, shard_counts)
    outputs: dict[str, dict[str, Any]] = {}
    names = {
        "endpoint": "common_resample_endpoint_bootstrap.parquet",
        "convexity": "common_resample_convexity_bootstrap.parquet",
        "sibling_removal": "common_resample_sibling_bootstrap.parquet",
    }
    for analysis_type, filename in names.items():
        paths = sorted(Path(job[7]) for job in jobs if job[1] == analysis_type)
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        expected_rows = required_counts[analysis_type] * int(config["bootstrap_shard_size"])
        if len(frame) != expected_rows:
            raise RuntimeError(
                f"Merged bootstrap row gate failed for {analysis_type}: "
                f"{len(frame)} != {expected_rows}"
            )
        output = root / "outputs/analysis" / filename
        atomic_parquet(
            output,
            frame.sort_values(["interface", "target", "family", "replicate"]),
        )
        outputs[filename] = {"row_count": len(frame), "sha256": sha256_file(output)}
    status = {
        "schema_version": "pier_v231_common_resample_status_v1",
        "completed_at": utc_now(),
        "passed": True,
        "replicates": int(config["bootstrap_replicates"]),
        "shared_global_multiplicity_per_replicate": True,
        "interval_wording": BOOTSTRAP_WORDING,
        "shard_count": len(jobs),
        "shard_counts": shard_counts,
        "pending": 0,
        "failed": 0,
        "numerical_feasibility_repairs": {
            "count": repair_summary["repaired_solution_count"],
            "diagnostics_sha256": repair_summary["diagnostics_sha256"],
        },
        "outputs": outputs,
    }
    atomic_write_json(root / "status/common_resample_bootstrap.json", status)
    return status
