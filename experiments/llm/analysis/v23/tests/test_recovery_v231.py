from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pier_llm_v23.bootstrap import (
    BOOTSTRAP_SCHEMA,
    _jobs,
    _pending_jobs,
    write_shard,
)
from pier_llm_v23.core import (
    REPAIR_DIAGNOSTIC_COLUMNS,
    validate_or_repair_simplex,
)
from pier_llm_v23.utils import load_json, project_root, sha256_file


def _validate(
    raw: np.ndarray,
    *,
    design: np.ndarray | None = None,
    target: np.ndarray | None = None,
    evaluation_design: np.ndarray | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> np.ndarray:
    x = np.asarray(design if design is not None else [[0.0, 1.0], [1.0, 0.0]])
    y = np.asarray(target if target is not None else x @ raw)
    return validate_or_repair_simplex(
        raw,
        design=x,
        target=y,
        mass=np.ones(len(y)),
        objective="mse",
        solver_status="Optimal",
        solver_objective=0.0,
        evaluation_design=evaluation_design,
        repair_context={
            "analysis_type": "convexity",
            "interface": "raw_endpoint",
            "model": "test/model",
            "target": "test/model",
            "family": "content_deletion",
            "split": 20260828,
            "bootstrap_replicate": 2,
            "shard_id": "test.parquet",
            "fit_role": "convex",
        },
        repair_diagnostics=diagnostics,
    )


def test_tiny_negative_weight_is_projected_and_diagnosed() -> None:
    raw = np.asarray([-4.2e-8, 1.000000042])
    diagnostics: list[dict[str, object]] = []
    repaired = _validate(raw, diagnostics=diagnostics)
    np.testing.assert_array_equal(repaired, np.asarray([0.0, 1.0]))
    assert len(diagnostics) == 1
    assert tuple(diagnostics[0]) == REPAIR_DIAGNOSTIC_COLUMNS
    assert diagnostics[0]["raw_minimum_weight"] == pytest.approx(-4.2e-8)
    assert diagnostics[0]["linf_weight_change"] <= 1e-7


def test_exact_simplex_solution_is_unchanged_without_diagnostic() -> None:
    raw = np.asarray([0.2, 0.3, 0.5])
    diagnostics: list[dict[str, object]] = []
    design = np.eye(3)
    repaired = _validate(raw, design=design, target=design @ raw, diagnostics=diagnostics)
    np.testing.assert_array_equal(repaired, raw)
    assert diagnostics == []


def test_negative_weight_beyond_authorized_tolerance_is_rejected() -> None:
    raw = np.asarray([-1.1e-7, 1.00000011])
    with pytest.raises(RuntimeError, match="raw minimum"):
        _validate(raw)


def test_sum_error_beyond_authorized_tolerance_is_rejected() -> None:
    raw = np.asarray([0.5, 0.50000011])
    with pytest.raises(RuntimeError, match="raw sum error"):
        _validate(raw)


def test_objective_change_gate_is_enforced() -> None:
    raw = np.asarray([-4.2e-8, 1.000000042])
    design = np.asarray([[0.0, 10.0]])
    target = design @ raw - 1.0
    with pytest.raises(RuntimeError, match="objective-change gate"):
        _validate(raw, design=design, target=target)


def test_fitted_prediction_change_gate_is_enforced() -> None:
    raw = np.asarray([-4.2e-8, 1.000000042])
    design = np.asarray([[0.0, 100.0]])
    with pytest.raises(RuntimeError, match="fitted-prediction gate"):
        _validate(raw, design=design, target=design @ raw)


def test_evaluation_prediction_change_gate_is_enforced() -> None:
    raw = np.asarray([-4.2e-8, 1.000000042])
    design = np.asarray([[0.0, 1.0]])
    evaluation = np.asarray([[0.0, 100.0]])
    with pytest.raises(RuntimeError, match="evaluation-prediction gate"):
        _validate(
            raw,
            design=design,
            target=design @ raw,
            evaluation_design=evaluation,
        )


def test_resume_preserves_valid_shard_and_selects_only_missing(tmp_path: Path) -> None:
    output = tmp_path / "valid.parquet"
    frame = pd.DataFrame({"replicate": [0]})
    job = (
        str(tmp_path),
        "endpoint",
        "raw_endpoint",
        "test/model",
        "content_deletion",
        0,
        0,
        str(output),
    )
    expected = {
        "schema_version": BOOTSTRAP_SCHEMA,
        "analysis_type": job[1],
        "interface": job[2],
        "target": job[3],
        "family": job[4],
        "replicate_start": job[5],
        "replicate_end": job[6],
    }
    write_shard(output, frame, expected)
    before = {
        path: sha256_file(path)
        for path in (output, output.with_suffix(".meta.json"), output.with_suffix(".complete.json"))
    }
    missing = (*job[:-1], str(tmp_path / "missing.parquet"))
    assert _pending_jobs([job, missing], resume=True) == [missing]
    after = {path: sha256_file(path) for path in before}
    assert after == before


def test_frozen_bootstrap_seed_and_job_count_are_unchanged() -> None:
    root = project_root()
    config = load_json(root / "configs/experiment_v23.json")
    assert config["bootstrap_seed"] == 20260828
    assert config["bootstrap_replicates"] == 1000
    assert config["bootstrap_shard_size"] == 100
    assert config["split_seeds"] == list(range(20260828, 20260838))
    assert len(_jobs(root)) == 800
