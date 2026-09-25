from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

from .bootstrap import run_primary_bootstraps
from .calibrated_bootstrap import run_selected_calibrated_bootstraps
from .controls import write_synthetic_controls
from .data import recheck_and_record_after, validate_and_record_before
from .packaging import build_results_package
from .plotting import make_all_figures
from .point_analysis import (
    run_convexity_analysis,
    run_endpoint_analysis,
    run_estimand_decomposition,
    run_primary_fit,
    run_removal_analyses,
)
from .reporting import write_all_tables, write_report
from .sensitivities import (
    run_calibration_sensitivity,
    run_interface_sensitivity,
    run_vector_sensitivity,
)
from .utils import (
    PHYSICAL_ROOT,
    load_json,
    stage_is_valid,
    stage_marker_path,
    utc_now,
    write_source_tree_manifest,
    write_stage_marker,
)


def _stage_01() -> dict[str, Any]:
    result = validate_and_record_before()
    outputs = [
        PHYSICAL_ROOT / "manifests/INPUTS_BEFORE.sha256",
        PHYSICAL_ROOT / "manifests/input_inventory_before.json",
        PHYSICAL_ROOT / "manifests/resolved_input_roots.json",
        PHYSICAL_ROOT / "manifests/reanalysis_environment.json",
    ]
    return {
        "outputs": outputs,
        "row_counts": {"input_inventory": result["inventory_file_count"]},
        **result,
    }


def _run_tests_and_controls() -> dict[str, Any]:
    junit = PHYSICAL_ROOT / "status/reanalysis_tests.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        f"--junitxml={junit}",
        "tests",
    ]
    completed = subprocess.run(command, cwd=PHYSICAL_ROOT, check=False)
    if not junit.is_file():
        raise RuntimeError("pytest did not produce its JUnit summary")
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise RuntimeError("Unable to parse pytest JUnit summary")
    summary = {
        "tests": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "return_code": completed.returncode,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Required tests failed: {summary}")
    controls = write_synthetic_controls()
    summary_path = PHYSICAL_ROOT / "controls/reanalysis_test_summary.json"
    from .utils import atomic_write_json

    atomic_write_json(summary_path, {"passed": True, "command": command, **summary})
    return {
        "outputs": [junit, summary_path, *controls["outputs"]],
        "row_counts": {
            "tests": summary["tests"],
            **controls["row_counts"],
        },
    }


def _stage_09(*, workers: int, resume: bool) -> dict[str, Any]:
    point = run_calibration_sensitivity()
    selected = run_selected_calibrated_bootstraps(workers=workers, resume=resume)
    return {
        "outputs": [*point["outputs"], *selected["outputs"]],
        "row_counts": {**point["row_counts"], **selected["row_counts"]},
        "warnings": selected.get("warnings", []),
    }


def _stage_12() -> dict[str, Any]:
    tables = write_all_tables()
    figures = make_all_figures()
    report = write_report(byte_identical=None)
    source_manifest = write_source_tree_manifest()
    return {
        "outputs": [*tables["outputs"], *figures["outputs"], report, source_manifest],
        "row_counts": {**tables["row_counts"], **figures["row_counts"]},
    }


def _stage_13() -> dict[str, Any]:
    result = recheck_and_record_after()
    return {
        "outputs": [
            PHYSICAL_ROOT / "manifests/INPUTS_AFTER.sha256",
            PHYSICAL_ROOT / "manifests/input_inventory_after.json",
        ],
        "row_counts": {"input_inventory_after": result["file_count"]},
        **result,
    }


def _refresh_stage_12_marker() -> None:
    path = stage_marker_path("12_tables_figures_report")
    marker = load_json(path)
    outputs = [PHYSICAL_ROOT / record["path"] for record in marker["outputs"]]
    write_stage_marker(
        "12_tables_figures_report",
        started_at=marker["started_at"],
        outputs=outputs,
        row_counts=marker.get("row_counts", {}),
        warnings=marker.get("warnings", []),
        success=True,
    )


def _stage_14() -> dict[str, Any]:
    write_report(byte_identical=True)
    _refresh_stage_12_marker()
    return build_results_package()


def _failure_marker(stage_name: str, started_at: str, error: BaseException) -> None:
    write_stage_marker(
        stage_name,
        started_at=started_at,
        outputs=[],
        success=False,
        error={
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
    )


def _run_stage(
    stage_name: str,
    function: Callable[[], dict[str, Any]],
    *,
    resume: bool,
) -> None:
    if resume and stage_is_valid(stage_name):
        print(f"[pipeline] skip valid stage {stage_name}", flush=True)
        return
    started_at = utc_now()
    print(f"[pipeline] start stage {stage_name}", flush=True)
    try:
        result = function()
        outputs = result.pop("outputs")
        row_counts = result.pop("row_counts", {})
        warnings = result.pop("warnings", [])
        previous = sorted((PHYSICAL_ROOT / "status").glob("stage_*.json"))
        write_stage_marker(
            stage_name,
            started_at=started_at,
            outputs=outputs,
            inputs=previous[-1:] if previous else [],
            row_counts=row_counts,
            warnings=warnings,
            success=True,
            extra=result,
        )
        print(f"[pipeline] complete stage {stage_name}", flush=True)
    except BaseException as error:
        _failure_marker(stage_name, started_at, error)
        print(f"[pipeline] failed stage {stage_name}: {error}", flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PIER_REANALYSIS_WORKERS", "48")),
    )
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.workers > 48:
        parser.error("--workers must be between 1 and 48")
    return arguments


def main() -> None:
    arguments = parse_args()
    stages: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("01_input_validation", _stage_01),
        ("02_unit_and_synthetic_tests", _run_tests_and_controls),
        ("03_trackwise_primary_fit", run_primary_fit),
        ("04_estimand_decomposition", run_estimand_decomposition),
        ("05_endpoint_analysis", run_endpoint_analysis),
        ("06_convexity_analysis", run_convexity_analysis),
        ("07_sibling_and_peer_removal", run_removal_analyses),
        (
            "08_primary_bootstrap",
            lambda: run_primary_bootstraps(
                workers=arguments.workers, resume=arguments.resume
            ),
        ),
        (
            "09_calibration_sensitivity",
            lambda: _stage_09(workers=arguments.workers, resume=arguments.resume),
        ),
        ("10_interface_sensitivity", run_interface_sensitivity),
        ("11_vector_sensitivity", run_vector_sensitivity),
        ("12_tables_figures_report", _stage_12),
        ("13_input_immutability_recheck", _stage_13),
        ("14_packaging", _stage_14),
    ]
    for stage_name, function in stages:
        _run_stage(stage_name, function, resume=arguments.resume)
    print("=" * 61, flush=True)
    print("V2.2 TRACKWISE REANALYSIS COMPLETE", flush=True)
    print("", flush=True)
    print("Result package:", flush=True)
    print(
        "/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS/"
        "artifacts/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS_RESULTS.zip",
        flush=True,
    )
    print("", flush=True)
    print("No B200 inference was performed.", flush=True)
    print("=" * 61, flush=True)


if __name__ == "__main__":
    main()
