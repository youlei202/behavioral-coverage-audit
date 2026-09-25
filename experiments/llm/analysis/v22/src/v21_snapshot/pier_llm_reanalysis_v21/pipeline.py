from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

from .bootstrap import run_bootstraps
from .calibration import run_calibration_analysis
from .convexity import run_convexity_analysis
from .generation_alignment import run_generation_alignment_analysis
from .input_validation import recheck_and_record_after, validate_and_record_before
from .interface_bias import run_interface_bias_analysis
from .packaging import build_results_package, write_source_tree_manifest
from .paired_effects import run_paired_endpoint_analysis
from .peer_removal import run_peer_removal_analysis
from .plotting import make_all_figures
from .reporting import write_corrected_report
from .tables import write_all_tables
from .track_analysis import run_track_analysis
from .utils import (
    PHYSICAL_ROOT,
    atomic_write_json,
    load_json,
    sha256_file,
    stage_is_valid,
    stage_marker_path,
    utc_now,
    write_stage_marker,
)


def _run_tests() -> dict[str, Any]:
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
    result = subprocess.run(command, cwd=PHYSICAL_ROOT, check=False)
    if not junit.is_file():
        raise RuntimeError("pytest did not produce its JUnit summary")
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise RuntimeError("Unable to parse pytest JUnit summary")
    summary = {
        "schema_version": "pier_reanalysis_tests_v21_v1",
        "timestamp": utc_now(),
        "command": command,
        "tests": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "return_code": result.returncode,
        "passed": result.returncode == 0,
    }
    path = PHYSICAL_ROOT / "outputs/controls/reanalysis_test_summary.json"
    atomic_write_json(path, summary)
    if result.returncode != 0:
        raise RuntimeError(f"Required tests failed: {summary}")
    return {"outputs": [path, junit], **summary}


def _stage_01() -> dict[str, Any]:
    summary = validate_and_record_before()
    return {
        "outputs": [
            PHYSICAL_ROOT / "outputs/manifests/INPUTS_BEFORE.sha256",
            PHYSICAL_ROOT / "outputs/manifests/input_inventory_before.json",
            PHYSICAL_ROOT / "outputs/manifests/reanalysis_environment.json",
            PHYSICAL_ROOT / "status/INPUT_VALIDATION_COMPLETE.json",
        ],
        **summary,
    }


def _stage_03() -> dict[str, Any]:
    paired = run_paired_endpoint_analysis()
    track = run_track_analysis()
    return {
        "outputs": [*paired["outputs"], *track["outputs"]],
        "paired": {key: value for key, value in paired.items() if key != "outputs"},
        "track": {key: value for key, value in track.items() if key != "outputs"},
    }


def _stage_10() -> dict[str, Any]:
    tables = write_all_tables()
    figures = make_all_figures()
    report = write_corrected_report(byte_identical=None)
    source_manifest = write_source_tree_manifest()
    return {
        "outputs": [*tables["outputs"], *figures["outputs"], report, source_manifest],
        "tables": {key: value for key, value in tables.items() if key != "outputs"},
        "figures": {key: value for key, value in figures.items() if key != "outputs"},
    }


def _refresh_stage_10_marker() -> None:
    path = stage_marker_path("10_tables_figures_report")
    if not path.is_file():
        return
    marker = load_json(path)
    outputs = [PHYSICAL_ROOT / record["path"] for record in marker.get("outputs", [])]
    write_stage_marker(
        "10_tables_figures_report",
        outputs,
        extra={key: value for key, value in marker.items() if key in {"tables", "figures"}},
    )


def _stage_11() -> dict[str, Any]:
    result = recheck_and_record_after()
    report = write_corrected_report(byte_identical=True)
    _refresh_stage_10_marker()
    return {
        "outputs": [
            PHYSICAL_ROOT / "outputs/manifests/INPUTS_AFTER.sha256",
            PHYSICAL_ROOT / "outputs/manifests/input_inventory_after.json",
            report,
        ],
        **result,
    }


def _failure_marker(stage_name: str, error: BaseException) -> None:
    atomic_write_json(
        stage_marker_path(stage_name),
        {
            "schema_version": "pier_reanalysis_stage_v21_v1",
            "stage": stage_name,
            "complete": False,
            "timestamp": utc_now(),
            "input_hashes": {},
            "outputs": [],
            "errors": [
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            ],
        },
    )


def _run_stage(
    stage_name: str,
    function: Callable[[], dict[str, Any]],
    *,
    resume: bool,
) -> dict[str, Any] | None:
    if resume and stage_is_valid(stage_name):
        print(f"[pipeline] skip valid stage {stage_name}", flush=True)
        return None
    print(f"[pipeline] start stage {stage_name}", flush=True)
    try:
        result = function()
        outputs = result.pop("outputs")
        before_manifest = PHYSICAL_ROOT / "outputs/manifests/INPUTS_BEFORE.sha256"
        input_hashes = (
            {"INPUTS_BEFORE.sha256": sha256_file(before_manifest)}
            if before_manifest.is_file()
            else {}
        )
        write_stage_marker(
            stage_name,
            outputs,
            input_hashes=input_hashes,
            extra=result,
        )
        print(f"[pipeline] complete stage {stage_name}", flush=True)
        return {"outputs": outputs, **result}
    except BaseException as error:
        _failure_marker(stage_name, error)
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
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    stages: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("01_input_validation", _stage_01),
        ("02_unit_and_synthetic_tests", _run_tests),
        ("03_paired_endpoint_and_track_analysis", _stage_03),
        ("04_peer_removal_analysis", run_peer_removal_analysis),
        ("05_convexity_analysis", run_convexity_analysis),
        (
            "06_bootstrap",
            lambda: run_bootstraps(workers=arguments.workers, resume=arguments.resume),
        ),
        ("07_calibration", run_calibration_analysis),
        ("08_label_bias_correction", run_interface_bias_analysis),
        ("09_generation_alignment_subsets", run_generation_alignment_analysis),
        ("10_tables_figures_report", _stage_10),
        ("11_input_immutability_recheck", _stage_11),
        ("12_package", build_results_package),
    ]
    for stage_name, function in stages:
        _run_stage(stage_name, function, resume=arguments.resume)
    print("=" * 64, flush=True)
    print("CPU-ONLY V2.1 CORRECTED REANALYSIS COMPLETE", flush=True)
    print("", flush=True)
    print("Result package:", flush=True)
    print(
        "/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS/artifacts/"
        "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS_RESULTS.zip",
        flush=True,
    )
    print("", flush=True)
    print("No B200 inference was run.", flush=True)
    print("Original V2 inputs are byte-identical.", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
