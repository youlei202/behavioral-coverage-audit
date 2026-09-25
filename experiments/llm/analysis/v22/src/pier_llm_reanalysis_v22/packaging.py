from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .utils import (
    LOGICAL_ROOT,
    PHYSICAL_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now,
)

PACKAGE_NAME = "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS_RESULTS"
ZIP_PATH = PHYSICAL_ROOT / "artifacts" / f"{PACKAGE_NAME}.zip"

REQUIRED_ANALYSIS = [
    "trackwise_primary_dose_results.parquet",
    "trackwise_primary_weights.parquet",
    "trackwise_overall_results.parquet",
    "trackwise_endpoint_effects.parquet",
    "estimand_decomposition.parquet",
    "cancellation_gap_results.parquet",
    "trackwise_convex_vs_single.parquet",
    "trackwise_sibling_removal.parquet",
    "trackwise_non_sibling_removal_null.parquet",
    "trackwise_single_peer_influence.parquet",
    "trackwise_calibrated_results.parquet",
    "trackwise_interface_sensitivity.parquet",
    "trackwise_vector_results.parquet",
    "trackwise_claim_survival.parquet",
    "solver_diagnostics.parquet",
    "V2_2_TRACKWISE_ESTIMAND_REPORT.md",
]

REQUIRED_BOOTSTRAP = [
    "endpoint_bootstrap_summary.csv",
    "convexity_bootstrap_summary.csv",
    "sibling_removal_bootstrap_summary.csv",
    "selected_calibrated_bootstrap_summary.csv",
    "BOOTSTRAP_SHARD_INVENTORY.json",
]

REQUIRED_FIGURES = [
    "fig_v22_trackwise_corrected_main.pdf",
    "fig_v22_trackwise_corrected_main.png",
    "fig_v22_estimand_decomposition.pdf",
    "fig_v22_trackwise_dose_trajectories.pdf",
    "fig_v22_cancellation_by_target.pdf",
    "fig_v22_convexity_gap.pdf",
    "fig_v22_sibling_removal.pdf",
    "fig_v22_calibration_sensitivity.pdf",
    "fig_v22_interface_sensitivity.pdf",
    "fig_v22_vector_support.pdf",
    "fig_v22_solver_diagnostics.pdf",
]


def _required_paths() -> list[Path]:
    paths = [PHYSICAL_ROOT / "outputs/analysis" / name for name in REQUIRED_ANALYSIS]
    paths.extend(
        PHYSICAL_ROOT / "outputs/bootstrap_summaries" / name
        for name in REQUIRED_BOOTSTRAP
    )
    paths.extend(PHYSICAL_ROOT / "figures" / name for name in REQUIRED_FIGURES)
    paths.extend(
        [
            PHYSICAL_ROOT / "controls/V2_2_CONTROLS.json",
            PHYSICAL_ROOT / "controls/synthetic_controls.parquet",
            PHYSICAL_ROOT / "manifests/INPUTS_BEFORE.sha256",
            PHYSICAL_ROOT / "manifests/INPUTS_AFTER.sha256",
            PHYSICAL_ROOT / "manifests/input_inventory_before.json",
            PHYSICAL_ROOT / "manifests/input_inventory_after.json",
            PHYSICAL_ROOT / "manifests/reanalysis_environment.json",
            PHYSICAL_ROOT / "manifests/reanalysis_source_tree.sha256",
            PHYSICAL_ROOT / "REPRODUCE_V2_2.md",
        ]
    )
    return paths


def verify_completeness() -> dict[str, Any]:
    missing = [str(path) for path in _required_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required V2.2 outputs are missing: {missing}")
    for number, name in enumerate(
        [
            "01_input_validation",
            "02_unit_and_synthetic_tests",
            "03_trackwise_primary_fit",
            "04_estimand_decomposition",
            "05_endpoint_analysis",
            "06_convexity_analysis",
            "07_sibling_and_peer_removal",
            "08_primary_bootstrap",
            "09_calibration_sensitivity",
            "10_interface_sensitivity",
            "11_vector_sensitivity",
            "12_tables_figures_report",
            "13_input_immutability_recheck",
        ],
        start=1,
    ):
        path = PHYSICAL_ROOT / "status" / f"stage_{name}.json"
        if not path.is_file() or load_json(path).get("success") is not True:
            raise RuntimeError(f"Required stage is incomplete: {number:02d} {name}")
    before = (PHYSICAL_ROOT / "manifests/INPUTS_BEFORE.sha256").read_bytes()
    after = (PHYSICAL_ROOT / "manifests/INPUTS_AFTER.sha256").read_bytes()
    if before != after:
        raise RuntimeError("Before/after immutable-input manifests differ")
    return {
        "required_output_count": len(_required_paths()),
        "source_inputs_byte_identical": True,
        "stages_verified": 13,
    }


def write_original_input_paths() -> Path:
    inventory = load_json(PHYSICAL_ROOT / "manifests/input_inventory_before.json")
    lines = [
        "# Immutable original inputs used by PIER V2.2\n",
        "# source | logical path | resolved physical path | SHA256\n",
    ]
    for record in inventory:
        lines.append(
            f"{record['source']} | {record['absolute_logical_path']} | "
            f"{record['resolved_physical_path']} | {record['sha256']}\n"
        )
    path = PHYSICAL_ROOT / "manifests/ORIGINAL_INPUT_PATHS_AND_HASHES.txt"
    atomic_write_text(path, "".join(lines))
    return path


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"),
    )


def _manifest_lines(root: Path) -> str:
    files = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"],
        key=lambda path: str(path.relative_to(root)),
    )
    return "".join(
        f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files
    )


def _write_zip(source_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(source_root.rglob("*"), key=str):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=str(Path(PACKAGE_NAME) / path.relative_to(source_root)),
                    )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_results_package() -> dict[str, Any]:
    completeness = verify_completeness()
    original_paths = write_original_input_paths()
    final_marker = PHYSICAL_ROOT / "status/V2_2_REANALYSIS_COMPLETE.json"
    atomic_write_json(
        final_marker,
        {
            "schema_version": "pier_reanalysis_complete_v22_v1",
            "complete": True,
            "completed_at": utc_now(),
            "result_package": str(
                LOGICAL_ROOT
                / "artifacts/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS_RESULTS.zip"
            ),
            "source_inputs_byte_identical": True,
            "model_inference_performed": False,
            "b200_inference_performed": False,
            **completeness,
        },
    )
    with tempfile.TemporaryDirectory(
        prefix="pier-v22-package-", dir=PHYSICAL_ROOT / "artifacts"
    ) as temporary_name:
        staging = Path(temporary_name) / PACKAGE_NAME
        staging.mkdir(parents=True)
        shutil.copy2(
            PHYSICAL_ROOT / "outputs/analysis/V2_2_TRACKWISE_ESTIMAND_REPORT.md",
            staging / "V2_2_TRACKWISE_ESTIMAND_REPORT.md",
        )
        shutil.copy2(PHYSICAL_ROOT / "REPRODUCE_V2_2.md", staging / "REPRODUCE_V2_2.md")
        shutil.copy2(original_paths, staging / "ORIGINAL_INPUT_PATHS_AND_HASHES.txt")
        _copy_tree(PHYSICAL_ROOT / "outputs/analysis", staging / "analysis")
        _copy_tree(
            PHYSICAL_ROOT / "outputs/bootstrap_summaries",
            staging / "bootstrap_summaries",
        )
        _copy_tree(PHYSICAL_ROOT / "configs", staging / "configs")
        _copy_tree(PHYSICAL_ROOT / "controls", staging / "controls")
        _copy_tree(PHYSICAL_ROOT / "figures", staging / "figures")
        _copy_tree(PHYSICAL_ROOT / "logs", staging / "logs")
        _copy_tree(PHYSICAL_ROOT / "manifests", staging / "manifests")
        _copy_tree(PHYSICAL_ROOT / "scripts", staging / "scripts")
        _copy_tree(PHYSICAL_ROOT / "src", staging / "src")
        _copy_tree(PHYSICAL_ROOT / "status", staging / "status")
        _copy_tree(PHYSICAL_ROOT / "outputs/tables", staging / "tables")
        atomic_write_text(staging / "MANIFEST.sha256", _manifest_lines(staging))
        _write_zip(staging, ZIP_PATH)
    if not ZIP_PATH.is_file() or ZIP_PATH.stat().st_size == 0:
        raise RuntimeError("Final V2.2 result ZIP was not created")
    return {
        "outputs": [ZIP_PATH, final_marker, original_paths],
        "row_counts": {"packaged_files": len(_required_paths())},
        "package_sha256": sha256_file(ZIP_PATH),
        "package_size": ZIP_PATH.stat().st_size,
    }
