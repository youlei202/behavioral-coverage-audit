from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .utils import (
    ORIGINAL_ROOT,
    PHYSICAL_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now,
)

PACKAGE_NAME = "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS_RESULTS"
PACKAGE_PATH = PHYSICAL_ROOT / "artifacts" / f"{PACKAGE_NAME}.zip"


REQUIRED_OUTPUTS = [
    "outputs/analysis/paired_endpoint_effects.parquet",
    "outputs/analysis/track_specific_pier.parquet",
    "outputs/analysis/track_aggregation_sensitivity.parquet",
    "outputs/analysis/peer_removal_null_corrected.parquet",
    "outputs/analysis/single_peer_removal_influence.parquet",
    "outputs/analysis/convex_vs_single_corrected.parquet",
    "outputs/analysis/calibrated_dose_trajectories.parquet",
    "outputs/analysis/label_bias_estimates.parquet",
    "outputs/analysis/label_bias_heldout_validation.parquet",
    "outputs/analysis/interface_corrected_pier.parquet",
    "outputs/analysis/interface_effect_survival.parquet",
    "outputs/analysis/generation_alignment_subsets.parquet",
    "outputs/analysis/generation_alignment_interface_check.parquet",
    "outputs/analysis/corrected_gold_candidates.parquet",
    "outputs/analysis/V2_1_CORRECTED_REANALYSIS_REPORT.md",
    "outputs/controls/label_bias_synthetic_control.json",
    "outputs/controls/reanalysis_test_summary.json",
    "outputs/manifests/INPUTS_BEFORE.sha256",
    "outputs/manifests/INPUTS_AFTER.sha256",
    "outputs/manifests/input_inventory_before.json",
    "outputs/manifests/input_inventory_after.json",
    "outputs/manifests/reanalysis_environment.json",
    "outputs/manifests/reanalysis_source_tree.sha256",
    "outputs/tables/table_v21_endpoint_effects.csv",
    "outputs/tables/table_v21_track_consistency.csv",
    "outputs/tables/table_v21_sibling_removal.csv",
    "outputs/tables/table_v21_convexity_gap.csv",
    "outputs/tables/table_v21_calibration_sensitivity.csv",
    "outputs/tables/table_v21_interface_sensitivity.csv",
    "outputs/tables/table_v21_generation_alignment.csv",
    "outputs/tables/table_v21_corrected_target_summary.csv",
    "outputs/tables/table_v21_corrected_gold_candidates.csv",
    "outputs/figures/fig_v21_corrected_main.pdf",
    "outputs/figures/fig_v21_corrected_main.png",
    "outputs/figures/fig_v21_calibrated_dose_trajectories.pdf",
    "outputs/figures/fig_v21_label_bias_vectors.pdf",
    "outputs/figures/fig_v21_generation_alignment_subset.pdf",
    "outputs/figures/fig_v21_single_peer_removal_influence.pdf",
    "outputs/figures/fig_v21_track_aggregation_sensitivity.pdf",
    "outputs/figures/fig_v21_interface_effect_survival.pdf",
    "logs/cpu_reanalysis_v21.log",
    "REPRODUCE_V2_1.md",
]


def _source_files() -> list[Path]:
    files: list[Path] = []
    for relative in ("src", "scripts", "configs"):
        for path in (PHYSICAL_ROOT / relative).rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if ".egg-info" in path.as_posix():
                continue
            files.append(path)
    return sorted(files, key=lambda path: str(path.relative_to(PHYSICAL_ROOT)))


def write_source_tree_manifest() -> Path:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(PHYSICAL_ROOT)}\n"
        for path in _source_files()
    ]
    output = PHYSICAL_ROOT / "outputs/manifests/reanalysis_source_tree.sha256"
    atomic_write_text(output, "".join(lines))
    return output


def require_complete_outputs() -> list[Path]:
    missing = [relative for relative in REQUIRED_OUTPUTS if not (PHYSICAL_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required V2.1 outputs: {missing}")
    if not load_json(PHYSICAL_ROOT / "outputs/manifests/input_inventory_after.json"):
        raise ValueError("The final input inventory is empty")
    return [PHYSICAL_ROOT / relative for relative in REQUIRED_OUTPUTS]


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)


def _copy_tree_files(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if ".pre_" in path.name:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if ".egg-info" in path.as_posix():
            continue
        _copy_file(path, destination / path.relative_to(source))


def _original_inputs_text() -> str:
    inventory = load_json(PHYSICAL_ROOT / "outputs/manifests/input_inventory_before.json")
    lines = [
        "# Original immutable V2 inputs\n",
        f"# Root: {ORIGINAL_ROOT}\n",
        "# SHA256  bytes  persistent path\n",
    ]
    for record in inventory:
        lines.append(
            f"{record['sha256']}  {record['size']}  {record['path']}\n"
        )
    return "".join(lines)


def _write_manifest(package_root: Path) -> Path:
    lines: list[str] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{sha256_file(path)}  {path.relative_to(package_root).as_posix()}\n")
    manifest = package_root / "MANIFEST.sha256"
    manifest.write_text("".join(lines), encoding="utf-8")
    return manifest


def _verify_staging_manifest(package_root: Path) -> None:
    manifest = package_root / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = package_root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Staging manifest verification failed: {relative}")


def _secret_scan(package_root: Path) -> None:
    patterns = [
        re.compile(rb"hf_[A-Za-z0-9]{20,}"),
        re.compile(rb"AKIA[0-9A-Z]{16}"),
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(rb"Bearer\s+[A-Za-z0-9._-]{20,}"),
    ]
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".png", ".parquet"}:
            continue
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in patterns):
            raise ValueError(f"Potential secret detected in package member: {path}")


def _build_staging(package_root: Path) -> None:
    report = PHYSICAL_ROOT / "outputs/analysis/V2_1_CORRECTED_REANALYSIS_REPORT.md"
    _copy_file(report, package_root / report.name)
    _copy_file(PHYSICAL_ROOT / "REPRODUCE_V2_1.md", package_root / "REPRODUCE_V2_1.md")
    for relative in ("outputs/analysis", "outputs/controls", "outputs/figures", "outputs/manifests", "outputs/tables"):
        source = PHYSICAL_ROOT / relative
        destination_name = relative.split("/")[-1]
        _copy_tree_files(source, package_root / destination_name)
    for relative in ("configs", "scripts", "src", "status"):
        _copy_tree_files(PHYSICAL_ROOT / relative, package_root / relative)
    _copy_tree_files(PHYSICAL_ROOT / "logs", package_root / "logs")

    bootstrap_summaries = package_root / "bootstrap_summaries"
    bootstrap_summaries.mkdir(parents=True, exist_ok=True)
    for name in (
        "table_v21_endpoint_effects.csv",
        "table_v21_sibling_removal.csv",
        "table_v21_convexity_gap.csv",
    ):
        _copy_file(PHYSICAL_ROOT / "outputs/tables" / name, bootstrap_summaries / name)
    shard_inventory: list[dict[str, Any]] = []
    for metadata_path in sorted((PHYSICAL_ROOT / "outputs/bootstrap").glob("*/*.meta.json")):
        metadata = load_json(metadata_path)
        shard_inventory.append(metadata)
    (bootstrap_summaries / "BOOTSTRAP_SHARD_INVENTORY.json").write_text(
        json.dumps(shard_inventory, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (package_root / "ORIGINAL_INPUT_PATHS_AND_HASHES.txt").write_text(
        _original_inputs_text(), encoding="utf-8"
    )
    _write_manifest(package_root)
    _verify_staging_manifest(package_root)
    _secret_scan(package_root)


def _zip_staging(package_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=(Path(PACKAGE_NAME) / path.relative_to(package_root)).as_posix(),
                    )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_archive(path: Path) -> dict[str, Any]:
    forbidden_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx"}
    required = {
        f"{PACKAGE_NAME}/V2_1_CORRECTED_REANALYSIS_REPORT.md",
        f"{PACKAGE_NAME}/REPRODUCE_V2_1.md",
        f"{PACKAGE_NAME}/MANIFEST.sha256",
        f"{PACKAGE_NAME}/ORIGINAL_INPUT_PATHS_AND_HASHES.txt",
        f"{PACKAGE_NAME}/status/REANALYSIS_COMPLETE.json",
        f"{PACKAGE_NAME}/figures/fig_v21_corrected_main.pdf",
        f"{PACKAGE_NAME}/tables/table_v21_corrected_gold_candidates.csv",
    }
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC integrity test failed")
        names = archive.namelist()
        missing = sorted(required.difference(names))
        if missing:
            raise ValueError(f"Archive required-member check failed: {missing}")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != PACKAGE_NAME:
                raise ValueError(f"Unsafe archive member path: {name}")
            if any(part in {"venv", ".venv", "site-packages", "hf_cache"} for part in pure.parts):
                raise ValueError(f"Forbidden environment/cache directory in archive: {name}")
            if pure.suffix.lower() in forbidden_suffixes:
                raise ValueError(f"Model-weight-like suffix in archive: {name}")
        manifest_name = f"{PACKAGE_NAME}/MANIFEST.sha256"
        manifest = archive.read(manifest_name).decode("utf-8")
        for line in manifest.splitlines():
            digest, relative = line.split("  ", 1)
            member = f"{PACKAGE_NAME}/{relative}"
            if member not in names:
                raise ValueError(f"Manifest member missing from archive: {member}")
            observed = hashlib.sha256(archive.read(member)).hexdigest()
            if observed != digest:
                raise ValueError(f"Manifest digest mismatch in archive: {member}")
    return {
        "archive_integrity": True,
        "manifest_verified": True,
        "required_members_present": True,
        "secret_scan_passed": True,
        "unsafe_paths_absent": True,
        "model_weight_suffixes_absent": True,
        "virtual_environment_absent": True,
        "member_count": len(names),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def build_results_package() -> dict[str, Any]:
    require_complete_outputs()
    complete_status = {
        "schema_version": "pier_reanalysis_complete_v21_v1",
        "complete": True,
        "timestamp": utc_now(),
        "stage": "complete",
        "original_inputs_byte_identical": True,
        "new_model_inference_run": False,
        "gpu_host_used": False,
        "package_path": str(PACKAGE_PATH),
        "errors": [],
    }
    complete_path = PHYSICAL_ROOT / "status/REANALYSIS_COMPLETE.json"
    atomic_write_json(complete_path, complete_status)
    with tempfile.TemporaryDirectory(prefix="pier_v21_package_", dir=PHYSICAL_ROOT / "artifacts") as name:
        package_root = Path(name) / PACKAGE_NAME
        package_root.mkdir(parents=True)
        _build_staging(package_root)
        _zip_staging(package_root, PACKAGE_PATH)
    validation = validate_archive(PACKAGE_PATH)
    return {
        "outputs": [PACKAGE_PATH, complete_path],
        "package_path": str(PACKAGE_PATH),
        "validation": validation,
    }
