from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .utils import LOGICAL_ROOT, atomic_write_json, atomic_write_text, sha256_file, utc_now

PACKAGE_NAME = "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS"


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        destination.mkdir(parents=True, exist_ok=True)
        return
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=False)


def _raw_inventory(root: Path) -> str:
    lines = [
        "# Raw Stage-2 candidate-score shards omitted from the ZIP",
        "# Exact persistent paths and SHA256 values follow.",
        "",
    ]
    for path in sorted((root / "outputs/raw_scores").glob("**/*")):
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {path.resolve()}")
    return "\n".join(lines) + "\n"


def _manifest_lines(package_root: Path) -> str:
    lines: list[str] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{sha256_file(path)}  {path.relative_to(package_root)}")
    return "\n".join(lines) + "\n"


def create_package(root: Path) -> dict[str, Any]:
    required = (
        root / "status/V2_3_1_AUTHORIZATION.json",
        root / "manifests/V2_3_1_AUTHORIZED_PATCH.diff",
        root / "outputs/validation/numerical_feasibility_repairs.parquet",
        root / "status/numerical_feasibility_repair_summary.json",
        root / "outputs/validation/numerical_feasibility_repair_sensitivity.parquet",
        root / "status/V2_3_1_PREPACKAGE_GATE.json",
        root / "outputs/analysis/V2_3_FINAL_INTERFACE_VALIDATION_REPORT.md",
        root / "manifests/input_inventory_before.json",
        root / "manifests/input_inventory_after.json",
        root / "manifests/environment_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(
        str(root / "outputs/tables" / f"table_{index}_required.csv")
        for index in range(1, 7)
        if not list((root / "outputs/tables").glob(f"table_{index}_*.csv"))
    )
    if missing:
        raise FileNotFoundError(f"Required package inputs are absent: {missing}")
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    final_zip = artifacts / f"{PACKAGE_NAME}.zip"
    with tempfile.TemporaryDirectory(prefix="pier_v23_package_", dir=artifacts) as temporary:
        package_root = Path(temporary) / PACKAGE_NAME
        package_root.mkdir(parents=True)
        shutil.copy2(
            root / "outputs/analysis/V2_3_FINAL_INTERFACE_VALIDATION_REPORT.md",
            package_root / "V2_3_FINAL_INTERFACE_VALIDATION_REPORT.md",
        )
        shutil.copy2(root / "REPRODUCE_V2_3.md", package_root / "REPRODUCE_V2_3.md")
        for source, name in (
            (root / "outputs/analysis", "analysis"),
            (root / "figures", "figures"),
            (root / "outputs/generation", "generation"),
            (root / "logs", "logs"),
            (root / "manifests", "manifests"),
            (root / "outputs/tables", "tables"),
            (root / "outputs/validation", "validation"),
            (root / "scripts", "scripts"),
            (root / "src", "src"),
            (root / "status", "status"),
        ):
            _copy_tree(source, package_root / name)
        volatile_statuses = (
            "cpu_postprocess_status.json",
            "final_package.json",
            "V2_3_INTERFACE_VALIDATION_COMPLETE.json",
            "V2_3_1_STAGE_C_RECOVERY_COMPLETE.json",
        )
        for name in volatile_statuses:
            (package_root / "status" / name).unlink(missing_ok=True)
        atomic_write_text(package_root / "RAW_STAGE2_PATHS_AND_HASHES.txt", _raw_inventory(root))
        atomic_write_text(package_root / "MANIFEST.sha256", _manifest_lines(package_root))
        temporary_zip = final_zip.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(PACKAGE_NAME) / path.relative_to(package_root))
        temporary_zip.replace(final_zip)
    with zipfile.ZipFile(final_zip, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Final ZIP CRC validation failed: {bad_member}")
    result = {
        "schema_version": "pier_v231_final_package_v1",
        "created_at": utc_now(),
        "path": str(LOGICAL_ROOT / "artifacts" / f"{PACKAGE_NAME}.zip"),
        "sha256": sha256_file(final_zip),
        "size": final_zip.stat().st_size,
        "raw_scores_omitted": True,
        "model_weights_included": False,
        "zip_crc_readable": True,
        "volatile_self_referential_statuses_omitted": list(volatile_statuses),
    }
    atomic_write_json(root / "status/final_package.json", result)
    return result
