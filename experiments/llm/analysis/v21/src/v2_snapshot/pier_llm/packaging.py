from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from .utils import sha256_file

PACKAGE_NAME = "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_RESULTS"

_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"https?://[^/@\s]+:[^/@\s]+@"),
)
_FORBIDDEN_COMPONENTS = {".git", "env", "hf_cache", "wheelhouse"}
_FORBIDDEN_FILENAMES = {
    ".git-credentials",
    ".netrc",
    "credentials.json",
    "token",
    "token.json",
}
_MODEL_WEIGHT_SUFFIXES = {".bin", ".ckpt", ".gguf", ".pt", ".pth", ".safetensors"}


def _iter_files(root: Path, relative_roots: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for relative in relative_roots:
        path = root / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file()
                and "__pycache__" not in item.parts
                and item.suffix not in {".pyc", ".pyo"}
            )
    return sorted(set(files), key=lambda path: str(path.relative_to(root)))


def _archive_destination(relative: Path) -> Path:
    if relative == Path("outputs/analysis/GOLDMINE_REPORT.md"):
        return Path(PACKAGE_NAME) / "GOLDMINE_REPORT.md"
    if relative == Path("REPRODUCE.md"):
        return Path(PACKAGE_NAME) / "REPRODUCE.md"
    if relative.parts[:2] == ("outputs", "analysis"):
        return Path(PACKAGE_NAME) / "analysis" / Path(*relative.parts[2:])
    if relative.parts[:2] == ("outputs", "controls"):
        return Path(PACKAGE_NAME) / "controls" / Path(*relative.parts[2:])
    if relative.parts[:2] == ("outputs", "figures"):
        return Path(PACKAGE_NAME) / "figures" / Path(*relative.parts[2:])
    if relative.parts[:2] == ("outputs", "generation_validation"):
        return Path(PACKAGE_NAME) / "generation_validation" / Path(*relative.parts[2:])
    if relative.parts[:2] == ("outputs", "raw_scores"):
        return Path(PACKAGE_NAME) / "raw_scores" / Path(*relative.parts[2:])
    if relative.parts[:2] == ("outputs", "tables"):
        return Path(PACKAGE_NAME) / "tables" / Path(*relative.parts[2:])
    return Path(PACKAGE_NAME) / relative


def build_results_package(
    root: Path,
    *,
    output_path: Path | None = None,
    include_raw_threshold_bytes: int = 10 * 1024**3,
    strict_validation: bool = False,
) -> Path:
    output_path = output_path or root / "artifacts" / f"{PACKAGE_NAME}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_roots = [
        "outputs/analysis",
        "configs",
        "outputs/controls",
        "outputs/figures",
        "outputs/generation_validation",
        "logs",
        "scripts",
        "src",
        "status",
        "outputs/tables",
        "REPRODUCE.md",
    ]
    raw_files = _iter_files(root, ["outputs/raw_scores"])
    raw_bytes = sum(path.stat().st_size for path in raw_files)
    include_raw = raw_bytes <= include_raw_threshold_bytes
    if include_raw:
        relative_roots.append("outputs/raw_scores")
    files = _iter_files(root, relative_roots)
    manifest_lines = [
        f"{sha256_file(path)}  {path.relative_to(root)}" for path in files
    ]
    raw_hash_lines = []
    if not include_raw:
        raw_hash_lines = [f"{sha256_file(path)}  {path}" for path in raw_files]
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in files:
                relative = path.relative_to(root)
                destination = _archive_destination(relative)
                archive.write(path, destination)
            archive.writestr(
                f"{PACKAGE_NAME}/MANIFEST.sha256", "\n".join(manifest_lines) + "\n"
            )
            if raw_hash_lines:
                archive.writestr(
                    f"{PACKAGE_NAME}/RAW_SCORE_PATHS_AND_HASHES.txt",
                    "\n".join(raw_hash_lines) + "\n",
                )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_results_package(output_path, strict=strict_validation)
    return output_path


def validate_results_package(
    path: Path, *, strict: bool = False
) -> dict[str, bool | int | str]:
    required = {
        f"{PACKAGE_NAME}/GOLDMINE_REPORT.md",
        f"{PACKAGE_NAME}/REPRODUCE.md",
        f"{PACKAGE_NAME}/MANIFEST.sha256",
    }
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt member in result package: {bad}")
        names = set(archive.namelist())
        missing = required.difference(names)
        if missing:
            raise ValueError(f"Result package is missing: {sorted(missing)}")
        for name in names:
            relative = Path(name).relative_to(PACKAGE_NAME)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe result-package member path: {name}")
            lowered_parts = {part.casefold() for part in relative.parts}
            if lowered_parts.intersection(_FORBIDDEN_COMPONENTS):
                raise ValueError(f"Forbidden cache/environment member in package: {name}")
            if relative.name.casefold() in _FORBIDDEN_FILENAMES:
                raise ValueError(f"Forbidden credential member in package: {name}")
            if relative.suffix.casefold() in _MODEL_WEIGHT_SUFFIXES:
                raise ValueError(f"Forbidden model-weight member in package: {name}")
            if "__pycache__" in lowered_parts or relative.suffix in {".pyc", ".pyo"}:
                raise ValueError(f"Python cache member in result package: {name}")

        manifest_name = f"{PACKAGE_NAME}/MANIFEST.sha256"
        manifest: dict[str, str] = {}
        for line in archive.read(manifest_name).decode("utf-8").splitlines():
            digest, separator, relative_text = line.partition("  ")
            if (
                separator != "  "
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not relative_text
                or relative_text in manifest
            ):
                raise ValueError("Malformed or duplicate MANIFEST.sha256 entry")
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe path in MANIFEST.sha256")
            destination = str(_archive_destination(relative))
            if destination not in names:
                raise ValueError(f"Manifest member is absent from package: {relative_text}")
            observed = hashlib.sha256(archive.read(destination)).hexdigest()
            if observed != digest:
                raise ValueError(f"Manifest SHA256 mismatch for: {relative_text}")
            manifest[relative_text] = destination
        allowed_unmanifested = {
            manifest_name,
            f"{PACKAGE_NAME}/RAW_SCORE_PATHS_AND_HASHES.txt",
        }
        unexpected = names.difference(manifest.values()).difference(allowed_unmanifested)
        if unexpected:
            raise ValueError(f"Package contains unmanifested members: {sorted(unexpected)}")

        scanned_text_members = 0
        for name in names:
            if Path(name).suffix.casefold() not in _TEXT_SUFFIXES:
                continue
            payload = archive.read(name)
            scanned_text_members += 1
            if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
                raise ValueError(f"Potential credential or secret found in package member: {name}")

        if strict:
            strict_required = {
                f"{PACKAGE_NAME}/analysis/bootstrap_headline_results.parquet",
                f"{PACKAGE_NAME}/analysis/calibration_sensitivity.parquet",
                f"{PACKAGE_NAME}/analysis/design_transfer_results.parquet",
                f"{PACKAGE_NAME}/analysis/disco_scalar_results.parquet",
                f"{PACKAGE_NAME}/analysis/disco_vector_results.parquet",
                f"{PACKAGE_NAME}/analysis/disco_weights.parquet",
                f"{PACKAGE_NAME}/analysis/ecosystem_coverage_results.parquet",
                f"{PACKAGE_NAME}/analysis/environment_manifest.json",
                f"{PACKAGE_NAME}/analysis/generation_validation.parquet",
                f"{PACKAGE_NAME}/analysis/option_permutation_results.parquet",
                f"{PACKAGE_NAME}/analysis/peer_removal_controls.parquet",
                f"{PACKAGE_NAME}/analysis/split_stability.parquet",
                f"{PACKAGE_NAME}/analysis/weight_ambiguity_audit.parquet",
                f"{PACKAGE_NAME}/controls/control_results.json",
                f"{PACKAGE_NAME}/status/B200_INFERENCE_COMPLETE.json",
                f"{PACKAGE_NAME}/status/B200_READY.json",
                f"{PACKAGE_NAME}/status/EXPERIMENT_COMPLETE.json",
                f"{PACKAGE_NAME}/status/postprocess_status.json",
            }
            strict_missing = strict_required.difference(names)
            if strict_missing:
                raise ValueError(
                    f"Complete result package is missing: {sorted(strict_missing)}"
                )
            required_prefix_counts = {
                "configs/": 1,
                "figures/": 9,
                "generation_validation/": 1,
                "logs/": 1,
                "raw_scores/": 1,
                "tables/": 5,
            }
            for relative_prefix, minimum in required_prefix_counts.items():
                prefix = f"{PACKAGE_NAME}/{relative_prefix}"
                count = sum(name.startswith(prefix) for name in names)
                if count < minimum:
                    raise ValueError(
                        f"Complete result package has only {count} members under "
                        f"{relative_prefix}; expected at least {minimum}"
                    )
            for status_name in ("EXPERIMENT_COMPLETE.json", "postprocess_status.json"):
                member = f"{PACKAGE_NAME}/status/{status_name}"
                completed = json.loads(archive.read(member))
                if completed.get("complete") is not True:
                    raise ValueError(f"Packaged status is not complete: {status_name}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "member_count": len(names),
        "manifest_entry_count": len(manifest),
        "scanned_text_member_count": scanned_text_members,
        "strict": strict,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
