from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PHYSICAL_ROOT = Path("/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS")
LOGICAL_ROOT = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS")
ORIGINAL_ROOT = Path("/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    return load_json(PHYSICAL_ROOT / "configs/reanalysis_v21.json")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
        + "\n",
    )


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def file_record(path: Path, *, base: Path = ORIGINAL_ROOT) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(base)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def sha256_lines(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        f"{record['sha256']}  {record['relative_path']}\n"
        for record in sorted(records, key=lambda item: item["relative_path"])
    )


def tree_records(paths: Iterable[Path], *, base: Path) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            files.append(path)
    unique = sorted(set(files), key=lambda candidate: str(candidate.relative_to(base)))
    return [file_record(path, base=base) for path in unique]


def output_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        record: dict[str, Any] = {
            "path": str(path.relative_to(PHYSICAL_ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".parquet":
            record["row_count"] = int(pq.ParquetFile(path).metadata.num_rows)
        elif path.suffix == ".csv":
            record["row_count"] = len(pd.read_csv(path))
        records.append(record)
    return records


def stage_marker_path(stage_name: str) -> Path:
    return PHYSICAL_ROOT / "status" / f"stage_{stage_name}.json"


def stage_is_valid(stage_name: str) -> bool:
    path = stage_marker_path(stage_name)
    if not path.exists():
        return False
    try:
        marker = load_json(path)
        if marker.get("complete") is not True:
            return False
        for record in marker.get("outputs", []):
            output = PHYSICAL_ROOT / record["path"]
            if not output.is_file() or output.stat().st_size != record["size"]:
                return False
            if sha256_file(output) != record["sha256"]:
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def write_stage_marker(
    stage_name: str,
    outputs: Iterable[Path],
    *,
    input_hashes: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    marker = {
        "schema_version": "pier_reanalysis_stage_v21_v1",
        "stage": stage_name,
        "complete": True,
        "timestamp": utc_now(),
        "input_hashes": input_hashes or {},
        "outputs": output_records(outputs),
        "errors": [],
    }
    if extra:
        marker.update(extra)
    path = stage_marker_path(stage_name)
    atomic_write_json(path, marker)
    return path


def environment_manifest() -> dict[str, Any]:
    distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    lock_path = ORIGINAL_ROOT / "env/requirements.lock.txt"
    return {
        "schema_version": "pier_reanalysis_environment_v21_v1",
        "timestamp": utc_now(),
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "requirements_lock_path": str(lock_path),
        "requirements_lock_sha256": sha256_file(lock_path),
        "packages": dict(sorted(distributions.items(), key=lambda item: item[0].lower())),
    }
