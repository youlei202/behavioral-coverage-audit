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

LOGICAL_ROOT = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS")
PHYSICAL_ROOT = Path(
    "/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    return load_json(PHYSICAL_ROOT / "configs/reanalysis_v22.json")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=json_default,
        )
        + "\n",
    )


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big")


def stable_u64(*parts: object) -> int:
    """Byte-identical split hash used by V2."""
    joined = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip(
        "-"
    )


def output_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "path": str(path.relative_to(PHYSICAL_ROOT)),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        record["row_count"] = int(pq.ParquetFile(path).metadata.num_rows)
    elif path.suffix == ".csv":
        record["row_count"] = int(len(pd.read_csv(path)))
    return record


def output_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [output_record(path) for path in paths]


def stage_marker_path(stage_name: str) -> Path:
    return PHYSICAL_ROOT / "status" / f"stage_{stage_name}.json"


def stage_is_valid(stage_name: str) -> bool:
    path = stage_marker_path(stage_name)
    if not path.is_file():
        return False
    try:
        marker = load_json(path)
        if marker.get("success") is not True:
            return False
        for record in marker.get("outputs", []):
            output = PHYSICAL_ROOT / record["path"]
            if not output.is_file() or output.stat().st_size != record["size"]:
                return False
            if sha256_file(output) != record["sha256"]:
                return False
            if output.suffix == ".parquet":
                observed = int(pq.ParquetFile(output).metadata.num_rows)
                if observed != record.get("row_count"):
                    return False
            elif output.suffix == ".csv":
                observed = int(len(pd.read_csv(output)))
                if observed != record.get("row_count"):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def write_stage_marker(
    stage_name: str,
    *,
    started_at: str,
    outputs: Iterable[Path],
    inputs: Iterable[Path] = (),
    row_counts: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    success: bool,
    error: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_list = list(outputs)
    input_list = [path for path in inputs if path.is_file()]
    marker: dict[str, Any] = {
        "schema_version": "pier_reanalysis_stage_v22_v1",
        "stage": stage_name,
        "started_at": started_at,
        "completed_at": utc_now(),
        "inputs": [
            {
                "path": str(path),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in input_list
        ],
        "outputs": output_records(output_list) if success else [],
        "row_counts": row_counts or {},
        "checksums": {
            str(path.relative_to(PHYSICAL_ROOT)): sha256_file(path)
            for path in output_list
            if success
        },
        "warnings": warnings or [],
        "success": bool(success),
        "error": error,
    }
    if extra:
        marker.update(extra)
    path = stage_marker_path(stage_name)
    atomic_write_json(path, marker)
    return path


def tree_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
        elif path.is_file():
            files.append(path)
    return sorted(set(files), key=str)


def write_source_tree_manifest() -> Path:
    files = tree_files(
        [
            PHYSICAL_ROOT / "configs/reanalysis_v22.json",
            PHYSICAL_ROOT / "env/requirements.lock.txt",
            PHYSICAL_ROOT / "pyproject.toml",
            PHYSICAL_ROOT / "scripts",
            PHYSICAL_ROOT / "src/pier_llm_reanalysis_v22",
            PHYSICAL_ROOT / "tests",
        ]
    )
    path = PHYSICAL_ROOT / "manifests/reanalysis_source_tree.sha256"
    lines = [
        f"{sha256_file(file)}  {file.relative_to(PHYSICAL_ROOT)}\n" for file in files
    ]
    atomic_write_text(path, "".join(lines))
    return path


def environment_manifest() -> dict[str, Any]:
    wanted = {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "cvxpy",
        "osqp",
        "clarabel",
        "matplotlib",
        "pytest",
    }
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and name.lower() in wanted:
            installed[name] = distribution.version
    lock = PHYSICAL_ROOT / "env/requirements.lock.txt"
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    cpuset = Path("/sys/fs/cgroup/cpuset.cpus.effective")
    return {
        "schema_version": "pier_reanalysis_environment_v22_v1",
        "timestamp": utc_now(),
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count_visible": os.cpu_count(),
        "cgroup_cpu_max": cpu_max.read_text(encoding="utf-8").strip()
        if cpu_max.is_file()
        else None,
        "cgroup_cpuset_effective": cpuset.read_text(encoding="utf-8").strip()
        if cpuset.is_file()
        else None,
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
        "requirements_lock_path": str(lock),
        "requirements_lock_sha256": sha256_file(lock),
        "packages": dict(sorted(installed.items(), key=lambda item: item[0].lower())),
    }


def weight_geometry(weights: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(weights, dtype=np.float64)
    positive = values[values > 0]
    ordered = np.sort(values)[::-1]
    return {
        "effective_peer_count": float(1.0 / np.square(values).sum()),
        "top1_weight_mass": float(ordered[:1].sum()),
        "top3_weight_mass": float(ordered[:3].sum()),
        "weight_entropy": float(-np.sum(positive * np.log(positive)))
        if positive.size
        else 0.0,
        "support_size_1e3": int(np.sum(values >= 1e-3)),
        "support_size_1e2": int(np.sum(values >= 1e-2)),
    }
