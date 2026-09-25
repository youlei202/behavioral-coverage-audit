from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import tempfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNBOOK_ROOT = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2")
PERSISTENT_ROOT = Path("/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2")


def project_root() -> Path:
    override = os.environ.get("PIER_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    if RUNBOOK_ROOT.exists():
        return RUNBOOK_ROOT.resolve()
    return PERSISTENT_ROOT.resolve()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u64(*parts: object) -> int:
    joined = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def slug(value: str) -> str:
    return value.replace("/", "--").replace(" ", "_")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as handle:
            for row in rows:
                payload = canonical_json_bytes(row)
                handle.write(payload)
                digest.update(payload)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return count, digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def tree_sha256(root: Path, excluded_parts: set[str] | None = None) -> str:
    excluded_parts = excluded_parts or {"__pycache__", ".pytest_cache", ".ruff_cache"}
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if any(part in excluded_parts for part in rel.parts):
            continue
        digest.update(str(rel).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def environment_basics() -> dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "os": platform.platform(),
        "hostname_class": socket.gethostname().split(".")[0],
        "python": platform.python_version(),
        "machine": platform.machine(),
    }

