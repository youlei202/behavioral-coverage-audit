from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

LOGICAL_ROOT = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION")


def project_root() -> Path:
    value = os.environ.get("PIER_V23_ROOT")
    return Path(value).resolve() if value else LOGICAL_ROOT.resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_u64(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_seed(*parts: Any) -> int:
    return stable_u64(*parts) % (2**32)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", value).strip("-")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    _atomic_replace(path, payload)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_replace(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def tree_sha256(root: Path, *, exclude: tuple[str, ...] = ("__pycache__",)) -> str:
    records: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in exclude for part in path.parts):
            continue
        records.append((str(path.relative_to(root)), sha256_file(path)))
    return sha256_bytes(canonical_json_bytes(records))


def file_manifest_sha256(records: Iterable[dict[str, Any]]) -> str:
    compact = [(row["source"], row["relative_path"], row["sha256"]) for row in records]
    return sha256_bytes(canonical_json_bytes(compact))
