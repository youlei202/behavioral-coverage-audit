#!/usr/bin/env python3
"""Recover only the archived 560-question design, using CPU and pinned dataset bytes.

No model weights, inference, interventions, or postprocessing are requested.
The original selection and serialization functions are compiled from the
checksum-verified repository source, without importing its unrelated TF-IDF
preparation code. Imports of numpy and datasets are deferred until after --help
and the existing-output checksum check.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DATASET_ID = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
DATASET_SPLIT = "test"
SAMPLE_SEED = 20260828
PER_CATEGORY = 40
EXPECTED_ROWS = 560
EXPECTED_SOURCE_ROWS = 12032
EXPECTED_SIZE = 458957
EXPECTED_SHA256 = "9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6"
EXPECTED_IDS_SHA256 = "cb7c178b616b77695a5bea82ac099f2943829560476ef1020f166a154f4119fd"
EXPECTED_CATEGORIES = {
    "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "health", "history", "law", "math", "other", "philosophy",
    "physics", "psychology",
}
SOURCE_HASHES = {
    "experiments/llm/inference/v2/src/pier_llm/data_prep.py":
        "1f8f86b978a1009f350251901005b47b4ff92b741b629932f6197adae28d7bac",
    "experiments/llm/inference/v2/src/pier_llm/utils.py":
        "5e033ca7715044879b4e3d1c36b9ae14a47520bf884f12930d9a6ea6462035fc",
    "experiments/llm/inference/v2/configs/experiment.json":
        "a0777e2f08b6e75f083531acf7bf397f6328ccc8e5e8d5f378e2e4ea04272eac",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _check_existing(output: Path) -> bool:
    if not output.exists() and not output.is_symlink():
        return False
    if not output.is_file():
        raise RuntimeError(f"Output exists and is not a regular file: {output}")
    payload = output.read_bytes()
    actual = _sha256(payload)
    if actual != EXPECTED_SHA256 or len(payload) != EXPECTED_SIZE:
        raise RuntimeError(
            f"Refusing to overwrite existing output {output}; SHA256={actual}, "
            f"size={len(payload)}. Select a new --output path."
        )
    return True


def _compile_functions(source: str, filename: Path, functions: set[str],
                       constants: set[str], namespace: dict[str, Any]) -> None:
    """Compile the exact upstream AST nodes, with no changes to their bodies."""
    tree = ast.parse(source, filename=str(filename))
    selected: list[ast.stmt] = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & constants:
                selected.append(node)
                found.update(names & constants)
    expected = functions | constants
    if found != expected:
        raise RuntimeError(f"Upstream definitions missing from {filename}: {sorted(expected - found)}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *selected], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(filename), "exec"), namespace)


def _load_source_functions(repo_root: Path) -> dict[str, Any]:
    sources: dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = repo_root / relative
        payload = path.read_bytes()
        actual = _sha256(payload)
        if actual != expected:
            raise RuntimeError(
                f"Source identity mismatch: {path}; expected {expected}, got {actual}. "
                "Use the audited repository snapshot; no output was written."
            )
        sources[relative] = payload.decode("utf-8")
    config = json.loads(sources["experiments/llm/inference/v2/configs/experiment.json"])
    for key, expected in {
        "dataset_id": DATASET_ID, "dataset_split": DATASET_SPLIT,
        "sample_seed": SAMPLE_SEED, "base_questions_per_category": PER_CATEGORY,
        "expected_categories": 14,
    }.items():
        if config[key] != expected:
            raise RuntimeError(f"Archived experiment configuration disagrees for {key}")
    manifest_path = repo_root / "experiments/llm/manifests/v2/data/manifests/dataset_source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["dataset_id"] != DATASET_ID
        or manifest["dataset_revision"] != DATASET_REVISION
        or manifest["split"] != DATASET_SPLIT
        or manifest["hashes"]["data/mmlu_pro_selected_560.jsonl"] != EXPECTED_SHA256
        or manifest["selected_id_sha256"] != EXPECTED_IDS_SHA256
        or manifest["raw_row_count"] != EXPECTED_SOURCE_ROWS
    ):
        raise RuntimeError("Archived dataset-source manifest disagrees with the recovery contract")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("CPU recovery requires numpy and datasets; no model runtime is needed") from exc
    namespace: dict[str, Any] = {
        "np": np, "json": json, "hashlib": hashlib, "os": os,
        "tempfile": tempfile, "defaultdict": defaultdict,
    }
    utils_relative = "experiments/llm/inference/v2/src/pier_llm/utils.py"
    _compile_functions(
        sources[utils_relative], repo_root / utils_relative,
        {"canonical_json_bytes", "stable_u64", "write_jsonl"}, set(), namespace,
    )
    prep_relative = "experiments/llm/inference/v2/src/pier_llm/data_prep.py"
    _compile_functions(
        sources[prep_relative], repo_root / prep_relative,
        {"_answer_index", "normalize_dataset_rows", "stratified_fixed_sample"},
        {"LABELS"}, namespace,
    )
    return namespace


def _verify_selected(selected: list[dict[str, Any]], functions: dict[str, Any]) -> bytes:
    counts = Counter(row["category"] for row in selected)
    if len(selected) != EXPECTED_ROWS or set(counts) != EXPECTED_CATEGORIES:
        raise RuntimeError(f"Selected sample has wrong row count/categories: {len(selected)}, {dict(counts)}")
    if any(count != PER_CATEGORY for count in counts.values()):
        raise RuntimeError(f"Selected sample is not 40 questions per category: {dict(counts)}")
    ids = {row["base_question_id"] for row in selected}
    if len(ids) != EXPECTED_ROWS:
        raise RuntimeError("Selected sample contains duplicate base_question_id values")
    id_hash = _sha256(functions["canonical_json_bytes"](sorted(ids)))
    if id_hash != EXPECTED_IDS_SHA256:
        raise RuntimeError(f"Selected-ID identity mismatch: {id_hash}; no output was written")
    payload = b"".join(functions["canonical_json_bytes"](row) for row in selected)
    if _sha256(payload) != EXPECTED_SHA256 or len(payload) != EXPECTED_SIZE:
        raise RuntimeError(
            f"Selected JSONL byte identity mismatch: SHA256={_sha256(payload)}, "
            f"size={len(payload)}; no output was written"
        )
    return payload


def _publish_verified(output: Path, selected: list[dict[str, Any]],
                      functions: dict[str, Any]) -> str:
    expected_payload = _verify_selected(selected, functions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="selected560-recovery-", dir=output.parent) as temporary:
        staged = Path(temporary) / "mmlu_pro_selected_560.jsonl"
        count, digest = functions["write_jsonl"](staged, selected)
        if count != EXPECTED_ROWS or digest != EXPECTED_SHA256 or staged.read_bytes() != expected_payload:
            raise RuntimeError("Upstream JSONL writer did not reproduce the verified payload")
        try:
            # Same-filesystem hard link is atomic and never replaces an existing destination.
            os.link(staged, output)
        except FileExistsError:
            if _check_existing(output):
                return "reused"
            raise
    return "recovered"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="Root of the audited public behavioral_coverage repository")
    parser.add_argument("--output", required=True, type=Path,
                        help="New selected-question JSONL path; a matching existing file is reused")
    parser.add_argument("--dataset-cache", type=Path,
                        help="Optional Hugging Face dataset cache directory")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    try:
        if _check_existing(output):
            status = "reused"
        else:
            functions = _load_source_functions(args.repo_root.expanduser().resolve())
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("CPU recovery requires the datasets package; --help needs only Python") from exc
            kwargs: dict[str, Any] = {
                "split": DATASET_SPLIT, "revision": DATASET_REVISION,
            }
            if args.dataset_cache is not None:
                kwargs["cache_dir"] = str(args.dataset_cache.expanduser().absolute())
            dataset = load_dataset(DATASET_ID, **kwargs)
            raw_rows = [dict(row) for row in dataset]
            if len(raw_rows) != EXPECTED_SOURCE_ROWS:
                raise RuntimeError(f"Pinned dataset has {len(raw_rows)} rows, expected {EXPECTED_SOURCE_ROWS}")
            normalized = functions["normalize_dataset_rows"](raw_rows)
            selected = functions["stratified_fixed_sample"](normalized, PER_CATEGORY, SAMPLE_SEED)
            status = _publish_verified(output, selected, functions)
        print(json.dumps({
            "status": status, "output": str(output), "row_count": EXPECTED_ROWS,
            "size_bytes": EXPECTED_SIZE, "sha256": EXPECTED_SHA256,
            "selected_id_sha256": EXPECTED_IDS_SHA256, "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION, "split": DATASET_SPLIT,
            "sample_seed": SAMPLE_SEED, "model_inference_run": False,
        }, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
