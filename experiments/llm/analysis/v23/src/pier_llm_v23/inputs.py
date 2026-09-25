from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .utils import (
    atomic_write_json,
    atomic_write_text,
    file_manifest_sha256,
    load_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)


def resolve_roots(root: Path) -> dict[str, dict[str, str]]:
    config = load_json(root / "configs/experiment_v23.json")
    records: dict[str, dict[str, str]] = {}
    for source in ("v2", "v22"):
        logical = Path(config["source_roots"][source])
        physical = logical.resolve(strict=True)
        records[source] = {
            "logical": str(logical),
            "physical": str(physical),
            "realpath_command_equivalent": os.path.realpath(logical),
        }
    atomic_write_json(root / "manifests/resolved_input_roots.json", records)
    return records


def _v2_paths(v2: Path) -> list[Path]:
    fixed = [
        "configs/experiment.json",
        "configs/resolved_models.json",
        "data/mmlu_pro_selected_560.jsonl",
        "data/mmlu_pro_generation_subset_140.jsonl",
        "data/interventions/irrelevant_context_v2.jsonl",
        "data/interventions/content_deletion_v2.jsonl",
        "data/manifests/scoring_prompts.jsonl",
        "data/manifests/generation_prompts.jsonl",
        "env/requirements.lock.txt",
        "src/pier_llm/data_prep.py",
        "src/pier_llm/inference.py",
        "src/pier_llm/solver.py",
        "src/pier_llm/analysis.py",
        "src/pier_llm/utils.py",
        "status/B200_READY.json",
        "status/B200_INFERENCE_COMPLETE.json",
        "status/EXPERIMENT_COMPLETE.json",
    ]
    result = [v2 / relative for relative in fixed]
    result.extend(sorted((v2 / "data/manifests/models").glob("*.json")))
    result.extend(sorted((v2 / "data/manifests/rendered_prompts").glob("*.jsonl")))
    result.extend(sorted((v2 / "outputs/raw_scores").glob("*/shard_*.parquet")))
    result.extend(sorted((v2 / "outputs/raw_scores").glob("*/shard_*.meta.json")))
    result.extend(sorted((v2 / "outputs/generation_validation").glob("*/shard_*.parquet")))
    result.extend(sorted((v2 / "outputs/generation_validation").glob("*/shard_*.meta.json")))

    resolved = load_json(v2 / "configs/resolved_models.json")
    snapshots = [Path(record["local_cache_path"]) for record in resolved["models"]]
    tiny = next(
        (v2 / "hf_cache/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots").iterdir()
    )
    snapshots.append(tiny)
    for snapshot in snapshots:
        result.extend(path for path in sorted(snapshot.rglob("*")) if path.is_file())
    return sorted(set(result), key=str)


def _v22_paths(v22: Path) -> list[Path]:
    fixed = [
        "configs/reanalysis_v22.json",
        "outputs/analysis/V2_2_TRACKWISE_ESTIMAND_REPORT.md",
        "outputs/analysis/trackwise_primary_weights.parquet",
        "outputs/analysis/trackwise_endpoint_effects.parquet",
        "outputs/analysis/trackwise_convex_vs_single.parquet",
        "outputs/analysis/trackwise_sibling_removal.parquet",
        "outputs/analysis/trackwise_non_sibling_removal_null.parquet",
        "outputs/analysis/trackwise_claim_survival.parquet",
        "outputs/analysis/trackwise_interface_sensitivity.parquet",
        "outputs/bootstrap_summaries/endpoint_bootstrap_summary.csv",
        "outputs/bootstrap_summaries/convexity_bootstrap_summary.csv",
        "outputs/bootstrap_summaries/sibling_removal_bootstrap_summary.csv",
        "status/V2_2_REANALYSIS_COMPLETE.json",
        "manifests/INPUTS_BEFORE.sha256",
        "manifests/INPUTS_AFTER.sha256",
    ]
    result = [v22 / relative for relative in fixed]
    result.extend(sorted((v22 / "src/pier_llm_reanalysis_v22").glob("*.py")))
    return sorted(set(result), key=str)


def source_paths(root: Path) -> dict[str, list[Path]]:
    resolutions = resolve_roots(root)
    v2 = Path(resolutions["v2"]["physical"])
    v22 = Path(resolutions["v22"]["physical"])
    paths = {"v2": _v2_paths(v2), "v22": _v22_paths(v22)}
    for source, values in paths.items():
        missing = [str(path) for path in values if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {source} inputs: {missing[:5]}")
    return paths


def _record(source: str, root: Path, path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(root))
    stat = path.stat()
    return {
        "source": source,
        "relative_path": relative,
        "logical_path": str(Path(f"/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_{'V2' if source == 'v2' else 'V2_2_TRACKWISE_REANALYSIS'}") / relative),
        "physical_path": str(path.resolve(strict=True)),
        "size": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def build_inventory(root: Path, *, workers: int = 12) -> list[dict[str, Any]]:
    resolutions = resolve_roots(root)
    paths = source_paths(root)
    work: list[tuple[str, Path, Path]] = []
    for source in ("v2", "v22"):
        source_root = Path(resolutions[source]["physical"])
        work.extend((source, source_root, path) for path in paths[source])
    print(f"[inputs] hashing {len(work)} immutable source files", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(lambda values: _record(*values), work))
    return sorted(records, key=lambda row: (row["source"], row["relative_path"]))


def _sha_lines(records: list[dict[str, Any]]) -> str:
    return "".join(
        f"{record['sha256']}  {record['source']}/{record['relative_path']}\n"
        for record in records
    )


def group_sha256(records: list[dict[str, Any]], source: str) -> str:
    subset = [record for record in records if record["source"] == source]
    return file_manifest_sha256(subset)


def record_before(root: Path) -> dict[str, Any]:
    records = build_inventory(root)
    atomic_write_json(root / "manifests/input_inventory_before.json", records)
    atomic_write_text(root / "manifests/INPUTS_BEFORE.sha256", _sha_lines(records))
    result = {
        "schema_version": "pier_v23_input_inventory_v1",
        "created_at": utc_now(),
        "file_count": len(records),
        "source_v2_sha256": group_sha256(records, "v2"),
        "source_v22_sha256": group_sha256(records, "v22"),
        "combined_sha256": file_manifest_sha256(records),
        "total_bytes": int(sum(record["size"] for record in records)),
    }
    atomic_write_json(root / "status/input_inventory_before.json", result)
    return result


def recheck_after(root: Path) -> dict[str, Any]:
    before = load_json(root / "manifests/input_inventory_before.json")
    after = build_inventory(root)
    atomic_write_json(root / "manifests/input_inventory_after.json", after)
    atomic_write_text(root / "manifests/INPUTS_AFTER.sha256", _sha_lines(after))
    left = {(row["source"], row["relative_path"]): (row["size"], row["sha256"]) for row in before}
    right = {(row["source"], row["relative_path"]): (row["size"], row["sha256"]) for row in after}
    if left != right:
        changed = sorted(set(left) | set(right))
        changed = [key for key in changed if left.get(key) != right.get(key)]
        raise RuntimeError(f"Immutable source inputs changed: {changed[:5]}")
    result = {
        "byte_identical": True,
        "checked_at": utc_now(),
        "file_count": len(after),
        "combined_sha256": file_manifest_sha256(after),
    }
    atomic_write_json(root / "status/input_inventory_after.json", result)
    return result


def compact_source_tree_digest(root: Path) -> str:
    inventory = load_json(root / "manifests/input_inventory_before.json")
    return sha256_bytes(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())
