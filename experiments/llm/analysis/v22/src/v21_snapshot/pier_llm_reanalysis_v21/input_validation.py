from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pier_llm.inference import GENERATION_SCHEMA, SCORE_SCHEMA

from .utils import (
    ORIGINAL_ROOT,
    PHYSICAL_ROOT,
    atomic_write_json,
    atomic_write_text,
    environment_manifest,
    file_record,
    load_config,
    load_json,
    sha256_file,
    sha256_lines,
)

FIXED_DATA_FILES = (
    "data/mmlu_pro_selected_560.jsonl",
    "data/mmlu_pro_generation_subset_140.jsonl",
    "data/interventions/irrelevant_context_v2.jsonl",
    "data/interventions/content_deletion_v2.jsonl",
    "data/interventions/option_permutation_control.jsonl",
    "data/manifests/scoring_prompts.jsonl",
    "data/manifests/generation_prompts.jsonl",
)


def _metadata_for(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def _verify_shards(relative: str, schema: str) -> pd.DataFrame:
    paths = sorted((ORIGINAL_ROOT / relative).glob("*/shard_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No shards beneath {relative}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        metadata_path = _metadata_for(path)
        metadata = load_json(metadata_path)
        digest = sha256_file(path)
        if metadata.get("schema_version") != schema or metadata.get("sha256") != digest:
            raise ValueError(f"Shard checksum/schema mismatch: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(metadata["row_count"]):
            raise ValueError(f"Shard row-count mismatch: {path}")
        if not frame["schema_version"].eq(schema).all():
            raise ValueError(f"Shard row schema mismatch: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_validated_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = load_config()
    for status_name in ("B200_INFERENCE_COMPLETE.json", "EXPERIMENT_COMPLETE.json"):
        status = load_json(ORIGINAL_ROOT / "status" / status_name)
        if status.get("complete") is not True:
            raise RuntimeError(f"Original completion marker is not complete: {status_name}")

    resolved = load_json(ORIGINAL_ROOT / "configs/resolved_models.json")
    model_ids = [record["id"] for record in resolved["models"]]
    if model_ids != config["model_roster"]:
        raise ValueError("Resolved V2 model roster or ordering differs from the fixed V2.1 roster")

    scores = _verify_shards("outputs/raw_scores", SCORE_SCHEMA)
    generation = _verify_shards("outputs/generation_validation", GENERATION_SCHEMA)
    if len(scores) != 125_440 or len(generation) != 3_360:
        raise ValueError(
            f"Unexpected validated row counts: scores={len(scores)}, generation={len(generation)}"
        )
    if scores.duplicated(["model_id", "prompt_id"]).any():
        raise ValueError("Duplicate (model_id, prompt_id) rows")
    if generation.duplicated(["model_id", "generation_prompt_id"]).any():
        raise ValueError("Duplicate (model_id, generation_prompt_id) rows")
    if scores.groupby("model_id").size().to_dict() != {model: 15_680 for model in model_ids}:
        raise ValueError("Per-model raw-score count mismatch")
    if generation.groupby("model_id").size().to_dict() != {model: 420 for model in model_ids}:
        raise ValueError("Per-model generation count mismatch")

    selected = pd.read_json(
        ORIGINAL_ROOT / "data/mmlu_pro_selected_560.jsonl", lines=True, dtype={"base_question_id": str}
    )
    category_counts = selected.groupby("category")["base_question_id"].nunique()
    if len(selected) != 560 or len(category_counts) != 14 or not category_counts.eq(40).all():
        raise ValueError("The fixed 560-question design is malformed")
    return scores, generation, selected, resolved


def _sidecar_for(relative: str) -> Path:
    name = Path(relative).name + ".sha256"
    return ORIGINAL_ROOT / "data/manifests" / name


def verify_fixed_file_hashes() -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative in FIXED_DATA_FILES:
        path = ORIGINAL_ROOT / relative
        sidecar = _sidecar_for(relative)
        expected, recorded_relative = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
        if recorded_relative != relative:
            raise ValueError(f"Hash sidecar path mismatch for {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Fixed-data hash mismatch for {relative}")
        verified[relative] = observed
    return verified


def input_paths() -> list[Path]:
    paths = [ORIGINAL_ROOT / relative for relative in FIXED_DATA_FILES]
    paths.extend(_sidecar_for(relative) for relative in FIXED_DATA_FILES)
    paths.extend(
        [
            ORIGINAL_ROOT / "configs/experiment.json",
            ORIGINAL_ROOT / "configs/resolved_models.json",
            ORIGINAL_ROOT / "status/B200_INFERENCE_COMPLETE.json",
            ORIGINAL_ROOT / "status/EXPERIMENT_COMPLETE.json",
            ORIGINAL_ROOT / "env/requirements.lock.txt",
        ]
    )
    for relative in ("outputs/raw_scores", "outputs/generation_validation"):
        paths.extend(sorted((ORIGINAL_ROOT / relative).glob("*/shard_*.parquet")))
        paths.extend(sorted((ORIGINAL_ROOT / relative).glob("*/shard_*.meta.json")))
    return sorted(set(paths), key=lambda path: str(path.relative_to(ORIGINAL_ROOT)))


def build_inventory() -> list[dict[str, Any]]:
    return [file_record(path, base=ORIGINAL_ROOT) for path in input_paths()]


def compare_inventories(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    before_map = {record["relative_path"]: record for record in before}
    after_map = {record["relative_path"]: record for record in after}
    differences: list[dict[str, Any]] = []
    for relative in sorted(set(before_map) | set(after_map)):
        left = before_map.get(relative)
        right = after_map.get(relative)
        if left != right:
            differences.append({"relative_path": relative, "before": left, "after": right})
    return differences


def validate_and_record_before() -> dict[str, Any]:
    scores, generation, selected, resolved = load_validated_inputs()
    fixed_hashes = verify_fixed_file_hashes()
    inventory = build_inventory()
    manifests = PHYSICAL_ROOT / "outputs/manifests"
    atomic_write_json(manifests / "input_inventory_before.json", inventory)
    atomic_write_text(manifests / "INPUTS_BEFORE.sha256", sha256_lines(inventory))
    atomic_write_json(manifests / "reanalysis_environment.json", environment_manifest())
    summary = {
        "schema_version": "pier_input_validation_v21_v1",
        "complete": True,
        "raw_score_rows": len(scores),
        "generation_rows": len(generation),
        "selected_question_rows": len(selected),
        "model_roster": [record["id"] for record in resolved["models"]],
        "fixed_file_hashes": fixed_hashes,
        "inventory_file_count": len(inventory),
    }
    atomic_write_json(PHYSICAL_ROOT / "status/INPUT_VALIDATION_COMPLETE.json", summary)
    return summary


def recheck_and_record_after() -> dict[str, Any]:
    before = load_json(PHYSICAL_ROOT / "outputs/manifests/input_inventory_before.json")
    after = build_inventory()
    atomic_write_json(PHYSICAL_ROOT / "outputs/manifests/input_inventory_after.json", after)
    atomic_write_text(
        PHYSICAL_ROOT / "outputs/manifests/INPUTS_AFTER.sha256", sha256_lines(after)
    )
    differences = compare_inventories(before, after)
    if differences:
        raise RuntimeError(f"Original V2 inputs changed during reanalysis: {differences[:3]}")
    return {
        "byte_identical": True,
        "file_count": len(after),
        "differences": [],
    }
