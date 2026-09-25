from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import LABELS, canonical_prompt, permute_options, validate_permutation_balance
from .utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    slug,
    utc_now,
)


def _source_prompt_id(base_id: str, condition: str) -> str:
    if condition == "clean":
        return f"{base_id}__clean"
    family, track_text = condition.rsplit("_t", 1)
    track = int(track_text)
    if family == "irrelevant_context":
        return f"{base_id}__irrelevant_context__t{track}__d512"
    if family == "content_deletion":
        return f"{base_id}__content_deletion__t{track}__d0p4"
    raise ValueError(condition)


def _condition_metadata(condition: str) -> tuple[str, float, int | None, str]:
    if condition == "clean":
        return "clean", 0.0, None, "C0"
    family, track_text = condition.rsplit("_t", 1)
    track = int(track_text)
    if family == "irrelevant_context":
        return family, 512.0, track, f"I{track}"
    if family == "content_deletion":
        return family, 0.4, track, f"D{track}"
    raise ValueError(condition)


def build_manifests(root: Path) -> dict[str, Any]:
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    selected = list(read_jsonl(v2 / "data/mmlu_pro_selected_560.jsonl"))
    generation_subset = list(read_jsonl(v2 / "data/mmlu_pro_generation_subset_140.jsonl"))
    if len(selected) != 560 or len(generation_subset) != 140:
        raise ValueError("Frozen V2 question counts are not 560/140")
    selected_by_id = {str(row["base_question_id"]): row for row in selected}
    generation_ids = {str(row["base_question_id"]) for row in generation_subset}
    if len(selected_by_id) != 560 or len(generation_ids) != 140:
        raise ValueError("Frozen question IDs are duplicated")
    for row in generation_subset:
        source = selected_by_id[str(row["base_question_id"])]
        if source != row:
            raise ValueError("Generation subset row is not byte-semantically identical to selected set")

    source_prompts = {
        str(row["prompt_id"]): row
        for row in read_jsonl(v2 / "data/manifests/scoring_prompts.jsonl")
    }
    score_rows: list[dict[str, Any]] = []
    expected_source_ids: set[str] = set()
    option_counts = Counter()
    for base in selected:
        base_id = str(base["base_question_id"])
        options = [str(value) for value in base["options"]]
        option_count = len(options)
        validate_permutation_balance(option_count)
        option_counts[option_count] += 1
        for condition in config["score_conditions"]:
            source_id = _source_prompt_id(base_id, condition)
            expected_source_ids.add(source_id)
            source = source_prompts.get(source_id)
            if source is None:
                raise KeyError(f"Frozen V2 prompt is missing: {source_id}")
            family, dose, track, semantic_condition = _condition_metadata(condition)
            if (
                str(source["base_question_id"]) != base_id
                or source["question"] != (base["question"] if family == "clean" else source["question"])
                or list(source["options"]) != options
                or int(source["semantic_answer_index"]) != int(base["answer_index"])
                or list(source["presented_to_original"]) != list(range(option_count))
            ):
                raise ValueError(f"Frozen V2 prompt semantics mismatch: {source_id}")
            if family == "clean" and source["question"] != base["question"]:
                raise ValueError(f"Clean question mismatch: {source_id}")
            for rotation in range(option_count):
                presented, semantic_to_visible, visible_to_semantic = permute_options(options, rotation)
                answer_index = semantic_to_visible[int(base["answer_index"])]
                content = canonical_prompt(str(source["question"]), presented)
                prompt_id = f"{base_id}__{semantic_condition}__cyclic_r{rotation:02d}"
                score_rows.append(
                    {
                        "schema_version": "pier_v23_semantic_score_prompt_v1",
                        "prompt_id": prompt_id,
                        "source_v2_prompt_id": source_id,
                        "base_question_id": base_id,
                        "category": str(base["category"]),
                        "semantic_condition": semantic_condition,
                        "condition": family,
                        "family": family,
                        "dose": dose,
                        "track": track,
                        "permutation_id": rotation,
                        "rotation_policy": "all cyclic rotations",
                        "option_count": option_count,
                        "semantic_to_visible": semantic_to_visible,
                        "visible_to_semantic": visible_to_semantic,
                        "presented_to_original": visible_to_semantic,
                        "options": presented,
                        "option_labels": list(LABELS[:option_count]),
                        "candidate_continuations": [f" ({label})" for label in LABELS[:option_count]],
                        "answer_index": answer_index,
                        "answer_label": LABELS[answer_index],
                        "semantic_answer_index": int(base["answer_index"]),
                        "semantic_answer_label": LABELS[int(base["answer_index"])],
                        "question": str(source["question"]),
                        "canonical_content": content,
                        "canonical_content_sha256": sha256_text(content),
                        "source_v2_canonical_content_sha256": source["canonical_content_sha256"],
                        "rotation0_reuse": rotation == 0,
                        "inference_required": rotation != 0,
                    }
                )

    semantic_path = root / "data/permutations/score_semantic_manifest.jsonl"
    inference_path = root / "data/permutations/score_inference_manifest.jsonl"
    generation_path = root / "data/generation/long_generation_manifest.jsonl"
    atomic_write_jsonl(semantic_path, score_rows)
    inference_rows = [row for row in score_rows if row["inference_required"]]
    atomic_write_jsonl(inference_path, inference_rows)
    generation_rows: list[dict[str, Any]] = []
    generation_conditions = set(config["generation_conditions"])
    condition_code = {condition: _condition_metadata(condition)[3] for condition in generation_conditions}
    generation_codes = set(condition_code.values())
    for row in score_rows:
        if row["base_question_id"] not in generation_ids or row["semantic_condition"] not in generation_codes:
            continue
        generation_rows.append(
            {
                **row,
                "schema_version": "pier_v23_long_generation_prompt_v1",
                "generation_prompt_id": f"generation__{row['prompt_id']}",
                "scoring_prompt_id": row["prompt_id"],
                "max_new_tokens": int(config["max_new_tokens"]),
                "do_sample": False,
            }
        )
    atomic_write_jsonl(generation_path, generation_rows)

    expected_all = sum(len(row["options"]) for row in selected) * 7
    expected_new = sum(len(row["options"]) - 1 for row in selected) * 7
    expected_generation = sum(len(row["options"]) for row in generation_subset) * 3
    if (len(score_rows), len(inference_rows), len(generation_rows)) != (
        expected_all,
        expected_new,
        expected_generation,
    ):
        raise AssertionError("Actual cyclic prompt counts differ from manifest-derived expectations")
    rotation0 = [row for row in score_rows if row["permutation_id"] == 0]
    for row in rotation0:
        if row["canonical_content_sha256"] != row["source_v2_canonical_content_sha256"]:
            raise ValueError(f"Rotation-0 canonical prompt differs from V2: {row['prompt_id']}")
    if {row["source_v2_prompt_id"] for row in rotation0} != expected_source_ids:
        raise AssertionError("Rotation-0 source prompt coverage is incomplete")

    result = {
        "schema_version": "pier_v23_prompt_manifest_status_v1",
        "created_at": utc_now(),
        "base_question_count": len(selected),
        "generation_question_count": len(generation_subset),
        "semantic_condition_count": 7,
        "generation_semantic_condition_count": 3,
        "option_count_distribution": dict(sorted(option_counts.items())),
        "score_all_rotation_rows_per_model": len(score_rows),
        "score_rotation0_reused_rows_per_model": len(rotation0),
        "score_new_inference_rows_per_model": len(inference_rows),
        "score_new_inference_rows_all_models": len(inference_rows) * 8,
        "generation_rows_per_model": len(generation_rows),
        "generation_rows_all_models": len(generation_rows) * 8,
        "score_manifest_sha256": sha256_file(semantic_path),
        "score_inference_manifest_sha256": sha256_file(inference_path),
        "generation_manifest_sha256": sha256_file(generation_path),
    }
    atomic_write_json(root / "status/prompt_manifest_status.json", result)
    return result


def _load_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    options: dict[str, Any] = {}
    if "mistral" in str(snapshot).casefold():
        options["fix_mistral_regex"] = True
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, **options
        )
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        tokenizer = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            **options,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer lacks both pad and EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _render_prompt(
    tokenizer: Any, content: str, frozen_system_message: str | None = None
) -> tuple[str, list[int]]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        if frozen_system_message is not None:
            messages.insert(0, {"role": "system", "content": frozen_system_message})
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = content
    if content not in rendered:
        raise ValueError("Canonical content is not byte-identical inside official chat wrapping")
    token_ids = [int(value) for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
    if not token_ids:
        raise ValueError("Rendered prompt tokenized to an empty sequence")
    return rendered, token_ids


def _candidate_manifest(tokenizer: Any, option_count: int) -> list[dict[str, Any]]:
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    rows: list[dict[str, Any]] = []
    for label in LABELS[:option_count]:
        continuation = f" ({label})"
        token_ids = [int(value) for value in tokenizer(continuation, add_special_tokens=False)["input_ids"]]
        decoded = tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not token_ids or special.intersection(token_ids) or decoded != continuation:
            raise ValueError(f"Candidate continuation round-trip failed: {continuation!r}")
        rows.append(
            {
                "label": label,
                "continuation": continuation,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "decoded": decoded,
            }
        )
    return rows


def _inventory_hash_map(root: Path) -> dict[str, tuple[int, str]]:
    inventory = load_json(root / "manifests/input_inventory_before.json")
    return {str(Path(row["physical_path"])): (int(row["size"]), str(row["sha256"])) for row in inventory}


def validate_and_render_models(root: Path) -> dict[str, Any]:
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    resolved = load_json(v2 / "configs/resolved_models.json")
    records = list(resolved["models"])
    if [row["id"] for row in records] != config["model_roster"]:
        raise ValueError("V2 model roster/order differs from the exact V2.3 roster")
    inventory = _inventory_hash_map(root)
    semantic_rows = list(read_jsonl(root / "data/permutations/score_semantic_manifest.jsonl"))
    rotation0_ids = {row["source_v2_prompt_id"]: row for row in semantic_rows if row["permutation_id"] == 0}
    option_counts = sorted({int(row["option_count"]) for row in semantic_rows})
    output_records: list[dict[str, Any]] = []
    for record in records:
        model_id = str(record["id"])
        snapshot = Path(record["local_cache_path"])
        if snapshot.name != record["revision"] or not snapshot.is_dir():
            raise FileNotFoundError(f"Exact cached revision unavailable for {model_id}")
        compatibility_path = Path(record["compatibility_manifest_path"])
        compatibility = load_json(compatibility_path)
        if compatibility["model_id"] != model_id or compatibility["revision"] != record["revision"]:
            raise ValueError(f"Compatibility manifest revision mismatch for {model_id}")
        for item in compatibility["file_inventory"]:
            path = (snapshot / item["path"]).resolve(strict=True)
            observed = inventory.get(str(path))
            if observed != (int(item["bytes"]), str(item["sha256"])):
                raise ValueError(f"Cached model inventory mismatch for {model_id}: {item['path']}")

        print(f"[render] {model_id}", flush=True)
        tokenizer = _load_tokenizer(snapshot)
        template_hash = sha256_text(str(getattr(tokenizer, "chat_template", "") or ""))
        if template_hash != compatibility["chat_template_sha256"]:
            raise ValueError(f"Chat-template hash mismatch for {model_id}")
        candidates = {str(count): _candidate_manifest(tokenizer, count) for count in option_counts}
        expected_candidates = compatibility["rendering"]["candidate_sequences_by_option_count"]
        for count in option_counts:
            if candidates[str(count)] != expected_candidates[str(count)]:
                raise ValueError(f"Candidate convention differs from V2 for {model_id}, m={count}")
        old_rendered = {
            row["prompt_id"]: row
            for row in read_jsonl(Path(compatibility["rendering"]["rendered_prompt_path"]))
            if row["prompt_id"] in rotation0_ids
        }
        if len(old_rendered) != len(rotation0_ids):
            raise ValueError(f"V2 rendered rotation-0 coverage incomplete for {model_id}")
        frozen_system_message: str | None = None
        if "strftime_now" in str(getattr(tokenizer, "chat_template", "")):
            sample = next(iter(old_rendered.values()))["rendered_prompt"]
            prefix = "<|start_of_role|>system<|end_of_role|>"
            suffix = "<|end_of_text|>"
            if prefix not in sample or suffix not in sample.split(prefix, 1)[1]:
                raise ValueError(f"Cannot freeze dynamic system date for {model_id}")
            frozen_system_message = sample.split(prefix, 1)[1].split(suffix, 1)[0]
        rendered_path = root / "data/permutations/rendered" / f"{slug(model_id)}.jsonl"
        semantic_by_id = {row["prompt_id"]: row for row in semantic_rows}
        rendered_rows: list[dict[str, Any]] = []
        maximum_tokens = 0
        model_limit = int(compatibility.get("model_max_length") or 0)
        if rendered_path.is_file():
            rendered_rows = list(read_jsonl(rendered_path))
            if len(rendered_rows) != len(semantic_rows):
                raise ValueError(f"Incomplete resumable rendered manifest for {model_id}")
            if len({row["prompt_id"] for row in rendered_rows}) != len(semantic_rows):
                raise ValueError(f"Duplicate resumable rendered prompts for {model_id}")
            for row in rendered_rows:
                prompt = semantic_by_id.get(row["prompt_id"])
                if prompt is None or row["canonical_content_sha256"] != prompt[
                    "canonical_content_sha256"
                ]:
                    raise ValueError(f"Resumable rendered prompt mismatch for {model_id}")
                maximum_tokens = max(maximum_tokens, int(row["prompt_token_count"]))
                if prompt["permutation_id"] == 0:
                    old = old_rendered[prompt["source_v2_prompt_id"]]
                    if (
                        row["rendered_prompt_sha256"] != old["rendered_prompt_sha256"]
                        or row["rendered_prompt"] != old["rendered_prompt"]
                    ):
                        raise ValueError(f"Resumable rotation-0 wrapper differs for {model_id}")
            print(f"[resume] validated rendered prompts for {model_id}", flush=True)
        else:
            for prompt in semantic_rows:
                rendered, token_ids = _render_prompt(
                    tokenizer, prompt["canonical_content"], frozen_system_message
                )
                maximum_tokens = max(maximum_tokens, len(token_ids))
                if model_limit and model_limit < 1_000_000 and len(token_ids) + 512 > model_limit:
                    raise ValueError(
                        f"Prompt plus generation exceeds model limit: {prompt['prompt_id']}"
                    )
                rendered_hash = sha256_text(rendered)
                if prompt["permutation_id"] == 0:
                    old = old_rendered[prompt["source_v2_prompt_id"]]
                    if (
                        rendered_hash != old["rendered_prompt_sha256"]
                        or rendered != old["rendered_prompt"]
                    ):
                        raise ValueError(
                            f"Rotation-0 official wrapping differs from V2: {model_id}"
                        )
                rendered_rows.append(
                    {
                        "schema_version": "pier_v23_rendered_prompt_v1",
                        "prompt_id": prompt["prompt_id"],
                        "canonical_content_sha256": prompt["canonical_content_sha256"],
                        "rendered_prompt": rendered,
                        "rendered_prompt_sha256": rendered_hash,
                        "prompt_token_count": len(token_ids),
                    }
                )
            atomic_write_jsonl(rendered_path, rendered_rows)
        output_records.append(
            {
                **record,
                "v23_rendered_prompt_path": str(rendered_path),
                "v23_rendered_prompt_sha256": sha256_file(rendered_path),
                "v23_rendered_prompt_count": len(rendered_rows),
                "v23_maximum_prompt_tokens": maximum_tokens,
                "v23_candidate_sequences_by_option_count": candidates,
                "v23_chat_template_sha256": template_hash,
                "v23_frozen_dynamic_system_message_sha256": (
                    sha256_text(frozen_system_message)
                    if frozen_system_message is not None
                    else None
                ),
                "tokenizer_revision": record["revision"],
            }
        )
        del tokenizer, old_rendered, rendered_rows
        gc.collect()
    output = {
        "schema_version": "pier_v23_resolved_models_v1",
        "created_at": utc_now(),
        "model_count": len(output_records),
        "models": output_records,
    }
    path = root / "configs/resolved_models_v23.json"
    atomic_write_json(path, output)
    status = {
        "passed": True,
        "model_count": len(output_records),
        "all_exact_revisions_cached": True,
        "all_chat_templates_matched_v2": True,
        "all_candidate_conventions_matched_v2": True,
        "all_rotation0_wrappers_matched_v2": True,
        "resolved_models_sha256": sha256_file(path),
        "models": [
            {
                "id": row["id"],
                "revision": row["revision"],
                "tokenizer_revision": row["tokenizer_revision"],
                "chat_template_sha256": row["v23_chat_template_sha256"],
                "rendered_prompt_sha256": row["v23_rendered_prompt_sha256"],
                "maximum_prompt_tokens": row["v23_maximum_prompt_tokens"],
            }
            for row in output_records
        ],
    }
    atomic_write_json(root / "status/model_cache_and_rendering.json", status)
    return status


def validate_rotation0_raw_scores(root: Path) -> dict[str, Any]:
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    models = load_json(root / "configs/resolved_models_v23.json")["models"]
    semantic = list(read_jsonl(root / "data/permutations/score_semantic_manifest.jsonl"))
    rotation0 = {row["source_v2_prompt_id"]: row for row in semantic if row["permutation_id"] == 0}
    inventory = _inventory_hash_map(root)
    validations: list[dict[str, Any]] = []
    for model in models:
        frames: list[pd.DataFrame] = []
        source_dir = v2 / "outputs/raw_scores" / slug(model["id"])
        for parquet in sorted(source_dir.glob("shard_*.parquet")):
            metadata = load_json(parquet.with_suffix(".meta.json"))
            observed = inventory.get(str(parquet.resolve(strict=True)))
            if observed is None or observed[1] != metadata["sha256"]:
                raise ValueError(f"V2 raw shard checksum mismatch: {parquet}")
            frame = pd.read_parquet(
                parquet,
                columns=[
                    "prompt_id",
                    "model_id",
                    "model_revision",
                    "candidate_continuations",
                    "probabilities",
                    "semantic_probabilities",
                    "presented_to_original",
                    "prompt_token_count",
                    "candidate_token_counts",
                ],
            )
            subset = frame[frame["prompt_id"].isin(rotation0)]
            if not subset.empty:
                frames.append(subset)
        if not frames:
            raise ValueError(f"No reusable V2 raw rows found for {model['id']}")
        found = pd.concat(frames, ignore_index=True)
        if len(found) != len(rotation0) or found["prompt_id"].duplicated().any():
            raise ValueError(f"Rotation-0 raw score coverage mismatch for {model['id']}")
        for row in found.itertuples(index=False):
            prompt = rotation0[str(row.prompt_id)]
            if (
                row.model_id != model["id"]
                or row.model_revision != model["revision"]
                or list(row.candidate_continuations) != prompt["candidate_continuations"]
                or list(row.presented_to_original) != list(range(prompt["option_count"]))
                or not np.allclose(row.probabilities, row.semantic_probabilities, atol=0.0, rtol=0.0)
                or not np.isclose(np.sum(row.probabilities), 1.0, atol=1e-12)
            ):
                raise ValueError(f"Rotation-0 raw compatibility failed: {model['id']} {row.prompt_id}")
        validations.append(
            {
                "model_id": model["id"],
                "revision": model["revision"],
                "reusable_row_count": len(found),
            }
        )
    status = {
        "passed": True,
        "reuse_policy": "validated exact V2 rotation 0",
        "rows_per_model": len(rotation0),
        "rows_all_models": len(rotation0) * len(models),
        "models": validations,
    }
    atomic_write_json(root / "status/rotation0_reuse_validation.json", status)
    return status


def prepare_all(root: Path) -> dict[str, Any]:
    prompts = build_manifests(root)
    models = validate_and_render_models(root)
    rotation0 = validate_rotation0_raw_scores(root)
    result = {"prompts": prompts, "models": models, "rotation0": rotation0, "passed": True}
    atomic_write_json(root / "status/prepared_assets.json", result)
    return result
