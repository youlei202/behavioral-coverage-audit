from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .core import LABELS, extract_final_label, remap_visible_to_semantic
from .prepare import _load_tokenizer
from .utils import (
    atomic_write_json,
    load_json,
    project_root,
    read_jsonl,
    sha256_file,
    slug,
    tree_sha256,
    utc_now,
)

SCORE_SCHEMA = "pier_v23_candidate_score_v1"
GENERATION_SCHEMA = "pier_v23_long_generation_v1"


def _load_model(record: dict[str, Any], device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    snapshot = Path(record["local_cache_path"])
    tokenizer = _load_tokenizer(snapshot)
    if record["model_adapter"] == "causal_lm":
        auto_class: Any = AutoModelForCausalLM
    elif record["model_adapter"] == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        auto_class = AutoModelForImageTextToText
    else:
        raise ValueError(f"Unsupported model adapter: {record['model_adapter']}")
    model = auto_class.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def _rendered_index(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(record["v23_rendered_prompt_path"])
    if sha256_file(path) != record["v23_rendered_prompt_sha256"]:
        raise ValueError(f"Rendered prompt checksum mismatch for {record['id']}")
    rows = {str(row["prompt_id"]): row for row in read_jsonl(path)}
    if len(rows) != int(record["v23_rendered_prompt_count"]):
        raise ValueError(f"Rendered prompt count mismatch for {record['id']}")
    return rows


def _candidate_token_ids(tokenizer: Any, labels: Sequence[str]) -> list[list[int]]:
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    result: list[list[int]] = []
    for label in labels:
        continuation = f" ({label})"
        ids = [int(value) for value in tokenizer(continuation, add_special_tokens=False)["input_ids"]]
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not ids or special.intersection(ids) or decoded != continuation:
            raise ValueError(f"Candidate tokenization invalid for {continuation!r}")
        result.append(ids)
    return result


def _padded_batch(sequences: list[list[int]], pad_token_id: int, device: str) -> tuple[Any, Any]:
    import torch

    maximum = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), maximum), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    for row_index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row_index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
    return input_ids, attention_mask


def score_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    rendered_by_id: dict[str, dict[str, Any]],
    *,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    sequences: list[list[int]] = []
    descriptors: list[tuple[int, int, list[int], int]] = []
    prompt_lengths: list[int] = []
    candidates_by_count: dict[int, list[list[int]]] = {}
    for prompt_index, prompt in enumerate(prompts):
        rendered_record = rendered_by_id[prompt["prompt_id"]]
        if rendered_record["canonical_content_sha256"] != prompt["canonical_content_sha256"]:
            raise ValueError(f"Rendered/canonical hash mismatch: {prompt['prompt_id']}")
        prompt_ids = [
            int(value)
            for value in tokenizer(rendered_record["rendered_prompt"], add_special_tokens=False)[
                "input_ids"
            ]
        ]
        if len(prompt_ids) != int(rendered_record["prompt_token_count"]):
            raise ValueError(f"Prompt token count drift: {prompt['prompt_id']}")
        prompt_lengths.append(len(prompt_ids))
        option_count = int(prompt["option_count"])
        candidates = candidates_by_count.setdefault(
            option_count, _candidate_token_ids(tokenizer, prompt["option_labels"])
        )
        for candidate_index, candidate_ids in enumerate(candidates):
            sequences.append(prompt_ids + candidate_ids)
            descriptors.append((prompt_index, candidate_index, candidate_ids, len(prompt_ids)))
    input_ids, attention_mask = _padded_batch(sequences, tokenizer.pad_token_id, device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        log_probabilities = torch.log_softmax(outputs.logits.float(), dim=-1)
    scores: list[list[float]] = [
        [math.nan] * int(prompt["option_count"]) for prompt in prompts
    ]
    candidate_lengths: list[list[int]] = [
        [0] * int(prompt["option_count"]) for prompt in prompts
    ]
    for sequence_index, descriptor in enumerate(descriptors):
        prompt_index, candidate_index, candidate_ids, prompt_length = descriptor
        score = 0.0
        for offset, token_id in enumerate(candidate_ids):
            score += float(log_probabilities[sequence_index, prompt_length + offset - 1, token_id])
        scores[prompt_index][candidate_index] = score
        candidate_lengths[prompt_index][candidate_index] = len(candidate_ids)
    del outputs, log_probabilities, input_ids, attention_mask

    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        log_likelihoods = np.asarray(scores[prompt_index], dtype=np.float64)
        shifted = np.exp(log_likelihoods - float(np.max(log_likelihoods)))
        visible = shifted / shifted.sum()
        semantic = remap_visible_to_semantic(visible, prompt["visible_to_semantic"])
        visible_argmax = int(np.argmax(visible))
        semantic_argmax = int(prompt["visible_to_semantic"][visible_argmax])
        rows.append(
            {
                "schema_version": SCORE_SCHEMA,
                "prompt_id": prompt["prompt_id"],
                "source_v2_prompt_id": prompt["source_v2_prompt_id"],
                "base_question_id": prompt["base_question_id"],
                "category": prompt["category"],
                "semantic_condition": prompt["semantic_condition"],
                "condition": prompt["condition"],
                "family": prompt["family"],
                "dose": float(prompt["dose"]),
                "track": prompt["track"],
                "permutation_id": int(prompt["permutation_id"]),
                "option_count": int(prompt["option_count"]),
                "semantic_to_visible": prompt["semantic_to_visible"],
                "visible_to_semantic": prompt["visible_to_semantic"],
                "candidate_continuations": prompt["candidate_continuations"],
                "candidate_sequence_log_likelihoods": log_likelihoods.tolist(),
                "visible_label_probabilities": visible.tolist(),
                "semantic_remapped_probabilities": semantic.tolist(),
                "argmax_visible_index": visible_argmax,
                "argmax_visible_label": prompt["option_labels"][visible_argmax],
                "argmax_semantic_option": semantic_argmax,
                "gold_semantic_option": int(prompt["semantic_answer_index"]),
                "gold_semantic_probability": float(semantic[int(prompt["semantic_answer_index"])]),
                "prompt_token_count": prompt_lengths[prompt_index],
                "candidate_token_counts": candidate_lengths[prompt_index],
                "canonical_prompt_hash": prompt["canonical_content_sha256"],
                "rendered_prompt_hash": rendered_by_id[prompt["prompt_id"]][
                    "rendered_prompt_sha256"
                ],
            }
        )
    return rows


def generate_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    rendered_by_id: dict[str, dict[str, Any]],
    *,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rendered = [rendered_by_id[row["scoring_prompt_id"]]["rendered_prompt"] for row in prompts]
    encoded = tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        tokenizer.padding_side = previous_padding_side
    eos_ids = getattr(tokenizer, "eos_token_id", None)
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) - {None}
    rows: list[dict[str, Any]] = []
    for row_index, prompt in enumerate(prompts):
        token_ids = [int(value) for value in generated[row_index, input_width:].detach().cpu().tolist()]
        eos_position = next((index for index, value in enumerate(token_ids) if value in eos_set), None)
        if eos_position is not None:
            effective_ids = token_ids[: eos_position + 1]
            eos_reached = True
        else:
            while token_ids and token_ids[-1] == tokenizer.pad_token_id:
                token_ids.pop()
            effective_ids = token_ids
            eos_reached = False
        text = tokenizer.decode(
            effective_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        visible_label, parser_rule = extract_final_label(text, prompt["option_labels"])
        visible_index = LABELS.index(visible_label) if visible_label is not None else None
        semantic_option = (
            int(prompt["visible_to_semantic"][visible_index]) if visible_index is not None else None
        )
        rows.append(
            {
                "schema_version": GENERATION_SCHEMA,
                "generation_prompt_id": prompt["generation_prompt_id"],
                "scoring_prompt_id": prompt["scoring_prompt_id"],
                "base_question_id": prompt["base_question_id"],
                "category": prompt["category"],
                "semantic_condition": prompt["semantic_condition"],
                "condition": prompt["condition"],
                "family": prompt["family"],
                "dose": float(prompt["dose"]),
                "track": prompt["track"],
                "permutation_id": int(prompt["permutation_id"]),
                "option_count": int(prompt["option_count"]),
                "semantic_to_visible": prompt["semantic_to_visible"],
                "visible_to_semantic": prompt["visible_to_semantic"],
                "full_generated_text": text,
                "visible_final_label": visible_label,
                "semantic_final_option": semantic_option,
                "parser_rule_used": parser_rule,
                "malformed": visible_label is None,
                "generation_token_count": len(effective_ids),
                "eos_reached": eos_reached,
                "max_new_tokens": 512,
                "do_sample": False,
                "gold_semantic_option": int(prompt["semantic_answer_index"]),
                "canonical_prompt_hash": prompt["canonical_content_sha256"],
                "rendered_prompt_hash": rendered_by_id[prompt["scoring_prompt_id"]][
                    "rendered_prompt_sha256"
                ],
            }
        )
    del generated, encoded
    return rows


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _completion_path(data_path: Path) -> Path:
    return data_path.with_suffix(".complete.json")


def validate_shard(
    data_path: Path,
    metadata_path: Path,
    *,
    schema: str,
    model_id: str,
    revision: str,
    manifest_sha256: str,
    expected_prompt_ids: list[str],
    prompt_key: str,
) -> bool:
    completion_path = _completion_path(data_path)
    try:
        metadata = load_json(metadata_path)
        completion = load_json(completion_path)
        expected = {
            "schema_version": schema,
            "model_id": model_id,
            "model_revision": revision,
            "manifest_sha256": manifest_sha256,
            "row_count": len(expected_prompt_ids),
            "prompt_range": {"start": expected_prompt_ids[0], "end": expected_prompt_ids[-1]},
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if completion.get("complete") is not True or completion.get("sha256") != metadata["sha256"]:
            return False
        if not data_path.is_file() or sha256_file(data_path) != metadata["sha256"]:
            return False
        frame = pd.read_parquet(data_path, columns=["schema_version", "model_id", prompt_key])
        return bool(
            len(frame) == len(expected_prompt_ids)
            and frame["schema_version"].eq(schema).all()
            and frame["model_id"].eq(model_id).all()
            and frame[prompt_key].tolist() == expected_prompt_ids
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _quarantine(root: Path, paths: Sequence[Path]) -> list[str]:
    quarantine = root / "outputs/quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace(":", "-")
    moved: list[str] = []
    for path in paths:
        if path.exists():
            destination = quarantine / f"{path.parent.name}--{path.name}--{stamp}"
            os.replace(path, destination)
            moved.append(str(destination))
    return moved


def write_shard(
    root: Path,
    output_dir: Path,
    shard_index: int,
    rows: list[dict[str, Any]],
    *,
    schema: str,
    record: dict[str, Any],
    manifest_sha256: str,
    prompt_key: str,
) -> tuple[Path, Path, Path]:
    data_path = output_dir / f"shard_{shard_index:05d}.parquet"
    metadata_path = output_dir / f"shard_{shard_index:05d}.meta.json"
    completion_path = _completion_path(data_path)
    prompt_ids = [str(row[prompt_key]) for row in rows]
    digest = _write_parquet_atomic(data_path, rows)
    metadata = {
        "schema_version": schema,
        "created_at": utc_now(),
        "model_id": record["id"],
        "model_revision": record["revision"],
        "manifest_sha256": manifest_sha256,
        "prompt_range": {"start": prompt_ids[0], "end": prompt_ids[-1]},
        "row_count": len(rows),
        "sha256": digest,
        "SHA256": digest,
        "completion_marker": str(completion_path),
        "data_path": str(data_path),
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(
        completion_path,
        {
            "schema_version": schema,
            "complete": True,
            "model_id": record["id"],
            "model_revision": record["revision"],
            "manifest_sha256": manifest_sha256,
            "prompt_range": metadata["prompt_range"],
            "row_count": len(rows),
            "sha256": digest,
            "completed_at": utc_now(),
        },
    )
    if not validate_shard(
        data_path,
        metadata_path,
        schema=schema,
        model_id=record["id"],
        revision=record["revision"],
        manifest_sha256=manifest_sha256,
        expected_prompt_ids=prompt_ids,
        prompt_key=prompt_key,
    ):
        _quarantine(root, [data_path, metadata_path, completion_path])
        raise RuntimeError(f"New shard failed validation: {data_path}")
    return data_path, metadata_path, completion_path


def _batched_with_oom_backoff(
    function: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    prompts: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    model_id: str,
    phase: str,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    import torch

    output: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    offset = 0
    while offset < len(prompts):
        current = prompts[offset : offset + batch_size]
        try:
            output.extend(function(current))
            offset += len(current)
        except torch.OutOfMemoryError:
            if not device.startswith("cuda") or batch_size == 1:
                raise
            previous = batch_size
            batch_size = max(1, batch_size // 2)
            event = {
                "timestamp": utc_now(),
                "model_id": model_id,
                "phase": phase,
                "offset": offset,
                "previous_batch_size": previous,
                "new_batch_size": batch_size,
            }
            events.append(event)
            print(json.dumps({"event": "cuda_oom_automatic_backoff", **event}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    return output, batch_size, events


def run_model_worker(root_text: str, record: dict[str, Any], gpu_index: int, resume: bool) -> dict[str, Any]:
    import torch

    root = Path(root_text)
    device = f"cuda:{gpu_index}" if gpu_index >= 0 else "cpu"
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)
    config = load_json(root / "configs/experiment_v23.json")
    score_path = root / "data/permutations/score_inference_manifest.jsonl"
    generation_path = root / "data/generation/long_generation_manifest.jsonl"
    score_prompts = list(read_jsonl(score_path))
    generation_prompts = list(read_jsonl(generation_path))
    score_sha = sha256_file(score_path)
    generation_sha = sha256_file(generation_path)
    rendered = _rendered_index(record)
    required = {row["prompt_id"] for row in score_prompts}
    required.update(row["scoring_prompt_id"] for row in generation_prompts)
    if missing := required.difference(rendered):
        raise ValueError(f"{record['id']} lacks {len(missing)} rendered prompts")
    score_dir = root / "outputs/raw_scores" / slug(record["id"])
    generation_dir = root / "outputs/generation" / slug(record["id"])
    score_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    score_shard_size = int(config["score_shard_size"])
    generation_shard_size = int(config["generation_shard_size"])
    scoring_batch = int(record["batch_size"])
    generation_batch = int(record["batch_size"])
    skipped_score = skipped_generation = 0
    quarantine_events: list[str] = []
    backoffs: list[dict[str, Any]] = []
    model = tokenizer = None

    def ensure_loaded() -> tuple[Any, Any]:
        nonlocal model, tokenizer
        if model is None:
            model, tokenizer = _load_model(record, device)
        return model, tokenizer

    try:
        for shard_index, start in enumerate(range(0, len(score_prompts), score_shard_size)):
            prompts = score_prompts[start : start + score_shard_size]
            prompt_ids = [row["prompt_id"] for row in prompts]
            data_path = score_dir / f"shard_{shard_index:05d}.parquet"
            meta_path = score_dir / f"shard_{shard_index:05d}.meta.json"
            valid = validate_shard(
                data_path,
                meta_path,
                schema=SCORE_SCHEMA,
                model_id=record["id"],
                revision=record["revision"],
                manifest_sha256=score_sha,
                expected_prompt_ids=prompt_ids,
                prompt_key="prompt_id",
            )
            if resume and valid:
                skipped_score += len(prompts)
                continue
            quarantine_events.extend(_quarantine(root, [data_path, meta_path, _completion_path(data_path)]))
            loaded_model, loaded_tokenizer = ensure_loaded()
            rows, scoring_batch, events = _batched_with_oom_backoff(
                lambda batch: score_prompt_batch(
                    loaded_model, loaded_tokenizer, batch, rendered, device=device
                ),
                prompts,
                batch_size=scoring_batch,
                device=device,
                model_id=record["id"],
                phase="candidate_scoring",
            )
            backoffs.extend(events)
            for row in rows:
                row["model_id"] = record["id"]
                row["model_revision"] = record["revision"]
                row["manifest_sha256"] = score_sha
            write_shard(
                root,
                score_dir,
                shard_index,
                rows,
                schema=SCORE_SCHEMA,
                record=record,
                manifest_sha256=score_sha,
                prompt_key="prompt_id",
            )

        for shard_index, start in enumerate(
            range(0, len(generation_prompts), generation_shard_size)
        ):
            prompts = generation_prompts[start : start + generation_shard_size]
            prompt_ids = [row["generation_prompt_id"] for row in prompts]
            data_path = generation_dir / f"shard_{shard_index:05d}.parquet"
            meta_path = generation_dir / f"shard_{shard_index:05d}.meta.json"
            valid = validate_shard(
                data_path,
                meta_path,
                schema=GENERATION_SCHEMA,
                model_id=record["id"],
                revision=record["revision"],
                manifest_sha256=generation_sha,
                expected_prompt_ids=prompt_ids,
                prompt_key="generation_prompt_id",
            )
            if resume and valid:
                skipped_generation += len(prompts)
                continue
            quarantine_events.extend(_quarantine(root, [data_path, meta_path, _completion_path(data_path)]))
            loaded_model, loaded_tokenizer = ensure_loaded()
            rows, generation_batch, events = _batched_with_oom_backoff(
                lambda batch: generate_prompt_batch(
                    loaded_model, loaded_tokenizer, batch, rendered, device=device
                ),
                prompts,
                batch_size=generation_batch,
                device=device,
                model_id=record["id"],
                phase="long_generation",
            )
            backoffs.extend(events)
            for row in rows:
                row["model_id"] = record["id"]
                row["model_revision"] = record["revision"]
                row["manifest_sha256"] = generation_sha
            write_shard(
                root,
                generation_dir,
                shard_index,
                rows,
                schema=GENERATION_SCHEMA,
                record=record,
                manifest_sha256=generation_sha,
                prompt_key="generation_prompt_id",
            )
    finally:
        del model, tokenizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "model_id": record["id"],
        "model_revision": record["revision"],
        "gpu_index": gpu_index,
        "device": device,
        "score_rows": len(score_prompts),
        "generation_rows": len(generation_prompts),
        "skipped_score_rows": skipped_score,
        "skipped_generation_rows": skipped_generation,
        "effective_scoring_batch_size": scoring_batch,
        "effective_generation_batch_size": generation_batch,
        "automatic_oom_backoffs": backoffs,
        "quarantined_paths": quarantine_events,
        "completed_at": utc_now(),
    }


def validate_all_outputs(root: Path, models: list[dict[str, Any]]) -> dict[str, Any]:
    config = load_json(root / "configs/experiment_v23.json")
    score_prompts = list(read_jsonl(root / "data/permutations/score_inference_manifest.jsonl"))
    generation_prompts = list(read_jsonl(root / "data/generation/long_generation_manifest.jsonl"))
    score_sha = sha256_file(root / "data/permutations/score_inference_manifest.jsonl")
    generation_sha = sha256_file(root / "data/generation/long_generation_manifest.jsonl")
    checks: list[dict[str, Any]] = []
    all_passed = True
    for record in models:
        model_checks: list[bool] = []
        for phase, prompts, shard_size, output_dir, schema, key, digest in (
            (
                "score",
                score_prompts,
                int(config["score_shard_size"]),
                root / "outputs/raw_scores" / slug(record["id"]),
                SCORE_SCHEMA,
                "prompt_id",
                score_sha,
            ),
            (
                "generation",
                generation_prompts,
                int(config["generation_shard_size"]),
                root / "outputs/generation" / slug(record["id"]),
                GENERATION_SCHEMA,
                "generation_prompt_id",
                generation_sha,
            ),
        ):
            for shard_index, start in enumerate(range(0, len(prompts), shard_size)):
                expected = [row[key] for row in prompts[start : start + shard_size]]
                data_path = output_dir / f"shard_{shard_index:05d}.parquet"
                passed = validate_shard(
                    data_path,
                    output_dir / f"shard_{shard_index:05d}.meta.json",
                    schema=schema,
                    model_id=record["id"],
                    revision=record["revision"],
                    manifest_sha256=digest,
                    expected_prompt_ids=expected,
                    prompt_key=key,
                )
                model_checks.append(passed)
                if not passed:
                    checks.append(
                        {"model_id": record["id"], "phase": phase, "shard": shard_index, "passed": False}
                    )
        passed = all(model_checks) and bool(model_checks)
        all_passed &= passed
        checks.append(
            {
                "model_id": record["id"],
                "score_rows": len(score_prompts),
                "generation_rows": len(generation_prompts),
                "passed": passed,
            }
        )
    return {"passed": all_passed, "models": checks}


def _runtime_manifest(root: Path, gpu_names: list[str]) -> dict[str, Any]:
    import torch

    packages = ["torch", "transformers", "accelerate", "numpy", "pandas", "pyarrow"]
    return {
        "created_at": utc_now(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "torch_cuda_runtime": torch.version.cuda,
        "driver_version": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "gpu_models": gpu_names,
        "gpu_count": len(gpu_names),
        "dtype": "bfloat16",
    }


def run_formal(root: Path, resume: bool) -> dict[str, Any]:
    import torch

    ready_path = root / "status/B200_READY_STAGE2.json"
    if not ready_path.is_file() or load_json(ready_path).get("ready") is not True:
        raise RuntimeError("B200_READY_STAGE2.json is absent or not ready")
    ready = load_json(ready_path)
    if tree_sha256(root / "src") != ready["source_tree_sha256"]:
        raise RuntimeError("V2.3 source tree changed after CPU prestage")
    if sha256_file(root / "scripts/run_b200_interface_v23.sh") != ready[
        "b200_launcher_sha256"
    ]:
        raise RuntimeError("Formal B200 launcher changed after CPU prestage")
    if sha256_file(root / "scripts/run_cpu_postprocess_v23.sh") != ready[
        "cpu_postprocess_launcher_sha256"
    ]:
        raise RuntimeError("Formal CPU postprocess launcher changed after CPU prestage")
    score_path = root / "data/permutations/score_inference_manifest.jsonl"
    generation_path = root / "data/generation/long_generation_manifest.jsonl"
    if sha256_file(score_path) != ready["score_inference_manifest_sha256"]:
        raise RuntimeError("Score inference manifest differs from B200 readiness marker")
    if sha256_file(generation_path) != ready["generation_manifest_sha256"]:
        raise RuntimeError("Generation manifest differs from B200 readiness marker")
    models = load_json(root / "configs/resolved_models_v23.json")["models"]
    if len(models) != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError(f"Formal run requires exactly eight models and GPUs; GPUs={torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(8)]
    if any("B200" not in name.upper() for name in gpu_names):
        raise RuntimeError(f"Every GPU must be an NVIDIA B200: {gpu_names}")
    atomic_write_json(root / "manifests/b200_environment.json", _runtime_manifest(root, gpu_names))
    failures: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=8, mp_context=context) as executor:
        futures = {
            executor.submit(run_model_worker, str(root), record, gpu_index, resume): record
            for gpu_index, record in enumerate(models)
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                workers.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "model_id": record["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    validation = validate_all_outputs(root, models) if not failures else {"passed": False, "models": []}
    status = {
        "schema_version": "pier_v23_b200_stage2_complete_v1",
        "completed_at": utc_now(),
        "complete": not failures and validation["passed"],
        "model_count": len(models),
        "gpu_count": 8,
        "gpu_models": gpu_names,
        "workers": sorted(workers, key=lambda row: row["gpu_index"]),
        "failures": failures,
        "validation": validation,
        "ready_marker_sha256": sha256_file(ready_path),
        "source_prompt_hashes_match_ready": True,
        "score_manifest_sha256": sha256_file(score_path),
        "generation_manifest_sha256": sha256_file(generation_path),
        "persistent_root": str(root.resolve()),
    }
    atomic_write_json(root / "status/b200_runtime_status.json", status)
    if not status["complete"]:
        raise RuntimeError(f"B200 inference incomplete; failures={len(failures)}")
    atomic_write_json(root / "status/B200_STAGE2_COMPLETE.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-model-record", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.cpu_model_record:
        result = run_model_worker(str(root), load_json(args.cpu_model_record), -1, args.resume)
    else:
        result = run_formal(root, args.resume)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not args.cpu_model_record:
        print("========================================================", flush=True)
        print("B200 STAGE-2 INTERFACE INFERENCE COMPLETE", flush=True)
        print("", flush=True)
        print("Shut down the 8×B200 host.", flush=True)
        print("", flush=True)
        print("Return to the 64-core CPU machine and run:", flush=True)
        print("", flush=True)
        print("tmux new -d -s pier_llm_v23_post \\", flush=True)
        print('  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION && \\', flush=True)
        print("   bash scripts/run_cpu_postprocess_v23.sh --resume \\", flush=True)
        print('   2>&1 | tee -a logs/cpu_postprocess_v23.log"', flush=True)
        print("========================================================", flush=True)


if __name__ == "__main__":
    main()
