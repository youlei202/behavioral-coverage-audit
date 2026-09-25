from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import re
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_prep import LABELS
from .utils import (
    atomic_write_json,
    project_root,
    read_jsonl,
    sha256_file,
    slug,
    utc_now,
)

SCORE_SCHEMA = "pier_raw_label_scores_v2"
GENERATION_SCHEMA = "pier_generation_validation_v2"
VALID_LABEL_RE = re.compile(r"(?i)(?:\(|\b)([A-J])(?:\)|\b)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_token_ids(tokenizer: Any, option_count: int) -> list[list[int]]:
    special = set(int(value) for value in getattr(tokenizer, "all_special_ids", []))
    result: list[list[int]] = []
    for label in LABELS[:option_count]:
        suffix = f" ({label})"
        ids = [int(value) for value in tokenizer(suffix, add_special_tokens=False)["input_ids"]]
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not ids or special.intersection(ids) or decoded != suffix:
            raise ValueError(
                f"Invalid candidate tokenization for {suffix!r}: ids={ids}, decoded={decoded!r}"
            )
        result.append(ids)
    return result


def _load_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_options: dict[str, Any] = {}
    if "mistral" in str(snapshot).casefold():
        tokenizer_options["fix_mistral_regex"] = True
    try:
        processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            **tokenizer_options,
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
            **tokenizer_options,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(record: dict[str, Any], device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    snapshot = Path(record["local_cache_path"])
    tokenizer = _load_tokenizer(snapshot)
    auto_class: Any
    if record["model_adapter"] == "causal_lm":
        auto_class = AutoModelForCausalLM
    elif record["model_adapter"] == "image_text_to_text":
        from transformers import AutoModelForImageTextToText

        auto_class = AutoModelForImageTextToText
    else:
        raise ValueError(f"Unsupported model adapter {record['model_adapter']!r}")
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


def _padded_batch(sequences: list[list[int]], pad_token_id: int, device: str) -> tuple[Any, Any]:
    import torch

    maximum = max(len(row) for row in sequences)
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
    rendered_by_id: dict[str, str],
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Score complete label continuations for a batch of distinct prompts."""
    import torch

    sequences: list[list[int]] = []
    descriptors: list[tuple[int, int, list[int], int]] = []
    prompt_lengths: list[int] = []
    candidates_by_count: dict[int, list[list[int]]] = {}
    for prompt_index, prompt in enumerate(prompts):
        rendered = rendered_by_id[prompt["prompt_id"]]
        prompt_ids = [
            int(value)
            for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
        ]
        if not prompt_ids:
            raise ValueError(f"Empty rendered prompt {prompt['prompt_id']}")
        prompt_lengths.append(len(prompt_ids))
        option_count = len(prompt["option_labels"])
        candidates = candidates_by_count.setdefault(
            option_count, _candidate_token_ids(tokenizer, option_count)
        )
        for candidate_index, candidate_ids in enumerate(candidates):
            sequences.append(prompt_ids + candidate_ids)
            descriptors.append((prompt_index, candidate_index, candidate_ids, len(prompt_ids)))
    input_ids, attention_mask = _padded_batch(sequences, tokenizer.pad_token_id, device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        log_probabilities = torch.log_softmax(outputs.logits.float(), dim=-1)
    scores: list[list[float]] = [
        [math.nan] * len(prompt["option_labels"]) for prompt in prompts
    ]
    candidate_lengths: list[list[int]] = [
        [0] * len(prompt["option_labels"]) for prompt in prompts
    ]
    for sequence_index, (prompt_index, candidate_index, candidate_ids, prompt_length) in enumerate(
        descriptors
    ):
        score = 0.0
        for offset, token_id in enumerate(candidate_ids):
            score += float(log_probabilities[sequence_index, prompt_length + offset - 1, token_id])
        scores[prompt_index][candidate_index] = score
        candidate_lengths[prompt_index][candidate_index] = len(candidate_ids)
    del outputs, log_probabilities, input_ids, attention_mask

    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        log_likelihoods = np.asarray(scores[prompt_index], dtype=np.float64)
        shifted = np.exp(log_likelihoods - np.max(log_likelihoods))
        probabilities = shifted / shifted.sum()
        presented_to_original = [int(value) for value in prompt["presented_to_original"]]
        semantic = np.empty_like(probabilities)
        for presented_index, original_index in enumerate(presented_to_original):
            semantic[original_index] = probabilities[presented_index]
        answer_index = int(prompt["answer_index"])
        wrong = np.delete(log_likelihoods, answer_index)
        argmax_index = int(np.argmax(probabilities))
        semantic_argmax = int(presented_to_original[argmax_index])
        rows.append(
            {
                "schema_version": SCORE_SCHEMA,
                "prompt_id": prompt["prompt_id"],
                "base_question_id": prompt["base_question_id"],
                "category": prompt["category"],
                "condition": prompt["condition"],
                "family": prompt["family"],
                "dose": float(prompt["dose"]),
                "track": prompt["track"],
                "permutation_index": prompt.get("permutation_index"),
                "option_labels": prompt["option_labels"],
                "candidate_continuations": [
                    f" ({label})" for label in prompt["option_labels"]
                ],
                "candidate_log_likelihoods": log_likelihoods.tolist(),
                "probabilities": probabilities.tolist(),
                "semantic_probabilities": semantic.tolist(),
                "presented_to_original": presented_to_original,
                "argmax_index": argmax_index,
                "argmax_label": prompt["option_labels"][argmax_index],
                "semantic_argmax_index": semantic_argmax,
                "answer_index": answer_index,
                "semantic_answer_index": int(prompt["semantic_answer_index"]),
                "correct": argmax_index == answer_index,
                "gold_probability": float(probabilities[answer_index]),
                "gold_vs_best_wrong_margin": float(log_likelihoods[answer_index] - wrong.max()),
                "prompt_token_count": prompt_lengths[prompt_index],
                "candidate_token_counts": candidate_lengths[prompt_index],
            }
        )
    return rows


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _validate_shard(
    data_path: Path,
    metadata_path: Path,
    *,
    schema: str,
    model_id: str,
    revision: str,
    prompt_manifest_sha256: str,
    expected_prompt_ids: list[str],
) -> bool:
    try:
        metadata = _load_json(metadata_path)
        if metadata["schema_version"] != schema:
            return False
        if metadata["model_id"] != model_id or metadata["model_revision"] != revision:
            return False
        if metadata["prompt_manifest_sha256"] != prompt_manifest_sha256:
            return False
        if metadata["row_count"] != len(expected_prompt_ids):
            return False
        if metadata["first_prompt_id"] != expected_prompt_ids[0]:
            return False
        if metadata["last_prompt_id"] != expected_prompt_ids[-1]:
            return False
        if not data_path.is_file() or sha256_file(data_path) != metadata["sha256"]:
            return False
        prompt_column = "generation_prompt_id" if schema == GENERATION_SCHEMA else "prompt_id"
        frame = pd.read_parquet(
            data_path, columns=["schema_version", "model_id", prompt_column]
        )
        return (
            len(frame) == len(expected_prompt_ids)
            and frame["schema_version"].eq(schema).all()
            and frame["model_id"].eq(model_id).all()
            and frame[prompt_column].tolist() == expected_prompt_ids
        )
    except Exception:
        return False


def _quarantine(root: Path, *paths: Path) -> None:
    quarantine = root / "outputs" / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace(":", "-")
    for path in paths:
        if path.exists():
            destination = quarantine / f"{path.parent.name}--{path.name}--{stamp}"
            os.replace(path, destination)


def _write_shard(
    root: Path,
    output_dir: Path,
    shard_index: int,
    rows: list[dict[str, Any]],
    *,
    schema: str,
    record: dict[str, Any],
    prompt_manifest_sha256: str,
    prompt_key: str,
) -> tuple[Path, Path]:
    data_path = output_dir / f"shard_{shard_index:05d}.parquet"
    metadata_path = output_dir / f"shard_{shard_index:05d}.meta.json"
    prompt_ids = [str(row[prompt_key]) for row in rows]
    digest = _write_parquet_atomic(data_path, rows)
    metadata = {
        "schema_version": schema,
        "timestamp": utc_now(),
        "model_id": record["id"],
        "model_revision": record["revision"],
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "first_prompt_id": prompt_ids[0],
        "last_prompt_id": prompt_ids[-1],
        "row_count": len(rows),
        "sha256": digest,
        "data_path": str(data_path),
    }
    atomic_write_json(metadata_path, metadata)
    if not _validate_shard(
        data_path,
        metadata_path,
        schema=schema,
        model_id=record["id"],
        revision=record["revision"],
        prompt_manifest_sha256=prompt_manifest_sha256,
        expected_prompt_ids=prompt_ids,
    ):
        _quarantine(root, data_path, metadata_path)
        raise RuntimeError(f"New shard failed validation: {data_path}")
    return data_path, metadata_path


def _rendered_index(record: dict[str, Any]) -> dict[str, str]:
    manifest = _load_json(Path(record["compatibility_manifest_path"]))
    path = Path(manifest["rendering"]["rendered_prompt_path"])
    if sha256_file(path) != manifest["rendering"]["rendered_prompt_sha256"]:
        raise ValueError(f"Rendered prompt hash mismatch for {record['id']}")
    return {row["prompt_id"]: row["rendered_prompt"] for row in read_jsonl(path)}


def _extract_generated_label(text: str, valid_labels: list[str]) -> str | None:
    valid = set(valid_labels)
    for match in VALID_LABEL_RE.finditer(text):
        label = match.group(1).upper()
        if label in valid:
            return label
    return None


def generate_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    rendered_by_id: dict[str, str],
    *,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rendered = [rendered_by_id[row["scoring_prompt_id"]] for row in prompts]
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=16,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    texts = tokenizer.batch_decode(
        generated[:, input_width:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.padding_side = previous_padding_side
    rows: list[dict[str, Any]] = []
    for prompt, text in zip(prompts, texts, strict=True):
        extracted = _extract_generated_label(text, prompt["option_labels"])
        gold_label = LABELS[int(prompt["answer_index"])]
        rows.append(
            {
                "schema_version": GENERATION_SCHEMA,
                "generation_prompt_id": prompt["generation_prompt_id"],
                "scoring_prompt_id": prompt["scoring_prompt_id"],
                "base_question_id": prompt["base_question_id"],
                "category": prompt["category"],
                "condition": prompt["condition"],
                "generated_text": text,
                "extracted_label": extracted,
                "gold_label": gold_label,
                "malformed": extracted is None,
                "generated_correct": extracted == gold_label,
                "max_new_tokens": 16,
                "do_sample": False,
            }
        )
    return rows


def _run_batched_with_oom_backoff(
    function: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    prompts: list[dict[str, Any]],
    *,
    batch_size: int,
    device: str,
    model_id: str,
    phase: str,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Run a shard in batches, reducing only CUDA batches that exceed device memory."""
    import torch

    if batch_size < 1:
        raise ValueError(f"Invalid batch size {batch_size} for {model_id}")
    rows: list[dict[str, Any]] = []
    backoffs: list[dict[str, Any]] = []
    offset = 0
    while offset < len(prompts):
        current = prompts[offset : offset + batch_size]
        retry_after_oom = False
        try:
            rows.extend(function(current))
        except torch.OutOfMemoryError:
            if not device.startswith("cuda") or batch_size == 1:
                raise
            previous_batch_size = batch_size
            batch_size = max(1, batch_size // 2)
            event = {
                "timestamp": utc_now(),
                "model_id": model_id,
                "phase": phase,
                "offset": offset,
                "previous_batch_size": previous_batch_size,
                "new_batch_size": batch_size,
            }
            backoffs.append(event)
            print(json.dumps({"event": "cuda_oom_batch_backoff", **event}), flush=True)
            retry_after_oom = True
        if retry_after_oom:
            gc.collect()
            torch.cuda.empty_cache()
            continue
        offset += len(current)
    return rows, batch_size, backoffs


def run_model_worker(root: Path, record: dict[str, Any], device: str, resume: bool) -> dict[str, Any]:
    import torch

    scoring_path = root / "data" / "manifests" / "scoring_prompts.jsonl"
    generation_path = root / "data" / "manifests" / "generation_prompts.jsonl"
    scoring_prompts = list(read_jsonl(scoring_path))
    generation_prompts = list(read_jsonl(generation_path))
    prompt_sha = sha256_file(scoring_path)
    generation_sha = sha256_file(generation_path)
    rendered = _rendered_index(record)
    missing = {row["prompt_id"] for row in scoring_prompts}.difference(rendered)
    if missing:
        raise ValueError(f"Missing {len(missing)} rendered prompts for {record['id']}")
    output_dir = root / "outputs" / "raw_scores" / slug(record["id"])
    generation_dir = root / "outputs" / "generation_validation" / slug(record["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(_load_json(root / "configs" / "experiment.json")["shard_size"])
    model = tokenizer = None
    skipped_score = 0
    skipped_generation = 0
    scoring_batch_size = int(record["batch_size"])
    generation_batch_size = int(record["batch_size"])
    batch_backoffs: list[dict[str, Any]] = []

    def ensure_loaded() -> tuple[Any, Any]:
        nonlocal model, tokenizer
        if model is None or tokenizer is None:
            model, tokenizer = _load_model(record, device)
        return model, tokenizer

    try:
        for shard_index, start in enumerate(range(0, len(scoring_prompts), shard_size)):
            batch_prompts = scoring_prompts[start : start + shard_size]
            expected = [row["prompt_id"] for row in batch_prompts]
            data_path = output_dir / f"shard_{shard_index:05d}.parquet"
            meta_path = output_dir / f"shard_{shard_index:05d}.meta.json"
            valid = _validate_shard(
                data_path,
                meta_path,
                schema=SCORE_SCHEMA,
                model_id=record["id"],
                revision=record["revision"],
                prompt_manifest_sha256=prompt_sha,
                expected_prompt_ids=expected,
            )
            if resume and valid:
                skipped_score += len(batch_prompts)
                continue
            if data_path.exists() or meta_path.exists():
                _quarantine(root, data_path, meta_path)
            loaded_model, loaded_tokenizer = ensure_loaded()
            rows, scoring_batch_size, backoffs = _run_batched_with_oom_backoff(
                lambda current, model=loaded_model, tokenizer=loaded_tokenizer: score_prompt_batch(
                    model,
                    tokenizer,
                    current,
                    rendered,
                    device=device,
                ),
                batch_prompts,
                batch_size=scoring_batch_size,
                device=device,
                model_id=record["id"],
                phase="scoring",
            )
            batch_backoffs.extend(backoffs)
            for row in rows:
                row["model_id"] = record["id"]
                row["model_revision"] = record["revision"]
                row["prompt_manifest_sha256"] = prompt_sha
            _write_shard(
                root,
                output_dir,
                shard_index,
                rows,
                schema=SCORE_SCHEMA,
                record=record,
                prompt_manifest_sha256=prompt_sha,
                prompt_key="prompt_id",
            )

        for shard_index, start in enumerate(range(0, len(generation_prompts), shard_size)):
            batch_prompts = generation_prompts[start : start + shard_size]
            expected = [row["generation_prompt_id"] for row in batch_prompts]
            data_path = generation_dir / f"shard_{shard_index:05d}.parquet"
            meta_path = generation_dir / f"shard_{shard_index:05d}.meta.json"
            valid = _validate_shard(
                data_path,
                meta_path,
                schema=GENERATION_SCHEMA,
                model_id=record["id"],
                revision=record["revision"],
                prompt_manifest_sha256=generation_sha,
                expected_prompt_ids=expected,
            )
            if resume and valid:
                skipped_generation += len(batch_prompts)
                continue
            if data_path.exists() or meta_path.exists():
                _quarantine(root, data_path, meta_path)
            loaded_model, loaded_tokenizer = ensure_loaded()
            rows, generation_batch_size, backoffs = _run_batched_with_oom_backoff(
                lambda current, model=loaded_model, tokenizer=loaded_tokenizer: generate_prompt_batch(
                    model,
                    tokenizer,
                    current,
                    rendered,
                    device=device,
                ),
                batch_prompts,
                batch_size=generation_batch_size,
                device=device,
                model_id=record["id"],
                phase="generation",
            )
            batch_backoffs.extend(backoffs)
            for row in rows:
                row["model_id"] = record["id"]
                row["model_revision"] = record["revision"]
                row["prompt_manifest_sha256"] = generation_sha
            _write_shard(
                root,
                generation_dir,
                shard_index,
                rows,
                schema=GENERATION_SCHEMA,
                record=record,
                prompt_manifest_sha256=generation_sha,
                prompt_key="generation_prompt_id",
            )
    finally:
        del model, tokenizer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "model_id": record["id"],
        "device": device,
        "scoring_rows": len(scoring_prompts),
        "generation_rows": len(generation_prompts),
        "skipped_scoring_rows": skipped_score,
        "skipped_generation_rows": skipped_generation,
        "effective_scoring_batch_size": scoring_batch_size,
        "effective_generation_batch_size": generation_batch_size,
        "batch_backoffs": batch_backoffs,
    }


def _gpu_queue(
    root_text: str,
    records: list[dict[str, Any]],
    gpu_index: int,
    resume: bool,
    result_queue: Any,
) -> None:
    root = Path(root_text)
    for record in records:
        try:
            result_queue.put(
                {"ok": True, "result": run_model_worker(root, record, f"cuda:{gpu_index}", resume)}
            )
        except Exception as exc:
            result_queue.put(
                {
                    "ok": False,
                    "model_id": record["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            return


def _validate_all_outputs(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    config = _load_json(root / "configs" / "experiment.json")
    expected_score = int(config["scoring_prompts_per_model"])
    expected_generation = int(config["generation_prompts_per_model"])
    results: list[dict[str, Any]] = []
    all_passed = True
    for record in records:
        score_files = sorted((root / "outputs" / "raw_scores" / slug(record["id"])).glob("shard_*.parquet"))
        generation_files = sorted(
            (root / "outputs" / "generation_validation" / slug(record["id"])).glob(
                "shard_*.parquet"
            )
        )
        score_count = sum(len(pd.read_parquet(path, columns=["prompt_id"])) for path in score_files)
        generation_count = sum(
            len(pd.read_parquet(path, columns=["generation_prompt_id"]))
            for path in generation_files
        )
        passed = score_count == expected_score and generation_count == expected_generation
        all_passed &= passed
        results.append(
            {
                "model_id": record["id"],
                "scoring_rows": score_count,
                "generation_rows": generation_count,
                "passed": passed,
            }
        )
    return {"passed": all_passed, "models": results}


def run_formal(root: Path, resume: bool) -> dict[str, Any]:
    import torch

    ready_path = root / "status" / "B200_READY.json"
    if not ready_path.exists() or not _load_json(ready_path).get("ready"):
        raise RuntimeError("B200_READY.json is absent or not ready; formal inference is forbidden")
    ready = _load_json(ready_path)
    prompt_path = root / "data" / "manifests" / "scoring_prompts.jsonl"
    if sha256_file(prompt_path) != ready["prompt_manifest_sha256"]:
        raise RuntimeError("Prompt manifest does not match B200 readiness marker")
    resolved = _load_json(root / "configs" / "resolved_models.json")
    records = list(resolved["models"])
    gpu_count = torch.cuda.device_count()
    if gpu_count < len(records):
        raise RuntimeError(f"Need at least {len(records)} visible GPUs, found {gpu_count}")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    assignments = [records[index::gpu_count] for index in range(gpu_count)]
    processes = [
        context.Process(
            target=_gpu_queue,
            args=(str(root), assignment, gpu_index, resume, result_queue),
            name=f"pier-gpu-{gpu_index}",
        )
        for gpu_index, assignment in enumerate(assignments)
        if assignment
    ]
    for process in processes:
        process.start()
    messages = [result_queue.get() for _ in records]
    for process in processes:
        process.join()
    failures = [message for message in messages if not message["ok"]]
    validation = _validate_all_outputs(root, records) if not failures else {"passed": False}
    status = {
        "schema_version": "pier_b200_inference_status_v2",
        "timestamp": utc_now(),
        "complete": not failures and validation["passed"],
        "workers": messages,
        "validation": validation,
    }
    atomic_write_json(root / "status" / "inference_status.json", status)
    if failures or not validation["passed"]:
        raise RuntimeError(f"Formal inference incomplete; failures={len(failures)}")
    marker = {
        **status,
        "ready_manifest_sha256": sha256_file(ready_path),
        "prompt_manifest_sha256": ready["prompt_manifest_sha256"],
    }
    atomic_write_json(root / "status" / "B200_INFERENCE_COMPLETE.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-model-record", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.cpu_model_record:
        record = _load_json(args.cpu_model_record)
        result = run_model_worker(root, record, "cpu", args.resume)
    else:
        result = run_formal(root, args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
