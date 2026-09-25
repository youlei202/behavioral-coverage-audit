from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from .data_prep import LABELS
from .utils import (
    atomic_write_json,
    atomic_write_text,
    project_root,
    read_jsonl,
    sha256_file,
    slug,
    utc_now,
    write_jsonl,
)


def _model_catalog(root: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads((root / "configs" / "models.json").read_text(encoding="utf-8"))


def _experiment(root: Path) -> dict[str, Any]:
    return json.loads((root / "configs" / "experiment.json").read_text(encoding="utf-8"))


def _safe_card_value(card_data: Any, key: str) -> Any:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(key)
    return getattr(card_data, key, None)


def _weight_patterns(siblings: list[Any]) -> tuple[list[str], list[str]]:
    names = [getattr(item, "rfilename", str(item)) for item in siblings]
    has_safetensors = any(name.endswith(".safetensors") for name in names)
    ignored = [
        "*.gguf",
        "*.onnx",
        "*.h5",
        "*.msgpack",
        "*.pth",
        "*.pt",
        "original/**",
        "onnx/**",
        "gguf/**",
        "flax_model*",
        "tf_model*",
    ]
    if has_safetensors:
        ignored.append("*.bin")
    return names, ignored


def _file_inventory(snapshot: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot)
        if ".cache" in relative.parts:
            continue
        inventory.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _parameter_count(info: Any) -> int | None:
    safetensors = getattr(info, "safetensors", None)
    parameters = getattr(safetensors, "parameters", None)
    if parameters is None and isinstance(safetensors, dict):
        parameters = safetensors.get("parameters")
    if isinstance(parameters, dict):
        numeric = [int(value) for key, value in parameters.items() if key != "total"]
        return int(parameters.get("total", sum(numeric)))
    return None


def _load_tokenizer_and_processor(snapshot: Path) -> tuple[Any, Any | None]:
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_options: dict[str, Any] = {}
    if "mistral" in str(snapshot).casefold():
        tokenizer_options["fix_mistral_regex"] = True
    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            **tokenizer_options,
        )
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception:
        processor = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            **tokenizer_options,
        )
    return tokenizer, processor


def _instantiate_meta(snapshot: Path) -> tuple[str, str, dict[str, Any]]:
    import torch
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    errors: dict[str, str] = {}
    candidates: list[tuple[str, Any]] = [("causal_lm", AutoModelForCausalLM)]
    image_text_auto = getattr(transformers, "AutoModelForImageTextToText", None)
    if image_text_auto is not None:
        candidates.append(("image_text_to_text", image_text_auto))
    for adapter, auto_class in candidates:
        try:
            with init_empty_weights():
                model = auto_class.from_config(
                    config, trust_remote_code=False, torch_dtype=torch.bfloat16
                )
            class_name = model.__class__.__name__
            del model
            gc.collect()
            return adapter, class_name, config.to_dict()
        except Exception as exc:
            errors[adapter] = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"No supported model adapter: {errors}")


def _render_prompt(tokenizer: Any, content: str) -> tuple[str, list[int]]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = content
    if content not in rendered:
        raise ValueError("Canonical user-visible content is not byte-identical inside rendered prompt")
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("Rendered prompt tokenized to an empty sequence")
    return rendered, [int(value) for value in token_ids]


def _candidate_manifest(tokenizer: Any, option_count: int) -> list[dict[str, Any]]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    result: list[dict[str, Any]] = []
    for label in LABELS[:option_count]:
        continuation = f" ({label})"
        token_ids = [
            int(value)
            for value in tokenizer(continuation, add_special_tokens=False)["input_ids"]
        ]
        if not token_ids:
            raise ValueError(f"Candidate {continuation!r} tokenized to empty")
        if special_ids.intersection(token_ids):
            raise ValueError(f"Candidate {continuation!r} contains special token IDs")
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != continuation:
            raise ValueError(
                f"Candidate suffix round-trip failed: intended={continuation!r}, decoded={decoded!r}"
            )
        result.append(
            {
                "label": label,
                "continuation": continuation,
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "decoded": decoded,
            }
        )
    return result


def _render_all_prompts(
    root: Path, model_id: str, tokenizer: Any, model_max_length: int
) -> dict[str, Any]:
    import hashlib

    scoring_path = root / "data" / "manifests" / "scoring_prompts.jsonl"
    scoring_sha = sha256_file(scoring_path)
    output_path = root / "data" / "manifests" / "rendered_prompts" / f"{slug(model_id)}.jsonl"
    option_counts: set[int] = set()
    maximum_tokens = 0
    rows: list[dict[str, Any]] = []
    for prompt in read_jsonl(scoring_path):
        rendered, token_ids = _render_prompt(tokenizer, prompt["canonical_content"])
        maximum_tokens = max(maximum_tokens, len(token_ids))
        option_counts.add(len(prompt["option_labels"]))
        if 0 < model_max_length < 1_000_000 and len(token_ids) + 16 > model_max_length:
            raise ValueError(
                f"Prompt {prompt['prompt_id']} requires {len(token_ids) + 16} tokens, "
                f"model limit is {model_max_length}"
            )
        rows.append(
            {
                "prompt_id": prompt["prompt_id"],
                "canonical_content_sha256": prompt["canonical_content_sha256"],
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "prompt_token_count": len(token_ids),
            }
        )
    count, digest = write_jsonl(output_path, rows)
    expected = _experiment(root)["scoring_prompts_per_model"]
    if count != expected:
        raise ValueError(f"Rendered prompt count {count} != {expected}")
    candidates = {
        str(option_count): _candidate_manifest(tokenizer, option_count)
        for option_count in sorted(option_counts)
    }
    return {
        "rendered_prompt_path": str(output_path),
        "rendered_prompt_count": count,
        "rendered_prompt_sha256": digest,
        "source_scoring_prompt_sha256": scoring_sha,
        "maximum_prompt_tokens": maximum_tokens,
        "candidate_sequences_by_option_count": candidates,
    }


def _validate_cached_model(root: Path, record: dict[str, Any]) -> bool:
    try:
        manifest_path = Path(record["compatibility_manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model_id"] != record["id"]:
            return False
        snapshot = Path(manifest["local_cache_path"])
        if not snapshot.is_dir():
            return False
        rendered = Path(manifest["rendering"]["rendered_prompt_path"])
        if sha256_file(rendered) != manifest["rendering"]["rendered_prompt_sha256"]:
            return False
        for item in manifest["file_inventory"]:
            path = snapshot / item["path"]
            if (
                not path.is_file()
                or path.stat().st_size != item["bytes"]
                or sha256_file(path) != item["sha256"]
            ):
                return False
        return True
    except Exception:
        return False


def _record_from_compatibility_manifest(
    root: Path, spec: dict[str, Any]
) -> dict[str, Any] | None:
    manifest_path = root / "data" / "manifests" / "models" / f"{slug(spec['id'])}.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            **spec,
            "revision": manifest["revision"],
            "parameter_count": manifest["parameter_count"],
            "tokenizer_class": manifest["tokenizer_class"],
            "processor_class": manifest["processor_class"],
            "model_class": manifest["model_class"],
            "model_adapter": manifest["model_adapter"],
            "chat_template_available": manifest["chat_template_available"],
            "license": manifest["license"],
            "access_status": "available",
            "local_cache_path": manifest["local_cache_path"],
            "weight_bytes": manifest["weight_bytes"],
            "rendered_prompt_sha256": manifest["rendering"]["rendered_prompt_sha256"],
            "compatibility_manifest_path": str(manifest_path),
        }
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def prestage_one(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(spec["id"], files_metadata=True)
    revision = info.sha
    _names, ignored = _weight_patterns(list(info.siblings or []))
    snapshot = Path(
        snapshot_download(
            repo_id=spec["id"],
            revision=revision,
            cache_dir=root / "hf_cache" / "hub",
            ignore_patterns=ignored,
            max_workers=8,
        )
    ).resolve()
    tokenizer, processor = _load_tokenizer_and_processor(snapshot)
    adapter, model_class, config = _instantiate_meta(snapshot)
    model_max_length = int(getattr(tokenizer, "model_max_length", 0) or 0)
    rendering = _render_all_prompts(root, spec["id"], tokenizer, model_max_length)
    inventory = _file_inventory(snapshot)
    weight_bytes = sum(
        item["bytes"]
        for item in inventory
        if item["path"].endswith((".safetensors", ".bin"))
    )
    if weight_bytes <= 0:
        raise ValueError("No complete model weight files found")
    chat_template = getattr(tokenizer, "chat_template", None)
    manifest = {
        "schema_version": "pier_model_compatibility_v2",
        "timestamp": utc_now(),
        "model_id": spec["id"],
        "revision": revision,
        "local_cache_path": str(snapshot),
        "parameter_count": _parameter_count(info),
        "broad_lineage": spec["broad_lineage"],
        "exact_sibling_group": spec["exact_sibling_group"],
        "provider_family": spec["provider_family"],
        "post_training_type": spec["post_training_type"],
        "license": _safe_card_value(info.card_data, "license"),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_compatibility_options": (
            {"fix_mistral_regex": True}
            if "mistral" in spec["id"].casefold()
            else {}
        ),
        "processor_class": processor.__class__.__name__ if processor is not None else None,
        "model_class": model_class,
        "model_adapter": adapter,
        "chat_template_available": bool(chat_template),
        "chat_template_sha256": (
            __import__("hashlib").sha256(chat_template.encode()).hexdigest()
            if chat_template
            else None
        ),
        "model_max_length": model_max_length,
        "config_model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "torch_dtype_declared": config.get("torch_dtype") or config.get("dtype"),
        "formal_dtype": "bfloat16",
        "batch_size": spec["batch_size"],
        "weight_bytes": weight_bytes,
        "file_inventory": inventory,
        "rendering": rendering,
    }
    manifest_path = root / "data" / "manifests" / "models" / f"{slug(spec['id'])}.json"
    atomic_write_json(manifest_path, manifest)
    return {
        **spec,
        "revision": revision,
        "parameter_count": manifest["parameter_count"],
        "tokenizer_class": manifest["tokenizer_class"],
        "processor_class": manifest["processor_class"],
        "model_class": model_class,
        "model_adapter": adapter,
        "chat_template_available": manifest["chat_template_available"],
        "license": manifest["license"],
        "access_status": "available",
        "local_cache_path": str(snapshot),
        "weight_bytes": weight_bytes,
        "rendered_prompt_sha256": rendering["rendered_prompt_sha256"],
        "compatibility_manifest_path": str(manifest_path),
    }


def _is_access_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(token in text for token in ("401", "403", "gated", "access to model", "repository not found"))


def run(root: Path, resume: bool = True) -> dict[str, Any]:
    catalog = _model_catalog(root)
    existing_path = root / "configs" / "resolved_models.json"
    existing_by_id: dict[str, dict[str, Any]] = {}
    if resume and existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_by_id = {row["id"]: row for row in existing.get("models", [])}
    if resume:
        for spec in [*catalog["preferred"], *catalog["fallback"]]:
            if spec["id"] not in existing_by_id:
                recovered = _record_from_compatibility_manifest(root, spec)
                if recovered is not None:
                    existing_by_id[spec["id"]] = recovered

    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    access_blocked: list[dict[str, Any]] = []
    preferred_failures: list[dict[str, Any]] = []
    candidates = [("preferred", item) for item in catalog["preferred"]]
    candidates.extend(("fallback", item) for item in catalog["fallback"])
    preferred_count = len(catalog["preferred"])
    for tier, spec in candidates:
        if len(selected) >= preferred_count:
            break
        cached = existing_by_id.get(spec["id"])
        if cached is not None and _validate_cached_model(root, cached):
            selected.append(cached)
            print(f"[resume] validated cached model {spec['id']}", flush=True)
            continue
        print(f"[prestage] {tier} model {spec['id']}", flush=True)
        try:
            selected.append(prestage_one(root, spec))
        except Exception as exc:
            failure = {
                "timestamp": utc_now(),
                "tier": tier,
                "model_id": spec["id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "access_related": _is_access_error(exc),
            }
            failures.append(failure)
            if tier == "preferred":
                preferred_failures.append(failure)
            if failure["access_related"] and tier == "preferred":
                access_blocked.append(failure)
            print(f"[failed] {spec['id']}: {type(exc).__name__}: {exc}", flush=True)

    lineages = {row["broad_lineage"] for row in selected}
    sibling_groups = Counter(
        row["exact_sibling_group"] for row in selected if row["exact_sibling_group"] is not None
    )
    controlled_pairs = sum(count >= 2 for count in sibling_groups.values())
    result = {
        "schema_version": "pier_resolved_models_v2",
        "timestamp": utc_now(),
        "selection_rule": "preferred roster in declared order; declared fallbacks only after recorded failure",
        "models": selected,
        "failures": failures,
        "preferred_failures": preferred_failures,
        "resolved_model_count": len(selected),
        "broad_lineage_count": len(lineages),
        "controlled_sibling_pair_count": controlled_pairs,
        "minimum_roster_passed": len(selected) >= 7 and len(lineages) >= 5 and controlled_pairs >= 1,
        "full_roster_passed": (
            len(selected) == preferred_count
            and {row["id"] for row in selected}
            == {row["id"] for row in catalog["preferred"]}
        ),
        "access_blocked": bool(access_blocked),
    }
    atomic_write_json(existing_path, result)
    atomic_write_json(root / "status" / "model_prestage_status.json", result)
    if access_blocked:
        lines = [
            "# Model access blocked",
            "",
            "At least one preferred gated model requires access or license acceptance.",
            "B200 readiness must not be issued until this is resolved.",
            "",
        ]
        lines.extend(f"- `{item['model_id']}`: {item['error']}" for item in access_blocked)
        atomic_write_text(root / "status" / "ACCESS_BLOCKED.md", "\n".join(lines) + "\n")
    else:
        blocked_path = root / "status" / "ACCESS_BLOCKED.md"
        if blocked_path.exists():
            history = root / "status" / "history"
            history.mkdir(parents=True, exist_ok=True)
            stamp = utc_now().replace(":", "-")
            os.replace(blocked_path, history / f"ACCESS_BLOCKED.resolved.{stamp}.md")
    if not result["minimum_roster_passed"]:
        raise RuntimeError(
            "Minimum scientific roster failed: "
            f"models={len(selected)}, lineages={len(lineages)}, sibling_pairs={controlled_pairs}"
        )
    if access_blocked:
        raise RuntimeError("Preferred model access is blocked; see status/ACCESS_BLOCKED.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run(args.root.resolve(), resume=not args.no_resume)
    print(json.dumps({key: value for key, value in result.items() if key != "models"}, indent=2))


if __name__ == "__main__":
    main()
