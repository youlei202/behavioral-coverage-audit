from __future__ import annotations

import gc
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import atomic_parquet
from .core import (
    LABELS,
    canonical_prompt,
    common_bootstrap_multiplicities,
    fit_mae_simplex,
    fit_mse_simplex,
    permute_options,
    total_variation,
    weighted_mae,
)
from .inference import (
    GENERATION_SCHEMA,
    SCORE_SCHEMA,
    _load_model,
    generate_prompt_batch,
    score_prompt_batch,
    validate_shard,
    write_shard,
)
from .prepare import _render_prompt
from .utils import atomic_write_json, load_json, read_jsonl, sha256_file, sha256_text, utc_now


def _tiny_snapshot(root: Path) -> Path:
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    snapshots = list(
        (v2 / "hf_cache/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots").iterdir()
    )
    if len(snapshots) != 1 or not snapshots[0].is_dir():
        raise FileNotFoundError("Exact cached SmolLM2-135M-Instruct snapshot is unavailable")
    return snapshots[0]


def run(root: Path) -> dict[str, Any]:
    output_root = root / "outputs/validation/smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_json(root / "configs/experiment_v23.json")
    v2 = Path(config["source_roots"]["v2"]).resolve(strict=True)
    selected = list(read_jsonl(v2 / "data/mmlu_pro_selected_560.jsonl"))
    base = next(row for row in selected if len(row["options"]) == 3)
    snapshot = _tiny_snapshot(root)
    record = {
        "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": snapshot.name,
        "local_cache_path": str(snapshot),
        "model_adapter": "causal_lm",
        "batch_size": 1,
    }
    prompts: list[dict[str, Any]] = []
    rendered: dict[str, dict[str, Any]] = {}
    model, tokenizer = _load_model(record, "cpu")
    try:
        for rotation in range(len(base["options"])):
            options, semantic_to_visible, visible_to_semantic = permute_options(
                base["options"], rotation
            )
            content = canonical_prompt(base["question"], options)
            prompt_id = f"smoke__{base['base_question_id']}__r{rotation}"
            wrapped, token_ids = _render_prompt(tokenizer, content)
            rendered[prompt_id] = {
                "prompt_id": prompt_id,
                "canonical_content_sha256": sha256_text(content),
                "rendered_prompt": wrapped,
                "rendered_prompt_sha256": sha256_text(wrapped),
                "prompt_token_count": len(token_ids),
            }
            answer_index = semantic_to_visible[int(base["answer_index"])]
            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "source_v2_prompt_id": f"{base['base_question_id']}__clean",
                    "base_question_id": str(base["base_question_id"]),
                    "category": base["category"],
                    "semantic_condition": "C0",
                    "condition": "clean",
                    "family": "clean",
                    "dose": 0.0,
                    "track": None,
                    "permutation_id": rotation,
                    "option_count": len(options),
                    "semantic_to_visible": semantic_to_visible,
                    "visible_to_semantic": visible_to_semantic,
                    "option_labels": list(LABELS[: len(options)]),
                    "candidate_continuations": [f" ({label})" for label in LABELS[: len(options)]],
                    "semantic_answer_index": int(base["answer_index"]),
                    "answer_index": answer_index,
                    "canonical_content_sha256": sha256_text(content),
                }
            )
        score_rows = score_prompt_batch(model, tokenizer, prompts, rendered, device="cpu")
        for row in score_rows:
            row.update(
                {
                    "model_id": record["id"],
                    "model_revision": record["revision"],
                    "manifest_sha256": "smoke-manifest",
                }
            )
        generation_prompts = [
            {
                **prompt,
                "generation_prompt_id": f"generation__{prompt['prompt_id']}",
                "scoring_prompt_id": prompt["prompt_id"],
            }
            for prompt in prompts
        ]
        generation_rows = generate_prompt_batch(
            model, tokenizer, generation_prompts, rendered, device="cpu"
        )
        for row in generation_rows:
            row.update(
                {
                    "model_id": record["id"],
                    "model_revision": record["revision"],
                    "manifest_sha256": "smoke-generation-manifest",
                }
            )
    finally:
        del model, tokenizer
        gc.collect()

    vectors = np.stack(
        [np.asarray(row["semantic_remapped_probabilities"], dtype=np.float64) for row in score_rows]
    )
    average = vectors.mean(axis=0)
    if not np.isclose(average.sum(), 1.0, atol=1e-12):
        raise AssertionError("Smoke permutation average is not normalized")
    score_dir = output_root / "raw"
    score_paths = write_shard(
        root,
        score_dir,
        0,
        score_rows,
        schema=SCORE_SCHEMA,
        record=record,
        manifest_sha256="smoke-manifest",
        prompt_key="prompt_id",
    )
    generation_dir = output_root / "generation"
    generation_paths = write_shard(
        root,
        generation_dir,
        0,
        generation_rows,
        schema=GENERATION_SCHEMA,
        record=record,
        manifest_sha256="smoke-generation-manifest",
        prompt_key="generation_prompt_id",
    )
    resume_skip = validate_shard(
        score_paths[0],
        score_paths[1],
        schema=SCORE_SCHEMA,
        model_id=record["id"],
        revision=record["revision"],
        manifest_sha256="smoke-manifest",
        expected_prompt_ids=[row["prompt_id"] for row in score_rows],
        prompt_key="prompt_id",
    )
    if not resume_skip:
        raise AssertionError("Smoke resume did not recognize a valid completed shard")

    rng = np.random.default_rng(20260829)
    design = rng.uniform(0.05, 0.95, size=(48, 4))
    truth = np.asarray([0.55, 0.25, 0.15, 0.05])
    target = design @ truth
    mass = np.tile([0.5, 1 / 6, 1 / 6, 1 / 6], 12)
    mse_weights = fit_mse_simplex(design, target, mass)
    mae_weights = fit_mae_simplex(design, target, mass)
    mse_error = weighted_mae(target, design @ mse_weights, mass)
    mae_error = weighted_mae(target, design @ mae_weights, mass)
    sibling_removed = fit_mse_simplex(design[:, 1:], target, mass)
    sibling_inflation = weighted_mae(target, design[:, 1:] @ sibling_removed, mass) - mse_error
    if max(mse_error, mae_error) > 1e-8 or sibling_inflation <= 0:
        raise AssertionError("Smoke endpoint/matched-objective/sibling mock failed")

    selected_mock = pd.DataFrame(
        {
            "base_question_id": [f"q{index}" for index in range(12)],
            "category": ["a"] * 6 + ["b"] * 6,
        }
    )
    bootstrap_rows = []
    for replicate in range(5):
        multiplicity = common_bootstrap_multiplicities(selected_mock, replicate, 20260828)
        bootstrap_rows.append(
            {
                "replicate": replicate,
                "total_multiplicity": sum(multiplicity.values()),
                "shared_multiplicity_digest": sha256_text(str(sorted(multiplicity.items()))),
            }
        )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    if not bootstrap_frame["total_multiplicity"].eq(12).all():
        raise AssertionError("Smoke common bootstrap multiplicity is malformed")
    atomic_parquet(output_root / "tiny_common_bootstrap.parquet", bootstrap_frame)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(np.arange(len(average)), average, color="#4c78a8")
    ax.set_title("V2.3 CPU smoke: permutation average")
    ax.set_xlabel("semantic option")
    ax.set_ylabel("probability")
    figure_path = output_root / "smoke_permutation_average.png"
    fig.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    package_path = output_root / "smoke_test_package.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*score_paths, *generation_paths, output_root / "tiny_common_bootstrap.parquet", figure_path]:
            archive.write(path, path.relative_to(output_root))
    with zipfile.ZipFile(package_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Smoke test package CRC validation failed")
    status = {
        "schema_version": "pier_v23_real_cpu_smoke_v1",
        "completed_at": utc_now(),
        "passed": True,
        "model_id": record["id"],
        "model_revision": record["revision"],
        "base_question_id": str(base["base_question_id"]),
        "option_count": len(base["options"]),
        "cyclic_rotation_count": len(prompts),
        "score_row_count": len(score_rows),
        "generation_row_count": len(generation_rows),
        "max_new_tokens": 512,
        "permutation_average": average.tolist(),
        "maximum_rotation_tv": max(total_variation(row, average) for row in vectors),
        "resume_skip_validated": resume_skip,
        "endpoint_mse_error": mse_error,
        "endpoint_mae_error": mae_error,
        "sibling_removal_inflation": sibling_inflation,
        "common_bootstrap_replicates": 5,
        "figure": str(figure_path),
        "test_package": str(package_path),
        "test_package_sha256": sha256_file(package_path),
    }
    atomic_write_json(root / "status/cpu_smoke_test.json", status)
    return status
