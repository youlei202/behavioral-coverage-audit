from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import clustered_bootstrap_with_refitting, run_synthetic_controls
from .inference import (
    SCORE_SCHEMA,
    _load_model,
    _validate_shard,
    _write_shard,
    generate_prompt_batch,
    score_prompt_batch,
)
from .packaging import build_results_package, validate_results_package
from .plotting import make_all_figures
from .solver import fit_simplex_projection
from .utils import (
    atomic_write_json,
    atomic_write_text,
    project_root,
    read_jsonl,
    sha256_file,
    utc_now,
    write_jsonl,
)

SMOKE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _render(tokenizer: Any, canonical_content: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": canonical_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = canonical_content
    if canonical_content not in rendered:
        raise ValueError("Smoke rendering changed canonical user content")
    return rendered


def _tiny_plot_frames() -> dict[str, pd.DataFrame]:
    targets = ["smoke-target-a", "smoke-target-b", "smoke-target-c"]
    scalar_rows = []
    vector_rows = []
    for split_seed in (11, 12):
        for target_index, target in enumerate(targets):
            for family_index, family in enumerate(("irrelevant_context", "content_deletion")):
                for dose in (0.0, 1.0, math.nan):
                    base = 0.01 + 0.006 * target_index + 0.004 * family_index
                    pier = base + (0.005 if dose == 1 else 0.0) + 0.001 * (split_seed - 11)
                    scalar_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "dose": dose,
                            "representation": "gold_probability",
                            "peer_set_condition": "all_peers",
                            "split_seed": split_seed,
                            "pier": pier,
                            "fit_selected_single_error": pier * 1.5,
                            "honest_convexity_gap": 1.5,
                            "uniform_peer_error": pier * 1.8,
                            "dose_specific_convex_error": pier * 0.95,
                            "affine_ridge_error": pier * 0.9,
                            "oracle_single_peer_error": pier * 0.8,
                        }
                    )
                    vector_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "dose": dose,
                            "representation": "probability_vector",
                            "peer_set_condition": "all_peers",
                            "split_seed": split_seed,
                            "weight_fit": "vector_fitted",
                            "mean_total_variation": pier * 1.2,
                            "mean_js_divergence": pier * 0.4,
                            "top1_agreement": 0.9,
                        }
                    )
    return {
        "scalar": pd.DataFrame(scalar_rows),
        "vector": pd.DataFrame(vector_rows),
        "peer_removal": pd.DataFrame(
            {
                "removal_type": ["exact_sibling", "broad_lineage"],
                "observed_inflation": [0.01, 0.02],
                "random_median_inflation": [0.008, 0.01],
            }
        ),
        "transfer": pd.DataFrame(
            {
                "transfer_type": ["irrelevant_low_to_high", "irrelevant_to_deletion"],
                "same_design_error": [0.02, 0.02],
                "transferred_error": [0.03, 0.04],
            }
        ),
        "stability": pd.DataFrame(
            {
                "median_rank_correlation": [0.9, 0.8],
                "minimum_rank_correlation": [0.7, 0.6],
            }
        ),
        "permutation": pd.DataFrame(
            {
                "model_id": targets,
                "probability_vector_tv": [0.01, 0.02, 0.03],
            }
        ),
        "generation": pd.DataFrame(
            {
                "model_id": targets,
                "score_generation_agreement": [True, True, False],
                "malformed": [False, False, True],
            }
        ),
        "ambiguity": pd.DataFrame(
            {"peer": targets, "interval_width": [0.1, 0.2, 0.15]}
        ),
        "coverage": pd.DataFrame(
            {
                "peer_subset_size": [1, 1, 2, 2, 3, 3],
                "pier": [0.05, 0.06, 0.03, 0.04, 0.02, 0.025],
            }
        ),
    }


def _test_plot_and_package(root: Path) -> dict[str, Any]:
    smoke_parent = root / "outputs" / "smoke_test"
    smoke_parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="workspace-", dir=smoke_parent))
    frames = _tiny_plot_frames()
    figure_paths = make_all_figures(workspace / "outputs" / "figures", **frames)
    atomic_write_text(
        workspace / "outputs" / "analysis" / "GOLDMINE_REPORT.md",
        "# Smoke GOLDMINE report\n\nSynthetic/tiny plotting and packaging validation only.\n",
    )
    atomic_write_text(workspace / "REPRODUCE.md", "# Smoke reproduction\n")
    atomic_write_text(workspace / "outputs" / "raw_scores" / "README.txt", "smoke\n")
    atomic_write_text(workspace / "outputs" / "tables" / "smoke.csv", "value\n1\n")
    atomic_write_text(workspace / "outputs" / "controls" / "smoke.json", "{}\n")
    atomic_write_text(workspace / "scripts" / "smoke.sh", "#!/bin/sh\nexit 0\n")
    atomic_write_text(workspace / "src" / "README.txt", "smoke\n")
    atomic_write_text(workspace / "configs" / "smoke.json", "{}\n")
    atomic_write_text(workspace / "status" / "smoke.json", "{}\n")
    package = build_results_package(
        workspace,
        output_path=smoke_parent / "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_SMOKE_RESULTS.zip",
    )
    package_validation = validate_results_package(package)
    return {
        "workspace": str(workspace),
        "figure_count": len(figure_paths),
        "figures": [str(path) for path in figure_paths],
        "package": package_validation,
    }


def run(root: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    status_path = root / "status" / "smoke_test_status.json"
    status: dict[str, Any] = {
        "schema_version": "pier_real_cpu_smoke_v2",
        "timestamp_started": utc_now(),
        "model_id": SMOKE_MODEL_ID,
        "passed": False,
        "checks": {},
    }
    atomic_write_json(status_path, status)
    model = tokenizer = None
    try:
        api = HfApi()
        revision = api.model_info(SMOKE_MODEL_ID).sha
        if revision is None:
            raise RuntimeError("Hugging Face did not return a smoke-model revision")
        snapshot = Path(
            snapshot_download(
                SMOKE_MODEL_ID,
                revision=revision,
                cache_dir=root / "hf_cache" / "hub",
                allow_patterns=[
                    "*.json",
                    "*.model",
                    "*.txt",
                    "*.safetensors",
                    "tokenizer*",
                    "special_tokens_map.json",
                ],
            )
        ).resolve()
        record = {
            "id": SMOKE_MODEL_ID,
            "revision": revision,
            "local_cache_path": str(snapshot),
            "model_adapter": "causal_lm",
            "batch_size": 2,
        }
        model, tokenizer = _load_model(record, "cpu")
        all_prompts = list(read_jsonl(root / "data" / "manifests" / "scoring_prompts.jsonl"))
        chosen: list[dict[str, Any]] = []
        chosen_base_ids: list[str] = []
        for prompt in all_prompts:
            if prompt["condition"] == "clean" and len(chosen_base_ids) < 4:
                chosen_base_ids.append(prompt["base_question_id"])
        wanted = set(chosen_base_ids)
        condition_order = {"clean": 0, "irrelevant_context": 1, "content_deletion": 2}
        for prompt in all_prompts:
            if prompt["base_question_id"] not in wanted:
                continue
            if prompt["condition"] == "clean" or (
                prompt["condition"] == "irrelevant_context"
                and float(prompt["dose"]) == 512.0
                and int(prompt["track"]) == 0
            ) or (
                prompt["condition"] == "content_deletion"
                and float(prompt["dose"]) == 0.4
                and int(prompt["track"]) == 0
            ):
                chosen.append(prompt)
        chosen.sort(
            key=lambda row: (
                chosen_base_ids.index(row["base_question_id"]),
                condition_order[row["condition"]],
            )
        )
        if len(chosen) != 12:
            raise ValueError(f"Expected 12 real smoke prompts, selected {len(chosen)}")
        rendered = {
            prompt["prompt_id"]: _render(tokenizer, prompt["canonical_content"])
            for prompt in chosen
        }
        smoke_manifest = root / "outputs" / "smoke_test" / "scoring_prompts.jsonl"
        _count, prompt_sha = write_jsonl(smoke_manifest, chosen)
        output_dir = root / "outputs" / "smoke_test" / "raw_scores"
        data_path = output_dir / "shard_00000.parquet"
        metadata_path = output_dir / "shard_00000.meta.json"
        expected_ids = [row["prompt_id"] for row in chosen]
        resume_valid = _validate_shard(
            data_path,
            metadata_path,
            schema=SCORE_SCHEMA,
            model_id=SMOKE_MODEL_ID,
            revision=revision,
            prompt_manifest_sha256=prompt_sha,
            expected_prompt_ids=expected_ids,
        )
        if resume_valid:
            rows = pd.read_parquet(data_path).to_dict(orient="records")
        else:
            rows = score_prompt_batch(model, tokenizer, chosen, rendered, device="cpu")
            for row in rows:
                row.update(
                    {
                        "model_id": SMOKE_MODEL_ID,
                        "model_revision": revision,
                        "prompt_manifest_sha256": prompt_sha,
                    }
                )
            data_path, metadata_path = _write_shard(
                root,
                output_dir,
                0,
                rows,
                schema=SCORE_SCHEMA,
                record=record,
                prompt_manifest_sha256=prompt_sha,
                prompt_key="prompt_id",
            )
            resume_valid = _validate_shard(
                data_path,
                metadata_path,
                schema=SCORE_SCHEMA,
                model_id=SMOKE_MODEL_ID,
                revision=revision,
                prompt_manifest_sha256=prompt_sha,
                expected_prompt_ids=expected_ids,
            )
        if not resume_valid:
            raise RuntimeError("Real smoke shard did not pass resume validation")
        generation_prompts = [
            {
                "generation_prompt_id": f"generation__{prompt['prompt_id']}",
                "scoring_prompt_id": prompt["prompt_id"],
                "base_question_id": prompt["base_question_id"],
                "category": prompt["category"],
                "condition": prompt["condition"],
                "option_labels": prompt["option_labels"],
                "answer_index": prompt["answer_index"],
            }
            for prompt in chosen[:3]
        ]
        generated = generate_prompt_batch(
            model, tokenizer, generation_prompts, rendered, device="cpu"
        )

        probabilities = np.stack(
            [np.asarray(row["semantic_probabilities"], dtype=np.float64) for row in rows]
        )
        real_gold = np.asarray([row["gold_probability"] for row in rows])
        scalar_peers = np.column_stack(
            [
                real_gold,
                np.clip(0.9 * real_gold + 0.05, 0, 1),
                np.clip(0.75 * real_gold + 0.1, 0, 1),
            ]
        )
        scalar_target = scalar_peers @ np.array([0.5, 0.3, 0.2])
        scalar_projection = fit_simplex_projection(scalar_peers[:6], scalar_target[:6])
        scalar_pier = float(
            np.mean(np.abs(scalar_target[6:] - scalar_peers[6:] @ scalar_projection.weights))
        )
        vector_peers = np.stack(
            [
                probabilities,
                0.9 * probabilities + 0.1 / probabilities.shape[1],
                0.8 * probabilities + 0.2 / probabilities.shape[1],
            ],
            axis=-1,
        )
        vector_target = np.tensordot(
            vector_peers, np.array([0.5, 0.3, 0.2]), axes=([-1], [0])
        )
        vector_projection = fit_simplex_projection(
            vector_peers[:6].reshape(-1, 3), vector_target[:6].reshape(-1)
        )
        vector_prediction = np.tensordot(
            vector_peers[6:], vector_projection.weights, axes=([-1], [0])
        )
        vector_tv = float(0.5 * np.abs(vector_target[6:] - vector_prediction).sum(axis=1).mean())
        metadata_fit = pd.DataFrame(
            {"base_question_id": [row["base_question_id"] for row in rows[:6]]}
        )
        metadata_eval = pd.DataFrame(
            {"base_question_id": [row["base_question_id"] for row in rows[6:]]}
        )
        bootstrap = clustered_bootstrap_with_refitting(
            metadata_fit,
            scalar_peers[:6],
            scalar_target[:6],
            metadata_eval,
            scalar_peers[6:],
            scalar_target[6:],
            replicates=5,
            seed=20260828,
        )
        control_frame, control_summary = run_synthetic_controls()
        if not control_summary["passed"]:
            raise RuntimeError("Smoke clone/mixture controls failed")
        plot_package = _test_plot_and_package(root)
        checks = {
            "model_revision": revision,
            "snapshot": str(snapshot),
            "rendered_prompt_count": len(rendered),
            "scored_prompt_count": len(rows),
            "generation_prompt_count": len(generated),
            "raw_shard": str(data_path),
            "raw_shard_sha256": sha256_file(data_path),
            "resume_skipped_valid_shard": resume_valid,
            "scalar_disco_pier": scalar_pier,
            "vector_disco_tv": vector_tv,
            "scalar_weights": scalar_projection.weights.tolist(),
            "vector_weights": vector_projection.weights.tolist(),
            "controls_passed": control_summary["passed"],
            "control_rows": control_frame.to_dict(orient="records"),
            "bootstrap_replicates": len(bootstrap),
            "plot_and_package": plot_package,
        }
        if scalar_pier > 1e-7 or vector_tv > 1e-7:
            raise RuntimeError(
                f"Smoke DISCO exact-mixture recovery failed: scalar={scalar_pier}, vector={vector_tv}"
            )
        status.update({"timestamp_completed": utc_now(), "passed": True, "checks": checks})
        atomic_write_json(status_path, status)
        return status
    except Exception as exc:
        status.update(
            {
                "timestamp_failed": utc_now(),
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        raise
    finally:
        del model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    result = run(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
