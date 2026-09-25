from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import traceback
from pathlib import Path
from typing import Any

from .analysis import run_point_analyses
from .bootstrap import run_common_bootstraps
from .inputs import recheck_after
from .packaging import create_package
from .plotting import create_figures
from .reporting import (
    build_claim_locking,
    create_numerical_sensitivity_audit,
    create_report,
    create_tables,
)
from .utils import (
    atomic_write_json,
    load_json,
    project_root,
    sha256_file,
    tree_sha256,
    utc_now,
)


def _environment_manifest(root: Path) -> dict[str, Any]:
    import torch

    packages = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface-hub",
        "safetensors",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "cvxpy",
        "matplotlib",
    ]
    models = load_json(root / "configs/resolved_models_v23.json")["models"]
    config = load_json(root / "configs/experiment_v23.json")
    b200 = load_json(root / "manifests/b200_environment.json")
    frozen_environment = load_json(root / "status/environment_status.json")
    locked_packages = frozen_environment["packages"]
    current_locked_packages = {
        name: importlib.metadata.version(name) for name in locked_packages
    }
    if current_locked_packages != locked_packages:
        raise RuntimeError("Locked package versions differ from the frozen environment")
    if platform.python_version() != "3.11.15":
        raise RuntimeError(
            f"Authorized recovery requires CPython 3.11.15, got {platform.python_version()}"
        )
    return {
        "schema_version": "pier_v231_environment_manifest_v1",
        "created_at": utc_now(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "pytorch_cuda_build": torch.version.cuda,
        "b200_runtime": b200,
        "models": [
            {
                "id": row["id"],
                "model_revision": row["revision"],
                "tokenizer_revision": row["tokenizer_revision"],
                "chat_template_sha256": row["v23_chat_template_sha256"],
            }
            for row in models
        ],
        "source_v2_sha256": load_json(root / "status/input_inventory_before.json")[
            "source_v2_sha256"
        ],
        "source_v22_sha256": load_json(root / "status/input_inventory_before.json")[
            "source_v22_sha256"
        ],
        "score_manifest_sha256": sha256_file(
            root / "data/permutations/score_semantic_manifest.jsonl"
        ),
        "score_inference_manifest_sha256": sha256_file(
            root / "data/permutations/score_inference_manifest.jsonl"
        ),
        "generation_manifest_sha256": sha256_file(
            root / "data/generation/long_generation_manifest.jsonl"
        ),
        "stage_a_source_tree_sha256": load_json(root / "status/B200_READY_STAGE2.json")[
            "source_tree_sha256"
        ],
        "authorized_v231_source_tree_sha256": tree_sha256(root / "src"),
        "authorized_patch_sha256": sha256_file(
            root / "manifests/V2_3_1_AUTHORIZED_PATCH.diff"
        ),
        "authorization_marker_sha256": sha256_file(
            root / "status/V2_3_1_AUTHORIZATION.json"
        ),
        "b200_launcher_sha256": sha256_file(root / "scripts/run_b200_interface_v23.sh"),
        "cpu_postprocess_launcher_sha256": sha256_file(
            root / "scripts/run_cpu_postprocess_v23.sh"
        ),
        "seeds": {
            "splits": config["split_seeds"],
            "bootstrap": config["bootstrap_seed"],
        },
        "runtime_environment_bitwise_identical": False,
        "bitwise_runtime_identity_claimed": False,
        "locked_package_versions_identical": True,
        "python_environment_deviation": {
            "prestaged_python": frozen_environment["python"],
            "executed_python": platform.python_version(),
            "authorized": True,
            "wording": (
                "Stage-C post-processing was executed under CPython 3.11.15 rather than "
                "the prestaged CPython 3.11.13 because the original interpreter was "
                "unavailable on the replacement CPU host. Python remained within the same "
                "3.11 ABI line and all locked package versions, frozen scientific inputs, "
                "manifests, Stage-B outputs, and hashes were unchanged, except for the "
                "explicitly authorized V2.3.1 numerical-feasibility patch."
            ),
        },
    }


def run(root: Path, resume: bool) -> dict[str, Any]:
    complete_marker = root / "status/B200_STAGE2_COMPLETE.json"
    if not complete_marker.is_file() or load_json(complete_marker).get("complete") is not True:
        raise RuntimeError("B200_STAGE2_COMPLETE.json is absent or incomplete")
    status_path = root / "status/cpu_postprocess_status.json"
    status: dict[str, Any] = {
        "schema_version": "pier_v231_cpu_postprocess_v1",
        "started_at": utc_now(),
        "complete": False,
        "stage": "point_analysis",
    }
    atomic_write_json(status_path, status)
    try:
        point_status_path = root / "status/point_analysis.json"
        if not (resume and point_status_path.is_file() and load_json(point_status_path).get("passed")):
            point = run_point_analyses(root)
            atomic_write_json(point_status_path, point)
        status["stage"] = "common_resample_bootstrap"
        atomic_write_json(status_path, status)
        run_common_bootstraps(root, resume=resume)
        status["stage"] = "claim_locking_tables_report_figures"
        atomic_write_json(status_path, status)
        claims = build_claim_locking(root)
        create_numerical_sensitivity_audit(root, claims)
        tables = create_tables(root, claims)
        figures = create_figures(root, tables, claims)
        report = create_report(root, claims)
        atomic_write_json(root / "manifests/environment_manifest.json", _environment_manifest(root))
        status["stage"] = "input_immutability_recheck"
        atomic_write_json(status_path, status)
        source_check = recheck_after(root)
        bootstrap_status = load_json(root / "status/common_resample_bootstrap.json")
        repair_status = load_json(
            root / "status/numerical_feasibility_repair_summary.json"
        )
        expected_hashes = {
            root / "status/B200_STAGE2_COMPLETE.json": "0ebc20e8487d2857a07e01ab814381a39ed43ab09af5e1439dcb3ad01e1cc3c6",
            root / "scripts/run_b200_interface_v23.sh": "242e8d7212eee56fb67323d86bbc3c77a95cc335aa27d9189517b1c59ed7f679",
            root / "scripts/run_cpu_postprocess_v23.sh": "44bf9d0397ce3e43a6ee595aca2f5facdf9d767ac5b1612a53df11e4b581b7f2",
            root / "data/permutations/score_inference_manifest.jsonl": "065d65f94e51a4897159511f67720bdc793309b237b2dd92e5dae00e6ec933a3",
            root / "data/generation/long_generation_manifest.jsonl": "d1436e2aefd2519ebb0e73e34ce703b541a542c17882dc931abb4a4a22437c5d",
        }
        hash_checks = {
            str(path): {
                "expected": expected,
                "actual": sha256_file(path),
                "passed": sha256_file(path) == expected,
            }
            for path, expected in expected_hashes.items()
        }
        quarantine_files = sorted(
            str(path) for path in (root / "outputs/quarantine").rglob("*") if path.is_file()
        )
        required_figure_names = {
            "fig_v23_interface_validated_main.pdf",
            "fig_v23_interface_validated_main.png",
            "fig_v23_permutation_sensitivity.pdf",
            "fig_v23_endpoint_interface_transfer.pdf",
            "fig_v23_matched_objective_convexity.pdf",
            "fig_v23_sibling_removal.pdf",
            "fig_v23_generation_permutation_stability.pdf",
            "fig_v23_score_generation_alignment.pdf",
            "fig_v23_common_bootstrap_comparison.pdf",
        }
        actual_figure_names = {path.name for path in figures}
        prepackage_passed = bool(
            source_check["byte_identical"]
            and bootstrap_status["passed"]
            and bootstrap_status["shard_count"] == 800
            and bootstrap_status["pending"] == 0
            and bootstrap_status["failed"] == 0
            and repair_status["passed"]
            and len(tables) == 6
            and required_figure_names.issubset(actual_figure_names)
            and all(record["passed"] for record in hash_checks.values())
            and not quarantine_files
        )
        prepackage_gate = {
            "schema_version": "pier_v231_prepackage_gate_v1",
            "checked_at": utc_now(),
            "passed": prepackage_passed,
            "source_inputs_byte_identical": source_check["byte_identical"],
            "bootstrap_shards": bootstrap_status["shard_count"],
            "bootstrap_pending": bootstrap_status["pending"],
            "bootstrap_failed": bootstrap_status["failed"],
            "repair_gates_passed": repair_status["passed"],
            "table_count": len(tables),
            "required_figures_present": required_figure_names.issubset(
                actual_figure_names
            ),
            "figure_files": sorted(actual_figure_names),
            "frozen_hash_checks": hash_checks,
            "quarantine_files": quarantine_files,
            "authorized_source_tree_sha256": tree_sha256(root / "src"),
            "authorized_patch_sha256": sha256_file(
                root / "manifests/V2_3_1_AUTHORIZED_PATCH.diff"
            ),
        }
        atomic_write_json(root / "status/V2_3_1_PREPACKAGE_GATE.json", prepackage_gate)
        if not prepackage_passed:
            raise RuntimeError("V2.3.1 prepackage gate failed")
        status["stage"] = "packaging"
        atomic_write_json(status_path, status)
        package = create_package(root)
        result = {
            "schema_version": "pier_v231_complete_v1",
            "complete": True,
            "completed_at": utc_now(),
            "report": str(report),
            "claim_count": len(claims),
            "figure_count": len(figures),
            "table_count": len(tables),
            "source_inputs_byte_identical": source_check["byte_identical"],
            "result_package": package["path"],
            "result_package_sha256": package["sha256"],
        }
        atomic_write_json(root / "status/V2_3_INTERFACE_VALIDATION_COMPLETE.json", result)
        atomic_write_json(root / "status/V2_3_1_STAGE_C_RECOVERY_COMPLETE.json", result)
        status.update(result)
        status["stage"] = "complete"
        atomic_write_json(status_path, status)
        return result
    except Exception as exc:
        status.update(
            {
                "complete": False,
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.root.resolve(strict=True), args.resume)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("========================================================", flush=True)
    print("V2.3.1 STAGE-C RECOVERY COMPLETE", flush=True)
    print("", flush=True)
    print("Bootstrap:", flush=True)
    print("  800 / 800 complete", flush=True)
    print("", flush=True)
    print("No new B200 inference was performed.", flush=True)
    print("", flush=True)
    print("Final ZIP:", flush=True)
    print(result["result_package"], flush=True)
    print("", flush=True)
    print("SHA256:", flush=True)
    print(result["result_package_sha256"], flush=True)
    print("========================================================", flush=True)


if __name__ == "__main__":
    main()
