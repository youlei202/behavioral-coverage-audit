from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from .core import run_synthetic_controls
from .inputs import record_before, resolve_roots
from .prepare import prepare_all
from .smoke import run as run_smoke
from .utils import (
    LOGICAL_ROOT,
    atomic_write_json,
    load_json,
    project_root,
    sha256_file,
    tree_sha256,
    utc_now,
)

REQUIRED_DIRECTORIES = (
    "artifacts",
    "bootstrap_shards",
    "configs",
    "data/generation",
    "data/permutations",
    "env",
    "figures",
    "hf_cache",
    "logs",
    "manifests",
    "outputs/analysis",
    "outputs/generation",
    "outputs/raw_scores",
    "outputs/tables",
    "outputs/validation",
    "scripts",
    "src",
    "status",
    "tests",
)


def validate_environment(root: Path) -> dict[str, Any]:
    import torch

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Pinned Stage-A Python must be 3.11, found {sys.version}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Pinned PyTorch CUDA build must be 12.8, found {torch.version.cuda}")
    if root.resolve() != LOGICAL_ROOT.resolve(strict=True):
        raise RuntimeError(f"Project root mismatch: {root.resolve()} vs {LOGICAL_ROOT.resolve()}")
    v2 = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2").resolve(strict=True)
    venv = (root / "env/venv").resolve(strict=True)
    wheelhouse = (root / "env/wheelhouse").resolve(strict=True)
    if venv != (v2 / "env/venv").resolve(strict=True):
        raise RuntimeError("V2.3 is not reusing the tested V2 environment")
    if wheelhouse != (v2 / "env/wheelhouse").resolve(strict=True):
        raise RuntimeError("V2.3 is not reusing the tested offline wheelhouse")
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
        "pytest",
        "ruff",
    ]
    source_lock = v2 / "env/requirements.lock.txt"
    result = {
        "schema_version": "pier_v23_cpu_environment_v1",
        "validated_at": utc_now(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "torch_cuda_build": torch.version.cuda,
        "visible_gpu_count": int(torch.cuda.device_count()),
        "environment_path": str(root / "env/venv"),
        "environment_resolved_path": str(venv),
        "offline_wheelhouse_path": str(root / "env/wheelhouse"),
        "offline_wheel_count": len(list(wheelhouse.glob("*.whl"))),
        "source_lock_path": str(source_lock),
        "source_lock_sha256": sha256_file(source_lock),
    }
    atomic_write_json(root / "status/environment_status.json", result)
    return result


def _run_checked(command: list[str], root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    record = {"command": command, "returncode": completed.returncode, "output": completed.stdout}
    if completed.returncode:
        raise RuntimeError(
            f"Check failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-12000:]}"
        )
    return record


def run_static_checks(root: Path) -> dict[str, Any]:
    python = str(Path(sys.executable))
    scripts = [str(path) for path in sorted((root / "scripts").glob("*.sh"))]
    commands = [
        ["bash", "-n", *scripts],
        [python, "-m", "compileall", "-q", "src", "tests"],
        [python, "-m", "ruff", "check", "src", "tests"],
        [python, "-m", "pytest", "-q"],
    ]
    if shutil.which("shellcheck"):
        commands.insert(1, ["shellcheck", *scripts])
    records = [_run_checked(command, root) for command in commands]
    result = {"passed": True, "checks": records}
    atomic_write_json(root / "status/static_and_unit_checks.json", result)
    return result


def disk_check(root: Path, prompts: dict[str, Any]) -> dict[str, Any]:
    free = shutil.disk_usage(root).free
    score_rows = int(prompts["score_new_inference_rows_all_models"])
    generation_rows = int(prompts["generation_rows_all_models"])
    expected_raw = score_rows * 2400 + generation_rows * 5000
    expected_analysis = 12 * 1024**3
    required = 3 * expected_raw + expected_analysis
    result = {
        "checked_at": utc_now(),
        "free_bytes": free,
        "expected_raw_bytes": expected_raw,
        "expected_analysis_and_package_bytes": expected_analysis,
        "required_free_bytes": required,
        "passed": free >= required,
    }
    atomic_write_json(root / "status/disk_check.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Insufficient persistent disk: {result}")
    return result


def _prestage_environment_manifest(
    root: Path, environment: dict[str, Any], assets: dict[str, Any]
) -> dict[str, Any]:
    models = load_json(root / "configs/resolved_models_v23.json")["models"]
    config = load_json(root / "configs/experiment_v23.json")
    return {
        "schema_version": "pier_v23_environment_manifest_prestage_v1",
        "created_at": utc_now(),
        "cpu_environment": environment,
        "models": [
            {
                "id": row["id"],
                "model_revision": row["revision"],
                "tokenizer_revision": row["tokenizer_revision"],
                "chat_template_sha256": row["v23_chat_template_sha256"],
                "rendered_prompt_sha256": row["v23_rendered_prompt_sha256"],
            }
            for row in models
        ],
        "score_manifest_sha256": assets["prompts"]["score_manifest_sha256"],
        "generation_manifest_sha256": assets["prompts"]["generation_manifest_sha256"],
        "seeds": {"splits": config["split_seeds"], "bootstrap": config["bootstrap_seed"]},
        "formal_dtype": config["dtype"],
    }


def _ready_marker(
    root: Path,
    environment: dict[str, Any],
    inventory: dict[str, Any],
    assets: dict[str, Any],
) -> dict[str, Any]:
    prompts = assets["prompts"]
    return {
        "ready": True,
        "schema_version": "pier_v23_b200_ready_stage2_v1",
        "created_at": utc_now(),
        "project_root": str(LOGICAL_ROOT),
        "model_count": 8,
        "base_question_count": prompts["base_question_count"],
        "score_semantic_condition_count": prompts["semantic_condition_count"],
        "score_rotation_count_policy": "all cyclic rotations",
        "score_all_rotation_rows_per_model": prompts["score_all_rotation_rows_per_model"],
        "score_rotation0_reused_rows_per_model": prompts[
            "score_rotation0_reused_rows_per_model"
        ],
        "score_new_inference_rows_per_model": prompts["score_new_inference_rows_per_model"],
        "generation_question_count": prompts["generation_question_count"],
        "generation_semantic_condition_count": prompts[
            "generation_semantic_condition_count"
        ],
        "generation_rotation_count_policy": "all cyclic rotations",
        "generation_rows_per_model": prompts["generation_rows_per_model"],
        "source_v2_sha256": inventory["source_v2_sha256"],
        "source_v22_sha256": inventory["source_v22_sha256"],
        "score_manifest_sha256": prompts["score_manifest_sha256"],
        "score_inference_manifest_sha256": prompts["score_inference_manifest_sha256"],
        "generation_manifest_sha256": prompts["generation_manifest_sha256"],
        "source_tree_sha256": tree_sha256(root / "src"),
        "environment_path": str(LOGICAL_ROOT / "env/venv"),
        "environment_lock_sha256": environment["source_lock_sha256"],
        "formal_b200_launcher": str(LOGICAL_ROOT / "scripts/run_b200_interface_v23.sh"),
        "formal_cpu_postprocess_launcher": str(
            LOGICAL_ROOT / "scripts/run_cpu_postprocess_v23.sh"
        ),
        "b200_launcher_sha256": sha256_file(root / "scripts/run_b200_interface_v23.sh"),
        "cpu_postprocess_launcher_sha256": sha256_file(
            root / "scripts/run_cpu_postprocess_v23.sh"
        ),
        "resolved_models_sha256": assets["models"]["resolved_models_sha256"],
        "all_tests_passed": True,
    }


def run(root: Path) -> dict[str, Any]:
    for relative in REQUIRED_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    ready_path = root / "status/B200_READY_STAGE2.json"
    if ready_path.exists():
        raise RuntimeError("B200_READY_STAGE2.json already exists; refusing to overwrite readiness")
    status_path = root / "status/prestage_cpu_v23.json"
    status: dict[str, Any] = {
        "schema_version": "pier_v23_cpu_prestage_v1",
        "started_at": utc_now(),
        "complete": False,
        "stage": "resolve_sources_and_environment",
    }
    atomic_write_json(status_path, status)
    try:
        resolve_roots(root)
        environment = validate_environment(root)
        status["stage"] = "immutable_input_inventory"
        atomic_write_json(status_path, status)
        inventory = record_before(root)
        status["stage"] = "prompt_manifests_model_caches_and_rendering"
        atomic_write_json(status_path, status)
        assets = prepare_all(root)
        status["stage"] = "synthetic_controls"
        atomic_write_json(status_path, status)
        controls = run_synthetic_controls()
        atomic_write_json(root / "status/synthetic_controls.json", controls)
        if not controls["passed"]:
            raise RuntimeError(f"Synthetic controls failed: {controls}")
        status["stage"] = "static_and_unit_checks"
        atomic_write_json(status_path, status)
        checks = run_static_checks(root)
        status["stage"] = "real_cpu_smoke"
        atomic_write_json(status_path, status)
        smoke = run_smoke(root)
        status["stage"] = "disk_and_launcher_gate"
        atomic_write_json(status_path, status)
        disk = disk_check(root, assets["prompts"])
        manifest = _prestage_environment_manifest(root, environment, assets)
        atomic_write_json(root / "manifests/environment_manifest_prestage.json", manifest)
        ready = _ready_marker(root, environment, inventory, assets)
        if not all(
            [
                assets["passed"],
                controls["passed"],
                checks["passed"],
                smoke["passed"],
                disk["passed"],
            ]
        ):
            raise AssertionError("A prestage gate was false")
        atomic_write_json(ready_path, ready)
        status.update(
            {
                "complete": True,
                "stage": "complete",
                "completed_at": utc_now(),
                "readiness_marker": str(ready_path),
                "readiness_marker_sha256": sha256_file(ready_path),
            }
        )
        atomic_write_json(status_path, status)
        return ready
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
    args = parser.parse_args()
    result = run(args.root.resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("========================================================", flush=True)
    print("B200_READY_STAGE2", flush=True)
    print("", flush=True)
    print("CPU prestage is complete.", flush=True)
    print("Do not edit code on B200.", flush=True)
    print("", flush=True)
    print("Open the 8×B200 host and run:", flush=True)
    print("", flush=True)
    print("tmux new -d -s pier_llm_v23_b200 \\", flush=True)
    print('  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION && \\', flush=True)
    print("   bash scripts/run_b200_interface_v23.sh --resume \\", flush=True)
    print('   2>&1 | tee -a logs/b200_interface_v23.log"', flush=True)
    print("========================================================", flush=True)


if __name__ == "__main__":
    main()
