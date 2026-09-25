from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import data_prep, model_prestage, smoke
from .analysis import run_synthetic_controls
from .solver import fit_simplex_projection, fit_simplex_slsqp
from .utils import (
    RUNBOOK_ROOT,
    atomic_write_json,
    project_root,
    sha256_file,
    stable_u64,
    tree_sha256,
    utc_now,
)

DIRECTORIES = [
    "artifacts",
    "configs",
    "data/interventions",
    "data/manifests/models",
    "data/manifests/rendered_prompts",
    "env/wheelhouse",
    "hf_cache",
    "logs",
    "outputs/analysis",
    "outputs/controls",
    "outputs/figures",
    "outputs/generation_validation",
    "outputs/raw_scores",
    "outputs/tables",
    "scripts",
    "src",
    "status",
    "tests",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_checked(command: list[str], root: Path) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    record = {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout,
    }
    if result.returncode:
        raise RuntimeError(
            f"Check failed ({result.returncode}): {' '.join(command)}\n{result.stdout[-8000:]}"
        )
    return record


def validate_environment(root: Path) -> dict[str, Any]:
    import torch

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Stage A requires Python 3.11, found {sys.version}")
    if torch.version.cuda is None:
        raise RuntimeError("Pinned PyTorch is not a CUDA wheel")
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
        "osqp",
        "clarabel",
        "matplotlib",
    ]
    versions = {package: importlib.metadata.version(package) for package in packages}
    wheel_count = len(list((root / "env" / "wheelhouse").glob("*.whl")))
    if wheel_count < len(packages):
        raise RuntimeError(f"Offline wheelhouse appears incomplete: only {wheel_count} wheels")
    lock_path = root / "env" / "requirements.lock.txt"
    if not lock_path.is_file() or lock_path.stat().st_size == 0:
        raise RuntimeError("Exact environment lock is absent")
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "versions": versions,
        "torch_cuda_build": torch.version.cuda,
        "wheel_count": wheel_count,
        "lock_sha256": sha256_file(lock_path),
    }
    atomic_write_json(root / "status" / "environment_status.json", result)
    return result


def validate_solver(root: Path, problem_count: int = 100) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for problem_index in range(problem_count):
        rng = np.random.default_rng(stable_u64(20260828, "solver", problem_index) % (2**32))
        real_shaped = problem_index >= problem_count // 2
        examples = 1400 if real_shaped else int(rng.integers(30, 180))
        peers = 7 if real_shaped else int(rng.integers(2, 9))
        design = rng.normal(size=(examples, peers))
        truth_weights = rng.dirichlet(np.ones(peers))
        if problem_index % 10 == 0:
            truth_weights[:] = 0
            truth_weights[0] = 1
        elif problem_index % 10 == 1:
            truth_weights[:] = 0
            truth_weights[:2] = [0.75, 0.25]
        if problem_index % 10 == 2 and peers >= 3:
            design[:, -1] = design[:, 0]
        target = design @ truth_weights
        if problem_index % 4:
            target += rng.normal(scale=0.02, size=examples)
        primary = fit_simplex_projection(design, target)
        independent_weights, independent_objective, independent = fit_simplex_slsqp(
            design, target
        )
        objective_difference = abs(primary.stage1_objective - independent_objective)
        permitted = max(5e-9, 1e-6 * max(1.0, primary.stage1_objective))
        passed = (
            primary.minimum_weight >= -1e-9
            and primary.weight_sum_error <= 1e-8
            and objective_difference <= permitted
        )
        rows.append(
            {
                "problem_index": problem_index,
                "real_shaped": real_shaped,
                "examples": examples,
                "peers": peers,
                "objective_difference": objective_difference,
                "permitted_objective_difference": permitted,
                "primary_weights": primary.weights.tolist(),
                "independent_weights": independent_weights.tolist(),
                "primary": primary.diagnostics(),
                "independent": independent,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"Solver validation failed on problem {problem_index}: {rows[-1]}")
    controls, control_summary = run_synthetic_controls()
    if not control_summary["passed"]:
        raise RuntimeError(f"Synthetic solver controls failed: {control_summary}")
    result = {
        "schema_version": "pier_solver_validation_v2",
        "timestamp": utc_now(),
        "problem_count": problem_count,
        "real_shaped_problem_count": sum(row["real_shaped"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "problems": rows,
        "controls": controls.to_dict(orient="records"),
    }
    atomic_write_json(root / "status" / "solver_validation.json", result)
    return result


def validate_data_determinism(root: Path) -> dict[str, Any]:
    first = data_prep.prepare(root)
    first_hashes = dict(first["hashes"])
    second = data_prep.prepare(root)
    if first_hashes != second["hashes"]:
        raise RuntimeError("Deterministic data regeneration changed one or more manifest hashes")
    result = {
        "passed": True,
        "dataset_revision": first["dataset_revision"],
        "hashes": first_hashes,
        "validation": first["validation"],
    }
    atomic_write_json(root / "status" / "data_validation.json", result)
    return result


def validated_data_or_regenerate(root: Path) -> dict[str, Any]:
    status_path = root / "status" / "data_validation.json"
    if status_path.exists():
        existing = _load_json(status_path)
        hashes = existing.get("hashes", {})
        if existing.get("passed") and hashes and all(
            (root / relative).is_file() and sha256_file(root / relative) == digest
            for relative, digest in hashes.items()
        ):
            print("[resume] validated deterministic data artifacts", flush=True)
            return existing
    return validate_data_determinism(root)


def run_static_and_functional_checks(root: Path) -> dict[str, Any]:
    python = str(Path(sys.executable))
    commands = [
        ["bash", "-n", *[str(path) for path in sorted((root / "scripts").glob("*.sh"))]],
        [python, "-m", "compileall", "-q", "src", "tests"],
        [python, "-m", "ruff", "check", "src", "tests"],
        [python, "-m", "mypy", "src/pier_llm"],
        [python, "-m", "pytest", "-q"],
    ]
    if shutil.which("shellcheck"):
        commands.insert(
            1, ["shellcheck", *[str(path) for path in sorted((root / "scripts").glob("*.sh"))]]
        )
    records = [_run_checked(command, root) for command in commands]
    result = {"passed": True, "checks": records}
    atomic_write_json(root / "status" / "static_functional_checks.json", result)
    return result


def disk_check(root: Path, resolved: dict[str, Any]) -> dict[str, Any]:
    cached_model_bytes = int(sum(row["weight_bytes"] for row in resolved["models"]))
    dataset_bytes = sum(
        path.stat().st_size for path in (root / "data").rglob("*") if path.is_file()
    )
    config = _load_json(root / "configs" / "experiment.json")
    model_count = len(resolved["models"])
    score_rows = model_count * int(config["scoring_prompts_per_model"])
    generation_rows = model_count * int(config["generation_prompts_per_model"])
    bytes_per_score_row = 900
    bytes_per_generation_row = 700
    expected_raw = score_rows * bytes_per_score_row + generation_rows * bytes_per_generation_row
    expected_temporary = max(int(config["shard_size"]) * bytes_per_score_row * model_count, 1)
    expected_final = int(expected_raw * 1.75 + 2 * 1024**3)
    free = shutil.disk_usage(root).free
    required_beyond_cache = 2 * expected_temporary + expected_final
    passed = free >= required_beyond_cache
    result = {
        "timestamp": utc_now(),
        "cached_model_bytes": cached_model_bytes,
        "dataset_bytes": dataset_bytes,
        "expected_raw_score_bytes": expected_raw,
        "expected_temporary_shard_bytes": expected_temporary,
        "expected_final_artifact_bytes": expected_final,
        "free_bytes_remaining": free,
        "required_free_bytes": required_beyond_cache,
        "passed": passed,
    }
    atomic_write_json(root / "status" / "disk_check.json", result)
    if not passed:
        raise RuntimeError(f"Disk check failed: {result}")
    return result


def _readiness(
    root: Path,
    resolved: dict[str, Any],
    data_status: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    prompt_path = root / "data" / "manifests" / "scoring_prompts.jsonl"
    resolved_path = root / "configs" / "resolved_models.json"
    config = _load_json(root / "configs" / "experiment.json")
    return {
        "ready": True,
        "timestamp": utc_now(),
        "project_root": str(RUNBOOK_ROOT),
        "persistent_project_root": str(root.resolve()),
        "resolved_model_count": resolved["resolved_model_count"],
        "broad_lineage_count": resolved["broad_lineage_count"],
        "controlled_sibling_pair_count": resolved["controlled_sibling_pair_count"],
        "base_question_count": data_status["validation"]["base_question_count"],
        "scoring_prompts_per_model": config["scoring_prompts_per_model"],
        "generation_prompts_per_model": config["generation_prompts_per_model"],
        "prompt_manifest_sha256": sha256_file(prompt_path),
        "source_tree_sha256": tree_sha256(root / "src"),
        "model_manifest_sha256": sha256_file(resolved_path),
        "formal_launcher": str(RUNBOOK_ROOT / "scripts" / "run_b200_inference.sh"),
        "environment_path": str(root / "env" / "venv"),
        "environment_lock_sha256": environment["lock_sha256"],
        "all_tests_passed": True,
        "checksums_ok": True,
    }


def run(root: Path) -> dict[str, Any]:
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    ready_path = root / "status" / "B200_READY.json"
    if ready_path.exists():
        raise RuntimeError(
            "B200_READY.json already exists; refusing to overwrite a prior readiness decision"
        )
    status_path = root / "status" / "prestage_status.json"
    status: dict[str, Any] = {
        "schema_version": "pier_cpu_prestage_v2",
        "timestamp_started": utc_now(),
        "complete": False,
        "stage": "environment",
        "errors": [],
    }
    atomic_write_json(status_path, status)
    try:
        environment = validate_environment(root)
        status["stage"] = "data_determinism"
        atomic_write_json(status_path, status)
        data_status = validated_data_or_regenerate(root)
        status["stage"] = "solver_validation"
        atomic_write_json(status_path, status)
        solver_status = validate_solver(root)
        status["stage"] = "static_and_functional_checks"
        atomic_write_json(status_path, status)
        check_status = run_static_and_functional_checks(root)
        status["stage"] = "real_cpu_smoke"
        atomic_write_json(status_path, status)
        smoke_status = smoke.run(root)
        status["stage"] = "full_model_compatibility"
        atomic_write_json(status_path, status)
        resolved = model_prestage.run(root, resume=True)
        if not resolved["minimum_roster_passed"] or resolved["access_blocked"]:
            raise RuntimeError("Resolved model roster is not eligible for B200 readiness")
        status["stage"] = "disk_check"
        atomic_write_json(status_path, status)
        disk_status = disk_check(root, resolved)
        marker = _readiness(root, resolved, data_status, environment)
        status.update(
            {
                "timestamp_completed": utc_now(),
                "complete": True,
                "stage": "complete",
                "checks": {
                    "environment": environment,
                    "data": data_status,
                    "solver_problem_count": solver_status["problem_count"],
                    "static_functional": check_status["passed"],
                    "real_cpu_smoke": smoke_status["passed"],
                    "resolved_roster": resolved["minimum_roster_passed"],
                    "disk": disk_status["passed"],
                },
            }
        )
        atomic_write_json(status_path, status)
        atomic_write_json(ready_path, marker)
        return marker
    except Exception as exc:
        status.update(
            {
                "timestamp_failed": utc_now(),
                "complete": False,
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
    result = run(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print("================ B200_READY ================")
    print("Open the 8×B200 host. The first formal command is in REPRODUCE.md.")
    print("============================================")


if __name__ == "__main__":
    main()
