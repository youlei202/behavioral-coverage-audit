#!/usr/bin/env python3
"""Check source identity and balanced-cache availability without legacy imports.

Exit 0: input files exist; Parquet row validation and provenance review remain.
Exit 2: averaged input not ready; reconstruction or missing inputs must be resolved.
Exit 3: inspected source or historically identified input differs; review the difference.
Never invokes model inference, old analysis entry points, or new empirical fits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
V2_NAME = "PIER_LLM_ECOSYSTEM_GOLDMINE_V2"
V23_NAME = "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION"
AVERAGE = Path("outputs/analysis/permutation_averaged_semantic_scores.parquet")
ROTATIONS = Path("outputs/validation/permutation_wise_semantic_scores.parquet")
QUESTIONS = Path("data/mmlu_pro_selected_560.jsonl")
QUESTION_SHA256 = "9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6"
QUESTION_BYTES = 458957


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def candidates(raw_root, run_name, relative, override=None):
    """Inspect explicit paths and the named runs under/beside the given root."""
    if override is not None:
        return [override.expanduser().resolve()]
    if raw_root is None:
        return []
    raw = raw_root.expanduser().resolve()
    possible = [raw / run_name / relative, raw / relative]
    if raw.name in {V2_NAME, V23_NAME}:
        possible.append(raw.parent / run_name / relative)
    return list(dict.fromkeys(possible))


def first_file(paths):
    return next((p for p in paths if p.is_file()), None)


def inspect_file(path, role, expected_hash=None, expected_bytes=None):
    actual_hash, actual_size = sha256(path), path.stat().st_size
    record = {"role": role, "path": str(path), "size_bytes": actual_size,
              "sha256": actual_hash, "historical_identity_verified": False}
    if expected_hash is None:
        record["status"] = "present_current_hash_recorded"
        record["provenance_note"] = (
            "A current digest is not proof of equality to the historical cache. "
            "Review original-run provenance and validate rows before fitting.")
    else:
        good = actual_hash == expected_hash and (
            expected_bytes is None or actual_size == expected_bytes)
        record.update(status="historical_match" if good else "historical_mismatch",
                      expected_sha256=expected_hash, expected_size_bytes=expected_bytes,
                      historical_identity_verified=good)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path,
                        default=os.environ.get("BEHAVIORAL_COVERAGE_RAW_DATA"),
                        help="Original V2/V23 run or their parent; no scores are copied")
    parser.add_argument("--balanced-cache", type=Path,
                        help="Explicit path to the averaged full-vector Parquet")
    parser.add_argument("--rotations-cache", type=Path,
                        help="Explicit path to the complete saved rotation Parquet")
    parser.add_argument("--questions", type=Path,
                        help="Explicit path to the original selected-560 JSONL")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    report = {
        "schema_version": "bca_xwvg_preflight_v2",
        "primary_protocol": "balanced_full_vector_endpoints",
        "repo_root": str(root),
        "empirical_experiment_run": False,
        "parquet_row_validation_run": False,
        "model_inference_run": False,
        "source_checks": [], "input_checks": [], "missing": [], "mismatch": [],
        "notes": [
            "The public archive contains fitted summaries, not the required probability cache.",
            "An available averaged cache still needs row-level and provenance validation.",
            "Canonical V2 scores are not an automatic primary-protocol fallback."
        ],
    }
    manifest = json.loads((PACKAGE / "evidence/source_manifest.json").read_text())
    for record in manifest["inspected_files"]:
        path = root / record["path"]
        actual = sha256(path) if path.is_file() else None
        status = "match" if actual == record["sha256"] else "missing" if actual is None else "changed"
        report["source_checks"].append({"path": record["path"], "status": status,
            "expected_sha256": record["sha256"], "actual_sha256": actual})
        if status != "match":
            report["mismatch"].append("Inspected source " + status + ": " + record["path"])

    average_paths = candidates(args.raw_root, V23_NAME, AVERAGE, args.balanced_cache)
    rotation_paths = candidates(args.raw_root, V23_NAME, ROTATIONS, args.rotations_cache)
    question_paths = candidates(args.raw_root, V2_NAME, QUESTIONS, args.questions)
    report["searched_paths"] = {
        "balanced_cache": list(map(str, average_paths)),
        "rotation_cache": list(map(str, rotation_paths)),
        "selected_questions": list(map(str, question_paths)),
    }
    average, rotations, questions = map(first_file, (average_paths, rotation_paths, question_paths))
    report["expected_paths"] = {
        "balanced_cache": str(Path(V23_NAME) / AVERAGE),
        "rotation_cache_if_rebuilding": str(Path(V23_NAME) / ROTATIONS),
        "selected_questions": str(Path(V2_NAME) / QUESTIONS),
    }
    if questions is None:
        report["missing"].append("Original selected-560 JSONL: " + str(Path(V2_NAME) / QUESTIONS))
    else:
        check = inspect_file(questions, "selected_questions", QUESTION_SHA256, QUESTION_BYTES)
        report["input_checks"].append(check)
        if check["status"] == "historical_mismatch":
            report["mismatch"].append("Selected-question file differs from archived digest/size")

    if average is not None:
        report["input_route"] = "saved_balanced_full_vector_average"
        report["input_checks"].append(inspect_file(average, "balanced_average"))
    elif rotations is not None:
        report["input_route"] = "saved_rotations_require_cpu_aggregation"
        report["input_checks"].append(inspect_file(rotations, "semantic_rotations"))
        report["missing"].append("Averaged cache must be rebuilt into the new run directory")
        report["notes"].append(
            "Verify every model/question/condition has rotations 0..option_count-1, "
            "then average semantic_remapped_probabilities. No new inference is needed "
            "if these records are complete and valid.")
    else:
        report["input_route"] = "primary_cache_unavailable"
        report["missing"].append("Balanced full-vector average or complete saved semantic rotations")
        report["notes"].append(
            "If only shards remain, follow the data map: V23 new rotations, V2 rotation 0, "
            "and score_semantic_manifest.jsonl are needed; inspect sidecars and mappings.")

    report["can_attempt_cpu_reconstruction"] = rotations is not None and average is None
    if report["mismatch"]:
        report["status"], code = "source_or_input_mismatch", 3
    elif average is not None and questions is not None:
        report["status"], code = "files_available_validation_required", 0
    elif rotations is not None and questions is not None:
        report["status"], code = "rebuild_balanced_cache_from_rotations_required", 2
    else:
        report["status"], code = "blocked_missing_primary_inputs", 2
    if args.raw_root is None and not (args.balanced_cache or args.rotations_cache):
        report["notes"].append(
            "Set --raw-root or BEHAVIORAL_COVERAGE_RAW_DATA, or supply explicit cache/question paths.")
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "missing_count": len(report["missing"]),
                     "mismatch_count": len(report["mismatch"]), "report": str(destination)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

