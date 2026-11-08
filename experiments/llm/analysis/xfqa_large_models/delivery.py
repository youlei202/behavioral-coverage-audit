"""Create and verify a portable XfqA CPU-replay archive after empirical checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

try:
    from .reporting import sha256, write_json, write_text
except ImportError:
    from reporting import sha256, write_json, write_text


EXCLUDED_DIRECTORIES = {"cache", "model_cache", "models", "runtime", ".venv", "venv", ".git", "__pycache__", ".pytest_cache", "hf_cache"}
EXCLUDED_FILENAMES = {"token", "stored_tokens", "hosts.yml", ".env", ".git-credentials", ".netrc"}
SOURCE_ROOTS = (
    "experiments/llm/analysis/xfqa_large_models",
    "experiments/llm/analysis/xwvg_reference_selection",
    "experiments/llm/analysis/v23",
    "experiments/llm/inference/v2",
)
RUNTIME_METADATA = (
    "analysis_requirements.lock.txt", "scoring_requirements.lock.txt",
    "analysis_environment.json", "scoring_environment.json", "placement.json",
    "initial_resource_observations.json",
)
REPLAY_SCRIPT = r'''#!/usr/bin/env python3
"""Verify every bundled file and replay recorded XfqA metrics on CPU.

No network, model weights, original absolute paths or original Git checkout are
required. --check-only validates archive identities and complete input counts;
the default additionally replays raw averaging, weights and uncertainty evidence.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit-output', type=Path)
    args=parser.parse_args()
    bundle=Path(__file__).resolve().parent
    manifest=json.loads((bundle/'DELIVERY_MANIFEST.json').read_text())
    if manifest['status']!='verified_complete_empirical_analysis':
        raise ValueError('Bundle is not marked as a completed measured experiment')
    seen=set()
    for row in manifest['files']:
        relative=Path(row['archive_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen:
            raise ValueError('Unsafe or duplicated manifest path')
        seen.add(relative.as_posix())
        path=bundle/relative
        if path.is_symlink() or not path.resolve().is_relative_to(bundle) or not path.is_file():
            raise ValueError('Missing or unsafe bundled file: '+str(relative))
        if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            raise ValueError('Bundled file checksum mismatch: '+str(relative))
    import pyarrow.parquet as pq
    root=bundle/'run'
    for filename, rows in [('data/new_rotation_scores.parquet',110838),
                           ('data/new_permutation_averaged_semantic_scores.parquet',11760),
                           ('data/permutation_averaged_semantic_scores.parquet',43120),
                           ('analysis/point_fit_coefficients.parquet',5840),
                           ('analysis/per_question_losses.parquet',6540800)]:
        if pq.read_metadata(root/filename).num_rows!=rows:
            raise ValueError('Incomplete empirical input: '+filename)
    proof=json.loads((root/'validation/numerical_replay.json').read_text())
    if proof['status']!='passed' or proof['input_fingerprint']!=manifest['input_fingerprint']:
        raise ValueError('Completed independent numerical audit identity mismatch')
    print(json.dumps({'bundle_hashes':'passed','verified_files':len(seen),'input_counts':'passed',
                      'scope':'portable input and integrity check' if args.check_only else 'full independent CPU numerical replay'}),flush=True)
    if args.check_only:
        return 0
    code=bundle/'code/experiments/llm/analysis/xfqa_large_models'
    command=[sys.executable,str(code/'verify.py'),'--repo-root',str(bundle/'code'),
             '--plan-root',str(root/'plan'),'--output-root',str(root),'--portable']
    if args.audit_output:
        destination=args.audit_output.expanduser().absolute()
        if destination.is_relative_to(bundle):
            raise ValueError('Write replay audit outside the immutable extracted bundle')
        command+=['--audit-output',str(destination)]
    return subprocess.call(command)

if __name__=='__main__':
    raise SystemExit(main())
'''


def _reject_name(relative):
    return (any(p in EXCLUDED_DIRECTORIES for p in relative.parts) or
            relative.name in EXCLUDED_FILENAMES or relative.name.startswith("id_ed25519") or
            relative.name.startswith("id_rsa") or relative.suffix in {".safetensors", ".bin", ".pt", ".pth", ".pyc", ".tmp", ".lock"})


def collect_files(repo, root, archive_name):
    """Use only the run and explicit source roots; never traverse an account home."""
    selected = {}
    omitted = []
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        rel_parent = parent.relative_to(root)
        keep_directories = []
        for name in sorted(names):
            child = parent / name
            relative = rel_parent / name
            if child.is_symlink():
                omitted.append({"path": str(relative), "reason": "symlink directory excluded; no external traversal"})
            elif name in EXCLUDED_DIRECTORIES:
                omitted.append({"path": str(relative), "reason": "weights, download cache, environment or generated interpreter cache; identities retained in provenance"})
            else:
                keep_directories.append(name)
        names[:] = keep_directories
        for name in sorted(files):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                omitted.append({"path": str(relative), "reason": "external or link-based input excluded; actual portable inputs must be regular files"})
                continue
            if _reject_name(relative) or name in {archive_name, archive_name + ".sha256"}:
                continue
            if relative.as_posix() in {"status/package_complete.json", "status/complete.json"}:
                continue  # Avoid a circular archive identity; include the validated gate snapshot below.
            if name.startswith(".") or ".tmp" in name:
                continue
            if "rendered" in relative.parts and path.suffix == ".jsonl":
                omitted.append({"path": str(relative), "reason": "bulk rendered prompt text omitted; native-template boundaries and exact hashes retained",
                                "bytes": path.stat().st_size, "sha256": sha256(path)})
                continue
            selected["run/" + relative.as_posix()] = path
    # Runtime binaries/environments stay excluded, but exact environment locks
    # and the frozen source snapshots are required reproducibility evidence.
    for name in RUNTIME_METADATA:
        path = root / "runtime" / name
        if path.is_file() and not path.is_symlink():
            selected["run/runtime/" + name] = path
    for name in ("frozen_analysis", "frozen_scoring"):
        for path in sorted((root / "runtime" / name).rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in {".py", ".json", ".txt", ".md"}:
                selected["run/" + path.relative_to(root).as_posix()] = path
    for relative in SOURCE_ROOTS:
        source = repo / relative
        if not source.is_dir():
            raise FileNotFoundError("Source dependency missing: " + relative)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink() or _reject_name(path.relative_to(source)):
                continue
            selected["code/" + path.relative_to(repo).as_posix()] = path
    required = ["run/inputs/mmlu_pro_selected_560.jsonl", "run/inputs/permutation_averaged_semantic_scores.parquet",
                "run/data/new_rotation_scores.parquet", "run/data/new_permutation_averaged_semantic_scores.parquet",
                "run/data/permutation_averaged_semantic_scores.parquet", "run/data/permutations/score_execution_manifest.jsonl",
                "run/model_lock.json", "run/plan/config/experiment.json", "run/analysis/point_fit_coefficients.parquet",
                "run/analysis/per_question_losses.parquet", "run/analysis/split_ids.json",
                "run/analysis/primary_refit_intervals.csv", "run/analysis/secondary_descriptive_intervals.csv",
                "run/validation/numerical_replay.json", "run/RUN_COMMANDS.sh", "run/resource_summary.json", "run/resource_summary.csv",
                "run/runtime/analysis_requirements.lock.txt", "run/runtime/scoring_requirements.lock.txt",
                "run/runtime/analysis_environment.json", "run/runtime/scoring_environment.json",
                "run/runtime/initial_resource_observations.json"]
    for name in required:
        if name not in selected:
            raise FileNotFoundError("Required portable artifact missing or excluded: " + name)
    return selected, omitted


def check_gates(root):
    replay_path = root / "validation/numerical_replay.json"
    report_path = root / "status/report_complete.json"
    replay, report = (json.loads(p.read_text()) for p in (replay_path, report_path))
    if replay.get("status") != "passed" or report.get("status") != "complete":
        raise ValueError("Delivery requires passed independent replay and completed report")
    fingerprint = replay["input_fingerprint"]
    if report["input_fingerprint"] != fingerprint:
        raise ValueError("Delivery report and numerical replay identities differ")
    analysis = root / "analysis"
    states = {}
    for name in ("point_complete.json", "primary_refit_complete.json", "secondary_complete.json", "bootstrap_complete.json"):
        record = json.loads((analysis / name).read_text())
        if record.get("status") != "complete" or record.get("input_fingerprint") != fingerprint:
            raise ValueError("Delivery requires completed identity-matched stage: " + name)
        for filename, digest in record.get("artifact_hashes", {}).items():
            if sha256(analysis / filename) != digest:
                raise ValueError("Delivery stage artifact changed: " + filename)
        states[name] = {k: record[k] for k in ("status", "input_fingerprint", "point_fit_count", "replicates", "fits", "contrast_count") if k in record}
    if states["point_complete.json"].get("point_fit_count") != 5840 or states["primary_refit_complete.json"].get("replicates") != 1000 or states["primary_refit_complete.json"].get("fits") != 100000 or states["secondary_complete.json"].get("replicates") != 2000:
        raise ValueError("Incomplete prescribed fit or resampling workload")
    for record in report["artifacts"]:
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("Report artifact changed before delivery: " + record["path"])
    assembly = json.loads((root / "status/assembly_complete.json").read_text())
    if assembly.get("status") != "complete" or assembly.get("real_three_model_scoring_complete") is not True:
        raise ValueError("Real three-model scoring and assembly are incomplete")
    for name, expected in (("new_rotation_rows", 110838), ("new_averaged_vectors", 11760), ("merged_averaged_vectors", 43120)):
        if assembly.get(name) != expected:
            raise ValueError("Delivery scoring count differs: " + name)
    for record in assembly["artifacts"].values():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("Assembled scoring artifact changed before delivery")
    return fingerprint, states, sha256(replay_path), sha256(report_path)


def run_package(repo_root, plan_root, output_root, resume=False):
    repo, plan, root = map(lambda p: Path(p).resolve(), (repo_root, plan_root, output_root))
    if plan != root / "plan":
        raise ValueError("Portable packaging expects the byte-verified plan at output_root/plan")
    fingerprint, states, replay_sha, report_sha = check_gates(root)
    archive_name = f"XfqA_Large_Model_Results_{root.name}.zip"
    archive = root / archive_name
    marker = root / "status/package_complete.json"
    if marker.exists():
        existing = json.loads(marker.read_text())
        if not resume or existing.get("input_fingerprint") != fingerprint or existing.get("numerical_replay_sha256") != replay_sha or existing.get("report_complete_sha256") != report_sha:
            raise ValueError("Existing archive publication identity differs; do not rewrite it")
        if sha256(archive) != existing["archive_sha256"] or existing.get("status") != "verified":
            raise ValueError("Published archive changed or lacks verification")
        return existing
    files, omitted = collect_files(repo, root, archive_name)
    inventory = [{"archive_path": name, "sha256": sha256(path), "bytes": path.stat().st_size}
                 for name, path in sorted(files.items())]
    bundle_name = "XfqA_Large_Model_Results"
    manifest = {"schema": "xfqa_delivery_manifest_v1", "status": "verified_complete_empirical_analysis",
                "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": root.name,
                "input_fingerprint": fingerprint, "numerical_replay_sha256": replay_sha, "report_complete_sha256": report_sha,
                "validated_stage_snapshot": states,
                "completion_scope": "All three real new models, the unchanged eight-model seed, all point analyses, prescribed refit/descriptive uncertainty and independent replay; archive verification follows creation.",
                "files": inventory, "omissions": omitted,
                "model_weights_included": False, "credentials_included": False,
                "anonymous_public_release": False,
                "privacy_note": "Private reproducibility bundle retains original machine/user paths in provenance; sanitize before a separate anonymous public release.",
                "replay": {"command": "python replay_cpu.py --audit-output /absolute/path/outside/bundle/replay_audit.json",
                           "input_check_command": "python replay_cpu.py --check-only",
                           "scope": "Checks all bundled hashes, then independently replays scoring aggregation, saved point/refit coefficients, all metrics and uncertainty; no model inference or optimizer rerun.",
                           "dependencies": "Use the recorded CPU analysis environment; numpy, pandas and pyarrow are required."}}
    readme = ("# XfqA larger-model results\n\n"
              "The measured-results report, response and figures are under `run/analysis/`. "
              "Inputs, raw rotation scores, fitted coefficients, question-level losses, bootstrap evidence "
              "and exact provenance are included. Model weights and rendered-text bulk are omitted as "
              "recorded in `DELIVERY_MANIFEST.json`.\n\n"
              "Use the recorded CPU environment. From this extracted directory, run:\n\n"
              "```bash\npython replay_cpu.py --check-only\n"
              "python replay_cpu.py --audit-output /absolute/path/outside/bundle/replay_audit.json\n```\n\n"
              "The default replay checks every bundled hash and then independently reconstructs metrics "
              "from saved probabilities and coefficients, including the primary weighted-refit evidence "
              "and descriptive resamples. It requires no historical absolute paths, model weights, "
              "network credentials or original Git checkout. It does not rerun model inference or refit "
              "the saved optimizers. Original external-paper/XWVG preservation was checked before packing; "
              "its audit is retained rather than requiring those external files on another machine.\n\n"
              "This is a private reproducibility bundle. Provenance contains original user, host and path "
              "metadata and has not been sanitized for an anonymous public release.\n")
    extra = {"DELIVERY_MANIFEST.json": json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
             "replay_cpu.py": REPLAY_SCRIPT, "README.md": readme}
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for name, path in sorted(files.items()):
            z.write(path, bundle_name + "/" + name)
        for name, content in extra.items():
            z.writestr(bundle_name + "/" + name, content)
    extraction = Path(tempfile.mkdtemp(prefix="xfqa-delivery-check-"))
    with zipfile.ZipFile(temporary) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entry")
        for info in z.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe ZIP entry")
        bad = z.testzip()
        if bad:
            raise ValueError("ZIP CRC failure: " + bad)
        z.extractall(extraction)
    bundle = extraction / bundle_name
    process = subprocess.run([sys.executable, str(bundle / "replay_cpu.py"), "--check-only"],
                             cwd=extraction, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError("Relocated portable replay input-check failed: " + process.stderr[-3000:])
    portable_audit_path = extraction / "portable_numerical_replay.json"
    print("Running complete independent CPU numerical replay from extracted XfqA bundle", flush=True)
    replay_started = time.monotonic()
    full_replay = subprocess.run([sys.executable, str(bundle / "replay_cpu.py"), "--audit-output", str(portable_audit_path)],
                                cwd=extraction, capture_output=True, text=True)
    full_replay_seconds = time.monotonic() - replay_started
    if full_replay.returncode:
        raise RuntimeError("Relocated complete CPU replay failed: " + full_replay.stderr[-4000:])
    portable_audit = json.loads(portable_audit_path.read_text())
    if portable_audit.get("status") != "passed" or portable_audit.get("input_fingerprint") != fingerprint or portable_audit.get("portable") is not True:
        raise ValueError("Relocated complete CPU replay returned a mismatched or incomplete audit")
    if archive.exists():
        raise FileExistsError("Refusing to replace an existing unverified archive: " + str(archive))
    os.link(temporary, archive)
    temporary.unlink()
    digest = sha256(archive)
    write_text(archive.with_suffix(".zip.sha256"), digest + "  " + archive.name + "\n")
    retained_portable_audit = root / "validation/portable_numerical_replay.json"
    write_text(retained_portable_audit, portable_audit_path.read_text())
    result = {"schema": "xfqa_delivery_v1", "status": "verified", "input_fingerprint": fingerprint,
              "numerical_replay_sha256": replay_sha, "report_complete_sha256": report_sha,
              "archive_path": str(archive), "archive_sha256": digest, "archive_bytes": archive.stat().st_size,
              "manifest_files": len(inventory), "zip_crc_verified": True, "bundle_hashes_verified": True,
              "portable_replay_verified": True,
              "portable_verification_scope": "Relocated default replay verified all bundled hashes, raw scoring aggregation, all point/refit coefficients, metrics and uncertainty using only files inside the extracted bundle; no model inference or optimizer fit was rerun.",
              "portable_full_replay_wall_seconds": full_replay_seconds,
              "portable_numerical_replay_path": str(retained_portable_audit),
              "portable_numerical_replay_sha256": sha256(retained_portable_audit),
              "portable_check_output": process.stdout.strip(), "extraction_test_path": str(bundle),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(marker, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "plan-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_package(args.repo_root, args.plan_root, args.output_root, args.resume), indent=2))
