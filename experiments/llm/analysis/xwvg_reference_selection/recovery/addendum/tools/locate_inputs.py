#!/usr/bin/env python3
"""Locate possible original XWVG inputs by names and metadata, without extraction."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


TARGETS = {
    "permutation_averaged_semantic_scores.parquet",
    "permutation_wise_semantic_scores.parquet",
    "mmlu_pro_selected_560.jsonl",
}
LEGACY_DIRS = (
    "PIER_LLM_ECOSYSTEM_GOLDMINE_V2",
    "PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION",
)
LEGACY_ROOTS = ("/work/Users/leiyo", "/work/Lei")
KNOWN_ARCHIVES = tuple(name + "_RESULTS.zip" for name in LEGACY_DIRS)
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", ".tox", ".nox",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".huggingface", "huggingface", "hf_cache",
    "hf-cache", "huggingface_cache", "transformers_cache",
    "torch_extensions", "wandb",
}
KNOWN_RELATIVE_INPUTS = (
    "outputs/analysis/permutation_averaged_semantic_scores.parquet",
    "outputs/validation/permutation_wise_semantic_scores.parquet",
    "data/mmlu_pro_selected_560.jsonl",
)


def is_result_zip(name):
    upper = name.upper()
    return upper.endswith(".ZIP") and "PIER" in upper and "RESULTS" in upper


def relevant_zip_member(name):
    # ZIP member names are treated only as metadata. Never extract or open them.
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not parts:
        return False
    basename = parts[-1]
    return (
        basename in TARGETS
        or basename.startswith("score_semantic_manifest")
        or ("raw_scores" in parts and basename.startswith("shard"))
    )


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, bounded search for possible original question/probability "
            "inputs and ZIP names containing PIER and RESULTS (including duplicate "
            "download suffixes). Writes one discovery JSON report. "
            "Names and sizes are evidence of location only; schemas are NOT validated."
        ),
        epilog=(
            "Roots: --repo-root, repeated --root, BEHAVIORAL_COVERAGE_RAW_DATA if set, "
            "/work/Users/leiyo and /work/Lei. Missing/inaccessible roots are recorded. "
            "No downloads, model loading, archive extraction, or full-filesystem search. "
            "Explicit root symlinks are resolved once; internal symlinks are skipped. "
            "An empty result does not establish that data were lost."
        ),
    )
    parser.add_argument("--repo-root", required=True, help="actual existing repository root")
    parser.add_argument("--root", action="append", default=[], help="additional permitted directory (repeatable)")
    parser.add_argument("--out", required=True, help="new JSON report path; existing files are never overwritten")
    parser.add_argument(
        "--max-entries", type=positive_int, default=250000,
        help=("global budget for directory entries plus inspected ZIP member names "
              "(default: 250000); finite direct-path probes are excluded"),
    )
    args = parser.parse_args(argv)
    args.out = Path(os.path.abspath(os.path.expanduser(args.out)))
    if os.path.lexists(args.out):
        parser.error("--out already exists; choose a new report path")
    return args


class Locator:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.used = 0
        self.files = {}
        self.archives = {}
        self.visited_dirs = set()
        self.skip_keys = set()
        self.report = {
            "tool": "locate_inputs.py", "report_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_validated": False,
            "searched_roots": [], "missing_roots": [], "permission_errors": [],
            "other_errors": [], "skipped_entries": [], "direct_probes": [],
            "truncated_scan": False, "truncation_details": [],
            "max_entries": max_entries,
            "entry_counts": {"filesystem": 0, "zip_members": 0},
            "files": [], "archives": [], "zip_errors": [],
            "limitations": [
                "No match does not prove loss: unavailable mounts, permissions, skipped directories and limits leave gaps.",
                "Only listed roots are searched; no parent or filesystem-wide search is performed.",
                "Loose files are inspected by name/type/size only, without reading their contents or validating schemas or hashes.",
                "ZIP central directories are read; member payloads are never read or extracted. A member name does not validate its data.",
                "The entry budget bounds inspected names. Python zipfile loads the central-directory metadata when opening each ZIP.",
                "Internal symlinks and the recorded cache/environment directories are intentionally excluded.",
                "Search reflects this run; files can move or change during or after discovery.",
            ],
        }

    def error(self, path, operation, exc):
        item = {"path": str(path), "operation": operation,
                "error_type": type(exc).__name__, "error": str(exc)}
        key = "permission_errors" if isinstance(exc, PermissionError) else "other_errors"
        self.report[key].append(item)

    def skip(self, path, reason):
        key = (str(path), reason)
        if key not in self.skip_keys:
            self.skip_keys.add(key)
            self.report["skipped_entries"].append({"path": str(path), "reason": reason})

    def truncated(self, path, phase):
        self.report["truncated_scan"] = True
        self.report["truncation_details"].append({"path": str(path), "phase": phase,
                                                   "reason": "max_entries reached"})

    def take(self, kind):
        if self.used >= self.max_entries:
            return False
        self.used += 1
        self.report["entry_counts"][kind] += 1
        return True

    def roots(self, requests):
        found = {}
        for raw, source in requests:
            requested = Path(os.path.abspath(os.path.expanduser(raw)))
            request = {"requested_path": str(requested), "source": source}
            try:
                # Only a deliberately supplied root may resolve through a symlink.
                resolved = requested.resolve(strict=True)
                root_stat = resolved.stat()
                if not stat.S_ISDIR(root_stat.st_mode):
                    self.report["missing_roots"].append(dict(request, reason="not_a_directory"))
                    continue
                if resolved == Path(resolved.anchor):
                    self.skip(resolved, "filesystem_root_refused")
                    self.report["other_errors"].append(dict(request, error="Full-filesystem search is refused."))
                    continue
            except FileNotFoundError:
                self.report["missing_roots"].append(dict(request, reason="not_present_or_mount_unavailable"))
                continue
            except (OSError, RuntimeError) as exc:
                self.error(requested, "resolve_root", exc)
                continue
            key = str(resolved)
            if key in found:
                found[key]["requests"].append(request)
                continue
            record = {"path": key, "requests": [request], "direct_probes_complete": False,
                      "recursive_scan_status": "not_started"}
            found[key] = record
            self.report["searched_roots"].append(record)
        return list(found.values())

    def archive_members(self, path, record):
        record.update({"member_listing_status": "not_started", "members": [],
                       "member_payloads_read": False, "schema_validated": False})
        if self.used >= self.max_entries:
            record["member_listing_status"] = "not_started_budget_exhausted"
            self.truncated(path, "zip_members")
            return
        try:
            with zipfile.ZipFile(path, "r") as archive:
                members = archive.infolist()
                record["total_member_count"] = len(members)
                inspected = 0
                for member in members:
                    if not self.take("zip_members"):
                        record["member_listing_status"] = "truncated"
                        self.truncated(path, "zip_members")
                        break
                    inspected += 1
                    if not member.is_dir() and relevant_zip_member(member.filename):
                        record["members"].append({
                            "name": member.filename, "size_bytes": member.file_size,
                            "compressed_size_bytes": member.compress_size,
                        })
                else:
                    record["member_listing_status"] = "complete"
                record["inspected_member_count"] = inspected
                record["members"].sort(key=lambda item: (
                    PurePosixPath(item["name"].replace("\\", "/")).name not in TARGETS,
                    item["name"],
                ))
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile,
                RuntimeError, EOFError, NotImplementedError) as exc:
            record["member_listing_status"] = "error"
            self.report["zip_errors"].append({"path": str(path),
                "error_type": type(exc).__name__, "error": str(exc)})
            if isinstance(exc, PermissionError):
                self.error(path, "list_zip_members", exc)

    def match(self, path, file_stat, discovery):
        if not stat.S_ISREG(file_stat.st_mode):
            self.skip(path, "matching_name_is_not_regular_file")
            return
        key = str(path)
        destination = self.archives if is_result_zip(path.name) else self.files
        if key in destination:
            if discovery not in destination[key]["discovered_by"]:
                destination[key]["discovered_by"].append(discovery)
            return
        record = {"path": key, "size_bytes": file_stat.st_size, "discovered_by": [discovery]}
        destination[key] = record
        if destination is self.archives:
            self.report["archives"].append(record)
            self.archive_members(path, record)
        else:
            record.update({"basename": path.name, "file_contents_read": False,
                           "schema_validated": False})
            self.report["files"].append(record)

    def direct_probes(self, root):
        base = Path(root["path"])
        relatives = set(TARGETS) | set(KNOWN_ARCHIVES) | set(KNOWN_RELATIVE_INPUTS)
        for legacy in LEGACY_DIRS:
            relatives.update(str(Path(legacy) / name) for name in KNOWN_RELATIVE_INPUTS)
        for relative in sorted(relatives):
            target = base / relative
            probe = {"path": str(target), "status": "missing"}
            self.report["direct_probes"].append(probe)
            current = base
            try:
                for component in Path(relative).parts:
                    current = current / component
                    current_stat = current.lstat()
                    if stat.S_ISLNK(current_stat.st_mode):
                        self.skip(current, "internal_symlink_not_followed")
                        probe["status"] = "skipped_symlink"
                        break
                else:
                    probe["status"] = "found"
                    self.match(target, current_stat, "known_path_probe")
            except (FileNotFoundError, NotADirectoryError):
                pass
            except OSError as exc:
                probe["status"] = "error"
                self.error(current, "direct_probe", exc)
        root["direct_probes_complete"] = True

    def scan(self, root):
        pending = [Path(root["path"])]
        incomplete = False
        root["recursive_scan_status"] = "started"
        while pending:
            if self.used >= self.max_entries:
                self.truncated(pending[-1], "filesystem")
                root["recursive_scan_status"] = "truncated"
                root["pending_directory_count"] = len(pending)
                return
            directory = pending.pop()
            if str(directory) in self.visited_dirs:
                self.skip(directory, "already_scanned_from_overlapping_root")
                continue
            self.visited_dirs.add(str(directory))
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not self.take("filesystem"):
                            self.truncated(directory, "filesystem")
                            root["recursive_scan_status"] = "truncated"
                            root["pending_directory_count"] = len(pending) + 1
                            return
                        path = Path(entry.path)
                        try:
                            if entry.is_symlink():
                                self.skip(path, "internal_symlink_not_followed")
                            elif entry.is_dir(follow_symlinks=False):
                                name = entry.name.lower()
                                if name in SKIP_DIRS or name.startswith("models--"):
                                    self.skip(path, "excluded_environment_or_model_cache_directory")
                                else:
                                    pending.append(path)
                            elif entry.name in TARGETS or is_result_zip(entry.name):
                                self.match(path, entry.stat(follow_symlinks=False), "bounded_name_scan")
                        except OSError as exc:
                            incomplete = True
                            self.error(path, "inspect_entry", exc)
            except OSError as exc:
                incomplete = True
                self.error(directory, "scan_directory", exc)
        root["recursive_scan_status"] = "finished_with_errors" if incomplete else "finished_with_recorded_exclusions"

    def run(self, requests):
        roots = self.roots(requests)
        # Try all finite known paths before any recursive filename search.
        for root in roots:
            self.direct_probes(root)
        for root in roots:
            self.scan(root)
        self.report["entries_used"] = self.used
        self.report["summary"] = {
            "candidate_loose_files": len(self.files), "candidate_result_archives": len(self.archives),
            "relevant_archive_members": sum(len(a["members"]) for a in self.archives.values()),
            "schema_validated": False,
            "interpretation": "Discovery only. Validate the required contents separately before analysis.",
        }
        return self.report


def main(argv=None):
    args = parse_args(argv)
    requests = [(args.repo_root, "--repo-root")]
    requests.extend((root, "--root") for root in args.root)
    env_root = os.environ.get("BEHAVIORAL_COVERAGE_RAW_DATA", "").strip()
    if env_root:
        requests.append((env_root, "BEHAVIORAL_COVERAGE_RAW_DATA"))
    requests.extend((root, "historically_recorded_root") for root in LEGACY_ROOTS)
    report = Locator(args.max_entries).run(requests)
    report["environment_root_is_set"] = bool(env_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The sole write is the explicitly requested report; never overwrite any file.
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print("Discovery report: {}".format(args.out))
    print("Candidate loose files: {}; result archives: {}; truncated: {}".format(
        report["summary"]["candidate_loose_files"],
        report["summary"]["candidate_result_archives"], report["truncated_scan"]))
    print("Contents/schema not validated. No match does not prove data loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
