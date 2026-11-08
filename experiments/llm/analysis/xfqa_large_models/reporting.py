"""Tables, scientific figures and prose from completed XfqA measurements.

This module reads saved analysis outputs.  It never selects models, fits a
scientific model, changes a split or performs a model forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


TABLE_COUNTS = {
    "scalar_main_split_results": 880,
    "scalar_removal_split_results": 4400,
    "scalar_directional_pairs": 240,
    "scalar_roster_expansion": 440,
    "vector_main_split_results": 440,
    "vector_directional_removals": 120,
    "vector_roster_expansion": 220,
}
FAMILY_LABELS = {"irrelevant_context": "Irrelevant context", "content_deletion": "Content deletion"}
MODEL_LABELS = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen-7B",
    "TIGER-Lab/General-Reasoner-Qwen2.5-7B": "General-Reasoner-7B",
    "microsoft/phi-4": "Phi-4",
    "microsoft/Phi-4-reasoning-plus": "Phi-4-reasoning-plus",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
    "allenai/OLMo-2-1124-13B-Instruct": "OLMo-2-13B",
    "ibm-granite/granite-3.3-8b-instruct": "Granite-8B",
    "google/gemma-3-12b-it": "Gemma-12B",
    "Qwen/Qwen2.5-32B-Instruct": "Qwen-32B",
    "Qwen/QwQ-32B": "QwQ-32B",
    "google/gemma-3-27b-it": "Gemma-27B",
}
PRIMARY_SCOPE = "1,000 shared-question refit bootstrap replicates; fitting and single-peer reselection included"
SECONDARY_SCOPE = ("2,000 descriptive frozen-fit common-question resamples; "
                   "fitting and selection uncertainty excluded")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    write_text(path, frame.to_csv(index=False, float_format="%.15g"))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 5) -> str:
    columns = columns or list(frame.columns)
    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.{digits}f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def implementation_check_summary(root: Path) -> tuple[str, dict, list[Path]]:
    """Read recorded test outcomes without adding overlapping runs together."""
    records = []
    sources = []
    for name, label, required in (
        ("implementation_tests.xml", "CPU implementation suite", True),
        ("delivery_tests.xml", "Later targeted packaging checks", False),
    ):
        path = root / "validation" / name
        if not path.exists() and not required:
            continue
        document = ET.parse(path).getroot()
        suites = [node for node in document.iter("testsuite") if node.findall("testcase")]
        counts = {key: sum(int(node.attrib.get(key, "0")) for node in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        cases = [case for suite in suites for case in suite.findall("testcase")]
        if len(cases) != counts["tests"] or not cases:
            raise ValueError("Test evidence has incomplete or inconsistent case counts: " + name)
        for field, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            if sum(case.find(tag) is not None for case in cases) != counts[field]:
                raise ValueError("Test evidence outcome counts disagree: " + name)
        passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        if passed < 0:
            raise ValueError("Invalid test outcome counts: " + name)
        records.append({"suite": label, "tests": counts["tests"], "passed": passed,
                        "failed": counts["failures"], "errors": counts["errors"], "skipped": counts["skipped"],
                        "source": "validation/" + name, "sha256": sha256(path),
                        "skipped_cases": [{"name": case.attrib["name"], "reason": case.find("skipped").attrib.get("message", "")}
                                          for case in cases if case.find("skipped") is not None]})
        sources.append(path)
    path = root / "validation/scoring_tests.txt"
    text = path.read_text()
    statuses = re.findall(r"^test\S+ .* \.\.\. (ok|FAIL|ERROR|skipped\b[^\n]*)$", text, re.MULTILINE)
    summary = re.search(r"^Ran (\d+) tests? in [^\n]+$", text, re.MULTILINE)
    if summary is None or not statuses or len(statuses) != int(summary.group(1)):
        raise ValueError("Scoring test evidence has incomplete or inconsistent case counts")
    passed, failed, errors = (statuses.count(status) for status in ("ok", "FAIL", "ERROR"))
    skipped = sum(status.startswith("skipped") for status in statuses)
    if not re.search(r"^(OK(?: \([^\n]*\))?|FAILED \([^\n]+\))$", text, re.MULTILINE):
        raise ValueError("Scoring test evidence has no completed unittest outcome")
    records.append({"suite": "Torch scoring environment", "tests": len(statuses), "passed": passed,
                    "failed": failed, "errors": errors, "skipped": skipped,
                    "source": "validation/scoring_tests.txt", "sha256": sha256(path), "skipped_cases": []})
    sources.append(path)
    frame = pd.DataFrame([{"Run": row["suite"], "Passed": row["passed"], "Failed": row["failed"],
                           "Errors": row["errors"], "Skipped": row["skipped"],
                           "Evidence": f"[{Path(row['source']).name}](../{row['source']})"} for row in records])
    explanation = ("These are separate recorded test runs, with overlapping test cases; their counts are not "
                   "summed into a unique-test total. The Torch environment run separately exercises scoring "
                   "checks that may be skipped by the CPU environment. Synthetic implementation controls "
                   "do not constitute measured model results.\n")
    skips = [f"`{case['name']}`: {case['reason']}" for row in records for case in row["skipped_cases"]]
    if skips:
        explanation += "\nRecorded skipped checks: " + "; ".join(skips) + ".\n"
    evidence = {"schema": "xfqa_implementation_check_summary_v1", "runs": records,
                "overlapping_runs": True, "unique_test_total": None}
    return markdown_table(frame) + "\n\n" + explanation, evidence, sources


def _selector(frame: pd.DataFrame, **keys: Any) -> pd.DataFrame:
    selected = frame
    for name, value in keys.items():
        if name not in selected:
            raise ValueError(f"Missing required selector column: {name}")
        selected = selected.loc[selected[name].fillna("").astype(str) == str(value)]
    return selected


def interval_row(intervals: pd.DataFrame, table: str, metric: str, **keys: Any) -> pd.Series:
    selected = _selector(intervals, table=table, metric=metric, **keys)
    if len(selected) != 1:
        raise ValueError(f"Expected one interval for {table}/{metric}/{keys}; found {len(selected)}")
    row = selected.iloc[0]
    for field in ("point", "low", "high"):
        if not np.isfinite(float(row[field])):
            raise ValueError(f"Nonfinite {field} interval: {keys}")
    if row.low > row.high:
        raise ValueError("Inverted percentile interval")
    return row


def aggregate_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean split estimates, with ratio-of-means separately named."""
    rows = []
    for name in ("scalar_main_split_results", "vector_main_split_results"):
        source = tables[name]
        keys = ["run_id", "panel", "target", "family", "response", "fit_loss", "evaluation_metric", "input_fingerprint"]
        for key, frame in source.groupby(keys, sort=False, dropna=False):
            record = dict(zip(keys, key))
            if frame.split_seed.nunique() != 10 or len(frame) != 10:
                raise ValueError(f"Summary needs exactly ten distinct splits: {key}")
            for metric in ("e_single", "e_group", "gamma", "relative_gamma_percent", "mse_single", "mse_group",
                           "single_top_option_agreement", "group_top_option_agreement", "train_single", "train_group"):
                if metric in frame:
                    values = pd.to_numeric(frame[metric], errors="raise")
                    record[metric] = (float(values.mean(skipna=False)) if metric == "relative_gamma_percent" else
                                      float(values.mean()) if values.notna().any() else np.nan)
            denominator = record["e_single"]
            record["macro_ratio_of_means_percent"] = (100 * record["gamma"] / denominator if denominator > 1e-14 else np.nan)
            record["split_count"] = 10
            record["source_table"] = name
            record["completion_state"] = "complete"
            record["peer_ids"] = frame.peer_ids.iloc[0]
            record["positive_gain_split_count"] = int((frame.gamma > 1e-12).sum())
            record["negative_gain_split_count"] = int((frame.gamma < -1e-12).sum())
            record["zero_gain_split_count"] = int((frame.gamma.abs() <= 1e-12).sum())
            rows.append(record)
    return pd.DataFrame(rows)


def _means(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, as_index=False, sort=False)[metrics].mean()


def _save_figure(fig, directory: Path, stem: str, caption: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fig.text(.015, .008, caption, ha="left", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, .105, 1, 1))
    paths = [directory / (stem + ".pdf"), directory / (stem + ".png")]
    fig.savefig(paths[0], bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def _interval_lines(ax, x, points, lows, highs, *, color, label=None):
    """Percentile intervals need not contain their point estimates."""
    ax.vlines(x, lows, highs, color=color, linewidth=1.4)
    ax.scatter(x, points, color=color, s=30, label=label, zorder=3)
    ax.scatter(x, lows, marker="_", color=color, s=30)
    ax.scatter(x, highs, marker="_", color=color, s=30)


def make_figures(config: dict, tables: dict[str, pd.DataFrame], primary: pd.DataFrame,
                 secondary: pd.DataFrame, directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    new = [m["id"] for m in config["new_models"]]
    families = list(config["families"])
    scalar = tables["scalar_main_split_results"]
    main = _means(_selector(scalar, panel="expanded11", fit_loss="mse"), ["target", "family"], ["e_single", "e_group", "gamma"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for ax, family in zip(axes, families):
        x = np.arange(3)
        for offset, metric, label, color in [(-.17, "e_single", "Fitting-selected single", "#6C7D8B"),
                                            (.17, "e_group", "Full ten-peer group", "#007F73")]:
            values = [interval_row(primary, "scalar_main_split_results", metric, panel="expanded11", target=t,
                                   family=family, fit_loss="mse") for t in new]
            _interval_lines(ax, x + offset, [v.point for v in values], [v.low for v in values],
                            [v.high for v in values], color=color, label=label)
        ax.set_xticks(x, [MODEL_LABELS[t] for t in new])
        ax.set_title(FAMILY_LABELS[family])
        ax.set_ylabel("Held-out reference-answer probability MAE")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="lower left", frameon=False)
    output = _save_figure(fig, directory, "large_target_coverage",
                         "MSE fitting; expanded eleven-model roster. Points average ten overlapping splits.\n"
                         "Bars: pointwise 95% intervals from 1,000 shared-question refit bootstrap replicates, including single-peer reselection.")

    pair = config["declared_pairs"][0]
    directions = [(pair["a"], pair["b"]), (pair["b"], pair["a"])]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    for column, family in enumerate(families):
        x = np.arange(2)
        for offset, metric, label, color in [(-.16, "partner_error", "Designated partner", "#6C7D8B"),
                                            (.16, "e_group", "Full ten-peer group", "#007F73")]:
            values = [interval_row(primary, "scalar_directional_pairs", metric, panel="expanded11", target=t,
                                   partner=p, family=family, fit_loss="mse") for t, p in directions]
            _interval_lines(axes[0, column], x + offset, [v.point for v in values], [v.low for v in values],
                            [v.high for v in values], color=color, label=label)
        for offset, metric, label, color in [(-.16, "extra_beyond_partner", "Extra beyond partner", "#007F73"),
                                            (.16, "removal_effect", "Partner-removal effect", "#BA5A32")]:
            values = [interval_row(primary, "scalar_directional_pairs", metric, panel="expanded11", target=t,
                                   partner=p, family=family, fit_loss="mse") for t, p in directions]
            _interval_lines(axes[1, column], x + offset, [v.point for v in values], [v.low for v in values],
                            [v.high for v in values], color=color, label=label)
        labels = [f"Target: {MODEL_LABELS[t]}\nPartner: {MODEL_LABELS[p]}" for t, p in directions]
        for row in range(2):
            axes[row, column].set_xticks(x, labels)
            axes[row, column].grid(axis="y", alpha=.2)
        axes[0, column].set_title(FAMILY_LABELS[family])
        axes[0, column].set_ylabel("Held-out scalar MAE")
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set_ylabel("Difference in held-out scalar MAE")
        axes[1, column].axhline(0, color="#333333", linewidth=.7)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=9)
    output += _save_figure(fig, directory, "large_pair_directionality",
                          "MSE fitting; expanded eleven-model roster. Direct partner distance is symmetric; group and refitted removal contrasts need not be.\n"
                          "Bars: pointwise 95% intervals from 1,000 shared-question refit bootstrap replicates; ten overlapping splits are averaged within each replicate.")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=True)
    old = config["original_roster"]
    for ax, family in zip(axes, families):
        values = [interval_row(secondary, "scalar_roster_expansion", "large_peer_gain", panel="expanded11",
                               target=t, family=family, fit_loss="mse") for t in old]
        y = np.arange(len(old))
        ax.hlines(y, [v.low for v in values], [v.high for v in values], color="#007F73")
        ax.scatter([v.point for v in values], y, color="#007F73", zorder=3)
        ax.set_yticks(y, [MODEL_LABELS[t] for t in old])
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Baseline-eight MAE − expanded-eleven MAE")
        ax.axvline(0, color="#333333", linewidth=.8)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    output += _save_figure(fig, directory, "roster_expansion",
                          "MSE fitting; all eight original targets, using identical saved responses in both rosters. Positive values indicate reduced error.\n"
                          "Bars: descriptive 95% frozen-fit common-question intervals (2,000 resamples); fitting and selection uncertainty are excluded.")
    return output


def _claims(config, tables, primary, secondary):
    claims = []
    new = [m["id"] for m in config["new_models"]]
    for target in new:
        for family in config["families"]:
            keys = dict(panel="expanded11", target=target, family=family, fit_loss="mse")
            source = _selector(tables["scalar_main_split_results"], **keys)
            interval = interval_row(primary, "scalar_main_split_results", "gamma", **keys)
            mean_single = float(source.e_single.mean())
            mean_group = float(source.e_group.mean())
            claims.append({"claim_id": f"large_target/{target}/{family}", "source_table": "scalar_main_split_results.csv",
                           "row_selector": keys, "split_seeds": sorted(source.split_seed.astype(int).tolist()),
                           "formula": "mean(e_single - e_group) across the ten fixed splits",
                           "e_single": mean_single, "e_group": mean_group,
                           "point": float(source.gamma.mean()), "low": float(interval.low), "high": float(interval.high),
                           "ratio_of_mean_gain_to_mean_single_percent": (100 * (mean_single - mean_group) / mean_single if mean_single > 1e-14 else None),
                           "uncertainty_scope": PRIMARY_SCOPE, "interval_contrast_id": str(interval.contrast_id)})
    for pair in config["declared_pairs"]:
        for target, partner in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
            for family in config["families"]:
                keys = dict(panel="expanded11", target=target, partner=partner, family=family, fit_loss="mse")
                source = _selector(tables["scalar_directional_pairs"], **keys)
                is_primary = pair["priority"] == "primary"
                intervals = primary if is_primary else secondary
                for metric in ("partner_error", "e_group", "extra_beyond_partner", "removal_effect", "directional_extra_difference"):
                    interval = interval_row(intervals, "scalar_directional_pairs", metric, **keys)
                    claims.append({"claim_id": f"direction/{target}/{partner}/{family}/{metric}",
                                   "source_table": "scalar_directional_pairs.csv", "row_selector": keys,
                                   "formula": f"mean({metric}) across the ten fixed splits", "point": float(source[metric].mean()),
                                   "low": float(interval.low), "high": float(interval.high),
                                   "uncertainty_scope": PRIMARY_SCOPE if is_primary else SECONDARY_SCOPE,
                                   "interval_contrast_id": str(interval.contrast_id)})
    for target in config["expanded_roster"]:
        for family in config["families"]:
            keys = dict(panel="expanded11", target=target, family=family, fit_loss="mse")
            source = _selector(tables["scalar_roster_expansion"], **keys)
            interval = interval_row(secondary, "scalar_roster_expansion", "large_peer_gain", **keys)
            claims.append({"claim_id": f"roster/{target}/{family}", "source_table": "scalar_roster_expansion.csv",
                           "row_selector": keys, "formula": "mean(comparator_error - expanded_error) across fixed splits",
                           "point": float(source.large_peer_gain.mean()), "low": float(interval.low), "high": float(interval.high),
                           "uncertainty_scope": SECONDARY_SCOPE, "interval_contrast_id": str(interval.contrast_id)})
    return claims


def make_written_outputs(config, tables, summary, primary, secondary) -> dict[str, str | dict]:
    claims = _claims(config, tables, primary, secondary)
    new_claims = [c for c in claims if c["claim_id"].startswith("large_target/")]
    rows = [{"Target": MODEL_LABELS[c["row_selector"]["target"]],
             "Family": FAMILY_LABELS[c["row_selector"]["family"]],
             "Single MAE": c["e_single"], "Group MAE": c["e_group"], "Gain": c["point"],
             "Refit 95% interval": f"[{c['low']:.5f}, {c['high']:.5f}]"} for c in new_claims]
    headline_table = markdown_table(pd.DataFrame(rows))
    pair = config["declared_pairs"][0]
    direction_claims = {c["claim_id"]: c for c in claims if c["claim_id"].startswith("direction/")}
    direction_sentences = []
    for family in config["families"]:
        prefix = f"direction/{pair['a']}/{pair['b']}/{family}/"
        reverse = f"direction/{pair['b']}/{pair['a']}/{family}/"
        left, right = direction_claims[prefix + "extra_beyond_partner"], direction_claims[reverse + "extra_beyond_partner"]
        difference = direction_claims[prefix + "directional_extra_difference"]
        direction_sentences.append(
            f"For {FAMILY_LABELS[family].lower()}, the extra reduction beyond the designated same-scale partner was "
            f"{left['point']:.5f} for Qwen-32B as target and {right['point']:.5f} for QwQ-32B as target; "
            f"the first-minus-second difference was {difference['point']:.5f} "
            f"(95% refit interval [{difference['low']:.5f}, {difference['high']:.5f}]).")
    unresolved_gain = [c for c in new_claims if c["low"] <= 0 <= c["high"]]
    gain_interpretation = ""
    if unresolved_gain:
        cells = [f"{MODEL_LABELS[c['row_selector']['target']]} under {FAMILY_LABELS[c['row_selector']['family']].lower()}"
                 for c in unresolved_gain]
        gain_interpretation = ("The primary gain interval includes zero for " + "; ".join(cells) +
                               ", so an improvement is not established in every new-target/family cell. ")
    unresolved_directions = [FAMILY_LABELS[family].lower() for family in config["families"]
                             if direction_claims[f"direction/{pair['a']}/{pair['b']}/{family}/directional_extra_difference"]["low"] <= 0 <=
                             direction_claims[f"direction/{pair['a']}/{pair['b']}/{family}/directional_extra_difference"]["high"]]
    direction_interpretation = ("The directional-difference interval includes zero for " + "; ".join(unresolved_directions) + ". "
                                if unresolved_directions else "")
    primary_removal_claims = [direction_claims[f"direction/{target}/{partner}/{family}/removal_effect"]
                              for family in config["families"]
                              for target, partner in ((pair["a"], pair["b"]), (pair["b"], pair["a"]))]
    if all(abs(c["point"]) <= 1e-12 for c in primary_removal_claims):
        direction_interpretation += ("Removing the designated same-scale partner changed point-estimate MAE by "
                                     "0.00000 in both directions and both families. The extra-coverage contrast "
                                     "therefore describes different full-roster reconstruction errors; it does "
                                     "not establish dependence on the designated partner. ")
    # The primary comparisons are fixed by the plan. These two secondary examples
    # are explicitly selected after inspecting the complete old-target table.
    roster_claims = [c for c in claims if c["claim_id"].startswith("roster/") and c["row_selector"]["target"] in config["original_roster"]]
    roster_scores = {target: float(np.mean([c["point"] for c in roster_claims if c["row_selector"]["target"] == target]))
                     for target in config["original_roster"]}
    ordered_targets = sorted(roster_scores, key=lambda target: (roster_scores[target], target))
    illustrative_targets = list(dict.fromkeys([ordered_targets[-1], ordered_targets[0]]))
    illustrative_claims = [c for target in illustrative_targets for c in roster_claims if c["row_selector"]["target"] == target]
    roster_sentences = []
    for target in illustrative_targets:
        cells = [c for c in illustrative_claims if c["row_selector"]["target"] == target]
        details = [f"{FAMILY_LABELS[c['row_selector']['family']].lower()}: {c['point']:.5f} [{c['low']:.5f}, {c['high']:.5f}]"
                   for c in cells]
        roster_sentences.append(f"{MODEL_LABELS[target]} ({'; '.join(details)})")
    roster_text = ("Adding the three large peers also changed reconstruction of the original targets. "
                   "As illustrative examples selected after inspecting the complete table, the largest and "
                   "smallest mean roster gains across the two families were for " + "; ".join(roster_sentences) +
                   ". These gains compare the original seven-peer group with the expanded ten-peer group, "
                   "using MSE fitting and held-out scalar MAE; brackets are descriptive 95% frozen-fit "
                   "intervals, which exclude fitting and selection uncertainty. Negative gains denote "
                   "increased held-out error. The complete table retains every original target.")
    # Fixed reporting scope: all six new-target/family cells, never a favorable subset.
    main_sentences = []
    for target in [m["id"] for m in config["new_models"]]:
        items = [c for c in new_claims if c["row_selector"]["target"] == target]
        details = [f"{FAMILY_LABELS[c['row_selector']['family']].lower()}: {c['e_single']:.5f} to {c['e_group']:.5f}, "
                   f"gain {c['point']:.5f} [{c['low']:.5f}, {c['high']:.5f}]" for c in items]
        main_sentences.append(f"{MODEL_LABELS[target]} ({'; '.join(details)})")
    primary_intro = ("With all ten other models available, the MSE-fitted fixed convex group gave the following "
                     "held-out reference-answer probability MAEs relative to a fitting-selected single peer "
                     "(brackets are pointwise 95% intervals for the gain from 1,000 shared-question refit replicates): ")
    response = ("# XfqA response draft\n\n"
                "Thank you for suggesting a larger-model roster. We have now extended the audit with "
                "Qwen2.5-32B-Instruct, QwQ-32B, and Gemma-3-27B-IT, while keeping the question sample, "
                "perturbations, response definition and fitting/evaluation splits fixed.\n\n"
                + primary_intro + "; ".join(main_sentences) + ".\n\n"
                + gain_interpretation + "\n\n"
                + " ".join(direction_sentences) + " " + direction_interpretation + "These directional contrasts compare each target's full-group "
                "reconstruction with its designated partner; the direct partner distance itself is symmetric.\n\n"
                + roster_text + "\n\n"
                "The accompanying tables include all eleven targets, both stress families, the original-eight "
                "and expanded-eleven peer rosters, MAE-fitting sensitivity, and matched-TV complete-option-vector "
                "sensitivity. The primary intervals refit the group and reselect the single peer within each "
                "question resample; other reported intervals are explicitly descriptive and hold fits fixed. "
                "The measured gains vary across targets and stress families, and enlarging the roster does "
                "not uniformly lower held-out error. This extension measures coverage relations in the tested mixed-size collection under the fixed "
                "candidate-label response interface. It does not isolate a causal parameter-count effect or "
                "measure unconstrained free-text reasoning performance.\n")
    methods = ("# Methods insert\n\n"
               "We extended the eight-model roster with Qwen/Qwen2.5-32B-Instruct, Qwen/QwQ-32B and "
               "google/gemma-3-27b-it, using checkpoint revisions frozen before scoring. We reused the eight "
               "models' supplied fresh XWVG option-probability measurements and scored only the three added "
               "models. The original 560 MMLU-Pro questions (40 per subject in 14 subjects), all cyclic option "
               "rotations, seven semantic conditions, and ten stratified 280/280 fitting/evaluation splits "
               "were held fixed. Each family assigns clean mass 1/2 and mass 1/6 to each of three maximum-stress "
               "realizations. BF16 scoring sums the log likelihood of every token in the prescribed candidate "
               "continuation, normalizes across valid options, remaps to semantic option order, and averages "
               "probabilities across complete cyclic groups. Native chat templates and QwQ's native reasoning "
               "prefix are preserved; Gemma uses text-only input through its verified model class.\n\n"
               "For each panel, target, family and split, we choose a single peer and fit a fixed simplex mixture "
               "using fitting questions only. The main scalar response is reference-answer probability: both "
               "selection and fitting minimize MSE, and evaluation reports held-out MAE. MAE fitting is a "
               "separate sensitivity. A matched-TV sensitivity fits and evaluates complete option distributions "
               "and also reports top-option agreement. The panels are all eleven targets against the other ten, "
               "the original eight against the other original seven, and the three new targets against the old "
               "eight. Removing a peer always triggers a new fit on the same fitting questions.\n\n"
               "Primary MSE uncertainty uses 1,000 subject-stratified common-question bootstrap replicates "
               "(seed 20261108), shared across models, families and splits. Each replicate refits the three new "
               "targets' full groups, reselects their single peers and refits both same-scale Qwen/QwQ removal "
               "directions. Fitting and evaluation masses are separately normalized within each split, and "
               "split estimates are averaged within the replicate before forming pointwise percentile "
               "intervals. Other contrasts use 2,000 common-question descriptive resamples (seed 20261109) "
               "with coefficients and selected peers fixed, excluding fitting/selection uncertainty. No "
               "independence of the overlapping splits, p-values or ecosystem-wide simultaneous coverage "
               "is asserted.\n")
    results = ("# Results insert\n\n" + primary_intro + "\n\n" + headline_table + "\n\n" + gain_interpretation +
               "\n\n" + " ".join(direction_sentences) + " " + direction_interpretation + "\n\n" + roster_text + "\n")
    limitations = ("# Limitations and execution scope\n\n"
                   "The empirical scope is the fixed three-checkpoint extension, one existing 560-question "
                   "benchmark sample and the specified endpoint stresses. The old eight vectors are reused "
                   "fresh XWVG measurements, not reconstructed historical scoring bytes. The three new "
                   "models have new measured responses under the matched prompt design.\n\n"
                   "Model size, family and post-training are not independently manipulated. QwQ-32B belongs "
                   "to the Qwen2.5-32B base family; it is not labelled a 32B General-Reasoner model or assumed "
                   "to be fine-tuned from the Instruct checkpoint. Candidate-label responses under its native "
                   "reasoning prefix are not generated reasoning outcomes. Scalar MAE and full-vector TV "
                   "refer to different representations and are displayed on separate scales.\n\n"
                   "Primary percentile intervals incorporate fitting and single-peer reselection but remain "
                   "pointwise. Secondary frozen-fit intervals omit fitting and selection uncertainty. The ten "
                   "fixed splits overlap and are not independent replication units. A small directional "
                   "difference does not establish equivalence. Test-set gains and removal effects can be "
                   "negative; all such rows remain in the tables.\n\n"
                   "Execution identities, native-template boundaries, placement, package versions, measured "
                   "resource use and any adapter validation are retained in the run's provenance, validation "
                   "and status records. The scientific scoring convention and analysis configuration are "
                   "fixed. Refer to those records for actual execution changes rather than treating pilot "
                   "projections as dedicated GPU hours.\n")
    direction_table = _means(_selector(tables["scalar_directional_pairs"], fit_loss="mse"),
                             ["target", "partner", "family"], ["partner_error", "e_group", "extra_beyond_partner", "removal_effect", "directional_extra_difference"])
    roster_table = _means(_selector(tables["scalar_roster_expansion"], fit_loss="mse"),
                          ["target", "family", "comparator_panel"], ["comparator_error", "expanded_error", "large_peer_gain"])
    report = ("# XfqA larger-model empirical report\n\n"
              "All three new checkpoints and all prescribed point analyses and uncertainty calculations "
              "are represented in these completed input tables. Final independent numerical replay is "
              "recorded separately in `validation/numerical_replay.json`; this document alone is not the "
              "run completion marker.\n\n"
              "## Large targets: primary scalar analysis\n\n" + headline_table + "\n\n"
              "Gain is selected-single MAE minus fixed-group MAE. The table averages ten overlapping "
              "splits; its intervals come from the full-refit shared-question bootstrap. All six declared "
              "target/family cells are shown, including zero or negative gains. " + gain_interpretation + "\n\n"
              "## Declared directional relations\n\n" + markdown_table(direction_table) + "\n\n"
              + " ".join(direction_sentences) + " " + direction_interpretation + "\n\n"
              "## Paired roster expansion\n\n" + markdown_table(roster_table) + "\n\n"
              "Positive large-peer gain means that the expanded roster reduces held-out MAE. Each "
              "comparison uses the same target responses and evaluation questions. Its intervals in "
              "`secondary_descriptive_intervals.csv` are frozen-fit descriptive intervals. " + roster_text + "\n\n"
              "## Full results and sensitivity\n\n" + markdown_table(summary, ["panel", "target", "family", "fit_loss", "evaluation_metric", "e_single", "e_group", "gamma"]) + "\n\n"
              "The scalar MAE-fit and complete-vector TV-fit results are separate analyses with matching "
              "training selection objectives. Top-option agreement, per-split values, all scalar peer "
              "removals, the six vector removal directions, saved coefficients and physical-question "
              "losses are retained in the accompanying machine-readable tables.\n\n"
              "## Provenance and uncertainty\n\n" + methods.split("\n\n", 1)[1] + "\n"
              "See `headline_evidence.json` for the exact table selectors, formulas, estimates and "
              "uncertainty sources behind the response and manuscript insert. See "
              "`LIMITATIONS_AND_DEVIATIONS.md` and the run's provenance for interpretation and execution scope.\n")
    return {"REPORT.md": report, "XfqA_RESPONSE_DRAFT.md": response, "METHODS_INSERT.md": methods,
            "RESULTS_INSERT.md": results, "LIMITATIONS_AND_DEVIATIONS.md": limitations,
            "headline_evidence.json": {"schema_version": "xfqa_headline_evidence_v1", "claims": claims,
                                       "primary_uncertainty": PRIMARY_SCOPE, "secondary_uncertainty": SECONDARY_SCOPE,
                                       "illustrative_roster_selection": {
                                           "scope": "post hoc illustrative examples from the complete original-target roster table",
                                           "rule": "largest and smallest mean MSE-fit scalar roster gain across the two declared families",
                                           "selected_targets": illustrative_targets,
                                           "candidate_targets": config["original_roster"]},
                                       "response_numerical_claims": [c["claim_id"] for c in new_claims] + [
                                           f"direction/{t}/{p}/{f}/{m}" for f in config["families"]
                                           for t, p in ((pair["a"], pair["b"]), (pair["b"], pair["a"]))
                                           for m in ("extra_beyond_partner", "directional_extra_difference", "removal_effect")] +
                                           [c["claim_id"] for c in illustrative_claims]}}


def run_report(repo_root, plan_root, output_root, resume=False):
    repo_root, plan_root, output_root = map(Path, (repo_root, plan_root, output_root))
    analysis = output_root / "analysis"
    config = json.loads((plan_root / "config/experiment.json").read_text())
    inputs = [analysis / f"{name}.csv" for name in TABLE_COUNTS] + [analysis / "primary_refit_intervals.csv", analysis / "secondary_descriptive_intervals.csv"]
    for required_marker in (output_root / "status/assembly_complete.json", analysis / "point_complete.json", analysis / "bootstrap_complete.json"):
        if not required_marker.exists() or json.loads(required_marker.read_text()).get("status") != "complete":
            raise ValueError(f"Report requires a completed empirical stage: {required_marker}")
    assembly = json.loads((output_root / "status/assembly_complete.json").read_text())
    manifest = json.loads((analysis / "analysis_manifest.json").read_text())
    if (assembly.get("real_three_model_scoring_complete") is not True or assembly.get("new_rotation_rows") != 110838 or
            assembly.get("merged_averaged_vectors") != 43120 or manifest["identity"].get("synthetic_fixture") is not False or
            manifest["identity"].get("scope") != "complete_expanded_experiment"):
        raise ValueError("Measured-results reports require complete real three-model data and a full empirical analysis identity")
    test_summary, test_evidence, test_sources = implementation_check_summary(output_root)
    if any(row["failed"] or row["errors"] for row in test_evidence["runs"]):
        raise ValueError("Recorded implementation checks contain failures or errors")
    inputs += test_sources
    identity = {str(p.relative_to(output_root)): sha256(p) for p in inputs}
    identity["reporting_source_sha256"] = sha256(Path(__file__))
    marker = output_root / "status/report_complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if not resume or saved.get("input_identity") != identity or saved.get("input_fingerprint") != manifest["input_fingerprint"]:
            raise ValueError("Existing report identity differs or --resume was not supplied")
        for record in saved["artifacts"]:
            if sha256(output_root / record["path"]) != record["sha256"]:
                raise ValueError(f"Completed report artifact changed: {record['path']}")
        return saved
    tables = {name: pd.read_csv(analysis / f"{name}.csv", keep_default_na=True) for name in TABLE_COUNTS}
    fingerprints = set()
    for name, frame in tables.items():
        if len(frame) != TABLE_COUNTS[name]:
            raise ValueError(f"Incomplete {name}: {len(frame)} != {TABLE_COUNTS[name]}")
        if not frame.completion_state.eq("complete").all():
            raise ValueError(f"Incomplete row in {name}")
        fingerprints.update(frame.input_fingerprint.astype(str))
    if len(fingerprints) != 1:
        raise ValueError("Mixed scientific input identities in report")
    if manifest["input_fingerprint"] != next(iter(fingerprints)):
        raise ValueError("Report inputs differ from the recorded empirical analysis identity")
    primary = pd.read_csv(analysis / "primary_refit_intervals.csv")
    secondary = pd.read_csv(analysis / "secondary_descriptive_intervals.csv")
    if not primary.replicates.eq(1000).all() or not secondary.replicates.eq(2000).all():
        raise ValueError("Prescribed uncertainty replicates are incomplete")
    for intervals in (primary, secondary):
        if set(intervals.input_fingerprint.astype(str)) != fingerprints or not intervals.completion_state.eq("complete").all():
            raise ValueError("Uncertainty and point tables have different input identities or incomplete rows")
    if not primary.includes_refitting.eq(True).all() or not primary.includes_single_reselection.eq(True).all():
        raise ValueError("Primary intervals must include refitting and single-peer reselection")
    if not secondary.includes_refitting.eq(False).all() or not secondary.includes_single_reselection.eq(False).all():
        raise ValueError("Secondary intervals must be explicitly frozen-fit")
    summary = aggregate_tables(tables)
    write_csv(analysis / "full_summary.csv", summary)
    written = make_written_outputs(config, tables, summary, primary, secondary)
    written["REPORT.md"] += "\n## Implementation verification\n\n" + test_summary
    written["implementation_check_summary.json"] = test_evidence
    for name, value in written.items():
        if name.endswith(".json"):
            write_json(analysis / name, _clean(value))
        else:
            write_text(analysis / name, value)
    figures = make_figures(config, tables, primary, secondary, analysis / "figures")
    outputs = [analysis / "full_summary.csv"] + [analysis / n for n in written] + figures
    result = {"status": "complete", "input_fingerprint": next(iter(fingerprints)), "input_identity": identity,
              "scope": "Reports and plots from completed empirical tables; independent replay is a separate required gate",
              "artifacts": [{"path": str(p.relative_to(output_root)), "sha256": sha256(p), "bytes": p.stat().st_size} for p in outputs]}
    write_json(marker, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "plan-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_report(args.repo_root, args.plan_root, args.output_root, args.resume), indent=2))
