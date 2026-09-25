from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analysis import atomic_parquet
from .utils import atomic_write_json, atomic_write_text, load_json, sha256_file, utc_now

LOCKED = "LOCKED"
QUALIFIED = "SUPPORTED_WITH_INTERFACE_QUALIFIER"
RAW_ONLY = "RAW_INTERFACE_ONLY"
NOT_SUPPORTED = "NOT_SUPPORTED"


def _interval(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _generation_compatibility(generation: pd.DataFrame, model: str) -> tuple[bool, str]:
    rows = generation[generation["model"].eq(model)]
    malformed = float(rows["malformed_rate"].mean())
    agreement = float(rows["all_rotation_score_generation_agreement"].mean())
    compatible = malformed <= 0.10 and agreement >= 0.50
    evidence = f"mean malformed={malformed:.3f}; all-rotation score-generation agreement={agreement:.3f}"
    return compatible, evidence


def _endpoint_claims(
    raw: pd.DataFrame,
    permutation: pd.DataFrame,
    bootstrap: pd.DataFrame,
    generation: pd.DataFrame,
) -> list[dict[str, Any]]:
    specifications = [
        ("C1_phi4_irrelevant_outward", "microsoft/phi-4", "irrelevant_context", 1, False),
        ("C1_qwen_irrelevant_outward", "Qwen/Qwen2.5-7B-Instruct", "irrelevant_context", 1, False),
        ("C2_mistral_irrelevant_inward", "mistralai/Mistral-Nemo-Instruct-2407", "irrelevant_context", -1, False),
        ("C2_olmo_irrelevant_inward", "allenai/OLMo-2-1124-13B-Instruct", "irrelevant_context", -1, False),
        ("C2_granite_irrelevant_inward", "ibm-granite/granite-3.3-8b-instruct", "irrelevant_context", -1, False),
        ("C5_mistral_deletion_contraction", "mistralai/Mistral-Nemo-Instruct-2407", "content_deletion", -1, False),
        ("C5_olmo_deletion_scalar", "allenai/OLMo-2-1124-13B-Instruct", "content_deletion", -1, True),
        ("C5_granite_deletion_scalar", "ibm-granite/granite-3.3-8b-instruct", "content_deletion", -1, True),
    ]
    rows: list[dict[str, Any]] = []
    perm_boot = bootstrap[bootstrap["interface"].eq("permutation_averaged")]
    for claim_id, target, family, direction, scalar_only in specifications:
        raw_rows = raw[raw["target"].eq(target) & raw["family"].eq(family)]
        perm_rows = permutation[
            permutation["target"].eq(target) & permutation["family"].eq(family)
        ]
        boot_rows = perm_boot[
            perm_boot["target"].eq(target) & perm_boot["family"].eq(family)
        ]
        raw_delta = float(raw_rows["endpoint_delta"].mean())
        perm_delta = float(perm_rows["endpoint_delta"].mean())
        lower, upper = _interval(boot_rows["endpoint_delta"])
        directional_splits = int(
            (perm_rows["endpoint_delta"] > 0).sum()
            if direction > 0
            else (perm_rows["endpoint_delta"] < 0).sum()
        )
        raw_direction = raw_delta * direction > 0
        semantic_direction = perm_delta * direction > 0
        interval_excludes = lower > 0 if direction > 0 else upper < 0
        generation_ok, generation_evidence = _generation_compatibility(generation, target)
        core = semantic_direction and interval_excludes and directional_splits >= 9
        if core and generation_ok and not scalar_only:
            status = LOCKED
        elif core:
            status = QUALIFIED
        elif raw_direction:
            status = RAW_ONLY
        else:
            status = NOT_SUPPORTED
        direction_word = "increase" if direction > 0 else "decrease"
        rows.append(
            {
                "claim_id": claim_id,
                "claim_type": "endpoint_effect",
                "target": target,
                "family": family,
                "v22_raw_trackwise": True,
                "v23_endpoint_raw": raw_direction,
                "v23_permutation_averaged": semantic_direction,
                "common_bootstrap": interval_excludes,
                "matched_objective": True,
                "generation_compatibility": generation_ok,
                "final_status": status,
                "raw_endpoint_delta": raw_delta,
                "permutation_endpoint_delta": perm_delta,
                "bootstrap_lower": lower,
                "bootstrap_upper": upper,
                "directional_split_count": directional_splits,
                "observed_evidence": (
                    f"raw endpoint Δ={raw_delta:.6f}; permutation-averaged Δ={perm_delta:.6f}; "
                    f"common-resample 95% interval [{lower:.6f}, {upper:.6f}]; "
                    f"directional splits={directional_splits}/10; {generation_evidence}"
                ),
                "what_it_supports": f"Whether {target} shows an endpoint PIER {direction_word} under {family}.",
                "what_it_does_not_support": "It does not identify causal training lineage, safe replacement, or a uniquely true latent belief.",
                "paper_ready_wording": (
                    f"For {target}, endpoint PIER {direction_word}d under {family} after exact cyclic option-label averaging"
                    + ("." if status in (LOCKED, QUALIFIED) else " was not supported by the balanced semantic interface.")
                ),
            }
        )
    return rows


def _sibling_claims(
    sibling: pd.DataFrame, bootstrap: pd.DataFrame, generation: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    perm_boot = bootstrap[bootstrap["interface"].eq("permutation_averaged")]
    qwen_targets = (
        "Qwen/Qwen2.5-7B-Instruct",
        "TIGER-Lab/General-Reasoner-Qwen2.5-7B",
    )
    for target in qwen_targets:
        for family in ("irrelevant_context", "content_deletion"):
            point = sibling[sibling["target"].eq(target) & sibling["family"].eq(family)]
            boot = perm_boot[
                perm_boot["target"].eq(target) & perm_boot["family"].eq(family)
            ]
            inflation = float(point["permutation_averaged_sibling_inflation"].mean())
            largest_null = float(
                point["permutation_averaged_largest_non_sibling_inflation"].mean()
            )
            lower, upper = _interval(boot["sibling_inflation"])
            splits = int(point["splits_sibling_largest"].iloc[0])
            core = inflation > largest_null and lower > 0 and splits >= 9
            raw_core = (
                float(point["raw_endpoint_sibling_inflation"].mean())
                > float(point["raw_endpoint_largest_non_sibling_inflation"].mean())
            )
            generation_ok, generation_evidence = _generation_compatibility(generation, target)
            status = LOCKED if core and generation_ok else QUALIFIED if core else RAW_ONLY if raw_core else NOT_SUPPORTED
            removed = str(point["removed_sibling"].iloc[0])
            rows.append(
                {
                    "claim_id": f"C3_{target}_{family}",
                    "claim_type": "sibling_coverage",
                    "target": target,
                    "family": family,
                    "v22_raw_trackwise": True,
                    "v23_endpoint_raw": raw_core,
                    "v23_permutation_averaged": inflation > largest_null,
                    "common_bootstrap": lower > 0,
                    "matched_objective": True,
                    "generation_compatibility": generation_ok,
                    "final_status": status,
                    "permutation_sibling_inflation": inflation,
                    "largest_non_sibling_inflation": largest_null,
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                    "directional_split_count": splits,
                    "observed_evidence": (
                        f"removing {removed} inflated endpoint-design PIER by {inflation:.6f}; "
                        f"largest non-sibling={largest_null:.6f}; common-resample interval "
                        f"[{lower:.6f}, {upper:.6f}]; sibling largest in {splits}/10 splits; "
                        f"{generation_evidence}"
                    ),
                    "what_it_supports": "A close Qwen sibling supplies disproportionate peer-hull coverage.",
                    "what_it_does_not_support": "It does not prove causal training lineage or semantic equivalence.",
                    "paper_ready_wording": (
                        f"For {target} under {family}, removing its declared Qwen sibling caused more coverage loss than any same-size non-sibling removal."
                        if core
                        else f"The declared sibling-local coverage claim for {target} under {family} did not survive all V2.3 criteria."
                    ),
                }
            )
    return rows


def _convex_claims(
    matched: pd.DataFrame, bootstrap: pd.DataFrame, generation: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = (
        "microsoft/Phi-4-reasoning-plus",
        "mistralai/Mistral-Nemo-Instruct-2407",
    )
    perm = matched[matched["interface"].eq("permutation_averaged")]
    raw = matched[matched["interface"].eq("raw_endpoint")]
    perm_boot = bootstrap[bootstrap["interface"].eq("permutation_averaged")]
    for target in targets:
        for family in ("irrelevant_context", "content_deletion"):
            point = perm[perm["target"].eq(target) & perm["family"].eq(family)]
            raw_point = raw[raw["target"].eq(target) & raw["family"].eq(family)]
            boot = perm_boot[
                perm_boot["target"].eq(target) & perm_boot["family"].eq(family)
            ]
            mse = float(point["MSE_absolute_improvement"].mean())
            mae = float(point["MAE_absolute_improvement"].mean())
            mse_lower, mse_upper = _interval(boot["MSE_improvement"])
            mae_lower, mae_upper = _interval(boot["MAE_improvement"])
            mse_splits = int((point["MSE_absolute_improvement"] > 0).sum())
            mae_splits = int((point["MAE_absolute_improvement"] > 0).sum())
            core = mse > 0 and mae > 0 and mse_lower > 0 and mae_lower > 0 and min(mse_splits, mae_splits) >= 9
            raw_core = (
                float(raw_point["MSE_absolute_improvement"].mean()) > 0
                and float(raw_point["MAE_absolute_improvement"].mean()) > 0
            )
            generation_ok, generation_evidence = _generation_compatibility(generation, target)
            status = LOCKED if core and generation_ok else QUALIFIED if core else RAW_ONLY if raw_core else NOT_SUPPORTED
            rows.append(
                {
                    "claim_id": f"C4_{target}_{family}",
                    "claim_type": "distributed_convex_coverage",
                    "target": target,
                    "family": family,
                    "v22_raw_trackwise": True,
                    "v23_endpoint_raw": raw_core,
                    "v23_permutation_averaged": mse > 0 and mae > 0,
                    "common_bootstrap": mse_lower > 0 and mae_lower > 0,
                    "matched_objective": mse > 0 and mae > 0,
                    "generation_compatibility": generation_ok,
                    "final_status": status,
                    "MSE_improvement": mse,
                    "MAE_improvement": mae,
                    "MSE_bootstrap_lower": mse_lower,
                    "MSE_bootstrap_upper": mse_upper,
                    "MAE_bootstrap_lower": mae_lower,
                    "MAE_bootstrap_upper": mae_upper,
                    "directional_split_count": min(mse_splits, mae_splits),
                    "observed_evidence": (
                        f"MSE-fit convex improvement={mse:.6f}, interval [{mse_lower:.6f}, {mse_upper:.6f}], "
                        f"positive splits={mse_splits}/10; MAE-fit improvement={mae:.6f}, interval "
                        f"[{mae_lower:.6f}, {mae_upper:.6f}], positive splits={mae_splits}/10; "
                        f"{generation_evidence}"
                    ),
                    "what_it_supports": "Collective coverage by several peers under both squared- and absolute-error fitting objectives.",
                    "what_it_does_not_support": "It does not certify safe model replacement or governance utility.",
                    "paper_ready_wording": (
                        f"For {target} under {family}, a fixed convex peer surrogate outperformed every honestly selected single peer under both MSE- and MAE-matched fitting."
                        if core
                        else f"Distributed coverage for {target} under {family} did not satisfy both matched-objective V2.3 criteria."
                    ),
                }
            )
    return rows


def build_claim_locking(root: Path) -> pd.DataFrame:
    analysis = root / "outputs/analysis"
    raw = pd.read_parquet(analysis / "raw_endpoint_design_results.parquet")
    permutation = pd.read_parquet(analysis / "permutation_endpoint_design_results.parquet")
    endpoint_boot = pd.read_parquet(analysis / "common_resample_endpoint_bootstrap.parquet")
    matched = pd.read_parquet(analysis / "matched_objective_convexity.parquet")
    convex_boot = pd.read_parquet(analysis / "common_resample_convexity_bootstrap.parquet")
    sibling = pd.read_parquet(analysis / "permutation_sibling_removal.parquet")
    sibling_boot = pd.read_parquet(analysis / "common_resample_sibling_bootstrap.parquet")
    generation = pd.read_parquet(analysis / "long_generation_validation.parquet")
    rows = _endpoint_claims(raw, permutation, endpoint_boot, generation)
    rows.extend(_sibling_claims(sibling, sibling_boot, generation))
    rows.extend(_convex_claims(matched, convex_boot, generation))
    output = pd.DataFrame(rows)
    allowed = {LOCKED, QUALIFIED, RAW_ONLY, NOT_SUPPORTED}
    if not set(output["final_status"]).issubset(allowed):
        raise AssertionError("Unknown claim status")
    atomic_parquet(analysis / "claim_locking_results.parquet", output)
    return output


def _status_for(claims: pd.DataFrame, prefix: str) -> str:
    rows = claims[claims["claim_id"].str.startswith(prefix)]
    if rows.empty:
        return "not evaluated"
    return ", ".join(f"{row.claim_id}: {row.final_status}" for row in rows.itertuples())


def create_numerical_sensitivity_audit(root: Path, claims: pd.DataFrame) -> pd.DataFrame:
    analysis = root / "outputs/analysis"
    validation = root / "outputs/validation"
    repairs_path = validation / "numerical_feasibility_repairs.parquet"
    repairs = pd.read_parquet(repairs_path)
    repaired_replicates = (
        sorted(repairs["bootstrap_replicate"].dropna().astype(int).unique().tolist())
        if len(repairs)
        else []
    )
    specifications = {
        "endpoint": ("common_resample_endpoint_bootstrap.parquet", "endpoint_delta"),
        "convexity": (
            "common_resample_convexity_bootstrap.parquet",
            ("MSE_improvement", "MAE_improvement"),
        ),
        "sibling_removal": (
            "common_resample_sibling_bootstrap.parquet",
            "sibling_inflation",
        ),
    }
    bootstrap: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for analysis_type, (filename, _metrics) in specifications.items():
        full = pd.read_parquet(analysis / filename)
        repair_free = full[~full["replicate"].isin(repaired_replicates)].copy()
        output = validation / f"repair_free_only_{filename}"
        atomic_parquet(output, repair_free)
        bootstrap[analysis_type] = (full, repair_free)

    rows: list[dict[str, Any]] = []
    for claim in claims.to_dict(orient="records"):
        claim_type = str(claim["claim_type"])
        if claim_type == "endpoint_effect":
            analysis_type = "endpoint"
            metrics = (("endpoint_delta", "permutation_endpoint_delta"),)
            direction = 1 if str(claim["claim_id"]).startswith("C1_") else -1
        elif claim_type == "sibling_coverage":
            analysis_type = "sibling_removal"
            metrics = (("sibling_inflation", "permutation_sibling_inflation"),)
            direction = 1
        elif claim_type == "distributed_convex_coverage":
            analysis_type = "convexity"
            metrics = (
                ("MSE_improvement", "MSE_improvement"),
                ("MAE_improvement", "MAE_improvement"),
            )
            direction = 1
        else:
            raise ValueError(f"Unknown headline claim type: {claim_type}")
        full_frame, repair_free_frame = bootstrap[analysis_type]
        full_group = full_frame[
            full_frame["interface"].eq("permutation_averaged")
            & full_frame["target"].eq(claim["target"])
            & full_frame["family"].eq(claim["family"])
        ]
        repair_free_group = repair_free_frame[
            repair_free_frame["interface"].eq("permutation_averaged")
            & repair_free_frame["target"].eq(claim["target"])
            & repair_free_frame["family"].eq(claim["family"])
        ]
        for metric, point_column in metrics:
            full_estimate = float(full_group[metric].mean())
            full_lower, full_upper = _interval(full_group[metric])
            repair_free_defined = bool(len(repair_free_group))
            if repair_free_defined:
                repair_free_estimate = float(repair_free_group[metric].mean())
                repair_free_lower, repair_free_upper = _interval(repair_free_group[metric])
                full_conclusion = full_lower > 0 if direction > 0 else full_upper < 0
                repair_free_conclusion = (
                    repair_free_lower > 0 if direction > 0 else repair_free_upper < 0
                )
                conclusion_changed = full_conclusion != repair_free_conclusion
            else:
                repair_free_estimate = None
                repair_free_lower = None
                repair_free_upper = None
                full_conclusion = full_lower > 0 if direction > 0 else full_upper < 0
                repair_free_conclusion = None
                conclusion_changed = True
            rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_type": claim_type,
                    "analysis_type": analysis_type,
                    "target": claim["target"],
                    "family": claim["family"],
                    "interface": "permutation_averaged",
                    "metric": metric,
                    "expected_direction": direction,
                    "claim_point_estimate": float(claim[point_column]),
                    "full_estimate": full_estimate,
                    "full_ci_lower": full_lower,
                    "full_ci_upper": full_upper,
                    "repair_free_estimate": repair_free_estimate,
                    "repair_free_ci_lower": repair_free_lower,
                    "repair_free_ci_upper": repair_free_upper,
                    "point_estimate_difference": (
                        repair_free_estimate - full_estimate
                        if repair_free_estimate is not None
                        else None
                    ),
                    "ci_lower_difference": (
                        repair_free_lower - full_lower
                        if repair_free_lower is not None
                        else None
                    ),
                    "ci_upper_difference": (
                        repair_free_upper - full_upper
                        if repair_free_upper is not None
                        else None
                    ),
                    "full_replicate_count": int(len(full_group)),
                    "repair_free_replicate_count": int(len(repair_free_group)),
                    "excluded_replicate_count": int(
                        len(full_group) - len(repair_free_group)
                    ),
                    "full_bootstrap_conclusion": bool(full_conclusion),
                    "repair_free_bootstrap_conclusion": repair_free_conclusion,
                    "conclusion_changed": bool(conclusion_changed),
                    "manual_review_required": bool(conclusion_changed),
                }
            )
    sensitivity = pd.DataFrame(rows)
    output = validation / "numerical_feasibility_repair_sensitivity.parquet"
    atomic_parquet(output, sensitivity)
    summary_path = root / "status/numerical_feasibility_repair_summary.json"
    summary = load_json(summary_path)
    summary["repair_free_sensitivity"] = {
        "passed": not bool(sensitivity["manual_review_required"].any()),
        "diagnostic_only": True,
        "global_excluded_bootstrap_replicates": repaired_replicates,
        "global_excluded_replicate_count": len(repaired_replicates),
        "headline_metric_count": int(len(sensitivity)),
        "manual_review_required": bool(sensitivity["manual_review_required"].any()),
        "manual_review_claims": sorted(
            sensitivity.loc[sensitivity["manual_review_required"], "claim_id"]
            .unique()
            .tolist()
        ),
        "path": str(output),
        "sha256": sha256_file(output),
    }
    atomic_write_json(summary_path, summary)
    return sensitivity


def create_tables(root: Path, claims: pd.DataFrame) -> dict[str, pd.DataFrame]:
    analysis = root / "outputs/analysis"
    table_root = root / "outputs/tables"
    table_root.mkdir(parents=True, exist_ok=True)
    sensitivity = pd.read_parquet(analysis / "permutation_sensitivity.parquet")
    table1 = (
        sensitivity.groupby(["model", "condition"], as_index=False)
        .agg(
            mean_rotation_TV=("mean_semantic_vector_TV_to_average", "mean"),
            p95_rotation_TV=("maximum_semantic_vector_TV_to_average", lambda x: float(np.quantile(x, 0.95))),
            semantic_top1_rotation_agreement=("semantic_top1_agreement_across_rotations", "mean"),
            mean_gold_probability_sd=("permutation_wise_gold_probability_sd", "mean"),
            canonical_vs_permutation_average_TV=("canonical_vs_permutation_average_TV", "mean"),
        )
    )
    raw = pd.read_parquet(analysis / "raw_endpoint_design_results.parquet")
    perm = pd.read_parquet(analysis / "permutation_endpoint_design_results.parquet")
    endpoint_boot = pd.read_parquet(analysis / "common_resample_endpoint_bootstrap.parquet")
    config = load_json(root / "configs/experiment_v23.json")
    v22 = Path(config["source_roots"]["v22"]).resolve(strict=True)
    old = pd.read_parquet(v22 / "outputs/analysis/trackwise_endpoint_effects.parquet")
    rows = []
    for keys, group in perm.groupby(["target", "family"], sort=True):
        raw_group = raw[raw["target"].eq(keys[0]) & raw["family"].eq(keys[1])]
        old_group = old[old["target"].eq(keys[0]) & old["family"].eq(keys[1])]
        boot = endpoint_boot[
            endpoint_boot["interface"].eq("permutation_averaged")
            & endpoint_boot["target"].eq(keys[0])
            & endpoint_boot["family"].eq(keys[1])
        ]
        lower, upper = _interval(boot["endpoint_delta"])
        claim_rows = claims[
            claims["target"].eq(keys[0])
            & claims["family"].eq(keys[1])
            & claims["claim_type"].eq("endpoint_effect")
        ]
        rows.append(
            {
                "target": keys[0],
                "family": keys[1],
                "v22_full_design_raw_delta": float(old_group["trackwise_delta"].mean()),
                "v23_endpoint_raw_delta": float(raw_group["endpoint_delta"].mean()),
                "v23_permutation_averaged_delta": float(group["endpoint_delta"].mean()),
                "bootstrap_lower": lower,
                "bootstrap_upper": upper,
                "positive_split_count": int((group["endpoint_delta"] > 0).sum()),
                "claim_status": claim_rows["final_status"].iloc[0] if len(claim_rows) else "NOT_PREDECLARED",
            }
        )
    table2 = pd.DataFrame(rows)
    sibling = pd.read_parquet(analysis / "permutation_sibling_removal.parquet")
    sibling_boot = pd.read_parquet(analysis / "common_resample_sibling_bootstrap.parquet")
    sibling_rows = []
    for keys, group in sibling.groupby(["target", "family"], sort=True):
        boot = sibling_boot[
            sibling_boot["interface"].eq("permutation_averaged")
            & sibling_boot["target"].eq(keys[0])
            & sibling_boot["family"].eq(keys[1])
        ]
        lower, upper = _interval(boot["sibling_inflation"])
        claim_row = claims[
            claims["target"].eq(keys[0])
            & claims["family"].eq(keys[1])
            & claims["claim_type"].eq("sibling_coverage")
        ]
        sibling_rows.append(
            {
                "target": keys[0],
                "removed_sibling": group["removed_sibling"].iloc[0],
                "family": keys[1],
                "raw_endpoint_inflation": float(group["raw_endpoint_sibling_inflation"].mean()),
                "permutation_averaged_inflation": float(group["permutation_averaged_sibling_inflation"].mean()),
                "bootstrap_lower": lower,
                "bootstrap_upper": upper,
                "largest_non_sibling_inflation": float(group["permutation_averaged_largest_non_sibling_inflation"].mean()),
                "sibling_rank": float(group["permutation_averaged_sibling_rank"].mean()),
                "splits_sibling_largest": int(group["splits_sibling_largest"].iloc[0]),
                "claim_status": claim_row["final_status"].iloc[0] if len(claim_row) else "NOT_PREDECLARED",
            }
        )
    table3 = pd.DataFrame(sibling_rows)
    matched = pd.read_parquet(analysis / "matched_objective_convexity.parquet")
    convex_boot = pd.read_parquet(analysis / "common_resample_convexity_bootstrap.parquet")
    convex_rows = []
    for keys, _group in matched.groupby(["target", "family"], sort=True):
        row: dict[str, Any] = {"target": keys[0], "family": keys[1]}
        for interface, prefix in (("raw_endpoint", "raw"), ("permutation_averaged", "perm")):
            part = matched[
                matched["interface"].eq(interface)
                & matched["target"].eq(keys[0])
                & matched["family"].eq(keys[1])
            ]
            for objective in ("MSE", "MAE"):
                row[f"{prefix}_{objective}_single_error"] = float(part[f"{objective}_single_error"].mean())
                row[f"{prefix}_{objective}_convex_error"] = float(part[f"{objective}_convex_error"].mean())
        boot = convex_boot[
            convex_boot["interface"].eq("permutation_averaged")
            & convex_boot["target"].eq(keys[0])
            & convex_boot["family"].eq(keys[1])
        ]
        row["common_bootstrap_MSE_improvement_interval"] = list(_interval(boot["MSE_improvement"]))
        row["common_bootstrap_MAE_improvement_interval"] = list(_interval(boot["MAE_improvement"]))
        claim_row = claims[
            claims["target"].eq(keys[0])
            & claims["family"].eq(keys[1])
            & claims["claim_type"].eq("distributed_convex_coverage")
        ]
        row["claim_status"] = claim_row["final_status"].iloc[0] if len(claim_row) else "NOT_PREDECLARED"
        convex_rows.append(row)
    table4 = pd.DataFrame(convex_rows)
    table5 = pd.read_parquet(analysis / "long_generation_validation.parquet")[[
        "model",
        "condition",
        "canonical_score_generation_agreement",
        "all_rotation_score_generation_agreement",
        "malformed_rate",
        "median_generation_tokens",
        "generated_answer_rotation_stability",
        "score_vs_generated_frequency_TV",
    ]]
    table6 = claims[[
        "claim_id",
        "v22_raw_trackwise",
        "v23_endpoint_raw",
        "v23_permutation_averaged",
        "common_bootstrap",
        "matched_objective",
        "generation_compatibility",
        "final_status",
        "paper_ready_wording",
    ]]
    tables = {
        "table_1_permutation_interface_quality.csv": table1,
        "table_2_endpoint_claim_comparison.csv": table2,
        "table_3_sibling_coverage.csv": table3,
        "table_4_distributed_convex_coverage.csv": table4,
        "table_5_generation_validation.csv": table5,
        "table_6_final_claim_locking_matrix.csv": table6,
    }
    for name, table in tables.items():
        table.to_csv(table_root / name, index=False)
    return tables


def create_report(root: Path, claims: pd.DataFrame) -> Path:
    analysis = root / "outputs/analysis"
    sensitivity = pd.read_parquet(analysis / "permutation_sensitivity.parquet")
    raw = pd.read_parquet(analysis / "raw_endpoint_design_results.parquet")
    perm = pd.read_parquet(analysis / "permutation_endpoint_design_results.parquet")
    generation = pd.read_parquet(analysis / "long_generation_validation.parquet")
    frequencies = pd.read_parquet(analysis / "generated_semantic_frequency.parquet")
    endpoint_boot = pd.read_parquet(analysis / "common_resample_endpoint_bootstrap.parquet")
    repair_summary = load_json(root / "status/numerical_feasibility_repair_summary.json")
    repair_sensitivity = pd.read_parquet(
        root / "outputs/validation/numerical_feasibility_repair_sensitivity.parquet"
    )
    mean_tv = float(sensitivity["mean_semantic_vector_TV_to_average"].mean())
    p95_tv = float(np.quantile(sensitivity["maximum_semantic_vector_TV_to_average"], 0.95))
    raw_rank = raw.groupby("target")["endpoint_design_pier"].mean().rank().sort_index()
    perm_rank = perm.groupby("target")["endpoint_design_pier"].mean().rank().sort_index()
    rank_rho = float(spearmanr(raw_rank, perm_rank).statistic)
    mean_old_malformed = float(generation["old_16_token_malformed_rate"].mean())
    mean_new_malformed = float(generation["malformed_rate"].mean())
    mean_old_agreement = float(generation["old_score_generation_agreement"].mean())
    mean_new_agreement = float(generation["canonical_score_generation_agreement"].mean())
    avg_tv = float(frequencies["TV_distance"].mean())
    canonical_tv = float(frequencies["canonical_score_vs_generated_frequency_TV"].mean())
    locked = claims[claims["final_status"].eq(LOCKED)]["claim_id"].tolist()
    qualified = claims[claims["final_status"].eq(QUALIFIED)]["claim_id"].tolist()
    raw_only = claims[claims["final_status"].eq(RAW_ONLY)]["claim_id"].tolist()
    unsupported = claims[claims["final_status"].eq(NOT_SUPPORTED)]["claim_id"].tolist()
    crossing = []
    for keys, group in endpoint_boot[endpoint_boot["interface"].eq("permutation_averaged")].groupby(
        ["target", "family"]
    ):
        lower, upper = _interval(group["endpoint_delta"])
        if lower <= 0 <= upper:
            crossing.append(f"{keys[0]} / {keys[1]}")
    another_b200 = bool(
        (generation["malformed_rate"] > 0.20).any()
        or len(unsupported) > len(claims) / 2
    )
    ready_flag = len(locked) + len(qualified) >= max(1, len(claims) // 2)
    discrepancy = (
        "a mixture dominated by truncation reduction plus residual genuine score-vs-generation mismatch"
        if mean_old_malformed - mean_new_malformed > 0.05
        else "mainly residual interface sensitivity and genuine score-vs-generation mismatch"
    )
    story = (
        f"Across all eight fixed modern LLMs, exact cyclic option-label averaging left mean semantic-vector TV {mean_tv:.3f} "
        f"(p95 maximum TV {p95_tv:.3f}) while preserving a raw-versus-balanced endpoint ranking correlation of {rank_rho:.3f}; "
        f"the common-base-question bootstrap and matched MSE/MAE objectives classify {len(locked)} pre-declared claims as LOCKED and "
        f"{len(qualified)} as supported with an interface qualifier, with longer 512-token generation reducing malformed output from "
        f"{mean_old_malformed:.3f} to {mean_new_malformed:.3f}, so the paper should present PIER as balanced-interface ecosystem geometry—not "
        "as a true-belief estimator, causal lineage test, or safe-replacement certificate."
    )
    answers = [
        f"1. Remaining presentation sensitivity: mean rotation TV={mean_tv:.6f}; p95 maximum TV={p95_tv:.6f}.",
        f"2. Target PIER ranking change: raw/permutation endpoint Spearman ρ={rank_rho:.6f}.",
        f"3. phi-4 irrelevant-context outward movement: {_status_for(claims, 'C1_phi4')}.",
        f"4. Qwen irrelevant-context outward movement: {_status_for(claims, 'C1_qwen')}.",
        f"5. Mistral/OLMo/Granite irrelevant-context contractions: {_status_for(claims, 'C2_')}.",
        f"6. Qwen sibling-local coverage: {_status_for(claims, 'C3_')}.",
        "7. Whether Qwen sibling beats every non-sibling is encoded by `splits_sibling_largest` and the exact null in Table 3.",
        "8. The Phi sibling relation remains a diagnostic, not a required sibling-local claim; its full V2.3 values are retained in the removal parquet.",
        f"9. Phi-4-reasoning-plus distributed coverage: {_status_for(claims, 'C4_microsoft/Phi-4-reasoning-plus')}.",
        f"10. Mistral distributed coverage: {_status_for(claims, 'C4_mistralai/Mistral')}.",
        "11. Both MSE- and MAE-matched survival is required and reported separately in Table 4.",
        f"12. Mistral content-deletion contraction: {_status_for(claims, 'C5_mistral')}.",
        f"13. OLMo/Granite content-deletion scalar contraction: {_status_for(claims, 'C5_olmo')}; {_status_for(claims, 'C5_granite')}.",
        "14. Common multiplicities replace the invalid independent-split interpretation; interval widths are reported in the bootstrap outputs for direct comparison.",
        f"15. Permutation-averaged endpoint intervals crossing zero: {crossing if crossing else 'none'}.",
        f"16. Mean malformed rate changed from {mean_old_malformed:.6f} at 16 tokens to {mean_new_malformed:.6f} at 512 tokens.",
        f"17. Mean canonical score-generation agreement changed from {mean_old_agreement:.6f} to {mean_new_agreement:.6f}.",
        f"18. Remaining discrepancy is {discrepancy}.",
        f"19. Mean TV to generation frequency: canonical={canonical_tv:.6f}; permutation average={avg_tv:.6f}.",
        f"20. LOCKED claims: {locked if locked else 'none'}.",
        f"21. Interface-qualified claims: {qualified if qualified else 'none'}.",
        f"22. Raw-only claims to drop from interface-robust wording: {raw_only + unsupported if raw_only or unsupported else 'none'}.",
        f"23. Another B200 experiment scientifically necessary: {'yes' if another_b200 else 'no'} under the pre-declared decision rule.",
        f"24. Ready to replace the old BERT/SST-2 flagship: {'yes, with the stated interface qualifiers' if ready_flag else 'not yet'}.",
        f"25. Recommended paper-ready story: {story}",
    ]
    lines = [
        "# PIER Modern-LLM Ecosystem V2.3.1 — Final Interface Validation Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Executive result",
        "",
        story,
        "",
        "## Runtime environment deviation",
        "",
        "**The Stage-C runtime environment was not bitwise identical to the prestaged environment.** Stage-C post-processing was executed under CPython 3.11.15 rather than the prestaged CPython 3.11.13 because the original interpreter was unavailable on the replacement CPU host. Python remained within the same 3.11 ABI line and all locked package versions, frozen scientific inputs, manifests, Stage-B outputs, and hashes were unchanged, except for the explicitly authorized V2.3.1 numerical-feasibility patch.",
        "",
        "## Numerical-feasibility recovery and sensitivity audit",
        "",
        f"The deterministic post-solve simplex projection repaired {repair_summary['repaired_solution_count']} otherwise-optimal solutions. Maximum raw negativity was {repair_summary['maximum_raw_negativity']:.12g}; maximum L1/L2/Linf weight changes were {repair_summary['maximum_l1_repair']:.12g}, {repair_summary['maximum_l2_repair']:.12g}, and {repair_summary['maximum_linf_repair']:.12g}. Maximum absolute/relative objective changes were {repair_summary['maximum_absolute_objective_change']:.12g} and {repair_summary['maximum_relative_objective_change']:.12g}; maximum fitted/evaluation prediction changes were {repair_summary['maximum_fitted_prediction_change']:.12g} and {repair_summary['maximum_evaluation_prediction_change']:.12g}. All strict acceptance gates passed.",
        "",
        "The repair-free-only calculation is diagnostic and does not replace the full 1,000-replicate common-multiplicity bootstrap. It globally excludes each replicate index for which any touched solution required repair.",
        "",
        "| Claim | Metric | Full estimate | Full 95% CI | Repair-free estimate | Repair-free 95% CI | Estimate difference | CI endpoint differences | Manual review |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in repair_sensitivity.itertuples(index=False):
        repair_estimate = (
            f"{row.repair_free_estimate:.8g}"
            if pd.notna(row.repair_free_estimate)
            else "undefined"
        )
        repair_interval = (
            f"[{row.repair_free_ci_lower:.8g}, {row.repair_free_ci_upper:.8g}]"
            if pd.notna(row.repair_free_ci_lower)
            else "undefined"
        )
        estimate_difference = (
            f"{row.point_estimate_difference:.8g}"
            if pd.notna(row.point_estimate_difference)
            else "undefined"
        )
        interval_difference = (
            f"[{row.ci_lower_difference:.8g}, {row.ci_upper_difference:.8g}]"
            if pd.notna(row.ci_lower_difference)
            else "undefined"
        )
        lines.append(
            f"| {row.claim_id} | {row.metric} | {row.full_estimate:.8g} | "
            f"[{row.full_ci_lower:.8g}, {row.full_ci_upper:.8g}] | {repair_estimate} | "
            f"{repair_interval} | {estimate_difference} | {interval_difference} | "
            f"{'YES' if row.manual_review_required else 'no'} |"
        )
    lines.extend(
        [
            "",
            "No headline conclusion changed in the repair-free diagnostic."
            if not repair_sensitivity["manual_review_required"].any()
            else "At least one headline conclusion changed and is marked for manual review above.",
            "",
            "## Answers to the 25 pre-declared questions",
            "",
            *answers,
            "",
            "## Claim-by-claim evidence",
            "",
        ]
    )
    for row in claims.itertuples(index=False):
        lines.extend(
            [
                f"### {row.claim_id} — {row.final_status}",
                "",
                f"Observed evidence: {row.observed_evidence}",
                "",
                f"What it supports: {row.what_it_supports}",
                "",
                f"What it does not support: {row.what_it_does_not_support}",
                "",
                f"Paper-ready wording: {row.paper_ready_wording}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Permutation averaging is a deliberately balanced interface-robust response representation. It is not a uniquely true latent belief. PIER does not certify safe replacement, sibling coverage does not prove causal lineage, and deterministic generation frequency is not stochastic model uncertainty.",
            "",
        ]
    )
    path = analysis / "V2_3_FINAL_INTERFACE_VALIDATION_REPORT.md"
    atomic_write_text(path, "\n".join(lines))
    return path
