from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .bootstrap import read_bootstrap_analysis
from .core import high_dose, summarize_split_effects
from .utils import PHYSICAL_ROOT, atomic_csv, atomic_parquet


def select_max_after_split_aggregation(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    condition_column: str,
    value_column: str,
) -> pd.DataFrame:
    aggregated = (
        frame.groupby([*group_columns, condition_column], as_index=False)[value_column]
        .mean()
    )
    selected = aggregated.loc[
        aggregated.groupby(group_columns)[value_column].idxmax()
    ]
    return selected.reset_index(drop=True)


def _bootstrap_summary(
    analysis_type: str,
    metric: str,
) -> pd.DataFrame:
    frame = read_bootstrap_analysis(analysis_type)
    rows: list[dict[str, Any]] = []
    for (target, family), group in frame.groupby(["target", "family"], sort=True):
        values = group[metric].astype(float).to_numpy()
        rows.append(
            {
                "target": target,
                "family": family,
                "bootstrap_metric": metric,
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_lower_95": float(np.quantile(values, 0.025)),
                "bootstrap_upper_95": float(np.quantile(values, 0.975)),
                "bootstrap_probability_gt_zero": float(np.mean(values > 0)),
                "bootstrap_probability_lt_zero": float(np.mean(values < 0)),
                "bootstrap_replicates": len(values),
            }
        )
    return pd.DataFrame(rows)


def _add_stability(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    same_sign = result[["positive_split_count", "negative_split_count"]].max(axis=1)
    interval_excludes_zero = (result["bootstrap_lower_95"] > 0) | (
        result["bootstrap_upper_95"] < 0
    )
    result["same_sign_split_count"] = same_sign.astype(int)
    result["evidence_status"] = np.where(
        (same_sign >= 9) & interval_excludes_zero, "stable", "exploratory"
    )
    return result


def endpoint_table() -> pd.DataFrame:
    effects = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/paired_endpoint_effects.parquet"
    )
    split = summarize_split_effects(
        effects, metric="delta_absolute", group_columns=["target", "family"]
    ).rename(
        columns={
            "mean": "mean_delta_absolute",
            "median": "median_delta_absolute",
            "standard_deviation": "delta_standard_deviation",
            "minimum": "minimum_delta_absolute",
            "maximum": "maximum_delta_absolute",
        }
    )
    split = split.drop(columns="metric")
    bootstrap = _bootstrap_summary("endpoint_effects", "delta_absolute")
    table = _add_stability(
        split.merge(bootstrap, on=["target", "family"], validate="one_to_one")
    )
    contrast = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/endpoint_stress_contrasts.parquet"
    )
    contrast_summary = (
        contrast.groupby("target")["stress_specific_contrast"]
        .agg(
            stress_specific_contrast_mean="mean",
            stress_specific_contrast_median="median",
            stress_specific_contrast_lower_95=lambda values: values.quantile(0.025),
            stress_specific_contrast_upper_95=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    return table.merge(contrast_summary, on="target", validate="many_to_one")


def track_table() -> pd.DataFrame:
    frame = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/track_specific_pier.parquet")
    high = frame[
        frame.apply(lambda row: math.isclose(float(row["dose"]), high_dose(row["family"])), axis=1)
    ].copy()
    track_delta = (
        high.groupby(["target", "family", "track"])["track_endpoint_delta"]
        .mean()
        .unstack("track")
        .rename(columns=lambda value: f"mean_track_{int(value)}_endpoint_delta")
        .reset_index()
    )
    summary = (
        high.drop_duplicates(["target", "family", "split_seed"])
        .groupby(["target", "family"], as_index=False)
        .agg(
            mean_aggregate_endpoint_delta=("aggregate_endpoint_delta", "mean"),
            mean_track_cancellation_gap=("track_cancellation_gap", "mean"),
            maximum_track_cancellation_gap=("track_cancellation_gap", "max"),
            mean_between_track_standard_deviation=(
                "between_track_standard_deviation",
                "mean",
            ),
            minimum_same_sign_track_count=("same_sign_track_count", "min"),
            mean_same_sign_track_count=("same_sign_track_count", "mean"),
        )
    )
    sensitivity = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/track_aggregation_sensitivity.parquet"
    )
    sensitivity_summary = sensitivity.groupby(["target", "family"], as_index=False).agg(
        mean_trackwise_fit_endpoint_delta=("trackwise_fit_endpoint_delta", "mean"),
        trackwise_endpoint_sign_agreement=(
            "trackwise_fit_endpoint_delta",
            lambda values: float(
                np.mean(
                    np.sign(values.to_numpy())
                    == np.sign(
                        sensitivity.loc[values.index, "primary_endpoint_delta"].to_numpy()
                    )
                )
            ),
        ),
        mean_weight_l1_difference=("weight_l1_difference", "mean"),
        mean_projected_response_difference=(
            "mean_absolute_projected_response_difference",
            "mean",
        ),
    )
    return summary.merge(track_delta, on=["target", "family"], validate="one_to_one").merge(
        sensitivity_summary, on=["target", "family"], validate="one_to_one"
    )


def sibling_table() -> pd.DataFrame:
    corrected = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/peer_removal_null_corrected.parquet"
    )
    exact = corrected[corrected["removal_type"].eq("exact_sibling")]
    summary = exact.groupby(["target", "family"], as_index=False).agg(
        mean_observed_inflation=("observed_inflation", "mean"),
        median_observed_inflation=("observed_inflation", "median"),
        mean_null_median_inflation=("null_median_inflation", "mean"),
        mean_excess_over_null_median=("observed_excess_over_null_median", "mean"),
        mean_observed_percentile=("observed_percentile", "mean"),
        splits_exceeding_every_null=("exceeds_every_null", "sum"),
        null_subset_count=("null_subset_count", "first"),
    )
    influence = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/single_peer_removal_influence.parquet"
    )
    sibling = influence[influence["is_exact_sibling"]]
    influence_summary = sibling.groupby(["target", "family"], as_index=False).agg(
        mean_sibling_removal_rank=("removal_rank", "mean"),
        splits_sibling_ranked_first=("removal_rank", lambda values: int(np.sum(values == 1))),
        mean_sibling_minus_best_non_sibling=(
            "inflation_minus_best_non_sibling",
            "mean",
        ),
    )
    bootstrap = _bootstrap_summary("sibling_removal", "sibling_removal_inflation")
    split_counts = exact.groupby(["target", "family"])["observed_inflation"].agg(
        positive_split_count=lambda values: int(np.sum(values > 0)),
        negative_split_count=lambda values: int(np.sum(values < 0)),
    ).reset_index()
    table = summary.merge(influence_summary, on=["target", "family"], validate="one_to_one")
    table = table.merge(bootstrap, on=["target", "family"], validate="one_to_one")
    table = table.merge(split_counts, on=["target", "family"], validate="one_to_one")
    return _add_stability(table)


def convexity_table() -> pd.DataFrame:
    frame = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/convex_vs_single_corrected.parquet"
    )
    summary = frame.groupby(["target", "family"], as_index=False).agg(
        mean_single_error=("single_error", "mean"),
        mean_convex_error=("convex_error", "mean"),
        mean_absolute_improvement=("absolute_improvement", "mean"),
        median_absolute_improvement=("absolute_improvement", "median"),
        improvement_standard_deviation=("absolute_improvement", "std"),
        mean_relative_improvement=("relative_improvement", "mean"),
        median_relative_improvement=("relative_improvement", "median"),
        mean_gap_ratio=("gap_ratio", "mean"),
        convex_wins_split_count=("absolute_improvement", lambda values: int(np.sum(values > 0))),
        positive_split_count=("absolute_improvement", lambda values: int(np.sum(values > 0))),
        negative_split_count=("absolute_improvement", lambda values: int(np.sum(values < 0))),
    )
    frequencies = (
        frame.groupby(["target", "family"])["fit_selected_single_peer"]
        .apply(lambda values: json.dumps(values.value_counts().to_dict(), sort_keys=True))
        .rename("fit_selected_single_peer_frequencies")
        .reset_index()
    )
    bootstrap = _bootstrap_summary("convex_vs_single", "absolute_improvement")
    table = summary.merge(frequencies, on=["target", "family"], validate="one_to_one")
    table = table.merge(bootstrap, on=["target", "family"], validate="one_to_one")
    return _add_stability(table)


def calibration_table() -> pd.DataFrame:
    frame = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/calibrated_dose_trajectories.parquet"
    )
    endpoints = frame.drop_duplicates(["target", "family", "split_seed"])
    summary = endpoints.groupby(["target", "family"], as_index=False).agg(
        mean_raw_endpoint_delta=("raw_endpoint_delta", "mean"),
        mean_calibrated_endpoint_delta=("calibrated_endpoint_delta", "mean"),
        endpoint_sign_agreement_rate=("endpoint_effect_sign_agreement", "mean"),
        mean_target_temperature=("target_temperature", "mean"),
        mean_rank_shift=("rank_shift", "mean"),
    )
    summary["calibration_induced_endpoint_change"] = (
        summary["mean_calibrated_endpoint_delta"] - summary["mean_raw_endpoint_delta"]
    )
    summary["calibration_reversed_endpoint_sign"] = (
        np.sign(summary["mean_calibrated_endpoint_delta"])
        != np.sign(summary["mean_raw_endpoint_delta"])
    )
    correlations: list[dict[str, Any]] = []
    for (family, dose, split_seed), group in frame.groupby(
        ["family", "dose", "split_seed"], sort=True
    ):
        correlations.append(
            {
                "family": family,
                "dose": dose,
                "split_seed": split_seed,
                "rank_correlation": float(
                    spearmanr(group["raw_pier"], group["calibrated_pier"]).statistic
                ),
            }
        )
    correlation_summary = pd.DataFrame(correlations).groupby("family", as_index=False).agg(
        mean_raw_calibrated_rank_correlation=("rank_correlation", "mean"),
        minimum_raw_calibrated_rank_correlation=("rank_correlation", "min"),
    )
    return summary.merge(correlation_summary, on="family", validate="many_to_one")


def interface_table() -> pd.DataFrame:
    survival = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/interface_effect_survival.parquet"
    )
    validation = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/label_bias_heldout_validation.parquet"
    )
    validation_summary = validation.groupby("model_id", as_index=False).agg(
        before_mean_heldout_permutation_tv=("before_mean_tv", "mean"),
        after_mean_heldout_permutation_tv=("after_mean_tv", "mean"),
        mean_heldout_tv_reduction=("mean_tv_reduction", "mean"),
        split_improvement_rate=("heldout_invariance_improved", "mean"),
        residual_95th_percentile_tv=("after_95th_percentile_tv", "mean"),
    ).rename(columns={"model_id": "target"})
    return survival.merge(validation_summary, on="target", validate="many_to_one")


def generation_table() -> pd.DataFrame:
    interface = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/generation_alignment_interface_check.parquet"
    )
    summary = interface.groupby(["model_id", "condition"], as_index=False).agg(
        raw_score_generation_agreement=("raw_score_generation_agreement", "mean"),
        raw_agreement_conditional_valid_label=(
            "raw_agreement_conditional_valid_label",
            "mean",
        ),
        bias_corrected_score_generation_agreement=(
            "bias_corrected_score_generation_agreement",
            "mean",
        ),
        bias_corrected_agreement_conditional_valid_label=(
            "bias_corrected_agreement_conditional_valid_label",
            "mean",
        ),
        malformed_or_no_label_rate=("malformed_or_no_label_rate", "mean"),
        enters_high_alignment_subset=("enters_high_alignment_subset", "first"),
    ).rename(columns={"model_id": "target"})
    summary["row_type"] = "alignment_summary"
    subsets = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/generation_alignment_subsets.parquet"
    )
    subset_summary = subsets.groupby(
        ["target", "family", "interface_condition", "ecosystem_condition"],
        as_index=False,
    ).agg(
        mean_endpoint_delta=("endpoint_delta", "mean"),
        mean_convexity_improvement=("convex_vs_single_absolute_improvement", "mean"),
        effect_sign_agreement_with_full=("effect_sign_agreement_with_full", "mean"),
        aggregate_rank_correlation_with_full=(
            "aggregate_rank_correlation_with_full",
            "first",
        ),
    )
    overall = summary[summary["condition"].eq("overall")].drop(columns="row_type")
    subset_summary = subset_summary.merge(
        overall, on="target", validate="many_to_one"
    )
    subset_summary["row_type"] = "ecosystem_sensitivity"
    subset_summary["condition"] = "not_applicable"
    return (
        pd.concat([summary, subset_summary], ignore_index=True, sort=False)
        .sort_values(["target", "row_type", "condition"])
        .reset_index(drop=True)
    )


def target_summary(endpoint: pd.DataFrame, convex: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/primary_dose_results.parquet")
    per_condition = (
        primary[primary["dose"].notna()]
        .groupby(["target", "family", "dose"], as_index=False)["convex_error"]
        .mean()
    )
    maximum_rows = select_max_after_split_aggregation(
        primary[primary["dose"].notna()],
        group_columns=["target", "family"],
        condition_column="dose",
        value_column="convex_error",
    ).rename(
        columns={"dose": "maximum_aggregate_pier_dose", "convex_error": "maximum_aggregate_pier"}
    )
    clean = per_condition[per_condition["dose"].eq(0.0)].rename(
        columns={"convex_error": "clean_pier_under_family_fitted_weights"}
    )
    overall = (
        primary[primary["dose"].isna()]
        .groupby(["target", "family"], as_index=False)["convex_error"]
        .mean()
        .rename(columns={"convex_error": "mean_overall_pier"})
    )
    columns = [
        "target",
        "family",
        "mean_delta_absolute",
        "bootstrap_lower_95",
        "bootstrap_upper_95",
        "same_sign_split_count",
        "evidence_status",
    ]
    convex_columns = [
        "target",
        "family",
        "mean_absolute_improvement",
        "mean_relative_improvement",
        "convex_wins_split_count",
        "bootstrap_lower_95",
        "bootstrap_upper_95",
        "evidence_status",
    ]
    endpoint_subset = endpoint[columns].rename(
        columns={
            "bootstrap_lower_95": "endpoint_bootstrap_lower_95",
            "bootstrap_upper_95": "endpoint_bootstrap_upper_95",
            "evidence_status": "endpoint_evidence_status",
        }
    )
    convex_subset = convex[convex_columns].rename(
        columns={
            "bootstrap_lower_95": "convexity_bootstrap_lower_95",
            "bootstrap_upper_95": "convexity_bootstrap_upper_95",
            "evidence_status": "convexity_evidence_status",
        }
    )
    table = endpoint_subset.merge(convex_subset, on=["target", "family"], validate="one_to_one")
    table = table.merge(
        clean[["target", "family", "clean_pier_under_family_fitted_weights"]],
        on=["target", "family"],
        validate="one_to_one",
    )
    table = table.merge(overall, on=["target", "family"], validate="one_to_one")
    table = table.merge(
        maximum_rows[
            ["target", "family", "maximum_aggregate_pier_dose", "maximum_aggregate_pier"]
        ],
        on=["target", "family"],
        validate="one_to_one",
    )
    aggregate = table.groupby("target")["mean_overall_pier"].transform("mean")
    table["aggregate_real_model_pier"] = aggregate
    return table


def _pick(
    frame: pd.DataFrame,
    value_column: str,
    *,
    largest: bool,
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.Series:
    candidates = frame[predicate(frame)] if predicate is not None else frame
    if candidates.empty:
        candidates = frame
    index = candidates[value_column].idxmax() if largest else candidates[value_column].idxmin()
    return candidates.loc[index]


def gold_candidates(
    endpoint: pd.DataFrame,
    convex: pd.DataFrame,
    sibling: pd.DataFrame,
    calibration: pd.DataFrame,
    interface: pd.DataFrame,
    generation: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        definition: str,
        row: pd.Series,
        value_column: str,
        *,
        lower: str | None = None,
        upper: str | None = None,
        status: str | None = None,
        sign_column: str | None = None,
        interface_column: str | None = None,
    ) -> None:
        family_value = row.get("family", "all")
        if pd.isna(family_value):
            family_value = "all"
        rows.append(
            {
                "target": row.get("target"),
                "family": family_value,
                "effect_definition": definition,
                "mean_effect": float(row[value_column]),
                "bootstrap_lower_95": float(row[lower]) if lower and pd.notna(row.get(lower)) else math.nan,
                "bootstrap_upper_95": float(row[upper]) if upper and pd.notna(row.get(upper)) else math.nan,
                "split_sign_count": int(row[sign_column]) if sign_column and pd.notna(row.get(sign_column)) else 0,
                "interface_survival_count": int(row[interface_column]) if interface_column and pd.notna(row.get(interface_column)) else 0,
                "evidence_status": status or str(row.get("evidence_status", "exploratory")),
            }
        )

    irrelevant = endpoint[endpoint["family"].eq("irrelevant_context")]
    deletion = endpoint[endpoint["family"].eq("content_deletion")]
    add(
        "largest stable positive irrelevant-context endpoint delta",
        _pick(
            irrelevant,
            "mean_delta_absolute",
            largest=True,
            predicate=lambda frame: frame["evidence_status"].eq("stable")
            & frame["mean_delta_absolute"].gt(0),
        ),
        "mean_delta_absolute",
        lower="bootstrap_lower_95",
        upper="bootstrap_upper_95",
        sign_column="same_sign_split_count",
    )
    add(
        "largest stable negative irrelevant-context endpoint delta",
        _pick(
            irrelevant,
            "mean_delta_absolute",
            largest=False,
            predicate=lambda frame: frame["evidence_status"].eq("stable")
            & frame["mean_delta_absolute"].lt(0),
        ),
        "mean_delta_absolute",
        lower="bootstrap_lower_95",
        upper="bootstrap_upper_95",
        sign_column="same_sign_split_count",
    )
    add(
        "largest stable negative deletion endpoint delta",
        _pick(
            deletion,
            "mean_delta_absolute",
            largest=False,
            predicate=lambda frame: frame["evidence_status"].eq("stable")
            & frame["mean_delta_absolute"].lt(0),
        ),
        "mean_delta_absolute",
        lower="bootstrap_lower_95",
        upper="bootstrap_upper_95",
        sign_column="same_sign_split_count",
    )
    best_convex = _pick(
        convex,
        "mean_absolute_improvement",
        largest=True,
        predicate=lambda frame: frame["evidence_status"].eq("stable"),
    )
    add(
        "largest stable convex-over-single absolute improvement",
        best_convex,
        "mean_absolute_improvement",
        lower="bootstrap_lower_95",
        upper="bootstrap_upper_95",
        sign_column="same_sign_split_count",
    )
    add(
        "largest stable convex-over-single relative improvement",
        _pick(
            convex,
            "mean_relative_improvement",
            largest=True,
            predicate=lambda frame: frame["evidence_status"].eq("stable"),
        ),
        "mean_relative_improvement",
        sign_column="same_sign_split_count",
    )
    add(
        "largest sibling-removal inflation beyond non-sibling null median",
        _pick(sibling, "mean_excess_over_null_median", largest=True),
        "mean_excess_over_null_median",
        lower="bootstrap_lower_95",
        upper="bootstrap_upper_95",
        sign_column="same_sign_split_count",
    )
    unique_summary = summary.drop_duplicates("target")
    low = _pick(unique_summary, "aggregate_real_model_pier", largest=False)
    high = _pick(unique_summary, "aggregate_real_model_pier", largest=True)
    add("lowest aggregate real-model PIER", low, "aggregate_real_model_pier")
    add("highest aggregate real-model PIER", high, "aggregate_real_model_pier")
    add(
        "largest calibration-induced reversal",
        calibration.loc[calibration["calibration_induced_endpoint_change"].abs().idxmax()],
        "calibration_induced_endpoint_change",
    )
    residual = interface.drop_duplicates(["target"])
    add(
        "largest residual option-permutation artifact after correction",
        _pick(residual, "after_mean_heldout_permutation_tv", largest=True),
        "after_mean_heldout_permutation_tv",
    )
    generation_unique = generation[
        generation["row_type"].eq("alignment_summary")
        & generation["condition"].eq("overall")
    ].copy()
    generation_unique["score_generation_mismatch"] = 1.0 - generation_unique[
        "raw_score_generation_agreement"
    ]
    add(
        "largest score-generation mismatch",
        _pick(generation_unique, "score_generation_mismatch", largest=True),
        "score_generation_mismatch",
    )
    survival = interface[interface["survives_all_four_interfaces"]].copy()
    if survival.empty:
        survival = interface[interface["interface_condition"].eq("raw")].copy()
    survival["absolute_effect"] = survival["raw_mean_endpoint_delta"].abs()
    strongest = _pick(survival, "absolute_effect", largest=True)
    add(
        "strongest effect surviving all four response-interface conditions",
        strongest,
        "raw_mean_endpoint_delta",
        status="stable" if int(strongest.get("interface_survival_count", 0)) == 4 else "exploratory",
        interface_column="interface_survival_count",
    )
    return pd.DataFrame(rows).sort_values("effect_definition").reset_index(drop=True)


def write_all_tables() -> dict[str, Any]:
    tables_dir = PHYSICAL_ROOT / "outputs/tables"
    endpoint = endpoint_table()
    track = track_table()
    sibling = sibling_table()
    convex = convexity_table()
    calibration = calibration_table()
    interface = interface_table()
    generation = generation_table()
    summary = target_summary(endpoint, convex)
    candidates = gold_candidates(
        endpoint,
        convex,
        sibling,
        calibration,
        interface,
        generation,
        summary,
    )
    mapping = {
        "table_v21_endpoint_effects.csv": endpoint,
        "table_v21_track_consistency.csv": track,
        "table_v21_sibling_removal.csv": sibling,
        "table_v21_convexity_gap.csv": convex,
        "table_v21_calibration_sensitivity.csv": calibration,
        "table_v21_interface_sensitivity.csv": interface,
        "table_v21_generation_alignment.csv": generation,
        "table_v21_corrected_target_summary.csv": summary,
        "table_v21_corrected_gold_candidates.csv": candidates,
    }
    outputs: list[Path] = []
    for name, frame in mapping.items():
        path = tables_dir / name
        atomic_csv(path, frame)
        outputs.append(path)
    candidate_path = PHYSICAL_ROOT / "outputs/analysis/corrected_gold_candidates.parquet"
    atomic_parquet(candidate_path, candidates)
    outputs.append(candidate_path)
    return {"outputs": outputs, "table_rows": {name: len(frame) for name, frame in mapping.items()}}
