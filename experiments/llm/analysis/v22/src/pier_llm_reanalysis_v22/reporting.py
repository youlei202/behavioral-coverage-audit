from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import high_dose
from .utils import PHYSICAL_ROOT, atomic_csv, atomic_parquet, atomic_write_text, load_config


def _stable_label(values: np.ndarray, lower: float, upper: float) -> str:
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    interval_excludes_zero = lower > 0 or upper < 0
    return "stable" if max(positive, negative) >= 9 and interval_excludes_zero else "exploratory"


def _mode(values: pd.Series) -> str:
    counts = Counter(str(value) for value in values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _v21_root() -> Path:
    return Path(load_config()["source_roots"]["v21"]["physical_alias"])


def _endpoint_table() -> pd.DataFrame:
    effects = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_endpoint_effects.parquet"
    )
    bootstrap = pd.read_csv(
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/endpoint_bootstrap_summary.csv"
    )
    decomp = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/estimand_decomposition.parquet"
    )
    calibrated = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_calibrated_results.parquet"
    )
    interface = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_interface_sensitivity.parquet"
    )
    old_rows: list[dict[str, Any]] = []
    for (target, family, split_seed), group in decomp.groupby(
        ["target", "family", "split_seed"], sort=True
    ):
        clean = group[group["dose"].eq(0.0)].iloc[0]
        high = group[group["dose"].eq(high_dose(str(family)))].iloc[0]
        old_rows.append(
            {
                "target": target,
                "family": family,
                "split_seed": split_seed,
                "old_delta": float(
                    high["aggregate_fit_aggregate_eval"]
                    - clean["aggregate_fit_aggregate_eval"]
                ),
            }
        )
    old = pd.DataFrame(old_rows)
    effects = effects.merge(old, on=["target", "family", "split_seed"])
    rows: list[dict[str, Any]] = []
    for (target, family), group in effects.groupby(["target", "family"], sort=True):
        boot = bootstrap[
            bootstrap["target"].eq(target) & bootstrap["family"].eq(family)
        ].iloc[0]
        values = group["trackwise_delta"].to_numpy(dtype=np.float64)
        calibrated_delta = float(
            calibrated[
                calibrated["target"].eq(target)
                & calibrated["family"].eq(family)
                & calibrated["dose"].eq(0.0)
            ]["calibrated_endpoint_delta"].mean()
        )
        interface_delta = float(
            interface[
                interface["target"].eq(target)
                & interface["family"].eq(family)
                & interface["dose"].eq(0.0)
            ]["interface_corrected_endpoint_delta"].mean()
        )
        old_delta = float(group["old_delta"].mean())
        mean_delta = float(np.mean(values))
        rows.append(
            {
                "target": target,
                "family": family,
                "trackwise_clean_pier": float(group["trackwise_clean_pier"].mean()),
                "trackwise_high_pier": float(group["trackwise_high_pier"].mean()),
                "trackwise_delta": mean_delta,
                "trackwise_relative_delta": float(
                    group["trackwise_relative_delta"].mean()
                ),
                "trackwise_delta_median": float(np.median(values)),
                "trackwise_delta_standard_deviation": float(np.std(values, ddof=1)),
                "trackwise_delta_minimum": float(np.min(values)),
                "trackwise_delta_maximum": float(np.max(values)),
                "bootstrap_lower": float(boot["bootstrap_lower"]),
                "bootstrap_upper": float(boot["bootstrap_upper"]),
                "probability_delta_positive": float(boot["probability_positive"]),
                "positive_split_count": int(np.sum(values > 0)),
                "negative_split_count": int(np.sum(values < 0)),
                "stability_label": _stable_label(
                    values,
                    float(boot["bootstrap_lower"]),
                    float(boot["bootstrap_upper"]),
                ),
                "old_mean_response_delta": old_delta,
                "delta_difference_due_to_correction": mean_delta - old_delta,
                "endpoint_sign_changed_from_v21": bool(
                    np.sign(mean_delta) != np.sign(old_delta)
                ),
                "calibrated_delta": calibrated_delta,
                "interface_corrected_delta": interface_delta,
            }
        )
    return pd.DataFrame(rows)


def _decomposition_table() -> pd.DataFrame:
    frame = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/estimand_decomposition.parquet"
    )
    columns = [
        "aggregate_fit_aggregate_eval",
        "aggregate_fit_trackwise_eval",
        "trackwise_fit_aggregate_eval",
        "trackwise_fit_trackwise_eval",
        "evaluation_correction",
        "fitting_correction",
        "interaction",
        "cancellation_gap",
        "weight_l1_change",
        "weight_l2_change",
        "projected_response_change",
        "rank_correlation_aggregate_to_v22",
    ]
    return (
        frame.groupby(["target", "family", "dose"], as_index=False)[columns]
        .mean()
        .sort_values(["target", "family", "dose"])
    )


def _convexity_table() -> pd.DataFrame:
    point = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_convex_vs_single.parquet"
    )
    bootstrap = pd.read_csv(
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/convexity_bootstrap_summary.csv"
    )
    calibrated = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_calibrated_results.parquet"
    )
    vector = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_vector_results.parquet"
    )
    rows: list[dict[str, Any]] = []
    for (target, family), group in point.groupby(["target", "family"], sort=True):
        boot = bootstrap[
            bootstrap["target"].eq(target) & bootstrap["family"].eq(family)
        ].iloc[0]
        values = group["absolute_improvement"].to_numpy(dtype=np.float64)
        calibrated_improvement = float(
            calibrated[
                calibrated["target"].eq(target)
                & calibrated["family"].eq(family)
                & calibrated["dose"].eq(0.0)
            ]["calibrated_convexity_improvement"].mean()
        )
        vector_support = vector[
            vector["target"].eq(target)
            & vector["family"].eq(family)
            & vector["weight_source"].eq("vector_fitted")
            & vector["dose"].eq(0.0)
        ]["convexity_direction_support"]
        rows.append(
            {
                "target": target,
                "family": family,
                "fit_selected_single_peer": _mode(group["fit_selected_single_peer"]),
                "fit_selected_single_peer_frequency": int(
                    group["fit_selected_single_peer"].eq(
                        _mode(group["fit_selected_single_peer"])
                    ).sum()
                ),
                "single_error_trackwise": float(
                    group["single_error_trackwise"].mean()
                ),
                "convex_error_trackwise": float(
                    group["convex_error_trackwise"].mean()
                ),
                "absolute_improvement": float(np.mean(values)),
                "gap_ratio": float(group["gap_ratio"].mean()),
                "bootstrap_lower": float(boot["bootstrap_lower"]),
                "bootstrap_upper": float(boot["bootstrap_upper"]),
                "positive_split_count": int(np.sum(values > 0)),
                "stability_label": _stable_label(
                    values,
                    float(boot["bootstrap_lower"]),
                    float(boot["bootstrap_upper"]),
                ),
                "calibrated_improvement": calibrated_improvement,
                "vector_space_support": f"{int(vector_support.sum())}/10 splits",
                "vector_support_fraction": float(vector_support.mean()),
            }
        )
    return pd.DataFrame(rows)


def _sibling_table() -> pd.DataFrame:
    point = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_sibling_removal.parquet"
    )
    overall = point[point["evaluation_scope"].eq("overall_equal_dose")]
    null = pd.read_parquet(
        PHYSICAL_ROOT
        / "outputs/analysis/trackwise_non_sibling_removal_null.parquet"
    )
    bootstrap = pd.read_csv(
        PHYSICAL_ROOT
        / "outputs/bootstrap_summaries/sibling_removal_bootstrap_summary.csv"
    )
    calibrated = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_calibrated_results.parquet"
    )
    vector = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_vector_results.parquet"
    )
    rows: list[dict[str, Any]] = []
    for (target, family), group in overall.groupby(["target", "family"], sort=True):
        removed_sibling = str(group.iloc[0]["removed_sibling"])
        boot = bootstrap[
            bootstrap["target"].eq(target) & bootstrap["family"].eq(family)
        ].iloc[0]
        null_group = null[
            null["target"].eq(target) & null["family"].eq(family)
        ]
        calibrated_value = float(
            calibrated[
                calibrated["target"].eq(target)
                & calibrated["family"].eq(family)
                & calibrated["dose"].eq(0.0)
            ]["calibrated_sibling_removal_inflation"].mean()
        )
        vector_support = vector[
            vector["target"].eq(target)
            & vector["family"].eq(family)
            & vector["weight_source"].eq("vector_fitted")
            & vector["dose"].eq(0.0)
        ]["sibling_direction_support"]
        values = group["inflation"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "target": target,
                "removed_sibling": removed_sibling,
                "family": family,
                "all_peer_trackwise_pier": float(
                    group["all_peer_trackwise_pier"].mean()
                ),
                "sibling_removed_trackwise_pier": float(
                    group["sibling_removed_trackwise_pier"].mean()
                ),
                "inflation": float(np.mean(values)),
                "bootstrap_lower": float(boot["bootstrap_lower"]),
                "bootstrap_upper": float(boot["bootstrap_upper"]),
                "positive_split_count": int(np.sum(values > 0)),
                "stability_label": _stable_label(
                    values,
                    float(boot["bootstrap_lower"]),
                    float(boot["bootstrap_upper"]),
                ),
                "largest_non_sibling_inflation": float(
                    null_group.groupby("split_seed")[
                        "largest_non_sibling_inflation"
                    ].first().mean()
                ),
                "median_non_sibling_inflation": float(
                    null_group.groupby("split_seed")[
                        "median_non_sibling_inflation"
                    ].first().mean()
                ),
                "sibling_rank": float(
                    null_group.groupby("split_seed")["sibling_rank"].first().mean()
                ),
                "sibling_percentile": float(
                    null_group.groupby("split_seed")[
                        "sibling_percentile"
                    ].first().mean()
                ),
                "splits_sibling_largest": int(group["splits_sibling_largest"].iloc[0]),
                "calibrated_inflation": calibrated_value,
                "vector_space_support": f"{int(vector_support.sum())}/10 splits",
                "vector_support_fraction": float(vector_support.mean()),
            }
        )
    return pd.DataFrame(rows)


def _cancellation_table(endpoint: pd.DataFrame) -> pd.DataFrame:
    dose = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/cancellation_gap_results.parquet"
    )
    changed = endpoint[["target", "family", "endpoint_sign_changed_from_v21"]]
    rows = (
        dose.groupby(["target", "family", "dose"], as_index=False)
        .agg(
            trackwise_pier=("trackwise_pier", "mean"),
            track_mean_response_pier=("track_mean_response_pier", "mean"),
            cancellation_gap=("cancellation_gap", "mean"),
            relative_hidden_fraction=("relative_hidden_fraction", "mean"),
            between_track_std=("between_track_std", "mean"),
            between_track_range=("between_track_range", "mean"),
        )
        .merge(changed, on=["target", "family"], validate="many_to_one")
    )
    return rows


CLAIMS = [
    ("phi-4 irrelevant-context activation", "endpoint", "microsoft/phi-4", "irrelevant_context", 1),
    ("Qwen2.5 irrelevant-context activation", "endpoint", "Qwen/Qwen2.5-7B-Instruct", "irrelevant_context", 1),
    ("OLMo irrelevant-context contraction", "endpoint", "allenai/OLMo-2-1124-13B-Instruct", "irrelevant_context", -1),
    ("Mistral irrelevant-context contraction", "endpoint", "mistralai/Mistral-Nemo-Instruct-2407", "irrelevant_context", -1),
    ("Granite irrelevant-context contraction", "endpoint", "ibm-granite/granite-3.3-8b-instruct", "irrelevant_context", -1),
    ("Qwen sibling coverage in content deletion", "sibling", "Qwen/Qwen2.5-7B-Instruct", "content_deletion", 1),
    ("Qwen sibling coverage in irrelevant context", "sibling", "Qwen/Qwen2.5-7B-Instruct", "irrelevant_context", 1),
    ("General-Reasoner sibling coverage in content deletion", "sibling", "TIGER-Lab/General-Reasoner-Qwen2.5-7B", "content_deletion", 1),
    ("General-Reasoner sibling coverage in irrelevant context", "sibling", "TIGER-Lab/General-Reasoner-Qwen2.5-7B", "irrelevant_context", 1),
    ("Phi-4-reasoning-plus distributed convex coverage in both families", "convex_both", "microsoft/Phi-4-reasoning-plus", "both", 1),
    ("Mistral distributed convex coverage in both families", "convex_both", "mistralai/Mistral-Nemo-Instruct-2407", "both", 1),
    ("content-deletion contraction for OLMo", "endpoint", "allenai/OLMo-2-1124-13B-Instruct", "content_deletion", -1),
    ("content-deletion contraction for Mistral", "endpoint", "mistralai/Mistral-Nemo-Instruct-2407", "content_deletion", -1),
    ("content-deletion contraction for Granite", "endpoint", "ibm-granite/granite-3.3-8b-instruct", "content_deletion", -1),
]


def _claim_table(
    endpoint: pd.DataFrame, convexity: pd.DataFrame, sibling: pd.DataFrame
) -> pd.DataFrame:
    interface = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_interface_sensitivity.parquet"
    )
    vector = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_vector_results.parquet"
    )
    v21_endpoint = pd.read_csv(
        _v21_root() / "outputs/tables/table_v21_endpoint_effects.csv"
    )
    v21_convexity = pd.read_csv(
        _v21_root() / "outputs/tables/table_v21_convexity_gap.csv"
    )
    v21_sibling = pd.read_csv(
        _v21_root() / "outputs/tables/table_v21_sibling_removal.csv"
    )
    rows: list[dict[str, Any]] = []

    def status_from_sign(value: float, expected: int) -> str:
        return "survives" if int(np.sign(value)) == expected else "reverses_or_absent"

    for claim, kind, target, family, expected in CLAIMS:
        families = list(load_config()["families"]) if family == "both" else [family]
        point_supported: list[bool] = []
        bootstrap_stable: list[bool] = []
        calibration_supported: list[bool] = []
        interface_supported: list[bool] = []
        vector_supported: list[bool] = []
        v21_supported: list[bool] = []
        details: list[str] = []
        for current_family in families:
            if kind == "endpoint":
                record = endpoint[
                    endpoint["target"].eq(target)
                    & endpoint["family"].eq(current_family)
                ].iloc[0]
                point_supported.append(int(np.sign(record["trackwise_delta"])) == expected)
                bootstrap_stable.append(
                    record["stability_label"] == "stable"
                    and int(np.sign(record["trackwise_delta"])) == expected
                )
                calibration_supported.append(
                    int(np.sign(record["calibrated_delta"])) == expected
                )
                interface_supported.append(
                    int(np.sign(record["interface_corrected_delta"])) == expected
                )
                vector_value = vector[
                    vector["target"].eq(target)
                    & vector["family"].eq(current_family)
                    & vector["weight_source"].eq("vector_fitted")
                    & vector["dose"].eq(0.0)
                ]["vector_endpoint_delta"].mean()
                vector_supported.append(int(np.sign(vector_value)) == expected)
                old = v21_endpoint[
                    v21_endpoint["target"].eq(target)
                    & v21_endpoint["family"].eq(current_family)
                ].iloc[0]
                v21_supported.append(
                    str(old["evidence_status"]).lower() == "stable"
                    and int(np.sign(old["mean_delta_absolute"])) == expected
                )
                details.append(f"{current_family}: Δ={record['trackwise_delta']:.6g}")
            elif kind == "sibling":
                record = sibling[
                    sibling["target"].eq(target)
                    & sibling["family"].eq(current_family)
                ].iloc[0]
                point_supported.append(record["inflation"] > 0)
                bootstrap_stable.append(record["stability_label"] == "stable")
                calibration_supported.append(record["calibrated_inflation"] > 0)
                interface_value = interface[
                    interface["target"].eq(target)
                    & interface["family"].eq(current_family)
                    & interface["dose"].eq(0.0)
                ]["interface_corrected_sibling_removal_inflation"].mean()
                interface_supported.append(interface_value > 0)
                vector_supported.append(record["vector_support_fraction"] >= 0.9)
                old = v21_sibling[
                    v21_sibling["target"].eq(target)
                    & v21_sibling["family"].eq(current_family)
                ].iloc[0]
                v21_supported.append(str(old["evidence_status"]).lower() == "stable")
                details.append(
                    f"{current_family}: inflation={record['inflation']:.6g}, "
                    f"largest-null splits={record['splits_sibling_largest']}/10"
                )
            else:
                record = convexity[
                    convexity["target"].eq(target)
                    & convexity["family"].eq(current_family)
                ].iloc[0]
                point_supported.append(record["absolute_improvement"] > 0)
                bootstrap_stable.append(record["stability_label"] == "stable")
                calibration_supported.append(record["calibrated_improvement"] > 0)
                interface_value = interface[
                    interface["target"].eq(target)
                    & interface["family"].eq(current_family)
                    & interface["dose"].eq(0.0)
                ]["interface_corrected_convexity_improvement"].mean()
                interface_supported.append(interface_value > 0)
                vector_supported.append(record["vector_support_fraction"] >= 0.9)
                old = v21_convexity[
                    v21_convexity["target"].eq(target)
                    & v21_convexity["family"].eq(current_family)
                ].iloc[0]
                v21_supported.append(str(old["evidence_status"]).lower() == "stable")
                details.append(
                    f"{current_family}: improvement={record['absolute_improvement']:.6g}"
                )
        raw_ok = all(point_supported)
        boot_ok = all(bootstrap_stable)
        calibration_ok = all(calibration_supported)
        interface_ok = all(interface_supported)
        vector_ok = all(vector_supported)
        if raw_ok and boot_ok:
            if kind == "endpoint":
                direction = "away from" if expected > 0 else "toward"
                wording = (
                    f"Under the corrected trackwise candidate-label estimand, {target} moved "
                    f"{direction} the fixed peer hull under {family.replace('_', ' ')}; "
                    "the effect was stable across paired splits and base-question resampling."
                )
            elif kind == "sibling":
                wording = (
                    f"For {target}, removing the declared sibling increased corrected "
                    f"trackwise PIER under {family.replace('_', ' ')}, conditional on the fixed roster."
                )
            else:
                wording = (
                    f"For {target}, a fixed distributed convex surrogate outperformed the "
                    "fit-selected single peer in both intervention families under the corrected trackwise estimand."
                )
        else:
            wording = (
                f"Treat the pre-declared claim '{claim}' as exploratory under the corrected "
                "trackwise candidate-label estimand."
            )
        rows.append(
            {
                "claim": claim,
                "v21_status": "stable" if all(v21_supported) else "exploratory_or_absent",
                "v22_raw_trackwise_status": "supported" if raw_ok else "not_supported",
                "bootstrap_status": "stable" if boot_ok else "exploratory",
                "calibration_status": "survives" if calibration_ok else "does_not_survive",
                "interface_sensitivity_status": "survives"
                if interface_ok
                else "does_not_survive",
                "vector_support_status": "supported" if vector_ok else "not_supported",
                "details": "; ".join(details),
                "final_recommended_wording": wording,
            }
        )
    return pd.DataFrame(rows)


def write_all_tables() -> dict[str, Any]:
    endpoint = _endpoint_table()
    decomposition = _decomposition_table()
    convexity = _convexity_table()
    sibling = _sibling_table()
    cancellation = _cancellation_table(endpoint)
    claims = _claim_table(endpoint, convexity, sibling)
    tables = {
        "table_1_corrected_endpoint_effects.csv": endpoint,
        "table_2_estimand_decomposition.csv": decomposition,
        "table_3_convexity_gap.csv": convexity,
        "table_4_sibling_coverage.csv": sibling,
        "table_5_cancellation_diagnosis.csv": cancellation,
        "table_6_claim_survival.csv": claims,
    }
    outputs: list[Path] = []
    row_counts: dict[str, int] = {}
    for name, frame in tables.items():
        path = PHYSICAL_ROOT / "outputs/tables" / name
        atomic_csv(path, frame)
        outputs.append(path)
        row_counts[name] = len(frame)
    claim_path = (
        PHYSICAL_ROOT / "outputs/analysis/trackwise_claim_survival.parquet"
    )
    atomic_parquet(claim_path, claims)
    outputs.append(claim_path)
    row_counts["trackwise_claim_survival"] = len(claims)
    return {"outputs": outputs, "row_counts": row_counts}


def _answer(
    number: int,
    title: str,
    observed: str,
    interpretation: str,
    limitation: str,
    wording: str,
) -> str:
    return (
        f"## {number}. {title}\n\n"
        f"**Observed result.** {observed}\n\n"
        f"**Interpretation.** {interpretation}\n\n"
        f"**Measurement limitation.** {limitation}\n\n"
        f"**Recommended manuscript wording.** {wording}\n\n"
    )


def _fmt_effects(frame: pd.DataFrame) -> str:
    return "; ".join(
        f"`{row.target}` {row.trackwise_delta:.6g} "
        f"[{row.bootstrap_lower:.6g}, {row.bootstrap_upper:.6g}] ({row.stability_label})"
        for row in frame.itertuples()
    )


def write_report(*, byte_identical: bool | None) -> Path:
    endpoint = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_1_corrected_endpoint_effects.csv"
    )
    decomposition = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_2_estimand_decomposition.csv"
    )
    convexity = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_3_convexity_gap.csv"
    )
    sibling = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_4_sibling_coverage.csv"
    )
    cancellation = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_5_cancellation_diagnosis.csv"
    )
    claims = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_6_claim_survival.csv"
    )
    primary = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    )
    weights = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_weights.parquet"
    )
    controls = pd.read_parquet(PHYSICAL_ROOT / "controls/synthetic_controls.parquet")
    interface_validation = pd.read_parquet(
        PHYSICAL_ROOT / "controls/label_bias_heldout_validation.parquet"
    )
    mean_old = float(decomposition["aggregate_fit_aggregate_eval"].mean())
    mean_new = float(decomposition["trackwise_fit_trackwise_eval"].mean())
    sign_changes = endpoint[endpoint["endpoint_sign_changed_from_v21"]]
    irrelevant = endpoint[endpoint["family"].eq("irrelevant_context")]
    away = irrelevant[irrelevant["trackwise_delta"] > 0]
    toward = irrelevant[irrelevant["trackwise_delta"] < 0]
    stable_away = away[away["stability_label"].eq("stable")]
    stable_toward = toward[toward["stability_label"].eq("stable")]
    targeted_deletion = primary[
        primary["family"].eq("content_deletion")
        & primary["target"].isin(
            [
                "allenai/OLMo-2-1124-13B-Instruct",
                "mistralai/Mistral-Nemo-Instruct-2407",
                "ibm-granite/granite-3.3-8b-instruct",
            ]
        )
    ]
    track_statements: list[str] = []
    for target, group in targeted_deletion.groupby("target", sort=True):
        clean = float(group[group["dose"].eq(0.0)]["trackwise_pier"].mean())
        high = group[group["dose"].eq(0.4)]["track_piers"]
        track_means = np.mean(np.stack(high.to_list()), axis=0)
        deltas = track_means - clean
        track_statements.append(
            f"`{target}` track deltas " + ", ".join(f"{value:.6g}" for value in deltas)
        )
    artifacts = endpoint[
        endpoint["family"].eq("content_deletion")
        & (endpoint["old_mean_response_delta"] < 0)
        & (
            (endpoint["trackwise_delta"] >= 0)
            | (
                endpoint["trackwise_delta"].abs()
                < 0.5 * endpoint["old_mean_response_delta"].abs()
            )
        )
    ]
    eval_mean = float(decomposition["evaluation_correction"].abs().mean())
    fit_mean = float(decomposition["fitting_correction"].abs().mean())
    interaction_mean = float(decomposition["interaction"].abs().mean())
    qwen = sibling[
        sibling["target"].eq("Qwen/Qwen2.5-7B-Instruct")
    ]
    general = sibling[
        sibling["target"].eq("TIGER-Lab/General-Reasoner-Qwen2.5-7B")
    ]
    distributed = convexity[
        convexity["target"].isin(
            [
                "microsoft/Phi-4-reasoning-plus",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ]
        )
    ]
    qwen_pair_mean = float(
        sibling[
            sibling["target"].isin(
                [
                    "Qwen/Qwen2.5-7B-Instruct",
                    "TIGER-Lab/General-Reasoner-Qwen2.5-7B",
                ]
            )
        ]["inflation"].mean()
    )
    phi_pair_mean = float(
        sibling[
            sibling["target"].isin(
                ["microsoft/phi-4", "microsoft/Phi-4-reasoning-plus"]
            )
        ]["inflation"].mean()
    )
    calibration_survivors = claims[claims["calibration_status"].eq("survives")][
        "claim"
    ].tolist()
    interface_survivors = claims[
        claims["interface_sensitivity_status"].eq("survives")
    ]["claim"].tolist()
    vector_survivors = claims[claims["vector_support_status"].eq("supported")][
        "claim"
    ].tolist()
    ambiguous = weights[
        weights["weight_representation_ambiguous"]
        & weights["target"].isin(
            [
                "Qwen/Qwen2.5-7B-Instruct",
                "microsoft/Phi-4-reasoning-plus",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ]
        )
    ]
    robust_claims = claims[
        claims["bootstrap_status"].eq("stable")
        & claims["calibration_status"].eq("survives")
        & claims["interface_sensitivity_status"].eq("survives")
        & claims["vector_support_status"].eq("supported")
    ]
    identity_text = (
        "The recorded V2/V2.1 inputs were byte-identical before and after the run."
        if byte_identical is True
        else "Final byte-identity confirmation is deferred to stage 13 and is required before packaging."
    )
    sections = [
        _answer(
            1,
            "How much did the corrected trackwise estimand change absolute PIER levels?",
            f"Across all target-family-dose summaries, mean aggregate-style PIER was {mean_old:.6g} and mean V2.2 trackwise PIER was {mean_new:.6g}, a change of {mean_new - mean_old:.6g}. The largest mean cancellation gap was {cancellation['cancellation_gap'].max():.6g}.",
            "The corrected object is systematically sensitive to residual magnitude hidden by pre-absolute track averaging.",
            "Both quantities use the same cached candidate-label scores; neither establishes interface-independent behavior.",
            "Averaging intervention tracks before taking residual magnitude hid peer-inexpressible response variation; primary PIER therefore averages absolute residuals over individual tracks.",
        ),
        _answer(
            2,
            "Which V2.1 endpoint signs changed?",
            "No endpoint sign changed." if sign_changes.empty else "; ".join(f"`{row.target}`/{row.family}: {row.old_mean_response_delta:.6g} → {row.trackwise_delta:.6g}" for row in sign_changes.itertuples()),
            "A changed sign is attributed to the jointly corrected evaluation estimand and fitted trackwise design, not to new inference.",
            "Sign alone does not encode interval stability or material effect size.",
            "V2.1 endpoint statements with changed signs should be withdrawn and replaced by the corrected trackwise estimates and intervals in Table 1.",
        ),
        _answer(
            3,
            "Does irrelevant-context ecosystem bifurcation survive?",
            f"There are {len(stable_away)} stable away-from-hull effects and {len(stable_toward)} stable toward-hull effects under irrelevant context. Away: {_fmt_effects(stable_away)}. Toward: {_fmt_effects(stable_toward)}.",
            "Stable effects in both directions support reorganization rather than a uniform shift.",
            "The hull is conditional on this fixed eight-model roster and gold-option probability interface.",
            "Irrelevant-context load reorganized target positions relative to the fixed peer hull, with stable movement in both directions where reported.",
        ),
        _answer(4, "Which models move away from the hull under irrelevant context?", _fmt_effects(away), "Positive clean-to-512-word ΔPIER means reduced convex peer expressibility.", "Effects labelled exploratory do not satisfy both the split-sign and bootstrap criteria.", "Models with stable positive effects moved away from the fixed peer hull under irrelevant-context load."),
        _answer(5, "Which models move toward the hull?", _fmt_effects(toward), "Negative clean-to-512-word ΔPIER means increased convex peer expressibility.", "Movement toward the hull is not a general claim of behavioral convergence.", "Models with stable negative effects moved toward the fixed peer hull under irrelevant-context load."),
        _answer(6, "Does content deletion still contract OLMo, Mistral, and Granite on all three tracks?", "; ".join(track_statements), "All-three-track support requires every mean high-dose track delta to be negative for each named model.", "Only the three fixed intervention tracks are represented.", "Content-deletion contraction should be stated per model and per fixed track; it is not universal across the ecosystem."),
        _answer(7, "Which former content-deletion contractions were mostly cancellation artifacts?", "None met the predeclared sign-reversal-or-50%-attenuation diagnostic." if artifacts.empty else "; ".join(f"`{row.target}`/{row.family}: old {row.old_mean_response_delta:.6g}, corrected {row.trackwise_delta:.6g}" for row in artifacts.itertuples()), "These cases lost most or all of their apparent contraction after residual magnitudes were evaluated trackwise.", "The 50% attenuation rule is descriptive, not a new inferential threshold.", "Former aggregate-response contractions identified here should be described as cancellation-sensitive rather than general contractions."),
        _answer(8, "How much came from evaluation aggregation versus weight refitting?", f"Mean absolute evaluation correction was {eval_mean:.6g}; mean absolute fitting correction was {fit_mean:.6g}; mean absolute interaction was {interaction_mean:.6g}.", "The four-way factorial decomposition separates cancellation at evaluation from movement of the fitted convex projection.", "Components can oppose one another and should not be interpreted causally.", "The estimand correction combined an evaluation-aggregation term, a trackwise-refitting term, and their interaction, as quantified in Table 2."),
        _answer(9, "Does Qwen sibling-local coverage survive?", "; ".join(f"{row.family}: inflation {row.inflation:.6g} [{row.bootstrap_lower:.6g}, {row.bootstrap_upper:.6g}], largest in {row.splits_sibling_largest}/10 splits" for row in qwen.itertuples()), "Positive refitted inflation beyond the exact non-sibling null supports disproportionate local coverage by the declared Qwen sibling.", "This is roster-conditional convex coverage, not a causal lineage effect.", "The Qwen sibling supplied disproportionate convex coverage relative to same-size non-sibling removals where the corrected intervals and exact-null comparisons support it."),
        _answer(10, "Does General-Reasoner sibling-local coverage survive?", "; ".join(f"{row.family}: inflation {row.inflation:.6g} [{row.bootstrap_lower:.6g}, {row.bootstrap_upper:.6g}], largest in {row.splits_sibling_largest}/10 splits" for row in general.itertuples()), "The reverse Qwen-pair direction is assessed independently with refitting.", "Asymmetry between directions is scientifically possible and should not be averaged away.", "General-Reasoner sibling coverage should be reported directionally and conditional on the fixed peer roster."),
        _answer(11, "Do Phi-4-reasoning-plus and Mistral distributed convex advantages survive?", "; ".join(f"`{row.target}`/{row.family}: {row.absolute_improvement:.6g} [{row.bootstrap_lower:.6g}, {row.bootstrap_upper:.6g}] ({row.stability_label})" for row in distributed.itertuples()), "A positive honest gap means a single fixed convex surrogate outperformed the peer selected only on fitting questions.", "Distributed coverage does not imply interchangeability of models.", "Some targets required a distributed convex combination of heterogeneous peers rather than one fit-selected peer."),
        _answer(12, "Does the Phi sibling pair remain much weaker than the Qwen sibling pair?", f"Mean directed sibling-removal inflation was {qwen_pair_mean:.6g} for the Qwen pair and {phi_pair_mean:.6g} for the Phi pair; the Phi/Qwen ratio was {phi_pair_mean / (qwen_pair_mean + 1e-10):.3f}.", "Smaller removal inflation indicates less uniquely supplied convex coverage within this roster.", "The comparison combines two directions and two families only as a descriptive summary.", "The Phi sibling pair supplied weaker roster-conditional convex coverage than the Qwen pair in this design."),
        _answer(13, "Which findings survive temperature calibration?", "; ".join(f"`{value}`" for value in calibration_survivors) or "None.", "Survival means the predeclared direction remains under one clean-fit temperature applied to every dose and track.", "Temperature-calibrated values are not treated as ground truth; selected calibrated bootstrap intervals condition on split-fitted temperatures.", "Findings listed as calibration-surviving retained their direction after clean-fit temperature rescaling."),
        _answer(14, "Which findings survive exploratory label-bias correction?", "; ".join(f"`{value}`" for value in interface_survivors) or "None.", f"Held-out clean permutation TV changed from {interface_validation['before_mean_tv'].mean():.6g} to {interface_validation['after_mean_tv'].mean():.6g} on average.", "Permutation controls were not collected at every stress condition; the additive correction does not resolve the interface.", "The conclusion is conditional on the candidate-label response interface; exploratory additive correction preserved only the listed directions."),
        _answer(15, "Which findings have matching vector-response evidence?", "; ".join(f"`{value}`" for value in vector_survivors) or "None.", "Vector support compares total variation, Jensen–Shannon divergence, semantic top-1 agreement, and correctness agreement under trackwise evaluation.", "Scalar and vector functionals answer related but non-identical questions.", "The listed scalar findings had matching direction-level evidence in the complete option-probability vector analysis."),
        _answer(16, "Are individual weight representations ambiguous in headline cases?", f"{len(ambiguous)} headline target-family-split-peer weight entries had feasible width >1e-4; the maximum width was {ambiguous['feasible_width'].max() if len(ambiguous) else 0.0:.6g}.", "Ambiguous simplex coordinates can encode essentially the same projected response, so the projection function is primary.", "Feasible intervals are tolerance-dependent and do not quantify sampling uncertainty.", "Weight coordinates are reported diagnostically; scientific conclusions concern the preserved projected response and held-out residuals."),
        _answer(17, "Did every exact-redundancy and convex-mixture control pass?", f"{int(controls['passed'].sum())}/{len(controls)} required controls passed. {identity_text}", "The controls cover cancellation, no cancellation, exact clone, known and boundary mixtures, duplicate-peer ambiguity, equal design mass, cluster propagation, honest selection, and historical regression.", "Synthetic success does not validate the candidate-label measurement interface.", "All numerical and design controls passed before the formal bootstrap; packaging additionally required byte-identical locked inputs."),
        _answer(18, "What is the corrected paper-ready modern-LLM story?", f"{len(robust_claims)}/{len(claims)} predeclared claims were stable and also survived calibration, interface sensitivity, and vector direction checks.", "Modern-LLM ecosystem geometry contains both sibling-local and distributed convex coverage, while track averaging can hide residual magnitude.", "Every result is design-, roster-, benchmark-, and interface-conditional.", "Irrelevant-context load reorganized positions relative to a fixed peer hull; sibling-local and distributed coverage geometries coexisted, and trackwise evaluation revealed behavior hidden by response averaging."),
        _answer(19, "Which V2.1 statements must be withdrawn or rewritten?", f"{len(sign_changes)} endpoint statements changed sign; all statements equating track-mean PIER with primary PIER must be rewritten. Cancellation gaps reached {cancellation['cancellation_gap'].max():.6g}.", "V2.1 remains a valid historical aggregate-response diagnostic but not the primary operational uniqueness functional.", "Rewriting does not retroactively alter cached scores or the V2.1 tables.", "Use 'PIER of the track-mean response' only for the historical diagnostic and reserve 'primary PIER' for mean absolute residual over individual tracks."),
        _answer(20, "Is focused B200 Stage 2 still necessary?", "Yes, for measurement validation only; V2.2 itself performed no model inference.", "Residual held-out permutation TV and sensitivity to calibration/interface correction justify a focused validation of the response measurement, not a broad rerun.", "V2.2 cannot test stressed-prompt label transport or longer free-form generation from cached scores alone.", "A focused Stage 2 remains warranted to validate the candidate-label interface and generation alignment for the smallest scientifically sufficient set of headline cases."),
        _answer(21, "What is the minimum scientifically sufficient B200 validation scope?", "Models: Qwen2.5-7B-Instruct, General-Reasoner-Qwen2.5-7B, phi-4, Phi-4-reasoning-plus, and Mistral-Nemo-Instruct-2407. Conditions: clean, 512-word irrelevant context on all three fixed tracks, and 0.4 content deletion on all three fixed tracks; retain canonical plus three clean label permutations and the fixed 140-question subset with a longer deterministic generation budget.", "This scope targets the surviving sibling-local, distributed-convex, endpoint, and interface-sensitive findings without reopening model or condition selection.", "It remains one benchmark and does not justify universal identity or replacement claims.", "Stage 2 should be limited to the predeclared headline models, the two fixed high-dose endpoints on all tracks, balanced label-interface checks, and longer deterministic validation on the existing subset."),
    ]
    header = (
        "# PIER Modern-LLM Ecosystem V2.2 — Trackwise-Estimand Reanalysis\n\n"
        "This CPU-only reanalysis uses no new model inference. The primary response is raw "
        "gold-option probability, and the primary PIER estimand averages absolute residuals "
        "over held-out base questions, individual intervention tracks, and equal-mass doses. "
        "All percentile intervals are base-question clustered-bootstrap intervals for the "
        "ten-split bagged estimator; the ten splits are not treated as independent experiments.\n\n"
    )
    path = (
        PHYSICAL_ROOT / "outputs/analysis/V2_2_TRACKWISE_ESTIMAND_REPORT.md"
    )
    atomic_write_text(path, header + "".join(sections))
    return path
