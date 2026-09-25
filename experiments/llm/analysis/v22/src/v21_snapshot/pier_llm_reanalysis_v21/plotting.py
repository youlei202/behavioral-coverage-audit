from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import high_dose
from .utils import PHYSICAL_ROOT

KEY_TARGETS = [
    "microsoft/phi-4",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-4-reasoning-plus",
    "mistralai/Mistral-Nemo-Instruct-2407",
]


def short_model(value: str) -> str:
    aliases = {
        "Qwen/Qwen2.5-7B-Instruct": "Qwen-7B",
        "TIGER-Lab/General-Reasoner-Qwen2.5-7B": "General-Reasoner",
        "microsoft/phi-4": "phi-4",
        "microsoft/Phi-4-reasoning-plus": "Phi-4-reasoning+",
        "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
        "allenai/OLMo-2-1124-13B-Instruct": "OLMo-2-13B",
        "ibm-granite/granite-3.3-8b-instruct": "Granite-8B",
        "google/gemma-3-12b-it": "Gemma-12B",
    }
    return aliases.get(value, value.rsplit("/", 1)[-1])


def _endpoint_forest(axis: Any, endpoint: pd.DataFrame) -> None:
    targets = list(dict.fromkeys(endpoint["target"]))
    positions = {target: index for index, target in enumerate(targets)}
    styles = {
        "irrelevant_context": ("o", "#b33f62", -0.13, "irrelevant 512-clean"),
        "content_deletion": ("s", "#3478a8", 0.13, "deletion 0.4-clean"),
    }
    for family, (marker, color, offset, label) in styles.items():
        group = endpoint[endpoint["family"].eq(family)]
        y = np.asarray([positions[target] + offset for target in group["target"]])
        x = group["mean_delta_absolute"].to_numpy()
        lower = x - group["bootstrap_lower_95"].to_numpy()
        upper = group["bootstrap_upper_95"].to_numpy() - x
        axis.errorbar(
            x,
            y,
            xerr=np.vstack([lower, upper]),
            fmt=marker,
            color=color,
            capsize=2,
            label=label,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(range(len(targets)), [short_model(target) for target in targets], fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("paired endpoint ΔPIER")
    axis.set_title("A. Corrected paired stress effects")
    axis.legend(fontsize=6, loc="best")


def _main_trajectory(axis: Any, interfaces: pd.DataFrame) -> None:
    group = interfaces[
        interfaces["target"].isin(KEY_TARGETS)
        & interfaces["family"].eq("irrelevant_context")
        & interfaces["dose"].notna()
    ]
    summarized = group.groupby(
        ["target", "interface_condition", "dose"], as_index=False
    )["pier"].mean()
    styles = {
        "raw": "-",
        "temperature_calibrated": "--",
        "label_bias_corrected": ":",
        "label_bias_corrected_then_temperature_calibrated": "-.",
    }
    colors = dict(zip(KEY_TARGETS, plt.cm.tab10.colors, strict=False))
    for (target, condition), rows in summarized.groupby(["target", "interface_condition"]):
        axis.plot(
            rows["dose"],
            rows["pier"],
            linestyle=styles[condition],
            color=colors[target],
            linewidth=1.0,
            alpha=0.9,
            label=f"{short_model(target)} / {condition.replace('_', ' ')}",
        )
    axis.set_xlabel("irrelevant-context dose (words)")
    axis.set_ylabel("PIER")
    axis.set_title("B. Four response interfaces")
    axis.legend(fontsize=4.5, ncol=2, loc="best")


def _convex_forest(axis: Any, convex: pd.DataFrame) -> None:
    ordered = convex.sort_values(["target", "family"]).reset_index(drop=True)
    y = np.arange(len(ordered))
    x = ordered["mean_absolute_improvement"].to_numpy()
    lower = x - ordered["bootstrap_lower_95"].to_numpy()
    upper = ordered["bootstrap_upper_95"].to_numpy() - x
    colors = np.where(ordered["family"].eq("irrelevant_context"), "#b33f62", "#3478a8")
    for index in range(len(ordered)):
        axis.errorbar(
            x[index],
            y[index],
            xerr=np.asarray([[lower[index]], [upper[index]]]),
            fmt="o",
            color=colors[index],
            capsize=2,
        )
    labels = [
        f"{short_model(row.target)} / {'irr' if row.family == 'irrelevant_context' else 'del'}"
        for row in ordered.itertuples()
    ]
    axis.set_yticks(y, labels, fontsize=6)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.invert_yaxis()
    axis.set_xlabel("single error − convex error")
    axis.set_title("C. Honest convexity improvement")


def _sibling_null(axis: Any, removal: pd.DataFrame) -> None:
    exact = removal[removal["removal_type"].eq("exact_sibling")].copy()
    groups = list(exact.groupby(["target", "family"], sort=True))
    for index, ((_target, _family), group) in enumerate(groups):
        null_values: list[float] = []
        for values in group["null_inflations"]:
            null_values.extend(float(value) for value in values)
        jitter = np.linspace(-0.18, 0.18, len(null_values)) if null_values else np.asarray([])
        axis.scatter(
            np.full(len(null_values), index) + jitter,
            null_values,
            s=8,
            color="#999999",
            alpha=0.55,
        )
        axis.scatter(
            [index],
            [group["observed_inflation"].mean()],
            marker="D",
            color="#d62728",
            s=28,
            zorder=3,
        )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(
        range(len(groups)),
        [
            f"{short_model(target)}\n{'irr' if family == 'irrelevant_context' else 'del'}"
            for (target, family), _group in groups
        ],
        rotation=55,
        ha="right",
        fontsize=6,
    )
    axis.set_ylabel("removal inflation")
    axis.set_title("D. Sibling removal vs non-sibling null")


def _track_panel(axis: Any, track: pd.DataFrame) -> None:
    high = track[
        track.apply(lambda row: float(row["dose"]) == high_dose(row["family"]), axis=1)
    ]
    summarized = high.groupby(["target", "family", "track"], as_index=False).agg(
        track_delta=("track_endpoint_delta", "mean"),
        aggregate_delta=("aggregate_endpoint_delta", "mean"),
        cancellation_gap=("track_cancellation_gap", "mean"),
    )
    summarized = summarized.sort_values(["target", "family", "track"])
    x = np.arange(len(summarized))
    axis.scatter(x, summarized["track_delta"], s=10, color="#4c78a8", label="track Δ")
    axis.scatter(
        x,
        summarized["aggregate_delta"],
        s=10,
        color="#e45756",
        marker="x",
        label="aggregate Δ",
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks([])
    axis.set_ylabel("endpoint ΔPIER")
    axis.set_title(
        f"E. Track consistency (max cancellation gap={summarized['cancellation_gap'].max():.3g})"
    )
    axis.legend(fontsize=6)


def _interface_validity(axis: Any, validation: pd.DataFrame, generation: pd.DataFrame) -> None:
    tv = validation.groupby("model_id", as_index=False).agg(
        before=("before_mean_tv", "mean"), after=("after_mean_tv", "mean")
    )
    agreement = generation[generation["condition"].eq("overall")].groupby(
        "model_id", as_index=False
    ).agg(
        raw=("raw_score_generation_agreement", "mean"),
        corrected=("bias_corrected_score_generation_agreement", "mean"),
    )
    merged = tv.merge(agreement, on="model_id", validate="one_to_one")
    axis.scatter(merged["before"], merged["raw"], label="raw", color="#e45756")
    axis.scatter(merged["after"], merged["corrected"], label="bias corrected", color="#4c78a8")
    for row in merged.itertuples():
        axis.annotate(short_model(row.model_id), (row.after, row.corrected), fontsize=5)
    axis.set_xlabel("held-out permutation TV")
    axis.set_ylabel("score-generation agreement")
    axis.set_title("F. Response-interface validity")
    axis.legend(fontsize=6)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _trajectory_figure(interfaces: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharey=False)
    styles = {
        "raw": ("raw", "-"),
        "temperature_calibrated": ("temperature", "--"),
        "label_bias_corrected": ("label bias", ":"),
        "label_bias_corrected_then_temperature_calibrated": ("bias + temperature", "-."),
    }
    for row_index, target in enumerate(KEY_TARGETS):
        for column_index, family in enumerate(("irrelevant_context", "content_deletion")):
            axis = axes[row_index, column_index]
            group = interfaces[
                interfaces["target"].eq(target)
                & interfaces["family"].eq(family)
                & interfaces["dose"].notna()
            ].groupby(["interface_condition", "dose"], as_index=False)["pier"].mean()
            for condition, rows in group.groupby("interface_condition"):
                label, linestyle = styles[condition]
                axis.plot(rows["dose"], rows["pier"], label=label, linestyle=linestyle)
            axis.set_title(f"{short_model(target)} — {family.replace('_', ' ')}", fontsize=9)
            axis.set_xlabel("dose")
            axis.set_ylabel("PIER")
            axis.legend(fontsize=6)
    fig.suptitle("V2.1 raw and interface-corrected dose trajectories")
    fig.tight_layout()
    _save(fig, path)


def make_all_figures() -> dict[str, Any]:
    figures = PHYSICAL_ROOT / "outputs/figures"
    endpoint = pd.read_csv(PHYSICAL_ROOT / "outputs/tables/table_v21_endpoint_effects.csv")
    convex = pd.read_csv(PHYSICAL_ROOT / "outputs/tables/table_v21_convexity_gap.csv")
    interfaces = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/interface_corrected_pier.parquet"
    )
    removal = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/peer_removal_null_corrected.parquet"
    )
    track = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/track_specific_pier.parquet")
    validation = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/label_bias_heldout_validation.parquet"
    )
    generation = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/generation_alignment_interface_check.parquet"
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    _endpoint_forest(axes[0, 0], endpoint)
    _main_trajectory(axes[0, 1], interfaces)
    _convex_forest(axes[1, 0], convex)
    _sibling_null(axes[1, 1], removal)
    _track_panel(axes[2, 0], track)
    _interface_validity(axes[2, 1], validation, generation)
    fig.suptitle("PIER Modern-LLM Ecosystem V2.1 corrected reanalysis", fontsize=16)
    fig.tight_layout()
    main_pdf = figures / "fig_v21_corrected_main.pdf"
    main_png = figures / "fig_v21_corrected_main.png"
    fig.savefig(main_pdf, bbox_inches="tight")
    fig.savefig(main_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    trajectory_path = figures / "fig_v21_calibrated_dose_trajectories.pdf"
    _trajectory_figure(interfaces, trajectory_path)

    estimates = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/label_bias_estimates.parquet")
    bias_summary = estimates.groupby(["model_id", "label"], as_index=False)["bias_estimate"].mean()
    fig, axis = plt.subplots(figsize=(10, 6))
    for model, group in bias_summary.groupby("model_id"):
        axis.plot(group["label"], group["bias_estimate"], marker="o", label=short_model(model))
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_title("Centered additive label-position bias estimates")
    axis.set_ylabel("candidate log-likelihood bias")
    axis.legend(fontsize=7, ncol=2)
    bias_path = figures / "fig_v21_label_bias_vectors.pdf"
    _save(fig, bias_path)

    subsets = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/generation_alignment_subsets.parquet"
    )
    subset_summary = subsets.groupby(
        ["target", "family", "interface_condition", "ecosystem_condition"], as_index=False
    )["endpoint_delta"].mean()
    fig, axis = plt.subplots(figsize=(11, 6))
    labels = [
        f"{short_model(row.target)} / {'irr' if row.family == 'irrelevant_context' else 'del'} / {row.interface_condition[:4]} / {'full' if row.ecosystem_condition.startswith('high') else 'restricted'}"
        for row in subset_summary.itertuples()
    ]
    axis.bar(np.arange(len(subset_summary)), subset_summary["endpoint_delta"], color="#4c78a8")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=75, ha="right", fontsize=5)
    axis.set_ylabel("endpoint ΔPIER")
    axis.set_title("Generation-alignment sensitivity ecosystems")
    subset_path = figures / "fig_v21_generation_alignment_subset.pdf"
    _save(fig, subset_path)

    influence = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/single_peer_removal_influence.parquet"
    )
    fig, axis = plt.subplots(figsize=(11, 6))
    non_sibling = influence[~influence["is_exact_sibling"]]
    sibling = influence[influence["is_exact_sibling"]]
    axis.scatter(
        [short_model(value) for value in non_sibling["target"]],
        non_sibling["inflation"],
        s=8,
        alpha=0.25,
        color="#777777",
        label="non-sibling removals",
    )
    axis.scatter(
        [short_model(value) for value in sibling["target"]],
        sibling["inflation"],
        s=18,
        alpha=0.8,
        color="#d62728",
        label="exact sibling",
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("held-out removal inflation")
    axis.set_title("Single-peer removal influence map")
    axis.legend(fontsize=7)
    influence_path = figures / "fig_v21_single_peer_removal_influence.pdf"
    _save(fig, influence_path)

    sensitivity = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/track_aggregation_sensitivity.parquet"
    )
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        sensitivity["primary_endpoint_delta"],
        sensitivity["trackwise_fit_endpoint_delta"],
        alpha=0.45,
    )
    limits = [
        min(sensitivity["primary_endpoint_delta"].min(), sensitivity["trackwise_fit_endpoint_delta"].min()),
        max(sensitivity["primary_endpoint_delta"].max(), sensitivity["trackwise_fit_endpoint_delta"].max()),
    ]
    axis.plot(limits, limits, color="black", linewidth=0.8)
    axis.set_xlabel("primary mean-response-fit endpoint Δ")
    axis.set_ylabel("trackwise-loss-fit endpoint Δ")
    axis.set_title("Track aggregation fitting sensitivity")
    track_path = figures / "fig_v21_track_aggregation_sensitivity.pdf"
    _save(fig, track_path)

    survival = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/interface_effect_survival.parquet"
    )
    fig, axis = plt.subplots(figsize=(12, 7))
    survival_plot = survival.copy()
    survival_plot["label"] = survival_plot.apply(
        lambda row: f"{short_model(row['target'])}/{row['family'][:3]}/{row['interface_condition'][:4]}",
        axis=1,
    )
    axis.bar(
        np.arange(len(survival_plot)),
        survival_plot["mean_endpoint_delta"],
        color=np.where(survival_plot["same_endpoint_sign_as_raw"], "#4c78a8", "#e45756"),
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(
        np.arange(len(survival_plot)), survival_plot["label"], rotation=80, ha="right", fontsize=5
    )
    axis.set_ylabel("mean endpoint ΔPIER")
    axis.set_title("Endpoint-effect survival across response interfaces")
    survival_path = figures / "fig_v21_interface_effect_survival.pdf"
    _save(fig, survival_path)

    outputs = [
        main_pdf,
        main_png,
        trajectory_path,
        bias_path,
        subset_path,
        influence_path,
        track_path,
        survival_path,
    ]
    return {"outputs": outputs, "figure_count": len(outputs)}
