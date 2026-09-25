from __future__ import annotations

# ruff: noqa: E402, I001

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import PHYSICAL_ROOT


MODEL_LABELS = {
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5",
    "TIGER-Lab/General-Reasoner-Qwen2.5-7B": "General-Reasoner",
    "microsoft/phi-4": "phi-4",
    "microsoft/Phi-4-reasoning-plus": "Phi-4-reasoning+",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
    "allenai/OLMo-2-1124-13B-Instruct": "OLMo-2",
    "ibm-granite/granite-3.3-8b-instruct": "Granite-3.3",
    "google/gemma-3-12b-it": "Gemma-3",
}


def _label(value: str) -> str:
    return MODEL_LABELS.get(value, value)


def _save(fig: plt.Figure, name: str, *, png: bool = False) -> list[Path]:
    outputs: list[Path] = []
    pdf = PHYSICAL_ROOT / "figures" / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    outputs.append(pdf)
    if png:
        raster = PHYSICAL_ROOT / "figures" / f"{name}.png"
        fig.savefig(raster, dpi=220, bbox_inches="tight")
        outputs.append(raster)
    plt.close(fig)
    return outputs


def _claim_matrix(claims: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            claims["v22_raw_trackwise_status"].eq("supported"),
            claims["bootstrap_status"].eq("stable"),
            claims["calibration_status"].eq("survives"),
            claims["interface_sensitivity_status"].eq("survives"),
            claims["vector_support_status"].eq("supported"),
        ]
    ).astype(float)


def make_main_figure() -> list[Path]:
    dose = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    )
    endpoint = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_1_corrected_endpoint_effects.csv"
    )
    cancellation = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_5_cancellation_diagnosis.csv"
    )
    convexity = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_3_convexity_gap.csv"
    )
    sibling = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_4_sibling_coverage.csv"
    )
    null = pd.read_parquet(
        PHYSICAL_ROOT
        / "outputs/analysis/trackwise_non_sibling_removal_null.parquet"
    )
    claims = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_6_claim_survival.csv"
    )
    mean_dose = dose.groupby(["target", "family", "dose"], as_index=False)[
        "trackwise_pier"
    ].mean()
    roster = list(MODEL_LABELS)
    family_doses = [
        (family, value)
        for family in ("irrelevant_context", "content_deletion")
        for value in sorted(mean_dose[mean_dose["family"].eq(family)]["dose"].unique())
    ]
    landscape = np.asarray(
        [
            [
                mean_dose[
                    mean_dose["target"].eq(target)
                    & mean_dose["family"].eq(family)
                    & mean_dose["dose"].eq(dose_value)
                ]["trackwise_pier"].iloc[0]
                for family, dose_value in family_doses
            ]
            for target in roster
        ]
    )
    fig, axes = plt.subplots(3, 2, figsize=(18, 21), constrained_layout=True)

    ax = axes[0, 0]
    image = ax.imshow(landscape, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(roster)), [_label(value) for value in roster])
    ax.set_xticks(
        range(len(family_doses)),
        [f"{'IC' if family == 'irrelevant_context' else 'CD'}\n{dose:g}" for family, dose in family_doses],
    )
    ax.set_title("A  Corrected trackwise PIER landscape")
    fig.colorbar(image, ax=ax, shrink=0.75, label="ten-split mean PIER")

    ax = axes[0, 1]
    ordered = endpoint.sort_values(["family", "trackwise_delta"])
    y = np.arange(len(ordered))
    colors = np.where(ordered["family"].eq("irrelevant_context"), "#7b3294", "#008837")
    ax.axvline(0, color="black", linewidth=0.8)
    for position, row, color in zip(y, ordered.itertuples(), colors, strict=True):
        ax.errorbar(
            row.trackwise_delta,
            position,
            xerr=[
                [row.trackwise_delta - row.bootstrap_lower],
                [row.bootstrap_upper - row.trackwise_delta],
            ],
            fmt="none",
            ecolor=color,
            alpha=0.75,
            capsize=2,
        )
    ax.scatter(ordered["trackwise_delta"], y, c=colors, s=35)
    ax.set_yticks(
        y,
        [
            f"{_label(row.target)} · {'IC' if row.family == 'irrelevant_context' else 'CD'}"
            for row in ordered.itertuples()
        ],
        fontsize=8,
    )
    ax.set_xlabel("high-dose − clean trackwise PIER")
    ax.set_title("B  Stress-induced movement with clustered-bootstrap intervals")

    ax = axes[1, 0]
    ax.scatter(
        cancellation["track_mean_response_pier"],
        cancellation["trackwise_pier"],
        c=cancellation["cancellation_gap"],
        cmap="magma",
        alpha=0.75,
    )
    limit = float(
        max(
            cancellation["track_mean_response_pier"].max(),
            cancellation["trackwise_pier"].max(),
        )
    )
    ax.plot([0, limit], [0, limit], linestyle="--", color="gray")
    for row in cancellation.nlargest(5, "cancellation_gap").itertuples():
        ax.annotate(
            f"{_label(row.target)}\n{row.family[:2]} {row.dose:g}",
            (row.track_mean_response_pier, row.trackwise_pier),
            fontsize=7,
        )
    ax.set_xlabel("PIER of the track-mean response")
    ax.set_ylabel("mean trackwise PIER")
    ax.set_title("C  Cancellation revealed")

    ax = axes[1, 1]
    coverage_records: list[tuple[str, float, str]] = []
    for row in sibling[
        sibling["target"].eq("Qwen/Qwen2.5-7B-Instruct")
    ].itertuples():
        coverage_records.append(
            (f"Qwen sibling\n{row.family[:2]}", row.inflation, "sibling-local")
        )
    for row in convexity[
        convexity["target"].isin(
            [
                "microsoft/Phi-4-reasoning-plus",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ]
        )
    ].itertuples():
        coverage_records.append(
            (
                f"{_label(row.target)}\n{row.family[:2]}",
                row.absolute_improvement,
                "distributed",
            )
        )
    labels, values, kinds = zip(*coverage_records, strict=True)
    ax.bar(
        range(len(values)),
        values,
        color=["#2166ac" if kind == "sibling-local" else "#b2182b" for kind in kinds],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("trackwise held-out advantage / inflation")
    ax.set_title("D  Sibling-local and distributed coverage geometries")

    ax = axes[2, 0]
    sibling_groups = sibling.sort_values(["target", "family"]).reset_index(drop=True)
    distributions = []
    names = []
    observed = []
    for row in sibling_groups.itertuples():
        group = null[
            null["target"].eq(row.target) & null["family"].eq(row.family)
        ]
        distributions.append(group["non_sibling_inflation"].to_numpy())
        names.append(f"{_label(row.target)}\n{row.family[:2]}")
        observed.append(row.inflation)
    positions = np.arange(1, len(distributions) + 1)
    ax.boxplot(distributions, positions=positions, showfliers=False)
    ax.scatter(positions, observed, color="#d73027", label="sibling removal", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(positions, names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("overall trackwise PIER inflation")
    ax.set_title("E  Sibling influence beyond exact non-sibling removal")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2, 1]
    matrix = _claim_matrix(claims)
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(
        range(5),
        ["raw\ntrackwise", "bootstrap", "calibrated", "interface", "vector"],
    )
    ax.set_yticks(
        range(len(claims)),
        [str(value) for value in claims["claim"]],
        fontsize=7,
    )
    ax.set_title("F  Dimension-specific claim robustness")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(claims), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    return _save(fig, "fig_v22_trackwise_corrected_main", png=True)


def _simple_figure(
    title: str,
    xlabel: str,
    ylabel: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str = "scatter",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if kind == "line":
        ax.plot(x, y, marker="o")
    elif kind == "bar":
        ax.bar(x, y)
    else:
        ax.scatter(x, y, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="gray", linewidth=0.7)
    return fig


def make_secondary_figures() -> list[Path]:
    outputs: list[Path] = []
    decomp = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_2_estimand_decomposition.csv"
    )
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    summary = decomp.groupby("family")[["evaluation_correction", "fitting_correction", "interaction"]].mean()
    summary.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mean change in PIER")
    ax.set_title("Four-way estimand decomposition")
    outputs += _save(fig, "fig_v22_estimand_decomposition")

    dose = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_primary_dose_results.parquet"
    )
    mean_dose = dose.groupby(["target", "family", "dose"], as_index=False)["trackwise_pier"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, family in zip(axes, ("irrelevant_context", "content_deletion"), strict=True):
        for target, group in mean_dose[mean_dose["family"].eq(family)].groupby("target"):
            ax.plot(group["dose"], group["trackwise_pier"], marker="o", label=_label(target))
        ax.set_title(family.replace("_", " "))
        ax.set_xlabel("dose")
        ax.set_ylabel("trackwise PIER")
    axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1))
    outputs += _save(fig, "fig_v22_trackwise_dose_trajectories")

    cancellation = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_5_cancellation_diagnosis.csv"
    )
    target_cancel = cancellation.groupby("target")["cancellation_gap"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh([_label(value) for value in target_cancel.index], target_cancel.values)
    ax.set_xlabel("mean cancellation gap")
    ax.set_title("Cancellation by target")
    outputs += _save(fig, "fig_v22_cancellation_by_target")

    convexity = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_3_convexity_gap.csv"
    ).sort_values("absolute_improvement")
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    labels = [f"{_label(row.target)} · {row.family[:2]}" for row in convexity.itertuples()]
    ax.barh(labels, convexity["absolute_improvement"], color="#762a83")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("single − convex overall trackwise error")
    ax.set_title("Honest convexity gap")
    outputs += _save(fig, "fig_v22_convexity_gap")

    sibling = pd.read_csv(
        PHYSICAL_ROOT / "outputs/tables/table_4_sibling_coverage.csv"
    ).sort_values("inflation")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    labels = [f"{_label(row.target)} · {row.family[:2]}" for row in sibling.itertuples()]
    ax.barh(labels, sibling["inflation"], color="#1b7837")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("sibling-removal inflation")
    ax.set_title("Directed sibling removal")
    outputs += _save(fig, "fig_v22_sibling_removal")

    calibration = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_calibrated_results.parquet"
    )
    subset = calibration[calibration["dose"].eq(0.0)].groupby(["target", "family"], as_index=False).agg(raw=("raw_endpoint_delta", "mean"), corrected=("calibrated_endpoint_delta", "mean"))
    fig = _simple_figure("Temperature-calibration sensitivity", "raw endpoint ΔPIER", "calibrated endpoint ΔPIER", subset["raw"].to_numpy(), subset["corrected"].to_numpy())
    ax = fig.axes[0]
    limits = [min(subset[["raw", "corrected"]].min()), max(subset[["raw", "corrected"]].max())]
    ax.plot(limits, limits, linestyle="--", color="gray")
    outputs += _save(fig, "fig_v22_calibration_sensitivity")

    interface = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_interface_sensitivity.parquet"
    )
    subset = interface[interface["dose"].eq(0.0)].groupby(["target", "family"], as_index=False).agg(raw=("raw_endpoint_delta", "mean"), corrected=("interface_corrected_endpoint_delta", "mean"))
    fig = _simple_figure("Exploratory interface sensitivity", "raw endpoint ΔPIER", "interface-corrected endpoint ΔPIER", subset["raw"].to_numpy(), subset["corrected"].to_numpy())
    ax = fig.axes[0]
    limits = [min(subset[["raw", "corrected"]].min()), max(subset[["raw", "corrected"]].max())]
    ax.plot(limits, limits, linestyle="--", color="gray")
    outputs += _save(fig, "fig_v22_interface_sensitivity")

    vector = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/trackwise_vector_results.parquet"
    )
    subset = vector[(vector["dose"].eq(0.0)) & vector["weight_source"].eq("vector_fitted")]
    fig = _simple_figure("Vector-response endpoint support", "scalar endpoint ΔPIER", "vector endpoint ΔTV", subset["scalar_endpoint_delta"].to_numpy(), subset["vector_endpoint_delta"].to_numpy())
    outputs += _save(fig, "fig_v22_vector_support")

    solver = pd.read_parquet(
        PHYSICAL_ROOT / "outputs/analysis/solver_diagnostics.parquet"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].hist(solver["normalized_weighted_objective"], bins=25, color="#2166ac")
    axes[0].set_title("Normalized weighted fit objective")
    axes[0].set_xlabel("weighted MSE")
    axes[1].hist(solver["maximum_projection_change"], bins=25, color="#b2182b")
    axes[1].set_title("Stage-2 projection preservation")
    axes[1].set_xlabel("maximum projected-response change")
    outputs += _save(fig, "fig_v22_solver_diagnostics")
    return outputs


def make_all_figures() -> dict[str, Any]:
    outputs = [*make_main_figure(), *make_secondary_figures()]
    return {"outputs": outputs, "row_counts": {"figures": len(outputs)}}
