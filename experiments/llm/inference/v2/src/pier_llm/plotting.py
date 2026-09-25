from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _placeholder(axis: Any, text: str) -> None:
    axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_pdf(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", metadata={"Creator": "PIER V2 pipeline"})
    plt.close(figure)


def _overall_scalar(scalar: pd.DataFrame) -> pd.DataFrame:
    if scalar.empty:
        return scalar.copy()
    mask = scalar["peer_set_condition"].eq("all_peers")
    return scalar[mask].copy()


def make_main_figure(
    output_dir: Path,
    scalar: pd.DataFrame,
    vector: pd.DataFrame,
    peer_removal: pd.DataFrame,
    transfer: pd.DataFrame,
) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    scalar_primary = _overall_scalar(scalar)
    scalar_primary = scalar_primary[scalar_primary["representation"].eq("gold_probability")]

    axis = axes[0, 0]
    landscape = scalar_primary[scalar_primary["dose"].notna()]
    if landscape.empty:
        _placeholder(axis, "PIER landscape unavailable")
    else:
        landscape = landscape.assign(
            design=landscape["family"].astype(str) + "\n" + landscape["dose"].astype(str)
        )
        pivot = landscape.pivot_table(index="target", columns="design", values="pier", aggfunc="mean")
        sns.heatmap(pivot, ax=axis, cmap="mako", cbar_kws={"label": "held-out PIER"})
    axis.set_title("A  PIER landscape")

    axis = axes[0, 1]
    if landscape.empty:
        _placeholder(axis, "Dose trajectories unavailable")
    else:
        summarized = (
            landscape.groupby(["target", "family", "dose"], as_index=False)["pier"].mean()
        )
        for (target, family), group in summarized.groupby(["target", "family"]):
            linestyle = "-" if family == "irrelevant_context" else "--"
            axis.plot(group["dose"], group["pier"], linestyle=linestyle, alpha=0.65, label=target)
        axis.set_xlabel("dose (family-specific units)")
        axis.set_ylabel("held-out PIER")
    axis.set_title("B  Dose trajectories")

    axis = axes[0, 2]
    overall = scalar_primary[scalar_primary["dose"].isna()]
    if overall.empty:
        _placeholder(axis, "Convexity comparison unavailable")
    else:
        axis.scatter(overall["fit_selected_single_error"], overall["pier"], alpha=0.65)
        limit = float(
            max(overall["fit_selected_single_error"].max(), overall["pier"].max(), 1e-6)
        )
        axis.plot([0, limit], [0, limit], color="black", linestyle=":", linewidth=1)
        axis.set_xlabel("fit-selected single-peer error")
        axis.set_ylabel("fixed-convex PIER")
    axis.set_title("C  Honest convexity gap")

    axis = axes[1, 0]
    if peer_removal.empty:
        _placeholder(axis, "Peer-removal controls unavailable")
    else:
        columns = [
            name
            for name in ("observed_inflation", "random_median_inflation")
            if name in peer_removal
        ]
        summary = peer_removal.groupby("removal_type")[columns].mean()
        summary.plot(kind="bar", ax=axis)
        axis.set_ylabel("PIER inflation")
        axis.legend(fontsize=8)
    axis.set_title("D  Lineage effect vs peer-count control")

    axis = axes[1, 1]
    vector_overall = vector[vector["dose"].isna()] if not vector.empty else vector
    if overall.empty or vector_overall.empty:
        _placeholder(axis, "Scalar/vector comparison unavailable")
    else:
        merged = overall.merge(
            vector_overall[["target", "family", "split_seed", "mean_total_variation"]],
            on=["target", "family", "split_seed"],
            how="inner",
        )
        axis.scatter(merged["pier"], merged["mean_total_variation"], alpha=0.65)
        axis.set_xlabel("scalar PIER")
        axis.set_ylabel("vector total variation")
    axis.set_title("E  Scalar-to-vector fidelity")

    axis = axes[1, 2]
    if transfer.empty:
        _placeholder(axis, "Design-transfer results unavailable")
    else:
        columns = ["same_design_error", "transferred_error"]
        summary = transfer.groupby("transfer_type")[columns].mean()
        summary.plot(kind="bar", ax=axis)
        axis.set_ylabel("held-out residual")
        axis.legend(fontsize=8)
    axis.set_title("F  Design-transfer penalty")

    pdf = output_dir / "fig_modern_llm_ecosystem_main.pdf"
    png = output_dir / "fig_modern_llm_ecosystem_main.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf, bbox_inches="tight", metadata={"Creator": "PIER V2 pipeline"})
    figure.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(figure)
    return [pdf, png]


def _simple_figure(path: Path, title: str, draw: Any) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    try:
        drawn = bool(draw(axis))
    except (KeyError, ValueError, TypeError):
        drawn = False
    if not drawn:
        _placeholder(axis, "No applicable rows in this run")
    axis.set_title(title)
    _save_pdf(figure, path)
    return path


def make_secondary_figures(
    output_dir: Path,
    scalar: pd.DataFrame,
    vector: pd.DataFrame,
    stability: pd.DataFrame,
    permutation: pd.DataFrame,
    generation: pd.DataFrame,
    ambiguity: pd.DataFrame,
    coverage: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []

    def representation(axis: Any) -> bool:
        overall = scalar[scalar["dose"].isna()]
        if overall.empty:
            return False
        pivot = overall.pivot_table(index="target", columns="representation", values="pier")
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("PIER (representation-specific scale)")
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_response_representation_sensitivity.pdf",
            "Response-representation sensitivity (descriptive point estimates)",
            representation,
        )
    )

    def split_plot(axis: Any) -> bool:
        if stability.empty:
            return False
        axis.scatter(
            np.arange(len(stability)), stability["median_rank_correlation"], label="median"
        )
        axis.scatter(
            np.arange(len(stability)), stability["minimum_rank_correlation"], label="minimum"
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("pairwise Spearman correlation")
        axis.legend()
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_split_stability.pdf", "Repeated-split rank stability", split_plot
        )
    )

    def permutation_plot(axis: Any) -> bool:
        if permutation.empty:
            return False
        summary = permutation.groupby("model_id")["probability_vector_tv"].mean().sort_values()
        summary.plot(kind="barh", ax=axis)
        axis.set_xlabel("mean clean-vs-permuted TV")
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_option_permutation_control.pdf",
            "Option-label permutation nuisance audit",
            permutation_plot,
        )
    )

    def generation_plot(axis: Any) -> bool:
        if generation.empty:
            return False
        summary = generation.groupby("model_id")[
            ["score_generation_agreement", "malformed"]
        ].mean()
        summary.plot(kind="bar", ax=axis)
        axis.set_ylim(0, 1)
        axis.set_ylabel("fraction")
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_generation_validation.pdf",
            "Candidate-score versus direct-generation validation",
            generation_plot,
        )
    )

    def ambiguity_plot(axis: Any) -> bool:
        if ambiguity.empty:
            return False
        summary = ambiguity.groupby("peer")["interval_width"].max().sort_values()
        summary.plot(kind="barh", ax=axis)
        axis.set_xlabel("maximum feasible weight interval width")
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_weight_ambiguity.pdf",
            "Projection-weight ambiguity audit",
            ambiguity_plot,
        )
    )

    def coverage_plot(axis: Any) -> bool:
        if coverage.empty:
            return False
        summary = coverage.groupby("peer_subset_size")["pier"].agg(["min", "median", "max"])
        axis.plot(summary.index, summary["median"], marker="o", label="median subset")
        axis.plot(summary.index, summary["min"], marker="o", label="best subset")
        axis.fill_between(summary.index, summary["min"], summary["max"], alpha=0.2)
        axis.set_xlabel("peer subset size")
        axis.set_ylabel("held-out PIER")
        axis.legend()
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_ecosystem_coverage_curve.pdf",
            "Exploratory real-ecosystem coverage curve",
            coverage_plot,
        )
    )

    def estimators(axis: Any) -> bool:
        overall = scalar[scalar["dose"].isna()]
        required = [
            "uniform_peer_error",
            "fit_selected_single_error",
            "pier",
            "dose_specific_convex_error",
            "affine_ridge_error",
            "oracle_single_peer_error",
        ]
        if overall.empty or not set(required).issubset(overall.columns):
            return False
        overall[required].mean().plot(kind="bar", ax=axis)
        axis.set_ylabel("held-out absolute error")
        axis.tick_params(axis="x", rotation=35)
        return True

    paths.append(
        _simple_figure(
            output_dir / "fig_estimator_class_comparison.pdf",
            "Estimator-class comparison",
            estimators,
        )
    )
    return paths


def make_all_figures(
    output_dir: Path,
    *,
    scalar: pd.DataFrame,
    vector: pd.DataFrame,
    peer_removal: pd.DataFrame,
    transfer: pd.DataFrame,
    stability: pd.DataFrame,
    permutation: pd.DataFrame,
    generation: pd.DataFrame,
    ambiguity: pd.DataFrame,
    coverage: pd.DataFrame,
) -> list[Path]:
    paths = make_main_figure(output_dir, scalar, vector, peer_removal, transfer)
    paths.extend(
        make_secondary_figures(
            output_dir,
            scalar,
            vector,
            stability,
            permutation,
            generation,
            ambiguity,
            coverage,
        )
    )
    return paths
