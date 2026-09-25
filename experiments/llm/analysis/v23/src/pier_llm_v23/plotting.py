from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATUS_COLORS = {
    "LOCKED": "#1b9e77",
    "SUPPORTED_WITH_INTERFACE_QUALIFIER": "#d95f02",
    "RAW_INTERFACE_ONLY": "#7570b3",
    "NOT_SUPPORTED": "#666666",
}


def _short(model: str) -> str:
    return {
        "Qwen/Qwen2.5-7B-Instruct": "Qwen",
        "TIGER-Lab/General-Reasoner-Qwen2.5-7B": "General Reasoner",
        "microsoft/phi-4": "phi-4",
        "microsoft/Phi-4-reasoning-plus": "Phi-4 reasoning+",
        "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
        "allenai/OLMo-2-1124-13B-Instruct": "OLMo-2",
        "ibm-granite/granite-3.3-8b-instruct": "Granite",
        "google/gemma-3-12b-it": "Gemma-3",
    }.get(model, model)


def _save(fig: Any, root: Path, name: str, png: bool = False) -> None:
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    if png:
        fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_figures(root: Path, tables: dict[str, pd.DataFrame], claims: pd.DataFrame) -> list[Path]:
    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    table1 = tables["table_1_permutation_interface_quality.csv"]
    table2 = tables["table_2_endpoint_claim_comparison.csv"]
    table3 = tables["table_3_sibling_coverage.csv"]
    table4 = tables["table_4_distributed_convex_coverage.csv"]
    table5 = tables["table_5_generation_validation.csv"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.axis("off")
    steps = ["semantic\nquestion", "cyclic label\nrotations", "semantic\nremap", "average", "balanced\nresponse"]
    x = np.linspace(0.08, 0.92, len(steps))
    for index, (position, text) in enumerate(zip(x, steps, strict=True)):
        ax.text(position, 0.52, text, ha="center", va="center", bbox={"boxstyle": "round", "fc": "#eef3f8"})
        if index:
            ax.annotate("", xy=(position - 0.08, 0.52), xytext=(x[index - 1] + 0.08, 0.52), arrowprops={"arrowstyle": "->"})
    ax.set_title("A  Balanced semantic response construction", loc="left", fontweight="bold")

    ax = axes[0, 1]
    endpoint = table2.groupby("target", as_index=False)[
        ["v23_endpoint_raw_delta", "v23_permutation_averaged_delta"]
    ].mean()
    positions = np.arange(len(endpoint))
    ax.bar(positions - 0.18, endpoint["v23_endpoint_raw_delta"], 0.36, label="raw endpoint")
    ax.bar(positions + 0.18, endpoint["v23_permutation_averaged_delta"], 0.36, label="permutation avg")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions, [_short(value) for value in endpoint["target"]], rotation=55, ha="right")
    ax.set_ylabel("mean endpoint ΔPIER")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("B  Raw vs balanced endpoint movement", loc="left", fontweight="bold")

    ax = axes[0, 2]
    qwen = table3[table3["target"].str.contains("Qwen", regex=False)].copy()
    labels = [f"{_short(row.target)}\n{row.family}" for row in qwen.itertuples()]
    positions = np.arange(len(qwen))
    ax.bar(positions - 0.18, qwen["permutation_averaged_inflation"], 0.36, label="sibling")
    ax.bar(positions + 0.18, qwen["largest_non_sibling_inflation"], 0.36, label="largest non-sibling")
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C  Qwen sibling-local coverage", loc="left", fontweight="bold")

    ax = axes[1, 0]
    distributed = table4[table4["target"].isin([
        "microsoft/Phi-4-reasoning-plus", "mistralai/Mistral-Nemo-Instruct-2407"
    ])].copy()
    labels = [f"{_short(row.target)}\n{row.family}" for row in distributed.itertuples()]
    mse = distributed["perm_MSE_single_error"] - distributed["perm_MSE_convex_error"]
    mae = distributed["perm_MAE_single_error"] - distributed["perm_MAE_convex_error"]
    positions = np.arange(len(distributed))
    ax.bar(positions - 0.18, mse, 0.36, label="MSE fit")
    ax.bar(positions + 0.18, mae, 0.36, label="MAE fit")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("single − convex held-out MAE")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("D  Distributed convex coverage", loc="left", fontweight="bold")

    ax = axes[1, 1]
    generation = table5.groupby("model", as_index=False)[
        ["canonical_score_generation_agreement", "malformed_rate"]
    ].mean()
    positions = np.arange(len(generation))
    ax.bar(positions - 0.18, generation["canonical_score_generation_agreement"], 0.36, label="agreement")
    ax.bar(positions + 0.18, generation["malformed_rate"], 0.36, label="malformed")
    ax.set_xticks(positions, [_short(value) for value in generation["model"]], rotation=55, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("E  Long-generation validation", loc="left", fontweight="bold")

    ax = axes[1, 2]
    codes = {status: index for index, status in enumerate(STATUS_COLORS)}
    matrix = np.asarray([[codes[value]] for value in claims["final_status"]])
    cmap = matplotlib.colors.ListedColormap(list(STATUS_COLORS.values()))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(codes) - 0.5)
    ax.set_yticks(np.arange(len(claims)), claims["claim_id"], fontsize=6)
    ax.set_xticks([0], ["status"])
    ax.set_title("F  Final claim-locking matrix", loc="left", fontweight="bold")
    _save(fig, figure_root, "fig_v23_interface_validated_main", png=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    quality = table1.groupby("model", as_index=False)["mean_rotation_TV"].mean().sort_values("mean_rotation_TV")
    ax.barh([_short(value) for value in quality["model"]], quality["mean_rotation_TV"], color="#4c78a8")
    ax.set_xlabel("mean semantic-vector TV to permutation average")
    ax.set_title("Permutation sensitivity")
    _save(fig, figure_root, "fig_v23_permutation_sensitivity")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(table2["v23_endpoint_raw_delta"], table2["v23_permutation_averaged_delta"], c="#e45756")
    limits = [float(min(table2["v23_endpoint_raw_delta"].min(), table2["v23_permutation_averaged_delta"].min())), float(max(table2["v23_endpoint_raw_delta"].max(), table2["v23_permutation_averaged_delta"].max()))]
    ax.plot(limits, limits, color="black", lw=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("raw endpoint ΔPIER")
    ax.set_ylabel("permutation-averaged endpoint ΔPIER")
    ax.set_title("Endpoint interface transfer")
    _save(fig, figure_root, "fig_v23_endpoint_interface_transfer")

    fig, ax = plt.subplots(figsize=(10, 5))
    improvements = table4.assign(
        mse=lambda d: d["perm_MSE_single_error"] - d["perm_MSE_convex_error"],
        mae=lambda d: d["perm_MAE_single_error"] - d["perm_MAE_convex_error"],
    )
    ax.scatter(improvements["mse"], improvements["mae"], c="#72b7b2")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("MSE-fit convex improvement")
    ax.set_ylabel("MAE-fit convex improvement")
    ax.set_title("Matched-objective convexity")
    _save(fig, figure_root, "fig_v23_matched_objective_convexity")

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(table3))
    ax.bar(positions - 0.18, table3["permutation_averaged_inflation"], 0.36, label="sibling")
    ax.bar(positions + 0.18, table3["largest_non_sibling_inflation"], 0.36, label="largest non-sibling")
    ax.set_xticks(positions, [f"{_short(row.target)}\n{row.family}" for row in table3.itertuples()], rotation=50, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Sibling removal exact null")
    _save(fig, figure_root, "fig_v23_sibling_removal")

    fig, ax = plt.subplots(figsize=(10, 5))
    stability = table5.groupby("model", as_index=False)["generated_answer_rotation_stability"].mean()
    ax.bar([_short(value) for value in stability["model"]], stability["generated_answer_rotation_stability"], color="#f2cf5b")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=50)
    ax.set_title("Generated-answer permutation stability")
    _save(fig, figure_root, "fig_v23_generation_permutation_stability")

    fig, ax = plt.subplots(figsize=(10, 5))
    alignment = table5.groupby("model", as_index=False)[["canonical_score_generation_agreement", "all_rotation_score_generation_agreement"]].mean()
    positions = np.arange(len(alignment))
    ax.bar(positions - 0.18, alignment["canonical_score_generation_agreement"], 0.36, label="rotation 0")
    ax.bar(positions + 0.18, alignment["all_rotation_score_generation_agreement"], 0.36, label="all rotations")
    ax.set_xticks(positions, [_short(value) for value in alignment["model"]], rotation=50, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Score-generation alignment")
    _save(fig, figure_root, "fig_v23_score_generation_alignment")

    fig, ax = plt.subplots(figsize=(10, 5))
    widths = table2.assign(width=lambda d: d["bootstrap_upper"] - d["bootstrap_lower"])
    ax.bar([f"{_short(row.target)}\n{row.family}" for row in widths.itertuples()], widths["width"], color="#b279a2")
    ax.tick_params(axis="x", rotation=55)
    ax.set_ylabel("common-resample 95% interval width")
    ax.set_title("Common-bootstrap comparison")
    _save(fig, figure_root, "fig_v23_common_bootstrap_comparison")

    return sorted(figure_root.glob("fig_v23_*"))
