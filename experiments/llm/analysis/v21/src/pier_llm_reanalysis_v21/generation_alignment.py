from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .core import FAMILIES, fixed_splits, high_dose, primary_fit
from .input_validation import load_validated_inputs
from .interface_bias import (
    LabelBiasFit,
    aggregate_interface_responses,
    corrected_semantic_probabilities,
)
from .utils import PHYSICAL_ROOT, atomic_parquet, load_config


def _fits_from_estimates(
    estimates: pd.DataFrame,
    model_ids: list[str],
    split_seed: int,
    fitting_ids: frozenset[str],
) -> dict[str, LabelBiasFit]:
    fits: dict[str, LabelBiasFit] = {}
    for model in model_ids:
        subset = estimates[
            estimates["model_id"].eq(model) & estimates["split_seed"].eq(split_seed)
        ]
        biases = {
            int(option_count): group.sort_values("label_position")[
                "bias_estimate"
            ].to_numpy(dtype=np.float64)
            for option_count, group in subset.groupby("option_count", sort=True)
        }
        fits[model] = LabelBiasFit(
            model_id=model,
            split_seed=split_seed,
            biases=biases,
            diagnostics=pd.DataFrame(),
            fitting_question_ids=fitting_ids,
        )
    return fits


def _raw_generation_merge(scores: pd.DataFrame, generation: pd.DataFrame) -> pd.DataFrame:
    lookup = scores[
        ["model_id", "prompt_id", "argmax_label"]
    ].rename(columns={"prompt_id": "scoring_prompt_id", "argmax_label": "raw_score_label"})
    merged = generation.merge(
        lookup, on=["model_id", "scoring_prompt_id"], validate="one_to_one"
    )
    merged["raw_score_generation_agreement"] = (
        merged["extracted_label"].notna()
        & merged["extracted_label"].eq(merged["raw_score_label"])
    )
    return merged


def _summary_record(group: pd.DataFrame, *, model: str, condition: str, split_seed: int) -> dict[str, Any]:
    valid = group["extracted_label"].notna()
    return {
        "model_id": model,
        "split_seed": split_seed,
        "condition": condition,
        "row_count": len(group),
        "valid_extracted_label_count": int(valid.sum()),
        "raw_score_generation_agreement": float(
            group["raw_score_generation_agreement"].mean()
        ),
        "raw_agreement_conditional_valid_label": float(
            group.loc[valid, "raw_score_generation_agreement"].mean()
        )
        if valid.any()
        else math.nan,
        "bias_corrected_score_generation_agreement": float(
            group["bias_corrected_score_generation_agreement"].mean()
        ),
        "bias_corrected_agreement_conditional_valid_label": float(
            group.loc[valid, "bias_corrected_score_generation_agreement"].mean()
        )
        if valid.any()
        else math.nan,
        "malformed_or_no_label_rate": float((~valid).mean()),
    }


def run_generation_alignment_analysis() -> dict[str, Any]:
    scores, generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    estimates = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/label_bias_estimates.parquet")
    merged_raw = _raw_generation_merge(scores, generation)
    overall_raw = merged_raw.groupby("model_id")["raw_score_generation_agreement"].mean()
    threshold = float(load_config()["high_alignment_threshold"])
    high_alignment = [
        model for model in model_ids if float(overall_raw.get(model, 0.0)) >= threshold
    ]
    run_restricted = len(high_alignment) >= 4
    score_lookup = scores.set_index(["model_id", "prompt_id"], drop=False)
    interface_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []

    for split in splits.values():
        fits = _fits_from_estimates(
            estimates, model_ids, split.seed, split.fitting_ids
        )
        merged = merged_raw.copy()
        corrected_labels: list[str] = []
        for _, generation_row in merged.iterrows():
            model = str(generation_row["model_id"])
            score_row = score_lookup.loc[(model, str(generation_row["scoring_prompt_id"]))]
            probabilities = corrected_semantic_probabilities(score_row, fits[model].biases)
            semantic_index = int(np.argmax(probabilities))
            corrected_labels.append(str(score_row["option_labels"][semantic_index]))
        merged["bias_corrected_score_label"] = corrected_labels
        merged["bias_corrected_score_generation_agreement"] = (
            merged["extracted_label"].notna()
            & merged["extracted_label"].eq(merged["bias_corrected_score_label"])
        )
        for model in model_ids:
            model_group = merged[merged["model_id"].eq(model)]
            interface_rows.append(
                _summary_record(
                    model_group, model=model, condition="overall", split_seed=split.seed
                )
            )
            for condition, condition_group in model_group.groupby("condition", sort=True):
                interface_rows.append(
                    _summary_record(
                        condition_group,
                        model=model,
                        condition=str(condition),
                        split_seed=split.seed,
                    )
                )

        for family in FAMILIES:
            aggregates = {
                "raw": aggregate_interface_responses(scores, family),
                "label_bias_corrected": aggregate_interface_responses(
                    scores, family, fits=fits
                ),
            }
            for interface_condition, aggregated in aggregates.items():
                ecosystems: list[tuple[str, list[str]]] = [
                    ("high_alignment_targets_full_peers", model_ids)
                ]
                if run_restricted:
                    ecosystems.append(
                        ("restricted_high_alignment_ecosystem", high_alignment)
                    )
                for ecosystem, ecosystem_models in ecosystems:
                    for target in high_alignment:
                        peers = [model for model in ecosystem_models if model != target]
                        if not peers:
                            continue
                        result = primary_fit(aggregated, target, peers, split)
                        frame = result.dose_results
                        clean = float(
                            frame[frame["dose"].eq(0.0)]["convex_error"].iloc[0]
                        )
                        high = float(
                            frame[frame["dose"].eq(high_dose(family))][
                                "convex_error"
                            ].iloc[0]
                        )
                        overall = frame[frame["dose"].isna()].iloc[0]
                        subset_rows.append(
                            {
                                "target": target,
                                "family": family,
                                "split_seed": split.seed,
                                "interface_condition": interface_condition,
                                "ecosystem_condition": ecosystem,
                                "target_count": len(high_alignment),
                                "peer_count": len(peers),
                                "high_alignment_threshold": threshold,
                                "high_alignment_models": high_alignment,
                                "endpoint_delta": high - clean,
                                "overall_pier": float(overall["convex_error"]),
                                "convex_vs_single_absolute_improvement": float(
                                    overall["absolute_improvement"]
                                ),
                                "fit_selected_single_peer": str(
                                    overall["fit_selected_single_peer"]
                                ),
                            }
                        )

    interface = pd.DataFrame(interface_rows)
    interface["enters_high_alignment_subset"] = interface["model_id"].isin(high_alignment)
    subsets = pd.DataFrame(subset_rows)
    full = subsets[
        subsets["ecosystem_condition"].eq("high_alignment_targets_full_peers")
    ][
        [
            "target",
            "family",
            "split_seed",
            "interface_condition",
            "endpoint_delta",
            "overall_pier",
        ]
    ].rename(
        columns={
            "endpoint_delta": "full_ecosystem_endpoint_delta",
            "overall_pier": "full_ecosystem_overall_pier",
        }
    )
    subsets = subsets.merge(
        full,
        on=["target", "family", "split_seed", "interface_condition"],
        validate="many_to_one",
    )
    subsets["effect_sign_agreement_with_full"] = (
        np.sign(subsets["endpoint_delta"])
        == np.sign(subsets["full_ecosystem_endpoint_delta"])
    )
    correlations: dict[tuple[str, str, str], float] = {}
    aggregate = subsets.groupby(
        ["ecosystem_condition", "family", "interface_condition", "target"],
        as_index=False,
    )["overall_pier"].mean()
    aggregate_full = aggregate[
        aggregate["ecosystem_condition"].eq("high_alignment_targets_full_peers")
    ]
    for (ecosystem, family, interface_condition), group in aggregate.groupby(
        ["ecosystem_condition", "family", "interface_condition"], sort=True
    ):
        reference = aggregate_full[
            aggregate_full["family"].eq(family)
            & aggregate_full["interface_condition"].eq(interface_condition)
        ][["target", "overall_pier"]].rename(columns={"overall_pier": "reference"})
        paired = group[["target", "overall_pier"]].merge(
            reference, on="target", validate="one_to_one"
        )
        correlation = (
            float(spearmanr(paired["overall_pier"], paired["reference"]).statistic)
            if len(paired) >= 2
            else math.nan
        )
        correlations[(str(ecosystem), str(family), str(interface_condition))] = correlation
    subsets["aggregate_rank_correlation_with_full"] = [
        correlations[(row.ecosystem_condition, row.family, row.interface_condition)]
        for row in subsets.itertuples()
    ]

    subset_path = PHYSICAL_ROOT / "outputs/analysis/generation_alignment_subsets.parquet"
    interface_path = (
        PHYSICAL_ROOT / "outputs/analysis/generation_alignment_interface_check.parquet"
    )
    atomic_parquet(subset_path, subsets)
    atomic_parquet(interface_path, interface)
    return {
        "outputs": [subset_path, interface_path],
        "high_alignment_models": high_alignment,
        "restricted_ecosystem_run": run_restricted,
        "subset_rows": len(subsets),
        "interface_rows": len(interface),
    }
