from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd
from pier_llm.analysis import aggregate_family_responses

from .core import FAMILIES, fixed_splits, primary_fit
from .input_validation import load_validated_inputs
from .utils import PHYSICAL_ROOT, atomic_parquet, stable_seed


def designated_removals(
    target: str,
    peers: list[str],
    records: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    target_record = records[target]
    exact_group = target_record.get("exact_sibling_group")
    exact = [
        peer
        for peer in peers
        if exact_group is not None and records[peer].get("exact_sibling_group") == exact_group
    ]
    lineage = [
        peer
        for peer in peers
        if records[peer].get("broad_lineage") == target_record.get("broad_lineage")
    ]
    return {"exact_sibling": exact, "broad_lineage": lineage}


def null_removal_subsets(
    all_peers: list[str],
    designated: list[str],
    *,
    seed_parts: tuple[Any, ...] = (),
    exact_limit: int = 5000,
    sampled_limit: int = 1000,
) -> tuple[list[tuple[str, ...]], str]:
    designated_set = set(designated)
    candidates = sorted(peer for peer in all_peers if peer not in designated_set)
    size = len(designated)
    if size <= 0 or size > len(candidates):
        raise ValueError("A same-size non-sibling null cannot be formed")
    possible = math.comb(len(candidates), size)
    if possible <= exact_limit:
        subsets = list(itertools.combinations(candidates, size))
        method = "exact_enumeration"
    else:
        rng = np.random.default_rng(stable_seed(20260828, "peer_removal_null", *seed_parts))
        target_count = min(sampled_limit, possible)
        chosen: set[tuple[str, ...]] = set()
        while len(chosen) < target_count:
            indices = rng.choice(len(candidates), size=size, replace=False)
            chosen.add(tuple(sorted(candidates[int(index)] for index in indices)))
        subsets = sorted(chosen)
        method = "deterministic_unique_sample"
    if any(designated_set.intersection(subset) for subset in subsets):
        raise AssertionError("A designated sibling entered the non-sibling null")
    return subsets, method


def _overall_error(
    aggregated: pd.DataFrame,
    target: str,
    peers: list[str],
    split: Any,
) -> float:
    result = primary_fit(aggregated, target, peers, split)
    return float(result.dose_results[result.dose_results["dose"].isna()]["convex_error"].iloc[0])


def run_peer_removal_analysis() -> dict[str, Any]:
    scores, _generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    records = {record["id"]: record for record in resolved["models"]}
    splits = fixed_splits(selected)
    corrected_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []

    for family in FAMILIES:
        aggregated = aggregate_family_responses(scores, family, "gold_probability")
        for split in splits.values():
            for target in model_ids:
                all_peers = [model for model in model_ids if model != target]
                baseline = _overall_error(aggregated, target, all_peers, split)
                removals = designated_removals(target, all_peers, records)
                single_records: list[dict[str, Any]] = []
                for removed_peer in all_peers:
                    retained = [peer for peer in all_peers if peer != removed_peer]
                    removed_error = _overall_error(aggregated, target, retained, split)
                    single_records.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "removed_peer": removed_peer,
                            "baseline_pier": baseline,
                            "removed_pier": removed_error,
                            "inflation": removed_error - baseline,
                            "is_exact_sibling": removed_peer in removals["exact_sibling"],
                            "is_broad_lineage": removed_peer in removals["broad_lineage"],
                        }
                    )
                single_frame = pd.DataFrame(single_records)
                single_frame["removal_rank"] = single_frame["inflation"].rank(
                    method="min", ascending=False
                ).astype(int)
                best_non_sibling = single_frame[~single_frame["is_exact_sibling"]][
                    "inflation"
                ].max()
                single_frame["best_non_sibling_inflation"] = float(best_non_sibling)
                single_frame["inflation_minus_best_non_sibling"] = np.where(
                    single_frame["is_exact_sibling"],
                    single_frame["inflation"] - best_non_sibling,
                    np.nan,
                )
                influence_rows.extend(single_frame.to_dict(orient="records"))

                for removal_type, designated in removals.items():
                    if not designated or len(all_peers) - len(designated) < 1:
                        continue
                    retained = [peer for peer in all_peers if peer not in designated]
                    observed_pier = _overall_error(aggregated, target, retained, split)
                    observed_inflation = observed_pier - baseline
                    subsets, null_method = null_removal_subsets(
                        all_peers,
                        designated,
                        seed_parts=(target, family, split.seed, removal_type),
                    )
                    null_inflations: list[float] = []
                    for subset in subsets:
                        null_retained = [peer for peer in all_peers if peer not in subset]
                        null_error = _overall_error(aggregated, target, null_retained, split)
                        null_inflations.append(null_error - baseline)
                    null_array = np.asarray(null_inflations, dtype=np.float64)
                    if null_method == "deterministic_unique_sample":
                        percentile = (1 + int(np.sum(null_array <= observed_inflation))) / (
                            1 + len(null_array)
                        )
                    else:
                        percentile = float(np.mean(null_array <= observed_inflation))
                    corrected_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "removal_type": removal_type,
                            "designated_removed_peers": designated,
                            "removed_peer_count": len(designated),
                            "baseline_pier": baseline,
                            "observed_removed_pier": observed_pier,
                            "observed_inflation": observed_inflation,
                            "null_method": null_method,
                            "null_subset_count": len(subsets),
                            "null_subsets": [list(subset) for subset in subsets],
                            "null_inflations": null_inflations,
                            "null_median_inflation": float(np.median(null_array)),
                            "null_lower_95": float(np.quantile(null_array, 0.025)),
                            "null_upper_95": float(np.quantile(null_array, 0.975)),
                            "observed_excess_over_null_median": observed_inflation
                            - float(np.median(null_array)),
                            "observed_percentile": percentile,
                            "observed_rank_among_null_plus_observed": 1
                            + int(np.sum(null_array < observed_inflation)),
                            "exceeds_every_null": bool(observed_inflation > np.max(null_array)),
                        }
                    )

    corrected_path = PHYSICAL_ROOT / "outputs/analysis/peer_removal_null_corrected.parquet"
    influence_path = PHYSICAL_ROOT / "outputs/analysis/single_peer_removal_influence.parquet"
    atomic_parquet(corrected_path, pd.DataFrame(corrected_rows))
    atomic_parquet(influence_path, pd.DataFrame(influence_rows))
    return {
        "outputs": [corrected_path, influence_path],
        "corrected_rows": len(corrected_rows),
        "influence_rows": len(influence_rows),
    }
