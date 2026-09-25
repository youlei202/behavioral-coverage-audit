from __future__ import annotations

from typing import Any

import pandas as pd

from .core import FAMILIES, fixed_splits, paired_endpoint_rows, primary_fit
from .input_validation import load_validated_inputs
from .utils import PHYSICAL_ROOT, atomic_parquet


def paired_endpoint_effect(
    frame: pd.DataFrame,
    *,
    split_column: str = "split_seed",
    dose_column: str = "dose",
    value_column: str = "pier",
    clean_dose: float = 0.0,
    high_dose: float,
) -> pd.DataFrame:
    """Return paired high-minus-clean effects without cross-split extrema."""
    clean = frame[frame[dose_column].eq(clean_dose)][[split_column, value_column]].rename(
        columns={value_column: "clean"}
    )
    high = frame[frame[dose_column].eq(high_dose)][[split_column, value_column]].rename(
        columns={value_column: "high"}
    )
    paired = clean.merge(high, on=split_column, validate="one_to_one")
    if len(paired) != frame[split_column].nunique():
        raise ValueError("Not every split has a clean/high endpoint pair")
    paired["delta"] = paired["high"] - paired["clean"]
    return paired.sort_values(split_column).reset_index(drop=True)


def run_paired_endpoint_analysis() -> dict[str, Any]:
    scores, _generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    dose_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        from pier_llm.analysis import aggregate_family_responses

        aggregated = aggregate_family_responses(scores, family, "gold_probability")
        for split in splits.values():
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                result = primary_fit(aggregated, target, peers, split)
                frame = result.dose_results.copy()
                frame["family"] = family
                dose_frames.append(frame)
                diagnostics = result.projection.diagnostics()
                for peer, weight in zip(peers, result.weights, strict=True):
                    weight_rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "peer": peer,
                            "weight": float(weight),
                            **diagnostics,
                        }
                    )
    primary = pd.concat(dose_frames, ignore_index=True)
    endpoints = paired_endpoint_rows(primary)
    dose_path = PHYSICAL_ROOT / "outputs/analysis/primary_dose_results.parquet"
    weight_path = PHYSICAL_ROOT / "outputs/analysis/primary_weights.parquet"
    endpoint_path = PHYSICAL_ROOT / "outputs/analysis/paired_endpoint_effects.parquet"
    atomic_parquet(dose_path, primary)
    atomic_parquet(weight_path, pd.DataFrame(weight_rows))
    atomic_parquet(endpoint_path, endpoints)
    return {
        "outputs": [dose_path, weight_path, endpoint_path],
        "primary_rows": len(primary),
        "endpoint_rows": len(endpoints),
    }
