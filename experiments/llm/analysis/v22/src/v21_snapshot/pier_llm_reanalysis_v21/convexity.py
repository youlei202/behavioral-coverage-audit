from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import PHYSICAL_ROOT, atomic_parquet


def run_convexity_analysis() -> dict[str, Any]:
    primary = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/primary_dose_results.parquet")
    overall = primary[primary["dose"].isna()].copy()
    columns = [
        "target",
        "family",
        "split_seed",
        "fit_selected_single_peer",
        "single_error",
        "convex_error",
        "absolute_improvement",
        "relative_improvement",
        "gap_ratio",
    ]
    corrected = overall[columns].sort_values(["target", "family", "split_seed"])
    if len(corrected) != 160:
        raise ValueError(f"Expected 160 corrected convexity rows, found {len(corrected)}")
    path = PHYSICAL_ROOT / "outputs/analysis/convex_vs_single_corrected.parquet"
    atomic_parquet(path, corrected)
    return {"outputs": [path], "rows": len(corrected)}
