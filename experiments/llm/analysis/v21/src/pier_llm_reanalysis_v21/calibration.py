from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pier_llm.analysis import aggregate_family_responses, fit_clean_temperature

from .core import FAMILIES, fixed_splits, high_dose, primary_fit
from .input_validation import load_validated_inputs
from .utils import PHYSICAL_ROOT, atomic_parquet


def run_calibration_analysis() -> dict[str, Any]:
    scores, _generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    splits = fixed_splits(selected)
    raw = pd.read_parquet(PHYSICAL_ROOT / "outputs/analysis/primary_dose_results.parquet")
    rows: list[dict[str, Any]] = []

    for split in splits.values():
        temperatures = {
            model: fit_clean_temperature(
                scores[scores["model_id"].eq(model)], split.fitting_ids
            )
            for model in model_ids
        }
        for family in FAMILIES:
            calibrated = aggregate_family_responses(
                scores, family, "gold_probability", temperatures
            )
            for target in model_ids:
                peers = [model for model in model_ids if model != target]
                calibrated_fit = primary_fit(calibrated, target, peers, split)
                calibrated_frame = calibrated_fit.dose_results[
                    calibrated_fit.dose_results["dose"].notna()
                ].copy()
                raw_frame = raw[
                    raw["target"].eq(target)
                    & raw["family"].eq(family)
                    & raw["split_seed"].eq(split.seed)
                    & raw["dose"].notna()
                ].copy()
                raw_clean = float(raw_frame[raw_frame["dose"].eq(0.0)]["convex_error"].iloc[0])
                raw_high = float(
                    raw_frame[raw_frame["dose"].eq(high_dose(family))]["convex_error"].iloc[0]
                )
                calibrated_clean = float(
                    calibrated_frame[calibrated_frame["dose"].eq(0.0)]["convex_error"].iloc[0]
                )
                calibrated_high = float(
                    calibrated_frame[
                        calibrated_frame["dose"].eq(high_dose(family))
                    ]["convex_error"].iloc[0]
                )
                for dose in sorted(raw_frame["dose"].astype(float).unique()):
                    raw_row = raw_frame[raw_frame["dose"].eq(dose)].iloc[0]
                    calibrated_row = calibrated_frame[
                        calibrated_frame["dose"].eq(dose)
                    ].iloc[0]
                    raw_improvement = float(raw_row["single_error"] - raw_row["convex_error"])
                    calibrated_improvement = float(
                        calibrated_row["single_error"] - calibrated_row["convex_error"]
                    )
                    raw_delta = raw_high - raw_clean
                    calibrated_delta = calibrated_high - calibrated_clean
                    rows.append(
                        {
                            "target": target,
                            "family": family,
                            "split_seed": split.seed,
                            "dose": float(dose),
                            "raw_pier": float(raw_row["convex_error"]),
                            "calibrated_pier": float(calibrated_row["convex_error"]),
                            "raw_endpoint_delta": raw_delta,
                            "calibrated_endpoint_delta": calibrated_delta,
                            "endpoint_effect_sign_agreement": bool(
                                np.sign(raw_delta) == np.sign(calibrated_delta)
                            ),
                            "raw_honest_convexity_improvement": raw_improvement,
                            "calibrated_honest_convexity_improvement": calibrated_improvement,
                            "target_temperature": temperatures[target],
                            "peer_ids": peers,
                            "peer_temperatures": [temperatures[peer] for peer in peers],
                            "raw_fit_selected_single_peer": str(
                                raw_row["fit_selected_single_peer"]
                            ),
                            "calibrated_fit_selected_single_peer": str(
                                calibrated_row["fit_selected_single_peer"]
                            ),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame["raw_rank"] = frame.groupby(["family", "split_seed", "dose"])[
        "raw_pier"
    ].rank(method="average")
    frame["calibrated_rank"] = frame.groupby(["family", "split_seed", "dose"])[
        "calibrated_pier"
    ].rank(method="average")
    frame["rank_shift"] = frame["calibrated_rank"] - frame["raw_rank"]
    if len(frame) != 800:
        raise ValueError(f"Expected 800 calibrated trajectory rows, found {len(frame)}")
    path = PHYSICAL_ROOT / "outputs/analysis/calibrated_dose_trajectories.parquet"
    atomic_parquet(path, frame)
    return {"outputs": [path], "rows": len(frame)}
