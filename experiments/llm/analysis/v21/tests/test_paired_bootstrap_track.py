from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from pier_llm.analysis import stratified_question_split

from pier_llm_reanalysis_v21.core import (
    best_single_for_draw,
    row_indices_for_draw,
    stratified_cluster_draw,
)
from pier_llm_reanalysis_v21.paired_effects import paired_endpoint_effect
from pier_llm_reanalysis_v21.track_analysis import cancellation_metrics


def test_paired_endpoint_uses_within_split_contrasts() -> None:
    frame = pd.DataFrame(
        {
            "split_seed": [1, 1, 2, 2],
            "dose": [0.0, 1.0, 0.0, 1.0],
            "pier": [0.0, 1.0, 100.0, 101.0],
        }
    )
    result = paired_endpoint_effect(frame, high_dose=1.0)
    assert result["delta"].tolist() == [1.0, 1.0]
    assert frame["pier"].max() - frame["pier"].min() == 101.0


def _selected_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"base_question_id": f"{category}-{index}", "category": category}
            for category in ("a", "b")
            for index in range(4)
        ]
    )


def test_stratified_cluster_draw_preserves_categories_and_clusters() -> None:
    selected = _selected_fixture()
    split = stratified_question_split(selected, 11)
    draw = stratified_cluster_draw(selected, split.fitting_ids, seed=22)
    source = selected[selected["base_question_id"].isin(split.fitting_ids)].set_index(
        "base_question_id"
    )
    assert Counter(source.loc[draw, "category"]) == Counter({"a": 2, "b": 2})
    metadata = pd.DataFrame(
        [
            {"base_question_id": identifier, "dose": dose}
            for identifier in sorted(split.fitting_ids)
            for dose in (0.0, 1.0)
        ]
    )
    rows = row_indices_for_draw(metadata, draw)
    drawn_rows = metadata.iloc[rows]
    for identifier, multiplicity in Counter(draw).items():
        assert int((drawn_rows["base_question_id"] == identifier).sum()) == 2 * multiplicity
    assert set(draw).isdisjoint(split.evaluation_ids)


def test_paired_conditions_use_identical_sample_multiplicities() -> None:
    metadata = pd.DataFrame(
        [
            {"base_question_id": identifier, "dose": dose}
            for identifier in ("q1", "q2")
            for dose in (0.0, 1.0)
        ]
    )
    draw = ["q1", "q1", "q2"]
    clean = metadata.iloc[row_indices_for_draw(metadata, draw, dose=0.0)]
    high = metadata.iloc[row_indices_for_draw(metadata, draw, dose=1.0)]
    assert Counter(clean["base_question_id"]) == Counter(high["base_question_id"])


def test_track_jensen_inequality_and_cancellation_example() -> None:
    metrics = cancellation_metrics(np.asarray([0.5, 0.5, 0.5]), 0.0)
    assert metrics["mean_of_trackwise_pier"] >= metrics["pier_of_mean_response"] - 1e-12
    assert metrics["track_cancellation_gap"] == 0.5


def test_honest_single_peer_is_selected_from_fitting_rows_only() -> None:
    metadata = pd.DataFrame(
        {
            "base_question_id": ["fit-1", "fit-2", "eval-1", "eval-2"],
            "dose": [0.0, 0.0, 0.0, 0.0],
        }
    )
    target = np.asarray([0.0, 1.0, 0.0, 1.0])
    design = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    selected = best_single_for_draw(metadata, design, target, ["fit-1", "fit-2"])
    assert selected == 0
    assert np.mean(np.abs(target[2:] - design[2:, selected])) > 0
