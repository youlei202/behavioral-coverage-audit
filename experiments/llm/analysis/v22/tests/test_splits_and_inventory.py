from __future__ import annotations

import pandas as pd

from pier_llm_reanalysis_v22.data import (
    compare_inventories,
    fixed_splits,
    stratified_question_split,
)


def _selected_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"base_question_id": f"{category}-{index}", "category": category}
            for category in ("a", "b")
            for index in range(4)
        ]
    )


def test_stratified_split_keeps_question_clusters_disjoint() -> None:
    split = stratified_question_split(_selected_fixture(), 20260828)
    assert len(split.fitting_ids) == 4
    assert len(split.evaluation_ids) == 4
    assert split.fitting_ids.isdisjoint(split.evaluation_ids)


def test_inventory_comparison_detects_metadata_or_hash_change() -> None:
    before = [
        {
            "source": "v2",
            "relative_path": "x",
            "size": 1,
            "mtime_ns": 2,
            "sha256": "a",
        }
    ]
    assert compare_inventories(before, [dict(before[0])]) == []
    changed = [dict(before[0], sha256="b")]
    assert compare_inventories(before, changed)
