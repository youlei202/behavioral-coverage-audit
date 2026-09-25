from __future__ import annotations

import pandas as pd

from pier_llm_reanalysis_v21.input_validation import compare_inventories
from pier_llm_reanalysis_v21.tables import select_max_after_split_aggregation
from pier_llm_reanalysis_v21.utils import file_record


def test_maximum_condition_is_selected_after_split_aggregation() -> None:
    frame = pd.DataFrame(
        {
            "target": ["m"] * 4,
            "family": ["f"] * 4,
            "split_seed": [1, 2, 1, 2],
            "dose": [0.0, 0.0, 1.0, 1.0],
            "pier": [100.0, 0.0, 60.0, 60.0],
        }
    )
    selected = select_max_after_split_aggregation(
        frame,
        group_columns=["target", "family"],
        condition_column="dose",
        value_column="pier",
    )
    assert selected.iloc[0]["dose"] == 1.0
    assert selected.iloc[0]["pier"] == 60.0


def test_before_after_hash_inventory_detects_mutation(tmp_path) -> None:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("locked input\n", encoding="utf-8")
    before = [file_record(fixture, base=tmp_path)]
    unchanged = [file_record(fixture, base=tmp_path)]
    assert compare_inventories(before, unchanged) == []
    fixture.write_text("changed input\n", encoding="utf-8")
    after = [file_record(fixture, base=tmp_path)]
    assert compare_inventories(before, after)
