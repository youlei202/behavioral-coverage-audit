from __future__ import annotations

import numpy as np
import pandas as pd

from pier_llm_v23.core import (
    common_bootstrap_multiplicities,
    cyclic_mapping,
    extract_final_label,
    fit_mae_simplex,
    fit_mse_simplex,
    remap_semantic_to_visible,
    remap_visible_to_semantic,
    run_synthetic_controls,
    validate_permutation_balance,
    weighted_mae,
)


def test_all_cyclic_sets_are_perfectly_balanced() -> None:
    for option_count in range(2, 11):
        validate_permutation_balance(option_count)


def test_round_trip_every_rotation() -> None:
    semantic = np.asarray([0.05, 0.15, 0.3, 0.5])
    for rotation in range(4):
        semantic_to_visible, visible_to_semantic = cyclic_mapping(4, rotation)
        visible = remap_semantic_to_visible(semantic, semantic_to_visible)
        recovered = remap_visible_to_semantic(visible, visible_to_semantic)
        np.testing.assert_array_equal(recovered, semantic)


def test_final_answer_parser_priorities() -> None:
    assert extract_final_label("reasoning\nFinal answer: (C)", list("ABCD")) == (
        "C",
        "explicit_final_answer",
    )
    assert extract_final_label("reasoning\n(D)", list("ABCD")) == (
        "D",
        "final_standalone_label",
    )
    assert extract_final_label("no usable answer", list("ABCD")) == (None, "unresolved")


def test_mse_and_mae_known_convex_mixture() -> None:
    rng = np.random.default_rng(4)
    design = rng.normal(size=(100, 4))
    truth = np.asarray([0.1, 0.2, 0.3, 0.4])
    target = design @ truth
    mass = np.ones(100)
    mse = fit_mse_simplex(design, target, mass)
    mae = fit_mae_simplex(design, target, mass)
    assert weighted_mae(target, design @ mse, mass) < 1e-9
    assert weighted_mae(target, design @ mae, mass) < 1e-9


def test_common_bootstrap_is_global_and_category_stratified() -> None:
    selected = pd.DataFrame(
        {
            "base_question_id": [f"q{index}" for index in range(12)],
            "category": ["a"] * 6 + ["b"] * 6,
        }
    )
    first = common_bootstrap_multiplicities(selected, 7, 20260828)
    second = common_bootstrap_multiplicities(selected, 7, 20260828)
    assert first == second
    assert sum(first[f"q{index}"] for index in range(6)) == 6
    assert sum(first[f"q{index}"] for index in range(6, 12)) == 6
    split_a = {key: first[key] for key in ("q0", "q1", "q6", "q7")}
    split_b = {key: first[key] for key in ("q1", "q2", "q7", "q8")}
    assert split_a["q1"] == split_b["q1"]
    assert split_a["q7"] == split_b["q7"]


def test_all_declared_synthetic_controls() -> None:
    result = run_synthetic_controls()
    assert result["passed"], result

