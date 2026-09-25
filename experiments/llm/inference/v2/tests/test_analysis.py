from __future__ import annotations

import numpy as np
import pandas as pd

from pier_llm.analysis import (
    aligned_response_arrays,
    clustered_bootstrap_with_refitting,
    design_transfer,
    fit_clean_temperature,
    fit_target_family,
    option_permutation_audit,
    stratified_question_split,
)


def _selected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"base_question_id": f"{category}-{index}", "category": category}
            for category in ("a", "b")
            for index in range(4)
        ]
    )


def test_split_has_no_question_leakage() -> None:
    selected = _selected()
    split = stratified_question_split(selected, 20260828)
    assert not split.fitting_ids.intersection(split.evaluation_ids)
    assert split.fitting_ids.union(split.evaluation_ids) == set(selected["base_question_id"])
    assert len(split.fitting_ids) == len(split.evaluation_ids) == 4


def test_cluster_bootstrap_refits_by_base_question() -> None:
    metadata = pd.DataFrame({"base_question_id": np.repeat(["a", "b", "c", "d"], 2)})
    design = np.column_stack([np.linspace(0, 1, 8), np.linspace(1, 0, 8)])
    target = design @ np.array([0.7, 0.3])
    result = clustered_bootstrap_with_refitting(
        metadata,
        design,
        target,
        metadata,
        design,
        target,
        replicates=7,
        seed=42,
    )
    assert len(result) == 7
    assert result["bootstrap_unit"].eq("base_question_id").all()
    assert result["metric"].max() < 1e-4


def test_honest_single_peer_is_selected_only_on_fit() -> None:
    selected = _selected()
    split = stratified_question_split(selected, 11)
    rows = []
    for base_id, category in selected.itertuples(index=False):
        is_fit = base_id in split.fitting_ids
        target = 0.2 if base_id.endswith(("0", "1")) else 0.8
        values = {
            "target": target,
            "fit-winner": target if is_fit else 1 - target,
            "eval-winner": target + (0.05 if is_fit else 0.0),
        }
        for model_id, value in values.items():
            for dose in (0.0, 1.0):
                rows.append(
                    {
                        "model_id": model_id,
                        "base_question_id": base_id,
                        "category": category,
                        "dose": dose,
                        "response": float(value),
                    }
                )
    result_rows, _weights, _projection = fit_target_family(
        pd.DataFrame(rows),
        "target",
        ["fit-winner", "eval-winner"],
        split,
        family="irrelevant_context",
        representation="gold_probability",
    )
    overall = next(row for row in result_rows if np.isnan(row["dose"]))
    assert overall["fit_selected_single_peer"] == "fit-winner"
    assert overall["fit_selected_single_error"] > 0


def test_temperature_calibration_and_option_remapping() -> None:
    rows = []
    for index in range(6):
        rows.append(
            {
                "model_id": "m",
                "base_question_id": f"q{index}",
                "condition": "clean",
                "candidate_log_likelihoods": [2.0, 0.0, -1.0],
                "answer_index": 0,
                "semantic_answer_index": 0,
                "presented_to_original": [0, 1, 2],
                "semantic_probabilities": [0.84, 0.11, 0.05],
            }
        )
    temperature = fit_clean_temperature(pd.DataFrame(rows), {f"q{i}" for i in range(3)})
    assert 0.05 <= temperature <= 20

    clean = pd.DataFrame(rows[:1])
    permuted = clean.copy()
    permuted["condition"] = "option_permutation"
    permuted["family"] = "option_permutation"
    permuted["permutation_index"] = 0
    permuted["semantic_probabilities"] = permuted["semantic_probabilities"].apply(list)
    audit = option_permutation_audit(pd.concat([clean, permuted], ignore_index=True))
    assert audit["probability_vector_tv"].iloc[0] == 0
    assert bool(audit["top1_semantic_agreement"].iloc[0])


def test_cross_design_transfer_runs_on_fit_only() -> None:
    selected = _selected()
    split = stratified_question_split(selected, 7)
    rows = []
    for model_index, model in enumerate(("t", "p1", "p2")):
        for base_id, category in selected.itertuples(index=False):
            for dose in (0.0, 1.0):
                rows.append(
                    {
                        "model_id": model,
                        "base_question_id": base_id,
                        "category": category,
                        "dose": dose,
                        "response": 0.1 * model_index + 0.01 * dose,
                    }
                )
    frame = pd.DataFrame(rows)
    result = design_transfer(frame, frame, "t", ["p1", "p2"], split)
    assert np.isfinite(result["transfer_penalty"])


def test_aligned_response_arrays_zero_pads_ragged_option_vectors() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_id": model,
                "base_question_id": question,
                "category": "a",
                "dose": 0.0,
                "response": response,
            }
            for model, first, second in (
                ("target", [0.75, 0.25], [0.2, 0.3, 0.5]),
                ("peer", [0.6, 0.4], [0.1, 0.2, 0.7]),
            )
            for question, response in (("q1", first), ("q2", second))
        ]
    )

    _metadata, target, design = aligned_response_arrays(frame, "target", ["peer"])

    assert target.shape == (2, 3)
    assert design.shape == (2, 3, 1)
    np.testing.assert_array_equal(target[0], [0.75, 0.25, 0.0])
    np.testing.assert_array_equal(design[0, :, 0], [0.6, 0.4, 0.0])


def test_aligned_response_arrays_rejects_cross_model_vector_length_mismatch() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_id": "target",
                "base_question_id": "q1",
                "category": "a",
                "dose": 0.0,
                "response": [0.75, 0.25],
            },
            {
                "model_id": "peer",
                "base_question_id": "q1",
                "category": "a",
                "dose": 0.0,
                "response": [0.5, 0.3, 0.2],
            },
        ]
    )

    with np.testing.assert_raises_regex(ValueError, "response-vector lengths differ"):
        aligned_response_arrays(frame, "target", ["peer"])
