from __future__ import annotations

import numpy as np
import pandas as pd
from pier_llm.analysis import fit_clean_temperature

from pier_llm_reanalysis_v21.interface_bias import fit_label_bias, synthetic_label_bias_control
from pier_llm_reanalysis_v21.peer_removal import null_removal_subsets


def test_peer_removal_null_excludes_designated_sibling_and_enumerates_exactly() -> None:
    subsets, method = null_removal_subsets(
        ["sibling", "a", "b", "c"], ["sibling"]
    )
    assert method == "exact_enumeration"
    assert len(subsets) == 3
    assert all("sibling" not in subset for subset in subsets)


def test_additive_label_bias_synthetic_recovery_and_no_leakage() -> None:
    control = synthetic_label_bias_control(seed=91)
    assert control["bias_correlation"] >= 0.99
    assert control["after_mean_tv"] < control["before_mean_tv"]
    assert control["evaluation_ids_used_in_fitting"] is False
    assert control["passed"] is True


def test_unseen_evaluation_option_count_gets_explicit_zero_correction() -> None:
    rows = []
    for identifier, option_count in (("fit", 2), ("evaluation", 3)):
        for presentation in range(4):
            mapping = np.roll(np.arange(option_count), presentation % option_count)
            rows.append(
                {
                    "model_id": "synthetic/model",
                    "base_question_id": identifier,
                    "condition": "clean" if presentation == 0 else "option_permutation",
                    "family": "clean" if presentation == 0 else "option_permutation",
                    "permutation_index": np.nan if presentation == 0 else presentation - 1,
                    "candidate_log_likelihoods": np.arange(option_count, dtype=float),
                    "presented_to_original": mapping,
                }
            )
    fit = fit_label_bias(pd.DataFrame(rows), {"fit"})
    assert np.array_equal(fit.biases[3], np.zeros(3))
    diagnostics = fit.diagnostics.set_index("option_count")
    assert bool(diagnostics.loc[3, "estimable"]) is False


def test_temperature_uses_only_clean_fitting_questions() -> None:
    fitting_rows = pd.DataFrame(
        {
            "base_question_id": ["fit-1", "fit-2", "fit-3"],
            "condition": ["clean", "clean", "clean"],
            "candidate_log_likelihoods": [
                np.asarray([2.0, 0.0]),
                np.asarray([0.0, 2.0]),
                np.asarray([1.0, -1.0]),
            ],
            "answer_index": [0, 1, 0],
        }
    )
    baseline = fit_clean_temperature(fitting_rows, {"fit-1", "fit-2", "fit-3"})
    nuisance = pd.DataFrame(
        {
            "base_question_id": ["eval", "fit-1"],
            "condition": ["clean", "irrelevant_context"],
            "candidate_log_likelihoods": [
                np.asarray([-100.0, 100.0]),
                np.asarray([-100.0, 100.0]),
            ],
            "answer_index": [0, 0],
        }
    )
    augmented = fit_clean_temperature(
        pd.concat([fitting_rows, nuisance], ignore_index=True),
        {"fit-1", "fit-2", "fit-3"},
    )
    assert augmented == baseline
