from __future__ import annotations

import numpy as np
import pandas as pd

from pier_llm_reanalysis_v22.data import fit_clean_temperature
from pier_llm_reanalysis_v22.sensitivities import fit_label_bias


def test_temperature_uses_only_clean_fitting_questions() -> None:
    fitting = pd.DataFrame(
        {
            "base_question_id": ["f1", "f2"],
            "condition": ["clean", "clean"],
            "candidate_log_likelihoods": [np.array([2.0, 0.0]), np.array([0.0, 2.0])],
            "answer_index": [0, 1],
        }
    )
    baseline = fit_clean_temperature(fitting, {"f1", "f2"})
    nuisance = pd.DataFrame(
        {
            "base_question_id": ["eval", "f1"],
            "condition": ["clean", "irrelevant_context"],
            "candidate_log_likelihoods": [
                np.array([-100.0, 100.0]),
                np.array([-100.0, 100.0]),
            ],
            "answer_index": [0, 0],
        }
    )
    augmented = fit_clean_temperature(
        pd.concat([fitting, nuisance], ignore_index=True), {"f1", "f2"}
    )
    assert baseline == augmented


def test_unseen_option_count_receives_explicit_zero_bias() -> None:
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
    fit = fit_label_bias(pd.DataFrame(rows), {"fit"}, split_seed=1)
    assert np.array_equal(fit.biases[3], np.zeros(3))
    diagnostics = fit.diagnostics.set_index("option_count")
    assert bool(diagnostics.loc[3, "estimable"]) is False
