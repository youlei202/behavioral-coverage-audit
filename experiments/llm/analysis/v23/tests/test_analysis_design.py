from __future__ import annotations

import numpy as np
import pandas as pd

from pier_llm_v23.analysis import _evaluate_endpoint
from pier_llm_v23.core import endpoint_row_weights, fit_mse_simplex


def test_endpoint_design_has_equal_clean_high_mass() -> None:
    metadata = pd.DataFrame(
        {
            "base_question_id": ["q0"] * 4 + ["q1"] * 4,
            "track": [-1, 0, 1, 2] * 2,
        }
    )
    weights = endpoint_row_weights(metadata)
    assert np.isclose(weights[metadata["track"].eq(-1)].sum(), 1.0)
    assert np.isclose(weights[metadata["track"].ne(-1)].sum(), 1.0)


def test_exact_clone_endpoint_pier_is_zero() -> None:
    metadata = pd.DataFrame(
        {
            "base_question_id": [f"q{index // 4}" for index in range(40)],
            "track": [-1, 0, 1, 2] * 10,
        }
    )
    rng = np.random.default_rng(9)
    design = rng.uniform(size=(40, 3))
    target = design[:, 1]
    weights = fit_mse_simplex(design, target, endpoint_row_weights(metadata))
    metrics = _evaluate_endpoint(
        metadata,
        target,
        design,
        weights,
        {f"q{index}" for index in range(10)},
    )
    assert metrics["endpoint_design_pier"] < 1e-9
