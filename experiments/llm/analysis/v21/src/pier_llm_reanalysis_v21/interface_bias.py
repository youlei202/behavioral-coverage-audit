from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pier_llm.analysis import fit_clean_temperature
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .core import FAMILIES, fixed_splits, high_dose, primary_fit
from .input_validation import load_validated_inputs
from .peer_removal import designated_removals
from .utils import PHYSICAL_ROOT, atomic_parquet, atomic_write_json


@dataclass(frozen=True)
class LabelBiasFit:
    model_id: str
    split_seed: int
    biases: dict[int, np.ndarray]
    diagnostics: pd.DataFrame
    fitting_question_ids: frozenset[str]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def _permutation_design(scores: pd.DataFrame, fitting_ids: Iterable[str]) -> pd.DataFrame:
    fitting = {str(value) for value in fitting_ids}
    subset = scores[
        scores["base_question_id"].astype(str).isin(fitting)
        & (scores["condition"].eq("clean") | scores["family"].eq("option_permutation"))
    ].copy()
    if subset.empty:
        raise ValueError("No clean/permutation rows for label-bias fitting")
    subset["option_count"] = subset["candidate_log_likelihoods"].map(len)
    presentation_counts = subset.groupby("base_question_id").size()
    if not presentation_counts.eq(4).all():
        raise ValueError("Each label-bias fitting question must have four presentations")
    return subset


def _fit_option_count(group: pd.DataFrame, option_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    questions = sorted(group["base_question_id"].astype(str).unique())
    question_index = {identifier: index for index, identifier in enumerate(questions)}
    semantic_parameter_count = len(questions) * (option_count - 1)
    bias_offset = semantic_parameter_count
    parameter_count = semantic_parameter_count + option_count - 1
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    observations: list[float] = []
    observation_index = 0

    for _, row in group.sort_values(["base_question_id", "condition", "permutation_index"]).iterrows():
        likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
        if likelihoods.size != option_count:
            raise ValueError("Inconsistent option count in sparse label-bias fit")
        centered = likelihoods - float(np.mean(likelihoods))
        mapping = [int(value) for value in row["presented_to_original"]]
        question_offset = question_index[str(row["base_question_id"])] * (option_count - 1)
        for presented_position, semantic_position in enumerate(mapping):
            if semantic_position < option_count - 1:
                row_indices.append(observation_index)
                column_indices.append(question_offset + semantic_position)
                values.append(1.0)
            else:
                for index in range(option_count - 1):
                    row_indices.append(observation_index)
                    column_indices.append(question_offset + index)
                    values.append(-1.0)
            if presented_position < option_count - 1:
                row_indices.append(observation_index)
                column_indices.append(bias_offset + presented_position)
                values.append(1.0)
            else:
                for index in range(option_count - 1):
                    row_indices.append(observation_index)
                    column_indices.append(bias_offset + index)
                    values.append(-1.0)
            observations.append(float(centered[presented_position]))
            observation_index += 1

    design = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(observations), parameter_count),
        dtype=np.float64,
    ).tocsr()
    solution = lsqr(
        design,
        np.asarray(observations, dtype=np.float64),
        atol=1e-12,
        btol=1e-12,
        iter_lim=1000,
    )
    coefficients = np.asarray(solution[0], dtype=np.float64)
    bias = np.empty(option_count, dtype=np.float64)
    bias[:-1] = coefficients[bias_offset:]
    bias[-1] = -float(np.sum(bias[:-1]))
    bias -= float(np.mean(bias))
    diagnostics = {
        "option_count": option_count,
        "fitting_question_count": len(questions),
        "observation_count": len(observations),
        "parameter_count": parameter_count,
        "lsqr_stop_code": int(solution[1]),
        "lsqr_iterations": int(solution[2]),
        "residual_norm": float(solution[3]),
        "normal_equation_residual_norm": float(solution[4]),
        "condition_estimate": float(solution[6]),
        "converged": bool(int(solution[1]) in {1, 2}),
        "estimable": True,
        "sum_to_zero_error": abs(float(np.sum(bias))),
    }
    if diagnostics["lsqr_iterations"] > 1000 or diagnostics["sum_to_zero_error"] > 1e-10:
        raise RuntimeError("Sparse additive label-bias fit failed its constraints")
    return bias, diagnostics


def fit_label_bias(
    scores: pd.DataFrame,
    fitting_ids: Iterable[str],
    *,
    split_seed: int = -1,
) -> LabelBiasFit:
    if scores["model_id"].nunique() != 1:
        raise ValueError("fit_label_bias expects exactly one model")
    model_id = str(scores["model_id"].iloc[0])
    fitting_set = frozenset(str(value) for value in fitting_ids)
    subset = _permutation_design(scores, fitting_set)
    biases: dict[int, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    for option_count, group in subset.groupby("option_count", sort=True):
        bias, record = _fit_option_count(group, int(option_count))
        biases[int(option_count)] = bias
        diagnostics.append(
            {
                "model_id": model_id,
                "split_seed": split_seed,
                **record,
            }
        )
    all_option_counts = sorted(
        {
            len(value)
            for value in scores.loc[
                scores["condition"].eq("clean")
                | scores["family"].eq("option_permutation"),
                "candidate_log_likelihoods",
            ]
        }
    )
    for option_count in all_option_counts:
        if option_count in biases:
            continue
        biases[option_count] = np.zeros(option_count, dtype=np.float64)
        diagnostics.append(
            {
                "model_id": model_id,
                "split_seed": split_seed,
                "option_count": option_count,
                "fitting_question_count": 0,
                "observation_count": 0,
                "parameter_count": 0,
                "lsqr_stop_code": 0,
                "lsqr_iterations": 0,
                "residual_norm": math.nan,
                "normal_equation_residual_norm": math.nan,
                "condition_estimate": math.nan,
                "converged": False,
                "estimable": False,
                "sum_to_zero_error": 0.0,
            }
        )
    return LabelBiasFit(
        model_id=model_id,
        split_seed=split_seed,
        biases=biases,
        diagnostics=pd.DataFrame(diagnostics),
        fitting_question_ids=fitting_set,
    )


def corrected_semantic_probabilities(row: pd.Series, biases: dict[int, np.ndarray]) -> np.ndarray:
    likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
    bias = biases.get(len(likelihoods))
    if bias is None:
        raise KeyError(f"No label-bias estimate for {len(likelihoods)} options")
    presented = _softmax(likelihoods - bias)
    semantic = np.empty_like(presented)
    for presented_index, original_index in enumerate(row["presented_to_original"]):
        semantic[int(original_index)] = presented[presented_index]
    return semantic


def _semantic_probabilities(
    row: pd.Series,
    *,
    biases: dict[int, np.ndarray] | None,
    temperature: float,
) -> np.ndarray:
    likelihoods = np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
    if biases is not None:
        likelihoods = likelihoods - biases[len(likelihoods)]
    presented = _softmax(likelihoods / temperature)
    semantic = np.empty_like(presented)
    for presented_index, original_index in enumerate(row["presented_to_original"]):
        semantic[int(original_index)] = presented[presented_index]
    return semantic


def fit_temperature_after_bias(
    scores: pd.DataFrame,
    fitting_ids: Iterable[str],
    biases: dict[int, np.ndarray],
) -> float:
    fitting = {str(value) for value in fitting_ids}
    subset = scores[
        scores["base_question_id"].astype(str).isin(fitting)
        & scores["condition"].eq("clean")
    ]
    logits = [
        np.asarray(row["candidate_log_likelihoods"], dtype=np.float64)
        - biases[len(row["candidate_log_likelihoods"])]
        for _, row in subset.iterrows()
    ]
    gold = subset["answer_index"].astype(int).tolist()

    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        losses = []
        for values, answer_index in zip(logits, gold, strict=True):
            probability = _softmax(values / temperature)[answer_index]
            losses.append(-math.log(max(float(probability), 1e-300)))
        return float(np.mean(losses))

    result = minimize_scalar(
        nll,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Joint label-bias/temperature fit failed: {result.message}")
    return float(math.exp(float(result.x)))


def aggregate_interface_responses(
    scores: pd.DataFrame,
    family: str,
    *,
    fits: dict[str, LabelBiasFit] | None = None,
    temperatures: dict[str, float] | None = None,
) -> pd.DataFrame:
    subset = scores[scores["condition"].eq("clean") | scores["family"].eq(family)].copy()
    temperatures = temperatures or {}
    responses: list[float] = []
    for _, row in subset.iterrows():
        model = str(row["model_id"])
        biases = fits[model].biases if fits is not None else None
        probabilities = _semantic_probabilities(
            row,
            biases=biases,
            temperature=float(temperatures.get(model, 1.0)),
        )
        responses.append(float(probabilities[int(row["semantic_answer_index"])]))
    subset["response"] = responses
    keys = ["model_id", "base_question_id", "category", "dose"]
    aggregated = subset.groupby(keys, as_index=False, sort=True).agg(
        response=("response", "mean"),
        track_count=("response", "size"),
    )
    nonzero = aggregated[~aggregated["dose"].astype(float).eq(0.0)]
    clean = aggregated[aggregated["dose"].astype(float).eq(0.0)]
    if not nonzero["track_count"].eq(3).all() or not clean["track_count"].eq(1).all():
        raise ValueError("Interface aggregation violated fixed track counts")
    return aggregated.drop(columns="track_count")


def heldout_label_bias_validation(
    model_scores: pd.DataFrame,
    evaluation_ids: Iterable[str],
    fit: LabelBiasFit,
) -> dict[str, Any]:
    evaluation = {str(value) for value in evaluation_ids}
    if fit.fitting_question_ids.intersection(evaluation):
        raise ValueError("Evaluation IDs were used in label-bias fitting")
    subset = model_scores[model_scores["base_question_id"].astype(str).isin(evaluation)]
    canonical = subset[subset["condition"].eq("clean")].set_index("base_question_id")
    permuted = subset[subset["family"].eq("option_permutation")]
    before_tv: list[float] = []
    after_tv: list[float] = []
    before_top: list[bool] = []
    after_top: list[bool] = []
    before_gold: list[float] = []
    after_gold: list[float] = []
    for _, row in permuted.iterrows():
        baseline = canonical.loc[str(row["base_question_id"])]
        raw_baseline = _semantic_probabilities(baseline, biases=None, temperature=1.0)
        raw_permuted = _semantic_probabilities(row, biases=None, temperature=1.0)
        corrected_baseline = corrected_semantic_probabilities(baseline, fit.biases)
        corrected_permuted = corrected_semantic_probabilities(row, fit.biases)
        gold = int(baseline["semantic_answer_index"])
        before_tv.append(float(0.5 * np.abs(raw_baseline - raw_permuted).sum()))
        after_tv.append(float(0.5 * np.abs(corrected_baseline - corrected_permuted).sum()))
        before_top.append(int(np.argmax(raw_baseline)) == int(np.argmax(raw_permuted)))
        after_top.append(
            int(np.argmax(corrected_baseline)) == int(np.argmax(corrected_permuted))
        )
        before_gold.append(abs(float(raw_baseline[gold] - raw_permuted[gold])))
        after_gold.append(abs(float(corrected_baseline[gold] - corrected_permuted[gold])))
    before_array = np.asarray(before_tv)
    after_array = np.asarray(after_tv)
    return {
        "model_id": fit.model_id,
        "split_seed": fit.split_seed,
        "evaluation_question_count": len(evaluation),
        "permutation_comparison_count": len(before_tv),
        "before_mean_tv": float(np.mean(before_array)),
        "after_mean_tv": float(np.mean(after_array)),
        "before_median_tv": float(np.median(before_array)),
        "after_median_tv": float(np.median(after_array)),
        "before_95th_percentile_tv": float(np.quantile(before_array, 0.95)),
        "after_95th_percentile_tv": float(np.quantile(after_array, 0.95)),
        "before_semantic_top1_agreement": float(np.mean(before_top)),
        "after_semantic_top1_agreement": float(np.mean(after_top)),
        "before_gold_probability_absolute_change": float(np.mean(before_gold)),
        "after_gold_probability_absolute_change": float(np.mean(after_gold)),
        "mean_tv_reduction": float(np.mean(before_array) - np.mean(after_array)),
        "heldout_invariance_improved": bool(np.mean(after_array) < np.mean(before_array)),
    }


def synthetic_label_bias_control(seed: int = 20260828) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    option_count = 5
    true_bias = np.asarray([-0.45, -0.1, 0.05, 0.2, 0.3], dtype=np.float64)
    true_bias -= true_bias.mean()
    rows: list[dict[str, Any]] = []
    permutations = [
        np.arange(option_count),
        np.asarray([1, 2, 3, 4, 0]),
        np.asarray([2, 4, 1, 0, 3]),
        np.asarray([4, 3, 0, 2, 1]),
    ]
    utilities: dict[str, np.ndarray] = {}
    for question_index in range(120):
        identifier = f"synthetic-{question_index:03d}"
        utility = rng.normal(0.0, 0.8, size=option_count)
        utility -= utility.mean()
        utilities[identifier] = utility
        for presentation_index, mapping in enumerate(permutations):
            presentation_intercept = rng.normal(0.0, 0.5)
            likelihoods = np.asarray(
                [
                    utility[int(semantic)]
                    + true_bias[presented]
                    + presentation_intercept
                    + rng.normal(0.0, 1e-5)
                    for presented, semantic in enumerate(mapping)
                ]
            )
            rows.append(
                {
                    "model_id": "synthetic/model",
                    "base_question_id": identifier,
                    "condition": "clean" if presentation_index == 0 else "option_permutation",
                    "family": "clean" if presentation_index == 0 else "option_permutation",
                    "permutation_index": np.nan if presentation_index == 0 else presentation_index - 1,
                    "candidate_log_likelihoods": likelihoods,
                    "presented_to_original": mapping,
                    "semantic_answer_index": 0,
                    "answer_index": int(np.where(mapping == 0)[0][0]),
                }
            )
    frame = pd.DataFrame(rows)
    fitting = {f"synthetic-{index:03d}" for index in range(80)}
    evaluation = {f"synthetic-{index:03d}" for index in range(80, 120)}
    fit = fit_label_bias(frame, fitting, split_seed=seed)
    recovered = fit.biases[option_count]
    correlation = float(np.corrcoef(true_bias, recovered)[0, 1])
    validation = heldout_label_bias_validation(frame, evaluation, fit)
    passed = bool(
        correlation >= 0.99
        and validation["after_mean_tv"] < validation["before_mean_tv"]
        and not fit.fitting_question_ids.intersection(evaluation)
    )
    return {
        "schema_version": "pier_label_bias_synthetic_v21_v1",
        "seed": seed,
        "true_centered_bias": true_bias.tolist(),
        "recovered_centered_bias": recovered.tolist(),
        "bias_correlation": correlation,
        "before_mean_tv": validation["before_mean_tv"],
        "after_mean_tv": validation["after_mean_tv"],
        "evaluation_ids_used_in_fitting": False,
        "passed": passed,
    }


def _fit_rows(fits: dict[str, LabelBiasFit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = list("ABCDEFGHIJ")
    for fit in fits.values():
        diagnostics = fit.diagnostics.set_index("option_count")
        for option_count, bias in sorted(fit.biases.items()):
            diagnostic = diagnostics.loc[option_count].to_dict()
            for position, value in enumerate(bias):
                rows.append(
                    {
                        "model_id": fit.model_id,
                        "split_seed": fit.split_seed,
                        "option_count": option_count,
                        "label_position": position,
                        "label": labels[position],
                        "bias_estimate": float(value),
                        **diagnostic,
                    }
                )
    return rows


def run_interface_bias_analysis() -> dict[str, Any]:
    synthetic = synthetic_label_bias_control()
    if not synthetic["passed"]:
        raise RuntimeError(f"Synthetic label-bias control failed: {synthetic}")
    synthetic_path = PHYSICAL_ROOT / "outputs/controls/label_bias_synthetic_control.json"
    atomic_write_json(synthetic_path, synthetic)

    scores, _generation, selected, resolved = load_validated_inputs()
    model_ids = [record["id"] for record in resolved["models"]]
    records = {record["id"]: record for record in resolved["models"]}
    splits = fixed_splits(selected)
    estimate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []

    for split in splits.values():
        fits = {
            model: fit_label_bias(
                scores[scores["model_id"].eq(model)],
                split.fitting_ids,
                split_seed=split.seed,
            )
            for model in model_ids
        }
        estimate_rows.extend(_fit_rows(fits))
        validation_rows.extend(
            heldout_label_bias_validation(
                scores[scores["model_id"].eq(model)], split.evaluation_ids, fits[model]
            )
            for model in model_ids
        )
        raw_temperatures = {
            model: fit_clean_temperature(
                scores[scores["model_id"].eq(model)], split.fitting_ids
            )
            for model in model_ids
        }
        joint_temperatures = {
            model: fit_temperature_after_bias(
                scores[scores["model_id"].eq(model)],
                split.fitting_ids,
                fits[model].biases,
            )
            for model in model_ids
        }
        interface_specs = {
            "raw": (None, {}),
            "temperature_calibrated": (None, raw_temperatures),
            "label_bias_corrected": (fits, {}),
            "label_bias_corrected_then_temperature_calibrated": (
                fits,
                joint_temperatures,
            ),
        }
        for family in FAMILIES:
            for interface_condition, (condition_fits, temperatures) in interface_specs.items():
                aggregated = aggregate_interface_responses(
                    scores,
                    family,
                    fits=condition_fits,
                    temperatures=temperatures,
                )
                for target in model_ids:
                    peers = [model for model in model_ids if model != target]
                    result = primary_fit(aggregated, target, peers, split)
                    frame = result.dose_results
                    clean = float(frame[frame["dose"].eq(0.0)]["convex_error"].iloc[0])
                    high = float(
                        frame[frame["dose"].eq(high_dose(family))]["convex_error"].iloc[0]
                    )
                    sibling = designated_removals(target, peers, records)["exact_sibling"]
                    sibling_effect = math.nan
                    if sibling:
                        retained = [peer for peer in peers if peer not in sibling]
                        removed = primary_fit(aggregated, target, retained, split)
                        all_overall = float(
                            frame[frame["dose"].isna()]["convex_error"].iloc[0]
                        )
                        removed_overall = float(
                            removed.dose_results[removed.dose_results["dose"].isna()][
                                "convex_error"
                            ].iloc[0]
                        )
                        sibling_effect = removed_overall - all_overall
                    for _, row in frame.iterrows():
                        interface_rows.append(
                            {
                                "target": target,
                                "family": family,
                                "split_seed": split.seed,
                                "interface_condition": interface_condition,
                                "dose": float(row["dose"]) if pd.notna(row["dose"]) else math.nan,
                                "pier": float(row["convex_error"]),
                                "single_error": float(row["single_error"]),
                                "convex_vs_single_absolute_improvement": float(
                                    row["absolute_improvement"]
                                ),
                                "convex_vs_single_relative_improvement": float(
                                    row["relative_improvement"]
                                ),
                                "fit_selected_single_peer": str(
                                    row["fit_selected_single_peer"]
                                ),
                                "endpoint_delta": high - clean,
                                "sibling_removal_inflation": sibling_effect,
                                "target_temperature": float(temperatures.get(target, 1.0)),
                            }
                        )

    estimates = pd.DataFrame(estimate_rows)
    validation = pd.DataFrame(validation_rows)
    interfaces = pd.DataFrame(interface_rows)
    overall = interfaces[interfaces["dose"].isna()].copy()
    overall["target_rank"] = overall.groupby(
        ["family", "split_seed", "interface_condition"]
    )["pier"].rank(method="average")
    endpoint = (
        interfaces[interfaces["dose"].eq(0.0)]
        .groupby(["target", "family", "interface_condition"], as_index=False)
        .agg(
            mean_endpoint_delta=("endpoint_delta", "mean"),
            median_endpoint_delta=("endpoint_delta", "median"),
            endpoint_delta_standard_deviation=("endpoint_delta", "std"),
            positive_split_count=("endpoint_delta", lambda values: int(np.sum(values > 0))),
            negative_split_count=("endpoint_delta", lambda values: int(np.sum(values < 0))),
            mean_convexity_improvement=("convex_vs_single_absolute_improvement", "mean"),
            mean_sibling_removal_inflation=("sibling_removal_inflation", "mean"),
        )
    )
    raw_sign = endpoint[endpoint["interface_condition"].eq("raw")][
        ["target", "family", "mean_endpoint_delta"]
    ].rename(columns={"mean_endpoint_delta": "raw_mean_endpoint_delta"})
    endpoint = endpoint.merge(raw_sign, on=["target", "family"], validate="many_to_one")
    endpoint["same_endpoint_sign_as_raw"] = (
        np.sign(endpoint["mean_endpoint_delta"])
        == np.sign(endpoint["raw_mean_endpoint_delta"])
    )
    survival = endpoint.groupby(["target", "family"])["same_endpoint_sign_as_raw"].transform(
        "sum"
    )
    endpoint["interface_survival_count"] = survival.astype(int)
    endpoint["survives_all_four_interfaces"] = endpoint["interface_survival_count"].eq(4)

    estimate_path = PHYSICAL_ROOT / "outputs/analysis/label_bias_estimates.parquet"
    validation_path = PHYSICAL_ROOT / "outputs/analysis/label_bias_heldout_validation.parquet"
    interface_path = PHYSICAL_ROOT / "outputs/analysis/interface_corrected_pier.parquet"
    survival_path = PHYSICAL_ROOT / "outputs/analysis/interface_effect_survival.parquet"
    atomic_parquet(estimate_path, estimates)
    atomic_parquet(validation_path, validation)
    atomic_parquet(interface_path, interfaces)
    atomic_parquet(survival_path, endpoint)
    return {
        "outputs": [
            synthetic_path,
            estimate_path,
            validation_path,
            interface_path,
            survival_path,
        ],
        "estimate_rows": len(estimates),
        "validation_rows": len(validation),
        "interface_rows": len(interfaces),
        "survival_rows": len(endpoint),
    }
