from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .utils import stable_seed, stable_u64

LABELS = tuple("ABCDEFGHIJ")

RAW_SIMPLEX_TOLERANCE = 1e-7
REPAIRED_MINIMUM_TOLERANCE = -1e-14
REPAIRED_SUM_TOLERANCE = 1e-12
REPAIR_PREDICTION_TOLERANCE = 1e-6
REPAIR_DIAGNOSTIC_COLUMNS = (
    "analysis_type",
    "interface",
    "model",
    "target",
    "family",
    "split",
    "bootstrap_replicate",
    "shard_id",
    "objective",
    "fit_role",
    "solver_status",
    "solver_objective",
    "raw_minimum_weight",
    "raw_sum_to_one_error",
    "repaired_minimum_weight",
    "repaired_sum_to_one_error",
    "l1_weight_change",
    "l2_weight_change",
    "linf_weight_change",
    "raw_objective",
    "repaired_objective",
    "absolute_objective_change",
    "relative_objective_change",
    "maximum_fitted_prediction_change",
    "maximum_evaluation_prediction_change",
)


def canonical_prompt(question: str, options: Sequence[str]) -> str:
    if not 2 <= len(options) <= len(LABELS):
        raise ValueError(f"Unsupported option count: {len(options)}")
    option_lines = "\n".join(f"({LABELS[index]}) {text}" for index, text in enumerate(options))
    return (
        "You are answering a multiple-choice question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Return only the label of the best option.\n"
        "Final answer:"
    )


def cyclic_mapping(option_count: int, rotation: int) -> tuple[list[int], list[int]]:
    """Return semantic->visible and visible->semantic mappings for pi_r(i)=(i+r)%m."""
    if option_count < 2 or not 0 <= rotation < option_count:
        raise ValueError((option_count, rotation))
    semantic_to_visible = [(semantic + rotation) % option_count for semantic in range(option_count)]
    visible_to_semantic = [0] * option_count
    for semantic, visible in enumerate(semantic_to_visible):
        visible_to_semantic[visible] = semantic
    return semantic_to_visible, visible_to_semantic


def permute_options(options: Sequence[str], rotation: int) -> tuple[list[str], list[int], list[int]]:
    semantic_to_visible, visible_to_semantic = cyclic_mapping(len(options), rotation)
    presented = [str(options[semantic]) for semantic in visible_to_semantic]
    return presented, semantic_to_visible, visible_to_semantic


def remap_visible_to_semantic(
    visible_values: Sequence[float], visible_to_semantic: Sequence[int]
) -> np.ndarray:
    visible = np.asarray(visible_values, dtype=np.float64)
    mapping = np.asarray(visible_to_semantic, dtype=np.int64)
    if visible.ndim != 1 or mapping.shape != visible.shape:
        raise ValueError("Visible vector and mapping must be aligned one-dimensional arrays")
    if sorted(mapping.tolist()) != list(range(len(mapping))):
        raise ValueError("Mapping is not a permutation")
    semantic = np.empty_like(visible)
    semantic[mapping] = visible
    return semantic


def remap_semantic_to_visible(
    semantic_values: Sequence[float], semantic_to_visible: Sequence[int]
) -> np.ndarray:
    semantic = np.asarray(semantic_values, dtype=np.float64)
    mapping = np.asarray(semantic_to_visible, dtype=np.int64)
    if semantic.ndim != 1 or mapping.shape != semantic.shape:
        raise ValueError("Semantic vector and mapping must be aligned one-dimensional arrays")
    visible = np.empty_like(semantic)
    visible[mapping] = semantic
    return visible


def softmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - float(np.max(array))
    result = np.exp(shifted)
    return result / result.sum()


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right)).sum())


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (p + q)

    def kl(values: np.ndarray, reference: np.ndarray) -> float:
        positive = values > 0
        return float(np.sum(values[positive] * np.log(values[positive] / reference[positive])))

    return 0.5 * kl(p, middle) + 0.5 * kl(q, middle)


def categorical_entropy(values: Sequence[int]) -> float:
    _, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def validate_permutation_balance(option_count: int) -> None:
    semantic_positions = np.zeros((option_count, option_count), dtype=np.int64)
    visible_semantics = np.zeros((option_count, option_count), dtype=np.int64)
    for rotation in range(option_count):
        semantic_to_visible, visible_to_semantic = cyclic_mapping(option_count, rotation)
        for semantic, visible in enumerate(semantic_to_visible):
            semantic_positions[semantic, visible] += 1
        for visible, semantic in enumerate(visible_to_semantic):
            visible_semantics[visible, semantic] += 1
    if not np.array_equal(semantic_positions, np.ones_like(semantic_positions)):
        raise AssertionError("A semantic option does not occupy each visible position exactly once")
    if not np.array_equal(visible_semantics, np.ones_like(visible_semantics)):
        raise AssertionError("A visible position does not receive each semantic option exactly once")


EXPLICIT_PATTERNS = (
    re.compile(r"(?is)final\s+answer\s*(?:is|:)?\s*\(?([A-J])\)?"),
    re.compile(r"(?is)(?<!final\s)answer\s*(?:is|:)?\s*\(?([A-J])\)?"),
    re.compile(r"(?is)the\s+answer\s+is\s*\(?([A-J])\)?"),
)
STANDALONE_LABEL = re.compile(r"(?i)(?<![A-Za-z0-9])\(?([A-J])\)?(?![A-Za-z0-9])")
REASONING_DELIMITER = re.compile(r"(?is)</think>|<\/analysis>|final\s*:")


def extract_final_label(text: str, valid_labels: Sequence[str]) -> tuple[str | None, str]:
    valid = {str(value).upper() for value in valid_labels}
    tail = text[-4096:]
    explicit: list[tuple[int, str]] = []
    for pattern in EXPLICIT_PATTERNS:
        for match in pattern.finditer(tail):
            label = match.group(1).upper()
            if label in valid:
                explicit.append((match.start(), label))
    if explicit:
        return max(explicit)[1], "explicit_final_answer"

    standalone = [
        (match.start(), match.group(1).upper())
        for match in STANDALONE_LABEL.finditer(tail)
        if match.group(1).upper() in valid
    ]
    if standalone:
        return standalone[-1][1], "final_standalone_label"

    delimiters = list(REASONING_DELIMITER.finditer(tail))
    if delimiters:
        after = tail[delimiters[-1].end() :]
        matches = [
            match.group(1).upper()
            for match in STANDALONE_LABEL.finditer(after)
            if match.group(1).upper() in valid
        ]
        if matches:
            return matches[-1], "post_reasoning_delimiter"
    return None, "unresolved"


@dataclass(frozen=True)
class Split:
    seed: int
    fitting_ids: frozenset[str]
    evaluation_ids: frozenset[str]


def fixed_split(selected: pd.DataFrame, seed: int) -> Split:
    unique = selected[["base_question_id", "category"]].drop_duplicates().copy()
    unique["base_question_id"] = unique["base_question_id"].astype(str)
    fitting: set[str] = set()
    evaluation: set[str] = set()
    for category, group in unique.groupby("category", sort=True):
        identifiers = sorted(
            group["base_question_id"], key=lambda value: (stable_u64(seed, category, value), value)
        )
        if len(identifiers) % 2:
            raise ValueError(f"Odd category size for {category}")
        midpoint = len(identifiers) // 2
        fitting.update(identifiers[:midpoint])
        evaluation.update(identifiers[midpoint:])
    all_ids = set(unique["base_question_id"])
    if fitting & evaluation or fitting | evaluation != all_ids:
        raise AssertionError("Fixed split leaked or lost identifiers")
    return Split(int(seed), frozenset(fitting), frozenset(evaluation))


def fixed_splits(selected: pd.DataFrame, seeds: Iterable[int]) -> dict[int, Split]:
    return {int(seed): fixed_split(selected, int(seed)) for seed in seeds}


def endpoint_row_weights(metadata: pd.DataFrame, multiplicities: np.ndarray | None = None) -> np.ndarray:
    track = metadata["track"].fillna(-1).astype(int).to_numpy()
    base = np.where(track == -1, 0.5, 0.5 / 3.0).astype(np.float64)
    if multiplicities is not None:
        base *= np.asarray(multiplicities, dtype=np.float64)
    return base


def _load_v22_projection() -> Any | None:
    source = Path("/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS/src")
    if not source.is_dir():
        return None
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from pier_llm_reanalysis_v22.v2_solver import fit_simplex_projection

        return fit_simplex_projection
    except (ImportError, OSError):
        return None


def project_probability_simplex(values: np.ndarray) -> np.ndarray:
    """Return the deterministic sorting-based Euclidean projection onto the simplex."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < 1 or not np.isfinite(vector).all():
        raise ValueError("Simplex projection requires a finite non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=np.float64)
    admissible = ordered - cumulative / indices > 0.0
    if not admissible.any():
        raise RuntimeError("Simplex projection found no active coordinate")
    rho = int(np.flatnonzero(admissible)[-1])
    threshold = float(cumulative[rho] / (rho + 1.0))
    return np.maximum(vector - threshold, 0.0)


def validate_or_repair_simplex(
    raw_weights: np.ndarray,
    *,
    design: np.ndarray,
    target: np.ndarray,
    mass: np.ndarray,
    objective: str,
    solver_status: str,
    solver_objective: float,
    evaluation_design: np.ndarray | None = None,
    repair_context: dict[str, Any] | None = None,
    repair_diagnostics: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Validate an optimal solver result and repair only authorized numerical drift."""
    raw = np.asarray(raw_weights, dtype=np.float64).reshape(-1).copy()
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    fit_mass = np.asarray(mass, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != raw.size or x.shape[0] != y.size:
        raise ValueError("Simplex repair design and solver weights are not aligned")
    if fit_mass.ndim != 1 or fit_mass.size != y.size:
        raise ValueError("Simplex repair mass is not aligned")
    if not np.isfinite(raw).all():
        raise RuntimeError("Simplex solver returned non-finite weights")
    if not np.isfinite(float(solver_objective)):
        raise RuntimeError("Simplex solver returned a non-finite objective")

    raw_minimum = float(raw.min())
    raw_sum_error = abs(float(raw.sum()) - 1.0)
    if raw_minimum < -RAW_SIMPLEX_TOLERANCE:
        raise RuntimeError(
            f"Simplex raw minimum {raw_minimum:.17g} exceeds authorized tolerance"
        )
    if raw_sum_error > RAW_SIMPLEX_TOLERANCE:
        raise RuntimeError(
            f"Simplex raw sum error {raw_sum_error:.17g} exceeds authorized tolerance"
        )

    already_strict = (
        raw_minimum >= REPAIRED_MINIMUM_TOLERANCE
        and raw_sum_error <= REPAIRED_SUM_TOLERANCE
    )
    if already_strict:
        return raw

    normalized_status = str(solver_status).strip().lower().split(":", maxsplit=1)[0]
    if normalized_status != "optimal":
        raise RuntimeError(f"Simplex feasibility repair requires Optimal status: {solver_status}")

    repaired = project_probability_simplex(raw)
    repaired_minimum = float(repaired.min())
    repaired_sum_error = abs(float(repaired.sum()) - 1.0)
    delta = repaired - raw
    l1_change = float(np.linalg.norm(delta, ord=1))
    l2_change = float(np.linalg.norm(delta, ord=2))
    linf_change = float(np.linalg.norm(delta, ord=np.inf))
    if not np.isfinite(repaired).all():
        raise RuntimeError("Simplex repair produced non-finite weights")
    if repaired_minimum < REPAIRED_MINIMUM_TOLERANCE:
        raise RuntimeError("Simplex repair failed the minimum-weight gate")
    if repaired_sum_error > REPAIRED_SUM_TOLERANCE:
        raise RuntimeError("Simplex repair failed the sum-to-one gate")
    if linf_change > RAW_SIMPLEX_TOLERANCE:
        raise RuntimeError("Simplex repair failed the Linf-change gate")

    if objective == "mse":
        metric = weighted_mse
    elif objective == "mae":
        metric = weighted_mae
    else:
        raise ValueError(f"Unknown simplex objective: {objective}")
    raw_prediction = x @ raw
    repaired_prediction = x @ repaired
    raw_objective = metric(y, raw_prediction, fit_mass)
    repaired_objective = metric(y, repaired_prediction, fit_mass)
    objective_change = abs(repaired_objective - raw_objective)
    relative_objective_change = objective_change / max(1.0, abs(raw_objective))
    objective_tolerance = max(
        1e-10,
        1e-7 * max(1.0, abs(raw_objective)),
    )
    maximum_fitted_change = float(np.max(np.abs(repaired_prediction - raw_prediction)))
    maximum_evaluation_change = 0.0
    if evaluation_design is not None:
        evaluation = np.asarray(evaluation_design, dtype=np.float64)
        if evaluation.ndim != 2 or evaluation.shape[1] != raw.size:
            raise ValueError("Simplex repair evaluation design is not aligned")
        if evaluation.shape[0]:
            maximum_evaluation_change = float(np.max(np.abs(evaluation @ delta)))
    if objective_change > objective_tolerance:
        raise RuntimeError(
            "Simplex repair failed the objective-change gate: "
            f"{objective_change:.17g} > {objective_tolerance:.17g}"
        )
    if maximum_fitted_change > REPAIR_PREDICTION_TOLERANCE:
        raise RuntimeError("Simplex repair failed the fitted-prediction gate")
    if maximum_evaluation_change > REPAIR_PREDICTION_TOLERANCE:
        raise RuntimeError("Simplex repair failed the evaluation-prediction gate")

    context = repair_context or {}
    record = {
        "analysis_type": context.get("analysis_type"),
        "interface": context.get("interface"),
        "model": context.get("model"),
        "target": context.get("target"),
        "family": context.get("family"),
        "split": context.get("split"),
        "bootstrap_replicate": context.get("bootstrap_replicate"),
        "shard_id": context.get("shard_id"),
        "objective": objective,
        "fit_role": context.get("fit_role"),
        "solver_status": str(solver_status),
        "solver_objective": float(solver_objective),
        "raw_minimum_weight": raw_minimum,
        "raw_sum_to_one_error": raw_sum_error,
        "repaired_minimum_weight": repaired_minimum,
        "repaired_sum_to_one_error": repaired_sum_error,
        "l1_weight_change": l1_change,
        "l2_weight_change": l2_change,
        "linf_weight_change": linf_change,
        "raw_objective": raw_objective,
        "repaired_objective": repaired_objective,
        "absolute_objective_change": objective_change,
        "relative_objective_change": relative_objective_change,
        "maximum_fitted_prediction_change": maximum_fitted_change,
        "maximum_evaluation_prediction_change": maximum_evaluation_change,
    }
    if repair_diagnostics is not None:
        repair_diagnostics.append(record)
    return repaired


def fit_mse_simplex(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    evaluation_design: np.ndarray | None = None,
    repair_context: dict[str, Any] | None = None,
    repair_diagnostics: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    positive = mass > 0
    x, y, mass = x[positive], y[positive], mass[positive]
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(mass) != len(y):
        raise ValueError("MSE design, target, and weights are not aligned")
    if x.shape[1] < 1 or len(y) < 1:
        raise ValueError("MSE fit is empty")
    square_root = np.sqrt(mass / mass.sum())
    solver = _load_v22_projection()
    if solver is not None:
        result = solver(x * square_root[:, None], y * square_root)
        fitted = np.asarray(result.weights, dtype=np.float64)
        solver_status = str(result.status_stage2)
        solver_objective = float(result.stage2_objective)
    else:
        peers = x.shape[1]

        def objective(candidate: np.ndarray) -> float:
            residual = y - x @ candidate
            return float(np.sum(mass * residual * residual) / mass.sum())

        result = minimize(
            objective,
            np.full(peers, 1.0 / peers),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * peers,
            constraints={"type": "eq", "fun": lambda candidate: float(candidate.sum() - 1.0)},
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not result.success:
            raise RuntimeError(f"MSE simplex solver failed: {result.message}")
        fitted = np.asarray(result.x, dtype=np.float64)
        solver_status = "Optimal"
        solver_objective = float(result.fun)
    return validate_or_repair_simplex(
        fitted,
        design=x,
        target=y,
        mass=mass,
        objective="mse",
        solver_status=solver_status,
        solver_objective=solver_objective,
        evaluation_design=evaluation_design,
        repair_context=repair_context,
        repair_diagnostics=repair_diagnostics,
    )


def fit_mae_simplex(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    evaluation_design: np.ndarray | None = None,
    repair_context: dict[str, Any] | None = None,
    repair_diagnostics: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    positive = mass > 0
    x, y, mass = x[positive], y[positive], mass[positive]
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(mass) != len(y):
        raise ValueError("MAE design, target, and weights are not aligned")
    rows, peers = x.shape
    objective = np.concatenate([np.zeros(peers), mass / mass.sum()])
    upper = np.block([[x, -np.eye(rows)], [-x, -np.eye(rows)]])
    upper_rhs = np.concatenate([y, -y])
    equality = np.concatenate([np.ones(peers), np.zeros(rows)])[None, :]
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=upper_rhs,
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * (peers + rows),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"MAE simplex LP failed: {result.message}")
    fitted = np.asarray(result.x[:peers], dtype=np.float64)
    return validate_or_repair_simplex(
        fitted,
        design=x,
        target=y,
        mass=mass,
        objective="mae",
        solver_status="Optimal",
        solver_objective=float(result.fun),
        evaluation_design=evaluation_design,
        repair_context=repair_context,
        repair_diagnostics=repair_diagnostics,
    )


def weighted_mae(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    mass = np.asarray(weights, dtype=np.float64)
    return float(np.sum(mass * np.abs(np.asarray(target) - np.asarray(prediction))) / mass.sum())


def weighted_mse(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    mass = np.asarray(weights, dtype=np.float64)
    residual = np.asarray(target) - np.asarray(prediction)
    return float(np.sum(mass * residual * residual) / mass.sum())


def select_single_peer(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray, objective: str
) -> int:
    metric = weighted_mse if objective == "mse" else weighted_mae
    errors = [metric(target, design[:, index], weights) for index in range(design.shape[1])]
    return int(np.argmin(np.asarray(errors)))


def common_bootstrap_multiplicities(
    selected: pd.DataFrame, replicate: int, seed: int
) -> dict[str, int]:
    result: dict[str, int] = {}
    unique = selected[["base_question_id", "category"]].drop_duplicates().copy()
    unique["base_question_id"] = unique["base_question_id"].astype(str)
    rng = np.random.default_rng(stable_seed(seed, "common-global", replicate))
    for _category, group in unique.groupby("category", sort=True):
        identifiers = sorted(group["base_question_id"].tolist())
        draws = rng.choice(identifiers, size=len(identifiers), replace=True)
        values, counts = np.unique(draws, return_counts=True)
        result.update({identifier: 0 for identifier in identifiers})
        result.update({str(value): int(count) for value, count in zip(values, counts, strict=True)})
    return result


def run_synthetic_controls() -> dict[str, Any]:
    for option_count in range(2, 11):
        validate_permutation_balance(option_count)

    known = np.asarray([0.05, 0.15, 0.25, 0.55])
    round_trip_error = 0.0
    invariant_vectors: list[np.ndarray] = []
    for rotation in range(len(known)):
        semantic_to_visible, visible_to_semantic = cyclic_mapping(len(known), rotation)
        visible = remap_semantic_to_visible(known, semantic_to_visible)
        recovered = remap_visible_to_semantic(visible, visible_to_semantic)
        round_trip_error = max(round_trip_error, float(np.max(np.abs(recovered - known))))
        invariant_vectors.append(recovered)
    invariant_average = np.mean(invariant_vectors, axis=0)
    invariant_variance = float(np.max(np.var(invariant_vectors, axis=0)))

    semantic_logits = np.asarray([0.8, 0.1, -0.2, -0.7])
    label_bias = np.asarray([0.7, -0.4, 0.2, -0.5])

    def balanced_with_bias(scale: float, interaction: float = 0.0) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for rotation in range(4):
            semantic_to_visible, visible_to_semantic = cyclic_mapping(4, rotation)
            visible_semantic_logits = np.asarray([semantic_logits[s] for s in visible_to_semantic])
            semantic_index = np.asarray(visible_to_semantic, dtype=np.float64)
            position = np.arange(4, dtype=np.float64)
            logits = visible_semantic_logits + scale * label_bias + interaction * semantic_index * position
            vectors.append(remap_visible_to_semantic(softmax(logits), visible_to_semantic))
        return np.mean(vectors, axis=0)

    finite_step = 1e-5
    prior_derivative = (balanced_with_bias(finite_step) - balanced_with_bias(-finite_step)) / (
        2 * finite_step
    )
    pure_label_prior_first_order_error = float(np.max(np.abs(prior_derivative)))
    baseline = balanced_with_bias(0.0)
    interaction_vector = balanced_with_bias(0.0, interaction=0.12)
    non_additive_remaining_tv = total_variation(baseline, interaction_vector)

    rng = np.random.default_rng(11)
    peers = rng.uniform(0.0, 1.0, size=(80, 4))
    weights = np.full(80, 1.0)
    exact_clone = peers[:, 2]
    clone_fit = fit_mse_simplex(peers, exact_clone, weights)
    clone_pier = weighted_mae(exact_clone, peers @ clone_fit, weights)
    truth = np.asarray([0.1, 0.2, 0.3, 0.4])
    mixture = peers @ truth
    mixture_fit = fit_mse_simplex(peers, mixture, weights)
    mixture_pier = weighted_mae(mixture, peers @ mixture_fit, weights)

    selected = pd.DataFrame(
        {
            "base_question_id": [f"q{index}" for index in range(8)],
            "category": ["a"] * 4 + ["b"] * 4,
        }
    )
    multiplicity = common_bootstrap_multiplicities(selected, 3, 19)
    overlapping_a = [multiplicity[f"q{index}"] for index in (0, 1, 4, 5)]
    overlapping_b = [multiplicity[f"q{index}"] for index in (1, 2, 5, 6)]
    common_dependence_passed = overlapping_a[1] == overlapping_b[0] and overlapping_a[3] == overlapping_b[2]

    x = np.column_stack([np.linspace(0, 1, 40), np.linspace(1, 0, 40), np.full(40, 0.5)])
    y = 0.35 * x[:, 0] + 0.65 * x[:, 1]
    mass = np.ones(40)
    mse_single = select_single_peer(x, y, mass, "mse")
    mae_single = select_single_peer(x, y, mass, "mae")
    mse_convex = fit_mse_simplex(x, y, mass)
    mae_convex = fit_mae_simplex(x, y, mass)
    matched_objective_passed = (
        weighted_mae(y, x @ mse_convex, mass) <= weighted_mae(y, x[:, mse_single], mass) + 1e-10
        and weighted_mae(y, x @ mae_convex, mass) <= weighted_mae(y, x[:, mae_single], mass) + 1e-10
    )

    result = {
        "permutation_balance": True,
        "round_trip_max_error": round_trip_error,
        "invariant_average_max_error": float(np.max(np.abs(invariant_average - known))),
        "invariant_variance": invariant_variance,
        "pure_label_prior_first_order_max_error": pure_label_prior_first_order_error,
        "non_additive_remaining_tv": non_additive_remaining_tv,
        "exact_clone_pier": clone_pier,
        "known_mixture_pier": mixture_pier,
        "common_bootstrap_dependence": common_dependence_passed,
        "matched_objective": matched_objective_passed,
    }
    result["passed"] = bool(
        round_trip_error < 1e-15
        and result["invariant_average_max_error"] < 1e-15
        and invariant_variance < 1e-30
        and pure_label_prior_first_order_error < 1e-8
        and non_additive_remaining_tv > 1e-5
        and clone_pier < 1e-9
        and mixture_pier < 1e-9
        and common_dependence_passed
        and matched_objective_passed
    )
    return result
