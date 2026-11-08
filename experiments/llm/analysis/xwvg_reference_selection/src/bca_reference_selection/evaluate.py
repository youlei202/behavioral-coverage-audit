"""Held-out scoring of saved fits and exact expected random-subset metrics."""
import numpy as np

from .fit import per_observation_tv, validate_inputs


def evaluate_candidate(fit, subset, target, peers, mask, observation_mass, question_ids=None):
    """question_ids, when supplied, has one base question ID per observation."""
    y, x, valid, mass = validate_inputs(target, peers[:, :, subset], mask, observation_mass)
    weights = np.asarray(fit.weights)
    if weights.shape != (len(subset),) or not np.all(np.isfinite(weights)):
        raise ValueError("Fit weights do not match the saved subset")
    if weights.min() < -1e-12 or abs(weights.sum() - 1) > 1e-10:
        raise ValueError("Saved weights violate the simplex")
    prediction = np.einsum("nap,p->na", x, weights)
    losses = per_observation_tv(y, prediction, valid)
    left = np.argmax(np.where(valid, y, -np.inf), axis=1)
    right = np.argmax(np.where(valid, prediction, -np.inf), axis=1)
    agreement = (left == right).astype(float)
    result = {"evaluation_tv": float(np.average(losses, weights=mass)),
              "top_option_agreement": float(np.average(agreement, weights=mass)),
              "per_observation_tv": losses,
              "per_observation_top_option_agreement": agreement,
              # Any simplex rounding can change evaluation predictions by at
              # most this L1 coefficient change because probabilities <= 1.
              "repair_evaluation_prediction_max_abs_bound": fit.repair_l1}
    if fit.raw_weights is not None:
        raw_prediction = np.einsum("nap,p->na", x, np.asarray(fit.raw_weights))
        result["repair_evaluation_prediction_max_abs"] = float(np.max(np.abs(prediction - raw_prediction)))
        result["repair_evaluation_tv_change"] = result["evaluation_tv"] - float(
            np.average(per_observation_tv(y, raw_prediction, valid), weights=mass))
    if question_ids is not None:
        ids = np.asarray(question_ids).astype(str)
        if ids.shape != (len(y),):
            raise ValueError("One question ID is required for each observation")
        order = list(dict.fromkeys(ids.tolist()))
        question_tv, question_agreement, question_mass = [], [], []
        for q in order:
            index = ids == q
            total = mass[index].sum()
            if total <= 0:
                raise ValueError(f"Question {q} has zero evaluation mass")
            question_tv.append(float(np.average(losses[index], weights=mass[index])))
            question_agreement.append(float(np.average(agreement[index], weights=mass[index])))
            question_mass.append(float(total))
        result.update(question_ids=order,
                      question_tv=np.asarray(question_tv),
                      question_top_option_agreement=np.asarray(question_agreement),
                      question_mass=np.asarray(question_mass))
    return result


def random_expected(evaluations):
    """Average each fitted subset's metrics, never its predicted distribution."""
    if not evaluations:
        raise ValueError("Random expectation requires every candidate evaluation")
    records = list(evaluations.values()) if isinstance(evaluations, dict) else list(evaluations)
    result = {key: float(np.mean([r[key] for r in records]))
              for key in ("evaluation_tv", "top_option_agreement")}
    for key in ("per_observation_tv", "per_observation_top_option_agreement",
                "question_tv", "question_top_option_agreement"):
        if key in records[0]:
            result[key] = np.mean(np.stack([r[key] for r in records]), axis=0)
    if "question_ids" in records[0]:
        if any(r["question_ids"] != records[0]["question_ids"] for r in records):
            raise ValueError("Random-subset question identities do not align")
        result["question_ids"] = records[0]["question_ids"]
        result["question_mass"] = records[0]["question_mass"]
    return result
