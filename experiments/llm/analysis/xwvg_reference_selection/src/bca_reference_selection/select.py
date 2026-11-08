"""Deterministic exhaustive selection; this module takes fitting results only."""
from itertools import combinations
import math

from .fit import fit_tv_simplex


def candidate_subsets(peer_count=7, budgets=(1, 2, 3, 7)):
    if len(set(budgets)) != len(budgets) or any(not 1 <= k <= peer_count for k in budgets):
        raise ValueError("Invalid or duplicate peer budgets")
    return {k: list(combinations(range(peer_count), k)) for k in budgets}


def fit_candidate_subsets(target, peers, mask, observation_mass, k, solver_config=None):
    return {subset: fit_tv_simplex(target, peers[:, :, subset], mask,
                                   observation_mass, solver_config)
            for subset in candidate_subsets(peers.shape[-1], (k,))[k]}


def choose_collective(candidate_fits, tie_tolerance=1e-9):
    if not candidate_fits or any(not math.isfinite(fit.train_tv) for fit in candidate_fits.values()):
        raise ValueError("Missing or nonfinite candidate fits")
    best = min(fit.train_tv for fit in candidate_fits.values())
    return min(subset for subset, fit in candidate_fits.items()
               if fit.train_tv <= best + tie_tolerance)


def choose_nearest(singleton_fits, k, tie_tolerance=1e-9):
    """Sequential tolerance ties use the lowest remaining fixed-roster index."""
    errors = {subset[0]: fit.train_tv for subset, fit in singleton_fits.items() if len(subset) == 1}
    if not 1 <= k <= len(errors) or any(not math.isfinite(v) for v in errors.values()):
        raise ValueError("Invalid budget or singleton fits")
    remaining, selected = set(errors), []
    for _ in range(k):
        best = min(errors[j] for j in remaining)
        index = min(j for j in remaining if errors[j] <= best + tie_tolerance)
        selected.append(index)
        remaining.remove(index)
    return tuple(sorted(selected))


def select_policies(all_fits, peer_count=7, budgets=(1, 2, 3, 7), tie_tolerance=1e-9):
    expected = candidate_subsets(peer_count, budgets)
    required = {s for subsets in expected.values() for s in subsets}
    if set(all_fits) != required:
        raise ValueError(f"Candidate enumeration incomplete: missing={sorted(required-set(all_fits))}; "
                         f"extra={sorted(set(all_fits)-required)}")
    singletons = {s: fit for s, fit in all_fits.items() if len(s) == 1}
    if len(singletons) != peer_count:
        raise ValueError("Selection requires every singleton score")
    return {k: {"collective": choose_collective({s: all_fits[s] for s in subsets}, tie_tolerance),
                "nearest_k": choose_nearest(singletons, k, tie_tolerance)}
            for k, subsets in expected.items()}


def check_training_invariants(all_fits, selections, peer_count=7, tolerance=1e-7):
    """Raise on numerical/enumeration failure; no held-out dominance is assumed."""
    previous = float("inf")
    worst_dominance = worst_random = worst_monotonicity = worst_subset_singleton = 0.0
    for k in sorted(selections):
        chosen = selections[k]
        collective = all_fits[chosen["collective"]].train_tv
        nearest = all_fits[chosen["nearest_k"]].train_tv
        candidates = [f.train_tv for s, f in all_fits.items() if len(s) == k]
        random = sum(candidates) / len(candidates)
        worst_dominance = max(worst_dominance, collective - nearest)
        worst_random = max(worst_random, collective - random)
        worst_monotonicity = max(worst_monotonicity, collective - previous)
        previous = collective
        if k == 1 and chosen["collective"] != chosen["nearest_k"]:
            raise RuntimeError("k=1 policy identity failed")
        if k == peer_count and chosen["collective"] != chosen["nearest_k"]:
            raise RuntimeError("Full-roster policy identity failed")
    for subset, fit in all_fits.items():
        if fit.status != 0 or not math.isfinite(fit.train_tv):
            raise RuntimeError("Candidate fit failed")
        singleton_min = min(all_fits[(j,)].train_tv for j in subset)
        worst_subset_singleton = max(worst_subset_singleton, fit.train_tv - singleton_min)
    diagnostics = {"collective_minus_nearest_training_max": worst_dominance,
                   "collective_minus_random_training_max": worst_random,
                   "collective_training_monotonicity_violation_max": worst_monotonicity,
                   "subset_minus_best_member_training_max": worst_subset_singleton,
                   "candidate_count": len(all_fits), "passed": True}
    if max(worst_dominance, worst_random, worst_monotonicity, worst_subset_singleton) > tolerance:
        raise RuntimeError(f"Training objective checks failed: {diagnostics}")
    return diagnostics
