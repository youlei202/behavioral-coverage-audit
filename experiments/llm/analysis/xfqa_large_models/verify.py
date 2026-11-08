"""Independent CPU replay of saved XfqA coefficients and measured responses.

No analysis evaluator or fitting routine is imported.  The verifier reconstructs
question splits, predictions, objectives, selected singles and contrasts from
the recorded scientific inputs.  It also checks raw cyclic-score averaging.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .reporting import TABLE_COUNTS, sha256, write_json
except ImportError:
    from reporting import TABLE_COUNTS, sha256, write_json

METRIC_ATOL = 1e-10
TRAIN_ATOL = 1e-7
PROBABILITY_ATOL = 1e-12
CONDITION_MASS = np.array([.5, 1/6, 1/6, 1/6], dtype=np.float64)
FAMILY_CONDITIONS = {"irrelevant_context": ["C0", "I0", "I1", "I2"],
                     "content_deletion": ["C0", "D0", "D1", "D2"]}


def json_list(value):
    return json.loads(value) if isinstance(value, str) else list(value)


def read_table_csv(path):
    """Preserve lexical identities even when a run ID contains only digits."""
    return pd.read_csv(path, dtype={"run_id": str, "input_fingerprint": str})


def _stable_u64(*parts):
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "big")


class Audit:
    def __init__(self):
        self.checks = []
        self.max_errors = defaultdict(float)

    def check(self, condition, label, **details):
        passed = bool(condition)
        self.checks.append({"name": label, "passed": passed, **details})
        if not passed:
            raise ValueError(f"Numerical replay failed: {label}: {details}")

    def close(self, left, right, label, atol=METRIC_ATOL):
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if a.shape != b.shape:
            raise ValueError(f"Numerical replay shape mismatch: {label}: {a.shape} != {b.shape}")
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise ValueError(f"Numerical replay NaN mismatch: {label}")
        finite = np.isfinite(a) & np.isfinite(b)
        if not np.all(finite | (np.isnan(a) & np.isnan(b))):
            raise ValueError(f"Nonfinite numerical replay data: {label}")
        error = float(np.max(np.abs(a[finite] - b[finite]), initial=0))
        self.max_errors[label] = max(self.max_errors[label], error)
        if error > atol:
            raise ValueError(f"Numerical replay mismatch: {label}: {error} > {atol}")


def independent_splits(questions, seeds):
    result = {}
    for seed in seeds:
        fit, evaluation = [], []
        for category in sorted({q["category"] for q in questions}):
            ids = [str(q["base_question_id"]) for q in questions if q["category"] == category]
            ids.sort(key=lambda q: (_stable_u64(seed, category, q), q))
            if len(ids) != 40:
                raise ValueError("Expected forty questions in each subject")
            fit += ids[:20]
            evaluation += ids[20:]
        result[int(seed)] = {"fitting_ids": sorted(fit), "evaluation_ids": sorted(evaluation)}
    return result


def expected_jobs(config):
    """Reconstruct the declared grid, independently of the production builder."""
    old = config["original_roster"]
    added = [r["id"] for r in config["new_models"]]
    models = old + added
    directions = {(p[a], p[b]) for p in config["declared_pairs"] for a, b in (("a", "b"), ("b", "a"))}
    records = {}
    for panel, targets, roster in (("expanded11", models, models), ("baseline8", old, old), ("new_on_old8", added, old)):
        for target in targets:
            peers = [m for m in roster if m != target]
            for family in sorted(config["families"]):
                for seed in config["split_seeds"]:
                    for loss in ("mse", "mae", "tv"):
                        removals = [None]
                        if panel == "expanded11":
                            removals += [p for p in peers if loss != "tv" or (target, p) in directions]
                        for removed in removals:
                            record = {"panel": panel, "target": target, "family": family, "split_seed": seed,
                                      "response": "complete_option_vector" if loss == "tv" else "reference_answer_probability",
                                      "fit_loss": loss, "peers": [p for p in peers if p != removed],
                                      "role": "group" if removed is None else "removal", "removed_peer": removed}
                            job_id = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                            records[job_id] = {"job_id": job_id, **record}
    if len(records) != 5840:
        raise ValueError("Independent expected job grid is not 5,840 fits")
    return records


class Responses:
    """Read complete ragged vectors without using the production loader."""
    def __init__(self, config, questions_path, cache_path, audit):
        self.config = config
        self.questions = [json.loads(line) for line in Path(questions_path).read_text().splitlines() if line.strip()]
        self.questions.sort(key=lambda row: str(row["base_question_id"]))
        self.qids = [str(q["base_question_id"]) for q in self.questions]
        self.qindex = {q: i for i, q in enumerate(self.qids)}
        self.categories = [q["category"] for q in self.questions]
        self.models = list(config["expanded_roster"])
        self.mindex = {m: i for i, m in enumerate(self.models)}
        self.conditions = list(config["conditions"])
        self.cindex = {c: i for i, c in enumerate(self.conditions)}
        self.counts = np.array([len(q["options"]) for q in self.questions])
        self.gold = np.array([q["answer_index"] for q in self.questions], dtype=int)
        audit.check(len(self.qids) == len(set(self.qids)) == 560, "560 unique selected questions")
        audit.check(len(set(self.categories)) == 14 and set(Counter(self.categories).values()) == {40}, "14 subjects, 40 questions each")
        frame = pd.read_parquet(cache_path)
        self.frame = frame
        audit.check(len(frame) == 43120 and not frame.duplicated(["model_id", "base_question_id", "semantic_condition"]).any(),
                    "43,120 unique merged semantic vectors")
        self.p = np.zeros((11, 560, 7, 10), dtype=np.float64)
        seen = np.zeros((11, 560, 7), dtype=bool)
        for row in frame.to_dict("records"):
            m, q, c = self.mindex[row["model_id"]], self.qindex[str(row["base_question_id"])], self.cindex[row["semantic_condition"]]
            values = np.asarray(row["permutation_averaged_semantic_probabilities"], dtype=float)
            n = int(self.counts[q])
            if values.shape != (n,) or not np.isfinite(values).all() or values.min() < -1e-12 or values.max() > 1 + 1e-12:
                raise ValueError("Invalid actual semantic option vector")
            if abs(values.sum() - 1) > config["numeric"]["prob_sum_atol"]:
                raise ValueError("Semantic option probabilities do not sum to one")
            if row["category"] != self.categories[q] or int(row["option_count"]) != n or int(row["gold_semantic_option"]) != self.gold[q]:
                raise ValueError("Merged cache question identity differs from selected questions")
            if seen[m, q, c]:
                raise ValueError("Repeated merged-cache primary key")
            self.p[m, q, c, :n] = values
            seen[m, q, c] = True
            if "permutation_averaged_gold_probability" in row:
                audit.close(row["permutation_averaged_gold_probability"], values[self.gold[q]], "merged scalar from full vector", PROBABILITY_ATOL)
            if "permutation_averaged_semantic_argmax" in row and int(row["permutation_averaged_semantic_argmax"]) != int(values.argmax()):
                raise ValueError("Merged vector argmax mismatch")
        audit.check(seen.all(), "Complete eleven-model/question/condition Cartesian grid")
        self.y = np.take_along_axis(self.p, np.broadcast_to(self.gold[None, :, None, None], (11, 560, 7, 1)), axis=-1)[..., 0]
        self.splits = independent_splits(self.questions, config["split_seeds"])
        self.fit_indices = {s: np.array([self.qindex[q] for q in r["fitting_ids"]]) for s, r in self.splits.items()}
        self.eval_indices = {s: np.array([self.qindex[q] for q in r["evaluation_ids"]]) for s, r in self.splits.items()}
        self.array_cache = {}

    def arrays(self, job):
        key = (job["target"], job["family"], job["fit_loss"] == "tv", tuple(job["peers"]))
        if key in self.array_cache:
            return self.array_cache[key]
        target = self.mindex[job["target"]]
        peers = [self.mindex[p] for p in job["peers"]]
        conditions = [self.cindex[c] for c in FAMILY_CONDITIONS[job["family"]]]
        source = self.p if job["fit_loss"] == "tv" else self.y
        y = source[target][:, conditions]
        x = np.moveaxis(source[peers][:, :, conditions], 0, -1)
        self.array_cache[key] = (y, x)
        return y, x


def replay_fit(responses, job, weights, multiplicity=None):
    y, x = responses.arrays(job)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(job["peers"]),) or not np.isfinite(w).all() or w.min() < -1e-14 or abs(w.sum() - 1) > 1e-12:
        raise ValueError("Saved final coefficients violate the simplex")
    train, evaluation = responses.fit_indices[int(job["split_seed"])], responses.eval_indices[int(job["split_seed"])]
    draw = np.ones(560) if multiplicity is None else np.asarray(multiplicity, dtype=float)
    fit_mass = draw[train, None] * CONDITION_MASS
    eval_mass = draw[evaluation, None] * CONDITION_MASS
    if fit_mass.sum() <= 0 or eval_mass.sum() <= 0:
        raise ValueError("Empty weighted fitting or evaluation split")
    fit_mass /= fit_mass.sum()
    eval_mass /= eval_mass.sum()
    prediction = np.einsum("...p,p->...", x, w)
    scalar = job["fit_loss"] != "tv"
    if scalar:
        fitting_residual = prediction[train] - y[train]
        singleton_residual = x[train] - y[train, :, None]
        transform = np.square if job["fit_loss"] == "mse" else np.abs
        train_objective = float(np.sum(fit_mass * transform(fitting_residual)))
        singleton_objectives = np.sum(fit_mass[..., None] * transform(singleton_residual), axis=(0, 1))
    else:
        train_objective = float(np.sum(fit_mass * .5 * np.abs(prediction[train] - y[train]).sum(axis=-1)))
        singleton_objectives = np.sum(fit_mass[..., None] * .5 * np.abs(x[train] - y[train, ..., None]).sum(axis=2), axis=(0, 1))
    selected = int(np.flatnonzero(singleton_objectives <= singleton_objectives.min() + 1e-12)[0])
    single = x[..., selected]
    if scalar:
        arrays = {"absolute_error": abs(prediction - y), "squared_error": (prediction - y)**2,
                  "single_absolute_error": abs(single - y), "single_squared_error": (single - y)**2}
        metrics = {"e_group": float(np.sum(eval_mass * arrays["absolute_error"][evaluation])),
                   "e_single": float(np.sum(eval_mass * arrays["single_absolute_error"][evaluation])),
                   "mse_group": float(np.sum(eval_mass * arrays["squared_error"][evaluation])),
                   "mse_single": float(np.sum(eval_mass * arrays["single_squared_error"][evaluation]))}
    else:
        valid = np.arange(10)[None, None, :] < responses.counts[:, None, None]
        target_arg = np.where(valid, y, -np.inf).argmax(axis=-1)
        arrays = {"total_variation": .5 * abs(prediction - y).sum(axis=-1),
                  "single_total_variation": .5 * abs(single - y).sum(axis=-1),
                  "top_option_agreement": (np.where(valid, prediction, -np.inf).argmax(axis=-1) == target_arg).astype(float),
                  "single_top_option_agreement": (np.where(valid, single, -np.inf).argmax(axis=-1) == target_arg).astype(float)}
        metrics = {"e_group": float(np.sum(eval_mass * arrays["total_variation"][evaluation])),
                   "e_single": float(np.sum(eval_mass * arrays["single_total_variation"][evaluation])),
                   "group_top_option_agreement": float(np.sum(eval_mass * arrays["top_option_agreement"][evaluation])),
                   "single_top_option_agreement": float(np.sum(eval_mass * arrays["single_top_option_agreement"][evaluation]))}
        metrics["agreement_gain"] = metrics["group_top_option_agreement"] - metrics["single_top_option_agreement"]
    metrics.update(train_group=train_objective, train_single=float(singleton_objectives[selected]), chosen_single=job["peers"][selected])
    metrics["gamma"] = metrics["e_single"] - metrics["e_group"]
    metrics["relative_gamma_percent"] = 100 * metrics["gamma"] / metrics["e_single"] if metrics["e_single"] > 1e-14 else np.nan
    kkt = None
    if job["fit_loss"] == "mse":
        gradient = 2 * np.sum(fit_mass[..., None] * x[train] * fitting_residual[..., None], axis=(0, 1))
        kkt = float(w @ gradient - gradient.min())
    return metrics, arrays, kkt


def _key(job):
    removed = job.get("removed_peer")
    if pd.isna(removed):
        removed = None
    return (job["panel"], job["target"], job["family"], int(job["split_seed"]), job["fit_loss"], job["role"], removed)


def verify_coefficients(responses, coefficients, jobs, audit):
    audit.check(len(coefficients) == 5840 and not coefficients.job_id.duplicated().any() and set(coefficients.job_id) == set(jobs),
                "Exact 5,840 nontrivial point-fit job inventory")
    metrics = {}
    weights = {}
    by_key = {_key(job): job_id for job_id, job in jobs.items()}
    backends = defaultdict(set)
    for row in coefficients.to_dict("records"):
        job_id = row["job_id"]
        job = jobs[job_id]
        for field in ("panel", "target", "family", "split_seed", "fit_loss", "role", "response"):
            if row[field] != job[field]:
                raise ValueError(f"Coefficient metadata differs from exact job inventory: {field}")
        removed = None if pd.isna(row["removed_peer"]) else row["removed_peer"]
        if removed != job["removed_peer"]:
            raise ValueError("Coefficient removal identity differs from exact job inventory")
        if json_list(row["peer_ids"]) != job["peers"]:
            raise ValueError("Saved available-peer order changed")
        if not bool(row["success"]):
            raise ValueError("Unsuccessful point fit")
        w = np.asarray(row["weights"], dtype=float)
        raw = np.asarray(row["raw_weights"], dtype=float)
        if raw.shape != w.shape or not np.isfinite(raw).all() or raw.min() < -1e-7 or abs(raw.sum() - 1) > 1e-7:
            raise ValueError("Raw simplex infeasibility exceeds prespecified tolerance")
        value, _, kkt = replay_fit(responses, job, w)
        audit.close(row["train_objective"], value["train_group"], "point direct training objective", TRAIN_ATOL)
        audit.close(row["solver_objective"], value["train_group"], "point solver objective", TRAIN_ATOL)
        audit.close(row["raw_min_weight"], raw.min(), "point raw minimum coefficient", PROBABILITY_ATOL)
        audit.close(row["raw_sum_error"], abs(raw.sum() - 1), "point raw simplex sum error", PROBABILITY_ATOL)
        _, x = responses.arrays(job)
        train = responses.fit_indices[int(job["split_seed"])]
        repair = float(np.max(np.abs(np.einsum("...p,p->...", x[train], w - raw))))
        if repair > 1e-6 or float(row["repair_prediction_max_abs"]) > 1e-6:
            raise ValueError("Coefficient repair changed fitting predictions beyond 1e-6")
        if value["train_group"] > value["train_single"] + TRAIN_ATOL:
            raise ValueError("Group fitting loss exceeds available fitting-selected single")
        if kkt is not None and (kkt > 1e-7 or kkt < -1e-10):
            raise ValueError(f"Independent MSE KKT gap exceeds fixed tolerance: {kkt}")
        backends[job["fit_loss"]].add(row["solver_backend"])
        weights[job_id], metrics[job_id] = w, value
    for loss, names in backends.items():
        allowed = ({"gram_active_set_v1_with_fixed_slsqp_recovery/active_set",
                    "gram_active_set_v1_with_fixed_slsqp_recovery/slsqp_recovery"} if loss == "mse" else
                   {"scipy_highs_sparse_mae_threads1"} if loss == "mae" else {"scipy_highs_sparse_tv_threads1"})
        audit.check(bool(names) and names <= allowed, f"Fixed {loss} solver algorithm and declared branches", backends=sorted(names))
    for job_id, job in jobs.items():
        if job["role"] == "removal":
            full_id = by_key[(job["panel"], job["target"], job["family"], job["split_seed"], job["fit_loss"], "group", None)]
            if metrics[job_id]["train_group"] + TRAIN_ATOL < metrics[full_id]["train_group"]:
                raise ValueError("Removal refit violates training nesting")
        elif job["panel"] != "expanded11":
            full_id = by_key[("expanded11", job["target"], job["family"], job["split_seed"], job["fit_loss"], "group", None)]
            if metrics[full_id]["train_group"] > metrics[job_id]["train_group"] + TRAIN_ATOL:
                raise ValueError("Roster expansion violates training nesting")
    audit.check(True, "All fitted coefficients, direct objectives, MSE optimality and nested fitting comparisons replayed")
    return metrics, weights, by_key


def _partner_error(responses, target, partner, family, seed, loss, draw=None):
    source = responses.p if loss == "tv" else responses.y
    conditions = [responses.cindex[c] for c in FAMILY_CONDITIONS[family]]
    residual = abs(source[responses.mindex[target]][:, conditions] - source[responses.mindex[partner]][:, conditions])
    if loss == "tv":
        residual = .5 * residual.sum(axis=-1)
    ids = responses.eval_indices[int(seed)]
    mass = np.ones(len(ids)) if draw is None else np.asarray(draw)[ids]
    return float(np.average(residual[ids] @ CONDITION_MASS, weights=mass))


def verify_tables(responses, tables, jobs, metrics, by_key, audit):
    expected_sets = {}
    for name, frame in tables.items():
        audit.check(len(frame) == TABLE_COUNTS[name], f"{name} expected rows", rows=len(frame))
        if not frame.completion_state.eq("complete").all():
            raise ValueError(f"Incomplete point table: {name}")
    for name, loss_filter in (("scalar_main_split_results", {"mse", "mae"}), ("vector_main_split_results", {"tv"})):
        table = tables[name]
        expected = {j for j, row in jobs.items() if row["role"] == "group" and row["fit_loss"] in loss_filter}
        audit.check(set(table.job_id) == expected and not table.job_id.duplicated().any(), f"{name} exact keys")
        for row in table.to_dict("records"):
            job, replay = jobs[row["job_id"]], metrics[row["job_id"]]
            for field in ("panel", "target", "family", "split_seed", "fit_loss", "response"):
                if row[field] != job[field]:
                    raise ValueError("Main table job metadata mismatch")
            if json_list(row["peer_ids"]) != job["peers"] or row["chosen_single"] != replay["chosen_single"]:
                raise ValueError("Main table available peers or fitting-only single selection mismatch")
            for metric, value in replay.items():
                if metric != "chosen_single" and metric in row:
                    audit.close(row[metric], value, f"main/{metric}")
    frame = tables["scalar_removal_split_results"]
    expected = {j for j, row in jobs.items() if row["role"] == "removal" and row["fit_loss"] != "tv"}
    audit.check(set(frame.removed_job_id) == expected and not frame.removed_job_id.duplicated().any(), "All scalar removed-peer fits represented exactly once")
    for row in frame.to_dict("records"):
        job = jobs[row["removed_job_id"]]
        full = by_key[("expanded11", job["target"], job["family"], job["split_seed"], job["fit_loss"], "group", None)]
        if row["full_job_id"] != full or row["removed_peer"] != job["removed_peer"]:
            raise ValueError("Scalar removal identities mismatch")
        e_full, e_without = metrics[full]["e_group"], metrics[row["removed_job_id"]]["e_group"]
        for field, value in {"e_full": e_full, "e_without": e_without, "removal_effect": e_without - e_full}.items():
            audit.close(row[field], value, f"scalar_removal/{field}")
    for name, losses in (("scalar_directional_pairs", ["mse", "mae"]), ("vector_directional_removals", ["tv"])):
        expected = {(pair[a], pair[b], family, seed, loss) for pair in responses.config["declared_pairs"]
                    for a, b in (("a", "b"), ("b", "a")) for family in responses.config["families"]
                    for seed in responses.config["split_seeds"] for loss in losses}
        seen = set()
        for row in tables[name].to_dict("records"):
            t, p, f, s, loss = (row[k] for k in ("target", "partner", "family", "split_seed", "fit_loss"))
            key = (t, p, f, int(s), loss)
            if key in seen:
                raise ValueError("Repeated directional key")
            seen.add(key)
            full = by_key[("expanded11", t, f, int(s), loss, "group", None)]
            removed = by_key[("expanded11", t, f, int(s), loss, "removal", p)]
            reverse = by_key[("expanded11", p, f, int(s), loss, "group", None)]
            if row["full_job_id"] != full or row["removed_job_id"] != removed:
                raise ValueError("Directional group/removal job identities mismatch")
            partner = _partner_error(responses, t, p, f, s, loss)
            audit.close(partner, _partner_error(responses, p, t, f, s, loss), "Direct partner distance symmetry", PROBABILITY_ATOL)
            values = {"partner_error": partner, "e_group": metrics[full]["e_group"],
                      "extra_beyond_partner": partner - metrics[full]["e_group"], "e_without": metrics[removed]["e_group"],
                      "removal_effect": metrics[removed]["e_group"] - metrics[full]["e_group"],
                      "directional_extra_difference": metrics[reverse]["e_group"] - metrics[full]["e_group"]}
            for field, value in values.items():
                audit.close(row[field], value, f"{name}/{field}")
        audit.check(seen == expected, f"{name} all declared direction keys")
    for name, losses in (("scalar_roster_expansion", ["mse", "mae"]), ("vector_roster_expansion", ["tv"])):
        expected = {(t, f, s, loss) for t in responses.models for f in responses.config["families"]
                    for s in responses.config["split_seeds"] for loss in losses}
        seen = set()
        for row in tables[name].to_dict("records"):
            t, f, s, loss = (row[k] for k in ("target", "family", "split_seed", "fit_loss"))
            key = (t, f, int(s), loss)
            if key in seen:
                raise ValueError("Repeated roster-expansion key")
            seen.add(key)
            comparator = "baseline8" if t in responses.config["original_roster"] else "new_on_old8"
            base = by_key[(comparator, t, f, int(s), loss, "group", None)]
            full = by_key[("expanded11", t, f, int(s), loss, "group", None)]
            if row["comparator_panel"] != comparator or row["comparator_job_id"] != base or row["expanded_job_id"] != full:
                raise ValueError("Roster comparison identities mismatch")
            values = {"comparator_error": metrics[base]["e_group"], "expanded_error": metrics[full]["e_group"],
                      "large_peer_gain": metrics[base]["e_group"] - metrics[full]["e_group"]}
            for field, value in values.items():
                audit.close(row[field], value, f"{name}/{field}")
        audit.check(seen == expected, f"{name} exact comparison keys")
    audit.check(True, "All main, directional, removal and roster-expansion point metrics independently replayed")


def _verify_per_question_metadata(group, job, *, run_id, input_fingerprint):
    expected_metadata = {k: job[k] for k in ("panel", "target", "family", "split_seed", "response", "fit_loss")}
    expected_metadata.update(run_id=run_id, input_fingerprint=input_fingerprint, completion_state="complete",
                             evaluation_metric="tv" if job["fit_loss"] == "tv" else "mae")
    for field, expected in expected_metadata.items():
        if field not in group or not group[field].eq(expected).all():
            raise ValueError("Per-question row metadata mismatch: " + field)
    if "peer_ids" not in group or group.peer_ids.nunique(dropna=False) != 1 or json_list(group.peer_ids.iloc[0]) != job["peers"]:
        raise ValueError("Per-question available-peer identity mismatch")


def verify_per_question(responses, path, jobs, weights, audit, *, run_id, input_fingerprint):
    identifiers = list(jobs)
    job_index = {j: i for i, j in enumerate(identifiers)}
    # A compact occupancy mask prevents silent duplicate/lost physical observations.
    seen = np.zeros((len(jobs), 560, 7), dtype=bool)
    @lru_cache(maxsize=64)
    def losses(job_id):
        return replay_fit(responses, jobs[job_id], weights[job_id])[1]
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        for job_id, group in frame.groupby("job_id", sort=False):
            if job_id not in jobs:
                raise ValueError("Unexpected per-question fit identifier")
            job = jobs[job_id]
            _verify_per_question_metadata(group, job, run_id=run_id, input_fingerprint=input_fingerprint)
            q = np.array([responses.qindex[str(v)] for v in group.base_question_id])
            c = np.array([responses.cindex[v] for v in group.semantic_condition])
            local = {v: i for i, v in enumerate(FAMILY_CONDITIONS[job["family"]])}
            local_c = np.array([local[v] for v in group.semantic_condition])
            if not np.isin(q, responses.eval_indices[job["split_seed"]]).all():
                raise ValueError("Fitting question appears in saved held-out losses")
            packed = q * 7 + c
            ji = job_index[job_id]
            if len(np.unique(packed)) != len(packed) or seen[ji, q, c].any():
                raise ValueError("Duplicated physical held-out question/condition/fit")
            seen[ji, q, c] = True
            audit.close(group.condition_mass.to_numpy(), CONDITION_MASS[local_c], "per-question condition mass", PROBABILITY_ATOL)
            expected = losses(job_id)
            for column in ("absolute_error", "squared_error", "total_variation", "top_option_agreement",
                           "single_absolute_error", "single_squared_error", "single_total_variation", "single_top_option_agreement"):
                values = (expected[column][q, local_c] if column in expected and
                          (not column.startswith("single_") or job["role"] == "group") else np.full(len(group), np.nan))
                audit.close(group[column].to_numpy(), values, "per-question/" + column)
            rows += len(group)
    audit.check(rows == 5840 * 280 * 4 and np.all(seen.sum(axis=(1, 2)) == 1120),
                "All 6,540,800 held-out physical observations replayed exactly once", rows=rows)


def verify_seed_and_rotations(responses, root, plan, audit):
    config = responses.config
    for name, expected in (("mmlu_pro_selected_560.jsonl", config["questions"]["sha256"]),
                           ("permutation_averaged_semantic_scores.parquet", config["baseline_cache"]["sha256"])):
        audit.check(sha256(root / "inputs" / name) == sha256(plan / "seed_inputs" / name) == expected,
                    "Unchanged byte-identical supplied input: " + name)
    baseline = pd.read_parquet(root / "inputs/permutation_averaged_semantic_scores.parquet")
    index = ["model_id", "base_question_id", "semantic_condition"]
    baseline["base_question_id"] = baseline.base_question_id.astype(str)
    merged = responses.frame.copy()
    merged["base_question_id"] = merged.base_question_id.astype(str)
    old = merged[merged.model_id.isin(config["original_roster"])].set_index(index).sort_index()
    baseline = baseline.set_index(index).sort_index()
    audit.check(old.index.equals(baseline.index), "Original eight-model semantic keys unchanged")
    for column in baseline:
        left, right = old[column].tolist(), baseline[column].tolist()
        for a, b in zip(left, right):
            if isinstance(b, (list, np.ndarray)):
                if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
                    raise ValueError("Original probability-vector bytes changed after merging")
            elif pd.isna(a) and pd.isna(b):
                continue
            elif a != b:
                raise ValueError(f"Original semantic metadata changed: {column}")
    audit.check(True, "All original probability vectors and metadata are exactly retained")
    raw = pd.read_parquet(root / "data/new_rotation_scores.parquet")
    audit.check(len(raw) == 110838 and set(raw.model_id) == {m["id"] for m in config["new_models"]}, "All 110,838 new raw rotation rows")
    audit.check(not raw.duplicated(["model_id", "prompt_id"]).any(), "No duplicate scored prompt per model")
    lock = json.loads((root / "model_lock.json").read_text())
    new_revisions = {r["id"]: r["revision"] for r in lock["models"]}
    audit.check(lock.get("status") == "complete" and set(new_revisions) == set(raw.model_id)
                and all(re.fullmatch(r"[0-9a-f]{40}", r) for r in new_revisions.values()), "Three frozen full model revisions")
    expected_revisions = {**config["original_model_revisions"], **new_revisions}
    for model, revision in expected_revisions.items():
        if set(merged.loc[merged.model_id == model, "model_revision"]) != {revision}:
            raise ValueError("Merged model revision differs from checkpoint lock")
    if not (root / "data/permutations/score_execution_manifest.jsonl").is_file():
        raise ValueError("Full execution manifest is required for raw-score replay")
    manifest = [json.loads(line) for line in (root / "data/permutations/score_execution_manifest.jsonl").read_text().splitlines() if line.strip()]
    canonical = {r["prompt_id"]: r for r in manifest}
    if len(canonical) != 36946:
        raise ValueError("Unexpected execution manifest coverage")
    for model, frame in raw.groupby("model_id", sort=False):
        audit.check(len(frame) == 36946 and int(frame.option_count.sum()) == 357196 and
                    int((frame.permutation_id == 0).sum()) == 3920 and set(frame.prompt_id) == set(canonical),
                    "Complete raw model scoring coverage: " + model)
        if set(frame.model_revision) != {new_revisions[model]}:
            raise ValueError("Raw score model revision mismatch")
        for row in frame.to_dict("records"):
            source = canonical[row["prompt_id"]]
            q, c = responses.qindex[str(row["base_question_id"])], responses.cindex[row["semantic_condition"]]
            n, rotation = int(responses.counts[q]), int(row["permutation_id"])
            if row["prompt_id"] != f"{responses.qids[q]}__{row['semantic_condition']}__cyclic_r{rotation:02d}":
                raise ValueError("Raw prompt identity does not match its cyclic group")
            if int(row["option_count"]) != n or not 0 <= rotation < n or int(row["gold_semantic_option"]) != responses.gold[q]:
                raise ValueError("Raw question metadata mismatch")
            if row["canonical_prompt_hash"] != source["canonical_content_sha256"]:
                raise ValueError("Raw canonical prompt hash mismatch")
            logits = np.asarray(row["candidate_sequence_log_likelihoods"], dtype=float)
            visible = np.asarray(row["visible_label_probabilities"], dtype=float)
            semantic = np.asarray(row["semantic_remapped_probabilities"], dtype=float)
            mapping = np.asarray(row["visible_to_semantic"], dtype=int)
            if logits.shape != (n,) or visible.shape != (n,) or semantic.shape != (n,) or not np.isfinite(logits).all():
                raise ValueError("Incomplete full-continuation scoring vectors")
            if not np.array_equal(mapping, (np.arange(n) - rotation) % n):
                raise ValueError("Raw semantic permutation mapping mismatch")
            normalized = np.exp(logits - logits.max())
            normalized /= normalized.sum()
            audit.close(visible, normalized, "raw normalized full-continuation likelihood", PROBABILITY_ATOL)
            remapped = np.zeros(n)
            remapped[mapping] = visible
            audit.close(semantic, remapped, "raw visible-to-semantic remapping", PROBABILITY_ATOL)
            if json_list(row["candidate_continuations"]) != [f" ({label})" for label in "ABCDEFGHIJ"[:n]]:
                raise ValueError("Candidate convention changed")
            if np.asarray(row["candidate_token_counts"]).shape != (n,) or min(row["candidate_token_counts"]) < 1:
                raise ValueError("Raw full-continuation token counts missing")
        for (qid, condition), group in frame.groupby(["base_question_id", "semantic_condition"], sort=False):
            q, c, m = responses.qindex[str(qid)], responses.cindex[condition], responses.mindex[model]
            n = int(responses.counts[q])
            if sorted(group.permutation_id) != list(range(n)):
                raise ValueError("Missing cyclic rotation before semantic averaging")
            actual = np.stack(group.sort_values("permutation_id").semantic_remapped_probabilities).mean(axis=0)
            audit.close(actual, responses.p[m, q, c, :n], "complete cyclic probability averaging", PROBABILITY_ATOL)
    audit.check(True, "All new raw likelihood normalizations, semantic mappings and cyclic averages independently replayed")


def relocated_run_path(recorded, root):
    """Resolve a provenance label inside the supplied run, never at old paths."""
    recorded = Path(recorded)
    if not recorded.is_absolute():
        relative = recorded
    elif recorded.is_relative_to(root):
        relative = recorded.relative_to(root)
    else:
        parts = recorded.parts
        candidates = [i for i, part in enumerate(parts) if part == "xfqa_large_models" and i + 2 < len(parts)]
        if not candidates:
            raise ValueError("Cannot relocate provenance path into this XfqA run: " + str(recorded))
        relative = Path(*parts[candidates[-1] + 2:])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe provenance path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Provenance path escapes the supplied run")
    return path


def verify_scoring_proofs(repo, root, plan, audit):
    assembly = json.loads((root / "status/assembly_complete.json").read_text())
    if assembly.get("status") != "complete" or assembly.get("real_three_model_scoring_complete") is not True:
        raise ValueError("Completed real three-model scoring proof is required")
    for name, expected in (("new_rotation_rows", 110838), ("new_averaged_vectors", 11760), ("merged_averaged_vectors", 43120)):
        if assembly.get(name) != expected:
            raise ValueError("Assembled score coverage differs: " + name)
    for item in assembly["artifacts"].values():
        path = relocated_run_path(item["path"], root)
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Assembly artifact checksum changed")
    provenance = json.loads((root / "provenance/measurement_provenance.json").read_text())
    lock = json.loads((root / "model_lock.json").read_text())
    config = json.loads((plan / "config/experiment.json").read_text())
    if provenance.get("status") != "complete" or provenance["model_lock_sha256"] != sha256(root / "model_lock.json") or provenance["protocol_sha256"] != sha256(plan / "config/experiment.json"):
        raise ValueError("Measurement provenance identity mismatch")
    inventory = provenance["new_measurements"]["models"]
    if {v["model_id"] for v in inventory} != {r["id"] for r in lock["models"]} or len(inventory) != 3:
        raise ValueError("Measurement provenance model inventory mismatch")
    for record in inventory:
        identity_path = relocated_run_path(record["identity_path"], root)
        complete_path = relocated_run_path(record["complete_path"], root)
        if sha256(identity_path) != record["identity_sha256"] or sha256(complete_path) != record["complete_sha256"]:
            raise ValueError("Scoring proof checksum changed")
        identity = json.loads(identity_path.read_text())
        fingerprint = identity.pop("input_fingerprint")
        calculated = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if fingerprint != calculated or fingerprint != record["input_fingerprint"]:
            raise ValueError("Scoring identity self-checksum mismatch")
        expected = {"precision": "bfloat16", "candidate_definition": "summed_full_continuation_log_likelihood",
                    "all_rotations_scored": True, "free_text_generation": False, "quantization": False, "truncation": False,
                    "measurement_origin": "fresh_endpoint_scoring"}
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError("Scoring definition differs from fixed full-continuation protocol")
        for relative, digest in identity["source_sha256"].items():
            if sha256(repo / relative) != digest:
                raise ValueError("Original scientific scorer dependency changed: " + relative)
        if sha256(repo / "experiments/llm/analysis/xfqa_large_models/scoring.py") != identity["new_adapter_sha256"]:
            raise ValueError("Measured new scoring adapter changed")
        if sha256(repo / "experiments/llm/analysis/xwvg_reference_selection/recovery/score_endpoints.py") != identity["original_driver_sha256"]:
            raise ValueError("Original scoring driver changed")
        complete = json.loads(complete_path.read_text())
        pilot_path = complete_path.with_name(complete_path.name.removesuffix("_complete.json") + "_pilot.json")
        pilot = json.loads(pilot_path.read_text())
        if (complete.get("status") != "complete" or complete.get("phase") != "score" or
                complete.get("input_fingerprint") != fingerprint or complete.get("completed_rows") != 36946 or
                complete.get("candidate_continuations") != 357196 or complete.get("frozen_original_score_function") is not True or
                complete.get("pilot_report_sha256") != sha256(pilot_path)):
            raise ValueError("Formal model scoring completion proof is incomplete")
        expected_class = next(m["expected_model_class"] for m in config["new_models"] if m["id"] == record["model_id"])
        if complete.get("actual_model_class") != expected_class or pilot.get("status") != "complete" or pilot.get("phase") != "pilot" or not pilot.get("worst_case_probe_passed") or not pilot.get("repeatability", {}).get("passed"):
            raise ValueError("Fixed pilot, native model class or maximum-workload validation failed")
        if len(pilot["formal_pilot_ids"]) != 42:
            raise ValueError("Fixed pilot does not contain the prescribed 42 prompts")
    raw_rows = 0
    prompt_sets = defaultdict(set)
    for record in provenance["raw_shards"]:
        paths = {key: relocated_run_path(record[key], root) for key in ("path", "metadata_path", "commit_path")}
        for key, digest_key in (("path", "sha256"), ("metadata_path", "metadata_sha256"), ("commit_path", "commit_sha256")):
            if sha256(paths[key]) != record[digest_key]:
                raise ValueError("Committed raw scoring shard/proof changed")
        metadata = json.loads(paths["metadata_path"].read_text())
        commit = json.loads(paths["commit_path"].read_text())
        if commit.get("complete") is not True or commit.get("sha256") != record["sha256"] or metadata["sha256"] != record["sha256"]:
            raise ValueError("Raw score shard is not atomically committed")
        if pq.read_metadata(paths["path"]).num_rows != record["rows"] or record["rows"] != len(metadata["prompt_ids"]):
            raise ValueError("Raw shard row coverage mismatch")
        seen = prompt_sets[metadata["model_id"]]
        ids = set(metadata["prompt_ids"])
        if len(ids) != len(metadata["prompt_ids"]) or seen & ids:
            raise ValueError("Repeated scoring prompts across committed shards")
        seen.update(ids)
        raw_rows += record["rows"]
    audit.check(raw_rows == 110838 and len(prompt_sets) == 3 and all(len(ids) == 36946 for ids in prompt_sets.values()),
                "Real measured scoring identities, pilots, original scorer hashes and all atomic raw shard proofs verified")


def _verify_preservation(repo_root, output_root, audit, portable):
    path = output_root / "validation/preservation_baseline.json"
    baseline = json.loads(path.read_text())
    audit.check(baseline.get("schema") == "xfqa_preservation_v1" or baseline.get("schema_version") == "xfqa_preservation_v1",
                "Preservation baseline schema")
    if portable:
        audit.check(True, "Original external files are outside portable replay; original preservation audit retained in bundle")
        return
    for entry in baseline["files"]:
        path = repo_root / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Original paper/XWVG preservation mismatch: " + entry["path"])
    audit.check(len(baseline["files"]) == baseline["file_count"], "Original paper and completed XWVG files remain unchanged", files=baseline["file_count"])


def verify_analysis_identity(repo, plan, root, manifest, audit):
    identity = manifest["identity"]
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(serialized.encode()).hexdigest() != manifest["input_fingerprint"]:
        raise ValueError("Analysis identity self-checksum mismatch")
    for key, path in (("cache_sha256", root / "data/permutation_averaged_semantic_scores.parquet"),
                      ("questions_sha256", root / "inputs/mmlu_pro_selected_560.jsonl"),
                      ("config_sha256", plan / "config/experiment.json"),
                      ("original_input_config_sha256", plan / "config/original_xwvg_input_config.json"),
                      ("model_lock_sha256", root / "model_lock.json")):
        if sha256(path) != identity[key]:
            raise ValueError("Analysis identity input checksum mismatch: " + key)
    for recorded, digest in identity["source_hashes"].items():
        parts = Path(recorded).parts
        if any(parts[i:i+2] == ("outputs", "xfqa_large_models") for i in range(len(parts)-1)):
            actual = relocated_run_path(recorded, root)
        else:
            indices = [i for i, part in enumerate(parts) if part == "experiments"]
            if not indices:
                raise ValueError("Cannot relocate recorded scientific source: " + recorded)
            actual = repo / Path(*parts[indices[0]:])
        if not actual.is_file() or sha256(actual) != digest:
            raise ValueError("Analysis scientific source checksum mismatch: " + recorded)
    report = json.loads((root / "status/report_complete.json").read_text())
    if report["input_identity"]["reporting_source_sha256"] != sha256(Path(__file__).with_name("reporting.py")):
        raise ValueError("Report-generation source differs from its recorded version")
    audit.check(True, "Analysis identity self-checksum, exact inputs, source hashes and report source verified")


def run_verify(repo_root, plan_root, output_root, resume=False, *, portable=False, audit_output=None):
    started = time.monotonic()
    repo, plan, root = map(Path, (repo_root, plan_root, output_root))
    audit = Audit()
    result = {"schema": "xfqa_numerical_replay_v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "portable": portable, "verifier_source_sha256": sha256(Path(__file__)),
              "tolerances": {"heldout_metric_absolute": METRIC_ATOL, "training_objective_absolute": TRAIN_ATOL,
                             "raw_probability_absolute": PROBABILITY_ATOL}}
    destination = Path(audit_output) if audit_output else (None if portable else root / "validation/numerical_replay.json")
    try:
        analysis = root / "analysis"
        config = json.loads((plan / "config/experiment.json").read_text())
        verify_scoring_proofs(repo, root, plan, audit)
        responses = Responses(config, root / "inputs/mmlu_pro_selected_560.jsonl", root / "data/permutation_averaged_semantic_scores.parquet", audit)
        verify_seed_and_rotations(responses, root, plan, audit)
        saved_splits = json.loads((analysis / "split_ids.json").read_text())
        for seed, sides in responses.splits.items():
            for side in ("fitting_ids", "evaluation_ids"):
                if saved_splits[str(seed)][side] != sides[side]:
                    raise ValueError("Original fixed question split changed")
        audit.check(True, "All ten fixed question splits independently reproduced")
        jobs = expected_jobs(config)
        coefficients = pd.read_parquet(analysis / "point_fit_coefficients.parquet")
        tables = {name: read_table_csv(analysis / f"{name}.csv") for name in TABLE_COUNTS}
        fingerprints = set(coefficients.input_fingerprint.astype(str))
        for table in tables.values():
            fingerprints.update(table.input_fingerprint.astype(str))
        audit.check(len(fingerprints) == 1, "One input fingerprint in all point tables and coefficients")
        manifest = json.loads((analysis / "analysis_manifest.json").read_text())
        audit.check(manifest["input_fingerprint"] == next(iter(fingerprints)) and
                    manifest["identity"]["synthetic_fixture"] is False and
                    manifest["identity"]["scope"] == "complete_expanded_experiment", "Full empirical analysis identity")
        verify_analysis_identity(repo, plan, root, manifest, audit)
        metrics, weights, by_key = verify_coefficients(responses, coefficients, jobs, audit)
        verify_tables(responses, tables, jobs, metrics, by_key, audit)
        verify_per_question(responses, analysis / "per_question_losses.parquet", jobs, weights, audit,
                            run_id=tables["scalar_main_split_results"].run_id.iloc[0], input_fingerprint=next(iter(fingerprints)))
        verify_uncertainty(responses, root, tables, jobs, metrics, weights, by_key, audit)
        verify_report_outputs(root, tables, audit)
        _verify_preservation(repo, root, audit, portable)
        result.update(status="passed", input_fingerprint=next(iter(fingerprints)),
                      counts={"new_rotation_rows": 110838, "merged_vectors": 43120, "point_fits": 5840,
                              "per_question_rows": 6540800, "primary_refit_replicates": 1000,
                              "primary_refit_fits": 100000, "secondary_replicates": 2000,
                              **{name: len(frame) for name, frame in tables.items()}},
                      checks=audit.checks, max_absolute_errors=dict(audit.max_errors), wall_seconds=time.monotonic() - started)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), checks=audit.checks,
                      max_absolute_errors=dict(audit.max_errors), wall_seconds=time.monotonic() - started)
        if destination:
            write_json(destination, result)
        raise
    if destination:
        write_json(destination, result)
    return result


INTERVAL_METRICS = {
    "scalar_main_split_results": ["e_single", "e_group", "gamma", "relative_gamma_percent", "mse_single", "mse_group"],
    "scalar_removal_split_results": ["e_full", "e_without", "removal_effect"],
    "scalar_directional_pairs": ["partner_error", "e_group", "extra_beyond_partner", "e_without", "removal_effect", "directional_extra_difference"],
    "scalar_roster_expansion": ["comparator_error", "expanded_error", "large_peer_gain"],
    "vector_main_split_results": ["e_single", "e_group", "gamma", "relative_gamma_percent", "single_top_option_agreement", "group_top_option_agreement", "agreement_gain"],
    "vector_directional_removals": ["partner_error", "e_group", "extra_beyond_partner", "e_without", "removal_effect", "directional_extra_difference"],
    "vector_roster_expansion": ["comparator_error", "expanded_error", "large_peer_gain"],
}
INTERVAL_KEYS = ["panel", "target", "family", "fit_loss", "partner", "removed_peer", "comparator_panel"]


def draw_multiplicities(responses, replicate, seed):
    """Independent reproduction of the archived stable-seed question bootstrap."""
    rng = np.random.default_rng(_stable_u64(seed, "common-global", replicate) % (2**32))
    counts = np.zeros(560, dtype=np.int16)
    for category in sorted(set(responses.categories)):
        qids = sorted(q for q, c in zip(responses.qids, responses.categories) if c == category)
        chosen, frequencies = np.unique(rng.choice(qids, size=len(qids), replace=True), return_counts=True)
        for q, count in zip(chosen, frequencies):
            counts[responses.qindex[q]] = count
    if counts.sum() != 560:
        raise ValueError("Independent bootstrap draw lost questions")
    for side in (responses.fit_indices, responses.eval_indices):
        if any(counts[ids].sum() == 0 for ids in side.values()):
            raise ValueError("Empty weighted bootstrap split side")
    return counts


def _contrast_descriptors(config, tables, primary=False):
    descriptors = {}
    allowed_targets = config["uncertainty"]["primary_refit"]["group_targets"]
    allowed_directions = {tuple(v) for v in config["uncertainty"]["primary_refit"]["removal_directions"]}
    for name, frame in tables.items():
        if primary:
            if name == "scalar_main_split_results":
                frame = frame[(frame.panel == "expanded11") & (frame.fit_loss == "mse") & frame.target.isin(allowed_targets)]
            elif name == "scalar_directional_pairs":
                mask = [(t, p) in allowed_directions for t, p in zip(frame.target, frame.partner)]
                frame = frame[(frame.fit_loss == "mse") & np.array(mask)]
            else:
                continue
        keys = [k for k in INTERVAL_KEYS if k in frame]
        columns = INTERVAL_METRICS[name][:4] if primary and name == "scalar_main_split_results" else INTERVAL_METRICS[name]
        for _, group in frame.groupby(keys, sort=True, dropna=False):
            if len(group) != 10 or group.split_seed.nunique() != 10:
                raise ValueError("An interval lacks ten distinct fixed splits")
            first = group.iloc[0]
            for metric in columns:
                key = {"table": name, **{k: str(first[k]) if k in group else "" for k in INTERVAL_KEYS}, "metric": metric}
                cid = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                descriptors[cid] = {**key, "point": float(group[metric].mean(skipna=False)), "rows": group.sort_values("split_seed").to_dict("records")}
    return descriptors


def verify_relative_omissions(path, descriptors, ordered_ids, values, audit):
    """Only numerically undefined relative percentages may be omitted."""
    evidence = json.loads(Path(path).read_text())
    records = {r["contrast_id"]: r for r in evidence["omissions"]}
    if len(records) != len(evidence["omissions"]) or evidence.get("absolute_contrasts_retained") is not True:
        raise ValueError("Invalid relative-percentage omission record")
    expected = set()
    keep = []
    for index, cid in enumerate(ordered_ids):
        d = descriptors[cid]
        undefined_point = not np.isfinite(d["point"])
        undefined_replicates = ~np.isfinite(values[:, index])
        if undefined_point or undefined_replicates.any():
            if d["metric"] != "relative_gamma_percent":
                raise ValueError("An absolute primary/secondary contrast is undefined")
            expected.add(cid)
            if cid not in records:
                raise ValueError("Undefined relative percentage lacks an omission record")
            record = records[cid]
            if undefined_point:
                actual = [int(r["split_seed"]) for r in d["rows"] if not np.isfinite(r[d["metric"]])]
                if record.get("stage") != "point" or record.get("undefined_split_seeds") != actual:
                    raise ValueError("Relative point omission does not identify its actual zero-denominator splits")
            elif record.get("stage") != "bootstrap" or record.get("undefined_replicates") != (np.flatnonzero(undefined_replicates) + 1).tolist():
                raise ValueError("Relative bootstrap omission does not identify its actual undefined replicates")
        else:
            keep.append(index)
    audit.check(set(records) == expected, "Only recorded undefined relative percentages omitted: " + Path(path).name,
                omitted=len(expected))
    ids = [ordered_ids[i] for i in keep]
    return {cid: descriptors[cid] for cid in ids}, ids, values[:, keep]


def _verify_interval_table(intervals, descriptors, values, ordered_ids, expected_replicates, method, audit):
    audit.check(set(intervals.contrast_id) == set(descriptors) == set(ordered_ids) and not intervals.contrast_id.duplicated().any(),
                method + " exact reported contrast inventory", contrasts=len(descriptors))
    if values.shape != (expected_replicates, len(ordered_ids)) or not np.isfinite(values).all():
        raise ValueError("Uncertainty replicate matrix dimensions or values are invalid")
    byid = intervals.set_index("contrast_id")
    for column, cid in enumerate(ordered_ids):
        row, descriptor = byid.loc[cid], descriptors[cid]
        if int(row.replicates) != expected_replicates or row.method != method:
            raise ValueError("Wrong uncertainty method or replicate count")
        for key in ["table", "metric", *INTERVAL_KEYS]:
            actual = "" if pd.isna(row[key]) else str(row[key])
            if actual != descriptor[key]:
                raise ValueError("Uncertainty contrast metadata mismatch: " + key)
        audit.close(row.point, descriptor["point"], method + " interval point")
        low, high = np.percentile(values[:, column], [2.5, 97.5])
        audit.close([row.low, row.high], [low, high], method + " percentile endpoints")
        refit = method == "primary_refit"
        if bool(row.includes_refitting) != refit or bool(row.includes_single_reselection) != refit:
            raise ValueError("Refit versus frozen-fit uncertainty flag mismatch")
        if not refit and "excludes fitting/selection uncertainty" not in row.uncertainty_scope:
            raise ValueError("Frozen-fit interval omits required scope disclosure")


def verify_primary_refits(responses, analysis, tables, jobs, by_key, audit):
    cfg = responses.config["uncertainty"]["primary_refit"]
    descriptors = _contrast_descriptors(responses.config, tables, primary=True)
    audit.check(len(descriptors) == 48, "48 prespecified primary refit contrasts")
    ordered_ids = list(descriptors)
    values = np.full((1000, len(ordered_ids)), np.nan)
    directions = {tuple(x) for x in cfg["removal_directions"]}
    primary_ids = {jid for jid, j in jobs.items() if j["panel"] == "expanded11" and j["fit_loss"] == "mse" and
                   ((j["role"] == "group" and j["target"] in cfg["group_targets"]) or
                    (j["role"] == "removal" and (j["target"], j["removed_peer"]) in directions))}
    audit.check(len(primary_ids) == 100, "100 independently enumerated refits per primary replicate")
    seen_replicates = set()
    fit_count = 0
    branches = Counter()
    paths = sorted((analysis / "checkpoints/primary_refit").glob("chunk_*"))
    audit.check(len(paths) == 50, "50 committed primary bootstrap chunks")
    for path in paths:
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("status") != "complete":
            raise ValueError("Incomplete primary-refit chunk")
        for name, digest in meta["artifact_hashes"].items():
            if sha256(path / name) != digest:
                raise ValueError("Primary-refit chunk checksum mismatch: " + name)
        with np.load(path / "multiplicities.npz", allow_pickle=False) as saved:
            replicate_ids = saved["replicates"].astype(int).tolist()
            question_ids = saved["question_ids"].astype(str).tolist()
            if len(set(question_ids)) != 560 or set(question_ids) != set(responses.qids):
                raise ValueError("Primary bootstrap question identities differ")
            reorder = [question_ids.index(q) for q in responses.qids]
            draws = np.asarray(saved["multiplicities"])[:, reorder]
        if replicate_ids != list(range(meta["replicate_start"], meta["replicate_end"] + 1)) or len(replicate_ids) != 20:
            raise ValueError("Primary chunk replicate identity mismatch")
        for i, replicate in enumerate(replicate_ids):
            if replicate in seen_replicates or not 1 <= replicate <= 1000:
                raise ValueError("Duplicate or unexpected primary replicate")
            seen_replicates.add(replicate)
            if not np.array_equal(draws[i], draw_multiplicities(responses, replicate, cfg["seed"])):
                raise ValueError("Primary question bootstrap draw differs from archived stable-seed algorithm")
        fits = pd.read_parquet(path / "fits.parquet")
        split_rows = pd.read_parquet(path / "split_metrics.parquet")
        if len(fits) != 2000 or len(split_rows) != 2000 or fits.duplicated(["replicate", "job_id"]).any():
            raise ValueError("Primary refit/metric row count mismatch")
        replay = {}
        for row in fits.to_dict("records"):
            replicate, jid = int(row["replicate"]), row["job_id"]
            if replicate not in replicate_ids or jid not in primary_ids:
                raise ValueError("Unexpected primary fitting job")
            job = jobs[jid]
            if json_list(row["peer_ids"]) != job["peers"]:
                raise ValueError("Primary refit available peers changed")
            draw = draws[replicate_ids.index(replicate)]
            actual, _, kkt = replay_fit(responses, job, row["weights"], draw)
            if not bool(row["success"]) or kkt is None or kkt > 1e-7 or kkt < -1e-10:
                raise ValueError("Primary weighted-refit feasibility or MSE KKT failure")
            raw = np.asarray(row["raw_weights"], dtype=float)
            final = np.asarray(row["weights"], dtype=float)
            if raw.shape != final.shape or not np.isfinite(raw).all() or raw.min() < -1e-7 or abs(raw.sum() - 1) > 1e-7:
                raise ValueError("Primary raw coefficients violate fixed simplex tolerance")
            _, x = responses.arrays(job)
            train = responses.fit_indices[job["split_seed"]]
            change = float(np.max(abs(np.einsum("...p,p->...", x[train], final - raw))))
            if change > 1e-6 or float(row["repair_prediction_max_abs"]) > 1e-6:
                raise ValueError("Primary refit numerical repair exceeded tolerance")
            allowed = {"gram_active_set_v1_with_fixed_slsqp_recovery/active_set", "gram_active_set_v1_with_fixed_slsqp_recovery/slsqp_recovery"}
            if row["solver_backend"] not in allowed:
                raise ValueError("Primary refit changed its fixed MSE algorithm")
            branches[row["solver_backend"]] += 1
            if actual["train_group"] > actual["train_single"] + TRAIN_ATOL:
                raise ValueError("Primary refit training dominance failure")
            for key in ("train_objective", "solver_objective"):
                audit.close(row[key], actual["train_group"], "primary weighted objective", TRAIN_ATOL)
            for key, value in actual.items():
                if key == "chosen_single":
                    if row[key] != value:
                        raise ValueError("Primary bootstrap did not reselect the fitting-only single peer")
                else:
                    audit.close(row[key], value, "primary fit metrics/" + key)
            replay[(replicate, jid)] = actual
        for replicate in replicate_ids:
            if {j for r, j in replay if r == replicate} != primary_ids:
                raise ValueError("A primary replicate lacks one of its 100 required refits")
        primary_split_keys = set()
        for row in split_rows.to_dict("records"):
            replicate = int(row["replicate"])
            t, f, s = row["target"], row["family"], int(row["split_seed"])
            full = by_key[("expanded11", t, f, s, "mse", "group", None)]
            measured = replay[(replicate, full)]
            if row["table"] == "scalar_main_split_results":
                key = (replicate, row["table"], t, f, s, "")
                expected = {k: v for k, v in measured.items() if k != "chosen_single"}
            elif row["table"] == "scalar_directional_pairs":
                p = row["partner"]
                if (t, p) not in directions:
                    raise ValueError("Undeclared primary directional comparison")
                key = (replicate, row["table"], t, f, s, p)
                removed = by_key[("expanded11", t, f, s, "mse", "removal", p)]
                reverse = by_key[("expanded11", p, f, s, "mse", "group", None)]
                error = _partner_error(responses, t, p, f, s, "mse", draws[replicate_ids.index(replicate)])
                expected = {"partner_error": error, "e_group": measured["e_group"],
                            "extra_beyond_partner": error - measured["e_group"],
                            "e_without": replay[(replicate, removed)]["e_group"],
                            "removal_effect": replay[(replicate, removed)]["e_group"] - measured["e_group"],
                            "directional_extra_difference": replay[(replicate, reverse)]["e_group"] - measured["e_group"]}
                if replay[(replicate, removed)]["train_group"] + TRAIN_ATOL < measured["train_group"]:
                    raise ValueError("Primary weighted removal training nesting failure")
            else:
                raise ValueError("Unexpected primary split-metric source table")
            if key in primary_split_keys:
                raise ValueError("Repeated primary split-metric key")
            primary_split_keys.add(key)
            for metric, value in expected.items():
                audit.close(row[metric], value, "primary split contrasts/" + metric)
        for column, cid in enumerate(ordered_ids):
            descriptor = descriptors[cid]
            selected = split_rows[split_rows.table == descriptor["table"]]
            for key in INTERVAL_KEYS:
                if descriptor[key]:
                    selected = selected[selected[key].fillna("").astype(str) == descriptor[key]]
            for replicate, group in selected.groupby("replicate"):
                if len(group) != 10 or group.split_seed.nunique() != 10:
                    raise ValueError("Primary replicate contrast omits or duplicates a fixed split")
                values[int(replicate) - 1, column] = group[descriptor["metric"]].mean(skipna=False)
        fit_count += len(fits)
    audit.check(seen_replicates == set(range(1, 1001)) and fit_count == 100000,
                "All 1,000 true refit replicates and 100,000 weighted fits replayed", fits=fit_count, branches=dict(branches))
    descriptors, ordered_ids, values = verify_relative_omissions(analysis / "primary_relative_omissions.json", descriptors, ordered_ids, values, audit)
    long = pd.read_parquet(analysis / "primary_refit_replicates.parquet")
    if len(long) != 1000 * len(ordered_ids) or long.duplicated(["replicate", "contrast_id"]).any():
        raise ValueError("Primary saved replicate contrast inventory mismatch")
    matrix = long.pivot(index="replicate", columns="contrast_id", values="value").reindex(index=range(1, 1001), columns=ordered_ids).to_numpy()
    audit.close(matrix, values, "primary aggregate replicate estimates")
    intervals = read_table_csv(analysis / "primary_refit_intervals.csv")
    _verify_interval_table(intervals, descriptors, values, ordered_ids, 1000, "primary_refit", audit)
    for field in ("table", "panel", "target", "family", "fit_loss", "metric", "run_id", "response", "evaluation_metric",
                  "peer_ids", "input_fingerprint", "completion_state", "partner", "removed_peer", "comparator_panel"):
        if field not in long or field not in intervals:
            raise ValueError("Primary replicate evidence lacks direct metadata: " + field)
        expected = long.contrast_id.map(intervals.set_index("contrast_id")[field]).fillna("").astype(str)
        if not long[field].fillna("").astype(str).eq(expected).all():
            raise ValueError("Primary replicate metadata differs from its exact interval contrast: " + field)


class IndependentFrozenReplay:
    def __init__(self, responses, draws, jobs, weights, by_key):
        self.responses, self.draws, self.jobs, self.weights, self.by_key = responses, draws, jobs, weights, by_key
        self.sample_masses = {seed: draws[:, indices] / draws[:, indices].sum(axis=1, keepdims=True)
                              for seed, indices in responses.eval_indices.items()}
        self.metric_cache = {}
        self.partner_cache = {}

    def metric(self, job_id, metric):
        key = (job_id, metric)
        if key not in self.metric_cache:
            job = self.jobs[job_id]
            _, losses, _ = replay_fit(self.responses, job, self.weights[job_id])
            columns = {"e_group": "total_variation" if job["fit_loss"] == "tv" else "absolute_error",
                       "e_single": "single_total_variation" if job["fit_loss"] == "tv" else "single_absolute_error",
                       "mse_group": "squared_error", "mse_single": "single_squared_error",
                       "group_top_option_agreement": "top_option_agreement", "single_top_option_agreement": "single_top_option_agreement"}
            ids = self.responses.eval_indices[job["split_seed"]]
            values = losses[columns[metric]][ids] @ CONDITION_MASS
            self.metric_cache[key] = self.sample_masses[job["split_seed"]] @ values
        return self.metric_cache[key]

    def partner(self, row):
        target, partner, family, seed, loss = (row[k] for k in ("target", "partner", "family", "split_seed", "fit_loss"))
        key = (*sorted([target, partner]), family, int(seed), loss == "tv")
        if key not in self.partner_cache:
            source = self.responses.p if loss == "tv" else self.responses.y
            c = [self.responses.cindex[v] for v in FAMILY_CONDITIONS[family]]
            values = abs(source[self.responses.mindex[target]][:, c] - source[self.responses.mindex[partner]][:, c])
            if loss == "tv":
                values = .5 * values.sum(axis=-1)
            ids = self.responses.eval_indices[int(seed)]
            self.partner_cache[key] = self.sample_masses[int(seed)] @ (values[ids] @ CONDITION_MASS)
        return self.partner_cache[key]

    def row_value(self, table, row, metric):
        get = self.metric
        if table in ("scalar_main_split_results", "vector_main_split_results"):
            jid = row["job_id"]
            if metric == "gamma":
                return get(jid, "e_single") - get(jid, "e_group")
            if metric == "relative_gamma_percent":
                single = get(jid, "e_single")
                return np.divide(100 * (single - get(jid, "e_group")), single,
                                 out=np.full_like(single, np.nan), where=single > 1e-14)
            if metric == "agreement_gain":
                return get(jid, "group_top_option_agreement") - get(jid, "single_top_option_agreement")
            return get(jid, metric)
        if table == "scalar_removal_split_results":
            if metric == "e_full":
                return get(row["full_job_id"], "e_group")
            if metric == "e_without":
                return get(row["removed_job_id"], "e_group")
            return get(row["removed_job_id"], "e_group") - get(row["full_job_id"], "e_group")
        if table in ("scalar_directional_pairs", "vector_directional_removals"):
            full = get(row["full_job_id"], "e_group")
            if metric == "partner_error":
                return self.partner(row)
            if metric == "e_group":
                return full
            if metric == "e_without":
                return get(row["removed_job_id"], "e_group")
            if metric == "extra_beyond_partner":
                return self.partner(row) - full
            if metric == "removal_effect":
                return get(row["removed_job_id"], "e_group") - full
            reverse = self.by_key[("expanded11", row["partner"], row["family"], int(row["split_seed"]), row["fit_loss"], "group", None)]
            return get(reverse, "e_group") - full
        if table in ("scalar_roster_expansion", "vector_roster_expansion"):
            if metric == "comparator_error":
                return get(row["comparator_job_id"], "e_group")
            if metric == "expanded_error":
                return get(row["expanded_job_id"], "e_group")
            return get(row["comparator_job_id"], "e_group") - get(row["expanded_job_id"], "e_group")
        raise ValueError("Unknown frozen-fit contrast table")


def verify_uncertainty(responses, root, tables, jobs, metrics, weights, by_key, audit):
    analysis = root / "analysis"
    expected_identity = next(iter(set(tables["scalar_main_split_results"].input_fingerprint)))
    for name in ("point_complete.json", "primary_refit_complete.json", "secondary_complete.json", "bootstrap_complete.json"):
        marker = json.loads((analysis / name).read_text())
        if marker.get("status") != "complete" or marker.get("input_fingerprint") != expected_identity:
            raise ValueError("Incomplete or mismatched empirical stage: " + name)
        for file, digest in marker.get("artifact_hashes", {}).items():
            if sha256(analysis / file) != digest:
                raise ValueError("Completion artifact checksum mismatch: " + file)
    verify_primary_refits(responses, analysis, tables, jobs, by_key, audit)
    descriptors = _contrast_descriptors(responses.config, tables, primary=False)
    audit.check(len(descriptors) == 2570, "All 2,570 secondary reported point-metric contrasts")
    with np.load(analysis / "secondary_frozen_fit_replicates.npz", allow_pickle=False) as evidence:
        saved_ids = evidence["contrast_ids"].astype(str).tolist()
        question_ids = evidence["question_ids"].astype(str).tolist()
        if set(question_ids) != set(responses.qids) or len(question_ids) != 560:
            raise ValueError("Frozen-fit resample question identity mismatch")
        if evidence["replicates"].tolist() != list(range(1, 2001)):
            raise ValueError("Frozen-fit replicate IDs differ")
        draws = evidence["multiplicities"][:, [question_ids.index(q) for q in responses.qids]]
        saved_values = evidence["values"]
    if len(saved_ids) != len(set(saved_ids)) or not set(saved_ids) <= set(descriptors):
        raise ValueError("Frozen-fit saved contrast IDs are unexpected or duplicated")
    for replicate in range(1, 2001):
        if not np.array_equal(draws[replicate - 1], draw_multiplicities(responses, replicate, 20261109)):
            raise ValueError("Frozen-fit common-question draws changed")
    audit.check(True, "All 2,000 frozen-fit multiplicity vectors independently regenerated")
    sampler = IndependentFrozenReplay(responses, draws, jobs, weights, by_key)
    ordered_ids = list(descriptors)
    values = np.empty((2000, len(ordered_ids)))
    for column, cid in enumerate(ordered_ids):
        descriptor = descriptors[cid]
        values[:, column] = np.mean([sampler.row_value(descriptor["table"], row, descriptor["metric"])
                                    for row in descriptor["rows"]], axis=0)
    descriptors, ordered_ids, values = verify_relative_omissions(analysis / "secondary_relative_omissions.json", descriptors, ordered_ids, values, audit)
    if set(saved_ids) != set(ordered_ids):
        raise ValueError("Frozen-fit evidence omits a defined contrast")
    saved_values = saved_values[:, [saved_ids.index(cid) for cid in ordered_ids]]
    audit.close(saved_values, values, "All independently replayed frozen-fit replicate contrasts")
    secondary = read_table_csv(analysis / "secondary_descriptive_intervals.csv")
    _verify_interval_table(secondary, descriptors, values, ordered_ids, 2000, "secondary_frozen_fit", audit)
    audit.check(True, "All primary and secondary percentile intervals independently reconstructed")


def verify_report_outputs(root, tables, audit):
    analysis = root / "analysis"
    marker = json.loads((root / "status/report_complete.json").read_text())
    if marker.get("status") != "complete":
        raise ValueError("Report is not complete")
    for record in marker["artifacts"]:
        if sha256(root / record["path"]) != record["sha256"]:
            raise ValueError("Report artifact changed after generation: " + record["path"])
    summary = read_table_csv(analysis / "full_summary.csv")
    expected = {}
    for name in ("scalar_main_split_results", "vector_main_split_results"):
        for key, group in tables[name].groupby(["panel", "target", "family", "fit_loss"]):
            expected[key] = group
    if len(summary) != len(expected) or len(summary) != 132:
        raise ValueError("Full summary omits a panel/target/family/loss cell")
    seen = set()
    for row in summary.to_dict("records"):
        key = tuple(row[k] for k in ("panel", "target", "family", "fit_loss"))
        if key in seen or key not in expected:
            raise ValueError("Repeated or invalid summary cell")
        seen.add(key)
        source = expected[key]
        for field in ("e_single", "e_group", "gamma", "relative_gamma_percent", "mse_single", "mse_group", "single_top_option_agreement", "group_top_option_agreement"):
            if field in source:
                value = source[field].mean(skipna=False) if field == "relative_gamma_percent" else source[field].mean()
                audit.close(row[field], float(value), "Full summary/" + field)
        denominator = source.e_single.mean()
        ratio = 100 * source.gamma.mean() / denominator if denominator > 1e-14 else np.nan
        audit.close(row["macro_ratio_of_means_percent"], ratio, "Full summary ratio of means")
    primary = read_table_csv(analysis / "primary_refit_intervals.csv").set_index("contrast_id")
    secondary = read_table_csv(analysis / "secondary_descriptive_intervals.csv").set_index("contrast_id")
    evidence = json.loads((analysis / "headline_evidence.json").read_text())
    claims = {c["claim_id"]: c for c in evidence["claims"]}
    if len(claims) != len(evidence["claims"]):
        raise ValueError("Duplicate prose claim identifier")
    allowed_numbers = set()
    for claim in claims.values():
        table_name = claim["source_table"].removesuffix(".csv")
        source = tables[table_name]
        for key, value in claim["row_selector"].items():
            source = source[source[key].astype(str) == str(value)]
        if len(source) != 10:
            raise ValueError("Prose evidence does not select ten fixed splits")
        if claim["claim_id"].startswith("large_target/"):
            metric = "gamma"
            audit.close(claim["e_single"], source.e_single.mean(), "Headline selected single")
            audit.close(claim["e_group"], source.e_group.mean(), "Headline group")
            denominator = source.e_single.mean()
            ratio = 100 * source.gamma.mean() / denominator if denominator > 1e-14 else np.nan
            audit.close(claim["ratio_of_mean_gain_to_mean_single_percent"], ratio, "Headline ratio formula")
        elif claim["claim_id"].startswith("roster/"):
            metric = "large_peer_gain"
        else:
            metric = claim["claim_id"].split("/")[-1]
        audit.close(claim["point"], source[metric].mean(), "Headline evidence point")
        cid = claim["interval_contrast_id"]
        interval = primary.loc[cid] if cid in primary.index else secondary.loc[cid]
        audit.close([claim["low"], claim["high"]], [interval.low, interval.high], "Headline evidence interval")
        for key in ("point", "low", "high", "e_single", "e_group"):
            if key in claim:
                allowed_numbers.add(f"{float(claim[key]):.5f}")
    for filename in ("XfqA_RESPONSE_DRAFT.md", "RESULTS_INSERT.md"):
        text = (analysis / filename).read_text()
        for number in re.findall(r"(?<![\w.])[-+]?\d+\.\d{5}(?!\d)", text):
            if number not in allowed_numbers:
                raise ValueError("Unsupported numerical value in response/insert: " + number)
        if not all(name in text for name in ("Qwen-32B", "QwQ-32B", "Gemma-27B")):
            raise ValueError("A new target is missing from the measured-results prose")
    audit.check(True, "Full summary and all numerical headline evidence replayed; prose includes all new targets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "plan-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_verify(args.repo_root, args.plan_root, args.output_root, args.resume,
                               portable=args.portable, audit_output=args.audit_output), indent=2))
