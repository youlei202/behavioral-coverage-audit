"""Tiny in-memory integration tests; no historical cache or empirical report."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from bca_reference_selection.data import Inputs
from bca_reference_selection import pipeline
from bca_reference_selection.reporting import summarize_strategy_results
from bca_reference_selection.resample import frozen_fit_resampling
from bca_reference_selection.splits import fixed_splits


class PipelineChecks(unittest.TestCase):
    def fixture(self):
        config = json.loads((pipeline.COMPONENT / "protocol/config/experiment.json").read_text())
        config["synthetic_fixture"] = True
        config["question_count"] = 4
        config["category_count"] = 1
        config["questions_per_category"] = 4
        rng = np.random.default_rng(438)
        probabilities = {f: rng.dirichlet(np.ones(3), size=(8, 4, 4)) for f in config["families"]}
        # Exact training mixture gives a measurable held-out target change.
        for values in probabilities.values():
            values[0] = .35 * values[1] + .65 * values[3]
        ids = tuple(f"q{i}" for i in range(4))
        inputs = Inputs(probabilities=probabilities, mask=np.ones((4, 4, 3), dtype=bool),
            question_ids=ids, categories=("subject",) * 4, model_ids=tuple(config["model_roster"]),
            masses={f: np.array([.5, 1/6, 1/6, 1/6]) for f in config["families"]},
            questions=pd.DataFrame({"base_question_id": ids, "category": ["subject"] * 4}),
            report={"synthetic_fixture": True})
        task = (0, "irrelevant_context", config["split_seeds"][0], set(ids[:2]), set(ids[2:]), False)
        return inputs, config, task

    def initialize(self, directory):
        inputs, config, task = self.fixture()
        pipeline.initialize(inputs, config, directory, "synthetic_test_fingerprint")
        return inputs, config, task

    def test_full_setting_output_and_resume_without_refitting(self):
        with TemporaryDirectory() as temporary:
            _, _, task = self.initialize(temporary)
            result = pipeline.fit_setting(task)
            directory = Path(result["path"])
            self.assertEqual(result["candidate_count"], 64)
            fits = pd.read_parquet(directory / "fits.parquet")
            evaluations = pd.read_parquet(directory / "evaluations.parquet")
            strategies = pd.read_parquet(directory / "strategies.parquet")
            losses = pd.read_parquet(directory / "losses.parquet")
            self.assertEqual(len(fits), 64)
            self.assertEqual(len(evaluations), 64)
            self.assertEqual(len(strategies), 12)
            self.assertEqual(len(losses), 24)
            self.assertEqual(fits.groupby("k").size().to_dict(), {1: 7, 2: 21, 3: 35, 7: 1})
            random = strategies[strategies.method == "uniform_random_expected"]
            self.assertTrue(random.selected_peers.isna().all())
            self.assertTrue(random.selected_weights.isna().all())
            for _, row in random.iterrows():
                self.assertAlmostEqual(row.evaluation_tv,
                    evaluations[evaluations.k == row.k].evaluation_tv.mean(), places=12)
            before = (directory / "selection.json").read_bytes()
            with patch.object(pipeline, "fit_tv_simplex", side_effect=AssertionError("Refit on resume")):
                again = pipeline.fit_setting(task)
            self.assertEqual(again["path"], result["path"])
            self.assertEqual(before, (directory / "selection.json").read_bytes())

    def test_changed_heldout_target_preserves_selection_and_weights(self):
        with TemporaryDirectory() as temporary:
            inputs, _, task = self.initialize(temporary)
            directory = Path(pipeline.fit_setting(task)["path"])
            before = (directory / "selection.json").read_bytes()
            fits_before = {p.name: p.read_bytes() for p in directory.glob("candidate_*.json")}
            evaluation_before = pd.read_parquet(directory / "evaluations.parquet").evaluation_tv.to_numpy()
            inputs.probabilities[task[1]][0, 2:, :, :] = [1., 0., 0.]
            with patch.object(pipeline, "fit_tv_simplex", side_effect=AssertionError("Evaluation altered fit")):
                pipeline.fit_setting(task)
            self.assertEqual(before, (directory / "selection.json").read_bytes())
            self.assertEqual(fits_before, {p.name: p.read_bytes() for p in directory.glob("candidate_*.json")})
            evaluation_after = pd.read_parquet(directory / "evaluations.parquet").evaluation_tv.to_numpy()
            self.assertGreater(float(np.max(np.abs(evaluation_after - evaluation_before))), .01)

    def test_corrupted_checkpoint_is_rejected(self):
        for valid_checksum in (False, True):
            with self.subTest(valid_checksum=valid_checksum), TemporaryDirectory() as temporary:
                _, _, task = self.initialize(temporary)
                directory = Path(pipeline.fit_setting(task)["path"])
                path = directory / "candidate_0_1.json"
                record = json.loads(path.read_text())
                record["fit"]["train_tv"] += .1
                if valid_checksum:
                    record["record_sha256"] = hashlib.sha256(pipeline.encoded(record["fit"])).hexdigest()
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Checkpoint.*mismatch"):
                    pipeline.fit_setting(task)

    def test_pilot_fits_reused_by_complete_setting(self):
        with TemporaryDirectory() as temporary:
            _, _, task = self.initialize(temporary)
            pilot = pipeline.fit_setting((*task[:-1], True))
            self.assertEqual(len(pilot["fits"]), 14)
            self.assertEqual(sum(r["k"] > 1 for r in pilot["fits"]), 7)
            fit = pipeline.fit_tv_simplex
            with patch.object(pipeline, "fit_tv_simplex", wraps=fit) as fitting:
                result = pipeline.fit_setting(task)
            self.assertEqual(fitting.call_count, 50)
            self.assertEqual(result["candidate_count"], 64)

    def test_full_protocol_grid_and_spawn_with_synthetic_probabilities(self):
        started = perf_counter()
        with TemporaryDirectory() as temporary:
            inputs, config, _ = self.initialize(temporary)
            splits = fixed_splits(inputs.questions, config["split_seeds"])
            tasks = [(t, f, seed, s.fitting_ids, s.evaluation_ids, False)
                     for t in range(8) for f in config["families"] for seed, s in splits.items()]
            self.assertEqual(len(tasks), 160)
            # One actual spawned process exercises pickling, process-local state,
            # and package-import thread limits; all others run serially for speed.
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                 initializer=pipeline.initialize,
                 initargs=(inputs, config, temporary, "synthetic_test_fingerprint")) as pool:
                first = pool.submit(pipeline.fit_setting, tasks[0]).result(timeout=60)
            directories = [Path(first["path"])]
            directories.extend(Path(pipeline.fit_setting(task)["path"]) for task in tasks[1:])
            frames = {name: pd.concat([pd.read_parquet(d / f"{name}.parquet") for d in directories],
                                     ignore_index=True)
                      for name in ("fits", "evaluations", "strategies", "losses")}
            self.assertEqual(len(frames["fits"]), 10240)
            self.assertEqual(int((frames["fits"].k > 1).sum()), 9120)
            self.assertEqual(len(frames["evaluations"]), 10240)
            self.assertEqual(len(frames["strategies"]), 1920)
            self.assertEqual(len(frames["losses"]), 3840)
            cell_keys = ["target", "family", "split_seed"]
            self.assertEqual(set(frames["fits"].groupby(cell_keys).size()), {64})
            self.assertEqual(set(frames["strategies"].groupby(cell_keys).size()), {12})
            self.assertFalse(frames["strategies"].duplicated(cell_keys + ["k", "method"]).any())
            for directory in directories:
                completed = json.loads((directory / "complete.json").read_text())
                for name, expected in completed["artifact_sha256"].items():
                    self.assertEqual(pipeline.digest(directory / name), expected)
            all_results, macro = summarize_strategy_results(frames["strategies"], config)
            self.assertEqual(len(all_results), 64)
            self.assertEqual(int(all_results.k.isin([2, 3]).sum()), 32)
            self.assertEqual(len(macro), 4)

            # A separate 80-question reporting fixture expands each pair of
            # constructed losses to 40 evaluation questions. The 4-question fit
            # fixture has legitimate zero-denominator draws and is unsuitable
            # for 2,000 bootstrap replicates. No resampling rule is changed.
            report_config = deepcopy(config)
            report_config.update(question_count=80, category_count=2, questions_per_category=40)
            questions = pd.DataFrame({"base_question_id": [f"r{i:02d}" for i in range(80)],
                                      "category": ["A"] * 40 + ["B"] * 40})
            report_splits = fixed_splits(questions, config["split_seeds"])
            expanded = []
            for seed, split in report_splits.items():
                block = frames["losses"].loc[frames["losses"].split_seed == seed].copy()
                qids = sorted(block.base_question_id.unique())
                evaluation_ids = sorted(split.evaluation_ids)
                mapping = pd.DataFrame({"base_question_id": np.repeat(qids, 20),
                                        "expanded_question_id": evaluation_ids})
                block = block.merge(mapping, on="base_question_id", validate="many_to_many")
                block = block.drop(columns="base_question_id").rename(
                    columns={"expanded_question_id": "base_question_id"})
                expanded.append(block)
            expanded = pd.concat(expanded, ignore_index=True)
            self.assertEqual(len(expanded), 76800)
            replicates, intervals, diagnostics = frozen_fit_resampling(
                expanded, questions, report_config, Path(temporary) / "constructed_resampling")
            self.assertEqual(replicates.replicate.nunique(), 2000)
            self.assertEqual(len(replicates), 8000)
            self.assertLess(diagnostics["all_ones_point_max_abs_error"], 1e-12)
            self.assertFalse(intervals.includes_selection_and_fitting_uncertainty.any())
            for _, row in macro.iterrows():
                point = intervals.loc[(intervals.scope == "macro") & (intervals.k == row.k) &
                                      (intervals.metric == "nearest_minus_collective_tv"), "estimate"].iloc[0]
                self.assertAlmostEqual(point, row.nearest_minus_collective_tv, places=12)
            saved = np.load(Path(temporary) / "constructed_resampling/validation/bootstrap_multiplicities.npz")
            self.assertEqual(saved["multiplicities"].shape, (2000, 80))
            np.testing.assert_array_equal(saved["multiplicities"][:, :40].sum(axis=1), np.full(2000, 40))
            np.testing.assert_array_equal(saved["multiplicities"][:, 40:].sum(axis=1), np.full(2000, 40))
        print(json.dumps({"synthetic_full_grid_smoke": "passed", "settings": 160,
            "candidates": 10240, "lp_fits": 9120, "strategy_rows": 1920,
            "shared_question_replicates": 2000, "spawned_settings": 1,
            "runtime_seconds": round(perf_counter() - started, 3)}), flush=True)


if __name__ == "__main__":
    unittest.main()
