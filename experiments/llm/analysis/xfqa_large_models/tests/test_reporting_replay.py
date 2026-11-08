"""Meaningful independent replay, scientific reporting and portable path checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

COMPONENT = Path(__file__).resolve().parents[1]
REPO = COMPONENT.parents[3]
sys.path.insert(0, str(COMPONENT))
import reporting
import verify
import delivery


@pytest.fixture
def plan():
    path = REPO / "outputs/xfqa_large_models/20261108/plan"
    if not path.exists():
        pytest.skip("Supplied XfqA protocol package is needed for exact-plan integration checks")
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_independent_job_inventory_matches_supplied_full_grid(plan):
    config = json.loads((plan / "config/experiment.json").read_text())
    reference = load_module(plan / "tools/build_analysis_jobs.py", "xfqa_test_supplied_jobs")
    expected = {row["job_id"]: row for row in reference.build_jobs(config)}
    assert verify.expected_jobs(config) == expected
    assert len(expected) == 5840
    assert all(row["target"] not in row["peers"] for row in expected.values())


def test_independent_splits_and_bootstrap_match_historical_algorithm(plan):
    sys.path.insert(0, str(REPO / "experiments/llm/analysis/xwvg_reference_selection/src"))
    sys.path.insert(0, str(REPO / "experiments/llm/analysis/v23/src"))
    from bca_reference_selection.splits import fixed_splits
    from pier_llm_v23.core import common_bootstrap_multiplicities
    questions = [json.loads(line) for line in (plan / "seed_inputs/mmlu_pro_selected_560.jsonl").read_text().splitlines()]
    seeds = list(range(20260828, 20260838))
    independent = verify.independent_splits(questions, seeds)
    reference = fixed_splits(pd.DataFrame(questions), seeds)
    for seed in seeds:
        assert set(independent[seed]["fitting_ids"]) == reference[seed].fitting_ids
        assert set(independent[seed]["evaluation_ids"]) == reference[seed].evaluation_ids
    qids = sorted(str(row["base_question_id"]) for row in questions)
    categories = {str(row["base_question_id"]): row["category"] for row in questions}
    response = SimpleNamespace(qids=qids, qindex={q: i for i, q in enumerate(qids)}, categories=[categories[q] for q in qids])
    response.fit_indices = {s: np.array([response.qindex[q] for q in v["fitting_ids"]]) for s, v in independent.items()}
    response.eval_indices = {s: np.array([response.qindex[q] for q in v["evaluation_ids"]]) for s, v in independent.items()}
    for seed in (20261108, 20261109):
        for replicate in (1, 20, 777, 1000):
            actual = verify.draw_multiplicities(response, replicate, seed)
            old = common_bootstrap_multiplicities(pd.DataFrame(questions), replicate, seed)
            assert actual.tolist() == [old[q] for q in qids]
            assert actual.sum() == 560


class SyntheticResponses:
    def __init__(self, vector=False):
        rng = np.random.default_rng(881)
        self.fit_indices = {11: np.arange(280)}
        self.eval_indices = {11: np.arange(280, 560)}
        self.counts = np.full(560, 3)
        self.x = rng.uniform(.1, .9, size=(560, 4, 2))
        if vector:
            raw = rng.uniform(.1, .9, size=(560, 4, 3, 2))
            raw /= raw.sum(axis=2, keepdims=True)
            self.x = np.zeros((560, 4, 10, 2))
            self.x[:, :, :3] = raw
        self.y = self.x @ np.array([.25, .75])

    def arrays(self, job):
        return self.y, self.x


@pytest.mark.parametrize("loss", ["mse", "mae", "tv"])
def test_independent_replay_exact_convex_target_and_fit_only_single(loss):
    source = SyntheticResponses(vector=loss == "tv")
    job = {"split_seed": 11, "fit_loss": loss, "peers": ["p0", "p1"]}
    metrics, arrays, gap = verify.replay_fit(source, job, [.25, .75])
    assert metrics["e_group"] < 1e-14
    assert metrics["train_group"] < 1e-14
    assert metrics["chosen_single"] == "p1"
    assert metrics["e_single"] > 0
    if loss == "mse":
        assert abs(gap) < 1e-14
    if loss == "tv":
        assert np.all(arrays["top_option_agreement"] == 1)
        assert metrics["group_top_option_agreement"] == 1


def test_bootstrap_evaluation_normalizes_within_each_fixed_split():
    source = SyntheticResponses()
    source.y = source.y.copy()
    source.y[280:, 0] += np.linspace(0, .1, 280)
    job = {"split_seed": 11, "fit_loss": "mse", "peers": ["p0", "p1"]}
    draw = np.ones(560)
    draw[280:420] = 4
    draw[420:] = 0
    metrics, _, _ = verify.replay_fit(source, job, [.25, .75], draw)
    expected = .5 * np.linspace(0, .1, 280)[:140].mean()
    assert metrics["e_group"] == pytest.approx(expected, abs=1e-14)
    assert metrics["e_group"] != pytest.approx(.5 * np.linspace(0, .1, 280).mean())


def test_audit_rejects_magnitude_nan_and_shape_corruption():
    audit = verify.Audit()
    audit.close([1., np.nan], [1., np.nan], "allowed undefined relative")
    with pytest.raises(ValueError, match="mismatch"):
        audit.close([.2], [.21], "metric")
    with pytest.raises(ValueError, match="NaN"):
        audit.close([np.nan], [0.], "metric")
    with pytest.raises(ValueError, match="shape"):
        audit.close([1., 2.], [1.], "metric")


@pytest.mark.parametrize("run_id", ["20261108", "00017"])
def test_numeric_csv_run_identity_matches_parquet_without_losing_lexical_identity(tmp_path, run_id):
    """A date-like run ID must survive CSV parsing and exact Parquet checks."""
    csv_path = tmp_path / "point_table.csv"
    pd.DataFrame({"run_id": [run_id], "input_fingerprint": ["0123456789"], "e_group": [.2]}).to_csv(csv_path, index=False)
    point = verify.read_table_csv(csv_path)
    assert point.run_id.iloc[0] == run_id
    assert point.input_fingerprint.iloc[0] == "0123456789"
    assert point.e_group.iloc[0] == pytest.approx(.2)
    job = {"panel": "expanded11", "target": "target", "family": "irrelevant_context", "split_seed": 20260828,
           "response": "reference_answer_probability", "fit_loss": "mse", "peers": ["peer"]}
    row = {k: v for k, v in job.items() if k != "peers"}
    row.update(run_id=run_id, input_fingerprint="0123456789", completion_state="complete",
               evaluation_metric="mae", peer_ids='["peer"]')
    parquet_path = tmp_path / "losses.parquet"
    pd.DataFrame([row]).to_parquet(parquet_path, index=False)
    saved = pd.read_parquet(parquet_path)
    verify._verify_per_question_metadata(saved, job, run_id=point.run_id.iloc[0],
                                        input_fingerprint=point.input_fingerprint.iloc[0])
    for changed_id in (run_id + "0", run_id + ".0"):
        corrupted = saved.copy()
        corrupted["run_id"] = changed_id
        with pytest.raises(ValueError, match="metadata mismatch: run_id"):
            verify._verify_per_question_metadata(corrupted, job, run_id=point.run_id.iloc[0],
                                                input_fingerprint=point.input_fingerprint.iloc[0])


def test_portable_provenance_never_reuses_original_absolute_path(tmp_path):
    root = tmp_path / "bundle/run"
    root.mkdir(parents=True)
    recorded = "/an/old/machine/outputs/xfqa_large_models/20261108/outputs/raw_scores/model/shard_00001.parquet"
    expected = root / "outputs/raw_scores/model/shard_00001.parquet"
    assert verify.relocated_run_path(recorded, root) == expected
    with pytest.raises(ValueError):
        verify.relocated_run_path("/other/private/path/token", root)
    with pytest.raises(ValueError):
        verify.relocated_run_path("../outside", root)


def test_relative_omission_must_be_undefined_and_preserves_absolute_contrasts(tmp_path):
    descriptors = {
        "gain": {"metric": "gamma", "point": .1, "rows": []},
        "relative": {"metric": "relative_gamma_percent", "point": np.nan,
                     "rows": [{"split_seed": 1, "relative_gamma_percent": np.nan}]},
    }
    path = tmp_path / "omissions.json"
    path.write_text(json.dumps({"absolute_contrasts_retained": True, "omissions": [
        {"contrast_id": "relative", "stage": "point", "undefined_split_seeds": [1]}]}))
    values = np.array([[.1, np.nan], [.2, 3.]])
    d, ids, v = verify.verify_relative_omissions(path, descriptors, ["gain", "relative"], values, verify.Audit())
    assert ids == ["gain"] and set(d) == {"gain"}
    assert v.tolist() == [[.1], [.2]]
    descriptors["gain"]["point"] = np.nan
    with pytest.raises(ValueError, match="absolute"):
        verify.verify_relative_omissions(path, descriptors, ["gain", "relative"], values, verify.Audit())


def test_summary_retains_negative_gains_and_separates_relative_estimands():
    rows = []
    for seed in range(10):
        rows.append({"run_id": "synthetic-fixture", "panel": "expanded11", "target": "t", "family": "irrelevant_context",
                     "response": "reference_answer_probability", "fit_loss": "mse", "evaluation_metric": "mae",
                     "input_fingerprint": "synthetic", "split_seed": seed, "peer_ids": '["p"]',
                     "e_single": .1 + seed / 100, "e_group": .12 + seed / 100, "gamma": -.02,
                     "relative_gamma_percent": -2 / (.1 + seed / 100)})
    scalar = pd.DataFrame(rows)
    vector = scalar.iloc[:0].copy()
    out = reporting.aggregate_tables({"scalar_main_split_results": scalar, "vector_main_split_results": vector})
    assert len(out) == 1
    assert out.iloc[0].negative_gain_split_count == 10
    assert out.iloc[0].gamma == pytest.approx(-.02)
    assert out.iloc[0].relative_gamma_percent != pytest.approx(out.iloc[0].macro_ratio_of_means_percent)


def test_report_refuses_missing_empirical_completion_without_writing_claims(tmp_path, plan):
    with pytest.raises(ValueError, match="completed empirical stage"):
        reporting.run_report(REPO, plan, tmp_path)
    assert not (tmp_path / "analysis/XfqA_RESPONSE_DRAFT.md").exists()


def test_report_keeps_overlapping_test_runs_separate_and_names_skips(tmp_path):
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "implementation_tests.xml").write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1">'
        '<testcase name="test_scoring_guard"/><testcase name="test_tokens">'
        '<skipped message="Torch required: run this test in the scoring environment"/>'
        '</testcase></testsuite></testsuites>')
    (validation / "delivery_tests.xml").write_text(
        '<testsuite tests="2" failures="0" errors="0" skipped="0">'
        '<testcase name="test_package"/><testcase name="test_runtime"/></testsuite>')
    (validation / "scoring_tests.txt").write_text(
        'test_scoring_guard (checks.Scoring.test_scoring_guard) ... ok\n'
        'test_tokens (checks.Scoring.test_tokens) ... ok\n\nRan 2 tests in 0.1s\n\nOK\n')
    text, evidence, sources = reporting.implementation_check_summary(tmp_path)
    assert [(r["passed"], r["failed"], r["errors"], r["skipped"]) for r in evidence["runs"]] == [
        (1, 0, 0, 1), (2, 0, 0, 0), (2, 0, 0, 0)]
    assert evidence["unique_test_total"] is None
    assert evidence["overlapping_runs"] is True
    assert "not summed into a unique-test total" in text
    assert "Torch required" in text
    assert "../validation/scoring_tests.txt" in text
    assert len(sources) == 3
    scoring = validation / "scoring_tests.txt"
    scoring.write_text(scoring.read_text().replace("Ran 2 tests", "Ran 3 tests"))
    with pytest.raises(ValueError, match="inconsistent case counts"):
        reporting.implementation_check_summary(tmp_path)


def test_packager_excludes_credentials_weights_environments_and_links():
    for path in ("cache/models/blob", "runtime/scoring/bin/python", "models/weights.safetensors",
                 "token", "hosts.yml", ".env", "id_ed25519_xwvg", "__pycache__/module.pyc"):
        assert delivery._reject_name(Path(path)), path
    for path in ("analysis/point_fit_coefficients.parquet", "analysis/checkpoints/primary_refit/chunk_0001_0020/fits.parquet",
                 "status/model_downloads/model.json", "data/new_rotation_scores.parquet", "RUN_COMMANDS.sh"):
        assert not delivery._reject_name(Path(path)), path


def test_packager_keeps_exact_environment_locks_and_frozen_sources_without_venv(tmp_path):
    repo, root = tmp_path / "repo", tmp_path / "run"
    required = (
        "inputs/mmlu_pro_selected_560.jsonl", "inputs/permutation_averaged_semantic_scores.parquet",
        "data/new_rotation_scores.parquet", "data/new_permutation_averaged_semantic_scores.parquet",
        "data/permutation_averaged_semantic_scores.parquet", "data/permutations/score_execution_manifest.jsonl",
        "model_lock.json", "plan/config/experiment.json", "analysis/point_fit_coefficients.parquet",
        "analysis/per_question_losses.parquet", "analysis/split_ids.json", "analysis/primary_refit_intervals.csv",
        "analysis/secondary_descriptive_intervals.csv", "validation/numerical_replay.json",
        "RUN_COMMANDS.sh", "resource_summary.json", "resource_summary.csv",
    )
    files = required + tuple("runtime/" + name for name in delivery.RUNTIME_METADATA) + (
        "runtime/frozen_analysis/analysis.py", "runtime/frozen_scoring/scoring.py",
        "runtime/analysis/bin/python", "runtime/scoring/lib/secret.py", "cache/model.safetensors",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic packaging fixture\n")
    for source in delivery.SOURCE_ROOTS:
        (repo / source).mkdir(parents=True)
    selected, _ = delivery.collect_files(repo, root, "result.zip")
    for name in delivery.RUNTIME_METADATA:
        assert "run/runtime/" + name in selected
    assert "run/runtime/frozen_analysis/analysis.py" in selected
    assert "run/runtime/frozen_scoring/scoring.py" in selected
    assert "run/resource_summary.csv" in selected
    assert "run/runtime/initial_resource_observations.json" in selected
    assert not any("/bin/" in name or "/lib/" in name or "/cache/" in name for name in selected)
    for relative in ("runtime/analysis_requirements.lock.txt", "runtime/initial_resource_observations.json", "resource_summary.csv"):
        path = root / relative
        path.unlink()
        with pytest.raises(FileNotFoundError, match=path.name):
            delivery.collect_files(repo, root, "result.zip")
        path.write_text("synthetic packaging fixture\n")
