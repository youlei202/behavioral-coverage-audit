# XWVG limited-reference selection

The completed fresh-scoring run, source modules, result files, and bundle reproduction entry point are indexed in [EXPERIMENT_FILES.md](EXPERIMENT_FILES.md).

This supplementary experiment implements the locked plan in `protocol/CODEX_TASK.md`. It uses cached V2.3 cyclic-permutation-averaged **complete semantic option probabilities**, fits fixed convex mixtures with sparse TV linear programs, and compares exhaustive collective selection, nearest-k selection, and exact uniform-random-subset expectation. The analysis CLI invokes no model inference. The separately authorized input recovery and fresh endpoint scoring workflow is documented in [recovery/README.md](recovery/README.md); its measurements retain explicit fresh-data provenance.

The full experiment contains eight targets, two stress families, ten original stratified splits, and budgets 1, 2, 3, 7: 160 settings, 10,240 candidates, and 9,120 nontrivial optimizations. Each question contributes clean mass 1/2 and three maximum-stress masses of 1/6. TV sums over valid options. All methods reuse the same candidate fits; training-only immutable selection records precede held-out evaluation. The primary budget is 2, secondary 3. Frozen-fit intervals use 2,000 common-question resamples and exclude selection/fitting uncertainty.

## Environment and checks

From the repository root, create a dedicated environment (the current session used `/tmp/xwvg-venv`):

```bash
python3 -m venv /tmp/xwvg-venv
/tmp/xwvg-venv/bin/pip install -e './experiments/llm/analysis/xwvg_reference_selection[test]'
/tmp/xwvg-venv/bin/python -m pytest experiments/llm/analysis/xwvg_reference_selection/tests -q
/tmp/xwvg-venv/bin/python -m unittest discover -s experiments/llm/analysis/xwvg_reference_selection/protocol/tests -v
```

Without installation, prefix commands with `PYTHONPATH=experiments/llm/analysis/xwvg_reference_selection/src`. Exact installed versions are captured in each run's manifest. The component caps BLAS/OpenMP threads before numerical imports and configures HiGHS to use one thread per worker. The fixed timing pilot runs serially and chooses at most four workers using CPU and available-memory limits.

## Required external data

Set `BEHAVIORAL_COVERAGE_RAW_DATA` to the original run parent, or supply explicit paths. The default empirical loader requires:

- `PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/outputs/analysis/permutation_averaged_semantic_scores.parquet`, column `permutation_averaged_semantic_probabilities`, 31,360 rows.
- `PIER_LLM_ECOSYSTEM_GOLDMINE_V2/data/mmlu_pro_selected_560.jsonl`, SHA-256 `9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6`, 458,957 bytes.

A complete `V23/outputs/validation/permutation_wise_semantic_scores.parquet` can replace the averaged input using `--rotations-cache`. CPU aggregation validates all rotations before saving a new derivative. If only shards remain, supply `--v2-root` and `--v23-root`; the adapter requires both sets of raw-score shards, every `.meta.json` sidecar, and `V23/data/permutations/score_semantic_manifest.jsonl`, checking historical prompt hashes, mappings, revisions and coverage. All derivative writes occur in the new output directory. No historical analysis launcher runs.

Frozen summary tables and scalar reference-answer caches cannot supply these vectors. The empirical CLI rejects protocol modifications and synthetic fixtures. Synthetic data are restricted to implementation tests.

## Run

Automatic discovery checks the explicit raw root, configured environment, repository and documented historical paths. A missing-input run exits 2 and writes exact requirements in `status/blocked_inputs.json`.

```bash
PYTHONPATH=experiments/llm/analysis/xwvg_reference_selection/src \
/tmp/xwvg-venv/bin/python -m bca_reference_selection.pipeline \
  --repo-root "$PWD" \
  --output-root "$PWD/outputs/xwvg_reference_selection/new_run" \
  --stage all --workers 4
```

Explicit inputs can be added as `--cache /absolute/path/to/cache.parquet --questions /absolute/path/to/mmlu_pro_selected_560.jsonl`. For a raw-run parent, use `--raw-root /absolute/path/to/original/runs`. Use a fresh output directory when input/config/source/package versions change. Add `--resume` only to continue an identity-matched run. Outputs cannot be placed in frozen data, paper/source/manifests directories, or historical PIER runs.

Stages: `check` validates input and saves exact question splits; `pilot` additionally times the fixed 84 pilot candidates; `point` fits/evaluates the complete grid; `report` validates saved completed settings and generates final summaries; `all` executes everything. Valid pilot fits are reused. Candidate records are atomic, checksum-verified, keyed by input/config/code/environment identity, and their training objectives are independently recomputed on resume. Failed settings retain successful candidates and prevent completion.

## Outputs and interpretation

The output contract is in `protocol/docs/04_IMPLEMENTATION_AND_DELIVERABLES.md`. A successful full run writes all candidate and strategy tables, per-question held-out metrics, 32-row main and 64-row all-budget tables, macro summaries, descriptive intervals, a two-panel PDF/PNG, `REPORT.md`, and a measured-results `XWVG_RESPONSE_DRAFT.md`. Only then is `status/complete.json` written. The 2,000 replicates yield 8,000 macro rows (one row per replicate and budget).

At missing data, only inventory/provenance/check reports are appropriate; no empirical chart or completed-results reviewer paragraph is generated. Technical completion does not require the comparison to favor collective selection. The interpretation concerns approximation of recorded option distributions on the existing benchmark, not task accuracy, safe target removal, inference savings or free-text equivalence.
