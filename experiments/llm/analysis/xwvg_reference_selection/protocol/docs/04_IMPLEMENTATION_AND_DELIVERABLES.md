# Implementation sequence and deliverables

This document specifies work for the receiving Codex. Apart from the bundled preflight and numerical reference, the proposed new pipeline does not yet exist. Commands below are the interface to implement, not claims of already completed execution.

## 1. Create a separate analysis component

Proposed destination:

```text
experiments/llm/analysis/xwvg_reference_selection/
  src/bca_reference_selection/
    __init__.py
    data.py
    splits.py
    fit.py
    select.py
    evaluate.py
    resample.py
    reporting.py
    pipeline.py
  tests/
```

Small modules may be combined where it improves clarity. The scientific protocol and output contract matter more than these proposed names. Do not overwrite historical V2/V2.2/V2.3 modules or add work to their old completion records.

Implement explicit `--repo-root`, `--cache`, `--questions`, `--config`, `--output-root`, `--stage`, `--workers`, and `--resume` arguments. No hidden import-time dependence on `/work/Lei` or `/work/Users/leiyo`.

## 2. Resolve primary inputs

Read the actual `02_CODE_AND_DATA_MAP.md` before implementing this step.

### Direct cache route

- `V23/outputs/analysis/permutation_averaged_semantic_scores.parquet`.
- `V2/data/mmlu_pro_selected_560.jsonl` for the exact original questions and subjects.
- Published roster and checkpoint revisions from the source configuration.
- Existing source/cache provenance where available; record when an old cache has no independently retained digest.

Here `V2` and `V23` refer to original run directories whose full names are in the map. These names are not assumed to be relative to the new component.

The averaged cache's primary key is `(model_id, base_question_id, semantic_condition)`. Require 31,360 rows and exact agreement of the question/category/condition/option-count sets across all eight models. Read `permutation_averaged_semantic_probabilities`; retain the gold-coordinate field only for input consistency checks, not as the primary target.

For each family, include its three maximum-stress tracks plus the single shared clean row. Normalize a missing clean track identifier to `-1` only according to the original code's convention. Assert the actual dose is zero for clean and the configured maximum for stressed rows. Do not blindly use `track != -1` as proof of a correct family/dose.

### CPU reconstruction route

If the average is absent, first look for the saved complete `outputs/validation/permutation_wise_semantic_scores.parquet`. It must include rotation 0 and all other cyclic rotations. A complete set has `permutation_id = 0,...,option_count-1` for each primary key. Every rotation must use the same semantic option identities, gold identity, category, family, dose and track.

Read `semantic_remapped_probabilities`, validate each distribution, average vectors over the complete rotation set, and write the result to the **new run output**. Match the original averaging logic and expected row count. Record the source hash and the aggregation code/config hash.

If only shards remain, reconstruct rotation 0 from the original V2 scores and the semantic prompt manifest, then combine it with the V2.3 new-rotation score shards. Verify sidecar hashes, exact prompt IDs, model revisions, schema and full cyclic completeness. Read the source implementation, but do not run a whole historical postprocessing campaign just to assemble this table. That campaign also launches unrelated fitting/bootstrap work and writes to original directories.

If complete balanced vectors cannot be recovered, produce `status/blocked_inputs.json`. Finish portable code and constructed-data checks where possible. The public vector summaries and correct-answer scalar cache are insufficient for the primary experiment.

## 3. Validate before fitting

Check the following once and save a structured validation report:

- Exact original 560 question IDs; 14 subjects and 40 questions per subject.
- Original eight-model roster and revisions where recorded.
- One clean and three maximum-stress observations per question/model/family.
- Full valid option vectors with identical option counts and semantic indexing across models.
- Finite probabilities, nonnegative coordinates, and unit mass to the fixed tolerance. Materially malformed distributions fail validation. If normalizing rounding-scale drift, log its maximum size and apply the same one-time rule to all inputs before fitting/evaluation.
- Zero padding for invalid option coordinates; masks are carried through fitting and evaluation.
- Ten original split seeds and exact deterministic split algorithm. Each side contains 280 questions, 20 per subject; no base question overlaps the two sides.
- Matching model rows by semantic keys, never by assumed file order.
- No method-specific row exclusions, silent missing-model substitution, or regenerated random splits.

The primary experiment has 2,240 observations per model/family, and 1,120 on each fitting/evaluation side. Each question carries total mass one before division by the number of questions: clean 1/2 and each stressed track 1/6.

## 4. Implement the sparse TV fitter

Use `reference/tv_core.py` as a checked starting point. It accepts target `[n,a]`, peer values `[n,a,p]`, a valid-option mask, and per-observation masses.

Flatten only valid coordinates. For coordinate `(r,a)`, use objective weight `mass[r] / (2 * sum(mass))`. The objective sums, rather than averages, over options. Use sparse identity and block matrices for absolute residual inequalities. Do not call the scalar dense MAE fitter after flattening the vector data.

Save each fit's solver status, objective, independently recomputed training TV, weight vector, simplex deviations, any rounding correction and solve time. A failed candidate cannot be silently excluded from exhaustive search. Retry only an explicitly documented numerical recovery with the same objective; if unresolved, label the corresponding setting incomplete.

For a singleton, evaluate its TV directly. Zero weights are allowed for selected peers. Use stable roster order and the fixed training-only tie tolerance. Do not add ridge penalties, positivity floors, input-dependent weights, validation-tuned temperatures, or new selector models.

## 5. Separate selection from evaluation

For each target/family/split:

1. Obtain all seven candidate peer columns in the fixed roster order excluding the target.
2. Create fitting and evaluation masks from base-question IDs.
3. Fit every subset of each prescribed size using fitting data only.
4. Rank individual peers by their fitting TV and record the nearest-k subset.
5. Select the collective subset by fitted training TV, with the fixed tie rule.
6. Write the chosen subsets, training objectives and weights to an immutable per-setting selection record.
7. Evaluate the saved candidates and selections on held-out observations.
8. Compute the random strategy by averaging each candidate subset's losses and agreement scores. Never average its probability predictions before scoring.

The same candidate fit may supply collective and nearest strategies. Retrieve it from the cache rather than refitting with different options.

Do not let held-out target values enter the selection function. One meaningful implementation check is to change evaluation target values after the selection record is written and verify that selected subsets and weights remain unchanged.

## 6. Timing pilot, run order and resume

The pilot is fixed in `config/experiment.json`: first three roster targets, both families, first split; lexicographic subset indices `[0,10,20]` for k=2, `[0,17,34]` for k=3, and the single k=7 subset. This is 42 nontrivial fits plus 42 singleton scores. Use zero-based subset indices.

Measure solve time and maximum resident memory. Project the full run using the separately measured k=2/k=3/k=7 timings. Start at one worker, then use at most four workers and reduce for available CPU/RAM. Set BLAS/OpenMP threads to one per worker. Do not change the experimental roster or protocol in response to pilot effect sizes.

Reuse valid pilot fits. Cache keys must include target, family, split, subset, response protocol, input hashes, effective configuration, source/code hash and solver version. Different weights or data cannot reuse a prior completion marker. Write candidate results atomically and recover incomplete settings on resume.

No fixed wall-time cutoff is imposed on the full point run in the default plan. Long-running jobs should checkpoint and report progress. A resource failure is a technical interruption, not a scientific negative result.

## 7. Output contract

All outputs live under a fresh run directory. Suggested files:

| File | Required content |
| --- | --- |
| `run_manifest.json` | Source/cache identities, effective config, versions, start/end, execution scope |
| `validation/input_checks.json` | All data, schema, option and probability checks |
| `validation/splits.json` | Actual fitting/evaluation IDs, subjects and ten seeds |
| `validation/numerical_checks.json` | Solver feasibility, endpoint and dominance checks |
| `validation/timing_pilot.json` | Fixed pilot cases, times, memory, projected workload |
| `candidate_fits.parquet` | 10,240 candidate records with selected columns, weights, training objective, status |
| `candidate_evaluations.parquet` | Held-out candidate metrics, including full group and singleton references |
| `strategy_split_results.csv` | 1,920 rows: 8 targets × 2 families × 10 splits × 4 budgets × 3 strategies |
| `heldout_question_losses.parquet` | Per-question TV and top-option agreement, already averaging its four conditions with declared masses |
| `main_results.csv` | 32 wide rows: target × family × budgets 2 and 3; strategy errors, absolute gains, relative gains, agreement |
| `all_budget_results.csv` | 64 wide rows: target × family × all four budgets |
| `macro_results.csv` | All targets/families/splits equally weighted; explicit primary k=2 and secondary k=3 |
| `frozen_fit_resampling.csv` | 2,000 replicate contrasts with shared question multiplicities |
| `descriptive_intervals.csv` | Percentiles with `includes_selection_and_fitting_uncertainty=false` |
| `figures/reference_selection.pdf` | One compact main figure |
| `figures/reference_selection.png` | Readable preview of the same figure |
| `REPORT.md` | Plain-language findings, negative/null cases, limitations and exact commands |
| `XWVG_RESPONSE_DRAFT.md` | Completed-results paragraph grounded in this run only |
| `status/complete.json` | Written only after every required setting and check completes |
| `status/blocked_inputs.json` | Exact unresolved files/fields if empirical execution is blocked |

The 1,920 strategy rows include the random strategy as an expectation over subsets; its selected subset and weight vector are null, not invented. Retain its underlying candidate records. The expected held-out per-question strategy table has 537,600 rows if represented as one row for each of the 1,920 settings × 280 questions.

Use at least these columns in each strategy row: protocol, target, family, split_seed, k, method, selected_peers, selected_weights, training_tv, evaluation_tv, top_option_agreement, fitting_questions, evaluation_questions, solver_complete, input_fingerprint. For random expected performance, training_tv also means the average of candidate training errors.

Use `nearest_minus_collective_tv` and `random_minus_collective_tv` as explicit gain names. Positive means collective selection is better. Compute the primary relative reduction from macro-average errors, not an average of cell-specific percentages. If a baseline denominator is zero, report the absolute difference and make the relative percentage null.

## 8. Figure and report

A suitable main figure has two panels: (a) macro-average held-out TV at k=1,2,3,7 for the three strategies; (b) per-target primary k=2 gains for both families. Include the complete target/family results in the table. Use ordinary numeric x positions for the budgets; do not imply that missing k=4,5,6 were measured.

The caption must describe the probability interface, endpoint weights, overlapping splits and descriptive interval scope. Show absolute error levels as well as gains. Use an independent plot file suitable for a paper; do not create illustrative empirical points.

The report answers these questions in order:

1. At k=2, how much did collective selection change held-out TV relative to nearest-k?
2. At k=3, does the secondary comparison agree?
3. Which targets/families benefit or lose? Are the differences substantial in absolute TV?
4. Does top-option agreement move in the same direction?
5. Are the selected peers actually different? Report selection frequencies without using them to choose a new headline task.
6. What is the exact uncertainty scope, computational cost and remaining limitation?

The response draft can state that a concrete selection decision improved only if the results show that. It should not claim better task accuracy, safe model removal, latency savings, free-text replacement, intrinsic model uniqueness, or an independent new benchmark. No outcome is a prerequisite for technical completion.

## 9. Commands

The following bundled commands exist now. Run them from the extracted plan directory. The small numerical reference requires the packages listed in `requirements-reference.txt`; the receiving Codex must record the versions actually used.

```bash
python3 tools/preflight.py --repo-root /path/to/behavioral-coverage-audit \
  --raw-root /path/to/original/experiment/parent \
  --out /path/to/new/run/preflight.json

python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The preflight also accepts `--balanced-cache`, `--rotations-cache`, and `--questions` as explicit path overrides. Exit 0 means the required files are present, **not** that Parquet rows or historical cache identity have been validated. Exit 2 identifies missing inputs or a required CPU aggregation step. Exit 3 identifies changed inspected source or a mismatch against a recorded historical digest; inspect and document the difference before proceeding. An updated local checkout may legitimately differ from the inspected snapshot, but it must not be silently claimed to match.

After implementing the new package, support this execution pattern:

```bash
PYTHONPATH=/path/to/repo/experiments/llm/analysis/xwvg_reference_selection/src \
python3 -m bca_reference_selection.pipeline \
  --repo-root /path/to/repo \
  --cache /path/to/V23/outputs/analysis/permutation_averaged_semantic_scores.parquet \
  --questions /path/to/V2/data/mmlu_pro_selected_560.jsonl \
  --config /path/to/plan/config/experiment.json \
  --output-root /path/to/new/run \
  --stage all --workers 4 --resume
```

Resolve real paths from the current environment. Do not run these illustrative `/path/to` strings literally, and do not assume that `make analysis` implements this task.
