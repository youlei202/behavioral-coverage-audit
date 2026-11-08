# XfqA larger-model extension

This component extends the measured roster with Qwen2.5-32B-Instruct,
QwQ-32B and Gemma-3-27B-IT. It uses the supplied eight-model probability cache
and original 560 questions unchanged. Original paper sources and completed XWVG
artifacts are read-only inputs; all writes use `outputs/xfqa_large_models/<run_id>/`.

The executable driver is `run.py`. Its phases are `preflight`, `prepare`,
`resolve`, `pilot`, `score`, `assemble`, `analyze`, `bootstrap`, `report`,
`verify`, `package`, and `all`. The `all` phase executes the actual stages.
`pilot` first verifies/downloads pinned assets and renders native templates.
`score` requires the completed outcome-blind pilot and maximum-workload probe.
No previously measured model is rescored.

## Run

Extract the supplied plan under a new run's `plan/` directory, retaining its
manifest and `seed_inputs/`. From the repository root:

```bash
/tmp/xwvg-venv/bin/python \
  experiments/llm/analysis/xfqa_large_models/run.py \
  --repo-root "$PWD" \
  --plan-root "$PWD/outputs/xfqa_large_models/20261108/plan" \
  --output-root "$PWD/outputs/xfqa_large_models/20261108" \
  --scoring-python /tmp/xwvg-scoring-venv/bin/python \
  --analysis-python /tmp/xwvg-venv/bin/python \
  --device cuda:0 --workers 4 --phase all --resume
```

These Python locations name the existing working environments, not a dependency
on a particular installation path. Supply explicit replacements on another
machine. The run saves complete analysis/scoring package locks in `runtime/`.
The scoring environment requires the compatible BF16 Torch/Transformers stack;
the analysis environment uses NumPy, SciPy, pandas, PyArrow, Matplotlib and psutil.

Without a placement file, one new model runs at a time on `--device`. Explicit
parallel placement is enabled with `--placement /absolute/path/placement.json`.
The file contains `slots`, each with `model_id`, `device`, `scoring_python`, and
an expected physical `expected_gpu_uuid`; remote slots also have `host`, `port`,
`user`, and `identity_file`. Duplicate devices/UUIDs are refused. Only use nodes
authorized for the current run. The actual 20261108 placement assigns the three
new models to three distinct B200s on the newly supplied node. Remote execution
uses the shared filesystem and local model assets, without forwarding HF tokens.

Run any individual phase by replacing `--phase all`. `--model-id` permits a
single declared model's pilot/score stage, but is rejected with `all`. Scientific
analysis and completion require all three new models. `--execution-manifest`
can select an already-prepared model-independent manifest; its original hash
and complete rotation coverage are still enforced. `--cache` can explicitly
select the completed eleven-model averaged probability file for CPU analysis.

## Implementation and evidence

| File | Responsibility |
| --- | --- |
| `run.py` | CLI entry point, local/remote stage orchestration, locking, resume checks, and final completion gates. |
| `preflight.py` | Package checksums, deep seed validation, original splits, the 5,840-job inventory, and preservation hashes. |
| `scoring.py` | Pinned model assets, native templates, complete-continuation BF16 scoring, resource reserves, atomic shards, and cyclic averaging. |
| `analysis_data.py` | Validated response tensors, model panels, question splits, and analysis input identities. |
| `analysis_math.py` | Simplex MSE fitting, sparse MAE/TV fitting, and direct numerical optimality checks. |
| `analysis.py` | Point fits, matched single-peer selection, peer removals, roster comparisons, and per-question losses. |
| `analysis_bootstrap.py` | Primary refit/reselection bootstrap and separate descriptive frozen-fit resampling. |
| `reporting.py` | Result summaries, three figures, manuscript inserts, response draft, and numerical claim evidence. |
| `verify.py` | Independent replay of scores, coefficients, metrics, intervals, report claims, and original-file preservation. |
| `delivery.py` | Portable result archive, hash checks, and full replay from an extracted copy. |
| `resources.py` | Measured GPU/CPU-stage accounting, preserved initial observations, and explicit unmeasured quantities. |
| `progress.py` | Read-only inspection of committed score shards and stage completion records. |

The six test modules cover the corresponding contracts:

| Test file | Coverage |
| --- | --- |
| `tests/test_analysis_math.py` | Fitting definitions, numerical checks, and response handling. |
| `tests/test_analysis_bootstrap.py` | Shared-question resampling, refitting, and resumable bootstrap behavior. |
| `tests/test_scoring.py` | Continuation-token boundaries, pinned assets, resource checks, and immutable staging. |
| `tests/test_driver.py` | Stage dispatch, placement identities, and completion gates. |
| `tests/test_reporting_replay.py` | Reporting evidence, independent replay, portable packaging, and string-valued run identities. |
| `tests/test_resources.py` | Resource evidence, aggregation, idempotence, and tamper detection. |

Synthetic fixtures are implementation checks and never empirical result substitutes.

The run's `inputs/` contains byte-identical read-only seed copies, original
splits and the declared job inventory. `data/permutations/` contains the exact
semantic/execution manifests and new-model rendered prompts. Atomic raw scores
are retained under `outputs/raw_scores/`. Required analysis tables and written
outputs are under `analysis/`. Per-model status and bootstrap chunks preserve
completed work after interruption. Actual commands are recorded in
`RUN_COMMANDS.sh`; model, source, input and runtime identities accompany results.

The primary uncertainty uses 1,000 shared within-subject question resamples,
refitting the prescribed MSE mixtures and reselecting single peers. Separate
2,000-replicate frozen-fit descriptive intervals exclude fitting/selection
uncertainty. The ten overlapping splits are averaged within each replicate.
The response concerns recorded candidate-label probabilities, including QwQ's
unaltered native reasoning prefix, rather than generated reasoning quality.

Use `--resume` only with unchanged locked inputs, model revisions, scoring
implementation and placement identity. Inspect failed-stage records before
retrying. A partial model/analysis/bootstrap result is not a completed run.
The final ZIP contains CPU replay inputs, coefficients, bootstrap evidence,
code, reports and hash inventories; model weights and credentials are excluded.
Packaging verifies archive hashes and executes the full saved-coefficient and
uncertainty replay from an extracted copy. The final completion record is written
only after that replay and the original-file preservation checks pass.

## Completed run and generated files

The completed run is stored locally under
`outputs/xfqa_large_models/20261108/`. It contains 110,838 new rotation-level
measurements, 43,120 merged probability vectors, 5,840 point fits, 1,000 primary
bootstrap replicates with 100,000 actual fits, and 2,000 frozen-fit replicates.
Independent numerical replay and the extracted-bundle replay passed. The
preservation audit confirmed that 18,874 original paper and completed XWVG files
were unchanged.

| Run-relative path | Contents |
| --- | --- |
| `analysis/REPORT.md` | Complete empirical report and interpretation limits. |
| `analysis/full_summary.csv` | All target/panel/loss aggregate results. |
| `analysis/scalar_main_split_results.csv`, `analysis/vector_main_split_results.csv` | Per-split single-peer and group comparisons. |
| `analysis/scalar_directional_pairs.csv`, `analysis/vector_directional_removals.csv` | Declared directional comparisons and refitted partner removals. |
| `analysis/scalar_removal_split_results.csv` | Exhaustive scalar peer-removal results for the expanded roster. |
| `analysis/scalar_roster_expansion.csv`, `analysis/vector_roster_expansion.csv` | Paired original-versus-expanded reference-group comparisons. |
| `analysis/point_fit_coefficients.parquet`, `analysis/per_question_losses.parquet` | Saved coefficients and physical-question replay evidence. |
| `analysis/primary_refit_intervals.csv`, `analysis/secondary_descriptive_intervals.csv` | Separately labelled primary and descriptive uncertainty. |
| `analysis/XfqA_RESPONSE_DRAFT.md` | Response draft grounded in the measured results. |
| `analysis/METHODS_INSERT.md`, `analysis/RESULTS_INSERT.md` | Manuscript-ready methods and results text. |
| `analysis/figures/` | Coverage, directionality, and roster-expansion figures in PNG/PDF. |
| `validation/numerical_replay.json`, `validation/portable_numerical_replay.json` | Original-location and extracted-bundle replay audits. |
| `status/complete.json` | Verified empirical completion record and final archive identity. |
| `XfqA_Large_Model_Results_20261108.zip` | Complete portable result bundle. |

These generated outputs, supplied seed inputs, downloaded weights, and environment
directories are not part of the source commit. The source remains independent of
the later conversation-specific readable exports.

## Progress inspection

Check an active run without loading weights or changing artifacts:

```bash
python3 experiments/llm/analysis/xfqa_large_models/progress.py \
  --output-root "$PWD/outputs/xfqa_large_models/20261108"
```
