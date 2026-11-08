# Code and data map

This document maps the proposed reference-selection experiment to the actual public source snapshot inspected for this package. Paths below are relative to the original behavioral-coverage repository. The proposed implementation location is `experiments/llm/analysis/xwvg_reference_selection/src/bca_reference_selection/`; that path is a new addition, not an existing repository component.

## 1. What the public repository does and does not contain

The public snapshot includes inference and analysis source, fixed configurations, model/dataset identities, paper tables, and selected fitted-result files. It does **not** include the question-level LLM probability caches needed to fit new reference subsets.

The repository inventory found:

| Item | Public snapshot | Meaning for this experiment |
|---|---|---|
| `outputs/raw_scores/*/shard_*.parquet` | No matching files | Original question/model score rows must come from the existing experiment workspace. New model inference is outside this package's default task. |
| `permutation_averaged_semantic_scores.parquet` | Not present | The most convenient balanced full-option cache is produced by the source, but is not bundled. |
| `mmlu_pro_selected_560.jsonl` | Not present | Its checksum and dataset identity are bundled; its full question records are external. |
| `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet` | Present | This is a 1,600-row table of fitted outcomes, not the per-question model output tensor. |
| `data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv` | Present | A scalar-column export of those fitted outcomes. It cannot support new subset fitting. |
| `data/frozen/llm/v23/*.csv` | Present | Final aggregate tables. They cannot reconstruct missing question-level probabilities. |
| `manifests/external_design_inputs.tsv` | Present | Hash references for external designs, prompts, and selected-question data. |
| `manifests/external_analysis_inputs.tsv` | Present | Hash references for cached fits and bootstrap outputs, not a bundled raw-score dataset. |

This agrees with `data/README.md`, `experiments/llm/README.md`, `docs/KNOWN_LIMITATIONS.md`, and `docs/FULL_INFERENCE_REPRODUCTION.md`. The existing `make analysis` route replays saved fits and bootstrap distributions; it does not generate new fits from question-level scores. A successful replay is therefore not completion of the proposed experiment.

Do not reconstruct probabilities from published mean errors, weights, or figure coordinates. Multiple underlying score tensors can produce the same summaries.

## 2. Locked primary input: the saved balanced full-option endpoint cache

### Exact producer and location

Existing source:

`experiments/llm/analysis/v23/src/pier_llm_v23/analysis.py`

Function:

`assemble_permutation_scores(root)`

External output produced by that function:

`PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/outputs/analysis/permutation_averaged_semantic_scores.parquet`

This file already preserves the **entire option distribution**, even though the main paper subsequently selects only the reference-answer coordinate. If this cache is available and passes validation, the new full-option experiment requires no new model inference.

### Cache identity and required companion metadata

No historical SHA-256 or byte size for either `permutation_averaged_semantic_scores.parquet` or `permutation_wise_semantic_scores.parquet` was found in the downloaded repository's `manifests/` or `experiments/llm/manifests/` archives. Record the SHA-256 and size of the supplied cache now, but label this as **current-input fingerprinting**, not verification against a historical cache digest. Historical prompt-manifest hashes and analysis-result hashes do not establish the byte identity of these score caches.

The default preflight requires both the cache (`--cache`) and the original selected-question file (`--questions`):

`PIER_LLM_ECOSYSTEM_GOLDMINE_V2/data/mmlu_pro_selected_560.jsonl`

The selected-question file has an archived size of 458,957 bytes and SHA-256 `9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6`, recorded in `manifests/external_design_inputs.tsv` and `experiments/llm/manifests/v2/data/manifests/dataset_source.json`. Match all cached question IDs, categories, valid option counts, and reference-answer indices to this file. Do not silently replace this historical sample check with a count-only check.

The original protocol metadata are available in `experiments/llm/analysis/v23/configs/experiment_v23.json`, `experiments/llm/analysis/v23/configs/resolved_models_v23.json`, and `experiments/llm/manifests/v23/status/prompt_manifest_status.json`. These specify the eight-model roster, clean and maximum-dose conditions, three tracks per family, and the historical option-count distribution: 3 options for 1 question, 4 for 33, 5 for 2, 6 for 1, 7 for 3, 8 for 22, 9 for 50, and 10 for 448. Validate the cache against these metadata; do not infer dose policy from file names.

### Relevant columns

| Column | Type and role |
|---|---|
| `model_id` | Exact checkpoint identifier; use it as model identity. |
| `base_question_id` | Question identity, stored/converted as a string. Split by this field. |
| `category` | One of the 14 subjects; use for stratification. |
| `semantic_condition` | Clean or a particular maximum-dose family/track condition. |
| `condition` | Used to identify clean rows. |
| `family` | `irrelevant_context` or `content_deletion` for stressed rows. |
| `dose` | 0 for clean; 512 words for maximum irrelevant context; 0.4 for maximum deletion. |
| `track` | Null for clean; one of three matched intervention realizations for stressed rows. |
| `option_count` | Number of valid options for this question. Do not assume every question has ten options. |
| `gold_semantic_option` | Reference-answer index in the original semantic ordering. Needed only for separately reported task-outcome metrics. |
| `permutation_averaged_semantic_probabilities` | List of `option_count` probabilities in the original option order. This is the input for full-option reconstruction. |
| `permutation_averaged_gold_probability` | The corresponding reference-answer probability. Retain for scalar sensitivity analysis; do not substitute it for the vector. |
| `permutation_averaged_semantic_argmax` | Argmax of the averaged vector. This is a score-based choice, not a generated answer. |

Expected historical dimensions are eight models, 560 questions, and seven conditions per question: one clean condition plus three tracks from each of two stress families. Thus the complete balanced cache has `8 * 560 * 7 = 31,360` rows. A single-family analysis has four rows per question/model: one clean and three stressed.

For that single-family endpoint design, give the clean row weight `1/2` and each stressed track weight `1/6`. Averaging rows equally would incorrectly give stressed conditions three times the total clean weight.

### CPU reconstruction from the complete saved rotation cache

If the averaged file is missing, first look for the complete saved cache:

`PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/outputs/validation/permutation_wise_semantic_scores.parquet`

`assemble_permutation_scores(root)` writes this file after combining reused rotation-zero rows with new V2.3 rotations. Its column names are verified from that writer and the V2.3 score producers:

- Identity: `schema_version`, `model_id`, `model_revision`, `prompt_id`, `base_question_id`, `category`.
- Condition: `semantic_condition`, `condition`, `family`, `dose`, `track`.
- Rotation and options: `permutation_id`, `option_count`, `gold_semantic_option`, `semantic_remapped_probabilities`.
- Additional provenance: `source_v2_prompt_id` for reused rows and `score_origin` after assembly.

The semantic probability column is already in original option order. Do not remap it a second time. The complete historical cache has 36,946 rotation rows per model, or 295,568 rows for eight models, according to the archived `prompt_manifest_status.json`. The historical SHA-256 and file size for this cache were not found; record its current fingerprint and validate its schema, keys, and contents without claiming a verified historical byte match.

Reconstruction needs only CPU operations: read the Parquet, check the original selected-question identities, group by `(model_id, base_question_id, semantic_condition)`, sort by `permutation_id`, require exactly `0, ..., option_count - 1`, and average `semantic_remapped_probabilities` across those rotations. Require identical category, condition, family, dose, track, option count, and gold index within each group. Validate every vector before averaging. Carry that metadata into the averaged rows and compute the derived gold coordinate and argmax. The result must contain exactly 31,360 rows with complete model/question/condition coverage.

Use the aggregation loop in `analysis.py::assemble_permutation_scores` as the reference implementation. There is no separate historical public aggregation function accepting an already assembled rotations DataFrame. A narrow new aggregation adapter should reproduce that loop without calling the whole postprocess or launching inference. Save the reconstructed cache as a new file, record both input and output fingerprints, and compare it numerically with a saved averaged cache if both are available. Parquet byte equality is not required for a newly serialized derivative.

### Reconstruction from original score shards, if needed

If neither saved cache exists but all original score shards are available, the same historical producer combines:

1. New V2.3 score shards under `PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/outputs/raw_scores/<model_slug>/shard_*.parquet`.
2. Validated rotation-zero rows reused from the V2 score shards.
3. `data/permutations/score_semantic_manifest.jsonl`, which supplies rotation mappings and source prompt identities.

Relevant existing functions are `_load_stage2_scores`, `load_rotation0_scores`, and `assemble_permutation_scores` in the V2.3 `analysis.py`. The V2.3 row schema is `pier_v23_candidate_score_v1`.

Full probability vectors appear in `semantic_remapped_probabilities`; rows are grouped by `(model_id, base_question_id, semantic_condition)`, sorted by `permutation_id`, and averaged only after the visible labels have been remapped to original semantic options. The producer requires exactly rotations `0, ..., option_count - 1` for every group.

Prefer loading a validated saved averaged cache. Calling the complete historical assembly pipeline just to read it introduces extra dependencies, including historical completion and generation-output checks. If reconstruction from rotation rows is necessary, use a narrow adapter that preserves these checks for score identity, mapping, and rotation completeness, and documents the new cache's hashes.

## 3. Optional sensitivity only: canonical five-dose raw scores

The locked primary experiment uses balanced full-option endpoints. Canonical five-dose scores are a separately labeled optional sensitivity, never an automatic fallback when the primary cache is absent. Fresh model inference is outside the default task.

### Exact source and external files

Inference writer:

`experiments/llm/inference/v2/src/pier_llm/inference.py::score_prompt_batch`

External shards:

`PIER_LLM_ECOSYSTEM_GOLDMINE_V2/outputs/raw_scores/<model_slug>/shard_*.parquet`

Each shard has a corresponding `shard_*.meta.json` sidecar. The row schema is `pier_raw_label_scores_v2`.

The historical complete scoring collection contains 125,440 rows: eight models with 15,680 rows each. The total includes clean, two intervention families, and option-permutation control rows. A canonical single-family audit uses one clean row plus four nonzero doses times three tracks: `13 * 560 = 7,280` rows per model. Exclude option-permutation controls from that canonical audit.

Relevant raw columns are:

- `model_id`, `model_revision`, `prompt_id`, `base_question_id`, `category`;
- `condition`, `family`, `dose`, `track`, `permutation_index`;
- `candidate_log_likelihoods`, `probabilities`, `semantic_probabilities`;
- `presented_to_original`, `answer_index`, `semantic_answer_index`;
- `semantic_argmax_index`, `gold_probability`, `prompt_token_count`.

`probabilities` are in visible-label order. `semantic_probabilities` are already remapped to original option order. `answer_index` and `semantic_answer_index` must not be interchanged.

### Validated loading and alignment

Existing implementation:

`experiments/llm/analysis/v22/src/pier_llm_reanalysis_v22/data.py`

| Function | Actual behavior | Reuse guidance |
|---|---|---|
| `load_validated_inputs()` | Checks V2 completion markers, exact roster, shard schema/checksum/row count, duplicate `(model_id, prompt_id)` keys, 125,440 total rows, and 560 selected questions with 40 in each category. | Reuse validation rules, but make roots explicit. Its exact historical counts intentionally reject a new roster or a different study size. |
| `semantic_probabilities(row, ...)` | Softmaxes candidate log likelihoods, optionally applies temperature or label-bias adjustment, then remaps visible to original options. | Use unmodified probabilities for the primary experiment. Calibration would change the observed response and must be a separate condition. |
| `track_response_frame(..., representation="probability_vector")` | Keeps clean and the named stress family, creates semantic vectors, normalizes null clean tracks to `-1`, and checks one clean/three stressed rows per question/dose. | Useful specification for the canonical adapter. |
| `aligned_track_arrays(frame, target, peers)` | Matches by question, category, dose, and track; checks missing peers and option-count agreement; zero-pads vectors. | Direct template for input validation and tensor layout. Add explicit cross-model reference-answer checks for task-outcome analysis. |
| `design_row_weights(metadata)` | Gives clean rows mass 1 and each nonzero-dose track mass 1/3, with each question-dose carrying equal mass. | Correct for five-dose fitting; not the balanced endpoint weights. |
| `vector_dose_metrics` / `vector_overall_tv` | Computes TV, Jensen–Shannon divergence, top-1 agreement, and correctness agreement; averages within dose then equally across doses. | Reuse metric definitions with explicit validity masks. |

Tensor layout returned by the vector alignment code:

- Target: `[n_observations, max_options]`.
- Peer design: `[n_observations, max_options, n_peers]`.
- Metadata includes `option_count` and `semantic_answer_index`.
- Prediction: `np.tensordot(design, weights, axes=([-1], [0]))`.

Zero padding is a storage device. Use only the first `option_count` entries for normalization, TV, and argmax. Do not treat padded positions as valid answers.

The two files in `experiments/llm/analysis/vector_reference/` are preserved paper-reference source copies. They import sibling utilities that are not supplied as a complete package in that directory. Use the complete V2.2 package or a new explicit adapter; do not assume `vector_reference/data.py` is a standalone runnable loader.

## 4. Question identity and split reproduction

Both implementations below use deterministic subject-stratified 50:50 splits:

- V2.2 `data.py::stratified_question_split` and `fixed_splits`.
- V2.3 `core.py::fixed_split` and `fixed_splits`.

Within a category, identifiers are ordered by `(stable_u64(seed, category, question_id), question_id)`, and split at the midpoint. The `stable_u64` algorithm is SHA-256 of the parts joined with `\x1f`, taking the first eight digest bytes as an unsigned big-endian integer. Do not replace this with Python's process-randomized `hash()` or a generic shuffled split while claiming historical split reproduction.

The historical split seeds are 20260828 through 20260837. Each gives 280 fitting and 280 evaluation questions. All doses, tracks, models, and option rotations of a question must stay on one side. The ten splits reuse the same 560 questions and are not independent datasets.

The selected dataset has pinned revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` of `TIGER-Lab/MMLU-Pro`, test split. The selected-question JSONL checksum is `9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6`. Both are recorded in the original manifest; do not fetch a floating dataset revision and call it the original sample.

A complete probability cache theoretically supplies enough `(base_question_id, category)` information to reproduce the splitting algorithm. This package nevertheless requires the original selected-question JSONL in the default preflight so that the original sample's identity, categories, options, and reference answers are checked against its archived SHA-256. If it is missing or fails that check, stop the default run; do not silently construct a replacement selected-question file from the cache.

Any inner training/validation partition for reference-subset selection must be formed entirely within the outer fitting questions. Save the resulting IDs and seed. The outer evaluation questions must not select subsets, budgets, solver settings, or result thresholds.

## 5. Convex solvers: reusable parts and hazards

### Squared-error solver

`experiments/llm/inference/v2/src/pier_llm/solver.py::fit_simplex_projection`

This accepts a two-dimensional scalar design or the three-dimensional vector design above. `_as_design` flattens observation/option coordinates. It enumerates simplex faces to find the squared-error fit, then obtains a deterministic minimum-norm representative while preserving its fitted response. It returns `ProjectionResult`, including feasibility and projection-preservation diagnostics.

Important implementation limits:

- The enumeration code rejects more than 15 peers. The present seven-peer target reconstruction lies within that limit.
- Enumeration costs grow exponentially. Repeatedly fitting every subset with this solver repeats face enumeration. Cache fits by `(split, target, subset, design, objective)` or use a verified small convex solver for the new analysis.
- For vector fitting, repeat the **same question/condition weight** across valid option coordinates. The desired objective sums coordinate losses per observed vector; do not divide each vector by its number of valid options unless that different objective is declared.
- The solver's flattened mean-loss scale is not the same numerical scale as mean per-vector squared distance. Its minimizer is unchanged under a constant rescaling, but do not mix the returned objective values when reporting errors.

The V2.2 `data.py::fit_projection` already applies square-root row weights and computes a separately normalized per-observation objective. It is a useful reference for weighted vector MSE.

### Absolute-error and total-variation fitting

`experiments/llm/analysis/v23/src/pier_llm_v23/core.py::fit_mae_simplex`

The existing function uses `scipy.optimize.linprog(method="highs")` with slack variables for absolute errors. It is intended for scalar arrays. Its constraint matrix contains **dense** `np.eye(rows)` blocks.

**Do not flatten full option vectors and pass them directly to this dense implementation.** For 280 fitting questions with four endpoint rows and up to ten options, flattening approaches 11,200 residual coordinates. A dense `2r` by `(r + p)` constraint matrix at this size alone approaches two gigabytes, before solver copies or intermediate allocations.

For the new TV-matched experiment, implement the same LP with sparse matrices:

- Create residual variables only for valid option coordinates.
- Use `scipy.sparse.eye`, `hstack`, and `vstack` for the signed constraints.
- Use `1/2` times the observation weight on each coordinate's absolute-error slack; normalize by total observation weight for reported mean TV.
- Enforce `w >= 0` and `sum(w) = 1`.
- Check solver success, finite coefficients, minimum weight, sum-to-one error, and objective recomputed from the unrounded solution.
- Log any bounded numerical repair and its change to fit/evaluation predictions. Do not silently clip materially invalid weights.

Single-peer selection and group fitting must optimize the same primary loss. The existing canonical vector audit selects its single peer by TV while fitting the group by vector MSE; it should not be copied unchanged as a TV-matched baseline.

### Existing numerical checks

V2.3 `core.py` provides `validate_or_repair_simplex`, `project_probability_simplex`, `weighted_mae`, `weighted_mse`, and `select_single_peer`. Read the repair policy before reusing it. The historical accepted tiny-feasibility repairs are not permission to normalize arbitrary infeasible new solutions.

V2.3 `fit_mse_simplex` first tries a hard-coded original V2.2 source path, then falls back to SLSQP if that path is unavailable. This can silently change the solver across machines. The new package should choose and record its solver explicitly.

## 6. Bootstrap reuse and limitations

V2.3 `core.py::common_bootstrap_multiplicities(selected, replicate, seed)` draws one within-category multiplicity vector over base questions. This is the right unit of resampling for correlated doses/tracks/rotations.

If the experiment uses repeated overlapping splits, reuse one multiplicity vector across all those splits in each replicate. Do not compute uncertainty by treating ten split means as ten independent samples.

For uncertainty that includes reference-selection variation, rerun the declared fitting and selection procedure within each bootstrap replicate, then evaluate with the corresponding question multiplicities. Resampling only saved evaluation errors conditions on the selected references and must be labeled accordingly. Existing paper bootstrap summaries cannot be transferred to new selected subsets.

## 7. Path and execution traps

The source snapshots retain original absolute paths. V2.2 `utils.py` defines hard-coded `LOGICAL_ROOT` and `PHYSICAL_ROOT`; V2.3 has a `PIER_V23_ROOT` override but still contains the fixed V2.2 solver-import path described above. Reading these files is not enough to make historical entry points portable.

For a full reproduction, `scripts/stage_experiments.py` creates a **new** external workspace and records path adaptations in `PATH_MIGRATION.json`. It does not modify an existing run. The `--raw-root` option copies only 44 hash-checked design/manifest files, approximately 1.55 GB, mostly rendered prompt text. It does not copy raw score shards or model weights.

The original formal GPU entry point expects eight B200 GPUs. This requirement belongs to that historical inference launcher; it is not a requirement for CPU analysis of already collected scores. Do not rerun the full model campaign when a validated vector cache is available.

The new experiment package uses explicit `--cache`, `--questions`, and `--output-root` arguments, loads only the needed caches, and leaves the original frozen outputs unchanged. Both primary inputs are required by default. New model inference is outside the default task. Use `pyarrow` or another standard Parquet implementation for nested probability arrays. The scalar CSV replay reader is not a loader for those arrays.

## 8. Required preflight report before new results

Produce one machine-readable report containing:

1. Input path, byte size, current SHA-256, source schema, and protocol name. Record the selected-question historical checksum match separately. For the two score caches, state that no historical cache checksum or size was found in the inspected archive; do not conflate a current fingerprint with historical identity verification.
2. Model roster and revisions where available; 560 distinct question IDs and 14 category counts for a historical-cache run.
3. Duplicate/missing-key checks and alignment of categories, reference answers, and valid option counts across models.
4. Probability finiteness, nonnegativity tolerance, normalization error, and any logged repairs.
5. Complete condition/track counts and, when rebuilding averages, complete cyclic rotation sets.
6. Saved outer and inner split question IDs, with zero overlap where required.
7. Tensor shape, valid-option mask, row-weight totals, selected primary loss, solver/version, and numerical tolerances.
8. Whether the primary cache was supplied directly or rebuilt on CPU from complete saved rotations/original score shards. A results-summary table is not an acceptable source for new fitting. Fresh model inference is outside the default task.

Proceed to the experiment only after this report passes. If the needed cache is absent, report the exact missing path/schema and stop before producing empirical selection curves. That outcome is a data prerequisite failure, not a negative scientific result.
