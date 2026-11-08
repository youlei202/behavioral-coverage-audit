# Fresh endpoint scoring, only if old probabilities cannot be recovered

This is a concrete implementation specification for the receiving Codex. The proposed driver below is **not yet implemented in this addendum**. Preserve the completed reference-selection implementation. First try the historical V23 results ZIP: its packaging rule includes the analysis and validation probability caches.

## 1. Exact scope

Collect only the locked eight checkpoints on the original 560 questions and seven semantic conditions: one clean condition, three 512-word irrelevant-context tracks, and three 0.4 content-deletion tracks. For every condition, score every cyclic option rotation, including rotation zero.

Historical option counts are 3:1 question, 4:33, 5:2, 6:1, 7:3, 8:22, 9:50, and 10:448. The relevant workload is:

| Quantity | Per model | All eight models |
|---|---:|---:|
| Balanced question/condition vectors after averaging | 3,920 | 31,360 |
| Complete cyclic-rotation prompt rows | 36,946 | 295,568 |
| Rotation-zero rows, historically reused from V2 | 3,920 | 31,360 |
| Candidate-label continuation sequences | 357,196 | 2,857,568 |

The final row counts candidate continuations under the original scorer, not independent GPU batches or free-text generations. Actual runtime must be measured. Do not call this a cheap CPU-only recovery. Once collected, the same cache supplies every split, target-selection setting and budget in the utility analysis.

No model training, intermediate-dose model scoring, free-text generation, original-paper refitting/bootstrap campaign, vision experiment, or traffic experiment is needed.

## 2. Prepare the exact dataset and prompt design on CPU

Use the original dataset revision and selection checks in `CODEX_CONTINUE.md`. To rebuild the complete original prompt files, use:

`experiments/llm/inference/v2/src/pier_llm/data_prep.py::build_from_rows(raw_rows, new_root, dataset_revision)`

Copy the original `configs/experiment.json` and `configs/stopwords_en.txt` to the new input workspace. Load `TIGER-Lab/MMLU-Pro`, test split, with the explicit revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`. The old `prepare()` queries a current dataset revision and must not be called unchanged.

Keep all 12,032 source rows in their original order during normalization. The irrelevant-context construction uses the unselected source pool and TF-IDF; the selected 560 questions alone do not define those interventions. Reuse the original text-generation functions and compare outputs with `dataset_source.json` hashes. Generating all original prompt texts on CPU is acceptable; only endpoints need GPU scoring. This permits identity checks against the existing full-design hashes.

Important expected hashes:

```text
selected560:
9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6

V2 scoring_prompts.jsonl:
199bd992d10009ac04bcdf212e13023f08ddc6d2f2abeaa26597855f5c2b5b3a
```

Use `experiments/llm/analysis/v23/src/pier_llm_v23/prepare.py::build_manifests` as the semantic-rotation specification. Make the source input root explicit in a new adapter. It creates conditions `C0`, `I0/I1/I2`, `D0/D1/D2` and rotations `0,...,option_count-1`.

The original manifest sets `inference_required=False` for rotation zero because its scores came from V2. In a fresh run, **score all rotations**. Do not filter the new scoring task by that old reuse flag. Keep the unchanged semantic design as provenance, and write a separate execution manifest with `rotation0_reuse=False` and `inference_required=True`. Record why its execution metadata differs from the historical manifest.

## 3. Locate pinned checkpoints and render prompts

Read the exact roster and model revisions from the completed analysis configuration and `experiments/llm/analysis/v23/configs/resolved_models_v23.json`. The latter contains the original adapters and renderer metadata. Relocate local paths; preserve model IDs and revisions.

Use existing complete local checkpoint snapshots when available. If downloading is necessary, request the exact pinned revision and report any real access restriction. The original V2 `model_prestage.run()` may resolve a current revision and allow a different fallback model; do not invoke it unchanged for this locked roster.

Reproduce the model-specific chat-template handling in V23 `prepare.py` and `inference.py`. Preserve candidate suffixes, thinking/template arguments, model class adapter, tokenizer behavior and precision. Do not replace full candidate-sequence scoring with a first-token approximation.

### Granite's date-dependent system message

The historical Granite chat template has a dynamic date. The original V23 preparation obtained a fixed system message from V2 rendered prompts. With those files absent, rebuild the fixed message from the pinned tokenizer template and the historical rendering date, then check its archived message hash:

```text
Historical date: 2026-08-28
System-message SHA256:
17278797738c84141b95f25ea868cb045e748327147dbf757795f9b4f19567c2
```

Check the actual archived compatibility record and renderer before assuming any template parameter name. Never allow the date to change between batches. If the original wrapper cannot be matched, record the exact fixed wrapper used in this fresh measurement, assign a new prompt/protocol fingerprint, and describe the difference in the report. Do not assert byte-identical restoration of the old interface or compare new errors with old paper values as if inputs were identical.

## 4. Score only the required candidate probabilities

Reuse the mathematical scoring behavior of:

`experiments/llm/analysis/v23/src/pier_llm_v23/inference.py::score_prompt_batch`

For each prompt, score each entire continuation ` (A)`, ` (B)`, and so on, using summed token log likelihood without length normalization. Normalize across valid labels, then use the original mapping to produce `semantic_remapped_probabilities`. Save all valid options, not just the gold-answer probability.

Implement a separate small runner around this scoring function. Required changes from the old orchestration are:

1. Render all prompts without requiring old rendered-file caches.
2. Score rotation zero with the same function as all other rotations.
3. Run only candidate scoring; do not require or generate the old free-text diagnostics.
4. Schedule the eight checkpoints over the devices actually available; record their identities and memory.
5. Preserve shard checksums, duplicate/missing-row checks, atomic writes and resume.
6. Adapt batch sizes using measured memory or OOM recovery, without truncating prompts, dropping models, quantizing weights, or changing the candidate likelihood definition silently.

The old launcher requires eight B200 GPUs and starts additional generation jobs. Those are properties of that historical runner. The new runner must validate its actual resource needs; it must not forge old readiness/completion markers or bypass real access controls. Use the pinned scoring precision. If one model does not fit the available setup, report that model and the measured constraint, then prepare a compatible command for the available experiment host.

Use a deterministic pilot fixed before viewing comparison results. Cover every model, all seven semantic conditions, rotation zero and nonzero rotations, and a range of prompt lengths. Record peak memory, candidate-sequence counts, throughput and a full-run projection. Continue all required scores when the environment supports them. Checkpoint continuously.

## 5. Assemble and validate the new balanced cache

Use the averaging logic of V23 `analysis.py::assemble_permutation_scores`, with explicit new input/output paths. Do not call its historical wrapper unchanged: it loads rotation-zero scores from V2 and validates unrelated generation outputs.

For every `(model_id, base_question_id, semantic_condition)`, require each rotation exactly once and complete metadata agreement. `semantic_remapped_probabilities` is already in semantic order; do not remap twice. Average the complete distributions across rotations and save:

`outputs/analysis/permutation_averaged_semantic_scores.parquet`

Expected total: 31,360 rows. Preserve the columns expected by the already implemented analysis, including `permutation_averaged_semantic_probabilities`, model and question identities, semantic condition, family, dose, track, option count and reference-answer index.

Record these as **fresh measurements**, with input/output hashes, exact revisions, precision, scoring dependencies and hardware. Run the established vector validation and original question splits, then resume the reference-selection analysis. No fitting-stage change is needed merely because the probabilities were freshly collected.

## 6. Proposed driver interface

The receiving Codex should implement an equivalent interface. These commands are a specification, not an existing executable in this package:

```bash
python scripts/reacquire_balanced_endpoints.py prepare \
  --source-repo /actual/repository \
  --run-root /new/endpoint-reacquisition

python scripts/reacquire_balanced_endpoints.py score \
  --run-root /new/endpoint-reacquisition --resume

python scripts/reacquire_balanced_endpoints.py assemble \
  --run-root /new/endpoint-reacquisition
```

Reuse the existing utility pipeline's real CLI after this stage. Do not invoke the historical full B200 shell scripts or full CPU postprocess merely to produce this cache.
