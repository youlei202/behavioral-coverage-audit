# Continue the implemented XWVG experiment: obtain real inputs

## Current state

The user reports that the new implementation and its component/constructed-data checks are complete. The reported working tree is:

`/work/Users/leiyo/GitHub/behavioral-coverage-audit`

The reported run directory is:

`outputs/xwvg_reference_selection/20261107_ready/`

Read that run's `REPORT.md`, actual CLI help, effective configuration, and existing source before acting. These paths come from the user's report; this addendum's author could not access that host. Do not recreate the whole implementation or discard existing work.

## Correction to the previous task

The previous plan had an unresolved prerequisite: access to the original question-level probability cache. It also disabled new model inference. Replace that missing-data stopping rule with the sequence below. The fixed scientific comparison remains unchanged. This execution task includes targeted fresh scoring if original outputs cannot be recovered; it does not require a rerun of the full paper's pipeline.

## 1. Establish what was actually searched

Run `tools/locate_inputs.py` from this addendum, passing the actual repository root and any known original data mount with repeated `--root` arguments. It checks the historically recorded `/work/Users/leiyo` and `/work/Lei` roots, the configured `BEHAVIORAL_COVERAGE_RAW_DATA`, and the explicit roots. It records missing roots, inaccessible directories, skipped entries, and limits. Inspect its `--help`; use its real flags.

Inspect relevant result-archive member lists as well as loose files. The historical archive names include:

```text
PIER_LLM_ECOSYSTEM_GOLDMINE_V2_RESULTS.zip
PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS.zip
```

The source repository's `manifests/llm_source_archive_identity.json` records original archive paths. **The V23 packaging implementation copies the entire `outputs/analysis` and `outputs/validation` directories.** Its `raw_scores_omitted: true` flag concerns raw shards, not these analysis caches. Prioritize this archive before considering new model scoring.

The archived V23 ZIP is 109,877,741 bytes and has SHA-256 `ba24123066a16aa29af60fa924589256fd7fe8566ecd9a2017f12dccaeb682a1`. Verify these if that exact original archive is found. Inspect these exact members:

```text
PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS/analysis/permutation_averaged_semantic_scores.parquet
PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS/validation/permutation_wise_semantic_scores.parquet
PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS/MANIFEST.sha256
```

The archive layout drops the external run's `outputs/` layer. A search only for the complete old run-relative path misses an unpacked result package. Check basenames as the supplied locator does. Use the ZIP's internal manifest to verify extracted member hashes. An archive name or packaging rule alone is not proof that a particular file has now been recovered; inspect the actual archive. Extract only relevant files into the new recovery workspace after checking paths, sizes, and identity.

Do not claim the data are deleted when a historical mount or machine is unavailable. Record whether the issue is a missing file, unavailable mount, inaccessible directory, incomplete search, or an archive that does not contain the input. A list of missing basenames without search coverage is insufficient.

## 2. Recover the selected questions independently

Use `tools/recover_selected_questions.py` with the actual `--repo-root` and a new `--output` path. Consult its help for local dataset cache support. The tool uses:

- `TIGER-Lab/MMLU-Pro`, test split;
- revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`;
- original normalization and stable subject-stratified selection code;
- seed `20260828`, 40 questions per each of 14 subjects;
- the original canonical JSONL serialization.

The reconstructed file must match SHA-256:

`9551186193cc86ca1fc275bdf94b1a43ce516e6df2bbc607f6e686437e43bab6`

The historical size is 458,957 bytes. A match supplies the original required file even if the old copy is gone. Record this as regeneration with historical identity verified. Do not reject it merely because the file was produced today. If the hash does not match, investigate the pinned input, source version and serializer; do not weaken the identity check or fabricate an expected hash.

## 3. Take the cheapest valid probability route

Use the first complete, validated route:

1. Saved `permutation_averaged_semantic_scores.parquet`: validate vectors and proceed directly.
2. Saved `permutation_wise_semantic_scores.parquet`: validate complete cyclic rotations and aggregate on CPU into the new workspace.
3. Complete original V2/V23 score shards and prompt mappings: reconstruct semantic rotations with the source's logic, then aggregate.
4. No recoverable outputs: implement and execute the narrowly scoped scoring task in `docs/FRESH_SCORE_ROUTE.md` using the available experiment machine and pinned checkpoints.

Do not silently replace the primary balanced endpoint experiment with the canonical five-dose analysis. Do not regenerate probability data with random numbers or reconstruct them from summary tables.

## 4. If fresh scoring is needed

Inspect local GPUs, available checkpoint caches, dataset access and installed scoring dependencies. Reuse exact model revision pins and scoring semantics. Save a concrete resource report and run a fixed small real-data timing/memory pilot. Use that measurement to choose batch sizes and schedule the complete scoring workload on available devices. Do not assume eight B200s merely because the old launcher required them; that was a historical orchestration requirement.

Continue the targeted collection when the required checkpoints and compute are available. Support checkpoint/resume and leave all historical data and results unchanged. If an external resource is unavailable, finish the preparation and command setup, then identify the exact unavailable model revision, data mount, machine, access permission or hardware constraint. Do not present another generic “two input files missing” report after this recovery workflow.

Freshly collected model outputs are new measurements under the same declared protocol, not recovered historical bytes. Record the hardware, numerical precision, scoring implementation, dependency versions, model revisions, prompt hashes and date. Do not relabel them as original historical outputs.

## 5. Resume the existing analysis

Once the balanced vectors and selected-question file validate, invoke the already implemented experiment using its actual CLI. Reuse compatible real-data checkpoints only. Constructed-data candidate results must not enter the real-data cache namespace.

Complete every original target/family/split and the fixed comparisons. Keep the original analysis protocol and uncertainty labels. Produce real result tables, one main figure, and a response draft based solely on those results. Existing tests need not be expanded merely to increase their count; check only changed ingestion/recovery/scoring behavior and required numerical invariants.

## Evidence to return

- Search coverage and recovered input paths, with verified identities where available.
- Selected-question restoration outcome and checksum.
- Cache-reuse / CPU-aggregation / fresh-scoring route actually used.
- For fresh scoring: measured pilot costs, completion counts and resume command.
- Real empirical analysis outputs, or a precisely named external resource that prevents the next concrete command.

Component tests and a complete run on constructed inputs show implementation behavior. They do not supply the empirical evidence requested by XWVG.
