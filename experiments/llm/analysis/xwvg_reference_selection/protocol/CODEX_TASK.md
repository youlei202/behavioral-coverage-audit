# Execute the XWVG utility experiment

## User objective

Implement and run one focused supplementary experiment for **Auditing Behavioral Coverage in Model Ecosystems**, ICLR submission 41357. Reviewer XWVG asks: **what can be done better with the coverage information, and what evidence supports that use?**

The experiment asks whether collective response coverage improves the selection of a limited number of reference models for reproducing one target's recorded response distributions on held-out questions.

Work autonomously on the reversible implementation and cached-data analyses. Follow the existing repository instructions. Complete the authorized implementation, checks, and available analyses. Ask for missing input only after identifying the exact missing files and completing the useful unblocked work.

## Read in this order

1. `config/experiment.json`.
2. `docs/01_EXPERIMENT_PROTOCOL.md`.
3. `docs/02_CODE_AND_DATA_MAP.md`.
4. `docs/03_STATISTICS_AND_COMPUTE.md`.
5. `docs/04_IMPLEMENTATION_AND_DELIVERABLES.md`.
6. Relevant actual source files in the user's repository. The map refers to inspected files; do not assume its paths still match a newer local checkout without checking.

## Fixed scientific choices

- Primary response: **complete cyclic-permutation-averaged semantic option probabilities** from the V2.3 balanced endpoint cache.
- Per-family design: one clean observation and three maximum-stress realizations per question, with masses 1/2 and 1/6 each.
- Every original target, both original stress families, and all ten original question splits.
- `k=2` primary; `k=3` secondary; `k=1,7` controls/reference.
- Collective subset search, nearest-k selection, and the exact expected performance of a uniform-random k-subset. All use the same fixed-convex TV fitting routine.
- Primary metric: held-out TV. Secondary metric: top-option agreement under the scored probability interface.
- Default resampling: 2,000 common-question **frozen-fit descriptive** replicates. Full selection/refitting bootstrap is disabled.
- No new model inference in the default task.

These choices were made from the paper, review, and source inspection, before running this supplementary experiment. This is a post-review supplementary analysis of an already studied question pool, not a preregistered independent validation.

## Scope and existing evidence

The public repository snapshot provides source and frozen summaries. It does not contain the primary per-question probability cache. The original experiment workspace may have that cache.

The key primary file is:

`PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION/outputs/analysis/permutation_averaged_semantic_scores.parquet`

Use its **`permutation_averaged_semantic_probabilities`** column. A scalar `response` column or the archived 1,600-row `trackwise_vector_results.parquet` is not a substitute.

If the averaged cache is absent but the complete saved semantic rotation outputs exist, implement a **pure CPU aggregation** in the new workspace. Verify one full cyclic set for every model/question/semantic condition and average the semantically remapped vectors. Preserve the original outputs. The historical `assemble_permutation_scores()` writes into its supplied run root and calls other validators; do not invoke it blindly against the old run. Use the inspected grouping logic and explicit input/output paths.

If only canonical V2 scores are available, report that the balanced primary is blocked. Canonical five-dose analysis is an optional separately labelled protocol; it does not silently become the primary result.

## Implementation destination

Create a new analysis component, proposed as:

`experiments/llm/analysis/xwvg_reference_selection/src/bca_reference_selection/`

Keep original raw data, historical analysis modules, `data/frozen`, published outputs, and old completion markers unchanged. Store new run outputs in a fresh user-owned directory such as `outputs/xwvg_reference_selection/<run_id>/`. Existing uncommitted work must remain intact.

Use explicit paths passed to the new loader. Historical modules contain machine-specific root constants, old bootstrap budgets, dense solver allocations, and sometimes unrelated inference validation. Reuse their relevant mathematical and alignment logic, not their orchestration by default.

## Required sequence

### A. Inventory and lock inputs

Locate the actual repository and original cached experiment roots from the current workspace, the user's configured `BEHAVIORAL_COVERAGE_RAW_DATA`, and the documented historical roots where those paths exist. Do not invent paths or claim that a source inspection establishes a cache's existence.

Run the bundled preflight. Its output is a first availability and byte-identity check. Add the row-level checks in the protocol before fitting. Save code revision/hash, input hashes, effective configuration, package versions, and exact question/split IDs. Compare existing provenance where available. A newly calculated SHA-256 identifies the file read now; it is not by itself proof that the file equals the historical cache.

### B. Build and check the implementation

Implement the portable full-vector loader, deterministic original splits, sparse TV fitter, subset enumeration, training-only selection, held-out evaluation, common frozen-fit resampling, result tables, plots, and final reporting.

The reference solver is a starting point and has constructed-data checks. Integrate it or implement an equivalent solver with the same objective. **Never use the historical dense `np.eye` MAE LP on flattened full vectors.** Verify vector masks and TV's option SUM carefully.

### C. Run a fixed timing and memory pilot

Use the pilot cases fixed in `config/experiment.json`. This pilot checks feasibility, not which targets to report. Keep its already valid fits for the full run. Save runtime and memory estimates and choose a safe worker count from resources. Use one BLAS thread per worker. Resume from input/config/code-matched checkpoints.

### D. Run the complete point analysis

Finish all 160 target/family/split settings and 64 candidates per setting. Compute singleton losses directly. Reuse each candidate fit across strategies. Held-out target values must never enter fitting, selection, tie-breaking, or tuning. Read-only input validation and the fixed pilot's timing measurements may inspect evaluation data. Save selected subsets and weights before generating the held-out policy comparisons. Random expected performance is the mean of the fitted subset losses, not the loss of an averaged prediction that uses more peers.

### E. Summarize without changing the experiment

Produce all-target tables, macro-average results, top-option agreement, and frozen-fit descriptive intervals. Distinguish technical completion from whether the scientific comparison favors collective selection. Do not drop unfavorable targets or change the objective, doses, subsets, or headline budget after seeing results.

### F. Return reviewable results

Provide a short plain-language report, exact numerical tables, one compact figure, reproducible commands, and a result-grounded draft response to XWVG. Only write a completed-results paragraph after the real-data run is complete. If inputs remain missing, return the implementation, checks, and a blocked-data report instead of a fabricated response.

## Completion criteria

- All required input, split, objective, solver, and enumeration checks pass.
- All candidate and strategy results are present, or the run is explicitly incomplete with exact failed settings.
- Metrics can be recomputed from saved held-out losses and saved selected weights.
- Frozen-fit intervals are labelled as excluding selection/fitting uncertainty. No p-values or significance claims derived from these descriptive intervals.
- Original source/data identities remain unchanged.
- The final statement answers the practical question only to the extent supported by the actual held-out results.

Do not stop after writing a plan if the original caches are available. This package is the plan; your job is to implement and run it.
