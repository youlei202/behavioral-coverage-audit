# Statistics and compute

**Status:** a locked analysis plan, not a report of completed measurements.

The primary design uses **balanced full-option probabilities at the two endpoints**: clean inputs and maximum stress. Each stress family is fitted separately. This choice follows the complete vector cache described in [Code and data map](02_CODE_AND_DATA_MAP.md). A canonical five-dose analysis, if run, is a separately labelled sensitivity analysis; it is not a fallback that can silently replace the primary design.

## 1. Defaults and the claim being evaluated

| Item | Default |
|---|---|
| Point estimates | All 8 targets, both families, all 10 original splits, budgets `1, 2, 3, 7` |
| Primary comparison | Collective selection versus pairwise-nearest selection at `k=2` |
| Supporting comparison | Collective selection versus the exact uniform-random-subset expectation at `k=2` |
| Secondary budget | `k=3`, with the same comparisons |
| Primary metric | Held-out mean total variation (TV) |
| Frozen-fit resampling | 2,000 common subject-stratified question resamples |
| Full selection-and-fitting bootstrap | Disabled |
| P-values or tests treating splits as independent | None |

The question is whether collective coverage helps choose a limited number of reference models that reproduce a target's recorded option probabilities on held-out questions. The budget counts available peer models. It is not a parameter, latency, token, or FLOP budget. A positive result supports this particular selection use; it does not establish better task accuracy, lower serving cost, or equivalent replacement of the target.

Freeze the protocol, configuration, data identities, solver settings, tie rules, and the timing-pilot list before inspecting the new comparisons. The original question pool has already been analysed in the manuscript. This additional held-out analysis is not a new independent benchmark or a retrospectively preregistered study.

## 2. Primary estimand and exact observation weights

Let `t` denote a target, `f` a stress family, `s` one of the ten fixed splits, and `E_s` its 280 evaluation questions. For policy `m`, define its fixed-fit loss on question `q` as

\[
\ell_{m,tfs,k}(q)
=\frac12\operatorname{TV}_{m,tfs,k}(q,\mathrm{clean})
+\frac16\sum_{r=1}^{3}
\operatorname{TV}_{m,tfs,k}(q,\mathrm{maximum\ stress},r).
\]

Here

\[
\operatorname{TV}(p,\widehat p)
=\frac12\sum_{a\in O_q}|p_a-\widehat p_a|,
\]

where `O_q` contains only the valid options. The first `1/2` in the question-loss formula assigns half the question's design mass to clean conditions; the `1/2` inside TV is the definition of TV. Both are required.

Each question has total mass one. Clean receives mass `1/2`; each of the three maximum-stress tracks receives `1/6`. Compute TV separately for each stress track before averaging. Do not average the track predictions before taking their discrepancy. Do not divide TV by the number of options, and do not assign mass to padded coordinates. Use these same weights for fitting, subset selection, evaluation, and top-option agreement.

The policy error for one split is

\[
E_{m,tfs,k}=\frac1{280}\sum_{q\in E_s}\ell_{m,tfs,k}(q).
\]

The primary contrast is

\[
\widehat\Delta_2
=\frac1{8\cdot2\cdot10}
\sum_{t,f,s}
\left(E_{\mathrm{nearest},tfs,2}
-E_{\mathrm{collective},tfs,2}\right).
\]

Positive values favour collective selection. The eight targets, two stress families, and budgets are fixed features of this experiment; they are not samples from populations of models or interventions.

Report the two macro-average errors and their absolute difference first. A relative reduction may accompany them:

\[
100\frac{\overline E_{\mathrm{nearest},2}
-\overline E_{\mathrm{collective},2}}
{\overline E_{\mathrm{nearest},2}}.
\]

Use the ratio of macro-average errors, not the average of target-specific percentages. If the comparator error is zero, report the relative reduction as undefined. Show every target/family result, including zero and negative differences. Apply the same definitions to the random reference and the secondary budget. Top-option agreement is a separate outcome and cannot choose a subset or revise the primary metric.

For the uniform-random reference, average the losses of all candidate subsets, including the per-question losses. Do not average their predictions: that would define a mixture that could use more than `k` peers.

## 3. Preserve the mean of split means

The ten splits overlap. A physical question may appear in evaluation a different number of times from another question. Let

\[
c_q=\#\{s:q\in E_s\},
\qquad
\overline d_{tf,k}(q)
=\frac1{c_q}\sum_{s:q\in E_s}
\left[\ell_{\mathrm{nearest},tfs,k}(q)
-\ell_{\mathrm{collective},tfs,k}(q)\right]
\]

for questions with `c_q > 0`. For one target/family, the mean of split contrasts is

\[
\frac1{10\cdot280}\sum_q c_q\,\overline d_{tf,k}(q).
\]

It generally differs from giving each distinct evaluated question equal weight after averaging its appearances. Keep the split-level denominators and then average the split means, both for the point estimate and for every resample. Save the evaluation-appearance counts as a diagnostic. Never treat the 160 target/family/split settings, the ten split means, the stress tracks, or individual option coordinates as independent replications.

## 4. Frozen-fit common-question resampling

This stage is inexpensive because it uses saved per-question held-out losses. It runs **after** the complete point-estimate calculation and does not fit any new mixtures.

### Reproduce the common multiplicities

Use the source algorithm in `experiments/llm/analysis/v23/src/pier_llm_v23/core.py::common_bootstrap_multiplicities`, with the configured resampling seed `20261107` and replicate IDs `1,...,2000`. This resampling seed is separate from the original split seeds. If porting the helper, preserve its stable seed derivation, category ordering, sorted question IDs, and random-generator calls. Do not replace it with a different per-split bootstrap while retaining the same label.

For each replicate:

1. Within each of the 14 subjects, draw 40 question IDs with replacement from its 40 original IDs.
2. Convert these draws into one multiplicity `b_q` for every original question, including zeros. Each subject's multiplicities sum to 40.
3. Reuse this **same vector** across all targets, both families, all budgets, all policies, and all ten splits.
4. Keep each question on its existing side of each split. A repeated draw changes its weight; it does not create a new identity or move it across the split.
5. Hold every fitted weight vector and every selected subset fixed. The random-subset reference also retains its exact average over the original candidate fits.

For each split, recompute

\[
E^{(b)}_{m,tfs,k}
=\frac{\sum_{q\in E_s}b_q\ell_{m,tfs,k}(q)}
{\sum_{q\in E_s}b_q},
\]

then form the paired contrast and average over the same targets, families, and splits as the point estimator. This is the source protocol's multiplicity-weighted partition normalization. Do not introduce an additional subject-by-subject renormalization within a restricted split without declaring a protocol change. All questions have the same number and design mass of endpoint observations, so weighting the original condition rows by `b_q` gives the same answer.

The two families share clean responses; the common question multiplicities preserve that link. Do not resample their clean observations independently. A zero evaluation denominator is an invalid replicate and must be surfaced, not replaced by a silently altered statistic.

Save the multiplicities or a reproducible compact representation, their checksum, all replicate-level macro contrasts, and the seed/algorithm metadata. The all-ones multiplicity vector must reproduce the point estimate exactly to numerical tolerance.

### Labels and reporting

Report the empirical 2.5th and 97.5th percentiles under the field name:

```text
frozen_fit_question_resampling_interval
includes_selection_and_fitting_uncertainty = false
```

Use this caption, or wording with the same meaning:

> Intervals describe common-question resampling of held-out losses with selected subsets and fitted weights fixed. They do not include uncertainty from relearning the selection rules. The ten overlapping splits are not treated as independent replications.

These are **descriptive frozen-fit resampling intervals**, not confidence intervals for the full selection-and-fitting procedure. Do not use them to report a significance claim, a p-value, or a probability that one method is superior. Split direction counts may be shown as descriptive stability summaries; do not apply a binomial test to them. The manuscript's existing refitting-bootstrap intervals cannot be transferred to this new experiment.

## 5. What a full-refit bootstrap would require

This stage is disabled by default. Enabling it is a separate compute decision based on a recorded, outcome-blind timing pilot.

A full replicate must use one common multiplicity vector as above, restrict it to each original fitting/evaluation partition, recompute all single-peer distances, refit all candidate subsets using weighted fitting data, reselect both policies, and evaluate on the weighted held-out data. Reusing only the original winning subsets omits selection variation. Refitting a winning subset while freezing the nearest-peer ranking is also incomplete.

Use the same paired question counts for all comparisons. Do not independently resample each split, regenerate splits after duplicating questions, or bootstrap condition rows. Any later replicate count must be fixed from the compute budget before inspecting bootstrap conclusions. Do not extend a run until its interval crosses a preferred threshold.

## 6. Workload: no hidden millions of fits

There are `8 * 2 * 10 = 160` target/family/split settings.

| Budget | Candidate subsets per setting | Candidates over all settings | Nontrivial optimizations |
|---|---:|---:|---:|
| `k=1` | 7 | 1,120 | 0; score each peer directly |
| `k=2` | 21 | 3,360 | 3,360 |
| `k=3` | 35 | 5,600 | 5,600 |
| `k=7` | 1 | 160 | 160 |
| **Total** | **64** | **10,240** | **9,120** |

Solve each candidate once. The nearest policy retrieves a candidate already fitted during enumeration. The random expectation uses the same complete candidate collection. Neither needs extra optimization. The full seven-peer fit is a reference, not a guaranteed lower bound on held-out error.

The default 2,000 frozen-fit resamples add **zero LP fits**. In contrast, 1,000 full refitting resamples would require about **9.12 million nontrivial fits**, in addition to the point run and scoring. Do not describe that option as a minor extension.

## 7. Deterministic timing pilot, memory, and threads

### Pilot before the full point run

Use the **first three targets in the locked source-roster order**, both configured stress families, and the first original split seed, `20260828`. Record the actual model IDs. For each of these six settings, use the following zero-based indices into the locked candidate-subset ordering:

| Budget | Pilot candidates per setting |
|---|---|
| `k=1` | All seven direct single-peer scores |
| `k=2` | Subset indices `[0, 10, 20]` among the 21 candidates |
| `k=3` | Subset indices `[0, 17, 34]` among the 35 candidates |
| `k=7` | The sole candidate |

The pilot contains **84 candidate records: 42 direct singleton scores and 42 nontrivial fits**. It samples computational work; it does not perform exhaustive selection or support empirical policy comparisons. Do not choose its targets or subsets because they produce favourable effects or run unusually quickly.

Start the pilot with one worker. Measure load time, fit time by budget, evaluation time, process peak memory, constraint shapes/nonzeros, solver status, and effective thread settings. Pilot candidate outputs can be reused in the complete run when their input and protocol hashes match. Resource decisions use timing, memory, and numerical diagnostics, not held-out effect sizes.

Write the projected remaining workload and the uncertainty in that projection. Checkpoint completed fits. The full point run has **9,120** nontrivial fits in total, including reusable pilot fits; after all 42 pilot fits succeed, 9,078 remain. There is no default hard wall-time abort for the full point run. If an actual job resource limit prevents completion, report an incomplete run with a continuation path; do not shrink the model roster, omit failed subsets, or change the primary budget to finish a different experiment.

### Sparse solver requirement

The primary endpoint design has `280 * 4 = 1,120` fitting observations per family and at most `11,200` valid option coordinates. A dense absolute-error LP inequality matrix can approach **2 GB** before temporary arrays and solver copies. Use the supplied sparse reference implementation as the numerical starting point; do not call the historical dense scalar `fit_mae_simplex` on flattened vectors.

For the optional canonical five-dose sensitivity, the corresponding maximum is `280 * 13 * 10 = 36,400` coordinates, and a dense matrix can exceed **21 GB**. Sparse construction is required there as well. These are upper-bound coordinate counts, not a reason to pad invalid coordinates into the LP.

### Parallelism rules

- Allow at most four independent solver workers for the full run, reduced as necessary to fit the recorded CPU and memory allocation. Start the timing pilot with one worker.
- Keep the projected aggregate worker peak memory below half of the memory available to the job. Account for loaded response tensors and solver copies. If even one worker does not fit, checkpoint and report the resource limitation.
- Cap BLAS/OpenMP threads at one per worker, using the appropriate `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` settings before importing numerical libraries. Also configure and record the effective HiGHS thread limit through settings supported by the installed version; do not assume BLAS controls govern the solver.
- Never multiply unrestricted solver threads by unrestricted process workers. Record the actual CPU allocation and keep total active threads within it.
- Group work by target/family/split so a worker can reuse aligned tensors across subsets. Save candidates under deterministic keys containing input/protocol hashes, target, family, split seed, and subset identities.
- Resume only validated completed candidates. A failed candidate leaves its exhaustive-selection setting incomplete; do not silently exclude it from selection or the random expectation.

No GPU or fresh model inference is required when the validated probability cache is available. If it is missing, follow the missing-data procedure in the data map rather than estimating new subset results from summary tables.

## 8. Optional future acceleration for two-peer fits

This optimization is **not required** for the default sparse-LP point run. It may be useful if full-refit inference is later requested.

For two peers, write the mixture as `alpha * p_a + (1-alpha) * p_b`, with `0 <= alpha <= 1`. For each valid flattened coordinate, set

\[
\delta_i=p_{a,i}-p_{b,i},\qquad
v_i=\frac{y_i-p_{b,i}}{\delta_i}.
\]

Coordinates with `delta_i=0` add a constant. The remaining weighted TV objective has the form

\[
\sum_i c_i|\delta_i|\,|\alpha-v_i|+\text{constant},
\]

where `c_i` contains the declared TV and observation weights. A weighted median of `v_i`, clipped to `[0,1]`, is an exact minimizer. Bootstrap multiplicities change the weights but not the breakpoint order, which can be cached.

Before adopting this optimization, verify objective equivalence to the sparse LP on deterministic controls and representative real candidate fits, including identical peers, boundary solutions, ragged options, and zero bootstrap masses. Lock its tie handling and numerical policy. It is a faster implementation of the same fitting objective, not a new scientific method or permission to change the experiment.
