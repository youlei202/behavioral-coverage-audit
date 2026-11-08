# Experiment protocol: selecting a fixed number of reference models

**Status:** a plan for an additional experiment. This document contains no new experimental results.

## 1. Question and intended use

An auditor wants a small set of reference models whose combined responses approximate a specified target model on future inputs from the audited design. The decision is which peers to include when at most two or three are available. Choosing the individually closest peers can select redundant responses; selecting peers jointly can account for how their responses combine.

The experiment asks:

> Does collective response coverage improve the selection of a fixed number of reference models for reproducing a target's option-probability distribution on held-out questions?

Each target receives its own reference set. This is a practical validation of the paper's coverage measurements, using a simple exhaustive selection rule. It is not a new claim about the novelty of subset optimization. It does not select one global model collection, prune a deployed ecosystem, or optimize inference cost. Those would require different objectives and evidence.

The available-model budget counts **at most \(k\) available peers**, not parameters, latency, tokens, or FLOPs. Some selected peers may receive zero fitted weight. The original target remains available for fitting and evaluation; the experiment does not authorize removing it from a system.

## 2. Locked scope and order of comparisons

Freeze this protocol, the configuration, input identities, solver tolerances, and tie rules before inspecting the new comparison results.

| Item | Required choice |
|---|---|
| Targets | All eight original language models |
| Candidate peers | The other seven models, separately for each target |
| Perturbation families | `irrelevant_context` and `content_deletion`, fitted separately |
| Questions | The original 560-question MMLU-Pro sample, 40 questions in each of 14 subjects |
| Outer partitions | The original ten stratified fitting/evaluation half-splits; preserve their question identities |
| Response | Complete semantic-option probability vectors averaged over all cyclic option rotations |
| Conditions | Clean and maximum stress in each family; one clean observation and three matched maximum-stress instances |
| Budgets | \(k=1,2,3,7\) |
| Primary comparison | Collective selection versus pairwise-nearest selection at \(k=2\) |
| Supporting baseline | Uniform-random selection at \(k=2\) |
| Secondary comparison | The same comparisons at \(k=3\) |
| Structural checks | \(k=1\) and \(k=7\) |
| Fit, selection, primary evaluation loss | Total variation (TV), with identical observation weights |
| Additional evaluation | Agreement with the target's highest-probability option |

The eight targets are Qwen2.5, General Reasoner, phi-4, Phi-4-reasoning-plus, Mistral NeMo, OLMo 2, Granite 3.3, and Gemma 3. Use their exact checkpoint identifiers and revisions from the source manifest `data/frozen/llm/v2/table1_ecosystem_manifest.csv`; short names are display labels only.

Maximum stress is 512 added words for irrelevant context and a 0.4 deleted-word fraction for content deletion. Give clean half of each question's mass. Divide the other half equally among the three maximum-stress instances, giving each instance mass \(1/6\). The expected original split sizes are 280 fitting and 280 evaluation questions; validate against the archived split definitions. Each model/family therefore supplies 2,240 vector observations, with 1,120 fitting and 1,120 evaluation observations in each half-split.

This uses the balanced endpoint observation design underlying the main Figure 2, while retaining **all option coordinates** and minimizing TV directly. The V2.3 source already preserves these complete vectors. Its `assemble_permutation_scores` function writes `outputs/analysis/permutation_averaged_semantic_scores.parquet`, including the column `permutation_averaged_semantic_probabilities`. The complete cache has \(8\times560\times7=31{,}360\) rows: one clean condition and six stressed conditions per model/question. Clean is shared between the two family-specific analyses, not duplicated within either analysis.

The canonical five-dose complete-option design from Appendix C.3 may be used only as a **separately labeled optional sensitivity analysis**. It is not the primary design and is not an automatic fallback. Its original squared-vector fits are not numerical expectations for the new matched-TV analysis.

**No new inference is part of the default run.** Use the validated balanced cache. If that cache is absent, reconstruct the same complete balanced vectors from saved, complete rotation-level probability records using CPU-only processing. If neither source is complete, exit as blocked and report what is missing. Do not silently switch to canonical responses or add checkpoints, new questions, new rotation scoring, longer generation, or other perturbations.

## 3. Data required before fitting

For each recorded observation, the adapter must recover the base question ID, subject, model identity, semantic condition, perturbation family, dose, instance or track identity, valid option identities, and the complete balanced probability vector. It must also recover the original split memberships. Read the full `permutation_averaged_semantic_probabilities` column, not the scalar `permutation_averaged_gold_probability` column.

Require the following checks before computing comparative results:

1. Align every model on the same semantic question, intervention instance, and valid option coordinates. Array position alone is insufficient unless its identity mapping is verified.
2. Keep all responses for a base question on the same side of every split. This includes all models, doses, and instances.
3. Mask padded coordinates. They must not become answer options or contribute probability mass.
4. Validate finite values, probability bounds, normalization, and completeness. Record any permitted roundoff correction under the numerical policy. Do not silently repair substantive violations.
5. Resolve repeated clean records according to the archived observation identity. A repeated clean record must not acquire extra mass. Conflicting copies must be investigated rather than averaged without justification.
6. Require a common complete set of valid observations for all methods. Do not silently drop a different set of questions or coordinates for different targets or policies.
7. Record file checksums, model revisions, question ordering, and split identities so the run can be reproduced.
8. When reconstructing a missing balanced cache, require the complete cyclic set for every model/question/semantic-condition group: rotation IDs \(0,\ldots,|O_q|-1\), each present once. Remap visible labels to the same semantic options before averaging. Missing rotations cannot be replaced with a canonical vector or an average over the available subset.

Summary tables, existing TV values, scalar reference-answer probabilities, fitted weights, and plots cannot reconstruct the required probability vectors. In particular, the archived canonical vector summary with repeated overall values across dose rows is not an input for the balanced experiment. If the balanced vectors cannot be validated or reconstructed from complete saved rotations, produce a precise missing-data report and leave the main experiment incomplete. Synthetic data can validate implementation only; it cannot supply empirical results.

## 4. Shared fitting and evaluation objective

Let \(t\) be the target, \(J_{-t}\) its seven peers, \(f\) a perturbation family, and \(A\) a set of base questions. Let \(O_q\) be the valid semantic options for question \(q\), and let \(d_f^{\max}\) be the maximum stress dose. Write

\[
\overline p_{j,q,d,r,a}
=\frac{1}{|O_q|}\sum_{\rho=0}^{|O_q|-1}
p^{(\rho),\mathrm{semantic}}_{j,q,d,r,a}
\]

for model \(j\)'s rotation-averaged probability assigned to valid semantic option \(a\) in intervention instance \(r\). Each term has already been remapped from its visible answer labels to semantic option identities. This average defines the response being audited; do not substitute an average of per-rotation losses.

The four endpoint observations and their weights are

\[
\Omega_f=\{(0,1),(d_f^{\max},1),(d_f^{\max},2),(d_f^{\max},3)\},
\qquad
\lambda_{0,1}=\tfrac12,
\quad
\lambda_{d_f^{\max},r}=\tfrac16.
\]

The clean-instance index is a mathematical placeholder; preserve the source's actual clean track identifier in stored records. For a candidate peer subset \(S\), use fixed weights satisfying

\[
w_j\geq 0,\qquad \sum_{j\in S}w_j=1.
\]

The reconstructed distribution is

\[
\widehat p_{S,w,q,d,r,a}
=\sum_{j\in S}w_j\overline p_{j,q,d,r,a}.
\]

One weight vector applies to all questions and all four endpoint observations within that target/family/split/subset fit. The objective is

\[
L_{t,f,A}(S,w)
=\frac{1}{|A|}\sum_{q\in A}
\sum_{(d,r)\in\Omega_f}\lambda_{d,r}
\frac{1}{2}\sum_{a\in O_q}
\left|\overline p_{t,q,d,r,a}-\widehat p_{S,w,q,d,r,a}\right|.
\]

Every question has equal mass. Clean and maximum stress each receive half of that mass; the three stress instances divide the stressed half equally. Compute TV separately for each instance before their weighted average. Do not average the three stressed vectors first. Sum the absolute coordinate differences within an observation; **do not divide TV by the number of options**. When vector coordinates are flattened for a solver, preserve these observation weights rather than making all flattened rows contribute an equal fraction of the objective.

For each outer split, fit

\[
\widehat w_S\in\arg\min_{w\in\Delta(S)}
L_{t,f,A_{\mathrm{fit}}}(S,w)
\]

and evaluate that same fitted response on \(A_{\mathrm{eval}}\). Evaluation never refits the weights or selects a subset.

All methods use the same implementation, data, loss, and feasibility criteria. TV fitting can have multiple minimizers. Use the locked solver, input ordering, and tie policy consistently, and preserve the returned feasible weights. Do not interpret a selected coefficient as a unique attribution or invoke the paper's squared-loss uniqueness result for this TV fit.

### Solver requirement

Implement the absolute-error program with **sparse** constraint matrices and `scipy.optimize.linprog`. The existing V2.3 `fit_mae_simplex` in `experiments/llm/analysis/v23/src/pier_llm_v23/core.py` constructs dense residual identity matrices with `np.eye`; it must not be reused unchanged for flattened complete-option observations. Sparse construction is an implementation requirement, not a change to the objective. Follow the package's solver and numerical-check instructions.

## 5. Three selection policies

For every \(k\), enumerate

\[
\mathcal S_k=\{S\subseteq J_{-t}:|S|=k\}.
\]

Solve each candidate subset once and cache its fitted weights and predictions. Zero weights are allowed, so enumeration of size-\(k\) sets implements an at-most-\(k\) budget. Reuse the same cached candidate fit whenever a policy selects it.

### 5.1 Collective selection

Select the subset with the smallest fitting-set TV:

\[
\widehat S_{\mathrm{collective},k}
\in\arg\min_{S\in\mathcal S_k}
L_{t,f,A_{\mathrm{fit}}}(S,\widehat w_S).
\]

Enumerate all subsets; this is not greedy selection. Resolve objective ties using the locked numerical tolerance and a stable order derived from model identities. Neither held-out TV nor held-out option agreement can break a tie.

### 5.2 Pairwise-nearest selection

For every peer \(j\), compute its fitting-set distance from the target using the same TV definition and observation weights. Select the \(k\) smallest distances, with the same stable model order for ties. Then use the TV-optimal convex fit for that selected subset.

Do not force equal weights on these peers or compare against only their best single member. Both deterministic policies receive exactly the same fitting opportunity after selection. Their difference is which peers they make available.

### 5.3 Uniform-random selection

Define a policy that selects one subset uniformly from \(\mathcal S_k\), then uses that subset's fitted convex weights. Since every subset is enumerated, report its **exact expected evaluation metric**:

\[
\overline L_{\mathrm{random},k}
=\frac{1}{|\mathcal S_k|}
\sum_{S\in\mathcal S_k}
L_{t,f,A_{\mathrm{eval}}}(S,\widehat w_S).
\]

Average each subset's metric, not its predictions. Averaging predictions across different subsets could use all seven peers and would define a different policy. Apply the same expectation rule to option agreement and to per-question metrics. There is no Monte Carlo seed or simulation error for this baseline. Retain the distribution across subsets for descriptive inspection.

## 6. Metrics and contrasts

For each target, family, split, and budget, the primary measured quantity is held-out mean TV. Smaller values indicate a better approximation of the measured response distribution.

Define the paired differences

\[
\Delta_{\mathrm{nearest},k}
=L_{\mathrm{nearest},k}-L_{\mathrm{collective},k},
\qquad
\Delta_{\mathrm{random},k}
=\overline L_{\mathrm{random},k}-L_{\mathrm{collective},k}.
\]

Positive values favor collective selection. Report absolute TV differences first. Relative reductions may accompany them when the comparator's error is positive; never replace an absolute effect with a percentage alone.

The primary comparison averages \(\Delta_{\mathrm{nearest},2}\) equally across all eight targets, both families, and the ten fixed splits. Show the full target/family breakdown so an average cannot hide opposing effects. The random comparison at \(k=2\) and the \(k=3\) comparisons are supporting and secondary results. Do not switch the primary budget after seeing results. The seven-peer fit is a full-roster reference, not a guaranteed lower bound on held-out error.

For each observation, also compare the maximum-probability semantic option of the target's balanced vector with that of the mixture's balanced vector. Aggregate agreement with the same question/endpoint/instance weights as TV. Resolve exact option ties by a fixed valid-option identity order. Do not average per-rotation argmax agreements. This metric must not influence selection or fitting.

Call this **top-option agreement under the balanced candidate-label interface**. It is neither free-form generation agreement nor verified generated-answer accuracy. Do not use the archived permissive generation parser for it. A TV improvement with lower top-option agreement is a result to report, not a reason to change the objective.

## 7. Required records and presentation

Preserve sufficient per-question held-out losses and agreement indicators for the statistical procedure in [Statistics and compute](03_STATISTICS_AND_COMPUTE.md). Preserve all candidate fits, not only selected ones.

The completed experiment should provide:

- A candidate-fit record for each target, family, split, budget, and subset: peer identities, fitted weights, fitting TV, evaluation TV, agreement, solver status, and feasibility diagnostics.
- A policy-level table containing all three methods for every target/family/split/budget setting. Mark uniform-random rows as exact expectations rather than assigning them a fictional selected subset.
- A main table with all 16 target/family combinations at \(k=2\): method errors, paired differences, and the uncertainty summaries specified separately.
- An equivalent secondary table for \(k=3\).
- A compact budget figure for \(k=1,2,3,7\), averaged equally across targets and families, plus the full underlying tables. Use the original model labels consistently.
- A result note that distinguishes completed measurements, limitations of the fixed benchmark, and any incomplete settings. Generate numerical statements from saved results rather than typing estimates from plots.

There are \(7,21,35,1\) candidate subsets at \(k=1,2,3,7\), respectively. The complete point-estimate run contains 160 target/family/split settings and 10,240 candidate fits. Singleton fits require no optimization, leaving 9,120 nontrivial fits. These counts exclude any resampling required by the separate statistical plan.

## 8. Validation of the implementation

Before accepting a complete run, require these structural checks within the locked numerical tolerances:

1. At \(k=1\), collective and pairwise-nearest selection agree, including tie handling. Uniform-random generally differs.
2. At \(k=7\), all three policies use the same candidate fit and give identical metrics.
3. On fitting data, collective selection is no worse than nearest selection or the random expectation at the same budget.
4. The best collective fitting loss is non-increasing over the available budgets. Do not impose this check on evaluation error.
5. Reuse of a candidate fit produces identical weights, predictions, and metrics across policies.
6. Candidate probabilities and mixture predictions pass the data and numerical checks. Target responses never appear among their own peer columns.
7. No evaluation outcome influences subset selection, fitting, budget selection, or tie handling.
8. Every required candidate succeeds or the affected setting is explicitly incomplete. Do not silently omit a failed candidate from exhaustive selection or the random expectation.

Passing these checks establishes that the experiment was implemented as specified. It does not require a favorable empirical result.

## 9. Statistical scope and interpretation

Follow [Statistics and compute](03_STATISTICS_AND_COMPUTE.md) for resampling, uncertainty, compute limits, and any permitted stages of execution. Do not treat overlapping splits as independent replications or improvise a cheaper interval with a different interpretation.

The existing 560-question pool has already supported exploratory analyses in the paper. Freezing this new protocol before computing its comparisons prevents further outcome-driven changes; it does not make the reused questions a new independent benchmark or make the study retrospectively preregistered. The held-out design evaluates a fitting-and-selection procedure within this fixed benchmark design.

Report the observed magnitude and the statistical evidence specified in the companion document:

| Possible result | Appropriate interpretation |
|---|---|
| Collective selection improves held-out TV at the primary budget | Evidence that considering combinations can improve this specific reference-model selection decision |
| Improvements are concentrated in particular targets or conditions | A target-specific benefit; show the complete pattern and limit the claim accordingly |
| Differences are negligible, uncertain, or unfavorable | The experiment does not establish additional selection value in the current roster and question pool |
| TV and top-option agreement move in different directions | Distributional approximation and reproducing the preferred option can favor different choices |
| Data or numerical requirements fail | An incomplete experiment; report the concrete missing input or failed condition |

Do not remove unfavorable targets, change the primary budget, change the response presentation, or introduce a new primary metric to obtain a positive conclusion. Even a substantial relative TV reduction does not establish identity of model behavior. The source manuscript's canonical complete-option errors are substantial, but the new balanced-vector errors have not been measured. Neither the old results nor an improvement in the new experiment establishes an equivalent replacement without separate evidence.

The strongest supported application claim, if the results warrant it, is that **the coverage audit helps choose a limited number of reference models whose fixed combination better reproduces a target's balanced option probabilities on held-out inputs**. Claims about capability preservation, safe replacement, inference savings, open-ended generation, and global ecosystem pruning remain outside this experiment.
