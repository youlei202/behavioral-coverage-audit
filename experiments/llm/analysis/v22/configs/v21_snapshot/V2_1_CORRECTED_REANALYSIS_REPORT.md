# PIER Modern-LLM Ecosystem V2.1 — Corrected CPU Reanalysis



This report reanalyzes the completed V2 scores without new inference. The primary estimand is controlled, design-dependent peer expressibility under a candidate-label response interface. The additive label-bias correction is exploratory and is not described as resolving the interface issue.



All endpoint intervals use 1000-replicate stratified base-question cluster bootstraps with refitting and synchronized clean/high samples. `Stable` means at least 9 of 10 paired split effects share a sign and the 95% interval excludes zero.

## 1. Corrected paired clean-to-512 effect for microsoft/phi-4

**Observed result.** Mean paired ΔPIER is 0.029838 across 10 splits, replacing the invalid unpaired `0.0867769` value.

**Interpretation.** The sign and magnitude now compare the two pre-declared endpoints within the same split and fitted intervention family.

**Measurement limitation.** The response remains gold-label probability under the candidate-label interface.

**Implication for Stage 2.** Retain phi-4 clean and 512-word conditions in the focused validation.

## 2. 95% clustered-bootstrap interval for that phi-4 effect

**Observed result.** The 1000-replicate interval is [0.0218622, 0.0360921]; P(Δ>0)=1.000.

**Interpretation.** The empirical evidence status is `stable` under the 9/10-sign plus interval rule.

**Measurement limitation.** The bootstrap refits within each fixed split and summarizes across those ten designs; it does not add benchmark diversity.

**Implication for Stage 2.** Use this interval, rather than an extreme-row contrast, to prioritize phi-4.

## 3. Models moving away from the peer hull under irrelevant context

**Observed result.** `microsoft/phi-4` 0.029838 (stable); `Qwen/Qwen2.5-7B-Instruct` 0.0181351 (stable); `microsoft/Phi-4-reasoning-plus` 0.0125407 (stable); `TIGER-Lab/General-Reasoner-Qwen2.5-7B` 0.00530363 (exploratory)

**Interpretation.** Positive paired endpoint deltas mean larger held-out residuals at 512 words.

**Measurement limitation.** Positive movement is design-dependent peer expressibility, not a general behavioral property.

**Implication for Stage 2.** Prioritize stable positive cases and interface-sensitive positive cases separately.

## 4. Models moving toward the peer hull under irrelevant context

**Observed result.** `google/gemma-3-12b-it` -1.04054e-07 (exploratory); `allenai/OLMo-2-1124-13B-Instruct` -0.0208305 (stable); `mistralai/Mistral-Nemo-Instruct-2407` -0.0219497 (stable); `ibm-granite/granite-3.3-8b-instruct` -0.0253374 (stable)

**Interpretation.** Negative paired endpoint deltas mean the target response is better covered by the fixed peer convex hull at 512 words.

**Measurement limitation.** The hull and response interface are fixed to this eight-model design.

**Implication for Stage 2.** Validate the strongest stable negative case at the same endpoint and tracks.

## 5. Content-deletion collapse across all three tracks

**Observed result.** No—not universally. All three mean track-specific endpoint deltas are negative for 3/8 targets: `allenai/OLMo-2-1124-13B-Instruct`, `ibm-granite/granite-3.3-8b-instruct`, `mistralai/Mistral-Nemo-Instruct-2407`. They are not all negative for: `Qwen/Qwen2.5-7B-Instruct`, `TIGER-Lab/General-Reasoner-Qwen2.5-7B`, `google/gemma-3-12b-it`, `microsoft/Phi-4-reasoning-plus`, `microsoft/phi-4`. The minimum split-level same-sign track count is 0/3.

**Interpretation.** The primary aggregate deletion decrease is therefore partly track-dependent and, for some targets, amplified by cancellation after averaging responses.

**Measurement limitation.** Only the three already fixed tracks are represented.

**Implication for Stage 2.** Carry all three tracks into Stage 2 rather than selecting the strongest one.

## 6. Track-cancellation gap

**Observed result.** The maximum split-level cancellation gap is 0.0470968 for `google/gemma-3-12b-it`/content_deletion; the mean across target-family summaries is 0.0137258.

**Interpretation.** A positive gap quantifies variation hidden by averaging tracks before applying the absolute residual.

**Measurement limitation.** This is a Jensen gap, not an independent uncertainty interval.

**Implication for Stage 2.** Use per-track reporting for cases with the largest gap.

## 7. Trackwise-loss fitting sensitivity

**Observed result.** Mean endpoint-sign agreement with the primary fit is 99.4%; mean projected-response difference is 0.00446905.

**Interpretation.** The primary conclusions are preserved to the extent indicated by those two diagnostics.

**Measurement limitation.** Trackwise fitting is a sensitivity analysis and does not replace the preregistered mean-response fit.

**Implication for Stage 2.** Escalate cases with sign disagreement or unusually large weight changes.

## 8. Corrected Qwen sibling-removal inflation

**Observed result.** content_deletion: 0.066172 (95% [0.0600071, 0.0711895]); irrelevant_context: 0.0609911 (95% [0.0539734, 0.0665685])

**Interpretation.** These are paired all-peer versus sibling-removed effects with refitting.

**Measurement limitation.** The effect measures convex coverage supplied by one designated sibling, not a causal lineage effect.

**Implication for Stage 2.** Include both Qwen siblings in any focused ecosystem validation.

## 9. Does Qwen sibling removal exceed every same-size non-sibling removal?

**Observed result.** content_deletion: 10/10 splits; irrelevant_context: 10/10 splits

**Interpretation.** The 10/10 result supports disproportionate sibling coverage relative to the exact enumerated non-sibling null in both families.

**Measurement limitation.** The null is small because only seven peers exist, although it now excludes the sibling by construction.

**Implication for Stage 2.** Retain the full peer roster when validating sibling coverage.

## 10. Sibling ranks in the single-peer removal influence map

**Observed result.** Qwen/Qwen2.5-7B-Instruct/content_deletion: mean rank 1, first in 10/10; Qwen/Qwen2.5-7B-Instruct/irrelevant_context: mean rank 1, first in 10/10; TIGER-Lab/General-Reasoner-Qwen2.5-7B/content_deletion: mean rank 1, first in 10/10; TIGER-Lab/General-Reasoner-Qwen2.5-7B/irrelevant_context: mean rank 1, first in 10/10; microsoft/Phi-4-reasoning-plus/content_deletion: mean rank 1.5, first in 5/10; microsoft/Phi-4-reasoning-plus/irrelevant_context: mean rank 3.3, first in 0/10; microsoft/phi-4/content_deletion: mean rank 1.3, first in 7/10; microsoft/phi-4/irrelevant_context: mean rank 6.7, first in 0/10

**Interpretation.** Rank 1 means the largest held-out inflation after individual peer removal within that split.

**Measurement limitation.** Rank is sensitive to the fixed roster and can tie.

**Implication for Stage 2.** Use the rank map to identify non-sibling comparators for focused validation.

## 11. Corrected convex-versus-single improvements for Phi-4-reasoning-plus and Mistral-Nemo

**Observed result.** microsoft/Phi-4-reasoning-plus/content_deletion: 0.0356547 (95% [0.0300318, 0.0425024], stable); microsoft/Phi-4-reasoning-plus/irrelevant_context: 0.0348953 (95% [0.0292383, 0.0409763], stable); mistralai/Mistral-Nemo-Instruct-2407/content_deletion: 0.0312352 (95% [0.0262536, 0.0429792], stable); mistralai/Mistral-Nemo-Instruct-2407/irrelevant_context: 0.0308598 (95% [0.025585, 0.0407138], stable)

**Interpretation.** Positive values mean the fixed convex surrogate outperformed a peer selected only on fitting questions.

**Measurement limitation.** The comparison uses overall held-out dose designs, not a best-case split.

**Implication for Stage 2.** Validate cases whose advantage is stable and scientifically material.

## 12. Do those convexity intervals exclude zero?

**Observed result.** microsoft/Phi-4-reasoning-plus/content_deletion: excludes zero; microsoft/Phi-4-reasoning-plus/irrelevant_context: excludes zero; mistralai/Mistral-Nemo-Instruct-2407/content_deletion: excludes zero; mistralai/Mistral-Nemo-Instruct-2407/irrelevant_context: excludes zero

**Interpretation.** Exclusion of zero is combined with the 9/10 split-sign rule for the `stable` label.

**Measurement limitation.** Percentile intervals quantify base-question sampling uncertainty within this benchmark.

**Implication for Stage 2.** Treat intervals including zero as exploratory rather than discarding them.

## 13. Most frequently selected honest single peer

**Observed result.** `mistralai/Mistral-Nemo-Instruct-2407` was selected in 63 of 160 target-family-split comparisons.

**Interpretation.** This identifies the dominant fit-only baseline comparator.

**Measurement limitation.** Selection frequency depends on the target roster and five-dose fitting design.

**Implication for Stage 2.** Include this peer when constructing the smallest comparison set.

## 14. Calibrated versus raw dose trajectories

**Observed result.** Mean raw/calibrated rank correlation across the family summaries is 0.616667; mean absolute endpoint change is 0.0119158.

**Interpretation.** Temperature calibration can shift both residual magnitudes and rankings while using one clean-fit temperature across all doses.

**Measurement limitation.** Temperature rescales candidate scores but does not remove label-position effects.

**Implication for Stage 2.** Keep raw and calibrated endpoints as paired reporting conditions.

## 15. Effects reversing after calibration

**Observed result.** TIGER-Lab/General-Reasoner-Qwen2.5-7B/irrelevant_context: raw 0.00530363 → calibrated -0.00667692; microsoft/Phi-4-reasoning-plus/content_deletion: raw 0.000283453 → calibrated -0.0114565; microsoft/Phi-4-reasoning-plus/irrelevant_context: raw 0.0125407 → calibrated -0.0141035

**Interpretation.** A sign reversal indicates sensitivity to score sharpness rather than a robust endpoint direction.

**Measurement limitation.** Calibration itself is estimated from clean fitting questions and is not ground truth.

**Implication for Stage 2.** Prioritize reversed cases for direct-generation validation.

## 16. Held-out permutation TV after additive label-bias correction

**Observed result.** Mean TV changed from 0.419826 to 0.39805; 96.2% of model-split fits improved held-out invariance.

**Interpretation.** The correction reduces the interface artifact when held-out TV decreases.

**Measurement limitation.** The correction is exploratory and bias estimated on clean permutations may not transport to stressed prompts.

**Implication for Stage 2.** Validate transport using fixed endpoint prompts and balanced labels.

## 17. Residual artifact after correction

**Observed result.** The largest residual mean TV is 0.462716 for `ibm-granite/granite-3.3-8b-instruct`; its residual mean 95th-percentile TV is 0.982537.

**Interpretation.** Nonzero residual TV shows that the additive correction does not fully account for presentation sensitivity.

**Measurement limitation.** Semantic utilities and label effects may interact non-additively.

**Implication for Stage 2.** Retain clean label permutations for the worst residual case.

## 18. Headline effects surviving label-bias correction

**Observed result.** Qwen/Qwen2.5-7B-Instruct/content_deletion; Qwen/Qwen2.5-7B-Instruct/irrelevant_context; TIGER-Lab/General-Reasoner-Qwen2.5-7B/content_deletion; TIGER-Lab/General-Reasoner-Qwen2.5-7B/irrelevant_context; allenai/OLMo-2-1124-13B-Instruct/content_deletion; allenai/OLMo-2-1124-13B-Instruct/irrelevant_context; google/gemma-3-12b-it/content_deletion; google/gemma-3-12b-it/irrelevant_context; ibm-granite/granite-3.3-8b-instruct/content_deletion; ibm-granite/granite-3.3-8b-instruct/irrelevant_context; microsoft/Phi-4-reasoning-plus/irrelevant_context; microsoft/phi-4/content_deletion; microsoft/phi-4/irrelevant_context; mistralai/Mistral-Nemo-Instruct-2407/content_deletion; mistralai/Mistral-Nemo-Instruct-2407/irrelevant_context

**Interpretation.** Survival here means aggregate endpoint sign agreement with raw across the split-fitted correction.

**Measurement limitation.** Sign agreement alone does not establish unchanged magnitude or validity under stressed permutations.

**Implication for Stage 2.** Prioritize effects that also have stable raw bootstrap evidence.

## 19. Effects surviving all four response-interface conditions

**Observed result.** Qwen/Qwen2.5-7B-Instruct/content_deletion; Qwen/Qwen2.5-7B-Instruct/irrelevant_context; TIGER-Lab/General-Reasoner-Qwen2.5-7B/content_deletion; allenai/OLMo-2-1124-13B-Instruct/content_deletion; allenai/OLMo-2-1124-13B-Instruct/irrelevant_context; google/gemma-3-12b-it/content_deletion; google/gemma-3-12b-it/irrelevant_context; ibm-granite/granite-3.3-8b-instruct/content_deletion; ibm-granite/granite-3.3-8b-instruct/irrelevant_context; microsoft/phi-4/content_deletion; microsoft/phi-4/irrelevant_context; mistralai/Mistral-Nemo-Instruct-2407/content_deletion; mistralai/Mistral-Nemo-Instruct-2407/irrelevant_context

**Interpretation.** These target-family effects retain the raw aggregate direction under temperature, label bias, and their joint condition.

**Measurement limitation.** The label-bias sensitivity uses ten split fits rather than an additional bootstrap.

**Implication for Stage 2.** These are the strongest candidates for minimal focused validation.

## 20. Score-generation agreement after label-bias correction

**Observed result.** Mean agreement changed from 59.3% to 52.6% across models.

**Interpretation.** The decrease shows that improved held-out permutation invariance did not translate into better score-generation alignment overall.

**Measurement limitation.** The existing reasoning-model generations are truncated and are not treated as ground-truth behavior.

**Implication for Stage 2.** Use a longer deterministic generation budget for the fixed subset.

## 21. Models in the pre-declared ≥70% high-alignment subset

**Observed result.** `Qwen/Qwen2.5-7B-Instruct`, `allenai/OLMo-2-1124-13B-Instruct`, `google/gemma-3-12b-it`, `mistralai/Mistral-Nemo-Instruct-2407`

**Interpretation.** Membership follows the fixed overall raw score-generation agreement threshold without post-hoc changes.

**Measurement limitation.** The threshold is a measurement screen, not a quality ranking.

**Implication for Stage 2.** Use G1 for all qualifying targets and G2 only when roster size permits.

## 22. Persistence in the restricted high-alignment ecosystem

**Observed result.** G2 ran for 4 targets; mean effect-sign agreement with G1 was 87.5%, and mean rank correlation was 0.500.

**Interpretation.** Agreement with G1 indicates how much the findings depend on low-alignment peers.

**Measurement limitation.** G2 changes both target and peer sets and is therefore a sensitivity ecosystem, not the primary design.

**Implication for Stage 2.** If G2 diverges, validate the affected target with both the full and restricted peer sets.

## 23. Invalid or misleading original V2 headlines and replacements

**Observed result.** The phi-4 `0.0867769` max-minus-min value is replaced by paired mean 0.029838 with 95% [0.0218622, 0.0360921]. Single-row lineage maxima are replaced by split-paired sibling means and the sibling-excluding null in Table V2.1 sibling removal. Maximum conditions are now selected only after target-family-dose split aggregation, and convexity headlines use target-family aggregates with fit-only peer selection.

**Interpretation.** The corrected summaries remove cross-split extrema and null contamination.

**Measurement limitation.** The replacements remain specific to the fixed design and response interface.

**Implication for Stage 2.** Use only V2.1 tables for Stage 2 prioritization.

## 24. Original V2 input byte identity

**Observed result.** Yes. The before/after inventories are byte-identical.

**Interpretation.** Matching SHA256, size, and nanosecond modification time confirms the reanalysis did not alter the locked inputs.

**Measurement limitation.** Byte identity does not by itself validate the scientific measurement model.

**Implication for Stage 2.** Stage 2 must write to a separate root as well.

## 25. Is focused B200 Stage 2 still scientifically necessary?

**Observed result.** Yes: residual option-permutation TV, truncated generation alignment, and interface-sensitive endpoint effects remain.

**Interpretation.** New inference is justified only for focused measurement validation, not to recompute the corrected CPU summaries.

**Measurement limitation.** Stage 1 cannot test stressed-prompt label transport or longer generation without new inference.

**Implication for Stage 2.** Proceed only with the minimal scope listed next.

## 26. Exact proposed Stage 2 scope

**Observed result.** Models: `Qwen/Qwen2.5-7B-Instruct`, `TIGER-Lab/General-Reasoner-Qwen2.5-7B`, `microsoft/Phi-4-reasoning-plus`, `microsoft/phi-4`, `mistralai/Mistral-Nemo-Instruct-2407`. Conditions: clean, irrelevant-context 512 words on each of the three fixed tracks, and content-deletion 0.4 on each of the three fixed tracks; validate canonical and the three fixed clean label permutations, and use a longer deterministic generation budget on the fixed 140-question generation subset.

**Interpretation.** This scope targets the unresolved response-interface and generation-alignment limitations identified by V2.1.

**Measurement limitation.** It remains one benchmark and the same fixed intervention design; broader claims would require separate work.

**Implication for Stage 2.** Do not add models, benchmark questions, or post-hoc doses before completing this focused validation.
