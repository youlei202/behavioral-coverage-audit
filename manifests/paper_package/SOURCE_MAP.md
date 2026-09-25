# Source map

Composite figures are assembled in section/appendix files using `subfigure`. Every panel below has exactly one standalone source and one TikZ picture.

- `fig1a_matched_audit` — Conceptual matched-audit schematic; no observations.
- `fig1b_peer_projection` — Conceptual projection onto a rectangle; target projection is on the boundary.
- `fig2a_group_gain` — 100*(single-group)/single. All eight targets; pointwise CI classification stored in llm_all_gains.csv.
- `fig2b_sibling_removal` — Stored mean removal effects and their recorded intervals; no inferred per-peer intervals.
- `fig2c_directional_coverage` — MSE-fit single and convex held-out MAE, both targets and both families.
- `fig2d_distributed_gain` — Absolute matched-objective gains; CIs copied from archived improvement intervals.
- `fig3a_context_trajectory` — Canonical five-dose trackwise-fit/trackwise-evaluation means. Values already averaged across splits in source table.
- `fig3b_deletion_trajectory` — Canonical five-dose trackwise-fit/trackwise-evaluation means. Values already averaged across splits in source table.
- `fig3c_balanced_effects` — Balanced endpoint changes and pointwise bootstrap intervals for named headline cases; all eight retained in appendix table.
- `fig4a_vision_stress` — All six archived curves, normalized by model-specific dose-zero residual. No CI reconstruction.
- `fig4b_vision_context` — Seven-model context audit; no averaging over separated shape-variant ecosystems.
- `fig4c_geometry_natural` — Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- `fig4d_geometry_shape` — Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- `figD_geometry_SININ_natural` — Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- `figD_geometry_SININ_shape` — Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- `figD_geometry_C_natural` — Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- `figD_geometry_C_shape` — Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- `fig4e_residual_mass` — Exact cumulative sorted residual mass at all 401 integer counts; zero included.
- `fig4f_residual_overlap` — Matched array-position overlap at every k=1..40; stable index tie-breaking, no fabricated intermediate fractions.
- `fig5a_traffic_utility` — All 31 exact city points; stated normalizations; flag only highlights numerical feasibility, no rows excluded.
- `fig5b_traffic_group` — All 31 archived points, identity reference; 26 improve, five do not. No claim of training-only single selection.
- `figA_interface_change` — All 16 matched-design endpoint effects; Spearman 0.93235, not the eight-target error-ranking correlation.
- `figA_cancellation` — Mean across all rows of trackwise_fit_aggregate_eval versus trackwise_fit_trackwise_eval; same fitted weights.
- `figF_bert_baseline` — Subgroup-mean differences at dose 0; no causal interpretation or invented intervals.
- `figF_bert_mid` — Subgroup-mean differences at dose 0.5; no causal interpretation or invented intervals.
- `figF_bert_controls` — All 30 exact dose-response values, including 1e-8 clone values.
## Current manuscript assembly and tables

- Main Figure 1: two conceptual panels, assembled in Section 2.
- Main Figure 2: balanced reference-answer group/removal comparisons, assembled in Section 4.
- Main Figure 3: canonical dose trajectories and balanced endpoints, assembled in Section 5.
- Main Figure 4: exact vision stress/context, fitting geometry and residual localization, assembled in Section 6.
- Appendix Figure 5: interface endpoint comparison and intervention aggregation, Appendix C.4.
- Appendix Figure 6: all six separately fitted PCA views, Appendix D.2.
- Appendix Figure 7: archived forecasting scatter comparisons, Appendix D.4. Historical source filenames still begin with `fig5`; these are not a main-text downstream experiment.
- Appendix Figure 8: sentiment-classifier subgroup/control panels, Appendix F.

The complete-option tables are not images or new plots. `vector_main.tex` and `vector_siblings.tex` are main-text Tables 2 and 3 in Section 4.4; `vector_all.tex` remains in Appendix C.3. They are generated from the same completed canonical vector audit by `build_vector_evidence.py`. Table 1 contains exact-response controls. The other tables retain model identities, complete primary comparisons, generation readouts, and archived geometry support in the appendices. The main text occupies pages 1–9; references start on page 10.
