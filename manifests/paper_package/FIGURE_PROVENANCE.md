# Figure provenance

All empirical coordinates are derived from bundled result tables or arrays. Only the two opening conceptual panels use illustrative geometry. Original observations are not silently repaired, filled in, or fitted anew.

The generated inputs use full-precision CSV values; decimal rounding in labels or table display is not a replacement for the stored values. Input byte-identity checks against the user ZIP files appear in `qa/input_identity.json`.

## `fig1a_matched_audit`

- Source: `figures/standalone/fig1a_matched_audit.tex`
- PDF: `figures/pdf/fig1a_matched_audit.pdf`
- Transformation: Conceptual matched-audit schematic; no observations.
- Data status: conceptual; no experimental observations.

## `fig1b_peer_projection`

- Source: `figures/standalone/fig1b_peer_projection.tex`
- PDF: `figures/pdf/fig1b_peer_projection.pdf`
- Transformation: Conceptual projection onto a rectangle; target projection is on the boundary.
- Data status: conceptual; no experimental observations.

## `fig2a_group_gain`

- Source: `figures/standalone/fig2a_group_gain.tex`
- PDF: `figures/pdf/fig2a_group_gain.pdf`
- Transformation: 100*(single-group)/single. All eight targets; pointwise CI classification stored in llm_all_gains.csv.
- Input: `data/table_v23_distributed_convex_coverage.csv`; SHA256 `7bf44c7d152ada13d3fddf69e3943d7a43d32ebaf508ce08c8c21f03e5f9d1e7`

## `fig2b_sibling_removal`

- Source: `figures/standalone/fig2b_sibling_removal.tex`
- PDF: `figures/pdf/fig2b_sibling_removal.pdf`
- Transformation: Stored mean removal effects and their recorded intervals; no inferred per-peer intervals.
- Input: `data/table_v23_sibling_coverage.csv`; SHA256 `8bb2a15c7c12561d75a6aaec84459164a823ba3d63ae5bedd330d3397bc414a0`
- Plot-ready tables: `data/figures/qwen_removal.csv`

## `fig2c_directional_coverage`

- Source: `figures/standalone/fig2c_directional_coverage.tex`
- PDF: `figures/pdf/fig2c_directional_coverage.pdf`
- Transformation: MSE-fit single and convex held-out MAE, both targets and both families.
- Input: `data/table_v23_distributed_convex_coverage.csv`; SHA256 `7bf44c7d152ada13d3fddf69e3943d7a43d32ebaf508ce08c8c21f03e5f9d1e7`
- Plot-ready tables: `data/figures/qwen_direction.csv`

## `fig2d_distributed_gain`

- Source: `figures/standalone/fig2d_distributed_gain.tex`
- PDF: `figures/pdf/fig2d_distributed_gain.pdf`
- Transformation: Absolute matched-objective gains; CIs copied from archived improvement intervals.
- Input: `data/table_v23_distributed_convex_coverage.csv`; SHA256 `7bf44c7d152ada13d3fddf69e3943d7a43d32ebaf508ce08c8c21f03e5f9d1e7`
- Plot-ready tables: `data/figures/distributed_MSE.csv`, `data/figures/distributed_MAE.csv`

## `fig3a_context_trajectory`

- Source: `figures/standalone/fig3a_context_trajectory.tex`
- PDF: `figures/pdf/fig3a_context_trajectory.pdf`
- Transformation: Canonical five-dose trackwise-fit/trackwise-evaluation means. Values already averaged across splits in source table.
- Input: `data/v22/table_2_estimand_decomposition.csv`; SHA256 `fdd9048ef6c6c8c49d753a60aa0ed1f5d16739e4371ab6860fd9cb8fc8f1ad3e`
- Plot-ready tables: `data/figures/fig3a_context_trajectory_Qwen2.5.csv`, `data/figures/fig3a_context_trajectory_phi-4.csv`, `data/figures/fig3a_context_trajectory_OLMo.csv`, `data/figures/fig3a_context_trajectory_Mistral.csv`

## `fig3b_deletion_trajectory`

- Source: `figures/standalone/fig3b_deletion_trajectory.tex`
- PDF: `figures/pdf/fig3b_deletion_trajectory.pdf`
- Transformation: Canonical five-dose trackwise-fit/trackwise-evaluation means. Values already averaged across splits in source table.
- Input: `data/v22/table_2_estimand_decomposition.csv`; SHA256 `fdd9048ef6c6c8c49d753a60aa0ed1f5d16739e4371ab6860fd9cb8fc8f1ad3e`
- Plot-ready tables: `data/figures/fig3b_deletion_trajectory_Qwen2.5.csv`, `data/figures/fig3b_deletion_trajectory_phi-4.csv`, `data/figures/fig3b_deletion_trajectory_OLMo.csv`, `data/figures/fig3b_deletion_trajectory_Mistral.csv`

## `fig3c_balanced_effects`

- Source: `figures/standalone/fig3c_balanced_effects.tex`
- PDF: `figures/pdf/fig3c_balanced_effects.pdf`
- Transformation: Balanced endpoint changes and pointwise bootstrap intervals for named headline cases; all eight retained in appendix table.
- Input: `data/table_v23_endpoint_claim_comparison.csv`; SHA256 `d2eb7971c0eaeca6259e50562258e95f03a1c2e7f96326a439273b298aecb3cf`
- Plot-ready tables: `data/figures/stress_effects_I.csv`, `data/figures/stress_effects_D.csv`

## `fig4a_vision_stress`

- Source: `figures/standalone/fig4a_vision_stress.tex`
- PDF: `figures/pdf/fig4a_vision_stress.pdf`
- Transformation: All six archived curves, normalized by model-specific dose-zero residual. No CI reconstruction.
- Input: `data/original/tables/exp10_imagenet_adv_pier.csv`; SHA256 `a496520f38f6a09220f8cf4afedc68a4338bcced10025812dff764784671ad2f`
- Plot-ready tables: `data/figures/vision_adv_ResNet50.csv`, `data/figures/vision_adv_ConvNeXtTiny.csv`, `data/figures/vision_adv_RobustResNet50.csv`, `data/figures/vision_adv_ResNet18.csv`, `data/figures/vision_adv_EfficientNetB0.csv`, `data/figures/vision_adv_ViT_B16.csv`

## `fig4b_vision_context`

- Source: `figures/standalone/fig4b_vision_context.tex`
- PDF: `figures/pdf/fig4b_vision_context.pdf`
- Transformation: Seven-model context audit; no averaging over separated shape-variant ecosystems.
- Input: `data/original/tables/exp11_shape_texture_pier.csv`; SHA256 `6d348c72d83da0533458e738cac31b274220953ef2e4737bf6aa32b779af0af9`
- Plot-ready tables: `data/figures/vision_context.csv`

## `fig4c_geometry_natural`

- Source: `figures/standalone/fig4c_geometry_natural.tex`
- PDF: `figures/pdf/fig4c_geometry_natural.pdf`
- Transformation: Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- Input: `data/original/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`; SHA256 `936bce6153ecdceb4d3135ce6dce444d840dd16ab4a45e5312c913933a763616`
- Plot-ready tables: `data/figures/fig4c_geometry_natural_hull.csv`, `data/figures/fig4c_geometry_natural_peers.csv`, `data/figures/fig4c_geometry_natural_mix.csv`, `data/figures/fig4c_geometry_natural_target.csv`

## `fig4d_geometry_shape`

- Source: `figures/standalone/fig4d_geometry_shape.tex`
- PDF: `figures/pdf/fig4d_geometry_shape.pdf`
- Transformation: Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- Input: `data/original/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`; SHA256 `118facb9febd47123c6efdb3615c996ca779476f3ae483f0a5a0d700047f835e`
- Plot-ready tables: `data/figures/fig4d_geometry_shape_hull.csv`, `data/figures/fig4d_geometry_shape_peers.csv`, `data/figures/fig4d_geometry_shape_mix.csv`, `data/figures/fig4d_geometry_shape_target.csv`

## `figD_geometry_SININ_natural`

- Source: `figures/standalone/figD_geometry_SININ_natural.tex`
- PDF: `figures/pdf/figD_geometry_SININ_natural.pdf`
- Transformation: Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- Input: `data/original/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_texture_natural.npz`; SHA256 `14a80e22fdc778a55aa8d14370701a83b16f6e897bef35b9115dd43f01fd11ea`
- Plot-ready tables: `data/figures/figD_geometry_SININ_natural_hull.csv`, `data/figures/figD_geometry_SININ_natural_peers.csv`, `data/figures/figD_geometry_SININ_natural_mix.csv`, `data/figures/figD_geometry_SININ_natural_target.csv`

## `figD_geometry_SININ_shape`

- Source: `figures/standalone/figD_geometry_SININ_shape.tex`
- PDF: `figures/pdf/figD_geometry_SININ_shape.pdf`
- Transformation: Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- Input: `data/original/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_shape_bias.npz`; SHA256 `abcc6b0c16b8ece1e023564f9ba03551819b3588377157622e9fb66ea21ae9e0`
- Plot-ready tables: `data/figures/figD_geometry_SININ_shape_hull.csv`, `data/figures/figD_geometry_SININ_shape_peers.csv`, `data/figures/figD_geometry_SININ_shape_mix.csv`, `data/figures/figD_geometry_SININ_shape_target.csv`

## `figD_geometry_C_natural`

- Source: `figures/standalone/figD_geometry_C_natural.tex`
- PDF: `figures/pdf/figD_geometry_C_natural.pdf`
- Transformation: Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- Input: `data/original/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_texture_natural.npz`; SHA256 `8e99141ec62af841388f01da2bb9cd4348c0d6bd92d09fd8134f2959b1c4974c`
- Plot-ready tables: `data/figures/figD_geometry_C_natural_hull.csv`, `data/figures/figD_geometry_C_natural_peers.csv`, `data/figures/figD_geometry_C_natural_mix.csv`, `data/figures/figD_geometry_C_natural_target.csv`

## `figD_geometry_C_shape`

- Source: `figures/standalone/figD_geometry_C_shape.tex`
- PDF: `figures/pdf/figD_geometry_C_shape.pdf`
- Transformation: Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.
- Input: `data/original/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_shape_bias.npz`; SHA256 `9e2957ad037165b338e14b164f4f58c0f0aacbfcbb2483c4383f7612642bf12f`
- Plot-ready tables: `data/figures/figD_geometry_C_shape_hull.csv`, `data/figures/figD_geometry_C_shape_peers.csv`, `data/figures/figD_geometry_C_shape_mix.csv`, `data/figures/figD_geometry_C_shape_target.csv`

## `fig4e_residual_mass`

- Source: `figures/standalone/fig4e_residual_mass.tex`
- PDF: `figures/pdf/fig4e_residual_mass.pdf`
- Transformation: Exact cumulative sorted residual mass at all 401 integer counts; zero included.
- Input: `data/original/artifacts/exp11/exp11_texture_natural_ConvNeXtTiny_residuals.npz`; SHA256 `e9351d7722083225b1b6f462153e08de341bd5017cf3f6de0c5d01af557a0389`
- Input: `data/original/artifacts/exp11/exp11_shape_bias_ConvNeXtTiny_residuals.npz`; SHA256 `af3ac995c011c920580a24fded748c89dfcdfa99234884f1f70a98c033c1a781`
- Plot-ready tables: `data/figures/residual_mass_texture_natural.csv`, `data/figures/residual_mass_shape_bias.csv`

## `fig4f_residual_overlap`

- Source: `figures/standalone/fig4f_residual_overlap.tex`
- PDF: `figures/pdf/fig4f_residual_overlap.pdf`
- Transformation: Matched array-position overlap at every k=1..40; stable index tie-breaking, no fabricated intermediate fractions.
- Input: `data/original/artifacts/exp11/exp11_shape_bias_ConvNeXtTiny_residuals.npz`; SHA256 `af3ac995c011c920580a24fded748c89dfcdfa99234884f1f70a98c033c1a781`
- Input: `data/original/artifacts/exp11/exp11_shape_bias_ShapeResNet50_ShapeResNet_residuals.npz`; SHA256 `a1f36bdde546da977669ca1f439b098ad215a34417d1205b6d3afc4cf6c7db2f`
- Input: `data/original/artifacts/exp11/exp11_texture_natural_ConvNeXtTiny_residuals.npz`; SHA256 `e9351d7722083225b1b6f462153e08de341bd5017cf3f6de0c5d01af557a0389`
- Input: `data/original/artifacts/exp11/exp11_texture_natural_ShapeResNet50_ShapeResNet_residuals.npz`; SHA256 `436a89ef9f380f74eea550e5aa66f9ca543c99c98aa8dccf8d720d0afba0fba8`
- Plot-ready tables: `data/figures/residual_overlap_texture_natural.csv`, `data/figures/residual_overlap_shape_bias.csv`

## `fig5a_traffic_utility`

- Source: `figures/standalone/fig5a_traffic_utility.tex`
- PDF: `figures/pdf/fig5a_traffic_utility.pdf`
- Transformation: All 31 exact city points; stated normalizations; flag only highlights numerical feasibility, no rows excluded.
- Input: `data/original/tables/exp14_multicity_tabular_summary.csv`; SHA256 `a62a135b13518432af5c242f3e95a2e481d8cd750021e658e292cb94d5d76700`
- Input: `data/original/tables/exp14_multicity_tabular_summary_peer_weights_long.csv`; SHA256 `3f47367b6c526573d794a48ca528747cd119db2b22eaecc4e25bdaaa9bdaed9f`
- Plot-ready tables: `data/figures/traffic_False.csv`, `data/figures/traffic_True.csv`

## `fig5b_traffic_group`

- Source: `figures/standalone/fig5b_traffic_group.tex`
- PDF: `figures/pdf/fig5b_traffic_group.pdf`
- Transformation: All 31 archived points, identity reference; 26 improve, five do not. No claim of training-only single selection.
- Input: `data/original/tables/exp14_multicity_tabular_summary.csv`; SHA256 `a62a135b13518432af5c242f3e95a2e481d8cd750021e658e292cb94d5d76700`
- Plot-ready tables: `data/figures/traffic_False.csv`, `data/figures/traffic_True.csv`

## `figA_interface_change`

- Source: `figures/standalone/figA_interface_change.tex`
- PDF: `figures/pdf/figA_interface_change.pdf`
- Transformation: All 16 matched-design endpoint effects; Spearman 0.93235, not the eight-target error-ranking correlation.
- Input: `data/table_v23_endpoint_claim_comparison.csv`; SHA256 `d2eb7971c0eaeca6259e50562258e95f03a1c2e7f96326a439273b298aecb3cf`
- Plot-ready tables: `data/figures/interface_effects.csv`

## `figA_cancellation`

- Source: `figures/standalone/figA_cancellation.tex`
- PDF: `figures/pdf/figA_cancellation.pdf`
- Transformation: Mean across all rows of trackwise_fit_aggregate_eval versus trackwise_fit_trackwise_eval; same fitted weights.
- Input: `data/v22/table_2_estimand_decomposition.csv`; SHA256 `fdd9048ef6c6c8c49d753a60aa0ed1f5d16739e4371ab6860fd9cb8fc8f1ad3e`
- Plot-ready tables: `data/figures/cancellation_fixed.csv`

## `figF_bert_baseline`

- Source: `figures/standalone/figF_bert_baseline.tex`
- PDF: `figures/pdf/figF_bert_baseline.pdf`
- Transformation: Subgroup-mean differences at dose 0; no causal interpretation or invented intervals.
- Input: `data/original/tables/exp6_bert_multicontext_pier.csv`; SHA256 `87fe79408890e3ca632d9fb5c5d5e9b7211a558bcd4d31e2d9a59e608edd6986`
- Plot-ready tables: `data/figures/figF_bert_baseline.csv`

## `figF_bert_mid`

- Source: `figures/standalone/figF_bert_mid.tex`
- PDF: `figures/pdf/figF_bert_mid.pdf`
- Transformation: Subgroup-mean differences at dose 0.5; no causal interpretation or invented intervals.
- Input: `data/original/tables/exp6_bert_multicontext_pier.csv`; SHA256 `87fe79408890e3ca632d9fb5c5d5e9b7211a558bcd4d31e2d9a59e608edd6986`
- Plot-ready tables: `data/figures/figF_bert_mid.csv`

## `figF_bert_controls`

- Source: `figures/standalone/figF_bert_controls.tex`
- PDF: `figures/pdf/figF_bert_controls.pdf`
- Transformation: All 30 exact dose-response values, including 1e-8 clone values.
- Input: `data/original/tables/exp4_bert_disco_dosesplit.csv`; SHA256 `211ef7fca15057ffecb402b234b63629788a6b11630f3ee0f9812d678e9d3826`
- Plot-ready tables: `data/figures/bert_control_Clone.csv`, `data/figures/bert_control_RoBERTa.csv`, `data/figures/bert_control_Finetuned.csv`

## Complete-option-distribution tables

Sources under `data/v22/vector_audit/`:
- `trackwise_vector_results.parquet`; SHA256 `1604bc3b71ba193fea699dec26ec24965230a73435ba69588e9f1bf82e1a05a9`
- `stage_11_vector_sensitivity.json`; SHA256 `4206cfc62c467b18ebe60771fde8c0cfb7c4f0992be7f85db1140f0a37f2d3a4`
- `trackwise_vector_scalar_columns.csv`; SHA256 `d3c8ad0a9b77f8c43def3f9119e32d9d99d98c4e29d8d4d5b6188592a502155e`
- `sensitivities.py`; SHA256 `4dcfa2aeb425c11abf97de5c0718adc72ceb1545ff81b6bdfca029938de0e252`
- `data.py`; SHA256 `a7bbe1d6b6ac2b68c32c43718a2715a974e499001d18331a12d8bfe5ed186f19`

The completed 1,600-row audit contains five dose rows for each target/family/split/weight-source combination. Scalar columns are exported at 17-digit precision. Existing overall statistics are deduplicated across dose before averaging ten splits. No fit, resampling, model inference, or new labels are computed. The transferred-weight and vector-refit paths are kept separate. Main and appendix tables are generated by `scripts/build_vector_evidence.py`.
