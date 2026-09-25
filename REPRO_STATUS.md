# Reproduction status

Level A **passed** in the organized repository and in a fresh source-only copy
with networking disabled by a Linux user/network namespace. No model inference,
new optimizer fit, or new bootstrap was run. Missing execution is `not_run`.

The current PDF has **22 pages**, with scientific main text on pages 1–9 and
references starting on page 10. All **26 standalone panels** (24 empirical and
2 conceptual), **8 manuscript figures**, and **10 tables** were rebuilt. Each
standalone source has exactly one TikZ picture. All **23 bibliography entries**
are cited; no undefined citations, undefined references, or missing figure files
remain. Visual spot checks of pages 4, 7 and 9 also found the intended layout.

All **77 bundled input/source identities**, **104 regenerated numerical/text
baselines**, and **450 copied-source mappings** passed verification. **No input
SHA256 mismatch occurred.** Original source locations were rehashed and remained
unchanged. The two extracted inline tables preserve the original table text.
The copied legacy protocol blobs match the recorded ISQED Git objects.

## Commands actually passed

After selecting the captured Python and pdfLaTeX environments:

```bash
# New repository, in dependency order; 12 tests passed.
make verify-inputs tables figures paper verify
make help
make environment

# External cached LLM split/bootstrap inputs; 11 tables matched.
BEHAVIORAL_COVERAGE_RAW_DATA=/work/Users/leiyo make analysis

# Fresh source-only copy: no PDFs, generated CSVs, build tree or Git metadata.
# replay/verify ran inside unshare -Urn with raw/model-root variables removed.
make replay
make verify
make clean
```

The final clean-room test also verified that cleaning removes generated products
without changing any frozen evidence. Its repository-independent environments
were already installed; environment installation itself is not an offline claim.
The delivered repository retains its compiled PDF and panel PDFs as ignored local
outputs. Build logs and temporary copies stay in external work storage.

## Every manuscript figure and table

The input column below gives every frozen input path. The full generating-script
and command mapping is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), and the
machine-readable labels/pages are checked against the rebuilt LaTeX auxiliary
file using `manifests/paper_items.json`.

| Component | Replay status | Input source | Output | Notes |
| --- | --- | --- | --- | --- |
| Figure 1 (`fig:framework`) | passed | Conceptual TikZ; no empirical inputs | `paper/main.pdf`, p. 3 | conceptual; 2 panel sources |
| Figure 2 (`fig:llm-coverage`) | passed | `data/frozen/llm/v23/table_v23_distributed_convex_coverage.csv`<br>`data/frozen/llm/v23/table_v23_sibling_coverage.csv` | `paper/main.pdf`, p. 5 | llm; 4 panel sources |
| Figure 3 (`fig:stress`) | passed | `data/frozen/llm/v22/table_2_estimand_decomposition.csv`<br>`data/frozen/llm/v23/table_v23_endpoint_claim_comparison.csv` | `paper/main.pdf`, p. 7 | llm; 3 panel sources |
| Figure 4 (`fig:vision`) | passed | `data/frozen/vision/artifacts/exp11/exp11_shape_bias_ConvNeXtTiny_residuals.npz`<br>`data/frozen/vision/artifacts/exp11/exp11_shape_bias_ShapeResNet50_ShapeResNet_residuals.npz`<br>`data/frozen/vision/artifacts/exp11/exp11_texture_natural_ConvNeXtTiny_residuals.npz`<br>`data/frozen/vision/artifacts/exp11/exp11_texture_natural_ShapeResNet50_ShapeResNet_residuals.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`<br>`data/frozen/vision/tables/exp10_imagenet_adv_pier.csv`<br>`data/frozen/vision/tables/exp11_shape_texture_pier.csv` | `paper/main.pdf`, p. 8 | vision; 6 panel sources |
| Figure 5 (`fig:interface-diagnostics`) | passed | `data/frozen/llm/v22/table_2_estimand_decomposition.csv`<br>`data/frozen/llm/v23/table_v23_endpoint_claim_comparison.csv` | `paper/main.pdf`, p. 17 | llm; 2 panel sources |
| Figure 6 (`fig:geometry-all`) | passed | `data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_texture_natural.npz` | `paper/main.pdf`, p. 20 | vision; 6 panel sources |
| Figure 7 (`fig:traffic`) | passed | `data/frozen/traffic/tables/exp14_multicity_tabular_summary.csv`<br>`data/frozen/traffic/tables/exp14_multicity_tabular_summary_peer_weights_long.csv` | `paper/main.pdf`, p. 21 | traffic; 2 panel sources |
| Figure 8 (`fig:legacy`) | passed | `data/frozen/bert_legacy/tables/exp4_bert_disco_dosesplit.csv`<br>`data/frozen/bert_legacy/tables/exp6_bert_multicontext_pier.csv` | `paper/main.pdf`, p. 22 | bert_legacy; 3 panel sources |
| Table 1 (`tab:controls`) | passed | `data/frozen/llm/v2/control_results.json` | `paper/main.pdf`, p. 4 | llm; Numerical table |
| Table 2 (`tab:vector-main`) | passed | `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet`<br>`data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv`<br>`data/frozen/llm/v22/vector_audit/stage_11_vector_sensitivity.json` | `paper/main.pdf`, p. 6 | llm; Numerical table |
| Table 3 (`tab:vector-sibling`) | passed | `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet`<br>`data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv`<br>`data/frozen/llm/v22/vector_audit/stage_11_vector_sensitivity.json` | `paper/main.pdf`, p. 6 | llm; Numerical table |
| Table 4 (`tab:models`) | passed | `data/frozen/llm/v2/table1_ecosystem_manifest.csv` | `paper/main.pdf`, p. 13 | llm; Numerical table |
| Table 5 (`tab:endpoints-full`) | passed | `data/frozen/llm/v23/table_v23_endpoint_claim_comparison.csv` | `paper/main.pdf`, p. 15 | llm; Numerical table |
| Table 6 (`tab:all-gains`) | passed | `data/frozen/llm/v23/table_v23_distributed_convex_coverage.csv` | `paper/main.pdf`, p. 15 | llm; Numerical table |
| Table 7 (`tab:sibling-full`) | passed | `data/frozen/llm/v23/table_v23_sibling_coverage.csv` | `paper/main.pdf`, p. 16 | llm; Numerical table |
| Table 8 (`tab:vector-all`) | passed | `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet`<br>`data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv`<br>`data/frozen/llm/v22/vector_audit/stage_11_vector_sensitivity.json` | `paper/main.pdf`, p. 17 | llm; Numerical table |
| Table 9 (`tab:generation`) | passed | `data/frozen/llm/v23/table_v23_generation_validation.csv` | `paper/main.pdf`, p. 18 | llm; Numerical table |
| Table 10 (`tab:geometry-support`) | passed | `data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_texture_natural.npz` | `paper/main.pdf`, p. 19 | vision; Numerical table |

## Scope by experimental family

| Family | Level A | Level B actually executed | Level C |
| --- | --- | --- | --- |
| Modern LLM | passed | 6 trackwise + 5 balanced-interface reporting tables from 32 hash-checked cached split/bootstrap/config files; independent scalar export from the 1,600-row vector Parquet | Documented with immutable checkpoint/dataset revisions and fresh-workspace adapter; `not_run` |
| Vision | passed | Residual concentration/overlap from all 400 evaluation positions per context; six stored PCA coordinate views; archived adversarial/context summaries | Source-derived commands documented; `not_run`; historical images/checkpoint/runtime identities incomplete |
| Traffic | passed | Exact 31-city archived summary/weight transformations; raw prediction-level analysis unavailable | Source-derived commands documented; `not_run`; historical data/training cache and invocation unresolved |
| Legacy BERT classifiers | passed | Exact archived summary transformations; raw prediction-level analysis unavailable | Source-derived commands documented; `not_run`; historical prediction caches and immutable revisions unavailable |

The LLM Level-B result starts from cached fitted outcomes and existing bootstrap
distributions. A complete CPU score-to-fit/bootstrap campaign was **not run**.
Vision coordinate replay is **not** a PCA refit: the archives omit the original
high-dimensional response vectors. No missing traffic pruning/tail curve was
reconstructed, and the archived minimum coefficient `-0.0048245451111272` was
retained. Historical failed V2.3 gates and the later authorized repair remain in
the provenance records. Canonical five-dose scalar/vector audits, balanced scalar
endpoints, transferred/refitted weights, and auxiliary generation are separate.

Full commands and unresolved inputs are documented in
[docs/FULL_INFERENCE_REPRODUCTION.md](docs/FULL_INFERENCE_REPRODUCTION.md) and
[docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md). No full-inference result,
public submission, remote push, author identity or blanket license was invented.

## Recorded evidence

Current runs write `build/qa/` and `build/analysis/verification.json`. The packaging
snapshots retained in the source manifest directory are:

- `manifests/verification/level_a.json`
- `manifests/verification/clean_room.json`
- `manifests/verification/cached_analysis.json`
- `manifests/verification/upstream.json`

These snapshots record this completed packaging verification; historical QA from
the upload remains separately labeled in `manifests/paper_package/`. PDF hashes
identify particular builds, not cross-machine pixel reproducibility.
