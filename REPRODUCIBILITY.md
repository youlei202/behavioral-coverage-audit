# Reproducibility

Level A verifies the uploaded frozen inputs, regenerates numerical tables and panel
sources, compiles all 26 panel PDFs, and performs a clean manuscript build. It uses
no GPU, network, model cache or old experiment directory. After environment setup:

```bash
make verify-inputs
make tables
make figures
make paper
make verify
```

`make replay` combines the build stages. `make clean` preserves frozen evidence
and maintained LaTeX while removing generated data and compiled products.

## Every manuscript figure and table

The inputs below are exact relative paths. Machine-readable mappings are in
`manifests/paper_items.json`; individual panels are expanded in
`docs/PAPER_FIGURE_MAP.md`. The item numbers come from the rebuilt manuscript's
labels, not historical filename prefixes.

| Paper item | Family | Frozen inputs | Generating script | Command | Output |
| --- | --- | --- | --- | --- | --- |
| Figure 1 (`fig:framework`) | conceptual | Conceptual: no empirical observations | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 3 |
| Figure 2 (`fig:llm-coverage`) | llm | `data/frozen/llm/v23/table_v23_distributed_convex_coverage.csv`<br>`data/frozen/llm/v23/table_v23_sibling_coverage.csv` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 5 |
| Figure 3 (`fig:stress`) | llm | `data/frozen/llm/v22/table_2_estimand_decomposition.csv`<br>`data/frozen/llm/v23/table_v23_endpoint_claim_comparison.csv` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 7 |
| Figure 4 (`fig:vision`) | vision | `data/frozen/vision/artifacts/exp11/exp11_shape_bias_ConvNeXtTiny_residuals.npz`<br>`data/frozen/vision/artifacts/exp11/exp11_shape_bias_ShapeResNet50_ShapeResNet_residuals.npz`<br>`data/frozen/vision/artifacts/exp11/exp11_texture_natural_ConvNeXtTiny_residuals.npz`<br>`data/frozen/vision/artifacts/exp11/exp11_texture_natural_ShapeResNet50_ShapeResNet_residuals.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`<br>`data/frozen/vision/tables/exp10_imagenet_adv_pier.csv`<br>`data/frozen/vision/tables/exp11_shape_texture_pier.csv` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 8 |
| Figure 5 (`fig:interface-diagnostics`) | llm | `data/frozen/llm/v22/table_2_estimand_decomposition.csv`<br>`data/frozen/llm/v23/table_v23_endpoint_claim_comparison.csv` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 17 |
| Figure 6 (`fig:geometry-all`) | vision | `data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_texture_natural.npz` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 20 |
| Figure 7 (`fig:traffic`) | traffic | `data/frozen/traffic/tables/exp14_multicity_tabular_summary.csv`<br>`data/frozen/traffic/tables/exp14_multicity_tabular_summary_peer_weights_long.csv` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 21 |
| Figure 8 (`fig:legacy`) | bert_legacy | `data/frozen/bert_legacy/tables/exp4_bert_disco_dosesplit.csv`<br>`data/frozen/bert_legacy/tables/exp6_bert_multicontext_pier.csv` | `scripts/build_assets.py` | `make tables figures paper` | `paper/main.pdf`, p. 22 |
| Table 1 (`tab:controls`) | llm | `data/frozen/llm/v2/control_results.json` | `scripts/generate_inline_tables.py` | `make tables paper` | `paper/main.pdf`, p. 4 |
| Table 2 (`tab:vector-main`) | llm | `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet`<br>`data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv`<br>`data/frozen/llm/v22/vector_audit/stage_11_vector_sensitivity.json` | `scripts/build_vector_evidence.py` | `make tables paper` | `paper/main.pdf`, p. 6 |
| Table 3 (`tab:vector-sibling`) | llm | `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet`<br>`data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv`<br>`data/frozen/llm/v22/vector_audit/stage_11_vector_sensitivity.json` | `scripts/build_vector_evidence.py` | `make tables paper` | `paper/main.pdf`, p. 6 |
| Table 4 (`tab:models`) | llm | `data/frozen/llm/v2/table1_ecosystem_manifest.csv` | `scripts/generate_inline_tables.py` | `make tables paper` | `paper/main.pdf`, p. 13 |
| Table 5 (`tab:endpoints-full`) | llm | `data/frozen/llm/v23/table_v23_endpoint_claim_comparison.csv` | `scripts/build_assets.py` | `make tables paper` | `paper/main.pdf`, p. 15 |
| Table 6 (`tab:all-gains`) | llm | `data/frozen/llm/v23/table_v23_distributed_convex_coverage.csv` | `scripts/build_assets.py` | `make tables paper` | `paper/main.pdf`, p. 15 |
| Table 7 (`tab:sibling-full`) | llm | `data/frozen/llm/v23/table_v23_sibling_coverage.csv` | `scripts/build_assets.py` | `make tables paper` | `paper/main.pdf`, p. 16 |
| Table 8 (`tab:vector-all`) | llm | `data/frozen/llm/v22/vector_audit/trackwise_vector_results.parquet`<br>`data/frozen/llm/v22/vector_audit/trackwise_vector_scalar_columns.csv`<br>`data/frozen/llm/v22/vector_audit/stage_11_vector_sensitivity.json` | `scripts/build_vector_evidence.py` | `make tables paper` | `paper/main.pdf`, p. 17 |
| Table 9 (`tab:generation`) | llm | `data/frozen/llm/v23/table_v23_generation_validation.csv` | `scripts/build_assets.py` | `make tables paper` | `paper/main.pdf`, p. 18 |
| Table 10 (`tab:geometry-support`) | vision | `data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SININ_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_SIN_texture_natural.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_shape_bias.npz`<br>`data/frozen/vision/artifacts/exp13/exp13_geometry_ShapeResNet50_ShapeResNet_texture_natural.npz` | `scripts/build_assets.py` | `make tables paper` | `paper/main.pdf`, p. 19 |

## Deeper replay

`make analysis` takes `BEHAVIORAL_COVERAGE_RAW_DATA` as the parent of the archived
PIER directories. It checks all 32 external input hashes, copies the 5.1 MB of
required cached split/bootstrap/configuration inputs into `build/analysis/`,
regenerates six trackwise and five balanced-interface tables with the archived
reporting modules, and compares all columns to the frozen tables. It independently
exports the scalar vector fields from Parquet. Input hashes are checked again
afterward. Original solver, inference and bootstrap execution are not invoked.

The regular panel replay computes image residual concentration and top-k overlap
from stored NPZ arrays, reads all six separately fitted PCA coordinate archives,
and derives traffic and legacy BERT summaries from their exact archived tables.
Missing high-dimensional PCA responses, city prediction traces and classifier
predictions limit deeper replay, as recorded in `docs/KNOWN_LIMITATIONS.md`.

Full model/data inference commands are in `docs/FULL_INFERENCE_REPRODUCTION.md`.
They were documented but not run during packaging. No missing execution is
represented by a zero performance value.

## Numerical comparisons

Frozen inputs use strict byte SHA256 identity. The 104 baseline numeric/text
outputs are also checked byte-for-byte under the captured replay environment.
Independent Parquet/CSV and cached-table checks allow the already documented
2e-15 floating conversion tolerance; the cumulative 400-element residual sum
check uses 2e-13 absolute arithmetic tolerance. These are serialization/arithmetic
checks, not new scientific decision thresholds. Original control gates and all
reported uncertainty, solver diagnostics and denominators remain unchanged.

The clean-room procedure creates a new copy without Git metadata or any compiled
PDF, generated CSV, log, or LaTeX auxiliary file and executes `make replay` and
`make verify`, with networking disabled when Linux user/network namespaces are
available. It then checks that `make clean` removes generated products and leaves
all frozen evidence unchanged. See `scripts/clean_room.py` and `REPRO_STATUS.md`
for recorded results.
