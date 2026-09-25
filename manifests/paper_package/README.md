# Auditing Behavioral Coverage in Model Ecosystems

Page-top-figure revision of the nine-page, claim-led ICLR-format manuscript. The narrative distinguishes **how well a collection reconstructs a target** from **which peers supply that reconstruction**, then tests this relation across recorded responses and controlled input conditions.

The scientific main text fills pages **1–9**. References begin on **page 10**. The added space is used for existing Phi controls, vector sibling-removal evidence, scalar-to-vector coefficient transfer, and the other archived shape-trained targets. Body fonts, text area, and scientific content are unchanged. All eight composite figures are placed at the top of their pages. Float placement is documented in `FLOAT_LAYOUT_NOTES.md`; the preceding content revision remains documented in `NINE_PAGE_REVISION_NOTES.md`.

## Compile the manuscript

```bash
bash compile.sh
```

Open `main.tex` in a TeX Live / Overleaf project. Precompiled panel PDFs and `main.bbl` are included. The Makefile regenerates the 26 standalone panels from their LaTeX sources. Required packages include `standalone`, TikZ, PGFPlots, `newtxtext`, `newtxmath`, and `subcaption`. The supplied conference style, text size, and margins are preserved. `.latexmkrc` prefers the usual BibTeX executable and handles the authoring environment's renamed executable.

## Replay the numerical figure and table inputs

```bash
make replay
bash compile.sh
```

Replay requires Python, NumPy, pandas, and SciPy. It uses bundled project-relative inputs only: no model inference, new fitting, resampling, or data download. `scripts/build_assets.py` regenerates the original measured panels; `scripts/build_vector_evidence.py` typesets the already completed full-option analysis. The latter verifies the source Parquet checksum and its completion marker, then uses the bundled scalar-column CSV export to avoid a Parquet dependency during ordinary replay.

For independent inspection of the scalar export, `scripts/read_scalar_parquet.py` is a narrow optional decoder for this archive's Snappy-compressed scalar columns. It requires Thrift and libsnappy; unsupported encodings fail explicitly. Its 1,600-row, 23-column export was checked against the raw source and numeric footer extrema. Nested weight-list columns remain in the original Parquet and are not used in the new tables. A standard Parquet reader may be used independently instead.

## Contents

- `sections/`: revised main text and three contribution statements.
- `appendix/`: proofs, audit protocols, complete numerical comparisons, and legacy cases.
- `figures/standalone/`: exactly one TikZ picture and one panel per `.tex` file.
- `figures/pdf/`: compiled panels, assembled with `subfigure` in the manuscript.
- `data/original/`: original cross-domain CSV/NPZ outputs, preserved byte-for-byte.
- `data/v2/`, `data/v22/`, `data/table_v23_*.csv`: archived language-model evidence.
- `data/v22/vector_audit/`: complete-option source Parquet, scalar export, archived analysis code, completion marker, and arithmetic summaries.
- `data/figures/`: full-precision plotting inputs.
- `tables/`: numerical tables including primary and complete-option comparisons.
- `FIGURE_PROVENANCE.md`, `SOURCE_MAP.md`: source-to-panel/table mapping.
- `WRITING_REVISION_NOTES.md`, `REVIEW_TO_REVISION.md`: author-facing revision decisions.
- `qa/`: input-identity, arithmetic, conversion, clean-build, and page checks.

## Evidence levels

The primary LLM result uses reference-answer probabilities balanced over cyclic option positions, an endpoint design, matched MSE/MAE fitting, and common-question bootstrap intervals. The recovered complete-option analysis uses canonical order, five doses, vector squared-loss group fitting and TV-based single-peer selection. It is a separate response diagnostic with descriptive split consistency; it is not a new balanced-vector or matched-loss bootstrap experiment.

Forecasting plots are retained in Appendix D. Their raw values include non-simplex coefficients and are not repaired or promoted to exact convex replacement certificates. Missing pruning trajectories are not reconstructed. Each vision PCA view uses its own archived fitting-sample target/context basis. These provenance distinctions are retained while the main narrative focuses on the measured coverage relations.

This is an author working package, not an independently anonymity-certified submission bundle. Source archives and provenance records retain identifying paths and repository names; review them before releasing source under double-blind rules.
