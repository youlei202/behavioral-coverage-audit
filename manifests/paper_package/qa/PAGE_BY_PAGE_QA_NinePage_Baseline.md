# Page-by-page QA — nine-page main text

The final PDF has 22 pages. Scientific main text occupies pages 1–9; References starts on page 10. Conference style, font sizes, margins, spacing settings and the 26 original standalone figures were not changed.

## Reviewed pages

| Page | Inspection |
|---|---|
| 1 | Abstract and introduction; typography and bottom boundary unchanged. |
| 2 | Contributions and formal response definition; no clipping. |
| 3 | Conceptual figure and definitions; no new figure layout changes. |
| 4 | Exact controls and start of LLM results; references resolve. |
| 5 | Main LLM figure and expanded quantitative Phi comparison; table/figure boundary clear. |
| 6 | Complete-option results; main Tables 2 and 3 both readable, with unchanged numerical values. |
| 7 | Coefficient-transfer continuation, stress curves and endpoint inference; labels clear. |
| 8 | Vision main figure and added support changes for the two other archived targets. |
| 9 | Residual localization, utility comparison, related work and Discussion. Final scientific line at ruler line 485; no bibliography used as main-text filler. |
| 10 | References begins at top of page. |
| 11 | Remaining references and reproducibility/AI statements; appendix not mixed into main text. |
| 12 | Appendix A projection/consistency proofs. |
| 13 | Proof continuation and model-roster table. |
| 14 | Candidate response, fit/evaluation design and bootstrap protocol. |
| 15 | Endpoint/all-target tables; compact and legible. |
| 16 | Sibling-effects table and full-option protocol; moved main table correctly referenced as Table 3. |
| 17 | Full-option all-target results and interface figures. |
| 18 | Generation readouts and cross-domain protocol. |
| 19 | Exact geometry/support table and forecasting limitations. |
| 20 | All six original PCA panels; source coordinates unchanged. |
| 21 | Archived traffic figures and additional theory. |
| 22 | Legacy sentiment diagnostics and final figure; no clipping. |

## Reproduction

A clean copy ran `make replay` and rebuilt all 26 standalone panels plus the manuscript. Its 22 rendered pages are pixel-identical to the reviewed version at 1.25x rendering scale. No undefined citations/references, duplicate labels or overfull boxes remain. Input data and standalone source identities are unchanged.
