# Manuscript source

This is the uploaded nine-page scientific manuscript, using the supplied cleaned bibliography. The 26 panels each have one standalone source and one TikZ picture; composite figures remain assembled with `subfigure` in the manuscript. Styles are under `styles/` and are supplied through the build's TeX/BibTeX search paths.

From the repository root run `make replay`, or from this directory run `make replay`. After panel generation, `make paper` performs a clean pdfLaTeX/BibTeX build and writes `main.pdf`. `make verify` checks pagination, all citation keys, expected tables, included figures and numerical identities. A direct TeX invocation without the supplied style search paths is not the supported build interface.

`tables/exact_controls.tex` and `tables/models.tex` replace two previously inline table blocks; they are generated from the archived JSON/CSV without changing display content. `make tables` regenerates all numerical tables and panel sources. Compiled PDFs are build products, not frozen inputs. The package's original build instructions and historical QA remain in `../manifests/paper_package/` for provenance.
