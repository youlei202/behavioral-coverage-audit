# Page-top figure layout revision

Base: `ICLR2027_Behavioral_Coverage_NinePage_Source.zip`.

This revision changes float placement only. Scientific prose, equations, figure captions, numerical inputs, standalone panel sources, panel scale, bibliography, and conference style are preserved.

## Final pagination

- Scientific main text: pages 1–9, with the final Discussion paragraph at the foot of page 9.
- References: begin on page 10.
- Total: 22 pages.
- Figures 1–4: at the top of pages 3, 5, 7, and 8.
- Figures 5–8: at the top of pages 17, 20, 21, and 22.

## Layout changes

- All eight composite `figure` environments use `[!t]`; their panels remain assembled with `subfigure`.
- The stress, vision, and final appendix figure declarations are queued before their section prose so that they can occupy the page-top float area.
- The float-page top glue is zero, including on the full-page PCA figure.
- A redundant appendix float barrier was removed to avoid a mostly empty additional page.
- The complete-option table is bottom-placed on page 17 so that the interface figure starts that page, with no table above it.
- The generators retain the same placement options when the numerical assets are replayed.
- Main-text fonts, line spacing, margins, and caption styling are unchanged.

## Verification

All eight figure groups begin at approximately 81.862 PDF points from the physical page top, just below the unchanged header area. No body text or table precedes a figure on its page. Figure position is checked from the embedded PDF form transformations, not just the `[!t]` declaration.

The 146 bundled data files, 26 standalone panel sources, and 26 compiled panels are byte-identical to the nine-page base. All scientific paragraphs, equations, and captions match after removing placement syntax and float-declaration relocation. Conference style and bibliography hashes also match.

A separate clean copy replayed the existing numerical assets and rebuilt every panel and the manuscript. All 22 page renders matched the inspected PDF at 108 dpi. No overfull boxes, undefined citations, or undefined cross-references were found.

See `qa/top_float_layout.json` and `qa/top_float_clean_build.json` for machine-readable checks.
