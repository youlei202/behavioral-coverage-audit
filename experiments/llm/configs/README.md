# Configuration index

Exact stage configurations are stored alongside their source trees to preserve import/build layouts:

- `../inference/v2/configs/`: experiment design, model roster, immutable resolved model revisions, stopwords.
- `../analysis/v21/configs/`: intermediate analysis design used by the later comparisons.
- `../analysis/v22/configs/`: trackwise design, ten fixed splits and canonical vector audit settings.
- `../analysis/v23/configs/`: balanced endpoint design, generation diagnostics and final model revisions.

These are byte-preserved upstream files. Portable run-specific configurations are made only in a new external workspace by `scripts/stage_experiments.py`; that command records every path/revision adaptation.
