# Legacy Transformer-classifier validation

These are BERT-family sentiment-classifier experiments on SST-2, not modern-LLM evidence. `src/` retains the ISQED dose-split control and multi-context sources. `make tables` rebuilds all plotted subgroup differences and the 30 archived dose/control values from their exact frozen tables.

The source preserves fitting/evaluation separation, deterministic interventions, and the distinction between clone, architectural and parametric controls. Raw classifier prediction caches, exact model revisions and a pinned historical environment were not recovered. Table replay is verified; full classifier inference is documented but not rerun.
