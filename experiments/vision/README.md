# Vision evidence

`src/` contains the selected ISQED adversarial-dose, joint-context, separated-context and geometry experiments at the recorded upstream commit. Shared library code is under `../shared/isqed/`.

`make tables` derives all six adversarial curves, the seven-model context comparison, six archived PCA views, 401-point cumulative residual curves and top-k overlap for k=1 through 40 from the exact frozen CSV/NPZ inputs. Fit/evaluation context ordering and the reference-class probability interface are preserved. Residual arrays retain 400 evaluation positions per context.

The stored geometry has separately fitted bases for each target/context. Replay uses exact archived coordinates, hulls, mixture weights and fit IDs; it does not refit PCA. High-dimensional response traces needed to re-estimate the bases are absent. Full data/checkpoint entry points and remaining provenance gaps are in `../../docs/FULL_INFERENCE_REPRODUCTION.md`.
