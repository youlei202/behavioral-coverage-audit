# Auditing Behavioral Coverage in Model Ecosystems

How accurately can a fixed combination of peer models reproduce a target's responses, and which peers supply that coverage?

The Peer-Inexpressible Residual (PIER) measures the difference between a target response and a fixed convex combination of its peers on matched observations. DISCO fits the combination on one sample and evaluates its residual on another. Comparing the group with a peer selected on the fitting sample, and refitting after peer removal, separates collective coverage from dependence on one related model. This is a response-reconstruction audit, not an assessment of a model's intrinsic value or of ensemble task accuracy.

The paper reports three main findings:

1. In the eight-model balanced reference-answer audit, six targets benefit from group approximation under both stress families and matched MSE/MAE fitting. Qwen2.5 relies on its reasoning-tuned sibling; the reverse direction benefits from other peers. The separate complete-option diagnostic retains this directionality and Mistral's group advantage.
2. Matched stress can increase or decrease residuals. Canonical five-dose trajectories, balanced endpoints, and complete-option responses use distinct designs and are reported separately.
3. In vision, context changes peer support and where residuals concentrate. The traffic archive separately illustrates that response reconstruction and downstream utility measure different quantities.

After the one-time setup in [ENVIRONMENT.md](ENVIRONMENT.md), rebuild from frozen evidence:

```bash
make replay
make paper
make verify
```

`make replay` already builds the manuscript; `make paper` also demonstrates a separate clean LaTeX/BibTeX rebuild. Outputs are `paper/main.pdf`, 26 standalone panel PDFs, generated CSVs, and ten manuscript tables. The scientific main text occupies pages 1–9, with references starting on page 10. Replay and verification need no model, GPU, network, or old experiment directory.

Inspect the complete figure/table mapping in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For the deeper replay from cached LLM split outcomes and bootstrap distributions:

```bash
export BEHAVIORAL_COVERAGE_RAW_DATA=/path/to/parent/of/archived/PIER/directories
make analysis
```

Every external input is checked against `manifests/external_analysis_inputs.tsv` before it is copied into a disposable workspace. This regenerates 11 upstream analysis tables and checks the vector scalar export directly against its Parquet. It does not refit models or resample bootstrap replicates. Vision residual localization and stored geometry, traffic normalizations, and legacy BERT table transformations are included in ordinary replay.

[Full inference instructions](docs/FULL_INFERENCE_REPRODUCTION.md) preserve the checkpoints, dataset revision, seeds, entry points, and separate environments. These inference commands were **not run during packaging**. Legacy dataset/checkpoint identities that were not archived remain unresolved; missing raw predictions and high-dimensional vision fitting traces are listed in [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

This repository verifies frozen-output replay. It does not claim that full inference was rerun, that canonical and balanced protocols are interchangeable, that archived traffic coefficients are exact simplex certificates, or that the stored PCA views share a coordinate system. `REPRO_STATUS.md` records the actual verification scope. No blanket license or author identity is inferred from the upload; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
