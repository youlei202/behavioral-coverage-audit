# Modern-LLM behavioral coverage

The scientific progression is frozen model-score collection, trackwise residual evaluation, sibling-removal and collective-versus-single comparisons, balanced answer-label endpoint validation, and a complete-option response diagnostic. Internal V2/V2.1/V2.2/V2.3 directory names identify provenance, not different manuscript claims to be combined into one estimator.

`inference/v2/` contains the original scoring and initial analysis code. `analysis/v21/` retains the intermediate dependency; `analysis/v22/` implements trackwise estimation and vector diagnostics; `analysis/v23/` implements balanced endpoints, matched losses, common bootstrap and the final authorized numerical recovery. `analysis/vector_reference/` preserves the exact two analysis files bundled with the paper. `manifests/` contains original configurations, environment/repair records and model/dataset identities. No model weights or environments are copied.

Use `make tables` for the uploaded paper arithmetic. Use `BEHAVIORAL_COVERAGE_RAW_DATA=/path/to/parent make analysis` for the checked replay from archived split outcomes and bootstrap distributions. The latter compares six trackwise and five balanced-interface tables with the paper's frozen inputs, then checks an independent Parquet scalar export. It does not execute the model or refit/resample.

The canonical five-dose and balanced endpoint designs remain separate. Reference-answer scalar and complete-option vector responses remain separate. MSE-matched and MAE-matched comparisons remain separate. Auxiliary generation diagnostics are not answer-quality ground truth. The original vector checksum/completion marker is mandatory; a failed marker or changed checksum aborts replay.

The archived source snapshots contain original machine paths for provenance. For full runs, use `scripts/stage_experiments.py` and `docs/FULL_INFERENCE_REPRODUCTION.md` to create a new external workspace with recorded path mappings and immutable revision pins. Full inference and score-to-fit bootstrap reruns were not executed during packaging.
