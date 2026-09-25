# Reproduce PIER Modern-LLM Ecosystem V2.3

The workflow has three hard-gated stages. All long stages run in `tmux`; model/data downloads and code edits are forbidden on the B200 host.

## Stage A — 64-core CPU prestage

```bash
tmux new -d -s pier_llm_v23_prestage \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION && \
   bash scripts/prestage_cpu_v23.sh \
   2>&1 | tee -a logs/prestage_cpu_v23.log"
```

Inspect with `tail -n 120 logs/prestage_cpu_v23.log`. Continue only when `status/B200_READY_STAGE2.json` exists and has `"ready": true`.

## Stage B — exactly 8×NVIDIA B200

The first meaningful B200 action is:

```bash
tmux new -d -s pier_llm_v23_b200 \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION && \
   bash scripts/run_b200_interface_v23.sh --resume \
   2>&1 | tee -a logs/b200_interface_v23.log"
```

The manifest-derived per-model counts are 33,026 new cyclic score prompts and 3,918 long-generation prompts. Rotation 0 contributes another 3,920 validated V2 rows per model. Do not edit code, download, install, or analyze on B200. Shut the host down only after `status/B200_STAGE2_COMPLETE.json` exists and is complete.

## Stage C — 64-core CPU analysis and package

```bash
tmux new -d -s pier_llm_v23_post \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION && \
   bash scripts/run_cpu_postprocess_v23.sh --resume \
   2>&1 | tee -a logs/cpu_postprocess_v23.log"
```

The final ZIP is `artifacts/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION_RESULTS.zip`. Raw candidate-score shards remain on persistent storage and are represented in the ZIP by exact paths and SHA256 values.

## Authorized V2.3.1 Stage-C recovery

The initial Stage-C run completed 781 of 800 common-multiplicity bootstrap shards and then stopped on 19 raw-endpoint Gemma convexity shards whose otherwise-optimal HiGHS solutions had machine-precision simplex violations. The only authorized scientific-code change is the deterministic sorting-based Euclidean projection of qualifying solver outputs onto the probability simplex, with the unmodified solver, objective, inputs, seeds, manifests, estimators, and claim gates. Existing valid shards are preserved by `--resume`.

Stage-C post-processing was executed under CPython 3.11.15 rather than the prestaged CPython 3.11.13 because the original interpreter was unavailable on the replacement CPU host. Python remained within the same 3.11 ABI line and all locked package versions, frozen scientific inputs, manifests, Stage-B outputs, and hashes were unchanged, except for the explicitly authorized V2.3.1 numerical-feasibility patch. The runtime environment is therefore not claimed to be bitwise identical.

The formal recovery command is:

```bash
tmux new -d -s pier_llm_v231_recover \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION && \
   bash scripts/run_cpu_postprocess_v23.sh --resume \
   2>&1 | tee -a logs/cpu_postprocess_v231_recovery.log"
```

Required recovery audit artifacts are:

- `status/V2_3_1_AUTHORIZATION.json`
- `manifests/V2_3_1_AUTHORIZED_PATCH.diff`
- `outputs/validation/numerical_feasibility_repairs.parquet`
- `status/numerical_feasibility_repair_summary.json`
- `outputs/validation/numerical_feasibility_repair_sensitivity.parquet`

The repair-free-only sensitivity calculation is diagnostic. It excludes all global bootstrap replicate indices for which any touched solution required repair and never replaces the full 1,000-replicate analysis.
