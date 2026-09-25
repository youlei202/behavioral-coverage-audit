# Reproduce the CPU-only V2.1 corrected reanalysis

The analysis consumes only the completed V2 raw-score and generation-validation shards. It does not perform model inference or access model weights.

```bash
bash /work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/scripts/bootstrap_runbook_path.sh
mkdir -p /work/Lei
ln -sfn /work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS \
  /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS

tmux new -d -s pier_llm_reanalysis_v21 \
  "bash -lc 'cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS && \
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PIER_REANALYSIS_WORKERS=48 && \
  bash scripts/run_cpu_reanalysis_v21.sh --resume 2>&1 | tee -a logs/cpu_reanalysis_v21.log'"
```

The runner validates every stage marker and output hash before skipping completed work. Bootstrap shards are independently checksummed and are reused only when their metadata remains valid.
