# Reproduce PIER Modern-LLM Ecosystem Gold-Mining V2

Persistent data live at `/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2`. The runbook-visible path `/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2` is a compatibility symlink and must be bootstrapped once on each new job host before any tmux session.

## Stage A — 64-vCPU CPU job

```bash
bash /work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/scripts/bootstrap_runbook_path.sh
tmux new -d -s pier_llm_prestage_v2 \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2 && \
   bash scripts/prestage_cpu.sh \
   2>&1 | tee -a logs/prestage_cpu.log"
```

Inspect with `tmux attach -t pier_llm_prestage_v2` or `tail -n 100 /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/logs/prestage_cpu.log`.

Do not open B200 until `status/B200_READY.json` exists and contains `"ready": true`.

## Stage B — 8×NVIDIA B200 job

Bootstrap the logical path first; this is a filesystem compatibility step, not formal model inference. Then make the following tmux command the first formal action:

```bash
bash /work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/scripts/bootstrap_runbook_path.sh
tmux new -d -s pier_llm_b200_v2 \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2 && \
   bash scripts/run_b200_inference.sh --resume \
   2>&1 | tee -a logs/b200_inference.log"
```

The launcher validates every completed shard and recomputes only missing/corrupt shards. Stop B200 immediately after `status/B200_INFERENCE_COMPLETE.json` is created.

## Stage C — 64-vCPU CPU job

```bash
bash /work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/scripts/bootstrap_runbook_path.sh
tmux new -d -s pier_llm_post_v2 \
  "cd /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2 && \
   bash scripts/run_cpu_postprocess.sh --resume \
   2>&1 | tee -a logs/cpu_postprocess.log"
```

The final package is `/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/artifacts/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_RESULTS.zip`.
