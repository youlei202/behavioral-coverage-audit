# Reproduce PIER Modern-LLM Ecosystem V2.2

This package performs CPU-only reanalysis of immutable V2 and V2.1 inputs. It does not run model inference.

From the project root:

```bash
bash scripts/run_cpu_trackwise_v22.sh --resume
```

The formal run is launched in tmux session `pier_llm_trackwise_v22`; it uses 48 workers by default, fixes numerical-library thread counts to one, clears `CUDA_VISIBLE_DEVICES`, validates every input shard, and resumes only checksum-valid stages and bootstrap shards.

The primary estimand is the equal-dose, equal-track mean of individual absolute residuals. `PIER of the track-mean response` is retained only as a diagnostic.
