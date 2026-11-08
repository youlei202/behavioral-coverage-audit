# Recover and reacquire the balanced endpoint inputs

The user explicitly authorized targeted fresh scoring after the addendum was supplied. This extends the original cached-data-only execution scope; the downstream scientific comparison remains unchanged. Keep the old implementation, old data, and original completion markers intact.

The checked addendum is preserved under `addendum/`. The original 560 questions and full V2/V23 text designs were regenerated from the pinned dataset and verified against archived hashes. All eight exact checkpoint snapshots and rendered prompt files are verified against their original file identities. New score outputs are marked `fresh_endpoint_scoring`; they are not historical score bytes.

## Actual workspace and environments

Current recovery workspace is `outputs/xwvg_reference_selection/20261107_recovery/`. It contains `inputs/`, `prepared_v2/`, `fresh_endpoints/`, `cache/`, and `validation/`. Large model/cache/output files are ignored by Git.

- `/tmp/xwvg-recovery-venv`: pinned dataset retrieval and original CPU design construction. Exact versions are recorded under the run's `validation/` directory.
- `/tmp/xwvg-scoring-venv`: Torch 2.9.1 with CUDA 12.8, Transformers 4.57.3, BF16 model scoring. The actual full package lock is `validation/scoring_requirements.lock.txt`. Python is 3.12.3 rather than historical 3.11; regenerated prompt hashes match exactly.
- `/tmp/xwvg-venv`: CPU reference selection, tables, resampling and figures.

Hugging Face credentials are stored by its official login flow in the user's standard cache; no token belongs in this repository or command history. The browser login completed and the fixed Gemma revision's access check passed. Future expired authorization must use the official login flow again.

## Commands

Use absolute paths for deployment. From this repository root, the implemented entry points are:

```bash
/tmp/xwvg-recovery-venv/bin/python \
  experiments/llm/analysis/xwvg_reference_selection/recovery/prepare_inputs.py \
  --repo-root "$PWD" \
  --dataset-cache "$PWD/outputs/xwvg_reference_selection/20261107_recovery/cache/datasets" \
  --output-root "$PWD/outputs/xwvg_reference_selection/20261107_recovery/prepared_v2" \
  --v23-output "$PWD/outputs/xwvg_reference_selection/20261107_recovery/fresh_endpoints" \
  --offline --resume
```

`prepare_models.py download` accepts `--source-repo`, `--run-root`, `--cache-dir` and optional `--model-id`. It fetches exact pinned revisions and verifies all loaded files against archived SHA-256 values. Mistral's unused `consolidated.safetensors` alternative is omitted; the exact indexed Hugging Face shards are preserved. `prepare_models.py render` uses the same arguments and requires verified local tokenizer assets. Granite's system-message date is fixed at 2026-08-28 and its archived hash is enforced.

`score_endpoints.py` accepts `--source-repo`, `--run-root`, `--model-id`, `--model-path`, `--device`, `--batch-size`, `--stage pilot|score`, and `--resume`. It scores all rotations, including zero, with the original complete continuation likelihood. It performs no free-text generation. Each model's fixed pilot samples every semantic condition, rotation-zero/nonzero group, and minimum/median/maximum prompt lengths. Its scores are reused during full scoring. A full score run requires a completed pilot. Batch sizes may decrease after a real CUDA OOM; no truncation or quantization is performed.

The serial campaign wrapper runs one model at a time on the available B200:

```bash
/tmp/xwvg-scoring-venv/bin/python \
  experiments/llm/analysis/xwvg_reference_selection/recovery/run_campaign.py \
  --source-repo "$PWD" \
  --run-root "$PWD/outputs/xwvg_reference_selection/20261107_recovery/fresh_endpoints" \
  --questions "$PWD/outputs/xwvg_reference_selection/20261107_recovery/prepared_v2/data/mmlu_pro_selected_560.jsonl" \
  --analysis-output "$PWD/outputs/xwvg_reference_selection/20261107_fresh_analysis" \
  --analysis-python /tmp/xwvg-venv/bin/python \
  --phase all --device cuda:0
```

Phases `pilot`, `score`, and `finish` are also available. `finish` invokes `assemble_scores.py`, then the unchanged scientific reference-selection protocol with `--input-provenance status/measurement_provenance.json`. The assembler requires all eight complete score collections, checks every committed shard and its semantic mappings, and produces 295,568 rotation rows followed by 31,360 averaged vectors. It refuses incomplete models or missing rotations.

The completed empirical campaign used `--scorer reference`, the frozen original scoring function, on all eight models. The local pilots measured 42 prompts per model; their adaptive final batch sizes were four except Gemma, which used two. Subsequent shared-node scoring used batch size one as documented below. Cold initialization contributes to these short-pilot projections, so recorded full-run shard timings are the appropriate production-throughput evidence.

An optional candidate-position-only log-softmax implementation (`score_math.py`) was investigated in a separate exact GPU comparison. Seven models matched the original outputs exactly, but Granite had real probability differences around `1e-7`; all repeated original-score outputs matched the original pilot cache. The prespecified all-eight exact-equality gate therefore failed. `score_selected_endpoints.py` refuses that failed acceptance report and is **not used for the empirical campaign**. No tolerance relaxation or model-dependent backend substitution was made. The comparison evidence is under `validation/score_math_gpu/`; its warmed timings reset the CUDA allocator between calls and are not production-throughput estimates. `fresh_selected_endpoints/` contains only preparatory input links and metadata, not empirical scores.

## Evidence and status

Use `fresh_endpoints/status/campaign.json`, per-model pilot/complete records, and `fresh_endpoints/logs/` for execution status. `status/measurement_provenance.json` is written only after validated full assembly. The downstream analysis's `status/complete.json` is written only after all 160 settings, 10,240 candidates, saved-metric checks, 2,000 frozen-fit resamples and reporting have finished. A pilot or synthetic test completion never substitutes for those empirical completion markers.

The campaign subsequently moved to two nodes using `run_distributed.py`: the current local B200 and the explicitly authorized shared B200 reached as `leiyo@ssh.cloud.sdu.dk` on port 2064. Port 2253 is an SSH alias of the current local GPU and is not an extra device. The user authorized coexistence with the resident vLLM job on 2064. No other user's process is stopped. Actual node, GPU UUID, driver and source/runtime identities are saved in `status/distributed_campaign.json` and `status/distributed_workers/`.

Shared execution uses the original scorer at batch size one, a 70 GiB PyTorch allocator cap, and a 16 GiB free-memory reserve monitored every two seconds. Each eligible model passed an isolated fixed-42 pilot plus an outcome-blind maximum-workload probe. These additional pilot rows are validation evidence, not added to the primary cache. Seven models passed; Gemma exceeded the 70 GiB cap and remains assigned to the exclusive local GPU. Different batch groupings are recorded as an execution change; no cross-batch bitwise-equality claim is made. The rejected candidate-position-only backend remains unused.

The distributed command is recorded in `validation/distributed_plan.json`; its explicit slot configuration is `validation/distributed_slots.json`. While model scoring remains incomplete, resume with the same arguments to `run_distributed.py` and `--exclusive-run` only after the prior campaign's workers have stopped. After all eight score collections are complete, resume only `run_campaign.py --phase finish` (or the recorded analysis command) so completed scoring reports remain unchanged. The controller refuses conflicting jobs, duplicated GPU UUIDs, unverified shared-pilot proofs and stale ownership locks. `--plan-only` performs a local read-only preview. It reuses committed score shards and passed shared pilots, then invokes assembly and the complete analysis only after all eight model jobs succeed.

Read progress without loading scores or weights:

```bash
python3 experiments/llm/analysis/xwvg_reference_selection/recovery/progress.py \
  --run-root "$PWD/outputs/xwvg_reference_selection/20261107_recovery/fresh_endpoints"
```

The shared scoring Python environment was copied to `20261107_recovery/runtime/scoring-venv/` and verified to match the original scoring versions. Invoke its `bin/python` directly; copied console-script shebangs still refer to the original `/tmp` environment.

Archive discovery records search bounds and unavailable historical roots. The bundled locator reached its entry limit in unrelated installed environments; a subsequent filename search completed for the listed roots and found no matching PIER RESULTS ZIP. Google Drive's available document search excludes ZIP/Parquet, so its empty result is not evidence that cloud archives do not exist. No claim of deleted data is made.
