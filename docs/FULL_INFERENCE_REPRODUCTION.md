# Full model/data inference reproduction

**These commands are documented but were not re-run as part of repository packaging.**

The verified result is frozen-output replay and the separately described cached-analysis replay. Full inference needs the original checkpoints/data, their access permissions, the archived environments, and suitable hardware. Missing historical identities below are unresolved, not guessed.

## Create a new external experiment workspace

Run from this repository after activating the relevant environment. `stage_experiments.py` copies source/configuration only into a **new** directory. It records path adaptations and pins the eight recorded model revisions and dataset revision; it disables fallback to a different model roster. It never modifies an existing run, launches inference, or copies model weights.

```bash
export BCA_REPO="$PWD"
export BCA_RUN_ROOT=/path/to/a/new/behavioral-coverage-run
export BEHAVIORAL_COVERAGE_RAW_DATA=/path/to/parent/of/archived/PIER/directories
export BEHAVIORAL_COVERAGE_MODEL_CACHE=/path/to/the/original/hf_cache
python scripts/stage_experiments.py --run-root "$BCA_RUN_ROOT" \
  --raw-root "$BEHAVIORAL_COVERAGE_RAW_DATA" \
  --model-cache "$BEHAVIORAL_COVERAGE_MODEL_CACHE"
```

`--raw-root` copies only the 44 hash-checked non-model design/manifest files in `external_design_inputs.tsv` (about 1.55 GB, mostly rendered prompt text). It does not copy score shards or model caches. `--model-cache` creates cache references in the new workspace. Omit `--raw-root` for a fresh dataset/prompt preparation from the recorded immutable dataset revision; this path requires downloads and remains unexecuted here. Model cache completeness and access must be checked during CPU prestaging, before starting any GPU stage.

The staged directories retain historical V2/V2.1/V2.2/V2.3 names solely for source compatibility. The repository's snapshots remain byte-preserved. `PATH_MIGRATION.json` records every path or revision-adapter change. Staging into an existing directory fails, including an incomplete earlier staging attempt. Never point it at an upstream frozen directory.

## Modern LLM protocol and environment

Use CPython 3.11 and the historical lock at `experiments/llm/manifests/v2/env/requirements.lock.txt` for model scoring. Torch is `2.9.1+cu128`, Transformers `4.57.3`, and the formal GPU stage expects **8 NVIDIA B200** devices. The original CPU stages were designed for a 64-core host, with 48 single-threaded bootstrap workers. V2.3 CPU recovery used Python 3.11.15 instead of the prestaged 3.11.13, as recorded in its environment/recovery manifests. Do not substitute the Level-A Python environment for the original inference environment.

Create/activate the historical environment in external work storage. An offline installation can use the original V2 wheelhouse with `pip install --no-index --find-links /path/to/wheelhouse -r "$BCA_REPO/experiments/llm/manifests/v2/env/requirements.lock.txt"`. Retain the CUDA-specific Torch build and recorded dependencies. The dependency lock and source files are packaged; the multi-gigabyte wheelhouse is external.

The dataset is `TIGER-Lab/MMLU-Pro`, `test` split, immutable revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`. The fixed sample has 560 questions, 40 from each of 14 subjects. `sample_seed=20260828`, `generation_seed=20260829`, and split seeds are 20260828 through 20260837. Each split assigns 280 questions to fitting and 280 to evaluation, with all related doses/tracks/rotations grouped by question. Exact selected IDs and their checksums are in the saved dataset manifest and external design files.

Scoring uses the literal candidate-label prompt in `pier_llm/data_prep.py`, short-label conditional sequence log likelihoods without length normalization, and normalized option probabilities. Content deletion uses literal blanks at fractions 0, 0.1, 0.2, 0.3, 0.4. Irrelevant context uses word budgets 0, 64, 128, 256, 512. There are three realizations per nonzero dose. The canonical five-dose, cyclic balanced endpoint, complete-option vector, and auxiliary generation designs remain distinct. The original V2 run has 15,680 score prompts and 420 generation prompts per model; V2.3 adds 33,026 cyclic-score prompts and 3,918 long-generation prompts per model, reusing 3,920 validated canonical rows. V2.3 generates at most 512 tokens in its auxiliary generation diagnostic, in bfloat16.

All eight checkpoint IDs and exact revisions are in `data/frozen/llm/v2/table1_ecosystem_manifest.csv`, `experiments/llm/inference/v2/configs/resolved_models.json`, and `docs/MODEL_REVISIONS.md`. Their compatibility manifests record tokenizer classes, prompt hashes, model adapters, weight file hashes, and known licensing metadata. Do not resolve floating model revisions for an exact reproduction.

## Model-score collection and analyses

Define the staged roots after environment activation. If you previously sourced the Level-A environment, remove its empty GPU mask before the GPU stage.

```bash
export BCA_V2="$BCA_RUN_ROOT/PIER_LLM_ECOSYSTEM_GOLDMINE_V2"
export BCA_V21="$BCA_RUN_ROOT/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS"
export BCA_V22="$BCA_RUN_ROOT/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS"
export BCA_V23="$BCA_RUN_ROOT/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# CPU prestage: dataset/prompt checks, model compatibility, pinned input identities.
cd "$BCA_V2"
PYTHONPATH="$BCA_V2/src" python -m pier_llm.prestage --root "$BCA_V2"
# Inspect status/B200_READY.json; continue only when its recorded gates pass.

# Long GPU stage, on the required host. This command is documentation, not a packaging action.
unset CUDA_VISIBLE_DEVICES
tmux new-session -d -s bca_scores \
  "cd '$BCA_V2' && PYTHONPATH='$BCA_V2/src' python -m pier_llm.inference --root '$BCA_V2' --resume > logs/inference.log 2>&1"
# After B200_INFERENCE_COMPLETE.json is successful, run the CPU postprocess:
PYTHONPATH="$BCA_V2/src" python -m pier_llm.postprocess --root "$BCA_V2" --resume

# Historical analysis dependency, then the trackwise estimand evaluation.
cd "$BCA_V21"
PYTHONPATH="$BCA_V21/src" python -m pier_llm_reanalysis_v21.pipeline --resume --workers 48
cd "$BCA_V22"
PYTHONPATH="$BCA_V22/src" python -m pier_llm_reanalysis_v22.pipeline --resume --workers 48

# Balanced answer-label endpoint preparation, scoring, and final CPU analysis.
cd "$BCA_V23"
PYTHONPATH="$BCA_V23/src" python -m pier_llm_v23.prestage --root "$BCA_V23"
# Inspect status/B200_READY_STAGE2.json before running the GPU stage.
tmux new-session -d -s bca_balanced_scores \
  "cd '$BCA_V23' && PYTHONPATH='$BCA_V23/src' python -m pier_llm_v23.inference --root '$BCA_V23' --resume > logs/inference.log 2>&1"
# After B200_STAGE2_COMPLETE.json is successful:
PYTHONPATH="$BCA_V23/src" python -m pier_llm_v23.postprocess --root "$BCA_V23" --resume
```

These commands are sequential stages, not a shell script to run without checking completion markers. The archived shell wrappers also retain their original host/quota gates. Logs, raw score shards, metadata, failures, solver diagnostics, and final completion records stay in the fresh run. Resume uses that run's own hashes; it is never used to mutate the old frozen run.

V2.2's complete-option stage writes `outputs/analysis/trackwise_vector_results.parquet` with 1,600 rows and the checksum-bearing stage-11 marker. V2.3's final report tables contain 16 target/family endpoint rows, eight directed sibling/family rows, 16 target/family group-comparison rows, and 24 auxiliary generation condition rows. The exact expected schemas/counts for paper inputs are in `manifests/input_schemas.json`. A future inference campaign need not yield the same serialized Parquet bytes across toolchains; compare protocol, numerical outputs and diagnostics without overwriting the archived evidence.

For the already available cached split/bootstrap artifacts, prefer the verified `make analysis` route. To rerun score-to-fit analyses without invoking models, first produce a separate run containing checksum-verified cached raw scores, selected questions and prompt manifests, then invoke the corresponding CPU analysis entry point above. Original code contains the estimators, solvers, cluster bootstrap, and configuration gates. This complete CPU refit/bootstrap campaign was not executed as part of packaging; the 11-table cached-summary replay was.

## Vision

Use the staged `ISQED` directory and an isolated legacy environment with Torch, TorchVision, NumPy, pandas, SciPy, CVXPY and the dependencies imported by the copied modules. Historical package/checkpoint revisions are incomplete; exact binary-level rerun equivalence cannot be promised. Preserve the ImageFolder directory/class ordering and the original natural and cue-conflict/stylized image sets. Supply the robust ResNet-50 checkpoint explicitly so a missing model is not silently treated as executed coverage.

```bash
cd "$BCA_RUN_ROOT/ISQED"
export PYTHONPATH="$PWD"
python -m experiments.exp10_imagenet_adv_pier \
  --data_root /path/to/imagenet/val --max_samples 500 \
  --robust_ckpt /path/to/resnet50_l2_eps3.ckpt
python -m experiments.exp11_shape_bias_texture \
  --natural_root /path/to/imagenet/val --shape_root /path/to/shape-images \
  --max_samples_per_context 800 --fit_fraction 0.5
python -m experiments.exp12_shape_bias_texture_seperated \
  --natural_root /path/to/imagenet/val --shape_root /path/to/shape-images \
  --max_samples_per_context 800 --fit_fraction 0.5
python -m experiments.exp13_image_model_geometry \
  --natural_root /path/to/imagenet/val --shape_root /path/to/shape-images \
  --max_samples_per_context 800 --fit_fraction 0.5
```

These are source-derived commands with the archived defaults, not a recovered historical shell transcript. The code fixes its base randomness at seed 0 and constructs deterministic per-input interventions with `experiments/utils.py` (base seed 2026). Contexts use matched fit/evaluation ordering; the stored observation interface is reference-class probability. Expected outputs are `results/tables/exp10_imagenet_adv_pier.csv`, `exp11_shape_texture_pier.csv`, `exp12_shape_texture_pier_separated.csv`, residual NPZ files under `results/artifacts/exp11/` and `exp12/`, and six geometry NPZ files under `exp13/`. Every geometry context/target has its own fitted PCA basis. GPU inference is needed for practical full vision runs; the original exact hardware allocation was not recovered.

## Traffic

The source uses UTD19 city detector data, seed 0, a 12-step prediction horizon, and the recorded HistGradientBoosting model path. Preserve original temporal train/validation/test splits and filtering. The archived data source/training cache and exact historical invocation were not recovered; the following command uses source defaults and is not a claim of an exact recreation of the 31-city run:

```bash
cd "$BCA_RUN_ROOT/ISQED"
PYTHONPATH="$PWD" python -m experiments.exp14_multicity_tabular_pier \
  --data_csv /path/to/utd19.csv --data_dir /path/to/traffic-work \
  --horizon_steps 12 --model_type histgb
```

Inspect the copied parser for all row caps, lags, filters and model settings. Expected outputs are `results/tables/exp14_multicity_tabular_summary.csv` and its peer-weight table. CPU resources suffice. Missing pruning/tail trajectories cannot be regenerated from the summaries and must not be substituted with interpolated curves.

## Legacy sentiment classifiers

These are SST-2 classifier experiments, not modern-LLM tests. The source uses the GLUE SST-2 validation split and TextAttack BERT, DistilBERT, RoBERTa, ALBERT and XLNet SST-2 checkpoints. Their IDs are in the copied experiment source; immutable checkpoint and dataset revisions were not recorded. Base randomness is seed 0, with deterministic context interventions as above. Use a separate legacy environment; the source's unpinned dependency file is preserved.

```bash
cd "$BCA_RUN_ROOT/ISQED"
PYTHONPATH="$PWD" python -m experiments.exp4_bert_audit
PYTHONPATH="$PWD" python -m experiments.exp6_bert_multicontext \
  --max_samples 1000 --min_context_size 80
```

The outputs are `results/tables/exp4_bert_disco_dosesplit.csv` and `exp6_bert_multicontext_pier.csv`. Missing original prediction caches and model revisions limit exact full reproduction. None of these inference commands was run during packaging; only the exact archived table transformations were replayed.
