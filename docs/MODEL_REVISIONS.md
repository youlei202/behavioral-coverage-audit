# Frozen modern-LLM checkpoint revisions

These identifiers, immutable revisions and parameter counts are copied from the authoritative frozen ecosystem manifest, not resolved against a current online model catalog. All eight participate in the modern-LLM audit. Historical attempted/fallback checkpoints in compatibility records are not additional executed ecosystem members.

| Checkpoint | Immutable revision | Recorded parameter count | Exact sibling group |
| --- | --- | ---: | --- |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 7,615,616,512 | qwen2.5-7b |
| `TIGER-Lab/General-Reasoner-Qwen2.5-7B` | `e05e2d49ca11fdf74825cb1547dc50a4e441f908` | 7,615,616,512 | qwen2.5-7b |
| `microsoft/phi-4` | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | 14,659,507,200 | phi-4-14b |
| `microsoft/Phi-4-reasoning-plus` | `69baf8528e1bcf05f475034d9e5dd32875ed125f` | 14,659,507,200 | phi-4-14b |
| `mistralai/Mistral-Nemo-Instruct-2407` | `04d8a90549d23fc6bd7f642064003592df51e9b3` | 12,247,782,400 | none recorded |
| `allenai/OLMo-2-1124-13B-Instruct` | `3a5c85baefbb1896a54d56fe2e76c0395627ddf4` | 13,716,198,400 | none recorded |
| `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | 8,170,864,640 | none recorded |
| `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | 12,187,325,040 | none recorded |

Source: `data/frozen/llm/v2/table1_ecosystem_manifest.csv`. Per-checkpoint weight-file hashes, rendering/tokenizer checks and compatibility details are in `experiments/llm/manifests/v2/data/manifests/models/`. Dataset: `TIGER-Lab/MMLU-Pro`, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`, test split. See `FULL_INFERENCE_REPRODUCTION.md` for the separate canonical, balanced and generation protocols. Full inference was not run during packaging.
