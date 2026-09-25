#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${PIER_PERSISTENT_ROOT}"

if [[ ! -f status/B200_READY.json ]]; then
  echo "B200_READY.json is absent; refusing formal inference." >&2
  exit 2
fi
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
  echo "Expected exactly 8 visible GPUs, found ${GPU_COUNT}." >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eiv 'B200' >/dev/null; then
  echo "At least one visible GPU is not an NVIDIA B200." >&2
  exit 2
fi

python -m pier_llm.inference --root "${PIER_PERSISTENT_ROOT}" "$@"
echo "================ B200 INFERENCE COMPLETE ================"
echo "Raw inference is complete and validated. Shut down B200."
echo "Return to CPU and launch the post-processing command in REPRODUCE.md."
echo "=========================================================="
