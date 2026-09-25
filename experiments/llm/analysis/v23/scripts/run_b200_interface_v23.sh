#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_v23.sh
source "${SCRIPT_DIR}/common_v23.sh"
cd "${PIER_V23_PHYSICAL_ROOT}"

if [[ ! -f status/B200_READY_STAGE2.json ]]; then
  echo "B200_READY_STAGE2.json is absent; refusing formal inference." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this is not the required B200 host." >&2
  exit 2
fi
mapfile -t PIER_V23_GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#PIER_V23_GPU_NAMES[@]}" -ne 8 ]]; then
  echo "Expected exactly 8 visible GPUs, found ${#PIER_V23_GPU_NAMES[@]}." >&2
  exit 2
fi
for PIER_V23_GPU_NAME in "${PIER_V23_GPU_NAMES[@]}"; do
  if [[ "${PIER_V23_GPU_NAME^^}" != *"B200"* ]]; then
    echo "Every visible GPU must be an NVIDIA B200; found ${PIER_V23_GPU_NAME}." >&2
    exit 2
  fi
done

python -m pier_llm_v23.inference --root "${PIER_V23_PHYSICAL_ROOT}" "$@"

