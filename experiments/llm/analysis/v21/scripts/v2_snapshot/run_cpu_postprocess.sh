#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
cd "${PIER_PERSISTENT_ROOT}"

if [[ ! -f status/B200_INFERENCE_COMPLETE.json ]]; then
  echo "B200_INFERENCE_COMPLETE.json is absent; refusing post-processing." >&2
  exit 2
fi
python -m pier_llm.postprocess --root "${PIER_PERSISTENT_ROOT}" "$@"
echo "========================================================="
echo "EXPERIMENT COMPLETE"
echo "Result package:"
echo "${PIER_LOGICAL_ROOT}/artifacts/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_RESULTS.zip"
echo "========================================================="
