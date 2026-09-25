#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_v23.sh
source "${SCRIPT_DIR}/common_v23.sh"
cd "${PIER_V23_PHYSICAL_ROOT}"

if [[ ! -f status/B200_STAGE2_COMPLETE.json ]]; then
  echo "B200_STAGE2_COMPLETE.json is absent; refusing CPU post-processing." >&2
  exit 2
fi

python -m pier_llm_v23.postprocess --root "${PIER_V23_PHYSICAL_ROOT}" "$@"

