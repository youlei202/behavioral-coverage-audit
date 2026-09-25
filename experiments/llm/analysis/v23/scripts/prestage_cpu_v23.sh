#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_v23.sh
source "${SCRIPT_DIR}/common_v23.sh"
cd "${PIER_V23_PHYSICAL_ROOT}"

python -m pier_llm_v23.prestage --root "${PIER_V23_PHYSICAL_ROOT}"

