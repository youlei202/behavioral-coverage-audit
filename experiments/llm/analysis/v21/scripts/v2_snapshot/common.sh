#!/usr/bin/env bash
set -Eeuo pipefail

PIER_LOGICAL_ROOT="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2"
PIER_PERSISTENT_ROOT="/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2"

if [[ ! -d "${PIER_PERSISTENT_ROOT}" ]]; then
  echo "Persistent project root is unavailable: ${PIER_PERSISTENT_ROOT}" >&2
  exit 2
fi
if [[ ! -e "${PIER_LOGICAL_ROOT}" ]]; then
  echo "Required logical path is absent: ${PIER_LOGICAL_ROOT}" >&2
  echo "Run scripts/bootstrap_runbook_path.sh before launching the tmux session." >&2
  exit 2
fi
if [[ "$(readlink -f "${PIER_LOGICAL_ROOT}")" != "$(readlink -f "${PIER_PERSISTENT_ROOT}")" ]]; then
  echo "Logical project path does not resolve to the persistent root." >&2
  exit 2
fi

export PIER_PROJECT_ROOT="${PIER_PERSISTENT_ROOT}"
export HF_HOME="${PIER_PERSISTENT_ROOT}/hf_cache"
export HF_HUB_CACHE="${PIER_PERSISTENT_ROOT}/hf_cache/hub"
export HF_DATASETS_CACHE="${PIER_PERSISTENT_ROOT}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${PIER_PERSISTENT_ROOT}/hf_cache/hub"
export XDG_CACHE_HOME="${PIER_PERSISTENT_ROOT}/env/cache"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export PYTHONHASHSEED="0"

PIER_VENV="${PIER_PERSISTENT_ROOT}/env/venv"
if [[ -x "${PIER_VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${PIER_VENV}/bin/activate"
fi
