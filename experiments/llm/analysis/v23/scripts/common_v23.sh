#!/usr/bin/env bash
set -Eeuo pipefail

PIER_V23_LOGICAL_ROOT="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION"
PIER_V23_PHYSICAL_ROOT="/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_3_INTERFACE_VALIDATION"

if [[ ! -d "${PIER_V23_LOGICAL_ROOT}" ]]; then
  echo "Required persistent project root is unavailable: ${PIER_V23_LOGICAL_ROOT}" >&2
  exit 2
fi
if [[ "$(readlink -f "${PIER_V23_LOGICAL_ROOT}")" != "${PIER_V23_PHYSICAL_ROOT}" ]]; then
  echo "Logical V2.3 root does not resolve to the required persistent physical root." >&2
  exit 2
fi

export PIER_V23_ROOT="${PIER_V23_PHYSICAL_ROOT}"
export PYTHONPATH="${PIER_V23_PHYSICAL_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${PIER_V23_PHYSICAL_ROOT}/hf_cache"
export HF_HUB_CACHE="${PIER_V23_PHYSICAL_ROOT}/hf_cache/hub"
export HF_DATASETS_CACHE="${PIER_V23_PHYSICAL_ROOT}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${PIER_V23_PHYSICAL_ROOT}/hf_cache/hub"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export PYTHONHASHSEED="0"
export OMP_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

PIER_V23_VENV="${PIER_V23_PHYSICAL_ROOT}/env/venv"
if [[ ! -x "${PIER_V23_VENV}/bin/python" ]]; then
  echo "Pinned V2.3 Python environment is unavailable: ${PIER_V23_VENV}" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${PIER_V23_VENV}/bin/activate"

