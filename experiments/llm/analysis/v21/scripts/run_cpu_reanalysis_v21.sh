#!/usr/bin/env bash
set -Eeuo pipefail

PIER_V21_ROOT="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS"
PIER_V2_ROOT="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2"
PIER_V21_PYTHON="/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/env/venv/bin/python"
PIER_V21_WORKERS="${PIER_REANALYSIS_WORKERS:-48}"
PIER_V21_RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      PIER_V21_RESUME=1
      shift
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "--workers requires an integer" >&2; exit 2; }
      PIER_V21_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ "${PIER_V21_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Worker count must be a positive integer" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PIER_REANALYSIS_WORKERS="${PIER_V21_WORKERS}"
export PYTHONPATH="${PIER_V21_ROOT}/src:${PIER_V21_ROOT}/src/v2_snapshot:${PIER_V2_ROOT}/src"

PIER_V21_ARGS=(--workers "${PIER_V21_WORKERS}")
if [[ "${PIER_V21_RESUME}" -eq 1 ]]; then
  PIER_V21_ARGS+=(--resume)
fi

cd "${PIER_V21_ROOT}"
exec "${PIER_V21_PYTHON}" -m pier_llm_reanalysis_v21.pipeline "${PIER_V21_ARGS[@]}"
