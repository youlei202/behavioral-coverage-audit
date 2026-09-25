#!/usr/bin/env bash
set -Eeuo pipefail

PIER_V22_ROOT="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS"
PIER_V22_PYTHON="${PIER_V22_ROOT}/env/venv/bin/python"
PIER_V22_WORKERS="${PIER_REANALYSIS_WORKERS:-48}"
PIER_V22_RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      PIER_V22_RESUME=1
      shift
      ;;
    --workers)
      [[ $# -ge 2 ]] || { echo "--workers requires an integer" >&2; exit 2; }
      PIER_V22_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ "${PIER_V22_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Worker count must be a positive integer" >&2
  exit 2
}
[[ "${PIER_V22_WORKERS}" -le 48 ]] || {
  echo "Worker count must not exceed 48" >&2
  exit 2
}
[[ -x "${PIER_V22_PYTHON}" ]] || {
  echo "Pinned V2.2 Python environment is unavailable: ${PIER_V22_PYTHON}" >&2
  exit 1
}
realpath -e /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2 >/dev/null
realpath -e /work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_1_REANALYSIS >/dev/null

PIER_V22_CPU_MAX="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
if [[ "${PIER_V22_CPU_MAX}" != "6400000 100000" ]]; then
  echo "Expected a 64-CPU cgroup quota; observed: ${PIER_V22_CPU_MAX:-unavailable}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PIER_REANALYSIS_WORKERS="${PIER_V22_WORKERS}"
export PYTHONPATH="${PIER_V22_ROOT}/src"

PIER_V22_ARGS=(--workers "${PIER_V22_WORKERS}")
if [[ "${PIER_V22_RESUME}" -eq 1 ]]; then
  PIER_V22_ARGS+=(--resume)
fi

cd "${PIER_V22_ROOT}"
exec "${PIER_V22_PYTHON}" -m pier_llm_reanalysis_v22.pipeline "${PIER_V22_ARGS[@]}"
