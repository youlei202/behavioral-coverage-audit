#!/usr/bin/env bash
# Source before Python/tests; caches and environments stay outside the source tree.
set -euo pipefail
BCA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BCA_WORK_ROOT="${BCA_WORK_ROOT:-$(dirname -- "$BCA_ROOT")/behavioral-coverage-audit-work}"
export TMPDIR="$BCA_WORK_ROOT/tmp"
export XDG_CACHE_HOME="$BCA_WORK_ROOT/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
