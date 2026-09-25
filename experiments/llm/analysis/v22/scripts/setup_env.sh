#!/usr/bin/env bash
set -Eeuo pipefail

PIER_V22_ROOT="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2_2_TRACKWISE_REANALYSIS"
PIER_V22_WHEELHOUSE="/work/Lei/PIER_LLM_ECOSYSTEM_GOLDMINE_V2/env/wheelhouse"

cd "${PIER_V22_ROOT}"
uv python install 3.11.13
uv venv env/venv --python 3.11.13
uv pip install \
  --python env/venv/bin/python \
  --offline \
  --find-links "${PIER_V22_WHEELHOUSE}" \
  --requirement env/requirements.lock.txt

env/venv/bin/python -c 'import clarabel, cvxpy, matplotlib, numpy, osqp, pandas, pyarrow, pytest, scipy, sklearn; print("V2.2 CPU environment ready")'
