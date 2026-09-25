#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

cd "${PIER_PERSISTENT_ROOT}"
mkdir -p env/wheelhouse env/cache logs status

if [[ ! -x env/venv/bin/python ]]; then
  uv venv --python 3.11 --seed env/venv
fi
# shellcheck disable=SC1091
source env/venv/bin/activate
if ! python -c 'import torch, transformers, datasets, cvxpy, pandas, pytest, ruff, mypy; assert torch.version.cuda == "12.8"' >/dev/null 2>&1; then
  python -m pip install --upgrade "pip==25.3" "setuptools==80.9.0" "wheel==0.45.1"
  python -m pip download --dest env/wheelhouse \
    "pip==25.3" "setuptools==80.9.0" "wheel==0.45.1"
  python -m pip download --requirement env/requirements.in --dest env/wheelhouse
  python -m pip install --no-index --find-links env/wheelhouse --requirement env/requirements.in
else
  echo "[resume] validated existing Python 3.11 CUDA environment and wheelhouse"
fi
python -m pip install --no-deps --editable .
python -m pip freeze --all | LC_ALL=C sort > env/requirements.lock.txt

python -m pier_llm.prestage --root "${PIER_PERSISTENT_ROOT}"
