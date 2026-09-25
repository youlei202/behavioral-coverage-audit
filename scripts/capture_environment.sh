#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BCA_ROOT"
mkdir -p build/environment
"${BCA_PYTHON:-python3}" --version > build/environment/python.txt 2>&1
"${BCA_PYTHON:-python3}" -m pip freeze > build/environment/pip-freeze.txt
if command -v pdflatex > /dev/null; then pdflatex --version > build/environment/pdflatex.txt; fi
if command -v bibtex > /dev/null; then bibtex --version > build/environment/bibtex.txt; fi
if command -v nvidia-smi > /dev/null; then
  nvidia-smi > build/environment/nvidia-smi.txt 2>&1 || true
else
  printf '%s\n' 'nvidia-smi unavailable; no GPU inference run' > build/environment/nvidia-smi.txt
fi
printf '%s\n' 'Environment captured in build/environment/'
