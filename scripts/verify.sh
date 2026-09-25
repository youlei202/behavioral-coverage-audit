#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BCA_ROOT"
"${BCA_PYTHON:-python3}" -m pytest
"${BCA_PYTHON:-python3}" scripts/verify_outputs.py
