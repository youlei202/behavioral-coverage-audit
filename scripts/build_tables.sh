#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$BCA_ROOT"
exec "${BCA_PYTHON:-python3}" scripts/replay.py tables "$@"
