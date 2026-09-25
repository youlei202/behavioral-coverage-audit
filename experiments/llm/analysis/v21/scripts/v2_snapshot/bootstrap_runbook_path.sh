#!/usr/bin/env bash
set -Eeuo pipefail

PIER_LOGICAL_PARENT="/work/Lei"
PIER_LOGICAL_ROOT="${PIER_LOGICAL_PARENT}/PIER_LLM_ECOSYSTEM_GOLDMINE_V2"
PIER_PERSISTENT_ROOT="/work/Users/leiyo/PIER_LLM_ECOSYSTEM_GOLDMINE_V2"

if [[ ! -d "${PIER_PERSISTENT_ROOT}" ]]; then
  echo "Persistent project root is unavailable: ${PIER_PERSISTENT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${PIER_LOGICAL_PARENT}" ]]; then
  echo "Creating ${PIER_LOGICAL_PARENT} requires platform/root authorization." >&2
  sudo install -d -o "$(id -un)" -g "$(id -gn)" "${PIER_LOGICAL_PARENT}"
fi
if [[ -e "${PIER_LOGICAL_ROOT}" || -L "${PIER_LOGICAL_ROOT}" ]]; then
  if [[ "$(readlink -f "${PIER_LOGICAL_ROOT}")" != "$(readlink -f "${PIER_PERSISTENT_ROOT}")" ]]; then
    echo "Refusing to replace an existing unrelated path: ${PIER_LOGICAL_ROOT}" >&2
    exit 2
  fi
else
  ln -s "${PIER_PERSISTENT_ROOT}" "${PIER_LOGICAL_ROOT}"
fi
readlink -f "${PIER_LOGICAL_ROOT}"
