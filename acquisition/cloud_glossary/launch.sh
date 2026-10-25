#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:?Specify smoke or acquire}"
count="${2:-20}"
trap 'printf "%s\n" "$?" > "'"$mode"'.exit"' EXIT
snapshot="$(<snapshot_path.txt)"
if [ "$mode" = smoke ]; then
    .venv/bin/python -u run.py smoke --snapshot "$snapshot" --output smoke
    .venv/bin/python verify.py smoke
elif [ "$mode" = acquire ]; then
    .venv/bin/python -u run.py acquire --snapshot "$snapshot" --output acquisition --rollouts "$count"
    .venv/bin/python verify.py acquisition
else
    exit 2
fi
