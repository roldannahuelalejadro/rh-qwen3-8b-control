#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'printf "%s\n" "$?" > pipeline.exit' EXIT
echo "Waiting for the running smoke test."
deadline=$((SECONDS + 7200))
while tmux has-session -t glossary-smoke 2>/dev/null; do
    if (( SECONDS >= deadline )); then
        echo "Smoke wait timeout; acquisition not started."
        exit 1
    fi
    sleep 10
done
if [ ! -f smoke.exit ] || [ "$(<smoke.exit)" != 0 ]; then
    echo "Smoke did not succeed; acquisition not started. Inspect smoke.log."
    exit 1
fi
echo "Smoke and capture verification passed. Starting 100 full rollouts."
bash launch.sh acquire 100 > acquire.log 2>&1
echo "100 rollouts and verification completed."
