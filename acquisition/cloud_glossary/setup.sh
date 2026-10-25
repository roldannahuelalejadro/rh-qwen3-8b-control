#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'printf "%s\n" "$?" > setup.exit' EXIT
python3 -m virtualenv .venv
.venv/bin/python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python prepare.py
.venv/bin/python run.py preflight --output preflight
