#!/usr/bin/env bash
# End-to-end run: generate data -> train -> evaluate -> sample screening.
set -euo pipefail
cd "$(dirname "$0")"

# Prefer the project venv (has torch); fall back to system python3.
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

$PY data_gen.py
$PY dataset.py
$PY train.py
$PY evaluate.py
$PY screen.py --job_id "$($PY -c "import json; print(json.load(open('data/split.json'))['test_job_ids'][0])")" --top_n 10
