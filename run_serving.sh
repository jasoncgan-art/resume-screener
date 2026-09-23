#!/usr/bin/env bash
# Run the resume-screener serving API locally.
# Usage:  bash run_serving.sh            # reads host/port from serving/config.yaml
#         SCREENER_PORT=9000 bash run_serving.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f artifacts/biencoder.pt ] || [ ! -f artifacts/dataset.pt ]; then
  echo "ERROR: artifacts/biencoder.pt and artifacts/dataset.pt are required." >&2
  echo "Run 'python train.py' first (this script never retrains)." >&2
  exit 1
fi

# Load host/port from config (env SCREENER_HOST/SCREENER_PORT override via uvicorn args below).
HOST="${SCREENER_HOST:-}"
PORT="${SCREENER_PORT:-}"

exec python -m uvicorn serving.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --workers 1
