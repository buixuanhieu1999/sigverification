#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BACKEND="${1:-auto}"
PORT="${SIGNATURE_GRADIO_PORT:-7860}"
HOST="${SIGNATURE_GRADIO_HOST:-127.0.0.1}"

if [[ "$BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    BACKEND="gpu"
  else
    BACKEND="cpu"
  fi
fi

case "$BACKEND" in
  gpu) RUNTIME_EXTRA="onnx-gpu" ;;
  cpu) RUNTIME_EXTRA="onnx-cpu" ;;
  *) echo "Usage: bash scripts/run_demo.sh [auto|gpu|cpu]" >&2; exit 2 ;;
esac

echo "Starting Gradio demo on http://${HOST}:${PORT} using ${BACKEND} runtime"
export SIGNATURE_GRADIO_HOST="$HOST"
export SIGNATURE_GRADIO_PORT="$PORT"
exec uv run --extra "$RUNTIME_EXTRA" --extra demo signature-verifier-demo
