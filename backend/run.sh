#!/bin/bash
# Start the backend server using uv

# Default settings
MODEL_NAME="${MODEL_NAME:-gpt2}"
DEVICE="${DEVICE:-}"
DTYPE="${DTYPE:-float16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting LLM Interpretability Workbench Backend"
echo "Model: $MODEL_NAME"
echo "Device: ${DEVICE:-auto}"
echo "Server: $HOST:$PORT"

export MODEL_NAME DEVICE DTYPE

# Run with uv
uv run uvicorn app.main:app --host $HOST --port $PORT --reload
