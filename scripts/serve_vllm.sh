#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 MODEL_OR_CHECKPOINT [additional vllm serve arguments...]" >&2
    exit 2
fi

MODEL=$1
shift

PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
REASONING_PARSER=${REASONING_PARSER:-}

if ! command -v vllm >/dev/null 2>&1; then
    echo "Error: vllm is not available on PATH." >&2
    exit 1
fi

args=(
    vllm serve "$MODEL"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --enable-prefix-caching
    --trust-remote-code
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
    args+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$REASONING_PARSER" ]]; then
    args+=(--reasoning-parser "$REASONING_PARSER")
fi

echo "Serving $MODEL at http://localhost:$PORT/v1" >&2
exec "${args[@]}" "$@"
