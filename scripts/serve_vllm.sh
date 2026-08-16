#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 MODEL_OR_CHECKPOINT [additional vllm serve arguments...]" >&2
    exit 2
fi

MODEL=$1
shift

PORT=${PORT:-8000}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
DTYPE=${DTYPE:-bfloat16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
REASONING_PARSER=${REASONING_PARSER:-}
MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}

if ! command -v vllm >/dev/null 2>&1; then
    echo "Error: vllm is not available on PATH." >&2
    exit 1
fi

args=(
    vllm serve "$MODEL"
    --host "$VLLM_HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --enable-prefix-caching
    --trust-remote-code
    --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
    args+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$REASONING_PARSER" ]]; then
    args+=(--reasoning-parser "$REASONING_PARSER")
fi

case "${ENFORCE_EAGER,,}" in
    1|true|yes)
        args+=(--enforce-eager)
        ;;
    0|false|no)
        ;;
    *)
        echo "Error: ENFORCE_EAGER must be true/false, yes/no, or 1/0." >&2
        exit 2
        ;;
esac

echo "Serving $MODEL at http://$VLLM_HOST:$PORT/v1" >&2
exec "${args[@]}" "$@"
