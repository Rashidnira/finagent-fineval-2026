#!/usr/bin/env bash
# Reproducible launcher for the shared-task inference server.
# Usage:  ./serve_vllm.sh [GPU_INDEX]
# Then:   export FINNLP_BASE_URL=http://127.0.0.1:${PORT}/v1
set -euo pipefail
cd "$(dirname "$0")"

GPU="${1:-0}"
PORT="${FINNLP_PORT:-8770}"
MODEL="${FINNLP_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"   # 32B, within the 70B rule cap
MAXLEN="${FINNLP_MAX_MODEL_LEN:-4096}"
UTIL="${FINNLP_GPU_UTIL:-0.70}"                          # shared box: do not take the whole card
PY="${FINNLP_VLLM_PYTHON:-python}"

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p logs
echo "serving $MODEL on GPU $GPU, port $PORT, ctx $MAXLEN, mem-util $UTIL"
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --quantization awq_marlin \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "${FINNLP_MAX_SEQS:-8}" \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --seed 0 \
  --disable-log-requests
