#!/usr/bin/env bash
# Reference commands for spinning up the VLM server locally.
# Pick ONE backend; all three expose an OpenAI-compatible /v1 endpoint.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
BACKEND="${BACKEND:-vllm}"

case "$BACKEND" in
  vllm)
    # ~16GB VRAM in bf16. Fits comfortably on a 3090 Ti.
    vllm serve "$MODEL" \
      --dtype bfloat16 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.85 \
      --port 8000
    ;;
  llamacpp)
    # Requires the GGUF model and matching mmproj vision projector.
    : "${GGUF:?set GGUF=/path/to/qwen3-vl-8b-instruct-Q5_K_M.gguf}"
    : "${MMPROJ:?set MMPROJ=/path/to/mmproj-qwen3-vl-8b-f16.gguf}"
    llama-server \
      -m "$GGUF" \
      --mmproj "$MMPROJ" \
      -ngl 99 \
      -c 16384 \
      --port 8080
    ;;
  sglang)
    python -m sglang.launch_server \
      --model-path "$MODEL" \
      --port 30000
    ;;
  *)
    echo "Unknown BACKEND: $BACKEND (use vllm|llamacpp|sglang)" >&2
    exit 1
    ;;
esac
