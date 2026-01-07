#!/usr/bin/env bash
RAY_memory_monitor_refresh_ms=0 \
RAY_memory_usage_threshold=0.8 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --served-model-name Qwen/Qwen3-4B-Thinking-2507 \
    --trust-remote-code \
    --port 8083 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.88 \
    --swap-space 2 \
    --max-model-len 16384 \
    --max-num-seqs 64