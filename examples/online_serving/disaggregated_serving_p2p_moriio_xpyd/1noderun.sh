#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=3
export HIP_VISIBLE_DEVICES=3
export VLLM_USE_V1=1 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ENABLE_DSV3=0  
export SAFETENSORS_FAST_GPU=1   

 vllm serve /apps/data/models/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/  \
    -tp 1 \
    --block-size 16 \
    --max_seq_len_to_capture 6144 \
    --max-num-batched-tokens 6144 \
    --host 0.0.0.0 \
    --port 20005 \
    --enforce-eager \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --served-model-name deepseek-ai/DeepSeek-R1 