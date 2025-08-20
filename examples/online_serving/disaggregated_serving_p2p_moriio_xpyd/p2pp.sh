#!/bin/bash

LOG_FILE="logs/vllm_serve_prefill_$(date +'%Y%m%d_%H-%M-%S').log"

set -ex
export GLOO_SOCKET_IFNAME=ens50f0
export NCCL_SOCKET_IFNAME=ens50f0

export CUDA_VISIBLE_DEVICES=4
export HIP_VISIBLE_DEVICES=4
# models--chwan--DeepSeek-V3-5layer/snapshots/38a0c0ee55158e7d2ac9a6af1de94c4dfe084872
export NCCL_NCHANNELS_PER_NET_PEER=1
#export VLLM_LOGGING_CONFIG_PATH=log.conf.json
#export NCCL_DEBUG=INFO 
export VLLM_RINGBUFFER_WARNING_INTERVAL=500 
export VLLM_RPC_TIMEOUT=1800000 
export    NCCL_IB_DISABLE=0
# mkdir -p profiler
# export VLLM_TORCH_PROFILER_DIR=./profiler

# 
{
  VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ENABLE_DSV3=0  SAFETENSORS_FAST_GPU=1  vllm serve /apps/data/models/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 \
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
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"4e10","kv_port":"21007","kv_connector_extra_config":{"proxy_ip":"10.235.192.54","proxy_port":"30001","http_port":"20005","send_type":"PUT","nccl_num_channels":"16"}}'
 } 2>&1 | tee -a "$LOG_FILE" &