#!/bin/bash

LOG_FILE="logs/vllm_serve_prefill_$(date +'%Y%m%d_%H-%M-%S').log"

set -ex
# export GLOO_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_IFNAME=eth0

export CUDA_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=1
export NCCL_NCHANNELS_PER_NET_PEER=1
#export VLLM_LOGGING_CONFIG_PATH=log.conf.json
#export NCCL_DEBUG=INFO 
export VLLM_RINGBUFFER_WARNING_INTERVAL=500 
export VLLM_RPC_TIMEOUT=1800000 
# export    NCCL_IB_DISABLE=1
# mkdir -p profiler
# export VLLM_TORCH_PROFILER_DIR=./profiler

export VLLM_USE_V1=1 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ENABLE_DSV3=0  
export SAFETENSORS_FAST_GPU=1   

{
 vllm serve /data/models/glm-4-9b-chat \
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
    --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_buffer_size":"4e10","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20005","send_type":"PUT","nccl_num_channels":"16"}}'
 } 2>&1 | tee -a "$LOG_FILE" & 