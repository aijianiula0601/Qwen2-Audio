#!/bin/bash
# Qwen2-Audio Environment Activation Script

# 设置项目路径
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL设置（根据网络环境调整）
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口修改
# export NCCL_IB_DISABLE=1  # 如果InfiniBand有问题则取消注释

# DeepSpeed设置
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

echo "Environment variables set for Qwen2-Audio multi-node training"
echo "Project root: $PROJECT_ROOT"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
