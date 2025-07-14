# Qwen2-Audio 多机训练指南

本文档详细介绍如何在多台机器上训练Qwen2-Audio模型。

## 前置条件

### 1. 环境要求
- 所有节点都安装了相同版本的Python、PyTorch、DeepSpeed等依赖
- 所有节点都有相同的项目代码和数据
- 所有节点之间网络连通良好，延迟较低

### 2. SSH配置
确保所有节点之间可以无密码SSH登录：

```bash
# 在每个节点上生成SSH密钥
ssh-keygen -t rsa -b 4096

# 将公钥复制到所有其他节点
ssh-copy-id user@node1.cluster.com
ssh-copy-id user@node2.cluster.com
# ... 对所有节点重复此操作

# 测试无密码登录
ssh user@node1.cluster.com "echo 'SSH connection successful'"
```

### 3. 共享存储（推荐）
建议使用共享存储（如NFS、GlusterFS等）来确保：
- 所有节点访问相同的训练数据
- 模型检查点能被所有节点访问
- 日志文件集中存储

## 配置方法

### 方法1：使用DeepSpeed Hostfile（推荐）

1. **创建hostfile**
```bash
# 创建 configs/hostfile.txt
vim configs/hostfile.txt
```

内容示例：
```
node0.cluster.com slots=8
node1.cluster.com slots=8
node2.cluster.com slots=8
node3.cluster.com slots=8
```

2. **在主节点启动训练**
```bash
# 在node0上执行
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_7b \
    --hostfile configs/hostfile.txt \
    --master-addr node0.cluster.com \
    --master-port 29500
```

### 方法2：手动在每个节点启动

在每个节点上分别执行相应的命令：

```bash
# 在主节点 (node0) 执行
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_7b \
    --nnodes 4 \
    --node-rank 0 \
    --master-addr node0.cluster.com \
    --master-port 29500 \
    --gpus-per-node 8

# 在工作节点 (node1) 执行  
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_7b \
    --nnodes 4 \
    --node-rank 1 \
    --master-addr node0.cluster.com \
    --master-port 29500 \
    --gpus-per-node 8

# 在工作节点 (node2) 执行
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_7b \
    --nnodes 4 \
    --node-rank 2 \
    --master-addr node0.cluster.com \
    --master-port 29500 \
    --gpus-per-node 8

# 在工作节点 (node3) 执行
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_7b \
    --nnodes 4 \
    --node-rank 3 \
    --master-addr node0.cluster.com \
    --master-port 29500 \
    --gpus-per-node 8
```

## 环境变量说明

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `NNODES` | 节点总数 | 4 |
| `NODE_RANK` | 当前节点序号 (0到nnodes-1) | 0, 1, 2, 3 |
| `MASTER_ADDR` | 主节点地址 | node0.cluster.com |
| `MASTER_PORT` | 主节点通信端口 | 29500 |
| `NPROC_PER_NODE` | 每个节点的GPU数量 | 8 |
| `WORLD_SIZE` | 总的进程数 (NNODES × NPROC_PER_NODE) | 32 |

## 配置示例

### 小规模集群 (2节点，每节点4GPU)
```bash
# 总计8个GPU
export NNODES=2
export GPUS_PER_NODE=4
export MASTER_ADDR="192.168.1.10"
export MASTER_PORT=29500

# 节点0
NODE_RANK=0 bash scripts/train_pretrain_multinode.sh --model qwen2_0.5b

# 节点1  
NODE_RANK=1 bash scripts/train_pretrain_multinode.sh --model qwen2_0.5b
```

### 大规模集群 (8节点，每节点8GPU)
```bash
# 总计64个GPU
export NNODES=8
export GPUS_PER_NODE=8
export MASTER_ADDR="node0.hpc.cluster"
export MASTER_PORT=29500

# 使用hostfile方式更方便管理
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_70b \
    --hostfile configs/hostfile_64gpu.txt
```

## 监控和调试

### 1. 检查训练状态
```bash
# 查看所有节点的GPU使用情况
for node in node0 node1 node2 node3; do
    echo "=== $node ==="
    ssh $node "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv"
done

# 查看训练日志
tail -f outputs/pretrain_*/pretrain_node*.log
```

### 2. 网络连通性测试
```bash
# 测试节点间通信
python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='tcp://node0:29500', world_size=4, rank=0)
print('Distributed initialization successful')
"
```

### 3. 常见问题排查

**问题1：节点间无法通信**
```bash
# 检查防火墙设置
sudo ufw status
sudo firewall-cmd --list-all

# 检查端口是否被占用
netstat -tulpn | grep 29500

# 测试端口连通性
telnet node0.cluster.com 29500
```

**问题2：NCCL通信错误**
```bash
# 设置NCCL调试信息
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果InfiniBand有问题
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
```

**问题3：GPU内存不足**
```bash
# 减少batch size或启用gradient checkpointing
# 修改configs/base_config.yaml中的per_device_train_batch_size
# 或使用更激进的DeepSpeed ZeRO配置
```

## 性能优化建议

### 1. 网络优化
- 使用高速网络（InfiniBand、100GbE等）
- 优化NCCL设置
- 确保节点间延迟 < 1ms

### 2. 存储优化  
- 使用高性能共享存储
- 启用数据预加载和缓存
- 使用SSD存储训练数据

### 3. 负载均衡
- 确保所有节点硬件配置相似
- 均匀分配数据加载任务
- 监控各节点GPU利用率

## 脚本参数完整列表

```bash
bash scripts/train_pretrain_multinode.sh [OPTIONS]

Multi-node Options:
  --nnodes NNODES            Number of nodes
  --node-rank NODE_RANK      Current node rank (0 to nnodes-1)  
  --master-addr MASTER_ADDR  Master node address
  --master-port MASTER_PORT  Master port for communication
  --gpus-per-node GPUS       Number of GPUs per node
  --hostfile HOSTFILE        DeepSpeed hostfile path

Model Options:
  --model MODEL_TYPE         Model type (qwen2_7b, qwen2_70b, llama3_8b, qwen2_0.5b)
  --config CONFIG_FILE       Path to base config file  
  --resume CHECKPOINT        Resume from checkpoint
```

## 高级配置

### 1. 自定义DeepSpeed配置
```bash
# 创建多机专用的DeepSpeed配置
cp configs/deepspeed_config.json configs/deepspeed_multinode_config.json

# 修改配置以适应多机环境
vim configs/deepspeed_multinode_config.json
```

### 2. 使用Slurm作业调度系统
```bash
#!/bin/bash
#SBATCH --job-name=qwen2-audio-multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

# 设置多机训练环境
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_PROCID
export GPUS_PER_NODE=8

# 启动训练
srun bash scripts/train_pretrain_multinode.sh --model qwen2_7b
```

这个指南提供了完整的多机训练配置方法，你可以根据自己的集群环境选择合适的方案。 