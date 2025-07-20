# Qwen2-Audio 多机训练完整指南

本文档详细介绍如何使用Qwen2-Audio项目的多机训练功能，包括环境配置、脚本使用和最佳实践。

## 📋 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
- [使用示例](#使用示例)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 🛠️ 环境要求

### 硬件要求
- 多台GPU服务器（推荐每台8张GPU）
- 节点间网络连通性良好
- 共享存储（推荐NFS或分布式文件系统）

### 软件要求
- Linux操作系统
- Python 3.8+
- PyTorch 2.0+
- DeepSpeed
- CUDA 11.8+
- SSH无密码登录配置

### 网络要求
- 节点间延迟 < 10ms
- 带宽 > 10Gbps（推荐）
- 端口29500开放（可配置）

## 🚀 快速开始

### 1. 环境准备

```bash
# 1. 设置多机环境
bash scripts/setup_multinode_env.sh --method conda --env-name qwen2-audio-multinode

# 2. 检查环境
bash scripts/check_environment.sh

# 3. 配置SSH无密码登录
ssh-keygen -t rsa -b 4096
ssh-copy-id user@node1.cluster.com
ssh-copy-id user@node2.cluster.com
ssh-copy-id user@node3.cluster.com
```

### 2. 一键启动训练

```bash
# 完整训练流程（推荐）
bash scripts/train_full_pipeline_multinode.sh \
    --model qwen2_7b \
    --nodes "node1,node2,node3" \
    --gpus-per-node 8
```

## 📜 脚本说明

### 1. 统一启动脚本

#### `scripts/launch_multinode_training.sh`
统一的多机训练管理脚本，支持环境检查、网络测试、训练启动等功能。

**主要功能：**
- 环境检查和验证
- 网络连通性测试
- GPU状态监控
- 训练启动和管理
- 日志查看和状态监控

**使用示例：**
```bash
# 检查环境
bash scripts/launch_multinode_training.sh check-env --nodes "node1,node2,node3"

# 检查网络
bash scripts/launch_multinode_training.sh check-network --nodes "node1,node2,node3"

# 启动训练
bash scripts/launch_multinode_training.sh launch \
    --model qwen2_7b \
    --stage pretrain \
    --nodes "node1,node2,node3"

# 查看状态
bash scripts/launch_multinode_training.sh status

# 查看日志
bash scripts/launch_multinode_training.sh logs
```

### 2. 分阶段训练脚本

#### `scripts/train_pretrain_multinode.sh`
预训练阶段的多机训练脚本。

**特点：**
- 支持大规模音频-文本数据训练
- 自动环境检查和激活
- 灵活的配置选项
- 详细的日志记录

#### `scripts/train_sft_multinode.sh`
监督微调阶段的多机训练脚本。

**特点：**
- 基于预训练模型进行微调
- 支持指令数据训练
- 自动检查点恢复

#### `scripts/train_dpo_multinode.sh`
直接偏好优化阶段的多机训练脚本。

**特点：**
- 基于人类偏好优化模型
- 需要TRL库支持
- 支持偏好数据训练

### 3. 完整流程脚本

#### `scripts/train_full_pipeline_multinode.sh`
一键启动完整训练流程的脚本。

**特点：**
- 自动执行pretrain → sft → dpo三个阶段
- 支持断点续训
- 支持跳过特定阶段
- 完整的错误处理和日志记录

## 💡 使用示例

### 示例1：完整训练流程

```bash
# 启动完整训练流程
bash scripts/train_full_pipeline_multinode.sh \
    --model qwen2_7b \
    --nodes "node1,node2,node3" \
    --gpus-per-node 8 \
    --master-port 29500
```

### 示例2：从特定阶段恢复

```bash
# 从SFT阶段开始，跳过预训练
bash scripts/train_full_pipeline_multinode.sh \
    --model qwen2_7b \
    --resume-stage sft \
    --resume-checkpoint outputs/pretrain_qwen2_7b_20241201_120000/ \
    --nodes "node1,node2,node3"
```

### 示例3：只训练特定阶段

```bash
# 只进行SFT训练
bash scripts/train_full_pipeline_multinode.sh \
    --model qwen2_7b \
    --skip-pretrain \
    --skip-dpo \
    --nodes "node1,node2,node3"
```

### 示例4：使用hostfile方式

```bash
# 创建hostfile
cat > configs/hostfile.txt << EOF
node1.cluster.com slots=8
node2.cluster.com slots=8
node3.cluster.com slots=8
EOF

# 使用hostfile启动
bash scripts/train_pretrain_multinode.sh \
    --model qwen2_7b \
    --hostfile configs/hostfile.txt \
    --master-addr node1.cluster.com
```

### 示例5：环境检查和监控

```bash
# 检查所有节点环境
bash scripts/launch_multinode_training.sh check-env \
    --nodes "node1,node2,node3" \
    --verbose

# 检查网络连通性
bash scripts/launch_multinode_training.sh check-network \
    --nodes "node1,node2,node3"

# 检查GPU状态
bash scripts/launch_multinode_training.sh check-gpu \
    --nodes "node1,node2,node3"
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NNODES` | 节点总数 | 1 |
| `NODE_RANK` | 当前节点序号 | 0 |
| `MASTER_ADDR` | 主节点地址 | localhost |
| `MASTER_PORT` | 主节点端口 | 29500 |
| `NPROC_PER_NODE` | 每节点GPU数量 | 8 |
| `WORLD_SIZE` | 总进程数 | NNODES × NPROC_PER_NODE |

### 配置文件

#### `configs/base_config.yaml`
基础训练配置文件，包含通用的训练参数。

#### `configs/models/`
模型特定配置文件目录：
- `qwen2_0.5b.yaml` - Qwen2-0.5B模型配置
- `qwen2_7b.yaml` - Qwen2-7B模型配置
- `qwen2_70b.yaml` - Qwen2-70B模型配置
- `llama3_8b.yaml` - LLaMA3-8B模型配置

#### `configs/deepspeed_config.json`
DeepSpeed配置文件，用于优化训练性能。

## 🐛 故障排除

### 常见问题

#### 1. 节点间无法通信

**症状：** NCCL通信错误，节点间无法建立连接

**解决方案：**
```bash
# 检查网络连通性
bash scripts/launch_multinode_training.sh check-network --nodes "node1,node2,node3"

# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

#### 2. GPU内存不足

**症状：** CUDA out of memory错误

**解决方案：**
```bash
# 减少batch size
# 修改configs/base_config.yaml中的per_device_train_batch_size

# 启用gradient checkpointing
# 在配置文件中设置gradient_checkpointing: true

# 使用更激进的DeepSpeed ZeRO配置
# 修改configs/deepspeed_config.json
```

#### 3. 环境不一致

**症状：** 不同节点上的Python环境或依赖版本不一致

**解决方案：**
```bash
# 检查环境一致性
bash scripts/launch_multinode_training.sh check-env --nodes "node1,node2,node3"

# 重新设置环境
bash scripts/setup_multinode_env.sh --method conda --env-name qwen2-audio-multinode
```

#### 4. 数据访问问题

**症状：** 某些节点无法访问训练数据

**解决方案：**
```bash
# 检查数据目录权限
ls -la data/pretrain/
ls -la data/sft/
ls -la data/dpo/

# 确保所有节点都能访问共享存储
# 检查NFS挂载或分布式文件系统配置
```

### 调试技巧

#### 1. 启用详细日志

```bash
# 设置详细日志
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 查看训练日志
tail -f logs/*/training.log
```

#### 2. 监控资源使用

```bash
# 监控GPU使用情况
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'

# 监控网络使用情况
iftop -i eth0
```

#### 3. 检查进程状态

```bash
# 检查训练进程
ps aux | grep train_.*_multinode.sh

# 检查DeepSpeed进程
ps aux | grep deepspeed
```

## 📈 最佳实践

### 1. 环境配置

- 使用conda环境确保依赖一致性
- 定期更新依赖包版本
- 使用共享存储避免数据重复

### 2. 网络优化

- 使用InfiniBand或高速以太网
- 配置NCCL环境变量优化通信
- 监控网络带宽使用

### 3. 训练优化

- 根据GPU内存调整batch size
- 使用gradient checkpointing节省内存
- 启用混合精度训练加速

### 4. 监控和日志

- 定期检查训练日志
- 监控GPU和网络使用情况
- 设置训练进度通知

### 5. 故障恢复

- 定期保存检查点
- 使用断点续训功能
- 备份重要配置文件

## 📞 技术支持

如果遇到问题，请：

1. 查看相关日志文件
2. 检查环境配置
3. 参考故障排除部分
4. 提交Issue到项目仓库

## 📚 相关文档

- [环境设置指南](ENVIRONMENT_SETUP.md)
- [多机环境配置](MULTI_NODE_ENVIRONMENT_SETUP.md)
- [训练配置说明](configs/README.md)
- [模型架构说明](models/README.md) 