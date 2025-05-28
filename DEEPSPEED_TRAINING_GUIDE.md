# Qwen2-Audio DeepSpeed 训练指南

本指南介绍如何使用DeepSpeed来训练Qwen2-Audio模型，以实现更高效的内存使用和更快的训练速度。

## 🚀 DeepSpeed 优势

### 内存优化
- **ZeRO-2**: 优化器状态分片，节省50%显存
- **ZeRO-3**: 参数分片，支持训练超大模型
- **CPU卸载**: 将优化器状态和参数卸载到CPU，进一步节省显存

### 性能提升
- **梯度压缩**: 减少通信开销
- **混合精度训练**: 使用FP16/BF16加速训练
- **激活检查点**: 以计算换内存

### 易用性
- **自动配置**: 批大小、学习率等参数自动调整
- **无缝集成**: 与Transformers库完美集成

## 📋 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support (推荐RTX 3080+, A100, H100)
- **内存**: 至少32GB系统内存
- **存储**: 至少2TB可用空间

### 软件依赖
```bash
# 安装DeepSpeed
pip install deepspeed>=0.12.0

# 验证安装
ds_report
```

## 🔧 配置文件说明

项目提供了三个预配置的DeepSpeed配置文件：

### Stage 1 配置 (`configs/deepspeed_stage1.json`)
- **ZeRO-2**: 适合预训练阶段
- **FP16**: 混合精度训练
- **无CPU卸载**: 最大化训练速度

### Stage 2 配置 (`configs/deepspeed_stage2.json`)
- **ZeRO-3**: 支持更大模型
- **CPU卸载**: 优化器和参数卸载到CPU
- **BF16**: 更稳定的混合精度
- **激活检查点**: 节省显存

### Stage 3 配置 (`configs/deepspeed_stage3.json`)
- **ZeRO-3**: 最大内存优化
- **CPU卸载**: 全面卸载策略
- **BF16**: 适合DPO训练

## 🎯 使用方法

### 方法一：统一脚本（推荐）

使用统一的DeepSpeed训练脚本：

```bash
# Stage 1: 预训练
bash scripts/train_deepspeed.sh \
    --stage stage1 \
    --model_path models/Qwen_Qwen2-7B \
    --data_path data/stage1_pretraining/train.jsonl \
    --output_dir checkpoints/qwen2-audio-stage1-deepspeed

# Stage 2: 监督微调
bash scripts/train_deepspeed.sh \
    --stage stage2 \
    --model_path checkpoints/qwen2-audio-stage1-deepspeed \
    --data_path data/stage2_sft/train.jsonl \
    --output_dir checkpoints/qwen2-audio-stage2-deepspeed

# Stage 3: DPO优化
bash scripts/train_deepspeed.sh \
    --stage stage3 \
    --model_path checkpoints/qwen2-audio-stage2-deepspeed \
    --data_path data/stage3_dpo/train.jsonl \
    --output_dir checkpoints/qwen2-audio-stage3-deepspeed \
    --freeze_audio_encoder
```

### 方法二：分阶段脚本

使用独立的阶段脚本：

```bash
# Stage 1
bash scripts/train_stage1_deepspeed.sh

# Stage 2  
bash scripts/train_stage2_deepspeed.sh

# Stage 3
bash scripts/train_stage3_deepspeed.sh
```

### 方法三：直接调用

直接使用deepspeed命令：

```bash
deepspeed --num_gpus=4 src/trainer.py \
    --model_name_or_path models/Qwen_Qwen2-7B \
    --data_path data/stage1_pretraining/train.jsonl \
    --output_dir checkpoints/stage1-deepspeed \
    --training_stage stage1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --deepspeed configs/deepspeed_stage1.json \
    --bf16 True \
    --report_to wandb
```

## ⚙️ 参数配置

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--stage` | 训练阶段 | 必需 |
| `--model_path` | 模型路径 | 必需 |
| `--data_path` | 数据路径 | 必需 |
| `--output_dir` | 输出目录 | 必需 |
| `--num_gpus` | GPU数量 | 4 |
| `--deepspeed_config` | DeepSpeed配置文件 | 自动选择 |

### 训练参数

| 参数 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|
| `--batch_size` | 4 | 2 | 1 |
| `--gradient_accumulation` | 4 | 8 | 16 |
| `--learning_rate` | 1e-4 | 5e-5 | 5e-6 |
| `--num_epochs` | 3 | 2 | 1 |

### 高级参数

```bash
# 自定义批大小和学习率
bash scripts/train_deepspeed.sh \
    --stage stage1 \
    --model_path models/Qwen_Qwen2-7B \
    --data_path data/stage1_pretraining/train.jsonl \
    --output_dir checkpoints/stage1-custom \
    --batch_size 8 \
    --gradient_accumulation 2 \
    --learning_rate 2e-4 \
    --num_epochs 5

# 使用自定义DeepSpeed配置
bash scripts/train_deepspeed.sh \
    --stage stage2 \
    --model_path checkpoints/stage1 \
    --data_path data/stage2_sft/train.jsonl \
    --output_dir checkpoints/stage2-custom \
    --deepspeed_config my_custom_config.json

# 冻结特定组件
bash scripts/train_deepspeed.sh \
    --stage stage3 \
    --model_path checkpoints/stage2 \
    --data_path data/stage3_dpo/train.jsonl \
    --output_dir checkpoints/stage3-frozen \
    --freeze_audio_encoder \
    --freeze_llm
```

## 📊 性能对比

### 内存使用对比

| 模型大小 | 传统训练 | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-3 |
|----------|----------|------------------|------------------|
| **7B** | 32GB | 16GB | 12GB |
| **13B** | 64GB | 32GB | 20GB |
| **72B** | 320GB | 160GB | 80GB |

### 训练速度对比

| 配置 | 传统训练 | DeepSpeed | 提升 |
|------|----------|-----------|------|
| **4×RTX 4090** | 100% | 120% | +20% |
| **8×A100** | 100% | 140% | +40% |

## 🔍 监控和调试

### 训练监控

```bash
# 查看训练日志
tail -f checkpoints/stage1-deepspeed/train.log

# 监控GPU使用
watch -n 1 nvidia-smi

# 查看DeepSpeed状态
deepspeed --num_gpus=4 src/trainer.py --help
```

### 常见问题

#### 1. 内存不足 (OOM)
```bash
# 解决方案：减少批大小或启用CPU卸载
bash scripts/train_deepspeed.sh \
    --stage stage2 \
    --batch_size 1 \
    --gradient_accumulation 16 \
    --deepspeed_config configs/deepspeed_stage2.json
```

#### 2. 训练速度慢
```bash
# 解决方案：调整配置或使用更少的CPU卸载
# 修改deepspeed配置文件中的offload设置
{
    "zero_optimization": {
        "offload_optimizer": {
            "device": "none"  # 禁用CPU卸载
        }
    }
}
```

#### 3. 梯度爆炸
```bash
# 解决方案：降低学习率或启用梯度裁剪
bash scripts/train_deepspeed.sh \
    --learning_rate 1e-5 \
    # DeepSpeed配置中已包含gradient_clipping: 1.0
```

## 🎛️ 自定义配置

### 创建自定义DeepSpeed配置

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto", 
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu", 
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "bf16": {
        "enabled": true
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    },
    "wall_clock_breakdown": false,
    "steps_per_print": 10
}
```

### 针对不同硬件的优化建议

#### RTX 4090 (24GB)
```json
{
    "zero_optimization": {
        "stage": 2,
        "cpu_offload": false
    },
    "fp16": {
        "enabled": true
    }
}
```

#### A100 (80GB)
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        }
    },
    "bf16": {
        "enabled": true
    }
}
```

#### H100 (80GB)
```json
{
    "zero_optimization": {
        "stage": 2,
        "cpu_offload": false
    },
    "bf16": {
        "enabled": true
    },
    "fp8": {
        "enabled": true
    }
}
```

## 📈 最佳实践

### 1. 阶段性训练策略
- **Stage 1**: 使用ZeRO-2，最大化训练速度
- **Stage 2**: 使用ZeRO-3，支持更大批大小
- **Stage 3**: 使用ZeRO-3 + CPU卸载，最大化内存效率

### 2. 批大小调优
```bash
# 从小批大小开始，逐步增加
for batch_size in 1 2 4 8; do
    bash scripts/train_deepspeed.sh \
        --stage stage1 \
        --batch_size $batch_size \
        --model_path models/Qwen_Qwen2-7B \
        --data_path data/stage1_pretraining/train.jsonl \
        --output_dir checkpoints/test-bs-$batch_size
done
```

### 3. 学习率调度
- **预训练**: 使用余弦退火调度
- **微调**: 使用线性warmup + 余弦退火
- **DPO**: 使用较小的固定学习率

### 4. 检查点管理
```bash
# 定期保存检查点
--save_steps 500
--save_total_limit 3

# 启用梯度检查点以节省内存
--gradient_checkpointing True
```

## 🔧 故障排除

### 常见错误及解决方案

#### 1. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 减少`per_device_train_batch_size`
- 增加`gradient_accumulation_steps`
- 启用CPU卸载
- 使用ZeRO-3配置

#### 2. DeepSpeed初始化失败
```
AssertionError: DeepSpeed requires a distributed backend
```
**解决方案**:
```bash
# 确保使用deepspeed命令启动
deepspeed --num_gpus=4 src/trainer.py ...

# 而不是
python src/trainer.py ...
```

#### 3. 模型保存失败
```
RuntimeError: Cannot save model with ZeRO stage 3
```
**解决方案**:
在DeepSpeed配置中添加：
```json
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

#### 4. 训练速度慢
**解决方案**:
- 禁用CPU卸载（如果内存足够）
- 使用更少的激活检查点
- 调整通信后端设置

## 📚 参考资料

- [DeepSpeed官方文档](https://deepspeed.readthedocs.io/)
- [ZeRO论文](https://arxiv.org/abs/1910.02054)
- [Transformers + DeepSpeed集成指南](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Qwen2-Audio论文](https://arxiv.org/abs/2407.10759)

## 🤝 贡献

如果您在使用DeepSpeed训练过程中遇到问题或有改进建议，欢迎：
1. 提交Issue报告问题
2. 创建Pull Request贡献代码
3. 分享您的训练经验和配置

---

**注意**: DeepSpeed训练需要较新的CUDA版本和驱动程序。建议使用CUDA 11.8+和相应的PyTorch版本。 