# Qwen2-Audio 训练数据和脚本说明

本文档说明了为Qwen2-Audio论文实现而创建的数据下载和训练脚本。

## 📁 文件结构

```
scripts/
├── download_training_data.sh     # 数据下载脚本
├── process_training_data.py      # 数据处理脚本  
├── setup_and_train.sh           # 完整训练流水线
├── train_stage1.sh              # Stage 1 预训练
├── train_stage2.sh              # Stage 2 监督微调
└── train_stage3.sh              # Stage 3 DPO优化

src/
├── model.py                     # 模型实现
├── processor.py                 # 数据处理器
├── trainer.py                   # 训练器实现
└── __init__.py                  # 包初始化

data/
├── raw/                         # 原始数据
├── processed/                   # 处理后数据
├── stage1_pretraining/          # Stage 1 训练数据
├── stage2_sft/                  # Stage 2 训练数据
└── stage3_dpo/                  # Stage 3 训练数据
```

## 🎯 论文中提到的训练数据集

### 根据Qwen2-Audio论文Table 1和相关描述，训练数据包括：

#### Stage 1 预训练数据 (~600k小时)
1. **ASR数据集**
   - LibriSpeech (960h): 英文语音识别
   - Common Voice 15.0 (50k+ h): 102种语言
   - FLEURS (12k h): 102种语言
   - AISHELL-2 (1k h): 中文语音
   - GigaSpeech (10k h): 英文语音

2. **音频理解数据集**
   - AudioCaps (50k clips): 音频描述
   - AudioSet-SL (2M clips): 音频分类
   - MusicCaps (5.5k clips): 音乐描述

3. **语音翻译数据集**
   - CoVoST2 (21种语言对): 语音翻译
   - Multi-lingual LibriSpeech: 多语言语音

#### Stage 2 监督微调数据
1. **对话数据集**
   - Fisher (2k h): 英文对话
   - SpokenWOZ (83h): 语音对话系统
   - IEMOCAP (12h): 情感对话
   
2. **情感和意图识别**
   - MELD (13h): 多模态情感识别
   - DAIC-WOZ: 抑郁症检测对话

3. **音频分类**
   - VocalSound (21k clips): 非语音声音
   - ESC-50: 环境声音分类

#### Stage 3 DPO优化数据
- 人工标注的偏好数据 (论文未详细描述具体来源)

## 🚀 使用方法

### 🎯 超级快速开始 (新手推荐)
```bash
# 使用预设配置一键开始
bash scripts/quick_start.sh
```
提供5种预设配置:
- 🔬 **开发测试**: 0.5B模型 + 小数据集 (2-4小时)
- 🎯 **标准训练**: 7B模型 + 完整数据集 (1-3天)
- 🏆 **高性能**: 72B模型 + 完整数据集 (1-2周)
- ⚡ **MoE高效**: 57B-A14B模型 + 完整数据集 (5-10天)
- 🛠️ **自定义**: 完全自定义配置

### 🔧 完整自定义 (高级用户)
```bash
# 完整配置流水线
bash scripts/setup_and_train.sh
```

运行时会提示选择基础模型:
1. **Qwen/Qwen2-0.5B** (最小模型, 适合测试, 需要4GB显存)
2. **Qwen/Qwen2-1.5B** (小模型, 适合开发, 需要8GB显存)  
3. **Qwen/Qwen2-7B** (推荐, 平衡性能和资源, 需要16GB显存)
4. **Qwen/Qwen2-72B** (大模型, 需要更多GPU, 需要80GB×8显存)
5. **Qwen/Qwen2-57B-A14B** (MoE模型, 高效大模型, 需要48GB×4显存)
6. **自定义模型路径** (支持其他Qwen2系列模型)

### 分步骤执行

#### 1. 下载数据
```bash
bash scripts/download_training_data.sh
```

#### 2. 处理数据
```bash
python scripts/process_training_data.py --data_root data
```

#### 3. 三阶段训练
```bash
# Stage 1: 预训练 (3-7天)
bash scripts/train_stage1.sh

# Stage 2: 监督微调 (1-3天)  
bash scripts/train_stage2.sh

# Stage 3: DPO优化 (6-12小时)
bash scripts/train_stage3.sh
```

## 📊 数据集下载状态

| 数据集 | 状态 | 大小 | 下载方式 | 备注 |
|--------|------|------|----------|------|
| LibriSpeech | ✅ | 60GB | 直接wget | 开源 |
| Common Voice | ⚠️ | 500GB | 需注册 | Mozilla账号 |
| FLEURS | ✅ | 100GB | HuggingFace | 开源 |
| AISHELL-2 | ⚠️ | 50GB | 需申请 | 学术授权 |
| AudioCaps | ✅ | 20GB | HuggingFace | 开源 |
| MusicCaps | ✅ | 5GB | HuggingFace | 开源 |
| CoVoST2 | ✅ | 100GB | HuggingFace | 开源 |
| MELD | ✅ | 10GB | HuggingFace | 开源 |
| VocalSound | ✅ | 5GB | GitHub | 开源 |
| Fisher | ❌ | 100GB | LDC购买 | 需付费 |
| IEMOCAP | ❌ | 12GB | USC申请 | 学术授权 |

**图例**: ✅ 可直接下载 | ⚠️ 需注册申请 | ❌ 需付费或特殊授权

## ⚙️ 训练配置

### 硬件要求

#### 不同模型大小的硬件需求:

| 模型大小 | 最低配置 | 推荐配置 | 显存需求 | 训练时间 |
|----------|----------|----------|----------|----------|
| **0.5B** | 1×RTX 3060 (12GB) | 1×RTX 4060 (16GB) | 4GB | 6-12小时 |
| **1.5B** | 1×RTX 3080 (16GB) | 1×RTX 4070 (20GB) | 8GB | 12-24小时 |
| **7B** | 1×RTX 4090 (24GB) | 2×RTX 4090 (48GB) | 16GB | 1-3天 |
| **72B** | 4×A100 (320GB) | 8×A100 (640GB) | 80GB×4 | 1-2周 |
| **57B-A14B** | 4×A100 (320GB) | 6×A100 (480GB) | 48GB×4 | 5-10天 |

#### 通用要求:
- **内存**: 模型大小 × 4 (如7B模型需要28GB内存)
- **存储**: 2TB+ (数据集 + 检查点)
- **网络**: 稳定连接用于模型下载

### 自动优化特性

脚本会根据选择的模型自动:
- 调整批处理大小和梯度累积
- 设置合适的学习率
- 启用大模型优化 (DeepSpeed ZeRO-3, 梯度检查点)
- 检查硬件兼容性并给出建议

### 训练参数

#### Stage 1 (预训练)
- 学习率: 1e-4
- 批大小: 4×4 (per GPU × gradient accumulation)
- 训练轮数: 3 epochs
- 序列长度: 2048

#### Stage 2 (监督微调)
- 学习率: 5e-5  
- 批大小: 2×8
- 训练轮数: 2 epochs
- 数据格式: 对话格式

#### Stage 3 (DPO)
- 学习率: 5e-6
- 批大小: 1×16
- 训练轮数: 1 epoch
- DPO Beta: 0.1

## 🔍 数据处理详情

### 音频预处理
- 采样率: 16kHz
- 最大时长: 30秒
- 特征提取: Whisper mel-spectrogram
- 池化步长: 2

### 文本格式
```json
{
    "audio_path": "path/to/audio.wav",
    "instruction": "请转录这段音频",
    "input": "",
    "output": "转录文本内容",
    "task_type": "asr|translation|classification|generation",
    "language": "zh|en|fr|de|...",
    "conversation": [
        {"role": "user", "content": "用户输入"},
        {"role": "assistant", "content": "助手回复"}
    ]
}
```

### 特殊标记
- `<|audio_bos|>`: 音频开始标记
- `<|audio_eos|>`: 音频结束标记  
- `<|AUDIO|>`: 音频占位符

## 📝 注意事项

1. **授权要求**: 某些数据集需要申请授权才能使用
2. **存储空间**: 完整数据集需要约2TB存储空间
3. **训练时间**: 完整三阶段训练需要1-2周时间
4. **显存优化**: 显存不足时可调整batch_size和gradient_accumulation
5. **网络要求**: 数据下载需要稳定的网络连接

## 🤝 贡献指南

如果您发现脚本有问题或想要改进，欢迎：
1. 提交Issue报告问题
2. 创建Pull Request贡献代码
3. 完善文档和使用说明

## 📚 参考资料

- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio)
- [训练指南](TRAINING_GUIDE.md)
- [数据集清单](data/dataset_sources.md) 