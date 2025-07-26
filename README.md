# LlamaAudio: Multimodal Audio-Language Model Framework

LlamaAudio是一个基于Llama系列语言模型的多模态音频语言模型框架，能够理解和处理音频内容并生成相应的文本回应。本框架参考了Qwen2-Audio的架构设计，支持各种Llama模型（如Llama3.3-70B等）。

## 🚀 主要特性

- **🦙 支持Llama系列模型**: 兼容Llama3.3-70B等各种Llama模型
- **🎵 音频理解能力**: 基于Whisper的强大音频编码器
- **🎯 三阶段训练策略**: 参考Qwen2-Audio论文的渐进式训练方法
- **📊 自动数据管理**: 自动下载和处理多种音频数据集
- **⚡ 高效训练**: 支持LoRA、DeepSpeed、ZeRO等优化技术
- **📊 分布式训练**: 支持多机多卡大规模训练
- **📈 丰富监控**: 集成TensorBoard日志记录
- **🔧 灵活配置**: 支持YAML配置文件和命令行参数
- **🎮 交互式演示**: 内置Gradio网页界面测试模型

## 📋 系统要求

- Python 3.8+
- CUDA 11.7+ (推荐12.0+)
- 至少16GB GPU内存（推荐32GB+用于大型模型）
- 对于Llama3.3-70B模型，推荐使用多卡或CPU offloading

## 🛠️ 安装

1. **克隆项目**
```bash
git clone <your-repo-url>
cd llama_audio
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **可选：安装优化组件**
```bash
# Flash Attention (推荐，提高训练效率)
pip install flash-attn

# xFormers (内存优化)
pip install xformers

# BitsAndBytes (量化支持)
pip install bitsandbytes
```

## 📁 项目结构

```
llama_audio/
├── llama_audio/              # 主模块
│   ├── models/               # 模型定义
│   │   ├── audio_encoder.py
│   │   ├── multimodal_connector.py
│   │   └── llama_audio_model.py
│   └── training/             # 训练模块
│       ├── trainer.py
│       ├── data_loader.py
│       └── utils.py
├── configs/                  # 配置文件
│   ├── stage1_training_config.yaml    # 阶段1配置
│   ├── stage2_training_config.yaml    # 阶段2配置
│   ├── stage3_training_config.yaml    # 阶段3配置
│   ├── deepspeed_stage1_config.json   # DeepSpeed阶段1
│   ├── deepspeed_stage2_config.json   # DeepSpeed阶段2
│   └── deepspeed_stage3_config.json   # DeepSpeed阶段3
├── scripts/                  # 训练脚本
│   ├── train.py              # 主训练脚本
│   ├── train_all_stages.sh   # 完整训练管道
│   ├── train_stage1.sh       # 阶段1训练
│   ├── train_stage2.sh       # 阶段2训练
│   ├── train_stage3.sh       # 阶段3训练
│   ├── interactive_demo.py   # 交互式演示
│   ├── evaluate.py           # 模型评估
│   └── data_download/        # 数据下载脚本
│       ├── download_stage1_data.sh
│       ├── download_stage2_data.sh
│       └── download_stage3_data.sh
├── docs/                     # 文档
│   └── MULTI_STAGE_TRAINING.md
├── requirements.txt
└── README.md
```

## 🎯 快速开始

### 多阶段训练（推荐方式）

LlamaAudio采用**三阶段渐进式训练策略**，参考Qwen2-Audio论文设计：

1. **Stage 1**: 音频-文本对齐（语音识别）- 6-12小时
2. **Stage 2**: 音频理解与描述 - 12-24小时  
3. **Stage 3**: 指令跟随与对话 - 8-16小时

#### Option 1: 一键完整训练（推荐）
```bash
# 自动运行全部三个阶段
bash scripts/train_all_stages.sh --auto_continue

# 或手动确认每个阶段
bash scripts/train_all_stages.sh
```

#### Option 2: 分阶段训练
```bash
# 阶段1：音频-文本对齐
bash scripts/data_download/download_stage1_data.sh
bash scripts/train_stage1.sh

# 阶段2：音频理解
bash scripts/data_download/download_stage2_data.sh
bash scripts/train_stage2.sh

# 阶段3：指令跟随
bash scripts/data_download/download_stage3_data.sh
bash scripts/train_stage3.sh
```

#### 多机分布式训练
```bash
# 多机多卡训练（推荐用于生产）
bash scripts/train_all_stages.sh \
    --num_nodes 4 \
    --node_rank 0 \
    --master_addr "192.168.1.100" \
    --auto_continue
```

### 训练数据集

训练过程自动下载以下数据集：

**Stage 1 (语音识别)**:
- LibriSpeech (~280K samples, ~50GB)
- CommonVoice (~150K samples, ~15GB)

**Stage 2 (音频理解)**:
- AudioCaps (~230K samples, ~8GB)
- Clotho (~24K samples, ~2GB)  
- WavCaps (~400K samples, ~50GB)
- FSD50K (~51K samples, ~25GB)

**Stage 3 (指令跟随)**:
- 自动生成的指令对话数据 (~405K samples)

### 监控训练进度

```bash
# 启动TensorBoard监控
tensorboard --logdir outputs/ --port 6006

# 查看特定阶段
tensorboard --logdir outputs/stage1_alignment/tensorboard --port 6007
```

### 测试训练结果

```bash
# 启动交互式演示
python scripts/interactive_demo.py \
    --model_path outputs/stage3_conversation/final_model \
    --port 7860 --share

# 模型评估
python scripts/evaluate.py \
    --model_path outputs/stage3_conversation/final_model
```

### 传统训练方式（可选）

如果您想使用传统的单阶段训练：

#### 1. 准备数据

数据格式支持JSONL，每行包含：
```json
{
  "audio_path": "path/to/audio.wav",
  "instruction": "请描述这段音频",
  "output": "这是一段包含钢琴演奏的音乐..."
}
```

#### 2. 配置和训练
```bash
# 编辑配置文件
vim configs/training_config.yaml

# 开始训练
python scripts/train.py --config configs/training_config.yaml
```

**详细的多阶段训练说明请参考：[多阶段训练指南](docs/MULTI_STAGE_TRAINING.md)**

## ⚙️ 高级配置

### DeepSpeed配置

对于大型模型（如Llama3.3-70B），推荐使用ZeRO Stage 3：

```bash
# 使用ZeRO Stage 3配置
bash scripts/run_distributed_training.sh \
    --deepspeed configs/deepspeed_zero3_config.json \
    --model_name "meta-llama/Llama-3.3-70B-Instruct"
```

### LoRA微调

在配置文件中启用LoRA以减少内存使用：

```yaml
model:
  use_lora: true
  lora_rank: 64
  lora_alpha: 16
  freeze_llm: false  # 可以设为false允许LoRA训练
```

### CPU Offloading

对于内存受限的环境，使用CPU offloading：

```json
// deepspeed_zero3_config.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  }
}
```

## 📊 监控训练

### TensorBoard
```bash
tensorboard --logdir outputs/llama_audio_training/tensorboard
```

### 训练日志
训练日志会保存在输出目录下的 `training.log` 文件中。

## 🎮 模型使用

训练完成后，可以这样使用模型：

```python
from llama_audio.models.llama_audio_model import LlamaAudioModel

# 加载训练好的模型
model = LlamaAudioModel.from_pretrained("outputs/llama_audio_training/final_model")

# 音频理解
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
audio_features = model.audio_processor(audio, sampling_rate=sr, return_tensors="pt")

# 生成回应
response = model.generate(
    audio_features=audio_features.input_features,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)

print(model.tokenizer.decode(response[0], skip_special_tokens=True))
```

## 🛠️ 故障排除

### 内存不足
1. 减少批次大小：`batch_size: 2`
2. 启用梯度累积：`gradient_accumulation_steps: 8`
3. 使用DeepSpeed ZeRO Stage 3
4. 启用CPU offloading

### 多卡同步问题
1. 检查NCCL环境：`export NCCL_DEBUG=INFO`
2. 确保所有节点网络连通
3. 检查防火墙设置

### 模型加载失败
1. 确保有Hugging Face访问权限
2. 设置代理：`export HF_ENDPOINT=https://hf-mirror.com`
3. 本地下载模型文件

## 📖 配置参考

### 支持的Llama模型
- `meta-llama/Llama-3.3-70B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`

### 支持的Whisper模型
- `openai/whisper-large-v3`
- `openai/whisper-large-v2`
- `openai/whisper-medium`
- `openai/whisper-small`

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

- 感谢Qwen2-Audio论文提供的架构设计思路
- 感谢Hugging Face Transformers库
- 感谢Microsoft DeepSpeed团队
- 感谢OpenAI Whisper项目

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

**注意**: 本框架需要足够的计算资源来训练大型模型。建议在开始训练前仔细评估硬件需求。 