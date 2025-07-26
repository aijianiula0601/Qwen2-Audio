# LlamaAudio Multi-Stage Training Guide

This document provides a comprehensive guide for training the LlamaAudio model using a three-stage training strategy inspired by Qwen2-Audio.

## 🎯 Training Overview

LlamaAudio follows a progressive three-stage training approach:

1. **Stage 1**: Audio-Text Alignment (Speech Recognition)
2. **Stage 2**: Audio Understanding & Description
3. **Stage 3**: Instruction Following & Conversation

Each stage builds upon the previous one, gradually increasing the model's audio-language capabilities.

## 📊 Training Stages Details

### Stage 1: Audio-Text Alignment Pre-training

**Objective**: Learn basic audio-text alignment through speech recognition tasks.

**Data Sources**:
- LibriSpeech (~280K samples, ~50GB)
- CommonVoice English (~150K samples, ~15GB)
- Total: ~430K samples, ~65GB

**Training Strategy**:
- Freeze LLM parameters
- Train audio encoder and multimodal connector
- Focus on speech recognition and transcription

**Key Files**:
- Config: `configs/stage1_training_config.yaml`
- DeepSpeed: `configs/deepspeed_stage1_config.json`
- Script: `scripts/train_stage1.sh`
- Data: `scripts/data_download/download_stage1_data.sh`

**Expected Duration**: 6-12 hours (depending on hardware)

### Stage 2: Audio Understanding & Description

**Objective**: Learn audio understanding and description capabilities.

**Data Sources**:
- AudioCaps (~230K audio-caption pairs, ~8GB)
- Clotho (~24K audio-caption pairs, ~2GB)
- WavCaps (~400K audio-caption pairs, ~50GB)
- FSD50K (~51K audio-label pairs, ~25GB)
- Total: ~705K samples, ~85GB

**Training Strategy**:
- Keep audio encoder frozen (from Stage 1)
- Train multimodal connector and LLM with LoRA
- Focus on audio description and content understanding

**Key Files**:
- Config: `configs/stage2_training_config.yaml`
- DeepSpeed: `configs/deepspeed_stage2_config.json`
- Script: `scripts/train_stage2.sh`
- Data: `scripts/data_download/download_stage2_data.sh`

**Expected Duration**: 12-24 hours (depending on hardware)

### Stage 3: Instruction Following & Conversation

**Objective**: Learn instruction following and multi-turn conversation capabilities.

**Data Sources**:
- Generated Instructions (~300K samples)
- Multi-turn Conversations (~100K samples)
- Additional Instructions (~5K samples)
- Total: ~405K samples

**Training Strategy**:
- Keep audio encoder frozen
- Continue training multimodal connector and LLM with LoRA
- Focus on instruction following and conversation

**Key Files**:
- Config: `configs/stage3_training_config.yaml`
- DeepSpeed: `configs/deepspeed_stage3_config.json`
- Script: `scripts/train_stage3.sh`
- Data: `scripts/data_download/download_stage3_data.sh`

**Expected Duration**: 8-16 hours (depending on hardware)

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

Run all three stages automatically:

```bash
# Full training pipeline with auto-continuation
bash scripts/train_all_stages.sh --auto_continue

# Or with manual confirmation between stages
bash scripts/train_all_stages.sh
```

### Option 2: Stage-by-Stage Training

Train each stage individually:

```bash
# Stage 1: Audio-Text Alignment
bash scripts/data_download/download_stage1_data.sh
bash scripts/train_stage1.sh

# Stage 2: Audio Understanding
bash scripts/data_download/download_stage2_data.sh
bash scripts/train_stage2.sh

# Stage 3: Instruction Following
bash scripts/data_download/download_stage3_data.sh
bash scripts/train_stage3.sh
```

## 📋 Prerequisites

### Hardware Requirements

**Minimum Configuration**:
- 1 node with 8 GPUs (e.g., 8x A100 40GB)
- 512GB RAM
- 250GB storage

**Recommended Configuration**:
- 2-4 nodes with 8 GPUs each
- 1TB+ RAM per node
- 500GB+ SSD storage

### Software Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## 🔧 Configuration

### Model Configuration

All stages use the same base model configuration but with different training strategies:

```yaml
model:
  llama_model_name: "meta-llama/Llama-3.3-70B-Instruct"
  whisper_model_name: "openai/whisper-large-v3"
  use_lora: true  # Stages 2 & 3
  lora_rank: 64
  lora_alpha: 16
```

### Multi-Node Training

For multi-node training, configure each node:

```bash
# Node 0 (master)
bash scripts/train_all_stages.sh \
  --num_nodes 4 \
  --node_rank 0 \
  --master_addr "10.0.0.1"

# Node 1
bash scripts/train_all_stages.sh \
  --num_nodes 4 \
  --node_rank 1 \
  --master_addr "10.0.0.1"

# Continue for other nodes...
```

## 📊 Data Preparation

### Automatic Data Download

Most datasets can be downloaded automatically:

```bash
# Stage 1
bash scripts/data_download/download_stage1_data.sh

# Stage 2 (includes auto-download script)
bash scripts/data_download/download_stage2_data.sh
cd data/stage2_understanding && python auto_download.py

# Stage 3 (generates from previous stages)
bash scripts/data_download/download_stage3_data.sh
```

### Manual Data Preparation

Some datasets require manual download due to licensing:

1. **CommonVoice**: Register at https://commonvoice.mozilla.org/
2. **Clotho**: Download from https://zenodo.org/record/4783391
3. **FSD50K**: Download from https://zenodo.org/record/4060432

Follow the instructions in the respective download scripts.

### Data Format

All training data uses JSONL format:

```json
{
  "audio_path": "/path/to/audio.wav",
  "instruction": "请描述这段音频的内容。",
  "output": "这段音频包含优美的钢琴音乐...",
  "task": "audio_description"
}
```

## 🎛️ Advanced Training Options

### Custom Model Selection

```bash
# Use different Llama model
bash scripts/train_all_stages.sh \
  --model_name "meta-llama/Llama-2-70B-Chat-hf"

# Use different Whisper model (modify config files)
# whisper_model_name: "openai/whisper-large-v2"
```

### Partial Training

```bash
# Train only specific stages
bash scripts/train_all_stages.sh --start_stage 2 --end_stage 3

# Resume from specific checkpoint
bash scripts/train_stage2.sh \
  --stage1_checkpoint "path/to/custom/checkpoint"
```

### Memory Optimization

```bash
# Use ZeRO-3 with CPU offloading for large models
# Modify deepspeed configs:
# - deepspeed_stage3_config.json for Stage 3
# - Enable CPU offloading for optimizer and parameters

# Cleanup intermediate checkpoints to save space
bash scripts/train_all_stages.sh --cleanup_intermediate
```

## 📈 Monitoring and Evaluation

### TensorBoard Monitoring

```bash
# Launch TensorBoard for all stages
tensorboard --logdir outputs/ --port 6006

# Stage-specific monitoring
tensorboard --logdir outputs/stage1_alignment/tensorboard --port 6007
tensorboard --logdir outputs/stage2_understanding/tensorboard --port 6008
tensorboard --logdir outputs/stage3_conversation/tensorboard --port 6009
```

### Model Evaluation

```bash
# Evaluate final model
python scripts/evaluate.py \
  --model_path outputs/stage3_conversation/final_model \
  --test_data_path data/test_data.jsonl

# Interactive demo
python scripts/interactive_demo.py \
  --model_path outputs/stage3_conversation/final_model \
  --port 7860 --share
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory Errors**:
```bash
# Reduce batch size in config files
# Use gradient accumulation
# Enable DeepSpeed ZeRO-3 with CPU offloading
```

**Data Loading Errors**:
```bash
# Verify data paths in config files
# Check audio file formats and accessibility
# Ensure sufficient disk space
```

**Multi-Node Issues**:
```bash
# Check network connectivity between nodes
# Verify NCCL installation and configuration
# Ensure consistent environments across nodes
```

**Checkpoint Loading Errors**:
```bash
# Verify checkpoint paths
# Check model configuration consistency
# Ensure LoRA state dict compatibility
```

### Performance Optimization

**Training Speed**:
- Use multiple nodes for distributed training
- Enable mixed precision (FP16/BF16)
- Optimize data loading with more workers
- Use faster storage (NVMe SSD)

**Memory Usage**:
- Enable DeepSpeed ZeRO-3
- Use CPU offloading for optimizer/parameters
- Reduce sequence lengths if possible
- Use gradient checkpointing

## 📚 Model Capabilities

After completing all three stages, LlamaAudio will have the following capabilities:

### Audio Understanding
- ✅ Audio transcription and speech recognition
- ✅ Music and sound analysis
- ✅ Environmental audio understanding
- ✅ Audio quality assessment

### Content Generation
- ✅ Audio description and captioning
- ✅ Audio content analysis
- ✅ Emotional interpretation
- ✅ Technical audio assessment

### Interaction
- ✅ Instruction following with audio input
- ✅ Multi-turn conversations about audio
- ✅ Question answering about audio content
- ✅ Audio-guided reasoning

## 🎯 Expected Results

### Performance Metrics

**Stage 1 (Alignment)**:
- Word Error Rate (WER): < 5% on LibriSpeech test-clean
- Character Error Rate (CER): < 3%
- Audio-text alignment quality: High

**Stage 2 (Understanding)**:
- BLEU score: > 20 on audio captioning tasks
- ROUGE-L: > 40 on description tasks
- Semantic similarity: > 0.7

**Stage 3 (Conversation)**:
- Instruction following accuracy: > 85%
- Response relevance: > 90%
- Conversation coherence: High

### Training Time

**Total Training Time**: 26-52 hours (depending on hardware)

**Breakdown**:
- Stage 1: 6-12 hours
- Stage 2: 12-24 hours  
- Stage 3: 8-16 hours

**Hardware Scaling**:
- 8 GPUs (1 node): ~50 hours total
- 16 GPUs (2 nodes): ~30 hours total
- 32 GPUs (4 nodes): ~20 hours total

## 🚀 Deployment

### Model Export

```bash
# Export final model for deployment
python scripts/export_model.py \
  --model_path outputs/stage3_conversation/final_model \
  --output_path deployed_models/llama_audio_final \
  --format "huggingface"
```

### Production Setup

```bash
# Launch production demo
python scripts/interactive_demo.py \
  --model_path deployed_models/llama_audio_final \
  --port 8080 \
  --share

# API server deployment (if implemented)
python scripts/api_server.py \
  --model_path deployed_models/llama_audio_final \
  --host 0.0.0.0 \
  --port 8000
```

## 📄 Additional Resources

- **Main README**: [README.md](../README.md)
- **Model Architecture**: [llama_audio/models/](../llama_audio/models/)
- **Training Scripts**: [scripts/](../scripts/)
- **Configuration Files**: [configs/](../configs/)
- **Example Data**: [examples/](../examples/)

## 🤝 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review configuration files and logs
3. Open an issue on the project repository
4. Consult the Qwen2-Audio paper for methodology details

## 🎉 Conclusion

The LlamaAudio multi-stage training framework provides a robust and efficient approach to training high-quality audio-language models. By following this guide, you can successfully train a production-ready model capable of understanding, describing, and conversing about audio content.

Good luck with your training! 🚀 