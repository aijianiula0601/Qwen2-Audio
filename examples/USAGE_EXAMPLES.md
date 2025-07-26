# LlamaAudio Usage Examples

This document provides practical examples of how to use the LlamaAudio framework for various audio-language tasks.

## 🚀 Quick Start Examples

### 1. Run Complete Training Pipeline

The easiest way to get started:

```bash
# Interactive quick start (recommended for beginners)
bash quick_start.sh

# Or run complete training automatically
bash scripts/train_all_stages.sh --auto_continue

# Monitor training progress
tensorboard --logdir outputs/ --port 6006
```

### 2. Stage-by-Stage Training

If you want more control over the training process:

```bash
# Stage 1: Audio-Text Alignment
bash scripts/data_download/download_stage1_data.sh
bash scripts/train_stage1.sh

# Wait for Stage 1 to complete, then:
bash scripts/data_download/download_stage2_data.sh  
bash scripts/train_stage2.sh

# Finally:
bash scripts/data_download/download_stage3_data.sh
bash scripts/train_stage3.sh
```

## 📊 Data Preparation Examples

### Custom Dataset Format

Prepare your own training data in JSONL format:

```json
{"audio_path": "/path/to/audio1.wav", "instruction": "请描述这段音频", "output": "这是一段钢琴音乐..."}
{"audio_path": "/path/to/audio2.wav", "instruction": "转录这段语音", "output": "Hello, this is a test recording..."}
```

### Data Preprocessing

```bash
# Convert audio files to required format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Batch conversion
for file in *.mp3; do
    ffmpeg -i "$file" -ar 16000 -ac 1 "${file%.mp3}.wav"
done
```

## 🔧 Configuration Examples

### Custom Model Configuration

Edit `configs/stage1_training_config.yaml`:

```yaml
model:
  llama_model_name: "meta-llama/Llama-3.1-8B-Instruct"  # Smaller model
  whisper_model_name: "openai/whisper-medium"           # Faster audio encoder
  
training:
  batch_size: 4          # Adjust for your GPU memory
  learning_rate: 1e-5    # Lower learning rate
  num_epochs: 2          # Fewer epochs for testing

hardware:
  use_fp16: true         # Enable mixed precision
  gradient_checkpointing: true  # Save memory
```

### Multi-Node Training Configuration

For distributed training across multiple machines:

```bash
# Machine 1 (master node)
bash scripts/train_all_stages.sh \
  --num_nodes 4 \
  --node_rank 0 \
  --master_addr "192.168.1.100" \
  --num_gpus 8 \
  --auto_continue

# Machine 2
bash scripts/train_all_stages.sh \
  --num_nodes 4 \
  --node_rank 1 \
  --master_addr "192.168.1.100" \
  --num_gpus 8 \
  --auto_continue

# Machines 3 and 4 (increment node_rank)
```

## 🎮 Interactive Demo Examples

### Basic Demo Usage

```bash
# Start interactive demo
python scripts/interactive_demo.py \
  --model_path outputs/stage3_conversation/final_model \
  --port 7860

# With public sharing
python scripts/interactive_demo.py \
  --model_path outputs/stage3_conversation/final_model \
  --port 7860 \
  --share
```

### Programmatic Model Usage

```python
from llama_audio.models.llama_audio_model import LlamaAudioModel
import librosa

# Load trained model
model = LlamaAudioModel.from_pretrained(
    "outputs/stage3_conversation/final_model"
)

# Load and process audio
audio, sr = librosa.load("test_audio.wav", sr=16000)

# Generate response
response = model.generate(
    audio=audio,
    instruction="请描述这段音频的内容",
    max_length=256,
    temperature=0.7
)

print(f"Model response: {response}")
```

## 📈 Evaluation Examples

### Comprehensive Evaluation

```bash
# Evaluate on multiple tasks
python scripts/evaluate.py \
  --model_path outputs/stage3_conversation/final_model \
  --test_data_dir data/test/ \
  --output_dir evaluation_results

# Single task evaluation
python scripts/evaluate.py \
  --model_path outputs/stage3_conversation/final_model \
  --test_data_path examples/sample_test_data.jsonl \
  --task audio_captioning \
  --output_dir test_evaluation
```

### Custom Evaluation Metrics

```python
# Custom evaluation script
from scripts.evaluate import LlamaAudioEvaluator

evaluator = LlamaAudioEvaluator(
    model_path="outputs/stage3_conversation/final_model"
)

# Test single audio file
response = evaluator.generate_response(
    audio_path="test_audio.wav",
    instruction="What instruments do you hear?"
)

print(f"Response: {response}")
```

## 🛠️ Advanced Usage Examples

### Memory-Optimized Training

For limited GPU memory:

```bash
# Use smaller batch size and gradient accumulation
bash scripts/train_stage1.sh \
  --config configs/stage1_training_config.yaml

# Edit config to include:
# batch_size: 2
# gradient_accumulation_steps: 8
# use_deepspeed: true
```

### Resume from Checkpoint

```bash
# Resume training from specific checkpoint
bash scripts/train_stage2.sh \
  --stage1_checkpoint "outputs/stage1_alignment/checkpoint-1000"

# Resume specific stage training
python scripts/train.py \
  --config configs/stage2_training_config.yaml \
  --resume_from_checkpoint "outputs/stage2_understanding/checkpoint-500"
```

### Custom Data Splitting

```python
# Split your dataset for training/validation
import json
import random

# Load your data
with open('your_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Split 80/20
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Save splits
with open('train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('val.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

## 🎯 Task-Specific Examples

### 1. Audio Captioning

```python
# Fine-tune for audio captioning
instruction_templates = [
    "请描述这段音频的内容。",
    "这段音频中包含什么？",
    "描述你听到的声音。",
    "分析这段音频的特征。"
]

# Generate training data
training_examples = []
for audio_path, caption in audio_caption_pairs:
    for template in instruction_templates:
        training_examples.append({
            "audio_path": audio_path,
            "instruction": template,
            "output": caption
        })
```

### 2. Music Analysis

```python
# Music-specific instructions
music_instructions = [
    "这段音乐的风格是什么？",
    "音乐中有哪些乐器？", 
    "描述这段音乐的节奏和旋律。",
    "这段音乐适合什么场景？"
]

# Example usage
response = model.generate(
    audio="classical_music.wav",
    instruction="分析这段古典音乐的特点",
    max_length=200
)
```

### 3. Speech Recognition

```python
# Speech-focused instructions
speech_instructions = [
    "请转录这段语音内容。",
    "这段音频说了什么？",
    "将语音转换为文字。"
]

# Batch processing
import os
speech_dir = "speech_samples/"
for audio_file in os.listdir(speech_dir):
    if audio_file.endswith('.wav'):
        response = model.generate(
            audio=os.path.join(speech_dir, audio_file),
            instruction="请转录这段语音内容。"
        )
        print(f"{audio_file}: {response}")
```

## 🔍 Monitoring and Debugging

### Training Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Training logs
tail -f outputs/stage1_alignment/training.log

# TensorBoard for multiple stages
tensorboard --logdir outputs/ --port 6006 &
tensorboard --logdir outputs/stage1_alignment/tensorboard --port 6007 &
tensorboard --logdir outputs/stage2_understanding/tensorboard --port 6008 &
```

### Debugging Common Issues

```bash
# Check data format
head -5 data/stage1_alignment/train.jsonl

# Validate audio files
python -c "
import librosa
import sys
try:
    audio, sr = librosa.load(sys.argv[1], sr=16000)
    print(f'Audio length: {len(audio)/sr:.2f}s, Sample rate: {sr}')
except Exception as e:
    print(f'Error: {e}')
" your_audio.wav

# Test model loading
python -c "
from llama_audio.models.llama_audio_model import LlamaAudioModel
model = LlamaAudioModel.from_pretrained('outputs/stage3_conversation/final_model')
print('Model loaded successfully!')
"
```

## 📱 Production Deployment Examples

### API Server Setup

```python
# Simple FastAPI server
from fastapi import FastAPI, UploadFile, File
from llama_audio.models.llama_audio_model import LlamaAudioModel
import tempfile
import librosa

app = FastAPI()
model = LlamaAudioModel.from_pretrained("outputs/stage3_conversation/final_model")

@app.post("/analyze_audio")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    instruction: str = "请描述这段音频的内容。"
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Generate response
    response = model.generate(
        audio_path=tmp_path,
        instruction=instruction
    )
    
    return {"response": response}

# Run with: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Batch Processing Script

```python
#!/usr/bin/env python3
# batch_process.py - Process multiple audio files

import os
import json
import argparse
from pathlib import Path
from llama_audio.models.llama_audio_model import LlamaAudioModel

def batch_process(audio_dir, output_file, instruction, model_path):
    # Load model
    model = LlamaAudioModel.from_pretrained(model_path)
    
    results = []
    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")
        
        try:
            response = model.generate(
                audio_path=str(audio_file),
                instruction=instruction
            )
            
            results.append({
                "audio_file": audio_file.name,
                "instruction": instruction,
                "response": response
            })
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_file", default="batch_results.json")
    parser.add_argument("--instruction", default="请描述这段音频的内容。")
    parser.add_argument("--model_path", required=True)
    
    args = parser.parse_args()
    batch_process(args.audio_dir, args.output_file, args.instruction, args.model_path)
```

## 🎉 Success Tips

### 1. **Start Small**
```bash
# Test with minimal configuration first
bash scripts/train_stage1.sh --num_gpus 1 --debug
```

### 2. **Monitor Resources**
```bash
# Keep an eye on disk space and memory
df -h
free -h
nvidia-smi
```

### 3. **Save Checkpoints**
```bash
# Enable frequent checkpointing in configs
save_steps: 500
save_total_limit: 3
```

### 4. **Use Validation Data**
```bash
# Always prepare validation data for monitoring
# Aim for 10-20% of your training data
```

### 5. **Experiment with Hyperparameters**
```yaml
# Try different learning rates
learning_rate: [1e-5, 2e-5, 5e-5]

# Adjust batch sizes based on memory
batch_size: [2, 4, 8]

# Different LoRA configurations
lora_rank: [16, 32, 64]
```

This comprehensive guide should help you get the most out of the LlamaAudio framework. Start with the basic examples and gradually explore more advanced features as you become comfortable with the system. 