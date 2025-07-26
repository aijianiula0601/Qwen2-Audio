#!/bin/bash

# LlamaAudio Training Environment Diagnostic Script
# This script checks the system configuration for optimal training

echo "========================================"
echo "LlamaAudio Training Environment Diagnosis"
echo "========================================"

# Check Python and PyTorch
echo "🐍 Python & PyTorch:"
python3 --version
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
"

# Check GPU information
echo ""
echo "🖥️  GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu --format=csv,noheader,nounits
    
    # Check for potential issues
    echo ""
    echo "🔍 GPU Health Check:"
    nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | while read line; do
        gpu_index=$(echo $line | cut -d',' -f1)
        temp=$(echo $line | cut -d',' -f2)
        if [ "$temp" -gt 80 ]; then
            echo "⚠️  GPU $gpu_index temperature is high: ${temp}°C"
        else
            echo "✅ GPU $gpu_index temperature OK: ${temp}°C"
        fi
    done
else
    echo "❌ nvidia-smi not available"
fi

# Check system memory
echo ""
echo "💾 System Memory:"
free -h
total_mem=$(free -g | awk '/^Mem:/{print $2}')
if [ "$total_mem" -lt 100 ]; then
    echo "⚠️  System memory might be insufficient for large models (< 100GB)"
else
    echo "✅ System memory looks sufficient for training"
fi

# Check disk space
echo ""
echo "💽 Disk Space:"
df -h . | tail -1
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$available_space" -lt 100 ]; then
    echo "⚠️  Available disk space might be insufficient (< 100GB)"
else
    echo "✅ Disk space looks sufficient"
fi

# Check required directories and files
echo ""
echo "📁 Required Files Check:"
files_to_check=(
    "configs/stage1_training_config_8b.yaml"
    "configs/deepspeed_config_8b.json"
    "scripts/train.py"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

# Check Python packages
echo ""
echo "📦 Python Packages Check:"
python3 -c "
packages = ['torch', 'transformers', 'deepspeed', 'accelerate', 'datasets', 'librosa', 'soundfile']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg} installed')
    except ImportError:
        print(f'❌ {pkg} missing')
"

# Memory optimization recommendations
echo ""
echo "🚀 Optimization Recommendations:"
echo "  For 8B model training:"
echo "  • Use micro_batch_size_per_gpu = 1"
echo "  • Enable gradient accumulation (8 steps recommended)"
echo "  • Use ZeRO Stage 2 for balance of memory and speed"
echo "  • Enable activation checkpointing"
echo "  • Consider FP16 mixed precision"
echo ""
echo "  If you encounter OOM errors:"
echo "  • Reduce max_audio_length (try 15 or 10 seconds)"
echo "  • Enable CPU offloading in DeepSpeed config"
echo "  • Use ZeRO Stage 3"
echo "  • Switch to 3B model configuration"

echo ""
echo "✅ Diagnosis complete!" 