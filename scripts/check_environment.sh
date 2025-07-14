#!/bin/bash

# Quick Environment Check for Qwen2-Audio Training
# 快速检查Qwen2-Audio训练环境

set -e

# Default conda environment name
CONDA_ENV_NAME="qwen2-audio-multinode"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --conda-env ENV_NAME   Conda environment name to check (default: qwen2-audio-multinode)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🔍 Qwen2-Audio Environment Check"
echo "================================="

# Check conda environment
if command -v conda &> /dev/null; then
    echo "✅ Conda found: $(conda --version)"
    
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo "✅ Environment '$CONDA_ENV_NAME' exists"
        
        # Activate environment and check packages
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV_NAME"
        
        if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
            echo "✅ Successfully activated environment: $CONDA_ENV_NAME"
            
            # Check Python and packages
            echo ""
            echo "📦 Package Versions:"
            echo "-------------------"
            python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

packages = ['torch', 'transformers', 'deepspeed', 'librosa', 'soundfile', 'gradio']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: not installed')
"
            
            # Check CUDA
            echo ""
            echo "🚀 CUDA Status:"
            echo "--------------"
            python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available')
"
            
            # Check distributed backends
            echo ""
            echo "🌐 Distributed Backends:"
            echo "----------------------"
            python -c "
import torch.distributed as dist
backends = ['nccl', 'gloo', 'mpi']
for backend in backends:
    available = getattr(dist, f'is_{backend}_available')()
    status = '✅' if available else '❌'
    print(f'{status} {backend}: {available}')
"
            
        else
            echo "❌ Failed to activate environment: $CONDA_ENV_NAME"
            exit 1
        fi
    else
        echo "❌ Environment '$CONDA_ENV_NAME' not found"
        echo "Available environments:"
        conda env list
        echo ""
        echo "💡 To create the environment:"
        echo "   bash scripts/setup_multinode_env.sh --method conda --env-name $CONDA_ENV_NAME"
        exit 1
    fi
else
    echo "❌ Conda not found"
    echo "💡 Install Miniconda or Anaconda first"
    exit 1
fi

echo ""
echo "🎉 Environment check completed!"
echo ""
echo "💡 Usage examples:"
echo "   # Single node training:"
echo "   bash scripts/train_pretrain.sh --model qwen2_0.5b"
echo ""
echo "   # Multi-node training:"
echo "   bash scripts/train_pretrain_multinode.sh --model qwen2_7b --nnodes 4 --node-rank 0" 