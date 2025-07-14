#!/bin/bash

# Quick Fix for PyTorch Installation Issues
# 快速修复PyTorch安装问题

set -e

# Default conda environment name
CONDA_ENV_NAME="qwen2-audio-multinode"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

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
            echo "  --conda-env ENV_NAME   Conda environment name (default: qwen2-audio-multinode)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "This script fixes PyTorch installation issues in existing conda environments"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "PyTorch Installation Fix"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install conda first."
    exit 1
fi

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    print_error "Environment '$CONDA_ENV_NAME' not found!"
    print_status "Available environments:"
    conda env list
    exit 1
fi

# Activate environment
print_status "Activating environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME"

# Remove problematic PyTorch packages
print_status "Removing existing PyTorch packages..."
pip uninstall torch torchaudio torchvision -y 2>/dev/null || true

# Install PyTorch via conda
print_status "Installing PyTorch 2.6.0 with CUDA 12.4 support via conda..."
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Verify PyTorch installation
print_status "Verifying PyTorch installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - check GPU drivers')
"

# Install remaining packages from cleaned requirements
print_status "Installing remaining packages (excluding PyTorch)..."

# Create filtered requirements
TEMP_REQ="/tmp/requirements_filtered_fix.txt"
if [ -f "requirements.txt" ]; then
    grep -v -E "^torch==|^torchaudio==|^torchvision==" requirements.txt > "$TEMP_REQ"
    print_status "Installing from filtered requirements:"
    cat "$TEMP_REQ"
    pip install -r "$TEMP_REQ"
    rm -f "$TEMP_REQ"
elif [ -f "requirements_clean.txt" ]; then
    print_status "Installing from requirements_clean.txt..."
    pip install -r requirements_clean.txt
else
    print_status "Installing essential packages manually..."
    pip install deepspeed==0.16.9 transformers==4.52.4 librosa==0.11.0 soundfile==0.13.1 gradio==5.33.0 numpy pandas
fi

# Final verification
print_header "Final Package Verification"
python -c "
import sys
print(f'Python executable: {sys.executable}')
print('')

packages = {
    'torch': 'PyTorch',
    'torchaudio': 'TorchAudio', 
    'transformers': 'Transformers',
    'deepspeed': 'DeepSpeed',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'gradio': 'Gradio',
    'numpy': 'NumPy',
    'pandas': 'Pandas'
}

for pkg, name in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError as e:
        print(f'❌ {name}: not installed - {e}')
"

# Check distributed backends
print_status ""
print_status "Checking distributed backends..."
python -c "
import torch.distributed as dist
backends = ['nccl', 'gloo', 'mpi']
for backend in backends:
    available = getattr(dist, f'is_{backend}_available')()
    status = '✅' if available else '❌'
    print(f'{status} {backend}: {available}')
"

print_header "Fix Complete"
print_status "Environment '$CONDA_ENV_NAME' has been successfully fixed!"
print_status ""
print_status "Next steps:"
print_status "1. Test the environment: bash scripts/check_environment.sh --conda-env $CONDA_ENV_NAME"
print_status "2. Start training: bash scripts/train_pretrain.sh --conda-env $CONDA_ENV_NAME --model qwen2_0.5b"
print_status "3. Multi-node training: bash scripts/train_pretrain_multinode.sh --conda-env $CONDA_ENV_NAME" 