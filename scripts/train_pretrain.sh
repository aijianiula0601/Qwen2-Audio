#!/bin/bash

# Qwen2-Audio Pretraining Script
# This script runs the pretraining stage of Qwen2-Audio

set -e

# Default configurations
MODEL_TYPE="qwen2_0.5b"  # qwen2_7b, qwen2_70b, llama3_8b, qwen2_0.5b
CONFIG_FILE="configs/base_config.yaml"
MODEL_CONFIG_DIR="configs/models"
NUM_GPUS=8
MASTER_PORT=29500

# Environment configurations
CONDA_ENV_NAME="qwen2-audio-multinode"  # Default conda environment name
SKIP_ENV_CHECK=false                   # Whether to skip environment activation

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --skip-env-check)
            SKIP_ENV_CHECK=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_TYPE     Model type (qwen2_7b, qwen2_70b, llama3_8b, qwen2_0.5b)"
            echo "  --config CONFIG_FILE   Path to base config file"
            echo "  --gpus NUM_GPUS        Number of GPUs to use"
            echo "  --port PORT            Master port for distributed training"
            echo "  --resume CHECKPOINT    Resume from checkpoint"
            echo "  --conda-env ENV_NAME   Conda environment name (default: qwen2-audio-multinode)"
            echo "  --skip-env-check       Skip environment activation and checks"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Environment setup and activation (reuse from multinode script)
setup_environment() {
    echo "=== Environment Setup ==="
    
    if [ "$SKIP_ENV_CHECK" = true ]; then
        echo "Skipping environment activation (--skip-env-check specified)"
        return
    fi
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        echo "Found conda, attempting to activate environment: $CONDA_ENV_NAME"
        
        # Initialize conda for this shell
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || {
            echo "Warning: Could not initialize conda. Trying alternative method..."
            # Try alternative conda initialization
            if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
                source "$HOME/miniconda3/etc/profile.d/conda.sh"
            elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
                source "$HOME/anaconda3/etc/profile.d/conda.sh"
            else
                echo "Warning: Could not find conda initialization script"
            fi
        }
        
        # Check if the environment exists
        if conda env list | grep -q "^$CONDA_ENV_NAME "; then
            echo "Activating conda environment: $CONDA_ENV_NAME"
            conda activate "$CONDA_ENV_NAME"
            
            # Verify activation
            if [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV_NAME" ]]; then
                echo "✅ Successfully activated conda environment: $CONDA_ENV_NAME"
                echo "Python path: $(which python)"
                echo "Python version: $(python --version 2>&1)"
            else
                echo "❌ Failed to activate conda environment: $CONDA_ENV_NAME"
                echo "Current environment: $CONDA_DEFAULT_ENV"
                echo "Available environments:"
                conda env list
                exit 1
            fi
        else
            echo "❌ Conda environment '$CONDA_ENV_NAME' not found!"
            echo "Available environments:"
            conda env list
            echo ""
            echo "Please create the environment first:"
            echo "  bash scripts/setup_multinode_env.sh --method conda --env-name $CONDA_ENV_NAME"
            exit 1
        fi
    else
        echo "Conda not found. Checking for virtual environment..."
        
        # Check for virtual environment
        if [ -f "$CONDA_ENV_NAME/bin/activate" ]; then
            echo "Activating virtual environment: $CONDA_ENV_NAME"
            source "$CONDA_ENV_NAME/bin/activate"
            echo "✅ Activated virtual environment: $CONDA_ENV_NAME"
            echo "Python path: $(which python)"
        elif [ -f "activate_env.sh" ]; then
            echo "Using project environment activation script"
            source activate_env.sh
        else
            echo "⚠️  No conda or virtual environment found."
            echo "Using system Python: $(which python)"
            echo "Python version: $(python --version 2>&1)"
            echo ""
            echo "Recommendation: Set up a dedicated environment:"
            echo "  bash scripts/setup_multinode_env.sh --method conda"
        fi
    fi
    
    # Verify key packages
    echo ""
    echo "=== Environment Verification ==="
    python -c "
import sys
print(f'Python executable: {sys.executable}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA version: {torch.version.cuda}')
        print(f'✅ GPU count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    sys.exit(1)

try:
    import deepspeed
    print(f'✅ DeepSpeed: {deepspeed.__version__}')
except ImportError as e:
    print(f'❌ DeepSpeed import failed: {e}')
    sys.exit(1)

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')
    sys.exit(1)
" || {
        echo "❌ Environment verification failed!"
        echo "Please check your Python environment and dependencies."
        exit 1
    }
    
    echo "✅ Environment verification passed!"
}

# Validate model type
case $MODEL_TYPE in
    qwen2_7b|qwen2_70b|llama3_8b|qwen2_0.5b|/home/huangjiahong.dracu/hjh/huggingface/Qwen2-0.5B)
        ;;
    *)
        echo "Error: Invalid model type '$MODEL_TYPE'"
        echo "Supported types: qwen2_7b, qwen2_70b, llama3_8b, qwen2_0.5b"
        exit 1
        ;;
esac

# Set model-specific config
MODEL_CONFIG_FILE="${MODEL_CONFIG_DIR}/${MODEL_TYPE}.yaml"

echo "=== Qwen2-Audio Pretraining ==="
echo "Model type: $MODEL_TYPE"
echo "Base config: $CONFIG_FILE"
echo "Model config: $MODEL_CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
echo "Conda environment: $CONDA_ENV_NAME"

# Setup and verify environment
setup_environment

# Check if config files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Base config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: Model config file not found: $MODEL_CONFIG_FILE"
    exit 1
fi

# Check if data exists
if [ ! -d "data/pretrain" ]; then
    echo "Error: Pretraining data not found. Please run scripts/download_data.sh first."
    exit 1
fi

# Setup environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
OUTPUT_DIR="outputs/pretrain_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Build training command
TRAINING_CMD="deepspeed \
    --num_gpus=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    training/train.py \
    --config $CONFIG_FILE \
    --model_config $MODEL_CONFIG_FILE \
    --stage pretrain"

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAINING_CMD="$TRAINING_CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
    echo "Resuming from checkpoint: $RESUME_CHECKPOINT"
fi

# Log command for debugging
echo "Training command:"
echo "$TRAINING_CMD"

# Create log file
LOG_FILE="${OUTPUT_DIR}/pretrain.log"
echo "Logging to: $LOG_FILE"

# Run training
echo "Starting pretraining..."
eval "$TRAINING_CMD" 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=== Pretraining completed successfully! ==="
    echo "Model saved to: $OUTPUT_DIR"
    echo "Next step: Run SFT with the pretrained checkpoint"
    echo "  bash scripts/train_sft.sh --model $MODEL_TYPE --resume $OUTPUT_DIR/pytorch_model.bin"
else
    echo "=== Pretraining failed! ==="
    echo "Check the log file: $LOG_FILE"
    exit 1
fi 