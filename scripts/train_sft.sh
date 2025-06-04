#!/bin/bash

# Qwen2-Audio Supervised Fine-Tuning Script
# This script runs the SFT stage of Qwen2-Audio

set -e

# Default configurations
MODEL_TYPE="qwen2_7b"  # qwen2_7b, qwen2_70b, llama3_8b
CONFIG_FILE="configs/base_config.yaml"
MODEL_CONFIG_DIR="configs/models"
NUM_GPUS=8
MASTER_PORT=29501

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
        --pretrain_checkpoint)
            PRETRAIN_CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_TYPE           Model type (qwen2_7b, qwen2_70b, llama3_8b)"
            echo "  --config CONFIG_FILE         Path to base config file"
            echo "  --gpus NUM_GPUS              Number of GPUs to use"
            echo "  --port PORT                  Master port for distributed training"
            echo "  --resume CHECKPOINT          Resume from SFT checkpoint"
            echo "  --pretrain_checkpoint PATH   Path to pretrained checkpoint"
            echo "  -h, --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate model type
case $MODEL_TYPE in
    qwen2_7b|qwen2_70b|llama3_8b)
        ;;
    *)
        echo "Error: Invalid model type '$MODEL_TYPE'"
        echo "Supported types: qwen2_7b, qwen2_70b, llama3_8b"
        exit 1
        ;;
esac

# Set model-specific config
MODEL_CONFIG_FILE="${MODEL_CONFIG_DIR}/${MODEL_TYPE}.yaml"

echo "=== Qwen2-Audio Supervised Fine-Tuning ==="
echo "Model type: $MODEL_TYPE"
echo "Base config: $CONFIG_FILE"
echo "Model config: $MODEL_CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"

# Check if config files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Base config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "Error: Model config file not found: $MODEL_CONFIG_FILE"
    exit 1
fi

# Check if SFT data exists
if [ ! -d "data/sft" ]; then
    echo "Error: SFT data not found. Please run scripts/download_data.sh first."
    exit 1
fi

# Determine checkpoint to use
CHECKPOINT_TO_USE=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    CHECKPOINT_TO_USE="$RESUME_CHECKPOINT"
    echo "Resuming SFT from checkpoint: $RESUME_CHECKPOINT"
elif [ -n "$PRETRAIN_CHECKPOINT" ]; then
    CHECKPOINT_TO_USE="$PRETRAIN_CHECKPOINT"
    echo "Starting SFT from pretrained checkpoint: $PRETRAIN_CHECKPOINT"
else
    echo "Warning: No checkpoint specified. Starting SFT from scratch."
    echo "Consider using --pretrain_checkpoint to load a pretrained model."
fi

# Setup environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
OUTPUT_DIR="outputs/sft_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Build training command
TRAINING_CMD="python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    training/train.py \
    --config $CONFIG_FILE \
    --model_config $MODEL_CONFIG_FILE \
    --stage sft"

# Add checkpoint if specified
if [ -n "$CHECKPOINT_TO_USE" ]; then
    TRAINING_CMD="$TRAINING_CMD --resume_from_checkpoint $CHECKPOINT_TO_USE"
fi

# Log command for debugging
echo "Training command:"
echo "$TRAINING_CMD"

# Create log file
LOG_FILE="${OUTPUT_DIR}/sft.log"
echo "Logging to: $LOG_FILE"

# Run training
echo "Starting supervised fine-tuning..."
eval "$TRAINING_CMD" 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=== SFT completed successfully! ==="
    echo "Model saved to: $OUTPUT_DIR"
    echo "Next step: Run DPO with the SFT checkpoint"
    echo "  bash scripts/train_dpo.sh --model $MODEL_TYPE --sft_checkpoint $OUTPUT_DIR/pytorch_model.bin"
else
    echo "=== SFT failed! ==="
    echo "Check the log file: $LOG_FILE"
    exit 1
fi 