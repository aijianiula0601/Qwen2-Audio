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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_TYPE     Model type (qwen2_7b, qwen2_70b, llama3_8b, qwen2_0.5b)"
            echo "  --config CONFIG_FILE   Path to base config file"
            echo "  --gpus NUM_GPUS        Number of GPUs to use"
            echo "  --port PORT            Master port for distributed training"
            echo "  --resume CHECKPOINT    Resume from checkpoint"
            echo "  -h, --help             Show this help message"
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