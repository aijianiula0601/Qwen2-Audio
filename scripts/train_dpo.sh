#!/bin/bash

# Qwen2-Audio Direct Preference Optimization Script
# This script runs the DPO stage of Qwen2-Audio

set -e

# Default configurations
MODEL_TYPE="qwen2_7b"  # qwen2_7b, qwen2_70b, llama3_8b
CONFIG_FILE="configs/base_config.yaml"
MODEL_CONFIG_DIR="configs/models"
NUM_GPUS=8
MASTER_PORT=29502

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
        --sft_checkpoint)
            SFT_CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_TYPE         Model type (qwen2_7b, qwen2_70b, llama3_8b)"
            echo "  --config CONFIG_FILE       Path to base config file"
            echo "  --gpus NUM_GPUS            Number of GPUs to use"
            echo "  --port PORT                Master port for distributed training"
            echo "  --resume CHECKPOINT        Resume from DPO checkpoint"
            echo "  --sft_checkpoint PATH      Path to SFT checkpoint"
            echo "  -h, --help                 Show this help message"
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

echo "=== Qwen2-Audio Direct Preference Optimization ==="
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

# Check if DPO data exists
if [ ! -d "data/dpo" ]; then
    echo "Error: DPO data not found. Please run scripts/download_data.sh first."
    exit 1
fi

# Determine checkpoint to use
CHECKPOINT_TO_USE=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    CHECKPOINT_TO_USE="$RESUME_CHECKPOINT"
    echo "Resuming DPO from checkpoint: $RESUME_CHECKPOINT"
elif [ -n "$SFT_CHECKPOINT" ]; then
    CHECKPOINT_TO_USE="$SFT_CHECKPOINT"
    echo "Starting DPO from SFT checkpoint: $SFT_CHECKPOINT"
else
    echo "Warning: No checkpoint specified. Starting DPO from scratch."
    echo "It's highly recommended to use --sft_checkpoint to load an SFT model."
fi

# Setup environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create output directory
OUTPUT_DIR="outputs/dpo_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Build training command
TRAINING_CMD="python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    training/train.py \
    --config $CONFIG_FILE \
    --model_config $MODEL_CONFIG_FILE \
    --stage dpo"

# Add checkpoint if specified
if [ -n "$CHECKPOINT_TO_USE" ]; then
    TRAINING_CMD="$TRAINING_CMD --resume_from_checkpoint $CHECKPOINT_TO_USE"
fi

# Log command for debugging
echo "Training command:"
echo "$TRAINING_CMD"

# Create log file
LOG_FILE="${OUTPUT_DIR}/dpo.log"
echo "Logging to: $LOG_FILE"

# Run training
echo "Starting Direct Preference Optimization..."
eval "$TRAINING_CMD" 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "=== DPO completed successfully! ==="
    echo "Final model saved to: $OUTPUT_DIR"
    echo "Training pipeline complete!"
    echo ""
    echo "You can now use the trained model for inference:"
    echo "  python demo/inference.py --model_path $OUTPUT_DIR"
else
    echo "=== DPO failed! ==="
    echo "Check the log file: $LOG_FILE"
    exit 1
fi 