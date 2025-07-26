#!/bin/bash

# LlamaAudio Distributed Training Script
# Supports multi-GPU and multi-node training with DeepSpeed

set -e

# Default configuration
NUM_GPUS=8
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
CONFIG_FILE="configs/training_config.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_config.json"
TRAIN_DATA_PATH="data/train.jsonl"
VAL_DATA_PATH="data/val.jsonl"
OUTPUT_DIR="outputs/llama_audio_training"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num_nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --train_data_path)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --val_data_path)
            VAL_DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus NUM          Number of GPUs per node (default: 8)"
            echo "  --num_nodes NUM         Number of nodes (default: 1)"
            echo "  --node_rank RANK        Rank of current node (default: 0)"
            echo "  --master_addr ADDR      Master node address (default: localhost)"
            echo "  --master_port PORT      Master node port (default: 29500)"
            echo "  --config FILE           Training config file (default: configs/training_config.yaml)"
            echo "  --deepspeed FILE        DeepSpeed config file (default: configs/deepspeed_config.json)"
            echo "  --train_data_path PATH  Training data path (default: data/train.jsonl)"
            echo "  --val_data_path PATH    Validation data path (default: data/val.jsonl)"
            echo "  --output_dir DIR        Output directory (default: outputs/llama_audio_training)"
            echo "  --model_name NAME       Llama model name (default: meta-llama/Llama-3.3-70B-Instruct)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Calculate total processes
TOTAL_PROCESSES=$((NUM_GPUS * NUM_NODES))

echo "========================================"
echo "LlamaAudio Distributed Training"
echo "========================================"
echo "Configuration:"
echo "  Number of GPUs per node: $NUM_GPUS"
echo "  Number of nodes: $NUM_NODES"
echo "  Total processes: $TOTAL_PROCESSES"
echo "  Current node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Config file: $CONFIG_FILE"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"
echo "  Model: $MODEL_NAME"
echo "  Train data: $TRAIN_DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================"

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
    exit 1
fi

if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "Warning: Training data file not found: $TRAIN_DATA_PATH"
fi

# Set environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$TOTAL_PROCESSES
export NODE_RANK=$NODE_RANK
export NCCL_DEBUG=INFO

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training with DeepSpeed
echo "Starting training..."

if [ "$NUM_NODES" -eq 1 ]; then
    # Single node training
    deepspeed --num_gpus=$NUM_GPUS \
              scripts/train.py \
              --config "$CONFIG_FILE" \
              --deepspeed "$DEEPSPEED_CONFIG" \
              --train_data_path "$TRAIN_DATA_PATH" \
              --val_data_path "$VAL_DATA_PATH" \
              --output_dir "$OUTPUT_DIR" \
              --llama_model_name "$MODEL_NAME"
else
    # Multi-node training
    deepspeed --num_gpus=$NUM_GPUS \
              --num_nodes=$NUM_NODES \
              --node_rank=$NODE_RANK \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              scripts/train.py \
              --config "$CONFIG_FILE" \
              --deepspeed "$DEEPSPEED_CONFIG" \
              --train_data_path "$TRAIN_DATA_PATH" \
              --val_data_path "$VAL_DATA_PATH" \
              --output_dir "$OUTPUT_DIR" \
              --llama_model_name "$MODEL_NAME"
fi

echo "Training completed!"

# Copy important files to output directory
echo "Copying configuration files to output directory..."
cp "$CONFIG_FILE" "$OUTPUT_DIR/"
cp "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/"

echo "All files saved to: $OUTPUT_DIR"
echo "TensorBoard logs available at: $OUTPUT_DIR/tensorboard"
echo ""
echo "To view training logs with TensorBoard, run:"
echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard" 