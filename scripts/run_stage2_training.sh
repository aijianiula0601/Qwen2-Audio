#!/bin/bash

# Stage 2: Multimodal Pretraining Script
# This script runs the second stage of LlamaAudio training

set -e

# Default configuration
NUM_GPUS=8
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29501
CONFIG_FILE="configs/stage2_training_config.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_zero3_config.json"
OUTPUT_DIR="outputs/stage2_multimodal"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
STAGE1_CHECKPOINT="outputs/stage1_alignment/final_model"

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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --stage1_checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        --help)
            echo "Stage 2 Training Script - Multimodal Pretraining"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus NUM          Number of GPUs per node (default: 8)"
            echo "  --num_nodes NUM         Number of nodes (default: 1)"
            echo "  --node_rank RANK        Rank of current node (default: 0)"
            echo "  --master_addr ADDR      Master node address (default: localhost)"
            echo "  --master_port PORT      Master node port (default: 29501)"
            echo "  --config FILE           Training config file (default: configs/stage2_training_config.yaml)"
            echo "  --deepspeed FILE        DeepSpeed config file (default: configs/deepspeed_zero3_config.json)"
            echo "  --output_dir DIR        Output directory (default: outputs/stage2_multimodal)"
            echo "  --model_name NAME       Llama model name (default: meta-llama/Llama-3.3-70B-Instruct)"
            echo "  --stage1_checkpoint DIR Stage 1 checkpoint path (default: outputs/stage1_alignment/final_model)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Stage 2 Focus: Multimodal Understanding"
            echo "- Trains LLM with LoRA on audio understanding tasks"
            echo "- Freezes audio encoder (from Stage 1)"
            echo "- Learns audio classification, QA, and reasoning"
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

echo "=========================================="
echo "LlamaAudio Stage 2: Multimodal Pretraining"
echo "=========================================="
echo "Configuration:"
echo "  Stage: 2 (Multimodal Understanding)"
echo "  Number of GPUs per node: $NUM_GPUS"
echo "  Number of nodes: $NUM_NODES"
echo "  Total processes: $TOTAL_PROCESSES"
echo "  Current node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Config file: $CONFIG_FILE"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"
echo "  Model: $MODEL_NAME"
echo "  Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Training Focus:"
echo "  ✓ LLM training with LoRA"
echo "  ✓ Audio understanding tasks"
echo "  ✓ Cross-modal attention"
echo "  ✗ Audio encoder (frozen from Stage 1)"
echo "=========================================="

# Check if required files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
    exit 1
fi

# Check for Stage 1 checkpoint
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "Error: Stage 1 checkpoint not found: $STAGE1_CHECKPOINT"
    echo "Please complete Stage 1 training first:"
    echo "  bash scripts/run_stage1_training.sh"
    exit 1
fi

# Check for Stage 2 data
STAGE2_TRAIN_DATA="data/stage2_multimodal/processed"
if [ ! -d "$STAGE2_TRAIN_DATA" ]; then
    echo "Warning: Stage 2 training data not found: $STAGE2_TRAIN_DATA"
    echo "Please run: bash scripts/download_data/download_stage2_data.sh"
    echo "Then process the data before training."
fi

# Set environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$TOTAL_PROCESSES
export NODE_RANK=$NODE_RANK
export NCCL_DEBUG=INFO

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log training configuration
echo "Starting Stage 2 training..."
echo "$(date): Stage 2 training started" >> "$OUTPUT_DIR/training.log"
echo "Using Stage 1 checkpoint: $STAGE1_CHECKPOINT" >> "$OUTPUT_DIR/training.log"

# Launch training with DeepSpeed
if [ "$NUM_NODES" -eq 1 ]; then
    # Single node training
    deepspeed --num_gpus=$NUM_GPUS \
              scripts/train.py \
              --config "$CONFIG_FILE" \
              --deepspeed "$DEEPSPEED_CONFIG" \
              --output_dir "$OUTPUT_DIR" \
              --llama_model_name "$MODEL_NAME" \
              --resume_from_checkpoint "$STAGE1_CHECKPOINT"
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
              --output_dir "$OUTPUT_DIR" \
              --llama_model_name "$MODEL_NAME" \
              --resume_from_checkpoint "$STAGE1_CHECKPOINT"
fi

# Check training completion
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 2 Training Completed Successfully!"
    echo "=========================================="
    echo "$(date): Stage 2 training completed successfully" >> "$OUTPUT_DIR/training.log"
    
    # Copy configuration files to output directory
    echo "Copying configuration files..."
    cp "$CONFIG_FILE" "$OUTPUT_DIR/"
    cp "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/"
    
    echo ""
    echo "Output files:"
    echo "  Model: $OUTPUT_DIR/final_model/"
    echo "  Logs: $OUTPUT_DIR/tensorboard/"
    echo "  Config: $OUTPUT_DIR/$(basename $CONFIG_FILE)"
    echo ""
    echo "Training Progress:"
    echo "  ✓ Stage 1: Audio-Text Alignment"
    echo "  ✓ Stage 2: Multimodal Understanding"
    echo "  ○ Stage 3: Instruction Tuning (next)"
    echo ""
    echo "Next Steps:"
    echo "1. Download Stage 3 data: bash scripts/download_data/download_stage3_data.sh"
    echo "2. Run Stage 3 training: bash scripts/run_stage3_training.sh"
    echo ""
    echo "To view training logs:"
    echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard"
    
else
    echo ""
    echo "=========================================="
    echo "Stage 2 Training Failed!"
    echo "=========================================="
    echo "$(date): Stage 2 training failed" >> "$OUTPUT_DIR/training.log"
    echo "Check the error messages above and training logs."
    exit 1
fi 