#!/bin/bash

# Stage 3: Instruction Tuning Script
# This script runs the third and final stage of LlamaAudio training

set -e

# Default configuration
NUM_GPUS=8
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29502
CONFIG_FILE="configs/stage3_training_config.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_zero3_config.json"
OUTPUT_DIR="outputs/stage3_instruction"
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
STAGE2_CHECKPOINT="outputs/stage2_multimodal/final_model"

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
        --stage2_checkpoint)
            STAGE2_CHECKPOINT="$2"
            shift 2
            ;;
        --help)
            echo "Stage 3 Training Script - Instruction Tuning"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus NUM          Number of GPUs per node (default: 8)"
            echo "  --num_nodes NUM         Number of nodes (default: 1)"
            echo "  --node_rank RANK        Rank of current node (default: 0)"
            echo "  --master_addr ADDR      Master node address (default: localhost)"
            echo "  --master_port PORT      Master node port (default: 29502)"
            echo "  --config FILE           Training config file (default: configs/stage3_training_config.yaml)"
            echo "  --deepspeed FILE        DeepSpeed config file (default: configs/deepspeed_zero3_config.json)"
            echo "  --output_dir DIR        Output directory (default: outputs/stage3_instruction)"
            echo "  --model_name NAME       Llama model name (default: meta-llama/Llama-3.3-70B-Instruct)"
            echo "  --stage2_checkpoint DIR Stage 2 checkpoint path (default: outputs/stage2_multimodal/final_model)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Stage 3 Focus: Instruction Following"
            echo "- Fine-tunes the model for instruction following"
            echo "- Enables conversational capabilities"
            echo "- Optimizes response quality and safety"
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
echo "LlamaAudio Stage 3: Instruction Tuning"
echo "=========================================="
echo "Configuration:"
echo "  Stage: 3 (Instruction Following)"
echo "  Number of GPUs per node: $NUM_GPUS"
echo "  Number of nodes: $NUM_NODES"
echo "  Total processes: $TOTAL_PROCESSES"
echo "  Current node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Config file: $CONFIG_FILE"
echo "  DeepSpeed config: $DEEPSPEED_CONFIG"
echo "  Model: $MODEL_NAME"
echo "  Stage 2 checkpoint: $STAGE2_CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Training Focus:"
echo "  ✓ Instruction following optimization"
echo "  ✓ Conversational capabilities"
echo "  ✓ Response quality enhancement"
echo "  ✗ Audio encoder (frozen)"
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

# Check for Stage 2 checkpoint
if [ ! -d "$STAGE2_CHECKPOINT" ]; then
    echo "Error: Stage 2 checkpoint not found: $STAGE2_CHECKPOINT"
    echo "Please complete Stage 2 training first:"
    echo "  bash scripts/run_stage2_training.sh"
    exit 1
fi

# Check for Stage 3 data
STAGE3_TRAIN_DATA="data/stage3_instruction/processed"
if [ ! -d "$STAGE3_TRAIN_DATA" ]; then
    echo "Warning: Stage 3 training data not found: $STAGE3_TRAIN_DATA"
    echo "Please run: bash scripts/download_data/download_stage3_data.sh"
    echo "Data will be automatically generated if not found."
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
echo "Starting Stage 3 training..."
echo "$(date): Stage 3 training started" >> "$OUTPUT_DIR/training.log"
echo "Using Stage 2 checkpoint: $STAGE2_CHECKPOINT" >> "$OUTPUT_DIR/training.log"

# Launch training with DeepSpeed
if [ "$NUM_NODES" -eq 1 ]; then
    # Single node training
    deepspeed --num_gpus=$NUM_GPUS \
              scripts/train.py \
              --config "$CONFIG_FILE" \
              --deepspeed "$DEEPSPEED_CONFIG" \
              --output_dir "$OUTPUT_DIR" \
              --llama_model_name "$MODEL_NAME" \
              --resume_from_checkpoint "$STAGE2_CHECKPOINT"
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
              --resume_from_checkpoint "$STAGE2_CHECKPOINT"
fi

# Check training completion
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Stage 3 Training Completed Successfully!"
    echo "=========================================="
    echo "$(date): Stage 3 training completed successfully" >> "$OUTPUT_DIR/training.log"
    
    # Copy configuration files to output directory
    echo "Copying configuration files..."
    cp "$CONFIG_FILE" "$OUTPUT_DIR/"
    cp "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/"
    
    echo ""
    echo "🎉 LlamaAudio Training Pipeline Completed!"
    echo "=========================================="
    echo "Training Progress:"
    echo "  ✓ Stage 1: Audio-Text Alignment"
    echo "  ✓ Stage 2: Multimodal Understanding"
    echo "  ✓ Stage 3: Instruction Tuning"
    echo ""
    echo "Final Model Output:"
    echo "  Model: $OUTPUT_DIR/final_model/"
    echo "  Logs: $OUTPUT_DIR/tensorboard/"
    echo "  Config: $OUTPUT_DIR/$(basename $CONFIG_FILE)"
    echo ""
    echo "Model Capabilities:"
    echo "  ✓ Audio understanding and analysis"
    echo "  ✓ Audio-based question answering"
    echo "  ✓ Instruction following with audio"
    echo "  ✓ Conversational audio interaction"
    echo ""
    echo "Usage Examples:"
    echo "1. Evaluate the model:"
    echo "   python scripts/evaluate.py --model_path $OUTPUT_DIR/final_model --test_data_path data/test.jsonl"
    echo ""
    echo "2. Interactive demo:"
    echo "   python scripts/demo.py --model_path $OUTPUT_DIR/final_model"
    echo ""
    echo "3. View training logs:"
    echo "   tensorboard --logdir $OUTPUT_DIR/tensorboard"
    echo ""
    echo "Congratulations! Your LlamaAudio model is ready for use! 🚀"
    
else
    echo ""
    echo "=========================================="
    echo "Stage 3 Training Failed!"
    echo "=========================================="
    echo "$(date): Stage 3 training failed" >> "$OUTPUT_DIR/training.log"
    echo "Check the error messages above and training logs."
    exit 1
fi 