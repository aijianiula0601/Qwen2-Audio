#!/bin/bash

# Complete LlamaAudio Training Pipeline
# This script runs all three stages of LlamaAudio training sequentially

set -e

# Configuration
NUM_GPUS=8
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
SKIP_DATA_DOWNLOAD=false
SKIP_STAGE1=false
SKIP_STAGE2=false
SKIP_STAGE3=false

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
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --skip_data_download)
            SKIP_DATA_DOWNLOAD=true
            shift
            ;;
        --skip_stage1)
            SKIP_STAGE1=true
            shift
            ;;
        --skip_stage2)
            SKIP_STAGE2=true
            shift
            ;;
        --skip_stage3)
            SKIP_STAGE3=true
            shift
            ;;
        --help)
            echo "Complete LlamaAudio Training Pipeline"
            echo ""
            echo "This script runs all three stages of training:"
            echo "  Stage 1: Audio-Text Alignment"
            echo "  Stage 2: Multimodal Understanding"
            echo "  Stage 3: Instruction Tuning"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus NUM          Number of GPUs per node (default: 8)"
            echo "  --num_nodes NUM         Number of nodes (default: 1)"
            echo "  --node_rank RANK        Rank of current node (default: 0)"
            echo "  --master_addr ADDR      Master node address (default: localhost)"
            echo "  --master_port PORT      Master node port (default: 29500)"
            echo "  --model_name NAME       Llama model name (default: meta-llama/Llama-3.3-70B-Instruct)"
            echo "  --skip_data_download    Skip data download steps"
            echo "  --skip_stage1           Skip Stage 1 training"
            echo "  --skip_stage2           Skip Stage 2 training"
            echo "  --skip_stage3           Skip Stage 3 training"
            echo "  --help                  Show this help message"
            echo ""
            echo "Hardware Requirements:"
            echo "  - Minimum: 4x GPU with 24GB VRAM each"
            echo "  - Recommended: 8x GPU with 40GB+ VRAM each"
            echo "  - For Llama3.3-70B: Multi-node setup recommended"
            echo ""
            echo "Estimated Training Time:"
            echo "  - Stage 1: 2-4 hours (8 GPUs)"
            echo "  - Stage 2: 4-8 hours (8 GPUs)"
            echo "  - Stage 3: 1-2 hours (8 GPUs)"
            echo "  - Total: 7-14 hours"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p logs
PIPELINE_LOG="logs/full_training_pipeline.log"

echo "=========================================="
echo "🚀 LlamaAudio Complete Training Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  GPUs per node: $NUM_GPUS"
echo "  Number of nodes: $NUM_NODES"
echo "  Node rank: $NODE_RANK"
echo "  Master address: $MASTER_ADDR"
echo ""
echo "Training Stages:"
echo "  $([ "$SKIP_STAGE1" = true ] && echo "○" || echo "●") Stage 1: Audio-Text Alignment"
echo "  $([ "$SKIP_STAGE2" = true ] && echo "○" || echo "●") Stage 2: Multimodal Understanding"
echo "  $([ "$SKIP_STAGE3" = true ] && echo "○" || echo "●") Stage 3: Instruction Tuning"
echo ""
echo "Data Download: $([ "$SKIP_DATA_DOWNLOAD" = true ] && echo "Skipped" || echo "Enabled")"
echo "=========================================="

# Log start time
echo "$(date): Starting LlamaAudio training pipeline" | tee -a "$PIPELINE_LOG"

# Stage 0: Data Download and Preparation
if [ "$SKIP_DATA_DOWNLOAD" = false ]; then
    echo ""
    echo "📥 Stage 0: Data Download and Preparation"
    echo "=========================================="
    echo "$(date): Starting data download" | tee -a "$PIPELINE_LOG"
    
    echo "Downloading Stage 1 data (Audio-Text Alignment)..."
    if bash scripts/download_data/download_stage1_data.sh; then
        echo "✓ Stage 1 data download completed"
        echo "$(date): Stage 1 data download completed" | tee -a "$PIPELINE_LOG"
    else
        echo "✗ Stage 1 data download failed"
        echo "$(date): Stage 1 data download failed" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
    
    echo "Downloading Stage 2 data (Multimodal Understanding)..."
    if bash scripts/download_data/download_stage2_data.sh; then
        echo "✓ Stage 2 data download completed"
        echo "$(date): Stage 2 data download completed" | tee -a "$PIPELINE_LOG"
    else
        echo "✗ Stage 2 data download failed"
        echo "$(date): Stage 2 data download failed" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
    
    echo "Downloading Stage 3 data (Instruction Tuning)..."
    if bash scripts/download_data/download_stage3_data.sh; then
        echo "✓ Stage 3 data download completed"
        echo "$(date): Stage 3 data download completed" | tee -a "$PIPELINE_LOG"
    else
        echo "✗ Stage 3 data download failed"
        echo "$(date): Stage 3 data download failed" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
    
    echo "✅ All data download completed!"
    
else
    echo "📥 Stage 0: Skipping data download (as requested)"
    echo "$(date): Data download skipped" | tee -a "$PIPELINE_LOG"
fi

# Stage 1: Audio-Text Alignment
if [ "$SKIP_STAGE1" = false ]; then
    echo ""
    echo "🎵 Stage 1: Audio-Text Alignment Training"
    echo "=========================================="
    echo "$(date): Starting Stage 1 training" | tee -a "$PIPELINE_LOG"
    
    if bash scripts/run_stage1_training.sh \
        --num_gpus "$NUM_GPUS" \
        --num_nodes "$NUM_NODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        --model_name "$MODEL_NAME"; then
        
        echo "✅ Stage 1 training completed successfully!"
        echo "$(date): Stage 1 training completed" | tee -a "$PIPELINE_LOG"
    else
        echo "❌ Stage 1 training failed!"
        echo "$(date): Stage 1 training failed" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
else
    echo ""
    echo "🎵 Stage 1: Skipping Audio-Text Alignment (as requested)"
    echo "$(date): Stage 1 training skipped" | tee -a "$PIPELINE_LOG"
fi

# Stage 2: Multimodal Understanding
if [ "$SKIP_STAGE2" = false ]; then
    echo ""
    echo "🧠 Stage 2: Multimodal Understanding Training"
    echo "=========================================="
    echo "$(date): Starting Stage 2 training" | tee -a "$PIPELINE_LOG"
    
    if bash scripts/run_stage2_training.sh \
        --num_gpus "$NUM_GPUS" \
        --num_nodes "$NUM_NODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port $((MASTER_PORT + 1)) \
        --model_name "$MODEL_NAME"; then
        
        echo "✅ Stage 2 training completed successfully!"
        echo "$(date): Stage 2 training completed" | tee -a "$PIPELINE_LOG"
    else
        echo "❌ Stage 2 training failed!"
        echo "$(date): Stage 2 training failed" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
else
    echo ""
    echo "🧠 Stage 2: Skipping Multimodal Understanding (as requested)"
    echo "$(date): Stage 2 training skipped" | tee -a "$PIPELINE_LOG"
fi

# Stage 3: Instruction Tuning
if [ "$SKIP_STAGE3" = false ]; then
    echo ""
    echo "💬 Stage 3: Instruction Tuning"
    echo "=========================================="
    echo "$(date): Starting Stage 3 training" | tee -a "$PIPELINE_LOG"
    
    if bash scripts/run_stage3_training.sh \
        --num_gpus "$NUM_GPUS" \
        --num_nodes "$NUM_NODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port $((MASTER_PORT + 2)) \
        --model_name "$MODEL_NAME"; then
        
        echo "✅ Stage 3 training completed successfully!"
        echo "$(date): Stage 3 training completed" | tee -a "$PIPELINE_LOG"
    else
        echo "❌ Stage 3 training failed!"
        echo "$(date): Stage 3 training failed" | tee -a "$PIPELINE_LOG"
        exit 1
    fi
else
    echo ""
    echo "💬 Stage 3: Skipping Instruction Tuning (as requested)"
    echo "$(date): Stage 3 training skipped" | tee -a "$PIPELINE_LOG"
fi

# Training Pipeline Completed
echo ""
echo "🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉"
echo "🎉 LlamaAudio Training Pipeline Completed! 🎉"
echo "🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉"
echo ""
echo "Training Summary:"
echo "  ✓ Stage 1: Audio-Text Alignment $([ "$SKIP_STAGE1" = true ] && echo "(Skipped)" || echo "(Completed)")"
echo "  ✓ Stage 2: Multimodal Understanding $([ "$SKIP_STAGE2" = true ] && echo "(Skipped)" || echo "(Completed)")"
echo "  ✓ Stage 3: Instruction Tuning $([ "$SKIP_STAGE3" = true ] && echo "(Skipped)" || echo "(Completed)")"
echo ""
echo "Final Model Location:"
echo "  📁 outputs/stage3_instruction/final_model/"
echo ""
echo "Model Capabilities:"
echo "  🎵 Audio understanding and analysis"
echo "  ❓ Audio-based question answering"
echo "  📝 Instruction following with audio"
echo "  💬 Conversational audio interaction"
echo ""
echo "Next Steps:"
echo "1. 📊 Evaluate your model:"
echo "   python scripts/evaluate.py --model_path outputs/stage3_instruction/final_model"
echo ""
echo "2. 🎮 Try interactive demo:"
echo "   python scripts/demo.py --model_path outputs/stage3_instruction/final_model"
echo ""
echo "3. 📈 View training logs:"
echo "   tensorboard --logdir outputs/stage3_instruction/tensorboard"
echo ""
echo "4. 🚀 Deploy your model for production use!"
echo ""

# Log completion
echo "$(date): LlamaAudio training pipeline completed successfully" | tee -a "$PIPELINE_LOG"

# Display timing information
END_TIME=$(date)
echo "Training completed at: $END_TIME"
echo "Check the complete log at: $PIPELINE_LOG"
echo ""
echo "🎊 Congratulations! Your LlamaAudio model is ready! 🎊" 