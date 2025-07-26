#!/bin/bash

# LlamaAudio Stage 1: Audio-Text Alignment Training Script
# This script runs the first stage of LlamaAudio training focused on audio-text alignment
# Default configuration uses Llama-3.2-3B-Instruct for memory efficiency

set -e

# Default configuration - Optimized for 8B model
NUM_GPUS=8  # 8B model needs more GPUs
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
CONFIG_FILE="configs/stage1_training_config_8b.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_config_8b.json"
OUTPUT_DIR="outputs/stage1_alignment_8b"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Default to 8B model
AUTO_DOWNLOAD_DATA=true
DRY_RUN=false

# Color functions for better output
print_header() {
    echo -e "\n\033[1;34m==================================================\033[0m"
    echo -e "\033[1;34m$1\033[0m"
    echo -e "\033[1;34m==================================================\033[0m"
}

print_info() {
    echo -e "\033[1;36m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

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
            # Auto-select config based on model name
            if [[ "$MODEL_NAME" == *"3B"* ]]; then
                CONFIG_FILE="configs/stage1_training_config_3b.yaml"
                DEEPSPEED_CONFIG="configs/deepspeed_config_3b.json"
                OUTPUT_DIR="outputs/stage1_alignment_3b"
                NUM_GPUS=4  # Adjust default GPUs for 3B model
            elif [[ "$MODEL_NAME" == *"8B"* ]]; then
                CONFIG_FILE="configs/stage1_training_config_8b.yaml"
                DEEPSPEED_CONFIG="configs/deepspeed_config_8b.json"
                OUTPUT_DIR="outputs/stage1_alignment_8b"
                NUM_GPUS=8  # Adjust default GPUs for 8B model
            else
                CONFIG_FILE="configs/stage1_training_config.yaml"
                DEEPSPEED_CONFIG="configs/deepspeed_config.json"
                OUTPUT_DIR="outputs/stage1_alignment"
                NUM_GPUS=8  # Default to 8 GPUs for other models
            fi
            shift 2
            ;;
        --auto_download)
            AUTO_DOWNLOAD_DATA=true
            shift
            ;;
        --no_auto_download)
            AUTO_DOWNLOAD_DATA=false
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "LlamaAudio Stage 1 Training: Audio-Text Alignment"
            echo ""
            echo "This script trains the audio encoder and connector for basic audio-text alignment"
            echo "using speech recognition tasks. Default configuration uses Meta-Llama-3.1-8B-Instruct."
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model Options:"
            echo "  --model_name NAME       Llama model name (default: meta-llama/Meta-Llama-3.1-8B-Instruct)"
            echo "                          Supported: *3B*, *8B*, or custom model"
            echo ""
            echo "Training Options:"
            echo "  --num_gpus NUM          Number of GPUs per node (default: 8 for 8B model)"
            echo "  --num_nodes NUM         Number of nodes (default: 1)"
            echo "  --node_rank RANK        Rank of current node (default: 0)"
            echo "  --master_addr ADDR      Master node address (default: localhost)"
            echo "  --master_port PORT      Master node port (default: 29500)"
            echo ""
            echo "Configuration Options:"
            echo "  --config FILE           Training config file (auto-selected based on model)"
            echo "  --deepspeed FILE        DeepSpeed config file (auto-selected based on model)"
            echo "  --output_dir DIR        Output directory (auto-selected based on model)"
            echo ""
            echo "Data Options:"
            echo "  --auto_download         Auto-download Stage 1 data if missing (default: true)"
            echo "  --no_auto_download      Skip automatic data download"
            echo ""
            echo "Other Options:"
            echo "  --dry_run               Show configuration without starting training"
            echo "  --help                  Show this help message"
            echo ""
            echo "Stage 1 Objectives:"
            echo "  • Learn basic audio-text alignment through speech recognition"
            echo "  • Train audio encoder (Whisper-large-v3) and multimodal connector"
            echo "  • Keep LLM parameters frozen (only audio components are trained)"
            echo "  • Prepare foundation for Stage 2 multimodal understanding"
            echo ""
            echo "Expected Datasets (auto-downloaded if --auto_download):"
            echo "  • LibriSpeech: ~280K speech samples (~50GB)"
            echo "  • CommonVoice: ~150K speech samples (~15GB)"
            echo "  • Total: ~430K samples, ~65GB storage required"
            echo ""
            echo "Hardware Requirements:"
            echo "  • 8B Model: 6-8x GPU with 24GB+ VRAM (recommended)"
            echo "  • 3B Model: 4x GPU with 16GB+ VRAM (memory-constrained option)"
            echo "  • Storage: 120GB free space for data and checkpoints"
            echo ""
            echo "Expected Training Time:"
            echo "  • 8B Model: 6-12 hours (8 GPUs)"
            echo "  • 3B Model: 4-8 hours (4 GPUs)"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
print_header "LlamaAudio Stage 1: Audio-Text Alignment Training"

echo ""
print_info "Configuration Summary:"
echo "  🎯 Stage: 1 (Audio-Text Alignment Pre-training)"
echo "  🦙 Model: $MODEL_NAME"
echo "  🖥️  GPUs per node: $NUM_GPUS"
echo "  🌐 Nodes: $NUM_NODES"
echo "  📁 Config file: $CONFIG_FILE"
echo "  ⚡ DeepSpeed: $DEEPSPEED_CONFIG"
echo "  📂 Output: $OUTPUT_DIR"
echo "  📊 Logging: TensorBoard (automatic)"
echo "  📥 Auto-download: $AUTO_DOWNLOAD_DATA"

if [ "$DRY_RUN" = true ]; then
    print_info "Dry run mode - configuration displayed above"
    exit 0
fi

echo ""

# Check prerequisites
print_info "Checking prerequisites..."

# Check if Python environment is ready
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

if ! command -v deepspeed &> /dev/null; then
    print_error "DeepSpeed not found. Please install: pip install deepspeed"
    exit 1
fi

# Check configuration files
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    print_info "Available configs:"
    ls configs/stage1_training_config*.yaml 2>/dev/null || echo "  No stage1 configs found"
    exit 1
fi

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    print_error "DeepSpeed config file not found: $DEEPSPEED_CONFIG"
    print_info "Available DeepSpeed configs:"
    ls configs/deepspeed_config*.json 2>/dev/null || echo "  No DeepSpeed configs found"
    exit 1
fi

print_success "Configuration files verified"

# Check and prepare data
TRAIN_DATA="data/stage1_alignment/train.jsonl"
VAL_DATA="data/stage1_alignment/val.jsonl"

if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    if [ "$AUTO_DOWNLOAD_DATA" = true ]; then
        print_warning "Stage 1 data not found. Downloading and preparing..."
        
        # Download data
        if [ -f "scripts/data_download/download_stage1_data.sh" ]; then
            print_info "Downloading Stage 1 datasets..."
            bash scripts/data_download/download_stage1_data.sh
        else
            print_error "Data download script not found: scripts/data_download/download_stage1_data.sh"
            exit 1
        fi
        
        # Prepare data if preparation script exists
        if [ -f "data/stage1_alignment/prepare_stage1_data.py" ]; then
            print_info "Preparing Stage 1 data..."
            cd data/stage1_alignment && python prepare_stage1_data.py && cd ../..
        fi
        
        # Verify data after download
        if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
            print_error "Data download/preparation failed"
            print_info "Please manually run:"
            echo "  bash scripts/data_download/download_stage1_data.sh"
            echo "  cd data/stage1_alignment && python prepare_stage1_data.py"
            exit 1
        fi
    else
        print_error "Training data not found and auto-download disabled"
        print_info "Please prepare Stage 1 data first:"
        echo "  bash scripts/data_download/download_stage1_data.sh"
        echo "  cd data/stage1_alignment && python prepare_stage1_data.py"
        exit 1
    fi
fi

print_success "Training data verified"

# Display data statistics
if [ -f "$TRAIN_DATA" ] && [ -f "$VAL_DATA" ]; then
    TRAIN_SAMPLES=$(wc -l < "$TRAIN_DATA" 2>/dev/null || echo "0")
    VAL_SAMPLES=$(wc -l < "$VAL_DATA" 2>/dev/null || echo "0")
    TOTAL_SAMPLES=$((TRAIN_SAMPLES + VAL_SAMPLES))
    
    echo ""
    print_info "Dataset Statistics:"
    echo "  📊 Training samples: $TRAIN_SAMPLES"
    echo "  📊 Validation samples: $VAL_SAMPLES"
    echo "  📊 Total samples: $TOTAL_SAMPLES"
fi

# Set environment variables for distributed training
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$((NUM_GPUS * NUM_NODES))
export NODE_RANK=$NODE_RANK
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Setup logging
LOG_FILE="$OUTPUT_DIR/training.log"
echo "Training started at $(date)" > "$LOG_FILE"

echo ""
print_header "Starting Stage 1 Training"

print_info "Training Details:"
echo "  🎯 Objective: Audio-text alignment through speech recognition"
echo "  ❄️  Frozen components: LLM parameters"
echo "  🔥 Trainable components: Audio encoder, multimodal connector"
echo "  📈 Expected duration: $([ "$MODEL_NAME" == *"3B"* ] && echo "4-8 hours (3B model)" || echo "6-12 hours (8B model)")"
echo "  📊 Metrics tracked: Transcription accuracy, alignment quality"
echo "  🔧 Optimization: DeepSpeed ZeRO, mixed precision (FP16)"

echo ""
print_info "Launching training with DeepSpeed..."

# Construct training command
TRAIN_CMD="deepspeed"

if [ "$NUM_NODES" -eq 1 ]; then
    print_info "Single node training mode"
    TRAIN_CMD="$TRAIN_CMD --num_gpus=$NUM_GPUS"
else
    print_info "Multi-node training mode"
    TRAIN_CMD="$TRAIN_CMD --num_gpus=$NUM_GPUS --num_nodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

TRAIN_CMD="$TRAIN_CMD scripts/train.py"
TRAIN_CMD="$TRAIN_CMD --config \"$CONFIG_FILE\""
TRAIN_CMD="$TRAIN_CMD --deepspeed \"$DEEPSPEED_CONFIG\""
TRAIN_CMD="$TRAIN_CMD --train_data_path \"$TRAIN_DATA\""
TRAIN_CMD="$TRAIN_CMD --val_data_path \"$VAL_DATA\""
TRAIN_CMD="$TRAIN_CMD --output_dir \"$OUTPUT_DIR\""
TRAIN_CMD="$TRAIN_CMD --llama_model_name \"$MODEL_NAME\""

# Note: --stage and logging options are configured in the YAML config file
# Stage 1 specific settings are in the config file's stage_info section

print_info "Executing command:"
echo "$TRAIN_CMD"
echo ""

# Launch training
eval $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"

# Check training completion
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    print_success "Stage 1 training completed successfully!"
    
    echo ""
    print_info "Training Results:"
    echo "  📂 Model checkpoint: $OUTPUT_DIR/final_model/"
    echo "  📋 Training logs: $OUTPUT_DIR/training.log"
    echo "  📈 TensorBoard logs: $OUTPUT_DIR/tensorboard/"
    echo "      View with: tensorboard --logdir=$OUTPUT_DIR/tensorboard/"
    echo "  ⚙️  Configuration backup: $OUTPUT_DIR/$(basename $CONFIG_FILE)"
    
    echo ""
    print_info "Next Steps:"
    echo "  1. 🔍 Review training logs and metrics"
    echo "  2. 📊 Evaluate model performance:"
    echo "     python scripts/evaluate.py --stage 1 --model_path $OUTPUT_DIR/final_model/"
    echo "  3. 📥 Prepare Stage 2 data:"
    echo "     bash scripts/data_download/download_stage2_data.sh"
    echo "  4. 🚀 Start Stage 2 training:"
    echo "     bash scripts/run_stage2_training.sh --stage1_checkpoint $OUTPUT_DIR/final_model/"
    
    # Copy configuration files for reference
    print_info "Backing up configuration files..."
    cp "$CONFIG_FILE" "$OUTPUT_DIR/"
    cp "$DEEPSPEED_CONFIG" "$OUTPUT_DIR/"
    
    # Create stage completion marker
    cat > "$OUTPUT_DIR/stage1_completed.json" << EOF
{
    "stage": 1,
    "status": "completed",
    "model_name": "$MODEL_NAME",
    "config_file": "$CONFIG_FILE",
    "output_dir": "$OUTPUT_DIR",
    "num_gpus": $NUM_GPUS,
    "num_nodes": $NUM_NODES,
    "timestamp": "$(date -Iseconds)",
    "next_stage": "stage2_understanding"
}
EOF
    
    echo ""
    print_success "🎯 Stage 1 Achievement: Basic audio-text alignment learned!"
    print_success "🚀 Ready for Stage 2: Audio understanding and description"
    
else
    echo ""
    print_error "Stage 1 training failed!"
    print_error "Check the training logs for details: $LOG_FILE"
    
    echo ""
    print_info "Common Issues and Solutions:"
    echo "  💾 Out of Memory:"
    echo "     - Reduce batch_size in config file"
    echo "     - Use more GPUs with smaller batch per GPU"
    echo "     - Switch to ZeRO stage 3 in DeepSpeed config"
    echo ""
    echo "  🖥️  CUDA Errors:"
    echo "     - Check GPU compatibility (CUDA 11.7+)"
    echo "     - Verify driver installation"
    echo "     - Reduce model size (use 3B instead of 8B)"
    echo ""
    echo "  📊 Data Loading Errors:"
    echo "     - Verify data paths and file formats"
    echo "     - Check disk space for datasets"
    echo "     - Re-run data preparation scripts"
    echo ""
    echo "  🤗 Model Loading Errors:"
    echo "     - Check Hugging Face access and authentication"
    echo "     - Verify model name spelling"
    echo "     - Check internet connection for model download"
    
    exit 1
fi

echo ""
print_header "Stage 1 Training Session Completed!"
echo "Session completed at $(date)" >> "$LOG_FILE"
