#!/bin/bash

# Qwen2-Audio Full Training Pipeline
# Runs all three stages: Pretrain -> SFT -> DPO

set -e

# Default configurations
MODEL_TYPE="qwen2_7b"
CONFIG_FILE="configs/base_config.yaml"
NUM_GPUS=8
SKIP_DATA_DOWNLOAD=false
SKIP_PRETRAIN=false
SKIP_SFT=false
SKIP_DPO=false

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
        --skip-data)
            SKIP_DATA_DOWNLOAD=true
            shift
            ;;
        --skip-pretrain)
            SKIP_PRETRAIN=true
            shift
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --skip-dpo)
            SKIP_DPO=true
            shift
            ;;
        --pretrain-only)
            SKIP_SFT=true
            SKIP_DPO=true
            shift
            ;;
        --sft-only)
            SKIP_PRETRAIN=true
            SKIP_DPO=true
            shift
            ;;
        --dpo-only)
            SKIP_PRETRAIN=true
            SKIP_SFT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_TYPE      Model type (qwen2_7b, qwen2_70b, llama3_8b)"
            echo "  --config CONFIG_FILE    Path to base config file"
            echo "  --gpus NUM_GPUS         Number of GPUs to use"
            echo "  --skip-data             Skip data download"
            echo "  --skip-pretrain         Skip pretraining stage"
            echo "  --skip-sft              Skip SFT stage"
            echo "  --skip-dpo              Skip DPO stage"
            echo "  --pretrain-only         Run only pretraining"
            echo "  --sft-only              Run only SFT"
            echo "  --dpo-only              Run only DPO"
            echo "  -h, --help              Show this help message"
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

echo "=== Qwen2-Audio Full Training Pipeline ==="
echo "Model type: $MODEL_TYPE"
echo "Config file: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "Pipeline stages:"
echo "  Data download: $([ "$SKIP_DATA_DOWNLOAD" = true ] && echo "SKIP" || echo "RUN")"
echo "  Pretraining: $([ "$SKIP_PRETRAIN" = true ] && echo "SKIP" || echo "RUN")"
echo "  SFT: $([ "$SKIP_SFT" = true ] && echo "SKIP" || echo "RUN")"
echo "  DPO: $([ "$SKIP_DPO" = true ] && echo "SKIP" || echo "RUN")"
echo ""

# Create main output directory
PIPELINE_OUTPUT_DIR="outputs/pipeline_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PIPELINE_OUTPUT_DIR"
echo "Pipeline output directory: $PIPELINE_OUTPUT_DIR"

# Log file for the entire pipeline
PIPELINE_LOG="${PIPELINE_OUTPUT_DIR}/pipeline.log"
echo "Pipeline log: $PIPELINE_LOG"

# Function to log with timestamp
log_with_timestamp() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$PIPELINE_LOG"
}

# Function to check if stage completed successfully
check_stage_success() {
    local stage_name=$1
    local checkpoint_path=$2
    
    if [ -d "$checkpoint_path" ] && [ -f "${checkpoint_path}/pytorch_model.bin" ]; then
        log_with_timestamp "$stage_name completed successfully!"
        return 0
    else
        log_with_timestamp "$stage_name failed or checkpoint not found!"
        return 1
    fi
}

# Start pipeline
log_with_timestamp "Starting Qwen2-Audio training pipeline for $MODEL_TYPE"

# Stage 0: Data Download
if [ "$SKIP_DATA_DOWNLOAD" = false ]; then
    log_with_timestamp "Stage 0: Data Download"
    
    if [ ! -d "data" ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
        log_with_timestamp "Downloading and preparing data..."
        bash scripts/download_data.sh 2>&1 | tee -a "$PIPELINE_LOG"
        
        if [ $? -eq 0 ]; then
            log_with_timestamp "Data download completed"
        else
            log_with_timestamp "Data download failed"
            exit 1
        fi
    else
        log_with_timestamp "Data already exists, skipping download"
    fi
fi

# Stage 1: Pretraining
PRETRAIN_CHECKPOINT=""
if [ "$SKIP_PRETRAIN" = false ]; then
    log_with_timestamp "Stage 1: Pretraining"
    
    # Run pretraining
    bash scripts/train_pretrain.sh --model "$MODEL_TYPE" --config "$CONFIG_FILE" --gpus "$NUM_GPUS" 2>&1 | tee -a "$PIPELINE_LOG"
    
    # Find the latest pretrain output directory
    PRETRAIN_CHECKPOINT=$(find outputs -name "pretrain_${MODEL_TYPE}_*" -type d | sort | tail -1)
    
    if check_stage_success "Pretraining" "$PRETRAIN_CHECKPOINT"; then
        log_with_timestamp "Pretrain checkpoint: $PRETRAIN_CHECKPOINT"
    else
        log_with_timestamp "Pretraining failed, exiting pipeline"
        exit 1
    fi
else
    log_with_timestamp "Skipping pretraining stage"
    # Look for existing pretrain checkpoint
    PRETRAIN_CHECKPOINT=$(find outputs -name "pretrain_${MODEL_TYPE}_*" -type d | sort | tail -1)
    if [ -n "$PRETRAIN_CHECKPOINT" ]; then
        log_with_timestamp "Using existing pretrain checkpoint: $PRETRAIN_CHECKPOINT"
    fi
fi

# Stage 2: Supervised Fine-tuning
SFT_CHECKPOINT=""
if [ "$SKIP_SFT" = false ]; then
    log_with_timestamp "Stage 2: Supervised Fine-tuning"
    
    # Build SFT command
    SFT_CMD="bash scripts/train_sft.sh --model $MODEL_TYPE --config $CONFIG_FILE --gpus $NUM_GPUS"
    if [ -n "$PRETRAIN_CHECKPOINT" ]; then
        SFT_CMD="$SFT_CMD --pretrain_checkpoint ${PRETRAIN_CHECKPOINT}/pytorch_model.bin"
        log_with_timestamp "Starting SFT from pretrain checkpoint: $PRETRAIN_CHECKPOINT"
    else
        log_with_timestamp "Starting SFT from scratch (no pretrain checkpoint)"
    fi
    
    # Run SFT
    eval "$SFT_CMD" 2>&1 | tee -a "$PIPELINE_LOG"
    
    # Find the latest SFT output directory
    SFT_CHECKPOINT=$(find outputs -name "sft_${MODEL_TYPE}_*" -type d | sort | tail -1)
    
    if check_stage_success "SFT" "$SFT_CHECKPOINT"; then
        log_with_timestamp "SFT checkpoint: $SFT_CHECKPOINT"
    else
        log_with_timestamp "SFT failed, exiting pipeline"
        exit 1
    fi
else
    log_with_timestamp "Skipping SFT stage"
    # Look for existing SFT checkpoint
    SFT_CHECKPOINT=$(find outputs -name "sft_${MODEL_TYPE}_*" -type d | sort | tail -1)
    if [ -n "$SFT_CHECKPOINT" ]; then
        log_with_timestamp "Using existing SFT checkpoint: $SFT_CHECKPOINT"
    fi
fi

# Stage 3: Direct Preference Optimization
DPO_CHECKPOINT=""
if [ "$SKIP_DPO" = false ]; then
    log_with_timestamp "Stage 3: Direct Preference Optimization"
    
    # Build DPO command
    DPO_CMD="bash scripts/train_dpo.sh --model $MODEL_TYPE --config $CONFIG_FILE --gpus $NUM_GPUS"
    if [ -n "$SFT_CHECKPOINT" ]; then
        DPO_CMD="$DPO_CMD --sft_checkpoint ${SFT_CHECKPOINT}/pytorch_model.bin"
        log_with_timestamp "Starting DPO from SFT checkpoint: $SFT_CHECKPOINT"
    else
        log_with_timestamp "Starting DPO from scratch (no SFT checkpoint)"
    fi
    
    # Run DPO
    eval "$DPO_CMD" 2>&1 | tee -a "$PIPELINE_LOG"
    
    # Find the latest DPO output directory
    DPO_CHECKPOINT=$(find outputs -name "dpo_${MODEL_TYPE}_*" -type d | sort | tail -1)
    
    if check_stage_success "DPO" "$DPO_CHECKPOINT"; then
        log_with_timestamp "DPO checkpoint: $DPO_CHECKPOINT"
    else
        log_with_timestamp "DPO failed, exiting pipeline"
        exit 1
    fi
else
    log_with_timestamp "Skipping DPO stage"
    # Look for existing DPO checkpoint
    DPO_CHECKPOINT=$(find outputs -name "dpo_${MODEL_TYPE}_*" -type d | sort | tail -1)
    if [ -n "$DPO_CHECKPOINT" ]; then
        log_with_timestamp "Using existing DPO checkpoint: $DPO_CHECKPOINT"
    fi
fi

# Pipeline completion
log_with_timestamp "=== Pipeline Completed Successfully! ==="

# Create summary
SUMMARY_FILE="${PIPELINE_OUTPUT_DIR}/pipeline_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Qwen2-Audio Training Pipeline Summary
=====================================

Model Type: $MODEL_TYPE
Config File: $CONFIG_FILE
GPUs Used: $NUM_GPUS
Pipeline Start: $(head -1 "$PIPELINE_LOG" | cut -d' ' -f1-2)
Pipeline End: $(tail -1 "$PIPELINE_LOG" | cut -d' ' -f1-2)

Checkpoints:
- Pretrain: $PRETRAIN_CHECKPOINT
- SFT: $SFT_CHECKPOINT  
- DPO: $DPO_CHECKPOINT

Final Model: $DPO_CHECKPOINT

Usage:
1. Inference:
   python demo/inference.py --model_path $DPO_CHECKPOINT --interactive

2. Test single audio:
   python demo/inference.py --model_path $DPO_CHECKPOINT --audio_path your_audio.wav --mode transcribe

3. Audio chat:
   python demo/inference.py --model_path $DPO_CHECKPOINT --audio_path your_audio.wav --mode chat --instruction "What do you hear?"

EOF

log_with_timestamp "Pipeline summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

echo ""
echo "Next steps:"
echo "1. Test your model: python demo/inference.py --model_path $DPO_CHECKPOINT --interactive"
echo "2. View full log: cat $PIPELINE_LOG"
echo "3. Check summary: cat $SUMMARY_FILE" 