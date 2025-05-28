#!/bin/bash

# Qwen2-Audio DeepSpeed Training Script
# Unified script for all three training stages with DeepSpeed optimization

set -e

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --stage STAGE                Training stage (stage1, stage2, stage3) [required]"
    echo "  --model_path PATH           Path to base model or previous checkpoint [required]"
    echo "  --data_path PATH            Path to training data [required]"
    echo "  --output_dir PATH           Output directory for checkpoints [required]"
    echo "  --deepspeed_config PATH     Path to DeepSpeed config file [optional]"
    echo "  --num_gpus NUM              Number of GPUs to use [default: 4]"
    echo "  --batch_size NUM            Per-device batch size [default: auto]"
    echo "  --gradient_accumulation NUM Gradient accumulation steps [default: auto]"
    echo "  --learning_rate RATE        Learning rate [default: auto]"
    echo "  --num_epochs NUM            Number of training epochs [default: auto]"
    echo "  --warmup_ratio RATIO        Warmup ratio [default: 0.03]"
    echo "  --max_length NUM            Maximum sequence length [default: 2048]"
    echo "  --dpo_beta BETA             DPO beta parameter for stage3 [default: 0.1]"
    echo "  --freeze_audio_encoder      Freeze audio encoder (for stage3)"
    echo "  --freeze_llm                Freeze LLM"
    echo "  --help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Stage 1 pretraining"
    echo "  $0 --stage stage1 --model_path models/Qwen_Qwen2-7B --data_path data/stage1_pretraining/train.jsonl --output_dir checkpoints/stage1"
    echo ""
    echo "  # Stage 2 supervised fine-tuning"
    echo "  $0 --stage stage2 --model_path checkpoints/stage1 --data_path data/stage2_sft/train.jsonl --output_dir checkpoints/stage2"
    echo ""
    echo "  # Stage 3 DPO optimization"
    echo "  $0 --stage stage3 --model_path checkpoints/stage2 --data_path data/stage3_dpo/train.jsonl --output_dir checkpoints/stage3 --freeze_audio_encoder"
}

# Default values
STAGE=""
MODEL_PATH=""
DATA_PATH=""
OUTPUT_DIR=""
DEEPSPEED_CONFIG=""
NUM_GPUS=4
BATCH_SIZE=""
GRADIENT_ACCUMULATION=""
LEARNING_RATE=""
NUM_EPOCHS=""
WARMUP_RATIO=0.03
MAX_LENGTH=2048
DPO_BETA=0.1
FREEZE_AUDIO_ENCODER=false
FREEZE_LLM=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --deepspeed_config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --warmup_ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --dpo_beta)
            DPO_BETA="$2"
            shift 2
            ;;
        --freeze_audio_encoder)
            FREEZE_AUDIO_ENCODER=true
            shift
            ;;
        --freeze_llm)
            FREEZE_LLM=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$STAGE" || -z "$MODEL_PATH" || -z "$DATA_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments"
    print_usage
    exit 1
fi

# Validate stage
if [[ "$STAGE" != "stage1" && "$STAGE" != "stage2" && "$STAGE" != "stage3" ]]; then
    echo "Error: Invalid stage. Must be stage1, stage2, or stage3"
    exit 1
fi

# Set default values based on stage
if [[ -z "$DEEPSPEED_CONFIG" ]]; then
    DEEPSPEED_CONFIG="configs/deepspeed_${STAGE}.json"
fi

if [[ -z "$BATCH_SIZE" ]]; then
    case $STAGE in
        stage1) BATCH_SIZE=4 ;;
        stage2) BATCH_SIZE=2 ;;
        stage3) BATCH_SIZE=1 ;;
    esac
fi

if [[ -z "$GRADIENT_ACCUMULATION" ]]; then
    case $STAGE in
        stage1) GRADIENT_ACCUMULATION=4 ;;
        stage2) GRADIENT_ACCUMULATION=8 ;;
        stage3) GRADIENT_ACCUMULATION=16 ;;
    esac
fi

if [[ -z "$LEARNING_RATE" ]]; then
    case $STAGE in
        stage1) LEARNING_RATE=1e-4 ;;
        stage2) LEARNING_RATE=5e-5 ;;
        stage3) LEARNING_RATE=5e-6 ;;
    esac
fi

if [[ -z "$NUM_EPOCHS" ]]; then
    case $STAGE in
        stage1) NUM_EPOCHS=3 ;;
        stage2) NUM_EPOCHS=2 ;;
        stage3) NUM_EPOCHS=1 ;;
    esac
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export WORLD_SIZE=$NUM_GPUS
export MASTER_ADDR=localhost
export MASTER_PORT=$((12345 + ${STAGE: -1}))

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== Qwen2-Audio DeepSpeed Training Configuration ==="
echo "Stage: $STAGE"
echo "Model Path: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Warmup Ratio: $WARMUP_RATIO"
echo "Max Length: $MAX_LENGTH"
if [[ "$STAGE" == "stage3" ]]; then
    echo "DPO Beta: $DPO_BETA"
fi
echo "Freeze Audio Encoder: $FREEZE_AUDIO_ENCODER"
echo "Freeze LLM: $FREEZE_LLM"
echo "=================================================="

# Build training command
TRAIN_CMD="deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT src/trainer.py"
TRAIN_CMD="$TRAIN_CMD --model_name_or_path $MODEL_PATH"
TRAIN_CMD="$TRAIN_CMD --data_path $DATA_PATH"
TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --training_stage $STAGE"
TRAIN_CMD="$TRAIN_CMD --per_device_train_batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --num_train_epochs $NUM_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --warmup_ratio $WARMUP_RATIO"
TRAIN_CMD="$TRAIN_CMD --max_seq_length $MAX_LENGTH"
TRAIN_CMD="$TRAIN_CMD --deepspeed $DEEPSPEED_CONFIG"
TRAIN_CMD="$TRAIN_CMD --bf16 True"
TRAIN_CMD="$TRAIN_CMD --tf32 True"
TRAIN_CMD="$TRAIN_CMD --dataloader_drop_last True"
TRAIN_CMD="$TRAIN_CMD --remove_unused_columns False"
TRAIN_CMD="$TRAIN_CMD --ddp_find_unused_parameters False"
TRAIN_CMD="$TRAIN_CMD --report_to wandb"
TRAIN_CMD="$TRAIN_CMD --run_name qwen2-audio-${STAGE}-deepspeed-$(date +%Y%m%d-%H%M%S)"

# Stage-specific parameters
case $STAGE in
    stage1)
        TRAIN_CMD="$TRAIN_CMD --logging_steps 50"
        TRAIN_CMD="$TRAIN_CMD --save_steps 500"
        TRAIN_CMD="$TRAIN_CMD --save_total_limit 3"
        ;;
    stage2)
        TRAIN_CMD="$TRAIN_CMD --logging_steps 10"
        TRAIN_CMD="$TRAIN_CMD --save_steps 200"
        TRAIN_CMD="$TRAIN_CMD --save_total_limit 3"
        TRAIN_CMD="$TRAIN_CMD --evaluation_strategy steps"
        TRAIN_CMD="$TRAIN_CMD --eval_steps 200"
        ;;
    stage3)
        TRAIN_CMD="$TRAIN_CMD --logging_steps 5"
        TRAIN_CMD="$TRAIN_CMD --save_steps 100"
        TRAIN_CMD="$TRAIN_CMD --save_total_limit 3"
        TRAIN_CMD="$TRAIN_CMD --evaluation_strategy steps"
        TRAIN_CMD="$TRAIN_CMD --eval_steps 100"
        TRAIN_CMD="$TRAIN_CMD --dpo_beta $DPO_BETA"
        ;;
esac

# Add freeze options
if [[ "$FREEZE_AUDIO_ENCODER" == "true" ]]; then
    TRAIN_CMD="$TRAIN_CMD --freeze_audio_encoder"
fi

if [[ "$FREEZE_LLM" == "true" ]]; then
    TRAIN_CMD="$TRAIN_CMD --freeze_llm"
fi

# Log the command
echo "Executing: $TRAIN_CMD"
echo ""

# Execute training
eval "$TRAIN_CMD 2>&1 | tee $OUTPUT_DIR/train.log"

echo ""
echo "Training completed successfully!"
echo "Model saved to: $OUTPUT_DIR"
echo "Training log saved to: $OUTPUT_DIR/train.log" 