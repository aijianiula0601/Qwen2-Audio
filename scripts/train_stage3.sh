#!/bin/bash

# Qwen2-Audio Stage 3 Training Script
# Direct Preference Optimization (DPO)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12347

# Training parameters (will be updated by setup script)
MODEL_NAME_OR_PATH="checkpoints/qwen2-audio-stage2"  # Stage 2 checkpoint
DATA_PATH="data/stage3_dpo/train.jsonl"  # DPO preference data
OUTPUT_DIR="checkpoints/qwen2-audio-stage3"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-5
NUM_EPOCHS=1
WARMUP_RATIO=0.1
MAX_LENGTH=2048
DPO_BETA=0.1

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch distributed training
torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --training_stage stage3 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --dpo_beta $DPO_BETA \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --dataloader_drop_last True \
    --bf16 True \
    --tf32 True \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name "qwen2-audio-stage3-$(date +%Y%m%d-%H%M%S)" \
    --freeze_audio_encoder True \
    --freeze_llm False

echo "Stage 3 training completed. Model saved to $OUTPUT_DIR" 