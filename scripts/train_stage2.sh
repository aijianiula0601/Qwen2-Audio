#!/bin/bash

# Qwen2-Audio Stage 2 Training Script
# Supervised Fine-Tuning with conversation data

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12346

# Training parameters (will be updated by setup script)
MODEL_NAME_OR_PATH="checkpoints/qwen2-audio-stage1"  # Stage 1 checkpoint
DATA_PATH="data/stage2_sft/train.jsonl"  # SFT conversation data
OUTPUT_DIR="checkpoints/qwen2-audio-stage2"
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=2
WARMUP_RATIO=0.03
MAX_LENGTH=2048

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch distributed training
torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --training_stage stage2 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --dataloader_drop_last True \
    --bf16 True \
    --tf32 True \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name "qwen2-audio-stage2-$(date +%Y%m%d-%H%M%S)" \
    --freeze_audio_encoder False

echo "Stage 2 training completed. Model saved to $OUTPUT_DIR" 