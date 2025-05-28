#!/bin/bash

# Qwen2-Audio Stage 1 Training Script
# Pre-training with natural language prompts

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12345

# Training parameters (will be updated by setup script)
MODEL_NAME_OR_PATH="models/Qwen_Qwen2-7B"  # Base LLM model path
DATA_PATH="data/stage1_pretraining/train.jsonl"  # Pre-training data
OUTPUT_DIR="checkpoints/qwen2-audio-stage1"
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=3
WARMUP_RATIO=0.03
MAX_LENGTH=2048

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch distributed training
torchrun --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    src/trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --warmup_ratio $WARMUP_RATIO \
    --max_seq_length $MAX_LENGTH \
    --stage "stage1" \
    --logging_steps 50 \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --fp16 \
    --ddp_find_unused_parameters False \
    --report_to wandb \
    --run_name "qwen2-audio-stage1" \
    2>&1 | tee $OUTPUT_DIR/train.log 