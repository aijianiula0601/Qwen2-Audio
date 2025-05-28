#!/bin/bash

# Qwen2-Audio Stage 3 Training Script with DeepSpeed
# Direct Preference Optimization (DPO)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=12347

# Training parameters (will be updated by setup script)
MODEL_NAME_OR_PATH="checkpoints/qwen2-audio-stage2-deepspeed"  # Stage 2 checkpoint
DATA_PATH="data/stage3_dpo/train.jsonl"  # DPO preference data
OUTPUT_DIR="checkpoints/qwen2-audio-stage3-deepspeed"
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=5e-6
NUM_EPOCHS=1
WARMUP_RATIO=0.1
MAX_LENGTH=2048
DPO_BETA=0.1
DEEPSPEED_CONFIG="configs/deepspeed_stage3.json"

# Create output directory
mkdir -p $OUTPUT_DIR

# Launch DeepSpeed training
deepspeed --num_gpus=4 \
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
    --deepspeed $DEEPSPEED_CONFIG \
    --bf16 True \
    --tf32 True \
    --report_to wandb \
    --run_name "qwen2-audio-stage3-deepspeed-$(date +%Y%m%d-%H%M%S)" \
    --freeze_audio_encoder True \
    --freeze_llm False \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "Stage 3 DeepSpeed training completed. Model saved to $OUTPUT_DIR" 