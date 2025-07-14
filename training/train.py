#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import argparse
from transformers import (
    AutoTokenizer,
    WhisperFeatureExtractor,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from torch.utils.data import DataLoader
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from datetime import datetime

pdj=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pdj)

from models.model import create_model_from_config, Qwen2AudioModel
from models.dataset import create_dataset, collate_fn
from training.dpo_trainer import DPOTrainer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioTrainingCallback(TrainerCallback):
    """Custom callback for audio training monitoring"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log additional metrics"""
        if logs and state.is_world_process_zero:
            # Debug print
            logger.info(f"Step {state.global_step}: Logging metrics to TensorBoard")
            logger.info(f"Logs content: {logs}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                logs['gpu_memory_gb'] = gpu_memory
            
            # Log learning rate
            if 'learning_rate' in logs:
                logs['lr'] = logs['learning_rate']
            
            # Log training loss
            if 'loss' in logs:
                logs['train_loss'] = logs['loss']
            
            # Log step and epoch
            logs['step'] = state.global_step
            logs['epoch'] = state.epoch
            
            # Log model parameters
            if model is not None:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logs['total_params'] = total_params
                logs['trainable_params'] = trainable_params
            
            # Debug print after adding metrics
            logger.info(f"Final logs to be written: {logs}")
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Custom save logic"""
        if state.is_world_process_zero:
            logger.info(f"Saving checkpoint at step {state.global_step}")
            
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log metrics at the end of each step"""
        if state.is_world_process_zero:
            logger.info(f"Step {state.global_step} completed")
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                logger.info(f"GPU Memory: {gpu_memory:.2f} GB")


def load_config(config_path: str, model_config_path: str = None) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge model-specific config if provided
    if model_config_path and os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Deep merge model config
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(config, model_config)
    
    return config


def setup_deepspeed_config(config: dict) -> str:
    """Setup DeepSpeed configuration"""
    deepspeed_config = config.get('training', {}).get('deepspeed', {})
    
    if deepspeed_config.get('enabled', False):
        config_file = deepspeed_config.get('config_file', 'configs/deepspeed_config.json')
        if os.path.exists(config_file):
            return config_file
        else:
            logger.warning(f"DeepSpeed config file {config_file} not found, disabling DeepSpeed")
            return None
    return None


def create_training_arguments(config, stage):
    """Create training arguments based on config"""
    training_config = config["training"]["hyperparameters"]
    
    # Get output directory
    output_dir = f"outputs/{stage}_{config['model']['llm_backbone']['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set TensorBoard log directory
    tensorboard_dir = f"tensorboard/{stage}_{config['model']['llm_backbone']['model_type']}"
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Ensure numeric values are correctly typed
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    warmup_ratio = float(training_config["warmup_ratio"])
    max_grad_norm = float(training_config["max_grad_norm"])
    
    # Get bf16 setting from config (default to False if not specified)
    bf16_enabled = training_config.get("bf16", False)
    tf32_enabled = training_config.get("tf32", False)
    gradient_checkpointing_enabled = training_config.get("gradient_checkpointing", False)
    
    # Debug: Print parameter types
    logger.info(f"Training parameters: lr={learning_rate} (type: {type(learning_rate)}), wd={weight_decay} (type: {type(weight_decay)})")
    logger.info(f"Mixed precision settings: bf16={bf16_enabled}, tf32={tf32_enabled}")
    logger.info(f"TensorBoard log directory: {tensorboard_dir}")
    
    # Create training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        num_train_epochs=int(training_config["num_train_epochs"]),
        max_steps=int(training_config["max_steps"]),
        logging_steps=int(training_config.get("logging_steps", 1)),  # Use config value or default to 1
        eval_steps=int(training_config["eval_steps"]),
        save_steps=int(training_config["save_steps"]),
        eval_strategy="no",  # Disable evaluation to avoid eval_dataset requirement
        save_strategy=training_config.get("save_strategy", "steps"),  # Keep save_strategy as is
        load_best_model_at_end=False,  # Disable since we're not evaluating
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),  # Add metric for best model
        greater_is_better=training_config.get("greater_is_better", False),  # Add greater is better
        report_to=training_config.get("report_to", ["tensorboard"]),  # Use config value or default
        logging_dir=training_config.get("logging_dir", tensorboard_dir),  # Use config value or default
        logging_first_step=True,  # Log the first step
        logging_nan_inf_filter=False,  # Don't filter out NaN/Inf values
        remove_unused_columns=training_config.get("remove_unused_columns", False),  # Use config value
        ddp_find_unused_parameters=False,
        deepspeed=config["training"]["deepspeed"]["config_file"] if config["training"]["deepspeed"]["enabled"] else None,
        torch_compile=False,  # Disable torch.compile
        bf16=bf16_enabled,  # Add bf16 parameter
        tf32=tf32_enabled,  # Add tf32 parameter
        gradient_checkpointing=gradient_checkpointing_enabled,  # Add gradient checkpointing
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 0)),  # Add dataloader workers
        group_by_length=training_config.get("group_by_length", False),  # Add group by length
    )
    
    # Debug: Print training arguments
    logger.info(f"Training arguments: {args}")
    
    return args


def train_pretrain_stage(config: dict, model: Qwen2AudioModel, tokenizer, feature_extractor):
    """Train pretraining stage"""
    logger.info("Starting pretraining stage...")
    
    # Create dataset
    train_dataset = create_dataset(config, tokenizer, feature_extractor, stage="pretrain")
    eval_dataset = None  # Add eval dataset if available
    
    # Create training arguments
    training_args = create_training_arguments(config, "pretrain")
    
    # Debug print training arguments
    logger.info("Training arguments:")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info(f"Logging directory: {training_args.logging_dir}")
    logger.info(f"Logging steps: {training_args.logging_steps}")
    logger.info(f"Report to: {training_args.report_to}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[AudioTrainingCallback()]
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    return trainer


def train_sft_stage(config: dict, model: Qwen2AudioModel, tokenizer, feature_extractor):
    """Train supervised fine-tuning stage"""
    logger.info("Starting SFT stage...")
    
    # Create dataset
    train_dataset = create_dataset(config, tokenizer, feature_extractor, stage="sft")
    eval_dataset = None  # Add eval dataset if available
    
    # Create training arguments
    training_args = create_training_arguments(config, "sft")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[AudioTrainingCallback()]
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    return trainer


def train_dpo_stage(config: dict, model: Qwen2AudioModel, tokenizer, feature_extractor):
    """Train DPO stage"""
    logger.info("Starting DPO stage...")
    
    # Create dataset
    train_dataset = create_dataset(config, tokenizer, feature_extractor, stage="dpo")
    eval_dataset = None  # Add eval dataset if available
    
    # Create training arguments
    training_args = create_training_arguments(config, "dpo")
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO hyperparameter
        data_collator=collate_fn
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2-Audio model")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--model_config", type=str, help="Path to model-specific config file")
    parser.add_argument("--stage", type=str, choices=["pretrain", "sft", "dpo"], required=True, help="Training stage")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.model_config)
    
    # Set training stage in config
    config['training']['stage'] = args.stage
    
    # Create model
    logger.info("Creating model...")
    model = create_model_from_config(config)
    tokenizer = model.tokenizer
    
    # Create feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config['model']['audio_encoder']['model_name']
    )
    
    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Train based on stage
    if args.stage == "pretrain":
        trainer = train_pretrain_stage(config, model, tokenizer, feature_extractor)
    elif args.stage == "sft":
        trainer = train_sft_stage(config, model, tokenizer, feature_extractor)
    elif args.stage == "dpo":
        trainer = train_dpo_stage(config, model, tokenizer, feature_extractor)
    else:
        raise ValueError(f"Unknown training stage: {args.stage}")
    
    logger.info(f"Training {args.stage} stage completed!")


if __name__ == "__main__":
    main() 