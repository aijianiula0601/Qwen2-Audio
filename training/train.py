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
            # Log GPU memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                logs['gpu_memory_gb'] = gpu_memory
            
            # Log learning rate
            if 'learning_rate' in logs:
                logs['lr'] = logs['learning_rate']
    
    def on_save(self, args, state, control, model=None, **kwargs):
        """Custom save logic"""
        if state.is_world_process_zero:
            logger.info(f"Saving checkpoint at step {state.global_step}")


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
    
    # Ensure numeric values are correctly typed
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    warmup_ratio = float(training_config["warmup_ratio"])
    max_grad_norm = float(training_config["max_grad_norm"])
    
    # Debug: Print parameter types
    logger.info(f"Training parameters: lr={learning_rate} (type: {type(learning_rate)}), wd={weight_decay} (type: {type(weight_decay)})")
    
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
        logging_steps=int(training_config["logging_steps"]),
        eval_steps=int(training_config["eval_steps"]),
        save_steps=int(training_config["save_steps"]),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=config["training"]["deepspeed"]["config_file"] if config["training"]["deepspeed"]["enabled"] else None,
        torch_compile=False,  # Disable torch.compile
    )
    
    return args


def train_pretrain_stage(config: dict, model: Qwen2AudioModel, tokenizer, feature_extractor):
    """Train pretraining stage"""
    logger.info("Starting pretraining stage...")
    
    # Create dataset
    train_dataset = create_dataset(config, tokenizer, feature_extractor, stage="pretrain")
    eval_dataset = None  # Add eval dataset if available
    
    # Create training arguments
    training_args = create_training_arguments(config, "pretrain")
    
    # logger.info("--------------------------------training_args:--------------------------------")
    # logger.info(training_args)
    # logger.info("------------------------------------------------------------------------------")
    
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