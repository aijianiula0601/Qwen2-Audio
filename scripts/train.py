#!/usr/bin/env python3
"""
LlamaAudio Training Script

This script trains LlamaAudio models with support for:
- DeepSpeed distributed training
- Multi-GPU and multi-node training
- TensorBoard logging
- Checkpointing and resuming
- Configuration file support
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_audio.models.llama_audio_model import LlamaAudioConfig
from llama_audio.training.trainer import LlamaAudioTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LlamaAudio model")
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    
    # Model arguments
    parser.add_argument("--llama_model_name", type=str, help="Llama model name")
    parser.add_argument("--whisper_model_name", type=str, help="Whisper model name")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient training")
    parser.add_argument("--freeze_llm", action="store_true", help="Freeze LLM parameters")
    
    # Training arguments
    parser.add_argument("--train_data_path", type=str, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.llama_model_name:
        config["model"]["llama_model_name"] = args.llama_model_name
    if args.whisper_model_name:
        config["model"]["whisper_model_name"] = args.whisper_model_name
    if args.train_data_path:
        config["training"]["train_data_path"] = args.train_data_path
    if args.val_data_path:
        config["training"]["val_data_path"] = args.val_data_path
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.use_lora:
        config["model"]["use_lora"] = True
    if args.freeze_llm:
        config["model"]["freeze_llm"] = True
    if args.resume_from_checkpoint:
        config["resume"]["resume_from_checkpoint"] = args.resume_from_checkpoint
    
    # Override DeepSpeed config path
    deepspeed_config = args.deepspeed
    if deepspeed_config is None and config["deepspeed"]["enabled"]:
        deepspeed_config = config["deepspeed"]["config_path"]
    
    # Create model configuration
    model_config = LlamaAudioConfig(**config["model"])
    
    # Create output directory
    output_dir = config["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting LlamaAudio training...")
    logger.info(f"Model: {model_config.llama_model_name}")
    logger.info(f"Audio Encoder: {model_config.whisper_model_name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Use LoRA: {model_config.use_lora}")
    logger.info(f"DeepSpeed: {deepspeed_config is not None}")
    
    # Initialize trainer
    trainer = LlamaAudioTrainer(
        model_config=model_config,
        training_config=config["training"],
        output_dir=output_dir,
        deepspeed_config=deepspeed_config,
        resume_from_checkpoint=config["resume"]["resume_from_checkpoint"]
    )
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
        
    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == "__main__":
    main() 