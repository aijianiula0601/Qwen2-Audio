#!/usr/bin/env python3
"""
Qwen2-Audio DeepSpeed Training Example

This script demonstrates how to train Qwen2-Audio models using DeepSpeed
for memory optimization and performance improvement.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trainer import Qwen2AudioTrainingArguments, train_qwen2_audio
from processor import Qwen2AudioProcessor


def create_deepspeed_config(stage: str, output_path: str):
    """Create a DeepSpeed configuration file for the given stage"""
    
    configs = {
        "stage1": {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
                "cpu_offload": False
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto"
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "activation_checkpointing": {
                "partition_activations": False,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False,
            "steps_per_print": 50
        },
        
        "stage2": {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto"
                }
            },
            "bf16": {
                "enabled": True
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False,
            "steps_per_print": 10
        },
        
        "stage3": {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto"
                }
            },
            "bf16": {
                "enabled": True
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False,
            "steps_per_print": 5
        }
    }
    
    config = configs.get(stage, configs["stage1"])
    
    # Save config to file
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio DeepSpeed Training Example")
    parser.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2", "stage3"],
                       help="Training stage")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to base model or checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                       help="Path to DeepSpeed config file")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=None,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--freeze_audio_encoder", action="store_true",
                       help="Freeze audio encoder")
    parser.add_argument("--freeze_llm", action="store_true",
                       help="Freeze LLM")
    
    args = parser.parse_args()
    
    # Set default values based on stage
    defaults = {
        "stage1": {"batch_size": 4, "gradient_accumulation": 4, "learning_rate": 1e-4, "num_epochs": 3},
        "stage2": {"batch_size": 2, "gradient_accumulation": 8, "learning_rate": 5e-5, "num_epochs": 2},
        "stage3": {"batch_size": 1, "gradient_accumulation": 16, "learning_rate": 5e-6, "num_epochs": 1},
    }
    
    stage_defaults = defaults[args.stage]
    batch_size = args.batch_size or stage_defaults["batch_size"]
    gradient_accumulation = args.gradient_accumulation or stage_defaults["gradient_accumulation"]
    learning_rate = args.learning_rate or stage_defaults["learning_rate"]
    num_epochs = args.num_epochs or stage_defaults["num_epochs"]
    
    # Create DeepSpeed config if not provided
    if args.deepspeed_config is None:
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        deepspeed_config = str(config_dir / f"deepspeed_{args.stage}_auto.json")
        create_deepspeed_config(args.stage, deepspeed_config)
        print(f"Created DeepSpeed config: {deepspeed_config}")
    else:
        deepspeed_config = args.deepspeed_config
    
    # Create training arguments
    training_args = Qwen2AudioTrainingArguments(
        output_dir=args.output_dir,
        training_stage=args.stage,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        logging_steps=10 if args.stage != "stage1" else 50,
        save_steps=200 if args.stage != "stage1" else 500,
        save_total_limit=3,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=deepspeed_config,
        bf16=True if args.stage != "stage1" else False,
        fp16=True if args.stage == "stage1" else False,
        tf32=True,
        report_to="wandb",
        run_name=f"qwen2-audio-{args.stage}-deepspeed-example",
        freeze_audio_encoder=args.freeze_audio_encoder,
        freeze_llm=args.freeze_llm,
    )
    
    # Add stage-specific parameters
    if args.stage in ["stage2", "stage3"]:
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = training_args.save_steps
    
    if args.stage == "stage3":
        training_args.dpo_beta = 0.1
    
    print("=== Training Configuration ===")
    print(f"Stage: {args.stage}")
    print(f"Model Path: {args.model_path}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"DeepSpeed Config: {deepspeed_config}")
    print(f"Batch Size: {batch_size}")
    print(f"Gradient Accumulation: {gradient_accumulation}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Freeze Audio Encoder: {args.freeze_audio_encoder}")
    print(f"Freeze LLM: {args.freeze_llm}")
    print("==============================")
    
    # Initialize processor
    processor = Qwen2AudioProcessor()
    
    # Start training
    try:
        trainer = train_qwen2_audio(
            model_name_or_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            training_args=training_args,
            processor=processor
        )
        print(f"Training completed successfully! Model saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 