"""
Training utilities for LlamaAudio framework.

This module provides utility functions for distributed training, optimization,
and other training-related functionality.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR, 
    CosineAnnealingWarmRestarts, ReduceLROnPlateau
)
import os
import logging
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
import deepspeed
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


def setup_distributed_training(backend: str = "nccl") -> Dict[str, Any]:
    """
    Setup distributed training environment.
    
    Args:
        backend: Distributed backend ("nccl", "gloo", "mpi")
        
    Returns:
        Dictionary containing distributed training info
    """
    # Check if distributed training is available
    if not dist.is_available():
        logger.warning("Distributed training not available")
        return {
            "is_distributed": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }
    
    # Initialize distributed training
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        return {
            "is_distributed": True,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        }
    
    else:
        logger.info("Single GPU/CPU training")
        return {
            "is_distributed": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
        }


def setup_deepspeed(
    model: nn.Module,
    config_path: str,
    model_parameters: Optional[Dict] = None,
) -> Tuple[Any, Any, Any, Any]:
    """
    Setup DeepSpeed training engine.
    
    Args:
        model: PyTorch model
        config_path: Path to DeepSpeed configuration file
        model_parameters: Model parameters for optimization
        
    Returns:
        Tuple of (model_engine, optimizer, lr_scheduler, training_data_loader)
    """
    try:
        # Initialize DeepSpeed
        model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
            model=model,
            config=config_path,
            model_parameters=model_parameters,
        )
        
        logger.info("DeepSpeed initialized successfully")
        return model_engine, optimizer, lr_scheduler, training_dataloader
    
    except Exception as e:
        logger.error(f"Failed to initialize DeepSpeed: {e}")
        raise


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine_with_warmup",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    scheduler_kwargs: Optional[Dict] = None,
    optimizer_kwargs: Optional[Dict] = None,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: PyTorch model
        learning_rate: Initial learning rate
        weight_decay: Weight decay factor
        optimizer_type: Type of optimizer ("adamw", "sgd")
        scheduler_type: Type of LR scheduler
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        scheduler_kwargs: Additional scheduler arguments
        optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Prepare parameters for optimization
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    optimizer_kwargs = optimizer_kwargs or {}
    
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=optimizer_kwargs.get("betas", (0.9, 0.999)),
            eps=optimizer_kwargs.get("eps", 1e-8),
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=optimizer_kwargs.get("momentum", 0.9),
            nesterov=optimizer_kwargs.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Create scheduler
    scheduler_kwargs = scheduler_kwargs or {}
    scheduler = None
    
    if scheduler_type == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    elif scheduler_type == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=scheduler_kwargs.get("num_cycles", 0.5),
        )
    
    elif scheduler_type == "cosine_annealing":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=scheduler_kwargs.get("eta_min", 0),
        )
    
    elif scheduler_type == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_kwargs.get("T_0", num_training_steps // 4),
            T_mult=scheduler_kwargs.get("T_mult", 2),
            eta_min=scheduler_kwargs.get("eta_min", 0),
        )
    
    elif scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_kwargs.get("mode", "min"),
            factor=scheduler_kwargs.get("factor", 0.5),
            patience=scheduler_kwargs.get("patience", 10),
            verbose=True,
        )
    
    elif scheduler_type == "sequential":
        # Linear warmup + cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=num_warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=scheduler_kwargs.get("eta_min", 0),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
    
    elif scheduler_type is None or scheduler_type == "none":
        scheduler = None
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    logger.info(f"Created optimizer: {optimizer_type}, scheduler: {scheduler_type}")
    
    return optimizer, scheduler


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    is_best: bool = False,
    deepspeed_engine: Optional[Any] = None,
):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        deepspeed_engine: DeepSpeed engine (if using DeepSpeed)
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Save with DeepSpeed if available
    if deepspeed_engine is not None:
        deepspeed_engine.save_checkpoint(save_path, tag=f"epoch_{epoch}_step_{step}")
        logger.info(f"DeepSpeed checkpoint saved to {save_path}")
    else:
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    # Save best model separately
    if is_best:
        best_path = save_path.replace(".pt", "_best.pt")
        if deepspeed_engine is not None:
            deepspeed_engine.save_checkpoint(best_path, tag="best")
        else:
            torch.save(checkpoint, best_path)
        logger.info(f"Best checkpoint saved to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with loaded checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
    }


def get_device() -> str:
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def calculate_gradient_norm(model: nn.Module) -> float:
    """Calculate the gradient norm of the model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        
    Returns:
        Original gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def warmup_lr_schedule(step: int, warmup_steps: int, base_lr: float) -> float:
    """
    Calculate learning rate with linear warmup.
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
        
    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    else:
        return base_lr


def get_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    free = total - allocated
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total,
    }


def log_model_info(model: nn.Module, logger: logging.Logger):
    """Log detailed model information."""
    param_counts = count_parameters(model)
    memory_stats = get_memory_usage()
    
    logger.info("=" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {param_counts['total']:,}")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"Frozen parameters: {param_counts['frozen']:,}")
    logger.info(f"Trainable ratio: {param_counts['trainable']/param_counts['total']*100:.2f}%")
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {memory_stats['allocated']:.2f} GB")
        logger.info(f"GPU memory reserved: {memory_stats['reserved']:.2f} GB")
        logger.info(f"GPU memory free: {memory_stats['free']:.2f} GB")
    
    logger.info("=" * 50) 