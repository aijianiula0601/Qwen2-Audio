"""
LlamaAudio Trainer with DeepSpeed and TensorBoard support.

This module implements the main training loop for LlamaAudio models with
support for distributed training, mixed precision, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
import json
from pathlib import Path
import deepspeed
from transformers import WhisperProcessor
import numpy as np

from ..models.llama_audio_model import LlamaAudioModel, LlamaAudioConfig
from .data_loader import AudioTextDataLoader, create_data_loaders
from .utils import (
    setup_distributed_training, setup_deepspeed, create_optimizer_and_scheduler,
    set_seed, save_checkpoint, load_checkpoint, get_device, format_time,
    calculate_gradient_norm, clip_gradients, get_memory_usage, log_model_info
)

logger = logging.getLogger(__name__)


class LlamaAudioTrainer:
    """
    Trainer class for LlamaAudio models with comprehensive training features.
    
    Features:
    - DeepSpeed integration for large-scale training
    - TensorBoard logging
    - Distributed training support
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing and resuming
    - Validation and evaluation
    """
    
    def __init__(
        self,
        model_config: LlamaAudioConfig,
        training_config: Dict[str, Any],
        output_dir: str,
        deepspeed_config: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: LlamaAudio model configuration
            training_config: Training configuration dictionary
            output_dir: Output directory for checkpoints and logs
            deepspeed_config: Path to DeepSpeed configuration file
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.deepspeed_config = deepspeed_config
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seed
        set_seed(training_config.get("seed", 42))
        
        # Setup distributed training
        self.distributed_info = setup_distributed_training()
        self.is_main_process = self.distributed_info["rank"] == 0
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.audio_processor = None
        self._init_model()
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.deepspeed_engine = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float("inf")
        self.start_time = time.time()
        
        # TensorBoard writer
        self.tb_writer = None
        if self.is_main_process:
            self.tb_writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")
        
        # Setup training components
        self._setup_training()
        
        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            self._resume_training()
        
        logger.info("LlamaAudio Trainer initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _init_model(self):
        """Initialize the LlamaAudio model."""
        self.model = LlamaAudioModel(self.model_config)
        self.tokenizer = self.model.tokenizer
        self.audio_processor = WhisperProcessor.from_pretrained(
            self.model_config.whisper_model_name
        )
        
        # Move model to appropriate device
        device = get_device()
        if not self.deepspeed_config:
            self.model = self.model.to(device)
        
        # Log model information
        if self.is_main_process:
            log_model_info(self.model, logger)
    
    def _setup_training(self):
        """Setup training components."""
        # Create data loaders
        self._setup_data_loaders()
        
        # Calculate training steps
        self.num_training_steps = self._calculate_training_steps()
        self.num_warmup_steps = int(
            self.num_training_steps * self.training_config.get("warmup_ratio", 0.1)
        )
        
        # Setup DeepSpeed or regular training
        if self.deepspeed_config:
            self._setup_deepspeed_training()
        else:
            self._setup_regular_training()
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders."""
        train_data_path = self.training_config["train_data_path"]
        val_data_path = self.training_config.get("val_data_path")
        
        self.train_loader, self.val_loader = create_data_loaders(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            tokenizer=self.tokenizer,
            audio_processor=self.audio_processor,
            batch_size=self.training_config.get("batch_size", 8),
            max_text_length=self.training_config.get("max_text_length", 512),
            max_audio_length=self.training_config.get("max_audio_length", 30.0),
            num_workers=self.training_config.get("num_workers", 4),
        )
        
        logger.info(f"Data loaders created: train={len(self.train_loader)}, val={len(self.val_loader) if self.val_loader else 0}")
    
    def _calculate_training_steps(self) -> int:
        """Calculate total number of training steps."""
        num_epochs = self.training_config.get("num_epochs", 3)
        gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 1)
        
        steps_per_epoch = len(self.train_loader) // gradient_accumulation_steps
        total_steps = steps_per_epoch * num_epochs
        
        return total_steps
    
    def _setup_deepspeed_training(self):
        """Setup DeepSpeed training."""
        # Get trainable parameters
        model_parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize DeepSpeed
        self.deepspeed_engine, self.optimizer, self.scheduler, _ = setup_deepspeed(
            model=self.model,
            config_path=self.deepspeed_config,
            model_parameters=model_parameters,
        )
        
        logger.info("DeepSpeed training setup completed")
    
    def _setup_regular_training(self):
        """Setup regular (non-DeepSpeed) training."""
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            model=self.model,
            learning_rate=self.training_config.get("learning_rate", 2e-5),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            optimizer_type=self.training_config.get("optimizer_type", "adamw"),
            scheduler_type=self.training_config.get("scheduler_type", "cosine_with_warmup"),
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_warmup_steps,
        )
        
        # Setup distributed training
        if self.distributed_info["is_distributed"]:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.distributed_info["local_rank"]],
                output_device=self.distributed_info["local_rank"],
            )
        
        logger.info("Regular training setup completed")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.training_config.get('num_epochs', 3)}")
        logger.info(f"Total training steps: {self.num_training_steps}")
        logger.info(f"Warmup steps: {self.num_warmup_steps}")
        
        num_epochs = self.training_config.get("num_epochs", 3)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Validation epoch
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_epoch_checkpoint(train_metrics, val_metrics)
            
            # Early stopping check
            if self._should_early_stop(val_metrics):
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed!")
        
        # Save final model
        self._save_final_model()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_steps = 0
        gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        log_interval = self.training_config.get("log_interval", 100)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            if not self.deepspeed_config:
                batch = {k: v.to(get_device()) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Forward pass
            if self.deepspeed_config:
                outputs = self.deepspeed_engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_features=batch["audio_features"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_features=batch["audio_features"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.deepspeed_config:
                self.deepspeed_engine.backward(loss)
            else:
                loss.backward()
            
            # Update parameters
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.deepspeed_config:
                    self.deepspeed_engine.step()
                else:
                    # Clip gradients
                    grad_norm = clip_gradients(self.model, max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                self.current_step += 1
                
                # Log training metrics
                if self.current_step % log_interval == 0 and self.is_main_process:
                    self._log_training_step(loss.item() * gradient_accumulation_steps, grad_norm if not self.deepspeed_config else 0.0)
            
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # Save checkpoint at intervals
            if (self.current_step % self.training_config.get("save_interval", 1000) == 0 and 
                self.is_main_process):
                self._save_step_checkpoint()
        
        avg_epoch_loss = epoch_loss / epoch_steps
        return {"train_loss": avg_epoch_loss}
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                if not self.deepspeed_config:
                    batch = {k: v.to(get_device()) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Forward pass
                if self.deepspeed_config:
                    outputs = self.deepspeed_engine(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        audio_features=batch["audio_features"],
                        audio_attention_mask=batch["audio_attention_mask"],
                        labels=batch["labels"],
                    )
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        audio_features=batch["audio_features"],
                        audio_attention_mask=batch["audio_attention_mask"],
                        labels=batch["labels"],
                    )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
        
        avg_val_loss = total_loss / total_steps
        return {"val_loss": avg_val_loss}
    
    def _log_training_step(self, loss: float, grad_norm: float):
        """Log training step metrics."""
        # Calculate learning rate
        if self.deepspeed_config:
            lr = self.deepspeed_engine.get_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]["lr"]
        
        # Get memory usage
        memory_stats = get_memory_usage()
        
        # Calculate training speed
        elapsed_time = time.time() - self.start_time
        steps_per_second = self.current_step / elapsed_time if elapsed_time > 0 else 0
        
        # Log to console
        logger.info(
            f"Step {self.current_step:6d} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Grad Norm: {grad_norm:.4f} | "
            f"Memory: {memory_stats['allocated']:.1f}GB | "
            f"Steps/s: {steps_per_second:.2f}"
        )
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, self.current_step)
            self.tb_writer.add_scalar("train/learning_rate", lr, self.current_step)
            self.tb_writer.add_scalar("train/grad_norm", grad_norm, self.current_step)
            self.tb_writer.add_scalar("train/memory_allocated", memory_stats['allocated'], self.current_step)
            self.tb_writer.add_scalar("train/steps_per_second", steps_per_second, self.current_step)
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]):
        """Log epoch-level metrics."""
        elapsed_time = time.time() - self.start_time
        
        log_msg = (
            f"Epoch {self.current_epoch:3d}/{self.training_config.get('num_epochs', 3)} | "
            f"Train Loss: {train_metrics['train_loss']:.4f}"
        )
        
        if val_metrics:
            log_msg += f" | Val Loss: {val_metrics['val_loss']:.4f}"
        
        log_msg += f" | Time: {format_time(elapsed_time)}"
        
        logger.info(log_msg)
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("epoch/train_loss", train_metrics['train_loss'], self.current_epoch)
            if val_metrics:
                self.tb_writer.add_scalar("epoch/val_loss", val_metrics['val_loss'], self.current_epoch)
            self.tb_writer.add_scalar("epoch/elapsed_time", elapsed_time, self.current_epoch)
    
    def _save_epoch_checkpoint(self, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]):
        """Save checkpoint at the end of each epoch."""
        if not self.is_main_process:
            return
        
        val_loss = val_metrics['val_loss'] if val_metrics else train_metrics['train_loss']
        is_best = val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            loss=val_loss,
            save_path=str(checkpoint_path),
            is_best=is_best,
            deepspeed_engine=self.deepspeed_engine,
        )
    
    def _save_step_checkpoint(self):
        """Save checkpoint at specified step intervals."""
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.current_step}.pt"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.current_step,
            loss=0.0,  # Will be updated later
            save_path=str(checkpoint_path),
            deepspeed_engine=self.deepspeed_engine,
        )
    
    def _save_final_model(self):
        """Save the final trained model."""
        if not self.is_main_process:
            return
        
        final_model_path = self.output_dir / "final_model"
        
        if self.deepspeed_config:
            # Save DeepSpeed model
            self.deepspeed_engine.save_checkpoint(str(final_model_path), tag="final")
        else:
            # Save regular model
            if hasattr(self.model, 'module'):
                model_to_save = self.model.module
            else:
                model_to_save = self.model
            
            model_to_save.save_pretrained(str(final_model_path))
        
        # Save training config
        with open(final_model_path / "training_config.json", "w") as f:
            json.dump(self.training_config, f, indent=2)
        
        logger.info(f"Final model saved to {final_model_path}")
    
    def _should_early_stop(self, val_metrics: Optional[Dict[str, float]]) -> bool:
        """Check if early stopping should be triggered."""
        early_stopping_patience = self.training_config.get("early_stopping_patience")
        if early_stopping_patience is None or val_metrics is None:
            return False
        
        # Implement early stopping logic here if needed
        # For now, just return False
        return False
    
    def _resume_training(self):
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from {self.resume_from_checkpoint}")
        
        checkpoint_info = load_checkpoint(
            checkpoint_path=self.resume_from_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=get_device(),
        )
        
        self.current_epoch = checkpoint_info["epoch"]
        self.current_step = checkpoint_info["step"]
        self.best_val_loss = checkpoint_info["loss"]
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.current_step}")
    
    def evaluate(self, test_data_path: str) -> Dict[str, float]:
        """Evaluate the model on test data."""
        logger.info("Starting evaluation...")
        
        # Create test data loader
        test_dataset = AudioTextDataset(
            data_path=test_data_path,
            tokenizer=self.tokenizer,
            audio_processor=self.audio_processor,
            max_text_length=self.training_config.get("max_text_length", 512),
            max_audio_length=self.training_config.get("max_audio_length", 30.0),
        )
        
        test_loader = AudioTextDataLoader(
            dataset=test_dataset,
            batch_size=self.training_config.get("eval_batch_size", 8),
            shuffle=False,
            num_workers=self.training_config.get("num_workers", 4),
            drop_last=False,
        )
        
        # Evaluation loop
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if not self.deepspeed_config:
                    batch = {k: v.to(get_device()) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_features=batch["audio_features"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
        
        avg_test_loss = total_loss / total_steps
        
        logger.info(f"Evaluation completed. Test loss: {avg_test_loss:.4f}")
        
        return {"test_loss": avg_test_loss}
    
    def cleanup(self):
        """Cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.distributed_info["is_distributed"]:
            torch.distributed.destroy_process_group()
        
        logger.info("Trainer cleanup completed") 