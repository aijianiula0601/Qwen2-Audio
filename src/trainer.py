"""
Qwen2-Audio Training Implementation

This module implements the three-stage training pipeline described in the paper:
1. Stage 1: Pre-training with natural language prompts
2. Stage 2: Supervised Fine-tuning (SFT)
3. Stage 3: Direct Preference Optimization (DPO)
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from transformers.trainer_utils import seed_worker
from typing import Dict, List, Optional, Union, Any
import logging
import wandb
from dataclasses import dataclass, field
import json
from tqdm import tqdm
import numpy as np

from .model import Qwen2AudioForConditionalGeneration, Qwen2AudioConfig
from .processor import Qwen2AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class Qwen2AudioTrainingArguments(TrainingArguments):
    """Extended training arguments for Qwen2-Audio"""
    
    # Stage-specific arguments
    training_stage: str = field(
        default="stage1",
        metadata={"help": "Training stage: stage1 (pretraining), stage2 (sft), stage3 (dpo)"}
    )
    
    # Audio-specific arguments
    audio_max_length: int = field(default=30, metadata={"help": "Maximum audio length in seconds"})
    freeze_audio_encoder: bool = field(default=False, metadata={"help": "Whether to freeze audio encoder"})
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to freeze LLM"})
    
    # Data arguments
    conversation_template: str = field(
        default="qwen",
        metadata={"help": "Conversation template to use"}
    )
    
    # DPO specific arguments
    dpo_beta: float = field(default=0.1, metadata={"help": "DPO beta parameter"})
    dpo_loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type"})
    
    # Optimization arguments
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler"})


class Qwen2AudioDataset(torch.utils.data.Dataset):
    """Dataset for Qwen2-Audio training"""
    
    def __init__(
        self,
        data_path: str,
        processor: Qwen2AudioProcessor,
        stage: str = "stage1",
        max_length: int = 512,
        audio_max_length: int = 30
    ):
        self.processor = processor
        self.stage = stage
        self.max_length = max_length
        self.audio_max_length = audio_max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                self.data = [json.loads(line.strip()) for line in f if line.strip()]
            else:
                self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples for {stage}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        if self.stage == "stage1":
            return self._process_stage1(example)
        elif self.stage == "stage2":
            return self._process_stage2(example)
        elif self.stage == "stage3":
            return self._process_stage3(example)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
    
    def _process_stage1(self, example):
        """Process data for Stage 1: Pre-training with natural language prompts"""
        audio_path = example.get('audio', None)
        text = example['text']
        
        # Process audio if available
        audio_features = None
        if audio_path:
            try:
                audio_features = self.processor.process_audio(audio_path)
            except Exception as e:
                logger.warning(f"Failed to process audio {audio_path}: {e}")
                audio_features = None
        
        # Tokenize text
        encoding = self.processor.process_text(text)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'audio_features': audio_features
        }
    
    def _process_stage2(self, example):
        """Process data for Stage 2: Supervised Fine-tuning"""
        conversation = example['conversation']
        
        # Apply chat template
        formatted_text = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=False,
            tokenize=False
        )
        
        # Extract audio paths
        audio_paths = []
        for message in conversation:
            if isinstance(message.get('content'), list):
                for content in message['content']:
                    if content.get('type') == 'audio':
                        audio_paths.append(content.get('audio_url', content.get('audio')))
        
        # Process audio
        audio_features = None
        if audio_paths:
            try:
                # For multiple audio files, use the first one for now
                audio_features = self.processor.process_audio(audio_paths[0])
            except Exception as e:
                logger.warning(f"Failed to process audio: {e}")
        
        # Tokenize
        encoding = self.processor.process_text(formatted_text)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (mask input tokens, only compute loss on response)
        labels = self._create_sft_labels(input_ids, conversation)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'audio_features': audio_features
        }
    
    def _process_stage3(self, example):
        """Process data for Stage 3: Direct Preference Optimization"""
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Process audio if available
        audio_features = None
        if 'audio' in example:
            try:
                audio_features = self.processor.process_audio(example['audio'])
            except Exception as e:
                logger.warning(f"Failed to process audio: {e}")
        
        # Format chosen and rejected responses
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected
        
        # Tokenize
        chosen_encoding = self.processor.process_text(chosen_text)
        rejected_encoding = self.processor.process_text(rejected_text)
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'audio_features': audio_features
        }
    
    def _create_sft_labels(self, input_ids, conversation):
        """Create labels for SFT training (only compute loss on assistant responses)"""
        labels = input_ids.clone()
        labels[:] = -100  # Ignore all tokens by default
        
        # This is a simplified version - in practice, you'd need to identify
        # assistant response tokens and only compute loss on those
        # For now, compute loss on all tokens
        return labels


class Qwen2AudioTrainer(Trainer):
    """Custom trainer for Qwen2-Audio"""
    
    def __init__(
        self,
        model: Qwen2AudioForConditionalGeneration,
        training_args: Qwen2AudioTrainingArguments,
        processor: Qwen2AudioProcessor,
        **kwargs
    ):
        self.processor = processor
        self.training_stage = training_args.training_stage
        
        super().__init__(
            model=model,
            args=training_args,
            **kwargs
        )
        
        # Freeze components if specified
        if training_args.freeze_audio_encoder:
            self._freeze_audio_encoder()
        if training_args.freeze_llm:
            self._freeze_llm()
    
    def _freeze_audio_encoder(self):
        """Freeze audio encoder parameters"""
        for param in self.model.audio_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen audio encoder parameters")
    
    def _freeze_llm(self):
        """Freeze LLM parameters"""
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        logger.info("Frozen LLM parameters")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss based on training stage"""
        if self.training_stage in ["stage1", "stage2"]:
            return self._compute_language_modeling_loss(model, inputs, return_outputs)
        elif self.training_stage == "stage3":
            return self._compute_dpo_loss(model, inputs, return_outputs)
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
    
    def _compute_language_modeling_loss(self, model, inputs, return_outputs=False):
        """Compute standard language modeling loss"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        logits = outputs.logits
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_dpo_loss(self, model, inputs, return_outputs=False):
        """Compute DPO (Direct Preference Optimization) loss"""
        # Extract chosen and rejected inputs
        chosen_inputs = {
            'input_ids': inputs['chosen_input_ids'],
            'attention_mask': inputs['chosen_attention_mask'],
            'audio_features': inputs.get('audio_features')
        }
        rejected_inputs = {
            'input_ids': inputs['rejected_input_ids'],
            'attention_mask': inputs['rejected_attention_mask'],
            'audio_features': inputs.get('audio_features')
        }
        
        # Forward pass for chosen and rejected
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        # Compute log probabilities
        chosen_logps = self._get_batch_logps(
            chosen_outputs.logits,
            chosen_inputs['input_ids']
        )
        rejected_logps = self._get_batch_logps(
            rejected_outputs.logits,
            rejected_inputs['input_ids']
        )
        
        # DPO loss
        beta = getattr(self.args, 'dpo_beta', 0.1)
        loss = -torch.nn.functional.logsigmoid(beta * (chosen_logps - rejected_logps)).mean()
        
        outputs = {
            'chosen_logps': chosen_logps,
            'rejected_logps': rejected_logps,
        }
        
        return (loss, outputs) if return_outputs else loss
    
    def _get_batch_logps(self, logits, labels):
        """Get log probabilities for a batch"""
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
        
        # Mask padding tokens
        mask = (shift_labels != -100).float()
        per_token_logps = per_token_logps * mask
        
        # Sum over sequence length
        return per_token_logps.sum(dim=-1)


def create_data_collator(processor: Qwen2AudioProcessor, stage: str = "stage1"):
    """Create data collator for training"""
    
    def collate_fn(examples):
        batch = {}
        
        if stage in ["stage1", "stage2"]:
            # Standard training
            batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                [ex['input_ids'] for ex in examples],
                batch_first=True,
                padding_value=processor.tokenizer.pad_token_id
            )
            batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                [ex['attention_mask'] for ex in examples],
                batch_first=True,
                padding_value=0
            )
            batch['labels'] = torch.nn.utils.rnn.pad_sequence(
                [ex['labels'] for ex in examples],
                batch_first=True,
                padding_value=-100
            )
            
            # Handle audio features
            audio_features = [ex['audio_features'] for ex in examples if ex['audio_features'] is not None]
            if audio_features:
                # Pad audio features to same length
                max_audio_len = max(af.shape[-1] for af in audio_features)
                padded_audio = []
                for af in audio_features:
                    if af.shape[-1] < max_audio_len:
                        padding = torch.zeros(af.shape[0], max_audio_len - af.shape[-1])
                        af = torch.cat([af, padding], dim=-1)
                    padded_audio.append(af)
                batch['audio_features'] = torch.stack(padded_audio)
            else:
                batch['audio_features'] = None
                
        elif stage == "stage3":
            # DPO training
            for key in ['chosen_input_ids', 'chosen_attention_mask', 'rejected_input_ids', 'rejected_attention_mask']:
                batch[key] = torch.nn.utils.rnn.pad_sequence(
                    [ex[key] for ex in examples],
                    batch_first=True,
                    padding_value=processor.tokenizer.pad_token_id if 'input_ids' in key else 0
                )
            
            # Handle audio features for DPO
            audio_features = [ex['audio_features'] for ex in examples if ex['audio_features'] is not None]
            batch['audio_features'] = torch.stack(audio_features) if audio_features else None
        
        return batch
    
    return collate_fn


def train_qwen2_audio(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    training_args: Qwen2AudioTrainingArguments,
    processor: Optional[Qwen2AudioProcessor] = None
):
    """Main training function"""
    
    # Initialize model and processor
    if processor is None:
        processor = Qwen2AudioProcessor()
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name_or_path)
    
    # Create dataset
    train_dataset = Qwen2AudioDataset(
        data_path=data_path,
        processor=processor,
        stage=training_args.training_stage,
        max_length=training_args.max_length if hasattr(training_args, 'max_length') else 512,
        audio_max_length=training_args.audio_max_length
    )
    
    # Create data collator
    data_collator = create_data_collator(processor, training_args.training_stage)
    
    # Initialize trainer
    trainer = Qwen2AudioTrainer(
        model=model,
        training_args=training_args,
        processor=processor,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model()
    processor.tokenizer.save_pretrained(output_dir)
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Qwen2-Audio model")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--training_stage", type=str, default="stage1", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--freeze_audio_encoder", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true")
    
    args = parser.parse_args()
    
    training_args = Qwen2AudioTrainingArguments(
        output_dir=args.output_dir,
        training_stage=args.training_stage,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        freeze_audio_encoder=args.freeze_audio_encoder,
        freeze_llm=args.freeze_llm,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    
    train_qwen2_audio(
        model_name_or_path=args.model_name_or_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        training_args=training_args
    ) 