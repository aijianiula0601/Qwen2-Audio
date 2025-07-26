"""
Data loading utilities for LlamaAudio training.

This module provides data loaders and datasets for multimodal audio-text training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import json
import os
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class AudioTextDataset(Dataset):
    """
    Dataset for audio-text pairs used in LlamaAudio training.
    
    Supports various audio formats and text instructions.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        audio_processor,
        max_text_length: int = 512,
        max_audio_length: int = 30.0,  # seconds
        sample_rate: int = 16000,
        audio_format: str = "whisper",  # "whisper" or "raw"
        instruction_templates: Optional[List[str]] = None,
        include_audio_tokens: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset JSON file or directory
            tokenizer: Tokenizer for text processing
            audio_processor: Audio processor (e.g., WhisperProcessor)
            max_text_length: Maximum text sequence length
            max_audio_length: Maximum audio duration in seconds
            sample_rate: Audio sample rate
            audio_format: Audio processing format
            instruction_templates: Templates for instruction formatting
            include_audio_tokens: Whether to include audio tokens in text
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.include_audio_tokens = include_audio_tokens
        
        # Default instruction templates
        self.instruction_templates = instruction_templates or [
            "Listen to the audio and respond: <audio>",
            "What do you hear in this audio? <audio>",
            "Describe the audio: <audio>",
            "Answer based on the audio: <audio>",
            "<audio> What is this?",
            "<audio> Please describe what you hear.",
        ]
        
        # Load dataset
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} audio-text pairs from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset from file or directory."""
        data = []
        
        if os.path.isfile(self.data_path):
            # Single JSON file
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.endswith('.jsonl'):
                    # JSONL format
                    for line in f:
                        data.append(json.loads(line.strip()))
                else:
                    # Regular JSON
                    data = json.load(f)
        
        elif os.path.isdir(self.data_path):
            # Directory with multiple files
            for file_path in Path(self.data_path).rglob("*.json*"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            data.append(json.loads(line.strip()))
                    else:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            data.extend(file_data)
                        else:
                            data.append(file_data)
        
        else:
            raise ValueError(f"Data path {self.data_path} not found")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample."""
        item = self.data[idx]
        
        # Load and process audio
        audio_path = item.get('audio_path', item.get('audio'))
        if audio_path and os.path.exists(audio_path):
            audio_features = self._process_audio(audio_path)
        else:
            # Create dummy audio if not available
            audio_features = torch.zeros(80, 3000)  # Whisper log-mel features
            logger.warning(f"Audio file not found for item {idx}: {audio_path}")
        
        # Process text
        text_data = self._process_text(item)
        
        return {
            "audio_features": audio_features,
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
            "labels": text_data["labels"],
            "text": text_data["text"],
            "audio_path": audio_path,
        }
    
    def _process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio file and extract features."""
        try:
            if self.audio_format == "whisper":
                # Use Whisper processor
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                # Trim to max length
                max_samples = int(self.max_audio_length * self.sample_rate)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                
                # Extract log-mel features
                audio_features = self.audio_processor(
                    audio, 
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                ).input_features.squeeze(0)
                
                return audio_features
            
            elif self.audio_format == "raw":
                # Raw audio waveform
                audio, sr = torchaudio.load(audio_path)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    audio = resampler(audio)
                
                # Convert to mono if stereo
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                
                # Trim to max length
                max_samples = int(self.max_audio_length * self.sample_rate)
                if audio.shape[1] > max_samples:
                    audio = audio[:, :max_samples]
                
                return audio.squeeze(0)
            
            else:
                raise ValueError(f"Unsupported audio format: {self.audio_format}")
        
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            # Return dummy features
            if self.audio_format == "whisper":
                return torch.zeros(80, 3000)
            else:
                return torch.zeros(int(self.max_audio_length * self.sample_rate))
    
    def _process_text(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process text data."""
        # Extract text components
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')
        conversation = item.get('conversation', [])
        
        # Format text based on available data
        if conversation:
            # Multi-turn conversation
            text = self._format_conversation(conversation)
        elif instruction and output_text:
            # Instruction-following format
            if self.include_audio_tokens:
                template = random.choice(self.instruction_templates)
                if '<audio>' in template:
                    instruction = template.replace('<audio>', '<audio>')
                else:
                    instruction = f"<audio> {instruction}"
            
            text = f"### Instruction: {instruction}\n### Response: {output_text}"
        
        elif input_text and output_text:
            # Input-output pairs
            if self.include_audio_tokens:
                input_text = f"<audio> {input_text}"
            text = f"{input_text}\n{output_text}"
        
        else:
            # Default format
            text = output_text or input_text or instruction
            if self.include_audio_tokens and '<audio>' not in text:
                text = f"<audio> {text}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "text": text,
        }
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format multi-turn conversation."""
        formatted_turns = []
        
        for turn in conversation:
            role = turn.get('role', turn.get('from', 'user'))
            content = turn.get('content', turn.get('value', ''))
            
            if role in ['user', 'human']:
                if self.include_audio_tokens and '<audio>' not in content:
                    content = f"<audio> {content}"
                formatted_turns.append(f"User: {content}")
            elif role in ['assistant', 'gpt', 'ai']:
                formatted_turns.append(f"Assistant: {content}")
            else:
                formatted_turns.append(f"{role}: {content}")
        
        return "\n".join(formatted_turns)


class AudioTextDataLoader:
    """Data loader wrapper for AudioTextDataset with collation."""
    
    def __init__(
        self,
        dataset: AudioTextDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset: AudioTextDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        # Separate components
        audio_features = [item["audio_features"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad audio features
        max_audio_len = max(af.shape[-1] for af in audio_features)
        padded_audio = []
        audio_attention_masks = []
        
        for af in audio_features:
            seq_len = af.shape[-1]
            if len(af.shape) == 2:  # [feat_dim, seq_len]
                pad_width = max_audio_len - seq_len
                padded = F.pad(af, (0, pad_width))
                padded_audio.append(padded)
                
                # Create attention mask
                mask = torch.cat([
                    torch.ones(seq_len),
                    torch.zeros(pad_width)
                ])
                audio_attention_masks.append(mask)
            
            elif len(af.shape) == 1:  # [seq_len] for raw audio
                pad_width = max_audio_len - seq_len
                padded = F.pad(af, (0, pad_width))
                padded_audio.append(padded)
                
                mask = torch.cat([
                    torch.ones(seq_len),
                    torch.zeros(pad_width)
                ])
                audio_attention_masks.append(mask)
        
        # Stack tensors
        audio_features_batch = torch.stack(padded_audio)
        audio_attention_mask_batch = torch.stack(audio_attention_masks)
        input_ids_batch = torch.stack(input_ids)
        attention_mask_batch = torch.stack(attention_mask)
        labels_batch = torch.stack(labels)
        
        return {
            "audio_features": audio_features_batch,
            "audio_attention_mask": audio_attention_mask_batch,
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_data_loaders(
    train_data_path: str,
    val_data_path: Optional[str],
    tokenizer,
    audio_processor,
    batch_size: int = 8,
    max_text_length: int = 512,
    max_audio_length: float = 30.0,
    num_workers: int = 4,
    **kwargs
) -> Tuple[AudioTextDataLoader, Optional[AudioTextDataLoader]]:
    """
    Create training and validation data loaders.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data (optional)
        tokenizer: Text tokenizer
        audio_processor: Audio processor
        batch_size: Batch size
        max_text_length: Maximum text length
        max_audio_length: Maximum audio duration
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = AudioTextDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        max_text_length=max_text_length,
        max_audio_length=max_audio_length,
        **kwargs
    )
    
    train_loader = AudioTextDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    # Validation dataset (optional)
    val_loader = None
    if val_data_path:
        val_dataset = AudioTextDataset(
            data_path=val_data_path,
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            max_text_length=max_text_length,
            max_audio_length=max_audio_length,
            **kwargs
        )
        
        val_loader = AudioTextDataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader) if val_loader else 0}")
    
    return train_loader, val_loader 