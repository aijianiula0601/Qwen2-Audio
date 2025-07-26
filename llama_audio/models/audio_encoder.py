"""
Audio Encoder based on Whisper for LlamaAudio framework.

This module implements the audio encoding component that converts
audio signals into feature representations compatible with Llama models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig
import logging
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)


class AudioEncoder(nn.Module):
    """
    Audio encoder based on Whisper encoder.
    
    This encoder extracts audio features and projects them to the LLM embedding space.
    """
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-large-v3",
        llm_hidden_size: int = 4096,  # Default for Llama-7B
        freeze_whisper: bool = True,
        use_conv_adapter: bool = True,
        adapter_hidden_size: int = 2048,
    ):
        """
        Initialize the audio encoder.
        
        Args:
            whisper_model_name: Pretrained Whisper model name
            llm_hidden_size: Hidden size of the target LLM
            freeze_whisper: Whether to freeze Whisper parameters
            use_conv_adapter: Whether to use convolutional adapter
            adapter_hidden_size: Hidden size of the adapter layers
        """
        super().__init__()
        
        self.whisper_model_name = whisper_model_name
        self.llm_hidden_size = llm_hidden_size
        self.freeze_whisper = freeze_whisper
        
        # Load Whisper encoder
        self.whisper_config = WhisperConfig.from_pretrained(whisper_model_name)
        self.whisper_encoder = WhisperModel.from_pretrained(
            whisper_model_name
        ).encoder
        
        # Freeze Whisper parameters if requested
        if freeze_whisper:
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False
            logger.info("Whisper encoder parameters frozen")
        
        # Audio feature adapter
        whisper_hidden_size = self.whisper_config.d_model
        
        if use_conv_adapter:
            # Convolutional adapter for temporal compression
            self.conv_adapter = nn.Sequential(
                nn.Conv1d(whisper_hidden_size, adapter_hidden_size, 
                         kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(adapter_hidden_size, adapter_hidden_size,
                         kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(adapter_hidden_size, llm_hidden_size,
                         kernel_size=3, stride=1, padding=1),
            )
        else:
            self.conv_adapter = None
            
        # Linear projection to LLM embedding space
        self.audio_projector = nn.Sequential(
            nn.Linear(whisper_hidden_size, adapter_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_size, llm_hidden_size),
        )
        
        # Learnable audio tokens for better alignment
        self.audio_start_token = nn.Parameter(torch.randn(1, 1, llm_hidden_size))
        self.audio_end_token = nn.Parameter(torch.randn(1, 1, llm_hidden_size))
        
    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the audio encoder.
        
        Args:
            audio_features: Audio input features [batch_size, seq_len, feat_dim]
            attention_mask: Attention mask for audio features
            
        Returns:
            Tuple of (audio_embeddings, audio_attention_mask)
        """
        batch_size = audio_features.shape[0]
        
        # Extract features using Whisper encoder
        with torch.set_grad_enabled(not self.freeze_whisper):
            whisper_outputs = self.whisper_encoder(
                input_features=audio_features,
                attention_mask=attention_mask,
            )
            
        audio_hidden_states = whisper_outputs.last_hidden_state  # [B, T, D]
        
        # Apply convolutional adapter if available
        if self.conv_adapter is not None:
            # Conv1d expects [B, D, T]
            audio_hidden_states = audio_hidden_states.transpose(1, 2)
            audio_hidden_states = self.conv_adapter(audio_hidden_states)
            audio_hidden_states = audio_hidden_states.transpose(1, 2)  # Back to [B, T, D]
            
            # Update attention mask for temporal compression
            if attention_mask is not None:
                # Downsample attention mask to match compressed sequence length
                original_length = attention_mask.shape[1]
                new_length = audio_hidden_states.shape[1]
                if new_length != original_length:
                    attention_mask = F.interpolate(
                        attention_mask.float().unsqueeze(1),
                        size=new_length,
                        mode='nearest'
                    ).squeeze(1).bool()
        
        # Project to LLM embedding space
        audio_embeddings = self.audio_projector(audio_hidden_states)  # [B, T, llm_hidden_size]
        
        # Add start and end tokens
        start_tokens = self.audio_start_token.expand(batch_size, -1, -1)
        end_tokens = self.audio_end_token.expand(batch_size, -1, -1)
        
        audio_embeddings = torch.cat([
            start_tokens,
            audio_embeddings,
            end_tokens
        ], dim=1)
        
        # Update attention mask for added tokens
        if attention_mask is not None:
            start_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            end_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([start_mask, attention_mask, end_mask], dim=1)
        
        return audio_embeddings, attention_mask
    
    def get_audio_feature_length(self, audio_length: int) -> int:
        """
        Calculate the output sequence length after processing.
        
        Args:
            audio_length: Input audio sequence length
            
        Returns:
            Output sequence length
        """
        if self.conv_adapter is not None:
            # Two conv layers with stride 2, then one with stride 1
            audio_length = (audio_length + 1) // 2  # First conv
            audio_length = (audio_length + 1) // 2  # Second conv
            # Third conv has stride 1, so no change
        
        # Add 2 for start and end tokens
        return audio_length + 2 