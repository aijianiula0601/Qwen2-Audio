"""
Qwen2-Audio Model Implementation

This module implements the Qwen2-Audio model as described in the paper:
"Qwen2-Audio Technical Report" (arXiv:2407.10759)

The model consists of:
1. Audio Encoder (based on Whisper-large-v3)
2. Large Language Model (Qwen-7B)
3. Audio-Language alignment layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WhisperModel,
    Qwen2Model, 
    Qwen2ForCausalLM,
    Qwen2Config,
    PreTrainedModel,
    AutoConfig,
    AutoModel
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class Qwen2AudioConfig:
    """Configuration for Qwen2-Audio model"""
    
    def __init__(
        self,
        audio_encoder_model_name: str = "openai/whisper-large-v3",
        llm_model_name: str = "Qwen/Qwen2-7B",
        audio_hidden_size: int = 1280,
        llm_hidden_size: int = 4096,
        max_audio_length: int = 30,  # seconds
        sampling_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 160,  # 10ms
        win_length: int = 400,  # 25ms
        audio_pooling_stride: int = 2,
        **kwargs
    ):
        self.audio_encoder_model_name = audio_encoder_model_name
        self.llm_model_name = llm_model_name
        self.audio_hidden_size = audio_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.max_audio_length = max_audio_length
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.audio_pooling_stride = audio_pooling_stride
        
        for key, value in kwargs.items():
            setattr(self, key, value)


class AudioEncoder(nn.Module):
    """Audio encoder based on Whisper-large-v3"""
    
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__()
        self.config = config
        
        # Load Whisper encoder
        self.whisper = WhisperModel.from_pretrained(
            config.audio_encoder_model_name
        ).encoder
        
        # Pooling layer to reduce sequence length
        self.pooling = nn.AvgPool1d(
            kernel_size=config.audio_pooling_stride,
            stride=config.audio_pooling_stride
        )
        
        # Projection layer to match LLM hidden size
        self.audio_projection = nn.Linear(
            config.audio_hidden_size,
            config.llm_hidden_size
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [batch_size, n_mels, seq_len]
        Returns:
            audio_embeddings: [batch_size, seq_len//stride, llm_hidden_size]
        """
        # Whisper expects [batch_size, n_mels, seq_len]
        encoder_outputs = self.whisper(audio_features)
        audio_embeddings = encoder_outputs.last_hidden_state
        
        # Apply pooling to reduce sequence length
        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
        audio_embeddings = audio_embeddings.transpose(1, 2)
        pooled_embeddings = self.pooling(audio_embeddings)
        # [batch_size, hidden_size, seq_len//stride] -> [batch_size, seq_len//stride, hidden_size]
        pooled_embeddings = pooled_embeddings.transpose(1, 2)
        
        # Project to LLM hidden size
        projected_embeddings = self.audio_projection(pooled_embeddings)
        
        return projected_embeddings


class Qwen2AudioForConditionalGeneration(PreTrainedModel):
    """
    Qwen2-Audio model for conditional generation
    
    This model combines an audio encoder (Whisper) with a language model (Qwen2)
    for audio-language understanding and generation tasks.
    """
    
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize audio encoder
        self.audio_encoder = AudioEncoder(config)
        
        # Initialize language model
        self.language_model = Qwen2ForCausalLM.from_pretrained(
            config.llm_model_name
        )
        
        # Special tokens for audio
        self.audio_start_token_id = None
        self.audio_end_token_id = None
        self.audio_token_id = None
        
        # Initialize special tokens
        self._init_special_tokens()
        
    def _init_special_tokens(self):
        """Initialize special tokens for audio input"""
        tokenizer = self.language_model.config.tokenizer if hasattr(self.language_model.config, 'tokenizer') else None
        
        # Add special tokens if they don't exist
        special_tokens = {
            '<|audio_bos|>': 'audio_start_token_id',
            '<|audio_eos|>': 'audio_end_token_id', 
            '<|AUDIO|>': 'audio_token_id'
        }
        
        # For now, use placeholder token IDs
        self.audio_start_token_id = 151643  # placeholder
        self.audio_end_token_id = 151644    # placeholder
        self.audio_token_id = 151645        # placeholder
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Prepare inputs for generation"""
        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
        }
        model_inputs.update(kwargs)
        return model_inputs
        
    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of Qwen2-Audio model
        
        Args:
            input_ids: [batch_size, seq_len] - Text token IDs
            audio_features: [batch_size, n_mels, audio_seq_len] - Audio mel spectrograms
            attention_mask: [batch_size, seq_len] - Attention mask for text
            labels: [batch_size, seq_len] - Labels for language modeling loss
            
        Returns:
            CausalLMOutputWithPast with loss, logits, etc.
        """
        batch_size, seq_len = input_ids.shape
        
        # Get text embeddings from language model
        text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        # Process audio if provided
        if audio_features is not None:
            audio_embeddings = self.audio_encoder(audio_features)
            
            # Find audio token positions in input_ids
            audio_token_positions = (input_ids == self.audio_token_id).nonzero(as_tuple=True)
            
            if len(audio_token_positions[0]) > 0:
                # Replace audio tokens with audio embeddings
                for i, (batch_idx, seq_idx) in enumerate(zip(*audio_token_positions)):
                    if i < audio_embeddings.shape[1]:  # Ensure we don't exceed audio sequence length
                        text_embeddings[batch_idx, seq_idx] = audio_embeddings[batch_idx, i]
        
        # Create combined attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=text_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        max_length: int = 256,
        **kwargs
    ):
        """Generate text given audio and text inputs"""
        
        # Prepare inputs
        model_inputs = self.prepare_inputs_for_generation(
            input_ids=input_ids,
            audio_features=audio_features,
            **kwargs
        )
        
        # Use language model's generate method with custom forward
        def custom_forward(**model_kwargs):
            return self.forward(**model_kwargs)
            
        # Temporarily replace forward method
        original_forward = self.language_model.forward
        self.language_model.forward = custom_forward
        
        try:
            generated_ids = self.language_model.generate(
                inputs_embeds=self._get_embeddings_for_generation(
                    input_ids, audio_features
                ),
                max_length=max_length,
                **kwargs
            )
        finally:
            # Restore original forward method
            self.language_model.forward = original_forward
            
        return generated_ids
    
    def _get_embeddings_for_generation(
        self, 
        input_ids: torch.Tensor, 
        audio_features: Optional[torch.Tensor] = None
    ):
        """Get embeddings for generation"""
        text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        if audio_features is not None:
            audio_embeddings = self.audio_encoder(audio_features)
            
            # Find audio token positions
            audio_token_positions = (input_ids == self.audio_token_id).nonzero(as_tuple=True)
            
            if len(audio_token_positions[0]) > 0:
                for i, (batch_idx, seq_idx) in enumerate(zip(*audio_token_positions)):
                    if i < audio_embeddings.shape[1]:
                        text_embeddings[batch_idx, seq_idx] = audio_embeddings[batch_idx, i]
        
        return text_embeddings


class Qwen2AudioModel(nn.Module):
    """
    Wrapper class for easier usage
    """
    
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__()
        self.model = Qwen2AudioForConditionalGeneration(config)
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


# Factory functions for easy model creation
def create_qwen2_audio_model(
    audio_encoder_name: str = "openai/whisper-large-v3",
    llm_name: str = "Qwen/Qwen2-7B"
) -> Qwen2AudioForConditionalGeneration:
    """Create a Qwen2-Audio model with default configuration"""
    
    config = Qwen2AudioConfig(
        audio_encoder_model_name=audio_encoder_name,
        llm_model_name=llm_name
    )
    
    return Qwen2AudioForConditionalGeneration(config)


def load_pretrained_qwen2_audio(model_path: str) -> Qwen2AudioForConditionalGeneration:
    """Load a pretrained Qwen2-Audio model"""
    
    # Load configuration
    config = Qwen2AudioConfig()  # This should be loaded from saved config
    
    # Create model
    model = Qwen2AudioForConditionalGeneration(config)
    
    # Load pretrained weights
    if model_path:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    return model 