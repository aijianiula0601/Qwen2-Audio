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
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class Qwen2AudioConfig(PretrainedConfig):
    """Configuration for Qwen2-Audio model"""
    
    model_type = "qwen2_audio"
    
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
        super().__init__(**kwargs)
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


class AudioEncoder(nn.Module):
    """Audio encoder based on Whisper-large-v3"""
    
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__()
        self.config = config
        
        # 加载 Whisper 编码器
        print("Loading Whisper model...")
        self.whisper = WhisperModel.from_pretrained(
            config.audio_encoder_model_name,
            device_map='auto'  # 自动选择设备
        ).encoder
        
        # 验证模型是否正确加载
        print("Model parameters:", sum(p.numel() for p in self.whisper.parameters()))
        print("Model device:", next(self.whisper.parameters()).device)
        print("Model dtype:", next(self.whisper.parameters()).dtype)
        
        # 确保模型在正确的设备上
        self.whisper = self.whisper.to(torch.float32)  # 确保使用 float32
        
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
        # 1. 检查输入
        print("1. Input audio_features stats:")
        print(f"   shape: {audio_features.shape}")
        print(f"   min/max: {audio_features.min().item()}, {audio_features.max().item()}")
        print(f"   mean/std: {audio_features.mean().item()}, {audio_features.std().item()}")
        
        # 2. Whisper 编码器处理
        encoder_outputs = self.whisper(audio_features)
        audio_embeddings = encoder_outputs.last_hidden_state
        print("2. After Whisper encoder:")
        print(f"   shape: {audio_embeddings.shape}")
        print(f"   min/max: {audio_embeddings.min().item()}, {audio_embeddings.max().item()}")
        print(f"   mean/std: {audio_embeddings.mean().item()}, {audio_embeddings.std().item()}")
        
        # 3. 池化层处理
        audio_embeddings = audio_embeddings.transpose(1, 2)
        pooled_embeddings = self.pooling(audio_embeddings)
        pooled_embeddings = pooled_embeddings.transpose(1, 2)
        print("3. After pooling:")
        print(f"   shape: {pooled_embeddings.shape}")
        print(f"   min/max: {pooled_embeddings.min().item()}, {pooled_embeddings.max().item()}")
        print(f"   mean/std: {pooled_embeddings.mean().item()}, {pooled_embeddings.std().item()}")
        
        # 4. 投影层处理
        projected_embeddings = self.audio_projection(pooled_embeddings)
        print("4. After projection:")
        print(f"   shape: {projected_embeddings.shape}")
        print(f"   min/max: {projected_embeddings.min().item()}, {projected_embeddings.max().item()}")
        print(f"   mean/std: {projected_embeddings.mean().item()}, {projected_embeddings.std().item()}")
        
        return projected_embeddings


class Qwen2AudioForConditionalGeneration(PreTrainedModel):
    """
    Qwen2-Audio model for conditional generation
    
    This model combines an audio encoder (Whisper) with a language model (Qwen2)
    for audio-language understanding and generation tasks.
    """
    
    config_class = Qwen2AudioConfig
    base_model_prefix = "qwen2_audio"
    
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
        self.audio_start_token_id = 151646
        self.audio_end_token_id = 151647
        self.audio_token_id = 151648
        
    def update_special_tokens(self, tokenizer):
        """Update special token IDs from tokenizer
        
        Args:
            tokenizer: The tokenizer instance with special tokens added
        """
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_end_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        
        print(f"✅ Updated special token IDs from tokenizer:")
        print(f"  audio_start_token_id: {self.audio_start_token_id}")
        print(f"  audio_end_token_id: {self.audio_end_token_id}")
        print(f"  audio_token_id: {self.audio_token_id}")
        
        # Validate that tokens were found
        if self.audio_token_id == tokenizer.unk_token_id:
            print("⚠️  Warning: <|AUDIO|> token not found in tokenizer vocabulary!")
        if self.audio_start_token_id == tokenizer.unk_token_id:
            print("⚠️  Warning: <|audio_bos|> token not found in tokenizer vocabulary!")
        if self.audio_end_token_id == tokenizer.unk_token_id:
            print("⚠️  Warning: <|audio_eos|> token not found in tokenizer vocabulary!")
        
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
        
        # Get combined embeddings using the same logic as generation
        inputs_embeds = self._get_embeddings_for_generation(input_ids, audio_features)
        
        # Create position IDs for the combined sequence
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        
        # Update attention mask and labels for the combined sequence if audio is present
        if audio_features is not None:
            # Find the position of audio token
            audio_token_positions = (input_ids == self.audio_token_id).nonzero(as_tuple=True)
            if len(audio_token_positions[0]) > 0:
                insert_pos = audio_token_positions[1][0] + 1  # +1 to insert after the audio token
                audio_seq_len = self.audio_encoder(audio_features).shape[1]
                total_seq_len = input_ids.shape[1] + audio_seq_len
                
                # Create new attention mask
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                new_attention_mask = torch.zeros(
                    (input_ids.shape[0], total_seq_len),
                    device=attention_mask.device
                )
                new_attention_mask[:, :insert_pos] = attention_mask[:, :insert_pos]
                new_attention_mask[:, insert_pos:insert_pos + audio_seq_len] = 1
                new_attention_mask[:, insert_pos + audio_seq_len:] = attention_mask[:, insert_pos:]
                attention_mask = new_attention_mask
                
                # Update labels if provided
                if labels is not None:
                    new_labels = torch.full(
                        (input_ids.shape[0], total_seq_len),
                        -100,
                        device=labels.device
                    )
                    new_labels[:, :insert_pos] = labels[:, :insert_pos]
                    new_labels[:, insert_pos + audio_seq_len:] = labels[:, insert_pos:]
                    labels = new_labels
        
        # If no audio features, use original attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        
        # Get the initial embeddings that combine text and audio
        inputs_embeds = self._get_embeddings_for_generation(input_ids, audio_features)
        
        # Create position IDs for the combined sequence
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        # Use the language model's generate method directly with inputs_embeds
        # No need to replace the forward method
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            max_length=max_length,
            **kwargs
        )
        
        return generated_ids
    
    def _get_embeddings_for_generation(
        self, 
        input_ids: torch.Tensor, 
        audio_features: Optional[torch.Tensor] = None
    ):
        """Get embeddings for generation by properly aligning audio and text features
        
        Args:
            input_ids: [batch_size, seq_len] - Text token IDs
            audio_features: [batch_size, n_mels, audio_seq_len] - Audio mel spectrograms
            
        Returns:
            combined_embeddings: [batch_size, total_seq_len, hidden_size] - Combined audio and text embeddings
        """

        # 检查输入
        if torch.isnan(input_ids).any():
            raise ValueError("input_ids contains NaN values")
        if audio_features is not None and torch.isnan(audio_features).any():
            raise ValueError("audio_features contains NaN values")
    

        # Get text embeddings
        text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        if audio_features is not None:
            # Get audio embeddings
            audio_embeddings = self.audio_encoder(audio_features)  # [batch_size, audio_seq_len, hidden_size]
            # Find the position of audio start token
            audio_start_positions = (input_ids == self.audio_token_id).nonzero(as_tuple=True)
            
            if len(audio_start_positions[0]) > 0:
                # Get the position where audio should be inserted
                insert_pos = audio_start_positions[1][0] + 1  # +1 to insert after the start token
                
                # Create a new tensor to hold combined embeddings
                batch_size, text_seq_len, hidden_size = text_embeddings.shape
                audio_seq_len = audio_embeddings.shape[1]
                total_seq_len = text_seq_len + audio_seq_len
                
                # Create combined embeddings
                combined_embeddings = torch.zeros(
                    (batch_size, total_seq_len, hidden_size),
                    device=text_embeddings.device
                )
                
                # Copy text embeddings before audio
                combined_embeddings[:, :insert_pos] = text_embeddings[:, :insert_pos]
                
                # Insert audio embeddings
                combined_embeddings[:, insert_pos:insert_pos + audio_seq_len] = audio_embeddings
                
                # Copy remaining text embeddings
                combined_embeddings[:, insert_pos + audio_seq_len:] = text_embeddings[:, insert_pos:]
                
                return combined_embeddings

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