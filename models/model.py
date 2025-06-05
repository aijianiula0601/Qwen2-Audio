import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    WhisperModel, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig
)
from typing import Optional, Tuple, List, Dict, Any
import math
import os


class AudioProjector(nn.Module):
    """Audio feature projector to map Whisper features to LLM hidden space"""
    
    def __init__(self, config):
        super().__init__()
        self.input_size = config.get('input_size', 1280)  # Whisper-large-v3 feature size
        self.hidden_size = config.get('hidden_size', 4096)
        self.intermediate_size = config.get('intermediate_size', 16384)
        self.num_layers = config.get('num_layers', 2)
        self.dtype = config.get('dtype', torch.bfloat16)
        layers = []
        current_size = self.input_size
        
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(current_size, self.intermediate_size, dtype=self.dtype))
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(self.intermediate_size, self.hidden_size, dtype=self.dtype))
            else:
                layers.append(nn.Linear(self.intermediate_size, self.intermediate_size, dtype=self.dtype))
                
            if i < self.num_layers - 1:
                layers.append(nn.GELU(approximate='tanh'))
                layers.append(nn.Dropout(0.1))
                
            current_size = self.intermediate_size if i == 0 else self.intermediate_size
            
        self.projector = nn.Sequential(*layers)
        
    def forward(self, audio_features):
        """
        Args:
            audio_features: [batch_size, sequence_length, input_size]
        Returns:
            projected_features: [batch_size, sequence_length, hidden_size]
        """
        return self.projector(audio_features)


class Qwen2AudioConfig(PretrainedConfig):
    """Configuration class for Qwen2-Audio model"""
    
    model_type = "qwen2_audio"
    
    def __init__(
        self,
        audio_encoder_name="openai/whisper-large-v3",
        llm_name="Qwen/Qwen2-7B",
        llm_type="qwen2",
        audio_projector_config=None,
        freeze_audio_encoder=False,
        freeze_llm=False,
        audio_start_token="<|audio_bos|>",
        audio_end_token="<|audio_eos|>",
        hidden_size=896,  # Add hidden_size for DeepSpeed auto-config
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_encoder_name = audio_encoder_name
        self.llm_name = llm_name
        self.llm_type = llm_type
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_llm = freeze_llm
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.hidden_size = hidden_size  # Add this for DeepSpeed
        
        if audio_projector_config is None:
            audio_projector_config = {
                'input_size': 1280,
                'hidden_size': hidden_size,  # Use the hidden_size parameter
                'intermediate_size': hidden_size * 4,  # 4 * hidden_size
                'num_layers': 2
            }
        # Ensure hidden_size in projector config matches
        audio_projector_config['hidden_size'] = hidden_size
        self.audio_projector_config = audio_projector_config


class Qwen2AudioModel(PreTrainedModel):
    """
    Qwen2-Audio model combining Whisper audio encoder with various LLM backbones
    """
    config_class = Qwen2AudioConfig
    
    def __init__(self, config: Qwen2AudioConfig):
        super().__init__(config)
        self.config = config
        
        torch_dtype = getattr(config, 'dtype', torch.float32)
        
        # Initialize audio encoder (Whisper)
        self.audio_encoder = WhisperModel.from_pretrained(
            config.audio_encoder_name,
            torch_dtype=torch_dtype
        ).encoder
        
        # Freeze audio encoder if specified
        if config.freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
                
        # Initialize LLM backbone
        # Check if it's a local path
        is_local_path = os.path.exists(config.llm_name) and os.path.isdir(config.llm_name)
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=is_local_path
        )
        
        # Freeze LLM if specified
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Initialize audio projector
        self.audio_projector = AudioProjector(config.audio_projector_config)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_name,
            trust_remote_code=True,
            local_files_only=is_local_path
        )
        
        # Add special tokens if not present
        special_tokens = [config.audio_start_token, config.audio_end_token]
        new_tokens = [token for token in special_tokens if token not in self.tokenizer.vocab]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.llm.resize_token_embeddings(len(self.tokenizer))
            
        self.audio_start_token_id = self.tokenizer.convert_tokens_to_ids(config.audio_start_token)
        self.audio_end_token_id = self.tokenizer.convert_tokens_to_ids(config.audio_end_token)
        
    def encode_audio(self, audio_values):
        """
        Encode audio using Whisper encoder
        Args:
            audio_values: [batch_size, n_mels, time] - mel-spectrogram features from WhisperFeatureExtractor
        Returns:
            audio_features: [batch_size, sequence_length, hidden_size]
        """
        with torch.no_grad() if self.config.freeze_audio_encoder else torch.enable_grad():
            # audio_values should already be mel-spectrograms from WhisperFeatureExtractor
            # Shape: [batch_size, n_mels, time]
            # Ensure audio_values has the correct dtype (match the model's dtype)
            target_dtype = next(self.audio_encoder.parameters()).dtype
            if audio_values.dtype != target_dtype:
                audio_values = audio_values.to(target_dtype)
                
            audio_features = self.audio_encoder(audio_values).last_hidden_state
                
        # Project audio features to LLM hidden space
        # Ensure consistency in data types
        if audio_features.dtype != next(self.audio_projector.parameters()).dtype:
            audio_features = audio_features.to(next(self.audio_projector.parameters()).dtype)
            
        projected_features = self.audio_projector(audio_features)
        return projected_features
    
    def prepare_inputs_embeds(self, input_ids, audio_features=None, attention_mask=None, labels=None):
        """
        Prepare input embeddings by inserting audio features at appropriate positions.
        This will expand the embedding sequence to insert audio features between <|audio_bos|> and <|audio_eos|>.
        Also returns updated attention_mask和labels（如有）。
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        new_embeds = []
        new_attention_mask = [] if attention_mask is not None else None
        new_labels = [] if labels is not None else None

        for i in range(batch_size):
            ids = input_ids[i]
            embeds = input_embeds[i]
            # 找到audio_start_token和audio_end_token
            try:
                start_idx = (ids == self.audio_start_token_id).nonzero(as_tuple=False).squeeze(-1).item()
                end_idx = (ids == self.audio_end_token_id).nonzero(as_tuple=False).squeeze(-1).item()
            except Exception:
                # 没有audio token，直接返回原始
                new_embeds.append(embeds)
                if new_attention_mask is not None:
                    new_attention_mask.append(attention_mask[i])
                if new_labels is not None:
                    new_labels.append(labels[i])
                continue
            # 拼接: [text_before, audio_embeds, text_after]
            before = embeds[:start_idx+1]  # 包含audio_bos
            after = embeds[end_idx:]       # 包含audio_eos
            audio_embeds = audio_features[i]  # [audio_seq, hidden]
            # 拼接
            concat_embeds = torch.cat([before, audio_embeds, after], dim=0)
            new_embeds.append(concat_embeds)
            # attention_mask
            if new_attention_mask is not None:
                before_mask = attention_mask[i][:start_idx+1]
                after_mask = attention_mask[i][end_idx:]
                audio_mask = torch.ones(audio_embeds.shape[0], dtype=before_mask.dtype, device=device)
                concat_mask = torch.cat([before_mask, audio_mask, after_mask], dim=0)
                new_attention_mask.append(concat_mask)
            # labels
            if new_labels is not None:
                before_labels = labels[i][:start_idx+1]
                after_labels = labels[i][end_idx:]
                # audio部分label设为-100
                audio_labels = torch.full((audio_embeds.shape[0],), -100, dtype=before_labels.dtype, device=device)
                concat_labels = torch.cat([before_labels, audio_labels, after_labels], dim=0)
                new_labels.append(concat_labels)
        # pad到最大长度
        max_len = max(x.shape[0] for x in new_embeds)
        def pad(x, value):
            if x.shape[0] < max_len:
                pad_shape = (max_len - x.shape[0],) + x.shape[1:]
                return torch.cat([x, torch.full(pad_shape, value, dtype=x.dtype, device=x.device)], dim=0)
            return x
        input_embeds = torch.stack([pad(x, 0.0) for x in new_embeds], dim=0)
        if new_attention_mask is not None:
            attention_mask = torch.stack([pad(x, 0) for x in new_attention_mask], dim=0)
        if new_labels is not None:
            labels = torch.stack([pad(x, -100) for x in new_labels], dim=0)
        return input_embeds, attention_mask, labels
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass of the Qwen2-Audio model
        """
        # Encode audio if provided
        audio_features = None
        if audio_values is not None:
            audio_features = self.encode_audio(audio_values)
        # Prepare input embeddings
        inputs_embeds, attention_mask, labels = self.prepare_inputs_embeds(
            input_ids=input_ids,
            audio_features=audio_features,
            attention_mask=attention_mask,
            labels=labels
        )
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **generation_kwargs
    ):
        """
        Generate responses given audio and text inputs
        """
        # Encode audio if provided
        audio_features = None
        if audio_values is not None:
            audio_features = self.encode_audio(audio_values)
        # Prepare input embeddings
        inputs_embeds, attention_mask, _ = self.prepare_inputs_embeds(
            input_ids=input_ids,
            audio_features=audio_features,
            attention_mask=attention_mask,
            labels=None
        )
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )
        return outputs


def create_model_from_config(config_dict: Dict[str, Any]) -> Qwen2AudioModel:
    """
    Create Qwen2AudioModel from configuration dictionary
    """
    model_config = config_dict.get('model', {})
    
    # Extract configuration
    audio_encoder_config = model_config.get('audio_encoder', {})
    llm_config = model_config.get('llm_backbone', {})
    projector_config = model_config.get('audio_projector', {})
    special_tokens = model_config.get('special_tokens', {})
    
    # Create model configuration
    qwen2_audio_config = Qwen2AudioConfig(
        audio_encoder_name=audio_encoder_config.get('model_name', 'openai/whisper-large-v3'),
        llm_name=llm_config.get('model_name', 'Qwen/Qwen2-7B'),
        llm_type=llm_config.get('model_type', 'qwen2'),
        audio_projector_config=projector_config,
        freeze_audio_encoder=audio_encoder_config.get('freeze_encoder', False),
        freeze_llm=llm_config.get('freeze_llm', False),
        audio_start_token=special_tokens.get('audio_start_token', '<|audio_bos|>'),
        audio_end_token=special_tokens.get('audio_end_token', '<|audio_eos|>'),
        hidden_size=projector_config.get('hidden_size', 896)
    )
    
    # Create and return model
    model = Qwen2AudioModel(qwen2_audio_config)
    return model 