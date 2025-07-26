"""
LlamaAudio Model: Multimodal Audio-Language Model based on Llama.

This module implements the main LlamaAudio model that combines audio encoding
with Llama language models for multimodal understanding and generation.
"""

import torch
import torch.nn as nn
from transformers import (
    LlamaModel, LlamaForCausalLM, LlamaConfig, LlamaTokenizer,
    AutoModel, AutoTokenizer, AutoConfig
)
from typing import Optional, Union, Tuple, Dict, Any, List
import logging
import warnings

from .audio_encoder import AudioEncoder
from .multimodal_connector import MultiModalConnector

logger = logging.getLogger(__name__)


class LlamaAudioConfig:
    """Configuration class for LlamaAudio model."""
    
    def __init__(
        self,
        llama_model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        whisper_model_name: str = "openai/whisper-large-v3",
        freeze_llm: bool = False,
        freeze_audio_encoder: bool = True,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_cross_attention: bool = False,
        max_audio_tokens: int = 2048,
        audio_adapter_hidden_size: int = 2048,
        use_conv_adapter: bool = True,
        **kwargs
    ):
        self.llama_model_name = llama_model_name
        self.whisper_model_name = whisper_model_name
        self.freeze_llm = freeze_llm
        self.freeze_audio_encoder = freeze_audio_encoder
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_cross_attention = use_cross_attention
        self.max_audio_tokens = max_audio_tokens
        self.audio_adapter_hidden_size = audio_adapter_hidden_size
        self.use_conv_adapter = use_conv_adapter
        
        # Additional configurations
        for key, value in kwargs.items():
            setattr(self, key, value)


class LlamaAudioModel(nn.Module):
    """
    LlamaAudio: Multimodal Audio-Language Model.
    
    This model combines Llama language models with audio understanding capabilities
    through a Whisper-based audio encoder and multimodal connector.
    """
    
    def __init__(self, config: LlamaAudioConfig):
        """
        Initialize LlamaAudio model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        
        # Initialize Llama model
        self._init_llama_model()
        
        # Initialize audio encoder
        self.audio_encoder = AudioEncoder(
            whisper_model_name=config.whisper_model_name,
            llm_hidden_size=self.llama_config.hidden_size,
            freeze_whisper=config.freeze_audio_encoder,
            use_conv_adapter=config.use_conv_adapter,
            adapter_hidden_size=config.audio_adapter_hidden_size,
        )
        
        # Initialize multimodal connector
        self.multimodal_connector = MultiModalConnector(
            llm_hidden_size=self.llama_config.hidden_size,
            max_audio_tokens=config.max_audio_tokens,
            use_cross_attention=config.use_cross_attention,
        )
        
        # Setup LoRA if enabled
        if config.use_lora:
            self._setup_lora()
        
        # Freeze LLM parameters if requested
        if config.freeze_llm:
            self._freeze_llm()
        
        # Extend vocabulary for special audio tokens
        self._extend_vocabulary()
        
        logger.info(f"LlamaAudio model initialized with {self._count_parameters():,} parameters")
        
    def _init_llama_model(self):
        """Initialize the Llama language model."""
        try:
            self.llama_config = AutoConfig.from_pretrained(self.config.llama_model_name)
            self.llama_model = LlamaForCausalLM.from_pretrained(
                self.config.llama_model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llama_model_name)
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load Llama model {self.config.llama_model_name}: {e}")
            raise
    
    def _setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            
            self.llama_model = get_peft_model(self.llama_model, lora_config)
            logger.info("LoRA setup completed")
            
        except ImportError:
            logger.warning("PEFT library not available. LoRA will be disabled.")
            self.config.use_lora = False
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            self.config.use_lora = False
    
    def _freeze_llm(self):
        """Freeze LLM parameters."""
        for param in self.llama_model.parameters():
            param.requires_grad = False
        logger.info("LLM parameters frozen")
    
    def _extend_vocabulary(self):
        """Extend vocabulary with special audio tokens."""
        special_tokens = ["<audio>", "<audio_start>", "<audio_end>"]
        
        # Add special tokens to tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Resize embedding layer
        original_vocab_size = self.llama_model.config.vocab_size
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size > original_vocab_size:
            self.llama_model.resize_token_embeddings(new_vocab_size)
            logger.info(f"Extended vocabulary from {original_vocab_size} to {new_vocab_size}")
        
        # Update special token IDs in multimodal connector
        self.multimodal_connector.audio_token_id = torch.tensor([self.tokenizer.convert_tokens_to_ids("<audio>")])
        self.multimodal_connector.audio_start_id = torch.tensor([self.tokenizer.convert_tokens_to_ids("<audio_start>")])
        self.multimodal_connector.audio_end_id = torch.tensor([self.tokenizer.convert_tokens_to_ids("<audio_end>")])
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass of LlamaAudio model.
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            audio_features: Audio input features [batch_size, audio_seq_len, feat_dim]
            audio_attention_mask: Audio attention mask [batch_size, audio_seq_len]
            labels: Target labels for training [batch_size, seq_len]
            use_cache: Whether to use past key values for generation
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs dictionary
        """
        batch_size = input_ids.shape[0] if input_ids is not None else audio_features.shape[0]
        
        # Process audio if provided
        audio_embeddings = None
        if audio_features is not None:
            audio_embeddings, audio_attention_mask = self.audio_encoder(
                audio_features, audio_attention_mask
            )
        
        # Get text embeddings
        if input_ids is not None:
            text_embeddings = self.llama_model.model.embed_tokens(input_ids)
        else:
            # Audio-only mode - create dummy text embeddings
            text_embeddings = torch.zeros(
                batch_size, 1, self.llama_config.hidden_size,
                device=audio_embeddings.device,
                dtype=audio_embeddings.dtype
            )
            attention_mask = torch.ones(batch_size, 1, device=audio_embeddings.device)
        
        # Combine modalities
        if audio_embeddings is not None:
            multimodal_embeddings, multimodal_attention_mask = self.multimodal_connector(
                text_embeddings=text_embeddings,
                audio_embeddings=audio_embeddings,
                text_attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask,
            )
        else:
            multimodal_embeddings = text_embeddings
            multimodal_attention_mask = attention_mask
        
        # Pass through Llama model
        outputs = self.llama_model(
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text responses given multimodal inputs.
        
        Args:
            input_ids: Input text token IDs
            audio_features: Audio input features
            attention_mask: Text attention mask
            audio_attention_mask: Audio attention mask
            max_length: Maximum total sequence length
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            
        Returns:
            Generated token IDs
        """
        # Prepare multimodal inputs
        batch_size = input_ids.shape[0] if input_ids is not None else audio_features.shape[0]
        
        # Process audio if provided
        audio_embeddings = None
        if audio_features is not None:
            audio_embeddings, audio_attention_mask = self.audio_encoder(
                audio_features, audio_attention_mask
            )
        
        # Get text embeddings
        if input_ids is not None:
            text_embeddings = self.llama_model.model.embed_tokens(input_ids)
        else:
            # Audio-only mode
            text_embeddings = torch.zeros(
                batch_size, 1, self.llama_config.hidden_size,
                device=audio_embeddings.device,
                dtype=audio_embeddings.dtype
            )
            input_ids = torch.zeros(batch_size, 1, device=audio_embeddings.device, dtype=torch.long)
            attention_mask = torch.ones(batch_size, 1, device=audio_embeddings.device)
        
        # Combine modalities
        if audio_embeddings is not None:
            multimodal_embeddings, multimodal_attention_mask = self.multimodal_connector(
                text_embeddings=text_embeddings,
                audio_embeddings=audio_embeddings,
                text_attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask,
            )
            
            # For generation, we need to handle the fact that we have embeddings instead of token IDs
            # We'll use a custom generation approach
            return self._generate_with_embeddings(
                multimodal_embeddings=multimodal_embeddings,
                attention_mask=multimodal_attention_mask,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs
            )
        else:
            # Text-only generation
            return self.llama_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
    
    def _generate_with_embeddings(
        self,
        multimodal_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using multimodal embeddings as input."""
        batch_size, seq_len, hidden_size = multimodal_embeddings.shape
        device = multimodal_embeddings.device
        
        # Initialize with empty token sequences for new generation
        generated_ids = []
        past_key_values = None
        
        # Determine generation length
        if max_new_tokens is not None:
            max_gen_length = max_new_tokens
        else:
            max_gen_length = max_length - seq_len
        
        # Initial forward pass with multimodal embeddings
        outputs = self.llama_model(
            inputs_embeds=multimodal_embeddings,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        past_key_values = outputs.past_key_values
        
        # Sample first token
        if do_sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    probs[i][indices_to_remove] = 0
                
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        generated_ids.append(next_token)
        
        # Update attention mask
        current_attention_mask = torch.cat([
            attention_mask,
            torch.ones(batch_size, 1, device=device)
        ], dim=1)
        
        # Continue generation
        for step in range(max_gen_length - 1):
            outputs = self.llama_model(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Sample next token
            if do_sample:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            
            # Update attention mask
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, 1, device=device)
            ], dim=1)
            
            # Check for EOS token
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        # Concatenate all generated tokens
        generated_sequence = torch.cat(generated_ids, dim=1)
        
        return generated_sequence
    
    def _count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get count of trainable parameters by component."""
        counts = {}
        
        # LLM parameters
        llm_total = sum(p.numel() for p in self.llama_model.parameters())
        llm_trainable = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
        counts["llm_total"] = llm_total
        counts["llm_trainable"] = llm_trainable
        
        # Audio encoder parameters
        audio_total = sum(p.numel() for p in self.audio_encoder.parameters())
        audio_trainable = sum(p.numel() for p in self.audio_encoder.parameters() if p.requires_grad)
        counts["audio_total"] = audio_total
        counts["audio_trainable"] = audio_trainable
        
        # Multimodal connector parameters
        connector_total = sum(p.numel() for p in self.multimodal_connector.parameters())
        connector_trainable = sum(p.numel() for p in self.multimodal_connector.parameters() if p.requires_grad)
        counts["connector_total"] = connector_total
        counts["connector_trainable"] = connector_trainable
        
        # Total
        counts["total"] = llm_total + audio_total + connector_total
        counts["total_trainable"] = llm_trainable + audio_trainable + connector_trainable
        
        return counts
    
    def save_pretrained(self, save_directory: str):
        """Save the model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        import json
        config_dict = {
            "llama_model_name": self.config.llama_model_name,
            "whisper_model_name": self.config.whisper_model_name,
            "freeze_llm": self.config.freeze_llm,
            "freeze_audio_encoder": self.config.freeze_audio_encoder,
            "use_lora": self.config.use_lora,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "use_cross_attention": self.config.use_cross_attention,
            "max_audio_tokens": self.config.max_audio_tokens,
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a pretrained model."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = LlamaAudioConfig(**config_dict)
        
        # Initialize model
        model = cls(config)
        
        # Load state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Model loaded from {model_path}")
        
        return model 