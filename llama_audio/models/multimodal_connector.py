"""
MultiModal Connector for LlamaAudio framework.

This module implements the connector between audio encoder and Llama LLM,
handling the fusion of audio and text modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MultiModalConnector(nn.Module):
    """
    Multimodal connector that handles the integration of audio and text features.
    
    This connector manages the positioning and attention mechanisms for
    multimodal inputs to the Llama model.
    """
    
    def __init__(
        self,
        llm_hidden_size: int = 4096,
        max_audio_tokens: int = 2048,
        use_cross_attention: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize the multimodal connector.
        
        Args:
            llm_hidden_size: Hidden size of the LLM
            max_audio_tokens: Maximum number of audio tokens
            use_cross_attention: Whether to use cross-attention mechanism
            dropout: Dropout rate
        """
        super().__init__()
        
        self.llm_hidden_size = llm_hidden_size
        self.max_audio_tokens = max_audio_tokens
        self.use_cross_attention = use_cross_attention
        
        # Audio position embeddings
        self.audio_position_embeddings = nn.Embedding(
            max_audio_tokens, llm_hidden_size
        )
        
        # Layer normalization for audio features
        self.audio_layer_norm = nn.LayerNorm(llm_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Cross-attention mechanism (optional)
        if use_cross_attention:
            self.cross_attention = CrossAttentionLayer(llm_hidden_size)
        
        # Audio-text fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 2, llm_hidden_size),
        )
        
        # Special tokens for multimodal processing
        self.register_buffer("audio_token_id", torch.tensor([32000]))  # Special audio token
        self.register_buffer("audio_start_id", torch.tensor([32001]))  # Audio start token
        self.register_buffer("audio_end_id", torch.tensor([32002]))    # Audio end token
        
    def forward(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multimodal connector.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_seq_len, hidden_size]
            audio_embeddings: Audio embeddings [batch_size, audio_seq_len, hidden_size]
            audio_positions: Positions where audio should be inserted
            text_attention_mask: Attention mask for text
            audio_attention_mask: Attention mask for audio
            
        Returns:
            Tuple of (multimodal_embeddings, multimodal_attention_mask)
        """
        batch_size = text_embeddings.shape[0]
        
        if audio_embeddings is None:
            # Text-only mode
            return text_embeddings, text_attention_mask
        
        # Process audio embeddings
        audio_seq_len = audio_embeddings.shape[1]
        
        # Add positional embeddings to audio
        if audio_positions is None:
            audio_positions = torch.arange(
                audio_seq_len, device=audio_embeddings.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure positions are within bounds
        audio_positions = torch.clamp(audio_positions, 0, self.max_audio_tokens - 1)
        
        audio_pos_emb = self.audio_position_embeddings(audio_positions)
        audio_embeddings = audio_embeddings + audio_pos_emb
        
        # Apply layer normalization and dropout
        audio_embeddings = self.audio_layer_norm(audio_embeddings)
        audio_embeddings = self.dropout(audio_embeddings)
        
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            audio_embeddings = self.cross_attention(
                query=audio_embeddings,
                key=text_embeddings,
                value=text_embeddings,
                attention_mask=text_attention_mask,
            )
        
        # Apply fusion layer to audio embeddings
        audio_embeddings = self.fusion_layer(audio_embeddings)
        
        # Concatenate text and audio embeddings
        multimodal_embeddings = torch.cat([text_embeddings, audio_embeddings], dim=1)
        
        # Concatenate attention masks
        if text_attention_mask is not None and audio_attention_mask is not None:
            multimodal_attention_mask = torch.cat([
                text_attention_mask, audio_attention_mask
            ], dim=1)
        elif text_attention_mask is not None:
            # Create attention mask for audio
            audio_mask = torch.ones(
                batch_size, audio_seq_len,
                device=text_attention_mask.device,
                dtype=text_attention_mask.dtype
            )
            multimodal_attention_mask = torch.cat([
                text_attention_mask, audio_mask
            ], dim=1)
        else:
            multimodal_attention_mask = None
        
        return multimodal_embeddings, multimodal_attention_mask
    
    def interleave_modalities(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleave text and audio embeddings based on special tokens.
        
        Args:
            text_embeddings: Text embeddings
            audio_embeddings: Audio embeddings
            text_tokens: Text token IDs
            audio_positions: Positions of audio tokens in text
            
        Returns:
            Tuple of (interleaved_embeddings, interleaved_attention_mask)
        """
        batch_size, text_seq_len, hidden_size = text_embeddings.shape
        audio_seq_len = audio_embeddings.shape[1]
        
        # Find audio token positions
        audio_token_mask = (text_tokens == self.audio_token_id)
        
        # Create output tensor
        max_seq_len = text_seq_len + audio_seq_len
        output_embeddings = torch.zeros(
            batch_size, max_seq_len, hidden_size,
            device=text_embeddings.device,
            dtype=text_embeddings.dtype
        )
        
        output_mask = torch.zeros(
            batch_size, max_seq_len,
            device=text_embeddings.device,
            dtype=torch.bool
        )
        
        for b in range(batch_size):
            audio_positions_b = torch.where(audio_token_mask[b])[0]
            
            if len(audio_positions_b) == 0:
                # No audio tokens, just copy text
                output_embeddings[b, :text_seq_len] = text_embeddings[b]
                output_mask[b, :text_seq_len] = True
            else:
                # Interleave text and audio
                output_pos = 0
                text_pos = 0
                audio_pos = 0
                
                for i in range(text_seq_len):
                    if i in audio_positions_b and audio_pos < audio_seq_len:
                        # Insert audio embeddings
                        audio_chunk_len = min(
                            audio_embeddings.shape[1] // len(audio_positions_b),
                            audio_seq_len - audio_pos
                        )
                        
                        output_embeddings[b, output_pos:output_pos + audio_chunk_len] = \
                            audio_embeddings[b, audio_pos:audio_pos + audio_chunk_len]
                        output_mask[b, output_pos:output_pos + audio_chunk_len] = True
                        
                        output_pos += audio_chunk_len
                        audio_pos += audio_chunk_len
                    else:
                        # Insert text embedding
                        output_embeddings[b, output_pos] = text_embeddings[b, text_pos]
                        output_mask[b, output_pos] = True
                        output_pos += 1
                    
                    text_pos += 1
                
                # Add remaining audio if any
                if audio_pos < audio_seq_len:
                    remaining_audio = audio_seq_len - audio_pos
                    output_embeddings[b, output_pos:output_pos + remaining_audio] = \
                        audio_embeddings[b, audio_pos:]
                    output_mask[b, output_pos:output_pos + remaining_audio] = True
        
        return output_embeddings, output_mask


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for multimodal fusion."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, \
            "hidden_size must be divisible by num_heads"
        
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of cross-attention."""
        batch_size, seq_len, _ = query.shape
        
        # Compute query, key, value projections
        Q = self.query_proj(query)  # [B, T_q, H]
        K = self.key_proj(key)      # [B, T_k, H]
        V = self.value_proj(value)  # [B, T_v, H]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [B, H, T_q, D]
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        output = self.output_proj(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output 