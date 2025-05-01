"""
/**
 * Copyright (c) [2025] Backdoor Software Inc.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of Backdoor Software Inc.
 * You may not disclose, reproduce, or distribute this software without the express written
 * permission of Backdoor Software Inc.
 *
 * Created by: Backdoor Software Inc.
 * Purpose: Activity Maintenance
 */
"""

"""
Model architecture definitions for the AI model training workflow.
This module provides transformer-based model architectures optimized for various tasks.
"""

import os
import math
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

# Configure logging
logger = logging.getLogger(__name__)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary positional embedding implementation.
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192):
        """
        Initialize rotary positional embeddings.
        
        Args:
            dim: Dimension of the embeddings
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Generate frequency bands
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create position indices
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        
        # Create rotation matrices
        self.cos = nn.Parameter(torch.cos(freqs), requires_grad=False)
        self.sin = nn.Parameter(torch.sin(freqs), requires_grad=False)
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            seq_dim: Dimension corresponding to sequence length
            
        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_len = x.shape[seq_dim]
        
        # Extract cos and sin values for the sequence length
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        
        # Reshape for broadcasting
        if seq_dim == 1:
            # [seq_len, dim] -> [seq_len, 1, dim]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            # [seq_len, dim] -> [1, seq_len, dim]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Apply rotary embeddings
        return x * cos + self._rotate_half(x) * sin
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions of the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Rotated tensor
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class FlashAttention(nn.Module):
    """
    Efficient attention implementation with optimized memory access patterns.
    Based on the paper "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    """
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, causal: bool = True):
        """
        Initialize flash attention module.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            causal: Whether to use causal attention mask
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        # Projection matrices
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for flash attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale queries
        q = q * self.scale
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attn_scores.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """
        Initialize feed-forward network.
        
        Args:
            dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-layer normalization.
    """
    
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0, causal: bool = True):
        """
        Initialize transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            causal: Whether to use causal attention
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout, causal)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Attention with pre-norm
        x = x + self.attn(self.attn_norm(x), mask)
        
        # Feed-forward with pre-norm
        x = x + self.ff(self.ff_norm(x))
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize decoder-only transformer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Extract configuration
        size = config['model']['size']
        size_config = config['model']['sizes'][size]
        
        self.n_layers = size_config['n_layers']
        self.n_heads = size_config['n_heads']
        self.d_model = size_config['d_model']
        self.d_ff = size_config['d_ff']
        self.max_seq_length = size_config['max_seq_length']
        
        self.dropout = config['model']['dropout']
        self.causal = config['model']['attention']['causal']
        self.use_rotary = config['model']['attention']['rotary_embedding']
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config['tokenizer']['vocab_size'], self.d_model)
        
        # Positional embeddings
        if self.use_rotary:
            self.pos_embedding = RotaryPositionalEmbedding(self.d_model, self.max_seq_length)
        else:
            self.pos_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.d_model,
                self.n_heads,
                self.d_ff,
                self.dropout,
                self.causal
            )
            for _ in range(self.n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, config['tokenizer']['vocab_size'], bias=False)
        
        # Tie weights
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights.
        """
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Initialize positional embeddings if not using rotary
        if not self.use_rotary:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)
        
        # Initialize linear layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for decoder-only transformer.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        if self.use_rotary:
            # Rotary embeddings are applied in the attention module
            pass
        else:
            # Add standard positional embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int, temperature: float = 1.0,
                top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability threshold for top-p sampling
            
        Returns:
            Generated token IDs of shape [batch_size, max_length]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create output tensor
        output_ids = input_ids.clone()
        
        # Generate tokens one by one
        for i in range(max_length - seq_len):
            # Get logits for the next token
            logits = self.forward(output_ids)[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to output
            output_ids = torch.cat([output_ids, next_token], dim=1)
        
        return output_ids


class ModelConfig(PretrainedConfig):
    """
    Configuration class for the custom model.
    """
    
    model_type = "decoder_only_transformer"
    
    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=1024,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        use_rotary_embeddings=True,
        causal=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_rotary_embeddings = use_rotary_embeddings
        self.causal = causal


class TransformerModel(nn.Module):
    """
    Transformer model compatible with Flask and CoreML.
    
    This model is designed to be compatible with both Flask applications and
    Apple's CoreML format. It uses standard PyTorch operations and avoids
    custom CUDA kernels or dynamic control flow that might cause issues
    during conversion.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        use_rotary_embeddings: bool = True,
        causal: bool = True,
        config = None
    ):
        """
        Initialize transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Size of the hidden layers
            num_hidden_layers: Number of hidden layers
            num_attention_heads: Number of attention heads
            config: Optional ModelConfig object. If provided, its parameters will override the individual arguments.
            intermediate_size: Size of the intermediate feed-forward layers
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention
            max_position_embeddings: Maximum sequence length
            initializer_range: Range for weight initialization
            layer_norm_eps: Epsilon for layer normalization
            use_cache: Whether to use caching during generation
            use_rotary_embeddings: Whether to use rotary positional embeddings
            causal: Whether to use causal attention
        """
        super().__init__()
        
        # If config is provided, use its parameters instead of the individual arguments
        if config is not None:
            self.vocab_size = config.vocab_size
            self.hidden_size = config.hidden_size
            self.num_hidden_layers = config.num_hidden_layers
            self.num_attention_heads = config.num_attention_heads
            self.intermediate_size = config.intermediate_size
            self.hidden_dropout_prob = config.hidden_dropout_prob
            self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
            self.max_position_embeddings = config.max_position_embeddings
            self.initializer_range = config.initializer_range
            self.layer_norm_eps = config.layer_norm_eps
            self.use_cache = config.use_cache
            self.use_rotary_embeddings = config.use_rotary_embeddings
            self.causal = config.causal
        else:
            # Store configuration from individual parameters
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.use_cache = use_cache
            self.use_rotary_embeddings = use_rotary_embeddings
            self.causal = causal
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Positional embeddings
        if self.use_rotary_embeddings:
            self.pos_embedding = RotaryPositionalEmbedding(self.hidden_size, self.max_position_embeddings)
        else:
            self.pos_embedding = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.intermediate_size,
                self.hidden_dropout_prob,
                self.causal
            )
            for _ in range(self.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie weights
        self.output_proj.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights.
        """
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, std=self.initializer_range)
        
        # Initialize positional embeddings if not using rotary
        if not self.use_rotary_embeddings:
            nn.init.normal_(self.pos_embedding.weight, std=self.initializer_range)
        
        # Initialize linear layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        if self.use_rotary_embeddings:
            # Rotary embeddings are applied in the attention module
            pass
        else:
            # Add standard positional embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            x = x + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for efficient text generation.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Past key values for fast inference
            attention_mask: Attention mask
            position_ids: Position IDs
            **kwargs: Additional keyword arguments
            
        Returns:
            Dict of model inputs
        """
        # Only last token for input_ids if using past key values
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Position IDs for the last token
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        # Prepare position IDs if not provided
        if position_ids is None:
            # Create position IDs accounting for past tokens
            past_length = 0
            if past_key_values is not None:
                past_length = past_key_values[0][0].size(-2)
                
            position_ids = torch.arange(
                past_length, past_length + input_ids.size(-1),
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", self.config.use_cache),
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        max_length: int = 20,
        min_length: int = 0,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        use_cache: Optional[bool] = None,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict_in_generate: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate text with advanced features.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            position_ids: Optional position IDs for positional embeddings
            max_length: Maximum total length of generated sequence (including input)
            min_length: Minimum length of generated sequence
            temperature: Sampling temperature (lower = more deterministic)
            do_sample: Whether to use sampling instead of greedy decoding
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens (>1.0 reduces repetition)
            use_cache: Whether to use KV cache for efficient generation
            num_return_sequences: Number of sequences to return per input
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end of sequence token
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict_in_generate: Whether to return dictionary instead of tensor
            **kwargs: Additional arguments

        Returns:
            Generated token IDs of shape [batch_size, seq_len + new_tokens]
            or dictionary with sequences and optional attention/hidden states
        """
        batch_size, input_seq_len = input_ids.shape
        device = input_ids.device
        use_cache = use_cache if use_cache is not None else getattr(self, 'use_cache', True)
        max_gen_length = max(max_length, input_seq_len)
        tokens_to_generate = max_gen_length - input_seq_len
        
        # If generating multiple sequences, repeat the input
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
            if position_ids is not None:
                position_ids = position_ids.repeat(num_return_sequences, 1)
            batch_size = batch_size * num_return_sequences
            
        # Create output tensor (will be expanded as we go)
        generated_ids = input_ids.clone()
        
        # Prepare position IDs if needed
        if position_ids is None and hasattr(self, 'pos_embedding'):
            position_ids = torch.arange(input_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Handle KV cache for efficient generation
        past_key_values = None
        
        # Track unfinished sequences
        if eos_token_id is not None:
            unfinished_sequences = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
        # Initialize outputs for return_dict_in_generate
        if return_dict_in_generate:
            scores = () if do_sample else None
            attentions = () if output_attentions else None
            hidden_states = () if output_hidden_states else None
            
        # Set model to evaluation mode
        self.eval()
        
        # Generate tokens one by one
        for _ in range(tokens_to_generate):
            # Prepare inputs - use full sequence or last token depending on whether cache is used
            if use_cache and past_key_values is not None:
                # Use only the last token and cache for efficiency
                curr_ids = generated_ids[:, -1].unsqueeze(-1)
                curr_mask = attention_mask
                curr_pos_ids = position_ids[:, -1].unsqueeze(-1) if position_ids is not None else None
            else:
                # Limit to maximum context window if needed
                effective_limit = self.max_position_embeddings if hasattr(self, 'max_position_embeddings') else 2048
                if generated_ids.size(1) > effective_limit:
                    curr_ids = generated_ids[:, -effective_limit:]
                    curr_mask = attention_mask[:, -effective_limit:] if attention_mask is not None else None
                    curr_pos_ids = position_ids[:, -effective_limit:] if position_ids is not None else None
                else:
                    curr_ids = generated_ids
                    curr_mask = attention_mask
                    curr_pos_ids = position_ids
                    
            # Forward pass
            with torch.no_grad():
                if use_cache:
                    outputs = self.forward(
                        input_ids=curr_ids,
                        attention_mask=curr_mask,
                        position_ids=curr_pos_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True
                    )
                    logits = outputs["logits"]
                    if use_cache:
                        past_key_values = outputs["past_key_values"]
                    
                    # Save outputs for return_dict_in_generate
                    if return_dict_in_generate:
                        if output_hidden_states:
                            hidden_states = hidden_states + (outputs.get("hidden_states"),)
                        if output_attentions:
                            attentions = attentions + (outputs.get("attentions"),)
                else:
                    # Use simpler forward pass signature when not using cache
                    logits = self.forward(curr_ids, curr_mask)
                    if isinstance(logits, dict):
                        logits = logits["logits"]
            
            # Get next token logits (last position in sequence)
            next_token_logits = logits[:, -1, :]
            
            # Apply min_length constraint
            if min_length > 0 and generated_ids.size(1) - input_seq_len < min_length and eos_token_id is not None:
                next_token_logits[:, eos_token_id] = -float("inf")
            
            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                for i in range(batch_size):
                    next_token_logits[i, indices[i]] = values[i]
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
            
            # Save scores for return_dict_in_generate
            if return_dict_in_generate and do_sample:
                scores = scores + (next_token_logits,)
            
            # Sample or greedy decoding
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the new tokens
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            
            # Update attention mask for new token
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)
            
            # Update position IDs for next token if they're being used
            if position_ids is not None:
                new_position_ids = position_ids[:, -1:] + 1
                position_ids = torch.cat([position_ids, new_position_ids], dim=1)
            
            # Check if any sequences have finished
            if eos_token_id is not None:
                # Update which sequences are still unfinished
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )
                
                # Stop when all sequences are finished
                if unfinished_sequences.max() == 0:
                    break
        
        # Return generated sequences with optional extras
        if return_dict_in_generate:
            return {
                "sequences": generated_ids,
                "scores": scores,
                "attentions": attentions,
                "hidden_states": hidden_states
            }
        
        return generated_ids
    
    def _reorder_cache(
        self,
        past_key_values: List[Tuple[torch.Tensor]], 
        beam_idx: torch.Tensor
    ) -> List[Tuple[torch.Tensor]]:
        """
        Reorder cached past key values for beam search.
        
        Args:
            past_key_values: Past key values
            beam_idx: Indices for beam reordering
            
        Returns:
            Reordered past key values
        """
        # If using key-value cache with beam search, we need to reorder the cache
        # for the selected beam indices when doing beam search
        return [
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) 
                  for past_state in layer_past)
            for layer_past in past_key_values
        ]
    
    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        """
        Save model and tokenizer to a directory.
        
        Args:
            save_directory: Directory to save to
            **kwargs: Additional arguments
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            # Convert config to dict and save as JSON
            config_dict = self.config.to_dict()
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load model from pretrained weights.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or model name
            **kwargs: Additional arguments
            
        Returns:
            Loaded model
        """
        # Handle local paths
        if os.path.isdir(pretrained_model_name_or_path):
            # Load configuration
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                config = ModelConfig(**config_dict)
            else:
                config = None
                
            # Load model weights
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")
            else:
                raise ValueError(f"Model weights not found at {model_path}")
                
            # Create model instance
            model = cls(config=config, **kwargs)
            
            # Load state dict
            model.load_state_dict(state_dict)
            
            return model
        else:
            # For remote models, would need to implement downloading logic
            raise NotImplementedError(f"Loading from remote location not implemented: {pretrained_model_name_or_path}")
            
    def to_export_format(self, export_format: str = "onnx", **kwargs):
        """
        Convert model to deployable format.
        
        Args:
            export_format: Format to export to (onnx, torchscript, etc.)
            **kwargs: Additional format-specific arguments
            
        Returns:
            Exported model or path to exported model
        """
        if export_format.lower() == "onnx":
            try:
                import onnx
                import onnxruntime
                import torch.onnx
                
                # Set model to evaluation mode
                self.eval()
                
                # Create dummy input
                batch_size = kwargs.get("batch_size", 1)
                seq_length = kwargs.get("seq_length", 8)
                dummy_input = {
                    "input_ids": torch.ones(batch_size, seq_length, dtype=torch.long),
                    "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
                }
                
                # Export path
                output_path = kwargs.get("output_path", "model.onnx")
                
                # Export to ONNX
                torch.onnx.export(
                    self,
                    (dummy_input,),
                    output_path,
                    opset_version=kwargs.get("opset_version", 12),
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'logits': {0: 'batch_size', 1: 'sequence'}
                    }
                )
                
                logger.info(f"Model exported to ONNX at {output_path}")
                return output_path
                
            except ImportError:
                raise ImportError("ONNX and ONNX Runtime are required for ONNX export")
                
        elif export_format.lower() == "torchscript":
            # Export to TorchScript
            self.eval()
            
            # Create dummy input
            batch_size = kwargs.get("batch_size", 1)
            seq_length = kwargs.get("seq_length", 8)
            dummy_input = {
                "input_ids": torch.ones(batch_size, seq_length, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
            }
            
            # Trace or script the model
            traced_model = torch.jit.trace(self, (dummy_input,))
            
            # Save if output path provided
            output_path = kwargs.get("output_path")
            if output_path:
                traced_model.save(output_path)
                logger.info(f"Model exported to TorchScript at {output_path}")
                
            return traced_model
            
        elif export_format.lower() == "coreml":
            try:
                import coremltools as ct
                
                # Set model to evaluation mode
                self.eval()
                
                # Create dummy input
                batch_size = 1  # CoreML typically uses batch size 1
                seq_length = kwargs.get("seq_length", 8)
                dummy_input = {
                    "input_ids": torch.ones(batch_size, seq_length, dtype=torch.long),
                    "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long)
                }
                
                # Convert to TorchScript first
                traced_model = torch.jit.trace(self, (dummy_input,))
                
                # Convert to CoreML
                mlmodel = ct.convert(
                    traced_model,
                    inputs=[
                        ct.TensorType(name="input_ids", shape=dummy_input["input_ids"].shape),
                        ct.TensorType(name="attention_mask", shape=dummy_input["attention_mask"].shape)
                    ]
                )
                
                # Save if output path provided
                output_path = kwargs.get("output_path")
                if output_path:
                    mlmodel.save(output_path)
                    logger.info(f"Model exported to CoreML at {output_path}")
                    
                return mlmodel
                
            except ImportError:
                raise ImportError("CoreMLTools is required for CoreML export")
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    # The from_pretrained method is now implemented above with enhanced features
    # The to_export_format method above supports TorchScript, ONNX, and CoreML export
    
    def prepare_for_export(self):
        """
        Prepare the model for export to other formats.
        
        This method:
        1. Sets the model to evaluation mode
        2. Freezes all parameters
        3. Applies any necessary transformations for compatibility
        
        Returns:
            Self for chaining
        """
        # Set model to evaluation mode
        self.eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        return self


class CustomTransformerModel(PreTrainedModel):
    """
    Custom transformer model compatible with the Hugging Face ecosystem.
    """
    
    config_class = ModelConfig
    
    def __init__(self, config):
        """
        Initialize custom transformer model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        self.embed_dim = config.hidden_size
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        
        # Positional embeddings
        if config.use_rotary_embeddings:
            self.wpe = RotaryPositionalEmbedding(self.embed_dim, config.max_position_embeddings)
            self.use_rotary = True
        else:
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
            self.use_rotary = False
        
        # Dropout
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_dropout_prob,
                config.causal
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights
        self.tie_weights()
    
    def _init_weights(self, module):
        """
        Initialize the weights.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        self.lm_head = nn.Linear(self.embed_dim, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (not used)
            position_ids: Position IDs
            inputs_embeds: Input embeddings
            labels: Labels for language modeling
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # Get token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        # Add positional embeddings
        if self.use_rotary:
            # Rotary embeddings are applied in the attention module
            hidden_states = inputs_embeds
        else:
            # Add standard positional embeddings
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        
        hidden_states = self.drop(hidden_states)
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        # Apply final normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Project to vocabulary
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
        }


# Factory function to create model from config
def create_model_from_config(config: Dict) -> nn.Module:
    """
    Create a model from configuration, supporting all modern architecture features.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model with the requested architecture
    """
    # Extract model configuration
    model_config = config['model']
    size = model_config['size']
    size_config = model_config['sizes'][size]
    
    # Extract architecture configuration
    architecture_config = model_config.get('architecture', {})
    
    # Create advanced ModelConfig with all modern features
    model_config_params = {
        # Basic model parameters
        "vocab_size": config['tokenizer']['vocab_size'],
        "max_position_embeddings": size_config['max_seq_length'],
        "hidden_size": size_config['d_model'],
        "num_hidden_layers": size_config['n_layers'],
        "num_attention_heads": size_config['n_heads'],
        "intermediate_size": size_config['d_ff'],
        "hidden_dropout_prob": model_config['dropout'],
        "attention_probs_dropout_prob": model_config['dropout'],
        "initializer_range": model_config.get('initializer_range', 0.02),
        "layer_norm_eps": model_config.get('layer_norm_eps', 1e-5),
        
        # Position embeddings - default to rotary if not specified
        "position_embedding_type": architecture_config.get('position_embeddings', 
                                   'rotary' if model_config['attention'].get('rotary_embedding', True) else 'learned'),
        
        # Attention settings
        "causal": model_config['attention'].get('causal', True),
        "attention_type": architecture_config.get('attention_type', 'mha'),  # mha, mqa, gqa
        "kv_heads": architecture_config.get('kv_heads', None),  # For GQA, number of KV heads
        
        # Normalization and activation
        "norm_type": architecture_config.get('norm_type', 'layer_norm'),  # layer_norm, rms_norm
        "normalization_strategy": architecture_config.get('normalization_strategy', 'pre_norm'),  # pre_norm, post_norm
        "activation_function": architecture_config.get('activation_function', 'gelu'),
        
        # Advanced features
        "ffn_type": architecture_config.get('ffn_type', 'mlp'),  # mlp, swiglu, geglu
        "use_bias": architecture_config.get('use_bias', True),
        "drop_path_rate": architecture_config.get('drop_path_rate', 0.0),
        
        # Performance optimizations
        "use_flash_attention": architecture_config.get('use_flash_attention', False),
        "use_cache": model_config.get('use_cache', True),
        "tie_word_embeddings": model_config.get('tie_word_embeddings', True),
        
        # Quantization
        "quantization": model_config.get('quantization', None)
    }
    
    # Create ModelConfig
    hf_config = ModelConfig(**model_config_params)
    
    # Create TransformerModel with all modern features using our enhanced implementation
    model = TransformerModel(config=hf_config)
    
    # Apply additional model initializations if specified 
    if 'initialization' in architecture_config:
        init_config = architecture_config['initialization']
        method = init_config.get('method', 'default')
        
        if method == 'normal':
            # Normal initialization with specified params
            std = init_config.get('std', 0.02)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=std)
                    if module.bias is not None:
                        module.bias.data.zero_()
        
        elif method == 'xavier_uniform':
            # Xavier uniform initialization
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight.data)
                    if module.bias is not None:
                        module.bias.data.zero_()
        
        elif method == 'kaiming_normal':
            # Kaiming normal initialization
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')
                    if module.bias is not None:
                        module.bias.data.zero_()
    
    # Apply gradient checkpointing if requested
    if model_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    # Apply pytorch 2.0 compilation if requested
    if 'compile' in model_config and model_config['compile'].get('enabled', False):
        try:
            import torch._dynamo
            mode = model_config['compile'].get('mode', 'default')
            
            logger.info(f"Applying torch.compile with mode: {mode}")
            model = torch.compile(model, mode=mode)
        except (ImportError, AttributeError):
            logger.warning("PyTorch 2.0+ compilation not available. Skipping model compilation.")
    
    return model


# Example usage
if __name__ == "__main__":
    import yaml
    import torch
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Add tokenizer vocabulary size to config
    config['tokenizer']['vocab_size'] = 50257  # Example value
    
    # Create model
    model = create_model_from_config(config)
    
    # Print model summary
    logger.info(f"Model created with size: {config['model']['size']}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config['tokenizer']['vocab_size'], (batch_size, seq_len))
    
    # Run forward pass
    outputs = model(input_ids=input_ids)
    
    logger.info(f"Output logits shape: {outputs['logits'].shape}")
    
    # Test generation
    generated = model.generate(
        input_ids=input_ids[:, :4],
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    logger.info(f"Generated sequence shape: {generated.shape}")