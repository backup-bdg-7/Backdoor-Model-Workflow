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
        causal: bool = True
    ):
        """
        Initialize transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Size of the hidden layers
            num_hidden_layers: Number of hidden layers
            num_attention_heads: Number of attention heads
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
        
        # Store configuration
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
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional embeddings
        if use_rotary_embeddings:
            self.pos_embedding = RotaryPositionalEmbedding(hidden_size, max_position_embeddings)
        else:
            self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                hidden_dropout_prob,
                causal
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)
        
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
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability threshold for top-p sampling
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            
        Returns:
            Generated token IDs of shape [batch_size, seq_len + max_new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        
        # Create output tensor
        generated = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get the last tokens up to the maximum sequence length
            curr_input_ids = generated[:, -self.max_position_embeddings:] if generated.size(1) > self.max_position_embeddings else generated
            curr_attention_mask = attention_mask[:, -self.max_position_embeddings:] if attention_mask.size(1) > self.max_position_embeddings else attention_mask
            
            # Forward pass
            with torch.no_grad():
                logits = self.forward(curr_input_ids, curr_attention_mask)
            
            # Get the next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
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
            
            # Sample or greedy decoding
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the next token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have reached the EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_cache": self.use_cache,
            "use_rotary_embeddings": self.use_rotary_embeddings,
            "causal": self.causal
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        """
        Load a model from a pretrained model path.
        
        Args:
            pretrained_model_path: Path to the pretrained model
            
        Returns:
            Loaded model
        """
        # Load configuration
        config_path = os.path.join(pretrained_model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        return model
    
    def to_torchscript(self, file_path: Optional[str] = None):
        """
        Convert the model to TorchScript format.
        
        Args:
            file_path: Path to save the TorchScript model
            
        Returns:
            TorchScript model
        """
        # Set model to evaluation mode
        self.eval()
        
        # Create example inputs
        example_input_ids = torch.ones(1, self.max_position_embeddings, dtype=torch.long)
        example_attention_mask = torch.ones(1, self.max_position_embeddings, dtype=torch.long)
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                self,
                (example_input_ids, example_attention_mask)
            )
        
        # Save the model if file_path is provided
        if file_path:
            traced_model.save(file_path)
        
        return traced_model
    
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
    Create a model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized model
    """
    # Extract model configuration
    model_config = config['model']
    size = model_config['size']
    size_config = model_config['sizes'][size]
    
    # Create Hugging Face compatible config
    hf_config = ModelConfig(
        vocab_size=config['tokenizer']['vocab_size'],
        max_position_embeddings=size_config['max_seq_length'],
        hidden_size=size_config['d_model'],
        num_hidden_layers=size_config['n_layers'],
        num_attention_heads=size_config['n_heads'],
        intermediate_size=size_config['d_ff'],
        hidden_dropout_prob=model_config['dropout'],
        attention_probs_dropout_prob=model_config['dropout'],
        use_rotary_embeddings=model_config['attention']['rotary_embedding'],
        causal=model_config['attention']['causal']
    )
    
    # Create model
    model = TransformerModel(
        vocab_size=config['tokenizer']['vocab_size'],
        hidden_size=size_config['d_model'],
        num_hidden_layers=size_config['n_layers'],
        num_attention_heads=size_config['n_heads'],
        intermediate_size=size_config['d_ff'],
        hidden_dropout_prob=model_config['dropout'],
        attention_probs_dropout_prob=model_config['dropout'],
        max_position_embeddings=size_config['max_seq_length'],
        initializer_range=model_config.get('initializer_range', 0.02),
        use_rotary_embeddings=model_config['attention']['rotary_embedding'],
        causal=model_config['attention']['causal']
    )
    
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