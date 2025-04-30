"""
Custom transformer model implementation compatible with HuggingFace ecosystem.
This module provides a transformer model that can be used with the HuggingFace ecosystem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import PreTrainedModel

from src.model.architecture import TransformerBlock, ModelConfig

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
        self.vocab_size = config.vocab_size
        
        # Word embeddings
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        
        # Position embeddings
        if not config.use_rotary_embeddings:
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight
    
    def get_input_embeddings(self):
        """Get word embeddings module."""
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        """Set word embeddings module."""
        self.wte = new_embeddings
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for fast inference
            inputs_embeds: Input embeddings
            labels: Labels for language modeling
            use_cache: Whether to use cache for fast inference
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary of outputs
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        # Initialize position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        # Add position embeddings
        if hasattr(self, 'wpe'):
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        
        # Apply dropout
        hidden_states = self.drop(hidden_states)
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = nn.functional.linear(hidden_states, self.wte.weight)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Format output
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states
            }
        else:
            return (loss, logits, hidden_states)
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate text with the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize generated sequence with input_ids
        generated = input_ids.clone()
        
        # If no attention mask is provided, create one
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Set model to evaluation mode
        self.eval()
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # If sequence length is getting too long, truncate
            if generated.shape[1] > self.config.max_position_embeddings:
                # Keep the last max_position_embeddings tokens
                generated = generated[:, -self.config.max_position_embeddings:]
                attention_mask = attention_mask[:, -self.config.max_position_embeddings:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                logits = outputs["logits"]
            
            # Get the next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][..., -1, None]
                next_token_logits[next_token_logits < indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add the sampled token to the generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Extend attention mask
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Check if any sequence has reached the EOS token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                # If all sequences have generated an EOS token, stop generation
                if (next_token == eos_token_id).all():
                    break
        
        return generated
