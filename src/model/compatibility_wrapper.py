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
Compatibility wrapper for model architecture.

This module provides a wrapper around the model architecture to ensure
compatibility with Flask applications and Apple's CoreML format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union, Tuple

class CompatibilityWrapper(nn.Module):
    """
    Wrapper around a model to ensure compatibility with Flask and CoreML.
    
    This wrapper:
    1. Standardizes the input/output interface
    2. Ensures all operations are supported by CoreML
    3. Provides helper methods for deployment
    """
    
    def __init__(self, base_model: nn.Module):
        """
        Initialize the wrapper.
        
        Args:
            base_model: The base model to wrap
        """
        super().__init__()
        self.base_model = base_model
        
        # Ensure the model is in evaluation mode
        self.base_model.eval()
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            
        Returns:
            Model outputs [batch_size, sequence_length, vocab_size]
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Forward pass through the base model
        outputs = self.base_model(input_ids, attention_mask)
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text from the model.
        
        Args:
            input_ids: Input token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            do_sample: Whether to use sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            
        Returns:
            Generated token IDs [batch_size, sequence_length + max_new_tokens]
        """
        # Check if the base model has a generate method
        if hasattr(self.base_model, 'generate'):
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )
        
        # If the base model doesn't have a generate method, implement it here
        batch_size, seq_length = input_ids.shape
        
        # Initialize generated sequence with input_ids
        generated = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            # Get the last tokens up to the maximum sequence length
            inputs = generated[:, -seq_length:] if generated.size(1) > seq_length else generated
            
            # Update attention mask
            if attention_mask is not None:
                mask = attention_mask[:, -seq_length:] if attention_mask.size(1) > seq_length else attention_mask
                # Extend mask if needed
                if inputs.size(1) > mask.size(1):
                    extension = torch.ones(batch_size, inputs.size(1) - mask.size(1), device=mask.device)
                    mask = torch.cat([mask, extension], dim=1)
            else:
                mask = torch.ones_like(inputs)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(inputs, mask)
            
            # Get the next token logits
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                for i in range(batch_size):
                    next_token_logits[i, indices[i]] = values[i]
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
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
    
    def save_for_deployment(self, path: str) -> None:
        """
        Save the model for deployment.
        
        Args:
            path: Path to save the model
        """
        # Save the model in TorchScript format
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
    
    def get_input_names(self) -> List[str]:
        """
        Get the names of the model inputs.
        
        Returns:
            List of input names
        """
        return ['input_ids', 'attention_mask']
    
    def get_output_names(self) -> List[str]:
        """
        Get the names of the model outputs.
        
        Returns:
            List of output names
        """
        return ['logits']
    
    def get_input_shapes(self) -> Dict[str, List[int]]:
        """
        Get the shapes of the model inputs.
        
        Returns:
            Dictionary mapping input names to their shapes
        """
        # Get the maximum sequence length from the base model
        max_position_embeddings = getattr(self.base_model, 'max_position_embeddings', 512)
        
        return {
            'input_ids': [1, max_position_embeddings],
            'attention_mask': [1, max_position_embeddings]
        }
    
    def get_output_shapes(self) -> Dict[str, List[int]]:
        """
        Get the shapes of the model outputs.
        
        Returns:
            Dictionary mapping output names to their shapes
        """
        # Get the vocabulary size from the base model
        vocab_size = getattr(self.base_model, 'vocab_size', 50000)
        max_position_embeddings = getattr(self.base_model, 'max_position_embeddings', 512)
        
        return {
            'logits': [1, max_position_embeddings, vocab_size]
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_type': type(self.base_model).__name__,
            'vocab_size': getattr(self.base_model, 'vocab_size', 50000),
            'hidden_size': getattr(self.base_model, 'hidden_size', 768),
            'num_hidden_layers': getattr(self.base_model, 'num_hidden_layers', 12),
            'num_attention_heads': getattr(self.base_model, 'num_attention_heads', 12),
            'max_position_embeddings': getattr(self.base_model, 'max_position_embeddings', 512)
        }