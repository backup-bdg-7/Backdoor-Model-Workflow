import re

# Read the architecture.py file
with open('src/model/architecture.py', 'r') as f:
    content = f.read()

# Find the TransformerModel's forward method and update it with proper gradient checkpointing
pattern = r'def forward\(self, input_ids: torch\.Tensor, attention_mask: Optional\[torch\.Tensor\] = None\) -> torch\.Tensor:.*?# Apply transformer blocks(.*?)# Apply final normalization'
replacement = r'''def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        # Apply transformer blocks with optional gradient checkpointing
        if hasattr(self, '_use_gradient_checkpointing') and self._use_gradient_checkpointing and self.training:
            # Custom function to create a forward function with only the inputs we need
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Apply blocks with gradient checkpointing
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, attention_mask
                )
        else:
            # Apply blocks normally
            for block in self.blocks:
                x = block(x, attention_mask)
        
        # Apply final normalization'''

# Use re.DOTALL to make . match newlines
updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the updated content back to the file
with open('src/model/architecture.py', 'w') as f:
    f.write(updated_content)
    
print("Updated gradient_checkpointing implementation in TransformerModel's forward method")
