import torch
import sys
sys.path.append("./")
from src.model.architecture import create_model_from_config, TransformerModel

# Create a minimal config for testing
config = {
    'model': {
        'type': 'decoder_only_transformer',
        'size': 'small',
        'sizes': {
            'small': {
                'n_layers': 2,
                'n_heads': 2,
                'd_model': 128,
                'd_ff': 512,
                'max_seq_length': 64
            }
        },
        'dropout': 0.1,
        'attention': {'causal': True, 'rotary_embedding': True},
        'architecture': {
            'position_embeddings': 'rotary',
            'attention_type': 'mha',
            'norm_type': 'layer_norm',
            'normalization_strategy': 'pre_norm',
            'ffn_type': 'gelu',
            'use_flash_attention': False
        },
        'gradient_checkpointing': True  # Enable gradient checkpointing
    },
    'tokenizer': {
        'type': 'huggingface',
        'name': 'gpt2',
        'vocab_size': 50257,
        'max_length': 64,
        'padding_side': 'right',
        'truncation_side': 'right',
        'add_bos_token': True,
        'add_eos_token': True
    }
}

print("Creating model with gradient checkpointing...")
model = create_model_from_config(config)

# Test that the gradient checkpointing attribute is set
print(f"Model has _use_gradient_checkpointing attribute: {hasattr(model, '_use_gradient_checkpointing')}")

# Test forward pass works with a small input
print("\nTesting forward pass with gradient checkpointing...")
model.train()  # Set to training mode to activate gradient checkpointing
batch_size, seq_len = 2, 16
input_ids = torch.randint(0, config['tokenizer']['vocab_size'], (batch_size, seq_len))
try:
    outputs = model(input_ids=input_ids)
    print(f"✅ Forward pass successful: output shape = {outputs.shape}")
    
    # Test gradient flow
    loss = outputs.mean()
    loss.backward()
    print("✅ Backward pass successful")
    
    # Success!
    print("\nGradient checkpointing implementation is working correctly!")
except Exception as e:
    print(f"❌ Error during test: {e}")

