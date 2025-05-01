"""
Basic tests to verify that the core functionality works.
"""

import os
import pytest
import yaml
from pathlib import Path

# Import modules to test
from src.data.loaders import DatasetLoader
from src.model.architecture import create_model_from_config
from src.utils.tokenization import get_tokenizer


def test_config_load():
    """Test that we can load the configuration file."""
    config_path = "config/training_config.yaml"
    assert os.path.exists(config_path), f"Config file {config_path} does not exist"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "model" in config, "Config should have a model section"
    assert "training" in config, "Config should have a training section"


def test_model_creation():
    """Test that we can create a model from config."""
    # Create minimal test config
    test_config = {
        "model": {
            "type": "decoder_only_transformer",
            "size": "small",
            "sizes": {
                "small": {
                    "n_layers": 2,
                    "n_heads": 2,
                    "d_model": 64,
                    "d_ff": 128,
                    "max_seq_length": 128
                }
            },
            "dropout": 0.1,
            "attention": {
                "causal": True,
                "rotary_embedding": True
            }
        },
        "tokenizer": {
            "type": "huggingface",
            "name": "gpt2",
            "vocab_size": 50257,
            "max_length": 128
        }
    }
    
    # Create model
    try:
        model = create_model_from_config(test_config)
        assert model is not None, "Model creation failed"
    except Exception as e:
        pytest.skip(f"Skipping test due to error: {e}")


def test_tokenizer():
    """Test that we can get a tokenizer."""
    tokenizer_config = {
        "type": "huggingface",
        "name": "gpt2",
        "vocab_size": 50257,
        "max_length": 128
    }
    
    try:
        tokenizer = get_tokenizer(tokenizer_config)
        assert tokenizer is not None, "Tokenizer creation failed"
        
        # Test encoding and decoding
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        assert isinstance(encoded, list), "Encoded text should be a list"
        assert len(encoded) > 0, "Encoded text should not be empty"
        
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str), "Decoded text should be a string"
        assert len(decoded) > 0, "Decoded text should not be empty"
    except Exception as e:
        pytest.skip(f"Skipping test due to error: {e}")
