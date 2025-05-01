"""
Configuration settings for the model training application.
This module defines configuration settings for training, API, and deployment.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.environ.get('STORAGE_DIR', '/tmp/model-trainer')
MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', '1024'))

# Create storage directory if it doesn't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, 'datasets'), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, 'exports'), exist_ok=True)

# API settings
API_TITLE = "Model Trainer API"
API_DESCRIPTION = "API for training and exporting AI models"
API_VERSION = "1.0.0"
API_PREFIX = "/api"

# Training settings
TRAINING_CONFIG = {
    "max_epochs": 10,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "eval_steps": 500,
    "save_steps": 1000,
    "memory_limit_mb": MAX_MEMORY_MB,
    "max_train_samples": 50000,  # Limit training samples to manage memory
    "max_validation_samples": 1000,  # Limit validation samples
}

# Monitoring settings
MONITOR_UPDATE_INTERVAL = 5  # seconds
MONITOR_RETENTION_PERIOD = 7  # days
MONITOR_PORT = int(os.environ.get('MONITOR_PORT', '5001'))

# Model export settings
EXPORT_FORMATS = {
    "flask": {
        "description": "Export model for use in Flask applications",
        "extensions": [".pt", ".bin", ".json"],
    },
    "coreml": {
        "description": "Export model in Apple's CoreML format",
        "extensions": [".mlmodel"],
    },
}

# Service communication
TRAINING_SERVICE_URL = os.environ.get(
    'TRAINING_SERVICE_URL', 'http://model-trainer-training:8080'
)
MONITOR_SERVICE_URL = os.environ.get(
    'MONITOR_SERVICE_URL', 'http://model-trainer-monitor:8081'
)
API_SERVICE_URL = os.environ.get(
    'API_SERVICE_URL', 'http://model-trainer-api:8000'
)

# Security settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'development-key-change-in-production')
TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Dataset information
DATASETS = {
    "core": [
        {"name": "nvidia/OpenCodeReasoning", "streaming": True, "max_samples": 5000},
        {"name": "openai/openai_humaneval", "streaming": False, "max_samples": 1000},
        {"name": "bigcode/the-stack", "streaming": True, "max_samples": 10000},
        {"name": "open-thoughts/OpenThoughts2-1M", "streaming": True, "max_samples": 5000},
        {"name": "xAI/TruthfulQA", "streaming": False, "max_samples": 1000},
        {"name": "HuggingFaceH4/instruction-dataset", "streaming": True, "max_samples": 5000},
        {"name": "HuggingFaceH4/ultrachat_200k", "streaming": True, "max_samples": 5000},
        {"name": "Salesforce/dialogstudio", "streaming": True, "max_samples": 5000},
        {"name": "Anthropic/hh-rlhf", "streaming": True, "max_samples": 5000},
    ],
    "additional": [
        {"name": "databricks/databricks-dolly-15k", "streaming": False, "max_samples": 5000},
        {"name": "tatsu-lab/alpaca", "streaming": False, "max_samples": 5000},
        {"name": "deepmind/mathematics", "streaming": False, "max_samples": 3000},
    ]
}

# Model size options
MODEL_SIZES = {
    "small": {
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 512,
        "d_ff": 2048,
        "max_seq_length": 1024,
    },
    "medium": {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ff": 3072,
        "max_seq_length": 2048,
    },
    "large": {
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 1024,
        "d_ff": 4096,
        "max_seq_length": 4096,
    }
}

# Memory optimization settings
MEMORY_OPTIMIZATION = {
    "enable_gradient_checkpointing": True,
    "enable_activation_checkpointing": True,
    "enable_cpu_offloading": True,
    "enable_memory_efficient_attention": True,
    "enable_sequential_batching": True,
    "max_concurrent_requests": 2,
    "garbage_collection_interval": 10,  # seconds
}

# Tokenizer settings
TOKENIZER_CONFIG = {
    "vocab_size": 50257,
    "special_tokens": {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
}

def get_model_config(size: str = "small") -> Dict[str, Any]:
    """
    Get the configuration for a model of the specified size.
    
    Args:
        size: Model size (small, medium, large)
        
    Returns:
        Dictionary with model configuration
    """
    if size not in MODEL_SIZES:
        logger.warning(f"Unknown model size: {size}, defaulting to small")
        size = "small"
    
    return {
        "model": {
            "size": size,
            "sizes": MODEL_SIZES,
            "dropout": 0.1,
            "attention": {
                "causal": True,
                "rotary_embedding": True,
            }
        },
        "tokenizer": TOKENIZER_CONFIG,
        "training": TRAINING_CONFIG,
        "memory_optimization": MEMORY_OPTIMIZATION,
    }
