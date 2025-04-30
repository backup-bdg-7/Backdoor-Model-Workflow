"""
Text preprocessing utilities for the AI model training workflow.
This module provides a TextPreprocessor class for tokenizing and processing text.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    A class to handle tokenization and preprocessing of text data.
    """
    
    def __init__(
        self,
        max_length: int = 512,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize the TextPreprocessor.
        
        Args:
            max_length: Maximum sequence length
            tokenizer_config: Configuration for tokenizer
            tokenizer_path: Path to save/load the tokenizer
        """
        self.max_length = max_length
        self.tokenizer_config = tokenizer_config or {
            "type": "bpe",
            "vocab_size": 50000,
            "special_tokens": {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]"
            }
        }
        self.tokenizer_path = tokenizer_path
        
        # Initialize tokenizer
        self.tokenizer = None
        
        # Try to load tokenizer if path is provided
        if tokenizer_path and os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
            try:
                self.load_tokenizer(os.path.join(tokenizer_path, "tokenizer.json"))
                logger.info(f"Loaded tokenizer from {tokenizer_path}")
            except Exception as e:
                logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")
    
    def create_tokenizer(self, config: Dict[str, Any]) -> Any:
        """
        Create a tokenizer based on configuration.
        
        Args:
            config: Tokenizer configuration
            
        Returns:
            Tokenizer instance
        """
        tokenizer_type = config.get("type", "bpe").lower()
        
        try:
            # Try to use transformers library
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            
            # Default to GPT-2 tokenizer if not specified
            model_name = config.get("model_name", "gpt2")
            
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure tokenizer
            tokenizer.model_max_length = self.max_length
            
            # Add special tokens if needed
            if "special_tokens" in config:
                special_tokens_dict = {}
                
                for key, value in config["special_tokens"].items():
                    if key.endswith("_token"):
                        special_tokens_dict[key] = value
                    else:
                        special_tokens_dict[f"{key}_token"] = value
                
                tokenizer.add_special_tokens(special_tokens_dict)
            
            return tokenizer
            
        except (ImportError, Exception) as e:
            logger.warning(f"Could not create transformers tokenizer: {e}. Falling back to custom tokenizer.")
            
            # Create a simple dictionary-based tokenizer
            class SimpleTokenizer:
                def __init__(self, vocab_size: int = 50000):
                    self.vocab_size = vocab_size
                    self.vocab = {}
                    self.inv_vocab = {}
                    
                    # Add special tokens
                    special_tokens = config.get("special_tokens", {})
                    for idx, (_, token) in enumerate(special_tokens.items()):
                        self.vocab[token] = idx
                        self.inv_vocab[idx] = token
                
                def tokenize(self, text: str) -> List[str]:
                    return text.split()
                
                def encode(self, text: str, max_length: Optional[int] = None, 
                         padding: Optional[str] = None, truncation: Optional[bool] = None) -> Dict[str, List[int]]:
                    tokens = self.tokenize(text)
                    input_ids = []
                    
                    for token in tokens:
                        if token in self.vocab:
                            input_ids.append(self.vocab[token])
                        else:
                            # Add unknown tokens to vocabulary if there's space
                            if len(self.vocab) < self.vocab_size:
                                idx = len(self.vocab)
                                self.vocab[token] = idx
                                self.inv_vocab[idx] = token
                                input_ids.append(idx)
                            else:
                                # Use UNK token ID (usually 1)
                                input_ids.append(1)
                    
                    # Apply truncation
                    if truncation and max_length and len(input_ids) > max_length:
                        input_ids = input_ids[:max_length]
                    
                    # Apply padding
                    attention_mask = [1] * len(input_ids)
                    if padding and max_length:
                        pad_token_id = self.vocab.get("[PAD]", 0)
                        pad_length = max_length - len(input_ids)
                        if pad_length > 0:
                            input_ids.extend([pad_token_id] * pad_length)
                            attention_mask.extend([0] * pad_length)
                    
                    return {"input_ids": input_ids, "attention_mask": attention_mask}
                
                def decode(self, ids: List[int]) -> str:
                    tokens = [self.inv_vocab.get(idx, "[UNK]") for idx in ids]
                    return " ".join(tokens)
                
                def train(self, texts: List[str]) -> None:
                    # Simple training: just add tokens to vocabulary
                    for text in texts:
                        tokens = self.tokenize(text)
                        for token in tokens:
                            if token not in self.vocab and len(self.vocab) < self.vocab_size:
                                idx = len(self.vocab)
                                self.vocab[token] = idx
                                self.inv_vocab[idx] = token
                
                def save(self, path: str) -> None:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # Save vocabulary
                    with open(path, "w") as f:
                        json.dump({"vocab": self.vocab, "inv_vocab": self.inv_vocab}, f)
            
            return SimpleTokenizer(vocab_size=config.get("vocab_size", 50000))
    
    def load_tokenizer(self, path: str) -> None:
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
        """
        try:
            from transformers import PreTrainedTokenizerFast
            
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
            
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load transformers tokenizer: {e}. Falling back to custom tokenizer.")
            
            # Load simple tokenizer
            with open(path, "r") as f:
                data = json.load(f)
            
            tokenizer = self.create_tokenizer(self.tokenizer_config)
            tokenizer.vocab = data.get("vocab", {})
            tokenizer.inv_vocab = data.get("inv_vocab", {})
            
            self.tokenizer = tokenizer
    
    def train_tokenizer(self, texts: List[str]) -> None:
        """
        Train tokenizer on texts.
        
        Args:
            texts: List of texts to train on
        """
        # Create tokenizer if not exists
        if self.tokenizer is None:
            self.tokenizer = self.create_tokenizer(self.tokenizer_config)
        
        # Train tokenizer
        self.tokenizer.train(texts)
        
        # Save tokenizer if path is provided
        if self.tokenizer_path:
            os.makedirs(self.tokenizer_path, exist_ok=True)
            self.tokenizer.save(os.path.join(self.tokenizer_path, "tokenizer.json"))
            logger.info(f"Saved tokenizer to {self.tokenizer_path}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and newlines.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = " ".join(text.split())
        return text
    
    def preprocess_text(self, text: str) -> Dict[str, List[int]]:
        """
        Preprocess text for model input.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Clean text
        text = self.clean_text(text)
        
        # Create tokenizer if not exists
        if self.tokenizer is None:
            self.tokenizer = self.create_tokenizer(self.tokenizer_config)
        
        # Tokenize text
        return self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
    
    def batch_preprocess(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with input_ids and attention_mask arrays
        """
        # Process each text
        processed = [self.preprocess_text(text) for text in texts]
        
        # Extract input_ids and attention_mask
        input_ids = [p["input_ids"] for p in processed]
        attention_mask = [p["attention_mask"] for p in processed]
        
        # Convert to numpy arrays
        return {
            "input_ids": np.array(input_ids),
            "attention_mask": np.array(attention_mask)
        }
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Create tokenizer if not exists
        if self.tokenizer is None:
            self.tokenizer = self.create_tokenizer(self.tokenizer_config)
        
        # Decode IDs
        return self.tokenizer.decode(ids)


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create temporary directory for tokenizer
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create processor
        processor = TextPreprocessor(
            max_length=128,
            tokenizer_config={"type": "bpe", "vocab_size": 1000},
            tokenizer_path=temp_dir
        )
        
        # Test texts
        texts = [
            "Hello, world!",
            "This is a test.",
            "How are you doing today?"
        ]
        
        # Train tokenizer
        processor.train_tokenizer(texts)
        
        # Test preprocessing
        for text in texts:
            processed = processor.preprocess_text(text)
            logger.info(f"Text: {text}")
            logger.info(f"Processed: {processed}")
            
            # Test decoding
            decoded = processor.decode(processed["input_ids"])
            logger.info(f"Decoded: {decoded}")
            logger.info("")
