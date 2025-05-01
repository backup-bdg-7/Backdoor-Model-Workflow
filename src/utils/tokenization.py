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
Tokenization utilities for the AI model training workflow.
This module provides tokenizers for text processing.
"""

import os
import json
import logging
import regex as re
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Some tokenizer features will be unavailable.")

class Tokenizer:
    """
    General-purpose tokenizer for natural language and code.
    Supports multiple tokenization approaches and can be configured
    for different use cases.
    """
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        special_tokens: Optional[Dict[str, str]] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab: Vocabulary mapping tokens to IDs
            merges: BPE merge operations
            unk_token: Unknown token string
            pad_token: Padding token string
            bos_token: Beginning of sequence token string
            eos_token: End of sequence token string
            special_tokens: Additional special tokens
        """
        # Set up vocabulary
        self.vocab = vocab or {}
        self.merges = merges or []
        
        # Add special tokens
        self.special_tokens = {
            "unk_token": unk_token,
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token
        }
        
        if special_tokens:
            self.special_tokens.update(special_tokens)
        
        # Create inverted vocabulary (ID -> token)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Set token IDs
        self._add_special_tokens_to_vocab()
        
        # Prepare regex pattern for splitting
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def _add_special_tokens_to_vocab(self) -> None:
        """Add special tokens to vocabulary."""
        for token in self.special_tokens.values():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.vocab) - 1] = token
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.vocab.get(self.special_tokens["unk_token"], 0)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.vocab.get(self.special_tokens["pad_token"], 0)
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.vocab.get(self.special_tokens["bos_token"], 0)
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.vocab.get(self.special_tokens["eos_token"], 0)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: Input token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        return self.convert_tokens_to_string(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return list(re.findall(self.pattern, text))
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to IDs.
        
        Args:
            tokens: Input tokens
            
        Returns:
            List of token IDs
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        """
        Convert IDs to tokens.
        
        Args:
            ids: Input token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of tokens
        """
        tokens = []
        special_token_values = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        for id in ids:
            token = self.inv_vocab.get(id, self.special_tokens["unk_token"])
            if token not in special_token_values:
                tokens.append(token)
        
        return tokens
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Convert tokens to string.
        
        Args:
            tokens: Input tokens
            
        Returns:
            Joined string
        """
        return " ".join(tokens).strip()
    
    def save(self, path: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab": self.vocab,
                "merges": self.merges,
                "special_tokens": self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'Tokenizer':
        """
        Load tokenizer from file.
        
        Args:
            path: Path to tokenizer file
            
        Returns:
            Loaded tokenizer
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab=data.get("vocab", {}),
            merges=data.get("merges", []),
            special_tokens=data.get("special_tokens", {})
        )
        
        logger.info(f"Tokenizer loaded from {path}")
        return tokenizer
    
    @classmethod
    def from_pretrained(cls, name_or_path: str) -> 'Tokenizer':
        """
        Load tokenizer from pretrained model or local path.
        
        Args:
            name_or_path: Pretrained model name or local path
            
        Returns:
            Loaded tokenizer
        """
        if os.path.isdir(name_or_path) or os.path.isfile(name_or_path):
            # Load from local path
            path = name_or_path
            if os.path.isdir(path):
                path = os.path.join(path, "tokenizer.json")
            return cls.from_file(path)
        else:
            # Try to load from tiktoken
            if TIKTOKEN_AVAILABLE:
                return TiktokenTokenizer.from_pretrained(name_or_path)
            else:
                raise ValueError(f"Could not load tokenizer from {name_or_path}")


class BPETokenizer(Tokenizer):
    """
    Byte-Pair Encoding tokenizer.
    
    This tokenizer uses BPE algorithm to tokenize text.
    """
    
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        **kwargs
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab: Vocabulary mapping tokens to IDs
            merges: BPE merge operations
            **kwargs: Additional arguments for base Tokenizer
        """
        super().__init__(vocab=vocab, merges=merges, **kwargs)
        
        # Initialize BPE cache
        self.cache = {}
        
        # Initialize BPE ranks
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
    
    def bpe(self, token: str) -> str:
        """
        Apply BPE algorithm to token.
        
        Args:
            token: Input token
            
        Returns:
            BPE-encoded token
        """
        if token in self.cache:
            return self.cache[token]
        
        word = list(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            
            pairs = self._get_pairs(word)
        
        result = ' '.join(word)
        self.cache[token] = result
        return result
    
    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """
        Get pairs of consecutive tokens.
        
        Args:
            word: List of tokens
            
        Returns:
            Set of token pairs
        """
        pairs = set()
        prev_char = word[0]
        
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        
        return pairs
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = super().tokenize(text)
        bpe_tokens = []
        
        for token in tokens:
            # Apply BPE
            bpe_token = self.bpe(token)
            bpe_tokens.extend(bpe_token.split())
        
        return bpe_tokens


class TiktokenTokenizer(Tokenizer):
    """
    Tokenizer based on tiktoken library.
    
    This provides compatibility with OpenAI's tokenizers.
    """
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        **kwargs
    ):
        """
        Initialize tiktoken tokenizer.
        
        Args:
            encoding_name: Name of the tiktoken encoding
            **kwargs: Additional arguments for base Tokenizer
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is not installed. Please install it with `pip install tiktoken`.")
        
        super().__init__(**kwargs)
        
        # Get tiktoken encoding
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Update vocab with tiktoken encoding
        for token, token_id in self.encoding._mergeable_ranks.items():
            token_str = token.decode('utf-8', errors='replace')
            self.vocab[token_str] = token_id
            self.inv_vocab[token_id] = token_str
        
        # Add special tokens to vocab
        self._add_special_tokens_to_vocab()
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs using tiktoken.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        ids = self.encoding.encode(text)
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text using tiktoken.
        
        Args:
            ids: Input token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            special_ids = [self.vocab.get(token) for token in self.special_tokens.values()]
            ids = [id for id in ids if id not in special_ids]
        
        return self.encoding.decode(ids)
    
    @classmethod
    def from_pretrained(cls, name_or_path: str) -> 'TiktokenTokenizer':
        """
        Load tokenizer from pretrained model name.
        
        Args:
            name_or_path: Pretrained model name or encoding name
            
        Returns:
            Loaded tokenizer
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is not installed. Please install it with `pip install tiktoken`.")
        
        # Check if name is directly a tiktoken encoding
        try:
            return cls(encoding_name=name_or_path)
        except KeyError:
            # Try to get associated encoding
            model_to_encoding = {
                "gpt-4": "cl100k_base",
                "gpt-3.5-turbo": "cl100k_base",
                "text-embedding-ada-002": "cl100k_base",
                "text-davinci-003": "p50k_base",
                "text-davinci-002": "p50k_base",
                "davinci": "r50k_base",
                "curie": "r50k_base",
                "babbage": "r50k_base",
                "ada": "r50k_base"
            }
            
            if name_or_path in model_to_encoding:
                return cls(encoding_name=model_to_encoding[name_or_path])
            else:
                raise ValueError(f"Unknown model or encoding name: {name_or_path}")


def create_tokenizer(
    tokenizer_type: str,
    vocab_size: int = 50257,
    **kwargs
) -> Tokenizer:
    """
    Create a new tokenizer instance.
    
    Args:
        tokenizer_type: Type of tokenizer ('bpe', 'tiktoken')
        vocab_size: Vocabulary size
        **kwargs: Additional arguments for tokenizer
        
    Returns:
        New tokenizer instance
    """
    if tokenizer_type.lower() == "bpe":
        return BPETokenizer(**kwargs)
    elif tokenizer_type.lower() == "tiktoken":
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken is not installed. Please install it with `pip install tiktoken`.")
        return TiktokenTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def get_tokenizer(config: Dict[str, Any]) -> Union[Tokenizer, Any]:
    """
    Get a tokenizer based on configuration.
    
    Args:
        config: Tokenizer configuration dictionary
        
    Returns:
        Tokenizer instance
    """
    tokenizer_type = config.get("type", "").lower()
    
    # Handle HuggingFace tokenizers
    if tokenizer_type == "huggingface":
        try:
            from transformers import AutoTokenizer
            
            # Get the tokenizer name
            name = config.get("name", "gpt2")
            
            # Optional parameters
            use_fast = config.get("use_fast", True)
            max_length = config.get("max_length", 1024)
            padding_side = config.get("padding_side", "right")
            truncation_side = config.get("truncation_side", "right")
            
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                name,
                use_fast=use_fast,
                model_max_length=max_length
            )
            
            # Set padding and truncation sides
            tokenizer.padding_side = padding_side
            tokenizer.truncation_side = truncation_side
            
            # Ensure padding token is set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    # Set default padding token
                    tokenizer.pad_token = tokenizer.eos_token = "</s>"
            
            # Add BOS/EOS tokens if specified
            if config.get("add_bos_token", False) and tokenizer.bos_token is None:
                tokenizer.bos_token = "<s>"
            
            if config.get("add_eos_token", False) and tokenizer.eos_token is None:
                tokenizer.eos_token = "</s>"
            
            logger.info(f"Loaded HuggingFace tokenizer: {name}")
            return tokenizer
        
        except ImportError:
            logger.warning("transformers not installed. Falling back to basic tokenizer.")
            return create_tokenizer("bpe", vocab_size=config.get("vocab_size", 50257))
    
    # Our own tokenizer implementations
    elif tokenizer_type in ["bpe", "tiktoken"]:
        return create_tokenizer(
            tokenizer_type=tokenizer_type,
            vocab_size=config.get("vocab_size", 50257),
            encoding_name=config.get("encoding_name", "cl100k_base")
        )
    
    # Handle unknown tokenizer types
    else:
        logger.warning(f"Unknown tokenizer type: {tokenizer_type}. Falling back to BPE tokenizer.")
        return create_tokenizer("bpe", vocab_size=config.get("vocab_size", 50257))


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple tokenizer
    tokenizer = Tokenizer()
    
    # Test tokenization
    text = "Hello, world! How are you doing today?"
    tokens = tokenizer.tokenize(text)
    print(f"Tokenized: {tokens}")
    
    # Test encoding/decoding
    ids = tokenizer.encode(text)
    print(f"Encoded: {ids}")
    
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
    
    # Test saving and loading
    tokenizer.save("tokenizer.json")
    loaded_tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Test with tiktoken if available
    if TIKTOKEN_AVAILABLE:
        print("\nTesting tiktoken tokenizer:")
        tiktoken_tokenizer = TiktokenTokenizer()
        tiktoken_ids = tiktoken_tokenizer.encode(text)
        print(f"Tiktoken Encoded: {tiktoken_ids}")
        tiktoken_decoded = tiktoken_tokenizer.decode(tiktoken_ids)
        print(f"Tiktoken Decoded: {tiktoken_decoded}")
