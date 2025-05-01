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
Data preprocessing utilities for the AI model training workflow.
This module provides functions to preprocess and transform datasets for training.
"""

import re
import unicodedata
import html
import logging
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from datasets import Dataset, IterableDataset
import nltk
from nltk.corpus import wordnet
import random

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    """
    A class to handle preprocessing of datasets for training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataPreprocessor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.preprocessing_config = config['data_processing']['preprocessing']
        self.augmentation_config = config['data_processing']['augmentation']
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by applying various transformations.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return text
        
        # Apply Unicode normalization
        if self.preprocessing_config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFKC', text)
        
        # Normalize whitespace
        if self.preprocessing_config.get('normalize_whitespace', True):
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove HTML entities
        if self.preprocessing_config.get('remove_html', True):
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def filter_by_length(self, text: str) -> bool:
        """
        Check if text meets length requirements.
        
        Args:
            text: Input text
            
        Returns:
            True if text meets length requirements, False otherwise
        """
        if not isinstance(text, str):
            return True
        
        min_length = self.preprocessing_config.get('min_length', 0)
        max_length = self.preprocessing_config.get('max_length', float('inf'))
        
        text_length = len(text)
        return min_length <= text_length <= max_length
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return text
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Check length requirements
        if not self.filter_by_length(text):
            return None
        
        return text
    
    def preprocess_code(self, code: str) -> str:
        """
        Preprocess code with code-specific transformations.
        
        Args:
            code: Input code
            
        Returns:
            Preprocessed code
        """
        if not isinstance(code, str):
            return code
        
        # Normalize whitespace but preserve indentation
        lines = code.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Preserve leading whitespace
            leading_space = len(line) - len(line.lstrip())
            normalized_line = re.sub(r'\s+', ' ', line[leading_space:]).rstrip()
            normalized_lines.append(' ' * leading_space + normalized_line)
        
        return '\n'.join(normalized_lines)
    
    def preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single example.
        
        Args:
            example: Input example
            
        Returns:
            Preprocessed example
        """
        # Create a copy to avoid modifying the original
        processed = dict(example)
        
        # Process input field
        if 'input' in processed:
            processed['input'] = self.preprocess_text(processed['input'])
        
        # Process output field
        if 'output' in processed:
            # Check if output is code
            if processed.get('task_type') == 'code_generation' or processed.get('task_type') == 'code_completion':
                processed['output'] = self.preprocess_code(processed['output'])
            else:
                processed['output'] = self.preprocess_text(processed['output'])
        
        # Process text field (for general datasets)
        if 'text' in processed:
            processed['text'] = self.preprocess_text(processed['text'])
        
        # Process code field
        if 'code' in processed:
            processed['code'] = self.preprocess_code(processed['code'])
        
        # Process question field
        if 'question' in processed:
            processed['question'] = self.preprocess_text(processed['question'])
        
        # Process answer field
        if 'answer' in processed:
            processed['answer'] = self.preprocess_text(processed['answer'])
        
        # Process instruction field
        if 'instruction' in processed:
            processed['instruction'] = self.preprocess_text(processed['instruction'])
        
        # Process response field
        if 'response' in processed:
            processed['response'] = self.preprocess_text(processed['response'])
        
        return processed
    
    def preprocess_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        """
        Preprocess a batch of examples.
        
        Args:
            batch: Batch of examples
            
        Returns:
            Preprocessed batch
        """
        result = {}
        
        # Process each field in the batch
        for key in batch:
            if key in ['input', 'text', 'question', 'instruction']:
                result[key] = [self.preprocess_text(text) for text in batch[key]]
            elif key in ['code']:
                result[key] = [self.preprocess_code(code) for code in batch[key]]
            elif key in ['output', 'answer', 'response']:
                # Check if output is code based on task_type
                if 'task_type' in batch and any(task in ['code_generation', 'code_completion'] for task in batch['task_type']):
                    result[key] = [self.preprocess_code(text) for text in batch[key]]
                else:
                    result[key] = [self.preprocess_text(text) for text in batch[key]]
            else:
                result[key] = batch[key]
        
        return result
    
    def augment_text(self, text: str) -> str:
        """
        Apply text augmentation techniques.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        if not isinstance(text, str) or not self.augmentation_config.get('enabled', False):
            return text
        
        # Get augmentation techniques
        techniques = self.augmentation_config.get('techniques', [])
        
        # Apply each technique with its probability
        for technique in techniques:
            name = technique['name']
            probability = technique.get('probability', 0.1)
            
            if random.random() < probability:
                if name == 'synonym_replacement':
                    text = self._synonym_replacement(text)
                elif name == 'random_insertion':
                    text = self._random_insertion(text)
                elif name == 'random_deletion':
                    text = self._random_deletion(text)
        
        return text
    
    def _synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words in the text with their synonyms.
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Text with synonyms replaced
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Choose n random words to replace
        n = min(n, len(words))
        random_word_indices = random.sample(range(len(words)), n)
        
        for idx in random_word_indices:
            word = words[idx]
            synonyms = self._get_synonyms(word)
            
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n words into the text.
        
        Args:
            text: Input text
            n: Number of words to insert
            
        Returns:
            Text with words inserted
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Choose n random words to get synonyms for
        n = min(n, len(words))
        random_word_indices = random.sample(range(len(words)), n)
        
        for idx in random_word_indices:
            word = words[idx]
            synonyms = self._get_synonyms(word)
            
            if synonyms:
                # Insert a synonym at a random position
                insert_position = random.randint(0, len(words))
                words.insert(insert_position, random.choice(synonyms))
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from the text with probability p.
        
        Args:
            text: Input text
            p: Probability of deleting each word
            
        Returns:
            Text with words deleted
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Randomly delete words with probability p
        kept_words = []
        for word in words:
            if random.random() >= p:
                kept_words.append(word)
        
        # If all words are deleted, keep a random one
        if not kept_words:
            kept_words = [random.choice(words)]
        
        return ' '.join(kept_words)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Input word
            
        Returns:
            List of synonyms
        """
        synonyms = []
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        
        return synonyms
    
    def augment_example(self, example: Dict) -> Dict:
        """
        Apply augmentation to a single example.
        
        Args:
            example: Input example
            
        Returns:
            Augmented example
        """
        if not self.augmentation_config.get('enabled', False):
            return example
        
        # Create a copy to avoid modifying the original
        augmented = dict(example)
        
        # Only augment input fields, not outputs
        if 'input' in augmented:
            augmented['input'] = self.augment_text(augmented['input'])
        
        if 'text' in augmented:
            augmented['text'] = self.augment_text(augmented['text'])
        
        if 'question' in augmented:
            augmented['question'] = self.augment_text(augmented['question'])
        
        if 'instruction' in augmented:
            augmented['instruction'] = self.augment_text(augmented['instruction'])
        
        return augmented
    
    def augment_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        """
        Apply augmentation to a batch of examples.
        
        Args:
            batch: Batch of examples
            
        Returns:
            Augmented batch
        """
        if not self.augmentation_config.get('enabled', False):
            return batch
        
        result = dict(batch)
        
        # Only augment input fields, not outputs
        for key in ['input', 'text', 'question', 'instruction']:
            if key in result:
                result[key] = [self.augment_text(text) for text in result[key]]
        
        return result
    
    def process_dataset(self, dataset: Union[Dataset, IterableDataset], 
                       augment: bool = False) -> Union[Dataset, IterableDataset]:
        """
        Process an entire dataset with preprocessing and optional augmentation.
        
        Args:
            dataset: Input dataset
            augment: Whether to apply augmentation
            
        Returns:
            Processed dataset
        """
        # Define processing function
        def process_fn(example):
            # Apply preprocessing
            processed = self.preprocess_example(example)
            
            # Apply augmentation if enabled
            if augment and self.augmentation_config.get('enabled', False):
                processed = self.augment_example(processed)
            
            return processed
        
        # Define batch processing function
        def batch_process_fn(batch):
            # Apply preprocessing
            processed = self.preprocess_batch(batch)
            
            # Apply augmentation if enabled
            if augment and self.augmentation_config.get('enabled', False):
                processed = self.augment_batch(processed)
            
            return processed
        
        # Apply processing to dataset
        if isinstance(dataset, IterableDataset):
            # For streaming datasets, use map without batching
            return dataset.map(process_fn)
        else:
            # For regular datasets, use batched processing
            return dataset.map(
                batch_process_fn,
                batched=True,
                num_proc=8  # Use multiple processes for efficiency
            )
    
    def unify_dataset_format(self, dataset: Union[Dataset, IterableDataset], 
                            task_type: str) -> Union[Dataset, IterableDataset]:
        """
        Unify dataset format to a consistent structure.
        
        Args:
            dataset: Input dataset
            task_type: Type of task (code_generation, dialogue, qa, etc.)
            
        Returns:
            Dataset with unified format
        """
        # Define mapping function based on task type and dataset structure
        def unify_fn(example):
            unified = {}
            
            # Determine input and output fields based on task type and available fields
            if task_type == 'code_generation' or task_type == 'code_completion':
                # For code tasks
                if 'prompt' in example:
                    unified['input'] = example['prompt']
                elif 'context' in example:
                    unified['input'] = example['context']
                elif 'question' in example:
                    unified['input'] = example['question']
                
                if 'completion' in example:
                    unified['output'] = example['completion']
                elif 'code' in example:
                    unified['output'] = example['code']
                elif 'solution' in example:
                    unified['output'] = example['solution']
                elif 'answer' in example:
                    unified['output'] = example['answer']
            
            elif task_type == 'dialogue' or task_type == 'conversation':
                # For dialogue tasks
                if 'prompt' in example:
                    unified['input'] = example['prompt']
                elif 'context' in example:
                    unified['input'] = example['context']
                elif 'question' in example:
                    unified['input'] = example['question']
                elif 'instruction' in example:
                    unified['input'] = example['instruction']
                
                if 'response' in example:
                    unified['output'] = example['response']
                elif 'completion' in example:
                    unified['output'] = example['completion']
                elif 'answer' in example:
                    unified['output'] = example['answer']
            
            elif task_type == 'qa' or task_type == 'question_answering':
                # For QA tasks
                if 'question' in example:
                    unified['input'] = example['question']
                elif 'prompt' in example:
                    unified['input'] = example['prompt']
                elif 'context' in example:
                    unified['input'] = example['context']
                
                if 'answer' in example:
                    unified['output'] = example['answer']
                elif 'response' in example:
                    unified['output'] = example['response']
                elif 'completion' in example:
                    unified['output'] = example['completion']
            
            elif task_type == 'instruction_following':
                # For instruction following tasks
                if 'instruction' in example:
                    unified['input'] = example['instruction']
                elif 'prompt' in example:
                    unified['input'] = example['prompt']
                elif 'question' in example:
                    unified['input'] = example['question']
                
                if 'response' in example:
                    unified['output'] = example['response']
                elif 'completion' in example:
                    unified['output'] = example['completion']
                elif 'answer' in example:
                    unified['output'] = example['answer']
            
            else:
                # For general tasks
                if 'text' in example:
                    # For unsupervised datasets, use text as both input and output
                    unified['input'] = example['text']
                    unified['output'] = example['text']
                elif 'input' in example and 'output' in example:
                    unified['input'] = example['input']
                    unified['output'] = example['output']
                elif 'prompt' in example and 'completion' in example:
                    unified['input'] = example['prompt']
                    unified['output'] = example['completion']
            
            # Add task type
            unified['task_type'] = task_type
            
            # Add metadata if available
            if 'metadata' in example:
                unified['metadata'] = example['metadata']
            else:
                # Create metadata from available fields
                metadata = {}
                for key in example:
                    if key not in ['input', 'output', 'task_type'] and key not in unified:
                        metadata[key] = example[key]
                
                if metadata:
                    unified['metadata'] = metadata
            
            return unified
        
        # Apply unification to dataset
        if isinstance(dataset, IterableDataset):
            return dataset.map(unify_fn)
        else:
            return dataset.map(unify_fn)
    
    def create_training_pairs(self, dataset: Union[Dataset, IterableDataset]) -> Union[Dataset, IterableDataset]:
        """
        Create training pairs from a unified dataset.
        
        Args:
            dataset: Input dataset with unified format
            
        Returns:
            Dataset with training pairs
        """
        def create_pairs_fn(example):
            # Create a training pair
            pair = {
                'input_text': example['input'],
                'target_text': example['output'],
                'task_type': example['task_type']
            }
            
            # Add metadata if available
            if 'metadata' in example:
                pair['metadata'] = example['metadata']
            
            return pair
        
        # Apply function to dataset
        if isinstance(dataset, IterableDataset):
            return dataset.map(create_pairs_fn)
        else:
            return dataset.map(create_pairs_fn)


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Test preprocessing
    text = "This is a   test with  multiple   spaces and\nnewlines."
    print(f"Original: '{text}'")
    print(f"Preprocessed: '{preprocessor.preprocess_text(text)}'")
    
    # Test augmentation
    if config['data_processing']['augmentation']['enabled']:
        augmented = preprocessor.augment_text(text)
        print(f"Augmented: '{augmented}'")
    
    # Test code preprocessing
    code = """def hello_world():
        print("Hello,   World!")
        
        # This is a comment
        return None"""
    print(f"Original code:\n{code}")
    print(f"Preprocessed code:\n{preprocessor.preprocess_code(code)}")
    
    # Test example preprocessing
    example = {
        'input': "What is the capital of France?",
        'output': "The capital of France is Paris.",
        'task_type': 'qa'
    }
    processed = preprocessor.preprocess_example(example)
    print(f"Processed example: {processed}")
    
    # Test augmentation
    if config['data_processing']['augmentation']['enabled']:
        augmented = preprocessor.augment_example(example)
        print(f"Augmented example: {augmented}")
    
    # Test unification
    unified = preprocessor.unify_dataset_format([example], 'qa')
    print(f"Unified example: {unified[0]}")
    
    # Test training pair creation
    pairs = preprocessor.create_training_pairs([unified[0]])
    print(f"Training pair: {pairs[0]}")