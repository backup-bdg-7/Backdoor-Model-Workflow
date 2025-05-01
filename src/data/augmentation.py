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
Data augmentation utilities for the AI model training workflow.
This module provides functions for augmenting text and other data types.
"""

import os
import logging
import random
import re
import copy
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
from collections import defaultdict

# Try to import optional dependencies
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.sentence as nas
    import nlpaug.flow as naflow
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False

try:
    from datasets import Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
if NLTK_AVAILABLE:
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)


class TextAugmenter:
    """
    A class to handle text data augmentation.
    """
    
    def __init__(
        self,
        techniques: Optional[List[str]] = None,
        probabilities: Optional[List[float]] = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        """
        Initialize the text augmenter.
        
        Args:
            techniques: List of augmentation techniques to use
            probabilities: List of probabilities for each technique
            seed: Random seed
            device: Device to use for model-based augmentations
        """
        self.techniques = techniques or ["synonym_replacement", "random_deletion", "random_swap", "random_insertion"]
        self.probabilities = probabilities or [0.25] * len(self.techniques)
        self.seed = seed
        self.device = device
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Check if required libraries are available
        if not NLTK_AVAILABLE:
            logger.warning("NLTK is not installed. Some augmentation techniques may not work.")
        
        # Initialize augmenters
        self._initialize_augmenters()
    
    def _initialize_augmenters(self) -> None:
        """
        Initialize augmenters for each technique.
        """
        self.augmenters = {}
        
        # Basic augmenters
        if "synonym_replacement" in self.techniques:
            self.augmenters["synonym_replacement"] = self._synonym_replacement
        
        if "random_deletion" in self.techniques:
            self.augmenters["random_deletion"] = self._random_deletion
        
        if "random_swap" in self.techniques:
            self.augmenters["random_swap"] = self._random_swap
        
        if "random_insertion" in self.techniques:
            self.augmenters["random_insertion"] = self._random_insertion
        
        # NLPAug augmenters
        if NLPAUG_AVAILABLE:
            if "contextual_word_embs" in self.techniques:
                if not TRANSFORMERS_AVAILABLE:
                    logger.warning("transformers is not installed. Contextual word embeddings augmentation will not work.")
                else:
                    self.augmenters["contextual_word_embs"] = naw.ContextualWordEmbsAug(
                        model_path='bert-base-uncased',
                        action="substitute",
                        device=self.device
                    )
            
            if "word_embs" in self.techniques:
                self.augmenters["word_embs"] = naw.WordEmbsAug(
                    model_type='word2vec',
                    action="substitute"
                )
            
            if "back_translation" in self.techniques:
                self.augmenters["back_translation"] = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de',
                    to_model_name='facebook/wmt19-de-en',
                    device=self.device
                )
            
            if "spelling" in self.techniques:
                self.augmenters["spelling"] = nac.SpellingAug()
            
            if "keyboard" in self.techniques:
                self.augmenters["keyboard"] = nac.KeyboardAug()
            
            if "sentence_paraphrase" in self.techniques:
                self.augmenters["sentence_paraphrase"] = nas.AbstSummAug(
                    model_path='facebook/bart-large-cnn',
                    device=self.device
                )
        
        # Transformers-based augmenters
        if TRANSFORMERS_AVAILABLE:
            if "masked_lm" in self.techniques:
                self.augmenters["masked_lm"] = self._initialize_masked_lm()
    
    def _initialize_masked_lm(self) -> Callable:
        """
        Initialize masked language model for augmentation.
        
        Returns:
            Function for masked language model augmentation
        """
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(self.device)
        fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=self.device if self.device != "cpu" else -1)
        
        def masked_lm_augment(text: str, prob: float = 0.15, top_k: int = 5) -> str:
            """
            Augment text using masked language model.
            
            Args:
                text: Input text
                prob: Probability of masking each token
                top_k: Number of top predictions to sample from
                
            Returns:
                Augmented text
            """
            tokens = tokenizer.tokenize(text)
            masked_tokens = tokens.copy()
            
            # Mask random tokens
            for i in range(len(tokens)):
                if random.random() < prob and tokens[i] not in ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]:
                    masked_tokens[i] = "[MASK]"
            
            # Convert back to text
            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
            
            # Fill masks
            if "[MASK]" in masked_text:
                try:
                    results = fill_mask(masked_text)
                    
                    # Handle multiple masks
                    if isinstance(results[0], list):
                        # Flatten results
                        all_results = []
                        for res_list in results:
                            all_results.extend(res_list)
                        
                        # Sort by position
                        all_results.sort(key=lambda x: x["index"])
                        
                        # Replace masks one by one
                        augmented_text = text
                        for result in all_results:
                            mask_token = result["token_str"]
                            mask_idx = result["index"]
                            
                            # Sample from top-k
                            sampled_result = random.choice(result[:top_k])
                            sampled_token = sampled_result["token_str"]
                            
                            # Replace mask
                            augmented_text = augmented_text.replace("[MASK]", sampled_token, 1)
                        
                        return augmented_text
                    else:
                        # Sample from top-k
                        sampled_result = random.choice(results[:top_k])
                        return sampled_result["sequence"]
                except Exception as e:
                    logger.warning(f"Error in masked language model augmentation: {e}")
                    return text
            
            return text
        
        return masked_lm_augment
    
    def _synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words in the text with their synonyms.
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Augmented text
        """
        if not NLTK_AVAILABLE:
            return text
        
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            
            if num_replaced >= n:
                break
        
        return " ".join(new_words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from the text with probability p.
        
        Args:
            text: Input text
            p: Probability of deleting each word
            
        Returns:
            Augmented text
        """
        if not NLTK_AVAILABLE:
            return text
        
        words = nltk.word_tokenize(text)
        
        # If there's only one word, return that word
        if len(words) == 1:
            return text
        
        # Randomly delete words with probability p
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        # If all words are deleted, just return a random word
        if len(new_words) == 0:
            return random.choice(words)
        
        return " ".join(new_words)
    
    def _random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words in the text.
        
        Args:
            text: Input text
            n: Number of pairs to swap
            
        Returns:
            Augmented text
        """
        if not NLTK_AVAILABLE:
            return text
        
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:  # Need at least 2 words to swap
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return " ".join(new_words)
    
    def _random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n words into the text.
        
        Args:
            text: Input text
            n: Number of words to insert
            
        Returns:
            Augmented text
        """
        if not NLTK_AVAILABLE:
            return text
        
        words = nltk.word_tokenize(text)
        new_words = words.copy()
        
        for _ in range(n):
            self._add_word(new_words)
        
        return " ".join(new_words)
    
    def _add_word(self, words: List[str]) -> None:
        """
        Add a random synonym of a random word to a random position in the text.
        
        Args:
            words: List of words
        """
        if not words:
            return
        
        random_word = random.choice([word for word in words if word.isalnum()])
        synonyms = self._get_synonyms(random_word)
        
        if synonyms:
            random_synonym = random.choice(list(synonyms))
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, random_synonym)
    
    def _get_synonyms(self, word: str) -> set:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Input word
            
        Returns:
            Set of synonyms
        """
        if not NLTK_AVAILABLE:
            return set()
        
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word:
                    synonyms.add(synonym)
        
        return synonyms
    
    def augment(self, text: str, n_aug: int = 1) -> List[str]:
        """
        Augment text using the specified techniques.
        
        Args:
            text: Input text
            n_aug: Number of augmentations to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(n_aug):
            aug_text = text
            
            # Apply augmentation techniques with their probabilities
            for technique, prob in zip(self.techniques, self.probabilities):
                if random.random() < prob:
                    augmenter = self.augmenters.get(technique)
                    if augmenter:
                        if callable(augmenter):
                            aug_text = augmenter(aug_text)
                        elif hasattr(augmenter, 'augment'):
                            try:
                                aug_text = augmenter.augment(aug_text)
                            except Exception as e:
                                logger.warning(f"Error applying {technique} augmentation: {e}")
            
            augmented_texts.append(aug_text)
        
        return augmented_texts
    
    def augment_batch(self, texts: List[str], n_aug: int = 1) -> List[str]:
        """
        Augment a batch of texts.
        
        Args:
            texts: List of input texts
            n_aug: Number of augmentations to generate per text
            
        Returns:
            List of augmented texts
        """
        all_augmented_texts = []
        
        for text in texts:
            augmented = self.augment(text, n_aug)
            all_augmented_texts.extend(augmented)
        
        return all_augmented_texts
    
    def augment_dataset(
        self,
        dataset: Any,
        text_column: str,
        n_aug: int = 1,
        keep_original: bool = True,
    ) -> Any:
        """
        Augment a dataset.
        
        Args:
            dataset: Input dataset
            text_column: Name of the column containing text
            n_aug: Number of augmentations to generate per example
            keep_original: Whether to keep original examples
            
        Returns:
            Augmented dataset
        """
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library is not installed. Dataset augmentation will not work.")
            return dataset
        
        # Handle different dataset types
        if isinstance(dataset, Dataset):
            return self._augment_hf_dataset(dataset, text_column, n_aug, keep_original)
        elif isinstance(dataset, DatasetDict):
            augmented_dataset = {}
            for split, ds in dataset.items():
                augmented_dataset[split] = self._augment_hf_dataset(ds, text_column, n_aug, keep_original)
            return DatasetDict(augmented_dataset)
        elif isinstance(dataset, dict):
            # Assume dictionary with 'train', 'validation', etc. keys
            augmented_dataset = {}
            for split, ds in dataset.items():
                augmented_dataset[split] = self._augment_hf_dataset(ds, text_column, n_aug, keep_original)
            return augmented_dataset
        elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            # Generic dataset-like object
            return self._augment_generic_dataset(dataset, text_column, n_aug, keep_original)
        else:
            logger.warning(f"Unsupported dataset type: {type(dataset)}. Returning original dataset.")
            return dataset
    
    def _augment_hf_dataset(
        self,
        dataset: Dataset,
        text_column: str,
        n_aug: int = 1,
        keep_original: bool = True,
    ) -> Dataset:
        """
        Augment a HuggingFace dataset.
        
        Args:
            dataset: Input dataset
            text_column: Name of the column containing text
            n_aug: Number of augmentations to generate per example
            keep_original: Whether to keep original examples
            
        Returns:
            Augmented dataset
        """
        # Create new examples
        new_examples = []
        
        # Keep track of columns
        columns = list(dataset.features.keys())
        
        # Process each example
        for example in dataset:
            # Keep original if requested
            if keep_original:
                new_examples.append({k: example[k] for k in columns})
            
            # Generate augmentations
            text = example[text_column]
            augmented_texts = self.augment(text, n_aug)
            
            # Create new examples with augmented text
            for aug_text in augmented_texts:
                new_example = {k: example[k] for k in columns}
                new_example[text_column] = aug_text
                new_examples.append(new_example)
        
        # Create new dataset
        return Dataset.from_dict({k: [example[k] for example in new_examples] for k in columns})
    
    def _augment_generic_dataset(
        self,
        dataset: Any,
        text_column: str,
        n_aug: int = 1,
        keep_original: bool = True,
    ) -> List[Dict]:
        """
        Augment a generic dataset.
        
        Args:
            dataset: Input dataset
            text_column: Name of the column containing text
            n_aug: Number of augmentations to generate per example
            keep_original: Whether to keep original examples
            
        Returns:
            Augmented dataset as list of dictionaries
        """
        # Create new examples
        new_examples = []
        
        # Process each example
        for example in dataset:
            # Handle different example formats
            if isinstance(example, dict):
                # Keep original if requested
                if keep_original:
                    new_examples.append(copy.deepcopy(example))
                
                # Generate augmentations
                text = example[text_column]
                augmented_texts = self.augment(text, n_aug)
                
                # Create new examples with augmented text
                for aug_text in augmented_texts:
                    new_example = copy.deepcopy(example)
                    new_example[text_column] = aug_text
                    new_examples.append(new_example)
            else:
                logger.warning(f"Unsupported example type: {type(example)}. Skipping.")
        
        return new_examples


class CodeAugmenter:
    """
    A class to handle code data augmentation.
    """
    
    def __init__(
        self,
        techniques: Optional[List[str]] = None,
        probabilities: Optional[List[float]] = None,
        seed: int = 42,
    ):
        """
        Initialize the code augmenter.
        
        Args:
            techniques: List of augmentation techniques to use
            probabilities: List of probabilities for each technique
            seed: Random seed
        """
        self.techniques = techniques or ["variable_renaming", "comment_modification", "whitespace_modification"]
        self.probabilities = probabilities or [0.3, 0.3, 0.3]
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
    
    def _variable_renaming(self, code: str) -> str:
        """
        Rename variables in code.
        
        Args:
            code: Input code
            
        Returns:
            Augmented code
        """
        # Simple regex-based variable extraction
        # This is a simplified approach and may not work for all programming languages
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Find all potential variables
        potential_vars = set(re.findall(variable_pattern, code))
        
        # Filter out keywords and built-ins (simplified for Python)
        python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
            'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
            'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
            'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
        }
        
        variables = [var for var in potential_vars if var not in python_keywords and len(var) > 1]
        
        # Create variable mapping
        var_mapping = {}
        for var in variables:
            if random.random() < 0.5:  # Only rename some variables
                new_name = f"var_{random.randint(1, 1000)}"
                var_mapping[var] = new_name
        
        # Replace variables
        augmented_code = code
        for old_name, new_name in var_mapping.items():
            # Use word boundaries to avoid partial replacements
            augmented_code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, augmented_code)
        
        return augmented_code
    
    def _comment_modification(self, code: str) -> str:
        """
        Modify comments in code.
        
        Args:
            code: Input code
            
        Returns:
            Augmented code
        """
        # Handle single-line comments (Python style)
        single_line_pattern = r'(#.+)$'
        
        # Replace single-line comments
        def replace_single_comment(match):
            if random.random() < 0.5:
                return "# Modified comment"
            else:
                return ""  # Remove comment
        
        augmented_code = re.sub(single_line_pattern, replace_single_comment, code, flags=re.MULTILINE)
        
        # Handle multi-line comments (Python style)
        multi_line_pattern = r'(""".*?""")|(\'\'\'.*?\'\'\')'
        
        # Replace multi-line comments
        def replace_multi_comment(match):
            if random.random() < 0.5:
                quote = '"""' if match.group(1) else "'''"
                return f"{quote}Modified multi-line comment{quote}"
            else:
                return ""  # Remove comment
        
        augmented_code = re.sub(multi_line_pattern, replace_multi_comment, augmented_code, flags=re.DOTALL)
        
        return augmented_code
    
    def _whitespace_modification(self, code: str) -> str:
        """
        Modify whitespace in code.
        
        Args:
            code: Input code
            
        Returns:
            Augmented code
        """
        lines = code.split('\n')
        augmented_lines = []
        
        for line in lines:
            # Randomly add or remove spaces around operators
            if random.random() < 0.3:
                # Add spaces around operators
                line = re.sub(r'([+\-*/=<>!&|^%]+)', r' \1 ', line)
            elif random.random() < 0.3:
                # Remove spaces around operators
                line = re.sub(r'\s*([+\-*/=<>!&|^%]+)\s*', r'\1', line)
            
            # Randomly add or remove indentation
            if line.strip() and random.random() < 0.2:
                if line.startswith('    '):
                    line = line[4:]  # Remove one level of indentation
                else:
                    line = '    ' + line  # Add one level of indentation
            
            augmented_lines.append(line)
        
        return '\n'.join(augmented_lines)
    
    def augment(self, code: str, n_aug: int = 1) -> List[str]:
        """
        Augment code using the specified techniques.
        
        Args:
            code: Input code
            n_aug: Number of augmentations to generate
            
        Returns:
            List of augmented code snippets
        """
        augmented_codes = []
        
        for _ in range(n_aug):
            aug_code = code
            
            # Apply augmentation techniques with their probabilities
            for technique, prob in zip(self.techniques, self.probabilities):
                if random.random() < prob:
                    if technique == "variable_renaming":
                        aug_code = self._variable_renaming(aug_code)
                    elif technique == "comment_modification":
                        aug_code = self._comment_modification(aug_code)
                    elif technique == "whitespace_modification":
                        aug_code = self._whitespace_modification(aug_code)
            
            augmented_codes.append(aug_code)
        
        return augmented_codes
    
    def augment_batch(self, codes: List[str], n_aug: int = 1) -> List[str]:
        """
        Augment a batch of code snippets.
        
        Args:
            codes: List of input code snippets
            n_aug: Number of augmentations to generate per snippet
            
        Returns:
            List of augmented code snippets
        """
        all_augmented_codes = []
        
        for code in codes:
            augmented = self.augment(code, n_aug)
            all_augmented_codes.extend(augmented)
        
        return all_augmented_codes


def augment_text(
    text: str,
    techniques: Optional[List[str]] = None,
    probabilities: Optional[List[float]] = None,
    n_aug: int = 1,
    seed: int = 42,
    device: str = "cpu",
) -> List[str]:
    """
    Augment text using various techniques.
    
    Args:
        text: Input text
        techniques: List of augmentation techniques to use
        probabilities: List of probabilities for each technique
        n_aug: Number of augmentations to generate
        seed: Random seed
        device: Device to use for model-based augmentations
        
    Returns:
        List of augmented texts
    """
    augmenter = TextAugmenter(
        techniques=techniques,
        probabilities=probabilities,
        seed=seed,
        device=device
    )
    
    return augmenter.augment(text, n_aug)


def augment_code(
    code: str,
    techniques: Optional[List[str]] = None,
    probabilities: Optional[List[float]] = None,
    n_aug: int = 1,
    seed: int = 42,
) -> List[str]:
    """
    Augment code using various techniques.
    
    Args:
        code: Input code
        techniques: List of augmentation techniques to use
        probabilities: List of probabilities for each technique
        n_aug: Number of augmentations to generate
        seed: Random seed
        
    Returns:
        List of augmented code snippets
    """
    augmenter = CodeAugmenter(
        techniques=techniques,
        probabilities=probabilities,
        seed=seed
    )
    
    return augmenter.augment(code, n_aug)


def augment_dataset(
    dataset: Any,
    text_column: str,
    techniques: Optional[List[str]] = None,
    probabilities: Optional[List[float]] = None,
    n_aug: int = 1,
    keep_original: bool = True,
    seed: int = 42,
    device: str = "cpu",
    is_code: bool = False,
) -> Any:
    """
    Augment a dataset.
    
    Args:
        dataset: Input dataset
        text_column: Name of the column containing text
        techniques: List of augmentation techniques to use
        probabilities: List of probabilities for each technique
        n_aug: Number of augmentations to generate per example
        keep_original: Whether to keep original examples
        seed: Random seed
        device: Device to use for model-based augmentations
        is_code: Whether the text is code
        
    Returns:
        Augmented dataset
    """
    if is_code:
        augmenter = CodeAugmenter(
            techniques=techniques,
            probabilities=probabilities,
            seed=seed
        )
    else:
        augmenter = TextAugmenter(
            techniques=techniques,
            probabilities=probabilities,
            seed=seed,
            device=device
        )
    
    return augmenter.augment_dataset(
        dataset=dataset,
        text_column=text_column,
        n_aug=n_aug,
        keep_original=keep_original
    )