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
Evaluation metrics for the AI model training workflow.
This module provides functions to compute various evaluation metrics.
"""

import logging
import re
import math
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute perplexity from logits and labels.
    
    Args:
        logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
        labels: Ground truth labels of shape [batch_size, seq_len]
        ignore_index: Index to ignore in labels
        
    Returns:
        Perplexity value
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Compute perplexity
    return math.exp(loss.item())

def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute token-level accuracy from logits and labels.
    
    Args:
        logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
        labels: Ground truth labels of shape [batch_size, seq_len]
        ignore_index: Index to ignore in labels
        
    Returns:
        Token accuracy value
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Get predictions
    preds = torch.argmax(shift_logits, dim=-1)
    
    # Create mask for valid tokens
    mask = (shift_labels != ignore_index)
    
    # Compute accuracy
    correct = ((preds == shift_labels) & mask).sum().item()
    total = mask.sum().item()
    
    if total > 0:
        return correct / total
    else:
        return 0.0

def compute_sequence_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute sequence-level accuracy from logits and labels.
    
    Args:
        logits: Predicted logits of shape [batch_size, seq_len, vocab_size]
        labels: Ground truth labels of shape [batch_size, seq_len]
        ignore_index: Index to ignore in labels
        
    Returns:
        Sequence accuracy value
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Get predictions
    preds = torch.argmax(shift_logits, dim=-1)
    
    # Create mask for valid tokens
    mask = (shift_labels != ignore_index)
    
    # Compute sequence accuracy
    correct_sequences = 0
    batch_size = shift_labels.size(0)
    
    for i in range(batch_size):
        seq_mask = mask[i]
        seq_preds = preds[i][seq_mask]
        seq_labels = shift_labels[i][seq_mask]
        
        if torch.all(seq_preds == seq_labels):
            correct_sequences += 1
    
    return correct_sequences / batch_size

def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """
    Compute BLEU scores for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of lists of reference texts
        
    Returns:
        Dictionary with BLEU scores
    """
    # Tokenize predictions and references
    tokenized_preds = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    tokenized_refs = [[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in references]
    
    # Initialize smoothing function
    smoothing = SmoothingFunction().method1
    
    # Compute BLEU scores
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_3 = 0.0
    bleu_4 = 0.0
    
    for pred, refs in zip(tokenized_preds, tokenized_refs):
        # Skip empty predictions or references
        if not pred or not any(refs):
            continue
        
        # Compute BLEU scores with different n-gram weights
        bleu_1 += sentence_bleu(refs, pred, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 += sentence_bleu(refs, pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 += sentence_bleu(refs, pred, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 += sentence_bleu(refs, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    # Compute average scores
    n = len(predictions)
    
    return {
        "bleu_1": bleu_1 / n if n > 0 else 0.0,
        "bleu_2": bleu_2 / n if n > 0 else 0.0,
        "bleu_3": bleu_3 / n if n > 0 else 0.0,
        "bleu_4": bleu_4 / n if n > 0 else 0.0
    }

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE scores
    """
    # Initialize Rouge
    rouge = Rouge()
    
    # Filter out empty predictions or references
    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if pred.strip() and ref.strip()]
    
    if not valid_pairs:
        return {
            "rouge_1_f": 0.0,
            "rouge_2_f": 0.0,
            "rouge_l_f": 0.0
        }
    
    # Unzip valid pairs
    valid_preds, valid_refs = zip(*valid_pairs)
    
    try:
        # Compute ROUGE scores
        scores = rouge.get_scores(valid_preds, valid_refs, avg=True)
        
        return {
            "rouge_1_f": scores["rouge-1"]["f"],
            "rouge_2_f": scores["rouge-2"]["f"],
            "rouge_l_f": scores["rouge-l"]["f"]
        }
    except Exception as e:
        logger.warning(f"Error computing ROUGE scores: {e}")
        return {
            "rouge_1_f": 0.0,
            "rouge_2_f": 0.0,
            "rouge_l_f": 0.0
        }

def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """
    Compute METEOR score for text generation.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        METEOR score
    """
    # Tokenize predictions and references
    tokenized_preds = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    tokenized_refs = [nltk.word_tokenize(ref.lower()) for ref in references]
    
    # Compute METEOR scores
    meteor_scores = []
    
    for pred, ref in zip(tokenized_preds, tokenized_refs):
        # Skip empty predictions or references
        if not pred or not ref:
            continue
        
        # Compute METEOR score
        score = meteor_score([ref], pred)
        meteor_scores.append(score)
    
    # Compute average score
    if meteor_scores:
        return sum(meteor_scores) / len(meteor_scores)
    else:
        return 0.0

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match score for question answering.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Exact match score
    """
    # Normalize predictions and references
    normalized_preds = [normalize_answer(pred) for pred in predictions]
    normalized_refs = [normalize_answer(ref) for ref in references]
    
    # Compute exact match
    exact_matches = sum(pred == ref for pred, ref in zip(normalized_preds, normalized_refs))
    
    return exact_matches / len(predictions) if predictions else 0.0

def compute_f1(predictions: List[str], references: List[str]) -> float:
    """
    Compute token-level F1 score for question answering.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        F1 score
    """
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Normalize and tokenize
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()
        
        # Skip empty predictions or references
        if not pred_tokens or not ref_tokens:
            continue
        
        # Compute precision, recall, and F1
        common = sum(1 for token in pred_tokens if token in ref_tokens)
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def normalize_answer(text: str) -> str:
    """
    Normalize answer text for evaluation.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def compute_pass_at_k(predictions: List[List[str]], references: List[str], k: int = 1) -> float:
    """
    Compute pass@k metric for code generation.
    
    Args:
        predictions: List of lists of predicted code samples (k samples per problem)
        references: List of reference code samples or expected outputs
        k: Number of samples to consider
        
    Returns:
        pass@k score
    """
    n_problems = len(references)
    n_correct = 0
    
    for i in range(n_problems):
        # Get predictions for the current problem
        problem_preds = predictions[i][:k]
        problem_ref = references[i]
        
        # Check if any prediction is correct
        for pred in problem_preds:
            if is_code_correct(pred, problem_ref):
                n_correct += 1
                break
    
    return n_correct / n_problems if n_problems > 0 else 0.0

def is_code_correct(prediction: str, reference: str) -> bool:
    """
    Check if predicted code is functionally correct by executing it and comparing outputs.
    
    Args:
        prediction: Predicted code
        reference: Reference code or expected output
        
    Returns:
        True if code is correct, False otherwise
    """
    import tempfile
    import subprocess
    import os
    import re
    
    # Extract expected output from reference if it contains output comments
    expected_output_match = re.search(r'# Expected output:\s*(.+)', reference)
    expected_output = expected_output_match.group(1).strip() if expected_output_match else None
    
    # If no expected output is found, normalize and compare the code directly
    if expected_output is None:
        return normalize_code(prediction) == normalize_code(reference)
    
    # Create temporary files for the code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        f.write(prediction.encode('utf-8'))
        pred_file = f.name
    
    try:
        # Execute the code and capture output
        result = subprocess.run(
            ['python', pred_file],
            capture_output=True,
            text=True,
            timeout=5  # Set a timeout to prevent infinite loops
        )
        
        # Check if execution was successful
        if result.returncode != 0:
            return False
        
        # Compare actual output with expected output
        actual_output = result.stdout.strip()
        return actual_output == expected_output
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(pred_file):
            os.remove(pred_file)

def normalize_code(code: str) -> str:
    """
    Normalize code for comparison.
    
    Args:
        code: Input code
        
    Returns:
        Normalized code
    """
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    
    return code

def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, task_type: str = 'general') -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        predictions: Predicted values
        labels: Ground truth labels
        task_type: Type of task (general, code, dialogue, qa)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Compute general metrics
    if task_type == 'general':
        # Compute perplexity if logits are available
        if hasattr(predictions, 'shape') and len(predictions.shape) == 3:
            metrics['perplexity'] = compute_perplexity(predictions, labels)
        
        # Compute token accuracy
        metrics['token_accuracy'] = compute_token_accuracy(predictions, labels)
    
    # Compute code-specific metrics
    elif task_type == 'code':
        # Compute token accuracy
        metrics['token_accuracy'] = compute_token_accuracy(predictions, labels)
        
        # Compute sequence accuracy
        metrics['sequence_accuracy'] = compute_sequence_accuracy(predictions, labels)
    
    # Compute dialogue-specific metrics
    elif task_type == 'dialogue':
        # Convert tensors to text for BLEU and ROUGE
        # This is a placeholder - in a real implementation, you would decode the tensors
        pred_texts = ["Placeholder prediction"] * len(predictions)
        ref_texts = ["Placeholder reference"] * len(labels)
        
        # Compute BLEU scores
        bleu_scores = compute_bleu(pred_texts, [[ref] for ref in ref_texts])
        metrics.update(bleu_scores)
        
        # Compute ROUGE scores
        rouge_scores = compute_rouge(pred_texts, ref_texts)
        metrics.update(rouge_scores)
    
    # Compute QA-specific metrics
    elif task_type == 'qa':
        # Convert tensors to text for exact match and F1
        # This is a placeholder - in a real implementation, you would decode the tensors
        pred_texts = ["Placeholder prediction"] * len(predictions)
        ref_texts = ["Placeholder reference"] * len(labels)
        
        # Compute exact match
        metrics['exact_match'] = compute_exact_match(pred_texts, ref_texts)
        
        # Compute F1 score
        metrics['f1'] = compute_f1(pred_texts, ref_texts)
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Compute metrics
    metrics = compute_metrics(logits, labels)
    
    logger.info(f"Computed metrics: {metrics}")
    
    # Test text-based metrics
    predictions = ["The cat sat on the mat.", "The dog ran in the park."]
    references = ["The cat sat on the mat.", "The dog walked in the park."]
    
    bleu_scores = compute_bleu(predictions, [[ref] for ref in references])
    logger.info(f"BLEU scores: {bleu_scores}")
    
    rouge_scores = compute_rouge(predictions, references)
    logger.info(f"ROUGE scores: {rouge_scores}")
    
    meteor_score = compute_meteor(predictions, references)
    logger.info(f"METEOR score: {meteor_score}")
    
    exact_match = compute_exact_match(predictions, references)
    logger.info(f"Exact match: {exact_match}")
    
    f1_score = compute_f1(predictions, references)
    logger.info(f"F1 score: {f1_score}")