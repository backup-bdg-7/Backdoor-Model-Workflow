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
Model evaluation utilities for the AI model training workflow.
This module provides functions for evaluating models with various metrics and visualizations.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score
)
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict

# Try to import optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasets import load_metric
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ModelEvaluator:
    """
    A class to handle model evaluation with various metrics and visualizations.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
        task_type: str = "text_generation",
        metrics: Optional[List[str]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to use for evaluation
            output_dir: Directory to save evaluation results
            task_type: Type of task (text_generation, classification, regression, etc.)
            metrics: List of metrics to compute
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: Weights & Biases project name
            wandb_run_name: Weights & Biases run name
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir or "./evaluation_results"
        self.task_type = task_type
        self.metrics = metrics or self._get_default_metrics()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Weights & Biases if enabled
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("Weights & Biases is not installed. Logging will be disabled.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project or "model_evaluation",
                    name=wandb_run_name or f"eval_{int(time.time())}",
                    config={
                        "task_type": task_type,
                        "metrics": metrics,
                    }
                )
    
    def _get_default_metrics(self) -> List[str]:
        """
        Get default metrics based on task type.
        
        Returns:
            List of default metrics
        """
        if self.task_type == "text_generation":
            return ["bleu", "rouge", "meteor", "perplexity"]
        elif self.task_type == "classification":
            return ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
        elif self.task_type == "regression":
            return ["mse", "mae", "r2"]
        else:
            return ["loss"]
    
    def evaluate(
        self,
        eval_dataset: Union[Dataset, DataLoader],
        batch_size: int = 16,
        max_samples: Optional[int] = None,
        num_beams: int = 4,
        max_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
        save_predictions: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Dataset or DataLoader for evaluation
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequences
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling for generation
            save_predictions: Whether to save predictions
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with evaluation results
        """
        # Prepare dataloader
        if isinstance(eval_dataset, DataLoader):
            dataloader = eval_dataset
        else:
            dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize results
        all_predictions = []
        all_references = []
        all_inputs = []
        all_labels = []
        all_logits = []
        all_losses = []
        
        # Evaluate model
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating", disable=not verbose)
            for i, batch in enumerate(progress_bar):
                # Limit number of samples if specified
                if max_samples is not None and i * batch_size >= max_samples:
                    break
                
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward pass
                if self.task_type == "text_generation":
                    # Generate text
                    input_ids = batch["input_ids"]
                    attention_mask = batch.get("attention_mask")
                    
                    # Store inputs
                    input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    all_inputs.extend(input_texts)
                    
                    # Generate outputs
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Decode outputs
                    predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    all_predictions.extend(predictions)
                    
                    # Get references if available
                    if "labels" in batch:
                        references = self.tokenizer.batch_decode(
                            batch["labels"], skip_special_tokens=True
                        )
                        all_references.extend(references)
                    
                    # Compute loss if labels are available
                    if "labels" in batch:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=batch["labels"]
                        )
                        all_losses.append(outputs.loss.item())
                
                elif self.task_type == "classification":
                    # Classification task
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    loss = outputs.loss if hasattr(outputs, "loss") else None
                    
                    # Store inputs, labels, and predictions
                    input_ids = batch["input_ids"]
                    input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    all_inputs.extend(input_texts)
                    
                    if "labels" in batch:
                        all_labels.extend(batch["labels"].cpu().numpy())
                    
                    all_logits.append(logits.cpu().numpy())
                    
                    if loss is not None:
                        all_losses.append(loss.item())
                
                elif self.task_type == "regression":
                    # Regression task
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    loss = outputs.loss if hasattr(outputs, "loss") else None
                    
                    # Store inputs, labels, and predictions
                    input_ids = batch["input_ids"]
                    input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    all_inputs.extend(input_texts)
                    
                    if "labels" in batch:
                        all_labels.extend(batch["labels"].cpu().numpy())
                    
                    all_logits.append(logits.cpu().numpy())
                    
                    if loss is not None:
                        all_losses.append(loss.item())
                
                else:
                    # Generic task
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else None
                    
                    if loss is not None:
                        all_losses.append(loss.item())
        
        # Concatenate logits if any
        if all_logits:
            all_logits = np.concatenate(all_logits, axis=0)
        
        # Compute predictions for classification and regression
        if self.task_type == "classification" and all_logits is not None:
            all_predictions = np.argmax(all_logits, axis=1).tolist()
        elif self.task_type == "regression" and all_logits is not None:
            all_predictions = all_logits.squeeze().tolist()
        
        # Compute metrics
        metrics_results = self._compute_metrics(
            predictions=all_predictions,
            references=all_references if all_references else all_labels,
            inputs=all_inputs,
            logits=all_logits,
            losses=all_losses,
        )
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(
                inputs=all_inputs,
                predictions=all_predictions,
                references=all_references if all_references else all_labels,
            )
        
        # Log to Weights & Biases if enabled
        if self.use_wandb:
            self._log_to_wandb(metrics_results, all_predictions, all_references, all_inputs)
        
        return metrics_results
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare batch for model input.
        
        Args:
            batch: Input batch
            
        Returns:
            Prepared batch
        """
        # Move tensors to device
        prepared_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared_batch[k] = v.to(self.device)
            else:
                prepared_batch[k] = v
        
        return prepared_batch
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of examples
            
        Returns:
            Collated batch
        """
        # Extract keys from the first example
        keys = batch[0].keys()
        
        # Initialize batch dictionary
        collated_batch = {}
        
        # Collate each key
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # Pad sequences to the same length
                values = [example[key] for example in batch]
                max_length = max(len(v) for v in values)
                
                # Create padded tensor
                padded_values = []
                for value in values:
                    padding_length = max_length - len(value)
                    if key == 'input_ids' or key == 'labels':
                        # Pad with tokenizer pad token ID
                        padded_value = value + [self.tokenizer.pad_token_id] * padding_length
                    else:
                        # Pad attention mask with 0
                        padded_value = value + [0] * padding_length
                    
                    padded_values.append(padded_value)
                
                # Convert to tensor
                collated_batch[key] = torch.tensor(padded_values)
            else:
                # For other keys, just collect values
                collated_batch[key] = [example[key] for example in batch]
        
        return collated_batch
    
    def _compute_metrics(
        self,
        predictions: List[Any],
        references: Optional[List[Any]] = None,
        inputs: Optional[List[str]] = None,
        logits: Optional[np.ndarray] = None,
        losses: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            inputs: Input texts
            logits: Model logits
            losses: Model losses
            
        Returns:
            Dictionary with metric results
        """
        metrics_results = {}
        
        # Compute loss if available
        if losses:
            metrics_results["loss"] = np.mean(losses)
        
        # Skip other metrics if no references are available
        if references is None or len(references) == 0:
            return metrics_results
        
        # Compute metrics based on task type
        if self.task_type == "text_generation":
            # Text generation metrics
            if "bleu" in self.metrics:
                metrics_results.update(self._compute_bleu(predictions, references))
            
            if "rouge" in self.metrics:
                metrics_results.update(self._compute_rouge(predictions, references))
            
            if "meteor" in self.metrics:
                metrics_results.update(self._compute_meteor(predictions, references))
            
            if "perplexity" in self.metrics and losses:
                metrics_results["perplexity"] = np.exp(np.mean(losses))
        
        elif self.task_type == "classification":
            # Classification metrics
            if "accuracy" in self.metrics:
                metrics_results["accuracy"] = accuracy_score(references, predictions)
            
            if "precision" in self.metrics:
                metrics_results["precision_macro"] = precision_score(
                    references, predictions, average="macro", zero_division=0
                )
                metrics_results["precision_weighted"] = precision_score(
                    references, predictions, average="weighted", zero_division=0
                )
            
            if "recall" in self.metrics:
                metrics_results["recall_macro"] = recall_score(
                    references, predictions, average="macro", zero_division=0
                )
                metrics_results["recall_weighted"] = recall_score(
                    references, predictions, average="weighted", zero_division=0
                )
            
            if "f1" in self.metrics:
                metrics_results["f1_macro"] = f1_score(
                    references, predictions, average="macro", zero_division=0
                )
                metrics_results["f1_weighted"] = f1_score(
                    references, predictions, average="weighted", zero_division=0
                )
            
            if "confusion_matrix" in self.metrics:
                metrics_results["confusion_matrix"] = confusion_matrix(
                    references, predictions
                ).tolist()
        
        elif self.task_type == "regression":
            # Regression metrics
            if "mse" in self.metrics:
                metrics_results["mse"] = mean_squared_error(references, predictions)
            
            if "mae" in self.metrics:
                metrics_results["mae"] = mean_absolute_error(references, predictions)
            
            if "r2" in self.metrics:
                metrics_results["r2"] = r2_score(references, predictions)
        
        return metrics_results
    
    def _compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dictionary with BLEU scores
        """
        # Tokenize predictions and references
        tokenized_predictions = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        tokenized_references = [nltk.word_tokenize(ref.lower()) for ref in references]
        
        # Compute BLEU scores
        bleu_scores = {}
        smoothing = SmoothingFunction().method1
        
        # Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4
        for n in range(1, 5):
            bleu_scores[f"bleu-{n}"] = corpus_bleu(
                [[ref] for ref in tokenized_references],
                tokenized_predictions,
                weights=tuple([1.0 / n] * n),
                smoothing_function=smoothing
            )
        
        return bleu_scores
    
    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dictionary with ROUGE scores
        """
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Compute ROUGE scores for each prediction-reference pair
        rouge_scores = defaultdict(list)
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key, score in scores.items():
                rouge_scores[f"{key}_precision"].append(score.precision)
                rouge_scores[f"{key}_recall"].append(score.recall)
                rouge_scores[f"{key}_fmeasure"].append(score.fmeasure)
        
        # Compute average scores
        avg_scores = {}
        for key, scores in rouge_scores.items():
            avg_scores[key] = np.mean(scores)
        
        return avg_scores
    
    def _compute_meteor(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute METEOR scores.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dictionary with METEOR scores
        """
        # Tokenize predictions and references
        tokenized_predictions = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        tokenized_references = [nltk.word_tokenize(ref.lower()) for ref in references]
        
        # Compute METEOR scores for each prediction-reference pair
        meteor_scores = []
        for pred, ref in zip(tokenized_predictions, tokenized_references):
            score = meteor_score([ref], pred)
            meteor_scores.append(score)
        
        return {"meteor": np.mean(meteor_scores)}
    
    def _save_predictions(
        self,
        inputs: List[str],
        predictions: List[Any],
        references: Optional[List[Any]] = None,
    ) -> None:
        """
        Save predictions to file.
        
        Args:
            inputs: Input texts
            predictions: Model predictions
            references: Ground truth references
        """
        # Create output file
        output_file = os.path.join(self.output_dir, f"predictions_{int(time.time())}.json")
        
        # Create output data
        output_data = []
        for i in range(len(inputs)):
            example = {
                "input": inputs[i],
                "prediction": predictions[i],
            }
            if references is not None and i < len(references):
                example["reference"] = references[i]
            
            output_data.append(example)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved predictions to {output_file}")
    
    def _log_to_wandb(
        self,
        metrics: Dict[str, Any],
        predictions: List[Any],
        references: Optional[List[Any]] = None,
        inputs: Optional[List[str]] = None,
    ) -> None:
        """
        Log evaluation results to Weights & Biases.
        
        Args:
            metrics: Evaluation metrics
            predictions: Model predictions
            references: Ground truth references
            inputs: Input texts
        """
        if not self.use_wandb:
            return
        
        # Log metrics
        wandb.log(metrics)
        
        # Log confusion matrix if available
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            wandb.log({"confusion_matrix_plot": wandb.Image(plt)})
            plt.close()
        
        # Log examples
        if inputs and predictions and references:
            examples_table = wandb.Table(columns=["Input", "Prediction", "Reference"])
            for i in range(min(len(inputs), 100)):  # Log up to 100 examples
                examples_table.add_data(inputs[i], predictions[i], references[i])
            
            wandb.log({"examples": examples_table})
    
    def visualize_results(
        self,
        metrics: Dict[str, Any],
        output_dir: Optional[str] = None,
        show_plots: bool = False,
    ) -> None:
        """
        Visualize evaluation results.
        
        Args:
            metrics: Evaluation metrics
            output_dir: Directory to save visualizations
            show_plots: Whether to show plots
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics summary
        metrics_summary = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        
        # Save metrics summary
        metrics_file = os.path.join(output_dir, "metrics_summary.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Create visualizations based on task type
        if self.task_type == "classification":
            # Visualize confusion matrix if available
            if "confusion_matrix" in metrics:
                cm = np.array(metrics["confusion_matrix"])
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
                if show_plots:
                    plt.show()
                plt.close()
            
            # Visualize metrics
            metrics_to_plot = [
                "accuracy", "precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted"
            ]
            metrics_values = [metrics.get(m, 0) for m in metrics_to_plot]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(metrics_to_plot, metrics_values)
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title("Classification Metrics")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f"{height:.3f}",
                    ha='center', va='bottom', rotation=0
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "classification_metrics.png"))
            if show_plots:
                plt.show()
            plt.close()
        
        elif self.task_type == "regression":
            # Visualize metrics
            metrics_to_plot = ["mse", "mae", "r2"]
            metrics_values = [metrics.get(m, 0) for m in metrics_to_plot]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics_to_plot, metrics_values)
            plt.title("Regression Metrics")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f"{height:.3f}",
                    ha='center', va='bottom', rotation=0
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "regression_metrics.png"))
            if show_plots:
                plt.show()
            plt.close()
        
        elif self.task_type == "text_generation":
            # Visualize BLEU scores
            bleu_metrics = {k: v for k, v in metrics.items() if k.startswith("bleu-")}
            if bleu_metrics:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(list(bleu_metrics.keys()), list(bleu_metrics.values()))
                plt.ylim(0, 1)
                plt.ylabel("Score")
                plt.title("BLEU Scores")
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f"{height:.3f}",
                        ha='center', va='bottom', rotation=0
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "bleu_scores.png"))
                if show_plots:
                    plt.show()
                plt.close()
            
            # Visualize ROUGE scores
            rouge_metrics = {k: v for k, v in metrics.items() if k.startswith("rouge")}
            if rouge_metrics:
                # Group by metric type
                rouge_groups = defaultdict(dict)
                for k, v in rouge_metrics.items():
                    parts = k.split("_")
                    metric = parts[0]
                    score_type = "_".join(parts[1:])
                    rouge_groups[metric][score_type] = v
                
                # Plot each ROUGE metric
                for metric, scores in rouge_groups.items():
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(list(scores.keys()), list(scores.values()))
                    plt.ylim(0, 1)
                    plt.ylabel("Score")
                    plt.title(f"{metric.upper()} Scores")
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height,
                            f"{height:.3f}",
                            ha='center', va='bottom', rotation=0
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{metric}_scores.png"))
                    if show_plots:
                        plt.show()
                    plt.close()
        
        logger.info(f"Saved visualizations to {output_dir}")


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_dataset: Union[Dataset, DataLoader],
    task_type: str = "text_generation",
    metrics: Optional[List[str]] = None,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    num_beams: int = 4,
    max_length: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    save_predictions: bool = True,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    visualize: bool = True,
    show_plots: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset or DataLoader for evaluation
        task_type: Type of task (text_generation, classification, regression, etc.)
        metrics: List of metrics to compute
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate
        num_beams: Number of beams for beam search
        max_length: Maximum length of generated sequences
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling for generation
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results
        save_predictions: Whether to save predictions
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name
        visualize: Whether to create visualizations
        show_plots: Whether to show plots
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with evaluation results
    """
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        task_type=task_type,
        metrics=metrics,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    
    # Evaluate model
    metrics_results = evaluator.evaluate(
        eval_dataset=eval_dataset,
        batch_size=batch_size,
        max_samples=max_samples,
        num_beams=num_beams,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        save_predictions=save_predictions,
        verbose=verbose,
    )
    
    # Visualize results if requested
    if visualize:
        evaluator.visualize_results(
            metrics=metrics_results,
            output_dir=output_dir,
            show_plots=show_plots,
        )
    
    return metrics_results