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
Training utilities for the AI model training workflow.
This module provides functions for training and fine-tuning models.
"""

import os
import logging
import time
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler, PreTrainedTokenizer
from datasets import Dataset as HFDataset, IterableDataset
import wandb
from tqdm import tqdm

from src.model.architecture import create_model_from_config
from src.utils.metrics import compute_metrics

# Configure logging
logger = logging.getLogger(__name__)

class TrainingArguments:
    """
    Arguments for model training.
    """
    
    def __init__(self, config: Dict, stage_name: str):
        """
        Initialize training arguments from configuration.
        
        Args:
            config: Training configuration
            stage_name: Name of the training stage
        """
        # Find the stage configuration
        stage_config = None
        for stage in config['training']['stages']:
            if stage['name'] == stage_name:
                stage_config = stage
                break
        
        if stage_config is None:
            raise ValueError(f"Training stage {stage_name} not found in configuration")
        
        # Set training parameters
        self.stage_name = stage_name
        self.datasets = stage_config['datasets']
        self.epochs = stage_config['epochs']
        
        # Learning rate settings
        lr_config = stage_config['learning_rate']
        self.learning_rate = lr_config['initial']
        self.min_learning_rate = lr_config['min']
        self.lr_schedule = lr_config['schedule']
        self.warmup_steps = lr_config['warmup_steps']
        
        # Optimizer settings
        optimizer_config = config['training']['optimizer']
        self.optimizer_name = optimizer_config['name']
        self.weight_decay = optimizer_config['weight_decay']
        self.beta1 = optimizer_config['beta1']
        self.beta2 = optimizer_config['beta2']
        self.epsilon = optimizer_config['eps']
        self.use_8bit_optimizer = optimizer_config['use_8bit']
        
        # Mixed precision settings
        self.mixed_precision = config['training']['mixed_precision']
        self.gradient_checkpointing = config['training']['gradient_checkpointing']
        self.gradient_clipping = config['training']['gradient_clipping']
        
        # Checkpointing settings
        checkpoint_config = config['training']['checkpointing']
        self.save_steps = checkpoint_config['save_steps']
        self.keep_last_n = checkpoint_config['keep_last_n']
        self.save_optimizer_state = checkpoint_config['save_optimizer_state']
        
        # Evaluation settings
        eval_config = config['training']['evaluation']
        self.eval_steps = eval_config['eval_steps']
        
        # Early stopping settings
        early_stopping_config = eval_config['early_stopping']
        self.early_stopping_enabled = early_stopping_config['enabled']
        self.early_stopping_patience = early_stopping_config['patience']
        self.early_stopping_metric = early_stopping_config['metric']
        self.early_stopping_mode = early_stopping_config['mode']
        
        # Batch size settings
        batch_config = config['data_processing']['batching']
        self.train_batch_size = batch_config['train_batch_size']
        self.eval_batch_size = batch_config['eval_batch_size']
        self.gradient_accumulation_steps = batch_config['gradient_accumulation_steps']
        self.dynamic_batching = batch_config['dynamic_batching']
        
        # Output directory
        self.output_dir = os.path.join(config['output_dir'], stage_name)
        
        # Resource settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Logging settings
        self.logging_steps = 10
        self.seed = config['seed']
        
        # Wandb settings
        self.use_wandb = True
        self.wandb_project = config['project_name']
        self.wandb_run_name = f"{config['project_name']}-{stage_name}"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


class Trainer:
    """
    Trainer class for model training and evaluation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Union[Dataset, HFDataset, IterableDataset],
        eval_dataset: Optional[Union[Dataset, HFDataset, IterableDataset]] = None,
        args: TrainingArguments = None,
        data_collator: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            args: Training arguments
            data_collator: Function to collate data samples into batches
            compute_metrics_fn: Function to compute evaluation metrics
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.data_collator = data_collator if data_collator is not None else self._default_collator
        self.compute_metrics_fn = compute_metrics_fn
        
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Move model to device
        self.model.to(args.device)
        
        # Enable gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if args.mixed_precision == 'fp16' or args.mixed_precision == 'bf16' else None
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if args.early_stopping_mode == 'min' else float('-inf')
        self.no_improvement_count = 0
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            wandb.config.update(vars(args))
    
    def _default_collator(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Default data collator that simply collates examples and converts to tensors.
        
        Args:
            examples: List of examples to collate
            
        Returns:
            Batch dictionary with tensors
        """
        # Extract keys from the first example
        keys = examples[0].keys()
        
        # Initialize batch dictionary
        batch = {}
        
        # Collate each key
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # Pad sequences to the same length
                values = [example[key] for example in examples]
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
                batch[key] = torch.tensor(padded_values)
            else:
                # For other keys, just collect values
                batch[key] = [example[key] for example in examples]
        
        return batch
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            Configured optimizer
        """
        # Prepare optimizer parameters
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer based on name
        if self.args.optimizer_name.lower() == 'adamw':
            if self.args.use_8bit_optimizer:
                try:
                    from bitsandbytes.optim import AdamW8bit
                    optimizer = AdamW8bit(
                        optimizer_grouped_parameters,
                        lr=self.args.learning_rate,
                        betas=(self.args.beta1, self.args.beta2),
                        eps=self.args.epsilon
                    )
                    logger.info("Using 8-bit AdamW optimizer")
                except ImportError:
                    logger.warning("bitsandbytes not installed, falling back to regular AdamW")
                    optimizer = optim.AdamW(
                        optimizer_grouped_parameters,
                        lr=self.args.learning_rate,
                        betas=(self.args.beta1, self.args.beta2),
                        eps=self.args.epsilon
                    )
            else:
                optimizer = optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.beta1, self.args.beta2),
                    eps=self.args.epsilon
                )
        elif self.args.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.epsilon
            )
        elif self.args.optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer_name}")
        
        return optimizer
    
    def _create_lr_scheduler(self) -> Any:
        """
        Create learning rate scheduler based on configuration.
        
        Returns:
            Configured learning rate scheduler
        """
        # Calculate total training steps
        if isinstance(self.train_dataset, IterableDataset):
            # For streaming datasets, estimate steps per epoch
            steps_per_epoch = 10000  # Arbitrary large number
        else:
            # For regular datasets, calculate steps per epoch
            steps_per_epoch = len(self.train_dataset) // (self.args.train_batch_size * self.args.gradient_accumulation_steps)
        
        total_training_steps = steps_per_epoch * self.args.epochs
        
        # Create scheduler based on schedule name
        if self.args.lr_schedule.lower() == 'linear':
            scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_training_steps
            )
        elif self.args.lr_schedule.lower() == 'cosine':
            scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_training_steps
            )
        elif self.args.lr_schedule.lower() == 'constant':
            scheduler = get_scheduler(
                "constant",
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_training_steps
            )
        else:
            raise ValueError(f"Unsupported learning rate schedule: {self.args.lr_schedule}")
        
        return scheduler
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for the model.
        
        Args:
            inputs: Input dictionary
            
        Returns:
            Prepared inputs
        """
        # Move tensors to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
        
        return inputs
    
    def _compute_loss(self, model_outputs: Dict, inputs: Dict) -> torch.Tensor:
        """
        Compute loss from model outputs and inputs.
        
        Args:
            model_outputs: Model outputs
            inputs: Model inputs
            
        Returns:
            Loss tensor
        """
        # If model returns loss directly, use it
        if "loss" in model_outputs and model_outputs["loss"] is not None:
            return model_outputs["loss"]
        
        # Otherwise, compute loss manually
        logits = model_outputs["logits"]
        labels = inputs["labels"]
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Dictionary with training metrics
        """
        # Create data loader
        train_dataloader = self._get_train_dataloader()
        
        # Set model to training mode
        self.model.train()
        
        # Initialize training metrics
        total_loss = 0.0
        epoch_loss = 0.0
        step_loss = 0.0
        step_count = 0
        start_time = time.time()
        
        # Training loop
        logger.info(f"Starting training for {self.args.epochs} epochs")
        
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            epoch_loss = 0.0
            step_count = 0
            
            # Iterate over batches
            for step, inputs in enumerate(train_dataloader):
                # Prepare inputs
                inputs = self._prepare_inputs(inputs)
                
                # Forward pass with mixed precision if enabled
                if self.args.mixed_precision in ['fp16', 'bf16']:
                    with autocast(dtype=torch.float16 if self.args.mixed_precision == 'fp16' else torch.bfloat16):
                        outputs = self.model(**inputs)
                        loss = self._compute_loss(outputs, inputs)
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                else:
                    # Regular forward and backward pass
                    outputs = self.model(**inputs)
                    loss = self._compute_loss(outputs, inputs)
                    loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                
                # Update metrics
                step_loss += loss.item() * self.args.gradient_accumulation_steps
                epoch_loss += loss.item() * self.args.gradient_accumulation_steps
                total_loss += loss.item() * self.args.gradient_accumulation_steps
                
                # Update parameters after accumulating gradients
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.args.gradient_clipping > 0:
                        if self.args.mixed_precision in ['fp16', 'bf16']:
                            self.scaler.unscale_(self.optimizer)
                        
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.gradient_clipping
                        )
                    
                    # Update parameters
                    if self.args.mixed_precision in ['fp16', 'bf16']:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Update learning rate
                    self.lr_scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Increment global step
                    self.global_step += 1
                    step_count += 1
                    
                    # Log metrics
                    if self.global_step % self.args.logging_steps == 0:
                        # Calculate metrics
                        avg_loss = step_loss / self.args.logging_steps
                        lr = self.optimizer.param_groups[0]['lr']
                        elapsed = time.time() - start_time
                        
                        # Log to console
                        logger.info(
                            f"Epoch: {epoch+1}/{self.args.epochs} | "
                            f"Step: {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.6f} | "
                            f"Time: {elapsed:.2f}s"
                        )
                        
                        # Log to wandb
                        if self.args.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/epoch": epoch + step / len(train_dataloader),
                                "train/global_step": self.global_step
                            })
                        
                        # Reset step loss
                        step_loss = 0.0
                        start_time = time.time()
                    
                    # Evaluate model
                    if self.args.eval_steps > 0 and self.global_step % self.args.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        # Log evaluation metrics
                        logger.info(f"Evaluation metrics: {eval_metrics}")
                        
                        # Log to wandb
                        if self.args.use_wandb:
                            wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                        
                        # Check for early stopping
                        if self.args.early_stopping_enabled:
                            metric_value = eval_metrics.get(
                                self.args.early_stopping_metric,
                                eval_metrics.get("eval_loss", float('inf'))
                            )
                            
                            if self._is_better_metric(metric_value):
                                # Save best model
                                self._save_checkpoint("best")
                                self.best_metric = metric_value
                                self.no_improvement_count = 0
                            else:
                                self.no_improvement_count += 1
                                logger.info(
                                    f"No improvement in {self.args.early_stopping_metric} for "
                                    f"{self.no_improvement_count} evaluations"
                                )
                                
                                if self.no_improvement_count >= self.args.early_stopping_patience:
                                    logger.info(
                                        f"Early stopping triggered after {self.no_improvement_count} "
                                        f"evaluations without improvement"
                                    )
                                    return {
                                        "train_loss": total_loss / self.global_step,
                                        "epoch": epoch + step / len(train_dataloader),
                                        "global_step": self.global_step
                                    }
                        
                        # Set model back to training mode
                        self.model.train()
                    
                    # Save checkpoint
                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"step-{self.global_step}")
            
            # End of epoch
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / step_count if step_count > 0 else 0.0
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch metrics
            logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} completed | "
                f"Loss: {avg_epoch_loss:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch": epoch + 1,
                    "train/epoch_time": epoch_time
                })
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch-{epoch+1}")
            
            # Evaluate at the end of each epoch
            eval_metrics = self.evaluate()
            
            # Log evaluation metrics
            logger.info(f"End of epoch evaluation metrics: {eval_metrics}")
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
            
            # Check for early stopping
            if self.args.early_stopping_enabled:
                metric_value = eval_metrics.get(
                    self.args.early_stopping_metric,
                    eval_metrics.get("eval_loss", float('inf'))
                )
                
                if self._is_better_metric(metric_value):
                    # Save best model
                    self._save_checkpoint("best")
                    self.best_metric = metric_value
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    logger.info(
                        f"No improvement in {self.args.early_stopping_metric} for "
                        f"{self.no_improvement_count} evaluations"
                    )
                    
                    if self.no_improvement_count >= self.args.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {self.no_improvement_count} "
                            f"evaluations without improvement"
                        )
                        break
        
        # End of training
        # Save final checkpoint
        self._save_checkpoint("final")
        
        # Calculate final metrics
        avg_loss = total_loss / self.global_step if self.global_step > 0 else 0.0
        
        # Log final metrics
        logger.info(
            f"Training completed | "
            f"Loss: {avg_loss:.4f} | "
            f"Steps: {self.global_step} | "
            f"Epochs: {self.args.epochs}"
        )
        
        # Return training metrics
        return {
            "train_loss": avg_loss,
            "epoch": self.args.epochs,
            "global_step": self.global_step
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if evaluation dataset is available
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided")
            return {}
        
        # Create data loader
        eval_dataloader = self._get_eval_dataloader()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize evaluation metrics
        eval_loss = 0.0
        eval_steps = 0
        all_preds = []
        all_labels = []
        
        # Evaluation loop
        logger.info("Starting evaluation")
        
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # Prepare inputs
                inputs = self._prepare_inputs(inputs)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = self._compute_loss(outputs, inputs)
                
                # Update metrics
                eval_loss += loss.item()
                eval_steps += 1
                
                # Collect predictions and labels for metrics
                if self.compute_metrics_fn is not None:
                    logits = outputs["logits"]
                    labels = inputs["labels"]
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=-1)
                    
                    # Collect predictions and labels
                    all_preds.append(preds.detach().cpu())
                    all_labels.append(labels.detach().cpu())
        
        # Calculate average loss
        avg_loss = eval_loss / eval_steps if eval_steps > 0 else 0.0
        
        # Initialize metrics dictionary
        metrics = {"eval_loss": avg_loss}
        
        # Compute additional metrics if function is provided
        if self.compute_metrics_fn is not None and all_preds and all_labels:
            # Concatenate predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute metrics
            additional_metrics = self.compute_metrics_fn(all_preds, all_labels)
            
            # Update metrics dictionary
            metrics.update(additional_metrics)
        
        return metrics
    
    def _get_train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for training.
        
        Returns:
            Training DataLoader
        """
        if isinstance(self.train_dataset, IterableDataset):
            # For streaming datasets, don't shuffle
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=4,
                pin_memory=True
            )
        else:
            # For regular datasets, shuffle
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                num_workers=4,
                pin_memory=True
            )
    
    def _get_eval_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for evaluation.
        
        Returns:
            Evaluation DataLoader
        """
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True
        )
    
    def _is_better_metric(self, metric_value: float) -> bool:
        """
        Check if the current metric is better than the best metric.
        
        Args:
            metric_value: Current metric value
            
        Returns:
            True if the current metric is better, False otherwise
        """
        if self.args.early_stopping_mode == 'min':
            return metric_value < self.best_metric
        else:
            return metric_value > self.best_metric
    
    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer and scheduler state if requested
        if self.args.save_optimizer_state:
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        
        # Save training arguments
        with open(os.path.join(checkpoint_dir, "training_args.yaml"), 'w') as f:
            yaml.dump(vars(self.args), f)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "no_improvement_count": self.no_improvement_count
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.yaml"), 'w') as f:
            yaml.dump(training_state, f)
        
        logger.info(f"Saved checkpoint: {checkpoint_name}")
        
        # Clean up old checkpoints if needed
        if self.args.keep_last_n > 0:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """
        Clean up old checkpoints to save space.
        """
        # Get all checkpoint directories
        checkpoint_dirs = []
        
        for name in os.listdir(self.args.output_dir):
            path = os.path.join(self.args.output_dir, name)
            if os.path.isdir(path) and name.startswith("step-"):
                checkpoint_dirs.append((name, path))
        
        # Sort by step number
        checkpoint_dirs.sort(key=lambda x: int(x[0].split("-")[1]))
        
        # Remove old checkpoints
        if len(checkpoint_dirs) > self.args.keep_last_n:
            for name, path in checkpoint_dirs[:-self.args.keep_last_n]:
                logger.info(f"Removing old checkpoint: {name}")
                shutil.rmtree(path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
        """
        # Load model
        self.model.from_pretrained(checkpoint_path)
        self.model.to(self.args.device)
        
        # Load optimizer and scheduler state if available
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        
        if os.path.exists(optimizer_path) and os.path.exists(scheduler_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path))
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.yaml")
        
        if os.path.exists(training_state_path):
            with open(training_state_path, 'r') as f:
                training_state = yaml.safe_load(f)
            
            self.global_step = training_state.get("global_step", 0)
            self.epoch = training_state.get("epoch", 0)
            self.best_metric = training_state.get("best_metric", float('inf') if self.args.early_stopping_mode == 'min' else float('-inf'))
            self.no_improvement_count = training_state.get("no_improvement_count", 0)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from global step {self.global_step}, epoch {self.epoch}")


# Example usage
if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model_from_config(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create dummy datasets
    class DummyDataset(Dataset):
        def __init__(self, size=1000, seq_len=128):
            self.size = size
            self.seq_len = seq_len
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Create random input IDs and labels
            input_ids = torch.randint(0, 50257, (self.seq_len,)).tolist()
            attention_mask = [1] * self.seq_len
            labels = input_ids.copy()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
    
    train_dataset = DummyDataset(size=1000)
    eval_dataset = DummyDataset(size=100)
    
    # Create training arguments
    args = TrainingArguments(config, "pretraining")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args
    )
    
    # Train model
    trainer.train()