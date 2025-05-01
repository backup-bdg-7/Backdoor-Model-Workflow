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
Distributed training utilities for the AI model training workflow.
This module provides functions for distributed training using DeepSpeed and PyTorch DDP.
"""

import os
import logging
import json
import yaml
import time
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

# Try to import optional dependencies
try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.model.training import Trainer, TrainingArguments
from src.utils.metrics import compute_metrics

# Configure logging
logger = logging.getLogger(__name__)


class DeepSpeedConfig:
    """
    Configuration for DeepSpeed.
    """
    
    def __init__(
        self,
        zero_stage: int = 2,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        fp16: bool = True,
        bf16: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: float = 1.0,
        optimizer: Dict = None,
        scheduler: Dict = None,
        output_dir: str = "./deepspeed_output",
    ):
        """
        Initialize DeepSpeed configuration.
        
        Args:
            zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
            offload_optimizer: Whether to offload optimizer states to CPU
            offload_param: Whether to offload parameters to CPU
            fp16: Whether to use FP16 mixed precision
            bf16: Whether to use BF16 mixed precision
            gradient_accumulation_steps: Number of gradient accumulation steps
            gradient_clipping: Gradient clipping value
            optimizer: Optimizer configuration
            scheduler: Scheduler configuration
            output_dir: Output directory for DeepSpeed
        """
        self.zero_stage = zero_stage
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param
        self.fp16 = fp16
        self.bf16 = bf16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.optimizer = optimizer or {}
        self.scheduler = scheduler or {}
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """
        Convert configuration to DeepSpeed JSON format.
        
        Returns:
            Dictionary with DeepSpeed configuration
        """
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "sub_group_size": 1e9,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "fp16": {
                "enabled": self.fp16,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.bf16
            },
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
            "zero_allow_untested_optimizer": True
        }
        
        return config
    
    def save_config(self, file_path: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save configuration
            
        Returns:
            Path to saved configuration
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, "ds_config.json")
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return file_path


class DistributedTrainer(Trainer):
    """
    Trainer class for distributed training with DeepSpeed or PyTorch DDP.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Union[Dataset, List[Dataset]],
        eval_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
        args: TrainingArguments = None,
        data_collator: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        deepspeed_config: Optional[Union[Dict, DeepSpeedConfig]] = None,
        use_deepspeed: bool = True,
        local_rank: int = -1,
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset(s)
            eval_dataset: Evaluation dataset(s)
            args: Training arguments
            data_collator: Function to collate data samples into batches
            compute_metrics_fn: Function to compute evaluation metrics
            deepspeed_config: DeepSpeed configuration
            use_deepspeed: Whether to use DeepSpeed
            local_rank: Local rank for distributed training
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            data_collator=data_collator,
            compute_metrics_fn=compute_metrics_fn
        )
        
        # Set up distributed training
        self.local_rank = local_rank if local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.distributed = self.local_rank != -1
        
        # Set up DeepSpeed
        self.use_deepspeed = use_deepspeed and DEEPSPEED_AVAILABLE
        if self.use_deepspeed and not DEEPSPEED_AVAILABLE:
            logger.warning("DeepSpeed is not installed. Falling back to PyTorch DDP.")
            self.use_deepspeed = False
        
        self.deepspeed_config = deepspeed_config
        if self.use_deepspeed and self.deepspeed_config is None:
            self.deepspeed_config = DeepSpeedConfig(
                zero_stage=2,
                fp16=True,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                gradient_clipping=self.args.gradient_clipping
            )
        
        # Initialize distributed environment
        if self.distributed and not dist.is_initialized():
            if self.local_rank == -1:
                raise ValueError(
                    "Distributed training requires LOCAL_RANK environment variable to be set"
                )
            
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            logger.info(f"Initialized distributed training with {self.world_size} GPUs")
        
        # Set up model for distributed training
        if self.distributed:
            if self.use_deepspeed:
                self._setup_deepspeed()
            else:
                self._setup_ddp()
    
    def _setup_deepspeed(self) -> None:
        """
        Set up DeepSpeed for distributed training.
        """
        if not self.use_deepspeed or not DEEPSPEED_AVAILABLE:
            return
        
        # Convert DeepSpeedConfig to dict if needed
        if isinstance(self.deepspeed_config, DeepSpeedConfig):
            ds_config_dict = self.deepspeed_config.to_dict()
            ds_config_path = self.deepspeed_config.save_config()
        else:
            ds_config_dict = self.deepspeed_config
            ds_config_path = os.path.join(self.args.output_dir, "ds_config.json")
            with open(ds_config_path, 'w') as f:
                json.dump(ds_config_dict, f, indent=2)
        
        # Update batch sizes in DeepSpeed config
        ds_config_dict["train_micro_batch_size_per_gpu"] = self.args.train_batch_size
        ds_config_dict["train_batch_size"] = self.args.train_batch_size * self.world_size
        
        # Initialize DeepSpeed
        model, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config_dict,
            dist_init_required=False
        )
        
        self.model = model
        self.optimizer = optimizer
        
        logger.info(f"Initialized DeepSpeed with config: {ds_config_path}")
    
    def _setup_ddp(self) -> None:
        """
        Set up PyTorch DDP for distributed training.
        """
        if not self.distributed:
            return
        
        # Move model to device
        self.model.to(self.args.device)
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )
        
        logger.info("Initialized PyTorch DDP")
    
    def _get_train_dataloader(self) -> DataLoader:
        """
        Get training dataloader with distributed sampler.
        
        Returns:
            Training dataloader
        """
        if not self.distributed:
            return super()._get_train_dataloader()
        
        # Create distributed sampler
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader
    
    def _get_eval_dataloader(self) -> Optional[DataLoader]:
        """
        Get evaluation dataloader with distributed sampler.
        
        Returns:
            Evaluation dataloader
        """
        if self.eval_dataset is None:
            return None
        
        if not self.distributed:
            return super()._get_eval_dataloader()
        
        # Create distributed sampler
        sampler = DistributedSampler(
            self.eval_dataset,
            num_replicas=self.world_size,
            rank=self.local_rank,
            shuffle=False
        )
        
        # Create dataloader
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def train(self) -> Dict[str, float]:
        """
        Train the model with distributed training.
        
        Returns:
            Dictionary with training metrics
        """
        # Set model to training mode
        self.model.train()
        
        # Get dataloader
        train_dataloader = self._get_train_dataloader()
        
        # Set up sampler for distributed training
        if self.distributed:
            train_dataloader.sampler.set_epoch(self.epoch)
        
        # Training loop
        if self.use_deepspeed:
            return self._train_with_deepspeed(train_dataloader)
        else:
            return super().train()
    
    def _train_with_deepspeed(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """
        Train the model with DeepSpeed.
        
        Args:
            train_dataloader: Training dataloader
            
        Returns:
            Dictionary with training metrics
        """
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
            
            # Set epoch for distributed sampler
            if self.distributed:
                train_dataloader.sampler.set_epoch(epoch)
            
            # Iterate over batches
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{self.args.epochs}",
                disable=self.local_rank != 0
            )
            
            for step, inputs in enumerate(progress_bar):
                # Prepare inputs
                inputs = self._prepare_inputs(inputs)
                
                # Forward and backward pass with DeepSpeed
                outputs = self.model(inputs)
                loss = outputs.loss
                
                # Update weights with DeepSpeed
                self.model.backward(loss)
                self.model.step()
                
                # Update metrics
                step_loss = loss.item()
                epoch_loss += step_loss
                total_loss += step_loss
                step_count += 1
                
                # Update progress bar
                if self.local_rank == 0:
                    progress_bar.set_postfix({
                        "loss": step_loss,
                        "avg_loss": epoch_loss / (step + 1),
                        "lr": self.model.get_lr()[0]
                    })
                
                # Evaluate if needed
                if self.args.eval_steps > 0 and step > 0 and step % self.args.eval_steps == 0:
                    eval_results = self.evaluate()
                    self.model.train()
                    
                    # Log evaluation results
                    if self.local_rank == 0:
                        logger.info(f"Evaluation results at step {step}: {eval_results}")
                        
                        # Log to wandb if enabled
                        if self.args.use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                "eval_loss": eval_results.get("eval_loss", 0),
                                "eval_perplexity": eval_results.get("perplexity", 0),
                                "step": step,
                                "epoch": epoch
                            })
                
                # Save checkpoint if needed
                if self.args.save_steps > 0 and step > 0 and step % self.args.save_steps == 0:
                    self.save_checkpoint()
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, loss: {epoch_loss/step_count:.4f}")
            
            # Evaluate at the end of each epoch
            eval_results = self.evaluate()
            self.model.train()
            
            # Log evaluation results
            if self.local_rank == 0:
                logger.info(f"Evaluation results at epoch {epoch+1}: {eval_results}")
                
                # Log to wandb if enabled
                if self.args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "eval_loss": eval_results.get("eval_loss", 0),
                        "eval_perplexity": eval_results.get("perplexity", 0),
                        "epoch": epoch + 1
                    })
            
            # Save checkpoint at the end of each epoch
            self.save_checkpoint()
            
            # Check for early stopping
            if self.args.early_stopping_enabled and self._check_early_stopping(eval_results):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # End of training
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f}s, avg loss: {total_loss/step_count:.4f}")
        
        # Save final model
        self.save_checkpoint(is_final=True)
        
        return {
            "loss": total_loss / step_count,
            "epoch": self.epoch,
            "learning_rate": self.model.get_lr()[0] if self.use_deepspeed else self.optimizer.param_groups[0]['lr'],
            "train_time": train_time
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Get dataloader
        eval_dataloader = self._get_eval_dataloader()
        if eval_dataloader is None:
            return {}
        
        # Initialize evaluation metrics
        total_loss = 0.0
        step_count = 0
        all_preds = []
        all_labels = []
        
        # Evaluation loop
        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # Prepare inputs
                inputs = self._prepare_inputs(inputs)
                
                # Forward pass
                if self.use_deepspeed:
                    outputs = self.model.forward(inputs)
                else:
                    outputs = self.model(**inputs)
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Update metrics
                total_loss += loss.item()
                step_count += 1
                
                # Store predictions and labels for metrics
                if "labels" in inputs:
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.append(preds.detach().cpu())
                    all_labels.append(inputs["labels"].detach().cpu())
        
        # Compute average loss
        avg_loss = total_loss / step_count if step_count > 0 else 0
        
        # Compute perplexity
        perplexity = math.exp(avg_loss)
        
        # Compute additional metrics if available
        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity
        }
        
        # Compute custom metrics if function is provided
        if self.compute_metrics_fn is not None and all_preds and all_labels:
            # Concatenate predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute metrics
            additional_metrics = self.compute_metrics_fn(all_preds, all_labels)
            metrics.update(additional_metrics)
        
        return metrics
    
    def save_checkpoint(self, is_final: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_final: Whether this is the final checkpoint
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"checkpoint-{self.global_step}" if not is_final else "final-model"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save with DeepSpeed if enabled
        if self.use_deepspeed:
            self.model.save_checkpoint(checkpoint_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"Saved DeepSpeed checkpoint to {checkpoint_dir}")
        else:
            # Save model and tokenizer
            if self.distributed:
                # Save model from first rank only
                if self.local_rank == 0:
                    unwrapped_model = self.model.module
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                    
                    logger.info(f"Saved model checkpoint to {checkpoint_dir}")
            else:
                self.model.save_pretrained(checkpoint_dir)
                self.tokenizer.save_pretrained(checkpoint_dir)
                
                logger.info(f"Saved model checkpoint to {checkpoint_dir}")
            
            # Save optimizer state if requested
            if self.args.save_optimizer_state:
                optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                torch.save(self.optimizer.state_dict(), optimizer_path)
                
                logger.info(f"Saved optimizer state to {optimizer_path}")
        
        # Save training arguments
        args_path = os.path.join(checkpoint_dir, "training_args.json")
        with open(args_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        if self.use_deepspeed:
            # Load with DeepSpeed
            self.model.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded DeepSpeed checkpoint from {checkpoint_path}")
        else:
            # Load model weights
            if self.distributed:
                # Load model for all ranks
                self.model.module.load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_path, "pytorch_model.bin"),
                        map_location=f"cuda:{self.local_rank}"
                    )
                )
            else:
                self.model.load_state_dict(
                    torch.load(
                        os.path.join(checkpoint_path, "pytorch_model.bin"),
                        map_location=self.args.device
                    )
                )
            
            logger.info(f"Loaded model checkpoint from {checkpoint_path}")
            
            # Load optimizer state if available
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optimizer_path) and self.args.save_optimizer_state:
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=self.args.device)
                )
                logger.info(f"Loaded optimizer state from {optimizer_path}")


def train_distributed(
    model: nn.Module,
    tokenizer: Any,
    train_dataset: Union[Dataset, List[Dataset]],
    eval_dataset: Optional[Union[Dataset, List[Dataset]]] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
    fp16: bool = True,
    bf16: bool = False,
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    use_deepspeed: bool = True,
    local_rank: int = -1,
    seed: int = 42,
    save_steps: int = 0,
    eval_steps: int = 0,
    logging_steps: int = 10,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train a model with distributed training.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset(s)
        eval_dataset: Evaluation dataset(s)
        config_path: Path to configuration file
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size per GPU
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        gradient_accumulation_steps: Number of gradient accumulation steps
        gradient_clipping: Gradient clipping value
        fp16: Whether to use FP16 mixed precision
        bf16: Whether to use BF16 mixed precision
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_param: Whether to offload parameters to CPU
        use_deepspeed: Whether to use DeepSpeed
        local_rank: Local rank for distributed training
        seed: Random seed
        save_steps: Save checkpoint every X steps (0 to disable)
        eval_steps: Evaluate every X steps (0 to disable)
        logging_steps: Log every X steps
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name
        
    Returns:
        Dictionary with training results
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set up output directory
    if output_dir is None:
        output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration if provided
    config = None
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create training arguments
    if config is not None:
        args = TrainingArguments(config, config['training']['active_stage'])
    else:
        # Create dummy config
        dummy_config = {
            "output_dir": output_dir,
            "seed": seed,
            "project_name": wandb_project or "distributed_training",
            "training": {
                "active_stage": "default",
                "stages": [
                    {
                        "name": "default",
                        "epochs": num_epochs,
                        "datasets": [],
                        "learning_rate": {
                            "initial": learning_rate,
                            "min": 0.0,
                            "schedule": "linear",
                            "warmup_steps": warmup_steps
                        }
                    }
                ],
                "optimizer": {
                    "name": "adamw",
                    "weight_decay": weight_decay,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                    "use_8bit": False
                },
                "mixed_precision": "fp16" if fp16 else ("bf16" if bf16 else "no"),
                "gradient_checkpointing": False,
                "gradient_clipping": gradient_clipping,
                "checkpointing": {
                    "save_steps": save_steps,
                    "keep_last_n": 3,
                    "save_optimizer_state": True
                },
                "evaluation": {
                    "eval_steps": eval_steps,
                    "early_stopping": {
                        "enabled": False,
                        "patience": 3,
                        "metric": "eval_loss",
                        "mode": "min"
                    }
                }
            },
            "data_processing": {
                "batching": {
                    "train_batch_size": batch_size,
                    "eval_batch_size": batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "dynamic_batching": False
                }
            }
        }
        args = TrainingArguments(dummy_config, "default")
    
    # Update arguments
    args.output_dir = output_dir
    args.train_batch_size = batch_size
    args.eval_batch_size = batch_size
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay
    args.warmup_steps = warmup_steps
    args.gradient_accumulation_steps = gradient_accumulation_steps
    args.gradient_clipping = gradient_clipping
    args.epochs = num_epochs
    args.save_steps = save_steps
    args.eval_steps = eval_steps
    args.logging_steps = logging_steps
    args.use_wandb = use_wandb
    args.wandb_project = wandb_project
    args.wandb_run_name = wandb_run_name
    
    # Set up DeepSpeed configuration
    deepspeed_config = DeepSpeedConfig(
        zero_stage=zero_stage,
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=gradient_clipping,
        output_dir=output_dir
    )
    
    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        deepspeed_config=deepspeed_config,
        use_deepspeed=use_deepspeed,
        local_rank=local_rank
    )
    
    # Train model
    results = trainer.train()
    
    # Evaluate final model
    eval_results = trainer.evaluate()
    results.update(eval_results)
    
    return results