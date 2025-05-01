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
Hyperparameter tuning utilities for the AI model training workflow.
This module provides functions for hyperparameter optimization using Ray Tune.
"""

import os
import logging
import yaml
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import torch
from functools import partial

# Ray Tune imports
try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.basic_variant import BasicVariantGenerator
    from ray.tune.utils.util import wait_for_gpu
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from src.model.training import Trainer, TrainingArguments
from src.model.architecture import create_model_from_config
from src.data.loaders import DatasetLoader
from src.utils.tokenization import get_tokenizer

# Configure logging
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    A class to handle hyperparameter optimization for model training.
    """
    
    def __init__(self, config_path: str, search_space: Dict = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config_path: Path to the configuration file
            search_space: Search space for hyperparameter optimization
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.search_space = search_space or self._get_default_search_space()
        
        # Check if Ray is available
        if not RAY_AVAILABLE:
            logger.warning("Ray Tune is not installed. Hyperparameter optimization will not be available.")
            logger.warning("Install Ray Tune with: pip install 'ray[tune]'")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _get_default_search_space(self) -> Dict:
        """
        Get default search space for hyperparameter optimization.
        
        Returns:
            Dictionary containing default search space
        """
        return {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "weight_decay": tune.loguniform(1e-6, 1e-3),
            "batch_size": tune.choice([8, 16, 32, 64]),
            "optimizer": tune.choice(["adam", "adamw"]),
            "scheduler": tune.choice(["linear", "cosine"]),
            "warmup_steps": tune.choice([100, 500, 1000]),
            "dropout": tune.uniform(0.1, 0.5),
        }
    
    def _train_function(self, config: Dict, checkpoint_dir: Optional[str] = None) -> Dict:
        """
        Training function for Ray Tune.
        
        Args:
            config: Hyperparameter configuration
            checkpoint_dir: Directory for checkpoints
            
        Returns:
            Dictionary with training results
        """
        # Update configuration with hyperparameters
        train_config = self._update_config_with_hyperparams(config)
        
        # Create output directory
        output_dir = os.path.join(train_config['output_dir'], f"tune_run_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save updated configuration
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(train_config, f)
        
        # Load datasets
        dataset_loader = DatasetLoader(self.config_path)
        
        # Get stage configuration
        stage_name = train_config['training']['active_stage']
        stage_config = None
        for stage in train_config['training']['stages']:
            if stage['name'] == stage_name:
                stage_config = stage
                break
        
        if stage_config is None:
            raise ValueError(f"Training stage {stage_name} not found in configuration")
        
        # Load datasets
        train_datasets = []
        for dataset_config in stage_config['datasets']:
            if dataset_config['split'] == 'train':
                dataset = dataset_loader.load_dataset(
                    dataset_config['name'],
                    subset=dataset_config.get('subset'),
                    split='train',
                    streaming=dataset_config.get('streaming', False)
                )
                train_datasets.append(dataset)
        
        eval_datasets = []
        for dataset_config in stage_config['datasets']:
            if dataset_config['split'] == 'validation':
                dataset = dataset_loader.load_dataset(
                    dataset_config['name'],
                    subset=dataset_config.get('subset'),
                    split='validation',
                    streaming=dataset_config.get('streaming', False)
                )
                eval_datasets.append(dataset)
        
        # Get tokenizer
        tokenizer = get_tokenizer(train_config['tokenizer'])
        
        # Create model
        model = create_model_from_config(train_config)
        
        # Create training arguments
        training_args = TrainingArguments(train_config, stage_name)
        
        # Update training arguments with hyperparameters
        training_args.learning_rate = config['learning_rate']
        training_args.weight_decay = config['weight_decay']
        training_args.train_batch_size = config['batch_size']
        training_args.optimizer_name = config['optimizer']
        training_args.lr_schedule = config['scheduler']
        training_args.warmup_steps = config['warmup_steps']
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_datasets[0] if train_datasets else None,
            eval_dataset=eval_datasets[0] if eval_datasets else None,
            args=training_args
        )
        
        # Load checkpoint if available
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            trainer.load_checkpoint(checkpoint_path)
        
        # Train model
        result = trainer.train()
        
        # Report metrics to Ray Tune
        metrics = {
            "loss": result["loss"],
            "eval_loss": result.get("eval_loss", float('inf')),
            "perplexity": result.get("perplexity", float('inf')),
            "learning_rate": result["learning_rate"],
            "epoch": result["epoch"]
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, "checkpoint")
        trainer.save_checkpoint(checkpoint_path)
        
        return metrics
    
    def _update_config_with_hyperparams(self, hyperparams: Dict) -> Dict:
        """
        Update configuration with hyperparameters.
        
        Args:
            hyperparams: Hyperparameter configuration
            
        Returns:
            Updated configuration
        """
        # Create a deep copy of the configuration
        config = self._load_config()
        
        # Update optimizer settings
        optimizer_config = config['training']['optimizer']
        optimizer_config['name'] = hyperparams['optimizer']
        optimizer_config['weight_decay'] = hyperparams['weight_decay']
        
        # Update learning rate settings
        stage_name = config['training']['active_stage']
        for stage in config['training']['stages']:
            if stage['name'] == stage_name:
                stage['learning_rate']['initial'] = hyperparams['learning_rate']
                stage['learning_rate']['schedule'] = hyperparams['scheduler']
                stage['learning_rate']['warmup_steps'] = hyperparams['warmup_steps']
        
        # Update batch size
        config['data_processing']['batching']['train_batch_size'] = hyperparams['batch_size']
        
        # Update model dropout
        if 'model' in config and 'dropout' in config['model']:
            config['model']['dropout'] = hyperparams['dropout']
        
        return config
    
    def optimize(
        self,
        num_samples: int = 10,
        num_epochs: int = 3,
        gpus_per_trial: float = 1,
        cpus_per_trial: int = 4,
        search_alg: str = "hyperopt",
        scheduler: str = "asha",
        metric: str = "eval_loss",
        mode: str = "min",
        max_concurrent_trials: Optional[int] = None,
        resume: bool = False,
        local_dir: str = "./ray_results",
        resources_per_trial: Optional[Dict] = None,
    ) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            num_samples: Number of hyperparameter combinations to try
            num_epochs: Number of epochs to train each trial
            gpus_per_trial: Number of GPUs to use per trial
            cpus_per_trial: Number of CPUs to use per trial
            search_alg: Search algorithm to use (hyperopt, bayesopt, optuna, random)
            scheduler: Scheduler to use (asha, pbt)
            metric: Metric to optimize
            mode: Optimization mode (min or max)
            max_concurrent_trials: Maximum number of concurrent trials
            resume: Whether to resume previous optimization
            local_dir: Directory to store results
            resources_per_trial: Resources to use per trial
            
        Returns:
            Dictionary with optimization results
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray Tune is not installed. Install with: pip install 'ray[tune]'")
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Set up search algorithm
        if search_alg == "hyperopt":
            search_algorithm = HyperOptSearch(metric=metric, mode=mode)
        elif search_alg == "bayesopt":
            search_algorithm = BayesOptSearch(metric=metric, mode=mode)
        elif search_alg == "optuna":
            search_algorithm = OptunaSearch(metric=metric, mode=mode)
        elif search_alg == "random":
            search_algorithm = BasicVariantGenerator()
        else:
            raise ValueError(f"Unsupported search algorithm: {search_alg}")
        
        # Limit concurrency if specified
        if max_concurrent_trials:
            search_algorithm = ConcurrencyLimiter(
                search_algorithm, max_concurrent=max_concurrent_trials
            )
        
        # Set up scheduler
        if scheduler == "asha":
            scheduler_algorithm = ASHAScheduler(
                metric=metric,
                mode=mode,
                max_t=num_epochs,
                grace_period=1,
                reduction_factor=2
            )
        elif scheduler == "pbt":
            scheduler_algorithm = PopulationBasedTraining(
                time_attr="training_iteration",
                metric=metric,
                mode=mode,
                perturbation_interval=1,
                hyperparam_mutations={
                    "learning_rate": lambda: np.random.uniform(1e-5, 1e-3),
                    "weight_decay": lambda: np.random.uniform(1e-6, 1e-3),
                }
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        
        # Set up reporter
        reporter = CLIReporter(
            metric_columns=["loss", "eval_loss", "perplexity", "learning_rate", "epoch"],
            parameter_columns=list(self.search_space.keys()),
            max_progress_rows=10,
            max_error_rows=1,
            max_report_frequency=300,  # Report every 5 minutes
        )
        
        # Set up resources per trial
        if resources_per_trial is None:
            resources_per_trial = {
                "cpu": cpus_per_trial,
                "gpu": gpus_per_trial
            }
        
        # Update config to limit epochs for tuning
        config = self._load_config()
        stage_name = config['training']['active_stage']
        for stage in config['training']['stages']:
            if stage['name'] == stage_name:
                stage['epochs'] = num_epochs
        
        # Run optimization
        analysis = tune.run(
            partial(self._train_function),
            config=self.search_space,
            num_samples=num_samples,
            scheduler=scheduler_algorithm,
            search_alg=search_algorithm,
            resources_per_trial=resources_per_trial,
            local_dir=local_dir,
            progress_reporter=reporter,
            resume=resume,
            verbose=2,
            fail_fast=True,
            checkpoint_at_end=True,
            checkpoint_freq=1,
            keep_checkpoints_num=2,
            checkpoint_score_attr=f"{mode}-{metric}",
            name=f"hpo_{int(time.time())}"
        )
        
        # Get best configuration
        best_config = analysis.get_best_config(metric=metric, mode=mode)
        best_result = analysis.get_best_trial(metric=metric, mode=mode).last_result
        
        logger.info(f"Best hyperparameters found: {best_config}")
        logger.info(f"Best result: {best_result}")
        
        # Save best configuration
        output_dir = os.path.join(self.config['output_dir'], "best_config")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'best_hyperparams.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Update original config with best hyperparameters
        best_full_config = self._update_config_with_hyperparams(best_config)
        
        with open(os.path.join(output_dir, 'best_config.yaml'), 'w') as f:
            yaml.dump(best_full_config, f)
        
        return {
            "best_config": best_config,
            "best_result": best_result,
            "all_results": analysis.results,
            "best_checkpoint": analysis.get_best_checkpoint(
                trial=analysis.get_best_trial(metric=metric, mode=mode),
                metric=metric,
                mode=mode
            )
        }


def optimize_hyperparameters(
    config_path: str,
    search_space: Optional[Dict] = None,
    num_samples: int = 10,
    num_epochs: int = 3,
    gpus_per_trial: float = 1,
    cpus_per_trial: int = 4,
    search_alg: str = "hyperopt",
    scheduler: str = "asha",
    metric: str = "eval_loss",
    mode: str = "min",
    max_concurrent_trials: Optional[int] = None,
    resume: bool = False,
    local_dir: str = "./ray_results",
) -> Dict:
    """
    Run hyperparameter optimization.
    
    Args:
        config_path: Path to the configuration file
        search_space: Search space for hyperparameter optimization
        num_samples: Number of hyperparameter combinations to try
        num_epochs: Number of epochs to train each trial
        gpus_per_trial: Number of GPUs to use per trial
        cpus_per_trial: Number of CPUs to use per trial
        search_alg: Search algorithm to use (hyperopt, bayesopt, optuna, random)
        scheduler: Scheduler to use (asha, pbt)
        metric: Metric to optimize
        mode: Optimization mode (min or max)
        max_concurrent_trials: Maximum number of concurrent trials
        resume: Whether to resume previous optimization
        local_dir: Directory to store results
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = HyperparameterOptimizer(config_path, search_space)
    return optimizer.optimize(
        num_samples=num_samples,
        num_epochs=num_epochs,
        gpus_per_trial=gpus_per_trial,
        cpus_per_trial=cpus_per_trial,
        search_alg=search_alg,
        scheduler=scheduler,
        metric=metric,
        mode=mode,
        max_concurrent_trials=max_concurrent_trials,
        resume=resume,
        local_dir=local_dir
    )