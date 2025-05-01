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
#!/usr/bin/env python


"""
Main script for training AI models using the enhanced training pipeline.

This script supports all advanced features:
- Modern architecture configurations (ALiBi, GQA, RMSNorm, etc.)
- Parameter-Efficient Fine-Tuning (LoRA, QLoRA, Adapters)
- Advanced distributed training (DeepSpeed, FSDP)
- Inference optimization for deployment
"""

import os
import sys
import logging
import argparse
import yaml
import torch
from typing import Dict, Optional, Union, List, Any

from src.data.loaders import DatasetLoader
from src.data.preprocessors import DataPreprocessor
from src.data.augmentation import TextAugmenter, augment_dataset
from src.model.architecture import create_model_from_config
from src.model.training import Trainer, TrainingArguments
from src.model.distributed_training import train_distributed, DeepSpeedConfig, DistributedTrainer
from src.model.advanced_distributed import train_with_fsdp, FSDPConfig, setup_distributed
from src.model.peft import apply_peft_to_model, prepare_for_qlora, LoRAConfig, AdapterConfig, PrefixTuningConfig
from src.model.inference_optimization import ContinuousBatchingServer, SpeculativeDecoder, KVCacheManager
from src.utils.tokenization import get_tokenizer
from src.utils.hyperparameter_tuning import optimize_hyperparameters
from src.utils.model_evaluation import evaluate_model
from src.model.optimization import optimize_model

# Try to import optional dependencies
try:
    from huggingface_hub import login, HfApi, create_repo
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    
try:
    import torch._dynamo
    TORCH_COMPILE_AVAILABLE = True
except ImportError:
    TORCH_COMPILE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AI models using the enhanced training pipeline.")
    
    # Basic configuration arguments
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--stage", type=str, default=None,
        help="Training stage to run (overrides config)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)"
    )
    
    # Distributed training arguments
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--strategy", type=str, default=None, choices=["deepspeed", "fsdp", "ddp"],
        help="Distributed training strategy (overrides config)"
    )
    parser.add_argument(
        "--deepspeed", action="store_true", 
        help="Enable DeepSpeed"
    )
    parser.add_argument(
        "--fsdp", action="store_true",
        help="Enable Fully Sharded Data Parallel (FSDP)"
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", type=str, default="full", choices=["full", "hybrid", "shard_grad_op"],
        help="FSDP sharding strategy"
    )
    parser.add_argument(
        "--fsdp_cpu_offload", action="store_true",
        help="Enable CPU offloading with FSDP"
    )
    parser.add_argument(
        "--fsdp_auto_wrap", type=str, default="transformer", choices=["transformer", "size_based"],
        help="FSDP auto-wrap policy"
    )
    parser.add_argument(
        "--zero_stage", type=int, default=None, choices=[0, 1, 2, 3],
        help="DeepSpeed ZeRO stage (overrides config)"
    )
    parser.add_argument(
        "--offload_optimizer", action="store_true",
        help="Enable CPU offloading of optimizer states with DeepSpeed"
    )
    parser.add_argument(
        "--offload_parameters", action="store_true",
        help="Enable CPU offloading of parameters with DeepSpeed"
    )
    
    # Mixed precision and optimization arguments
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16", action="store_true",
        help="Enable BF16 mixed precision training"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Enable gradient checkpointing for memory efficiency"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Enable PyTorch 2.0+ compilation (torch.compile)"
    )
    parser.add_argument(
        "--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"],
        help="PyTorch compilation mode"
    )
    
    # PEFT arguments
    parser.add_argument(
        "--peft", action="store_true",
        help="Enable Parameter-Efficient Fine-Tuning (PEFT)"
    )
    parser.add_argument(
        "--peft_method", type=str, default=None, choices=["lora", "qlora", "adapter", "prefix_tuning"],
        help="PEFT method to use (overrides config)"
    )
    parser.add_argument(
        "--lora_r", type=int, default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32,
        help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--lora_target_modules", type=str, default=None,
        help="Comma-separated list of target modules for LoRA (e.g., 'q_proj,k_proj,v_proj')"
    )
    parser.add_argument(
        "--qlora_bits", type=int, default=4, choices=[4, 8],
        help="Quantization bits for QLoRA"
    )
    parser.add_argument(
        "--adapter_dim", type=int, default=64,
        help="Bottleneck dimension for adapter tuning"
    )
    parser.add_argument(
        "--prefix_length", type=int, default=20,
        help="Prefix length for prefix tuning"
    )
    parser.add_argument(
        "--merge_lora_weights", action="store_true",
        help="Merge LoRA weights into base model at the end of training"
    )
    
    # Evaluation and hyperparameter tuning arguments
    parser.add_argument(
        "--hyperparameter_tuning", action="store_true",
        help="Run hyperparameter tuning"
    )
    parser.add_argument(
        "--evaluate_only", action="store_true",
        help="Only evaluate the model, don't train"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Evaluation batch size (overrides config)"
    )
    
    # Model optimization arguments
    parser.add_argument(
        "--optimize_model", action="store_true",
        help="Optimize the model after training"
    )
    parser.add_argument(
        "--optimization_type", type=str, default="dynamic_quantization",
        choices=[
            "dynamic_quantization", "static_quantization", "int8", "int4", 
            "pruning", "onnx", "gptq", "awq", "int8_float16_mixed"
        ],
        help="Type of model optimization to apply"
    )
    
    # Inference optimization arguments
    parser.add_argument(
        "--inference_mode", action="store_true",
        help="Optimize the model for inference after training"
    )
    parser.add_argument(
        "--continuous_batching", action="store_true",
        help="Enable continuous batching for efficient inference"
    )
    parser.add_argument(
        "--kv_cache_optimization", action="store_true",
        help="Enable KV cache optimization for inference"
    )
    parser.add_argument(
        "--speculative_decoding", action="store_true",
        help="Enable speculative decoding for faster inference"
    )
    parser.add_argument(
        "--draft_model_path", type=str, default=None,
        help="Path to draft model for speculative decoding"
    )
    
    # HuggingFace Hub integration
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace API token for accessing gated datasets and model uploads"
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id", type=str, default=None,
        help="Model ID for HuggingFace Hub (format: 'username/model_name')"
    )
    parser.add_argument(
        "--hub_private", action="store_true",
        help="Make the uploaded model private on HuggingFace Hub"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def update_config_with_args(config: Dict, args) -> Dict:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy of the config
    import copy
    updated_config = copy.deepcopy(config)
    
    # Basic configuration updates
    if args.stage:
        updated_config['training']['active_stage'] = args.stage
    
    if args.output_dir:
        updated_config['output_dir'] = args.output_dir
    
    if args.batch_size:
        updated_config['data_processing']['batching']['train_batch_size'] = args.batch_size
        updated_config['data_processing']['batching']['eval_batch_size'] = args.batch_size
    
    if args.eval_batch_size:
        updated_config['data_processing']['batching']['eval_batch_size'] = args.eval_batch_size
    
    if args.epochs:
        # Update epochs for the active stage
        active_stage = updated_config['training']['active_stage']
        for stage in updated_config['training']['stages']:
            if stage['name'] == active_stage:
                stage['epochs'] = args.epochs
    
    if args.learning_rate:
        # Update learning rate for the active stage
        active_stage = updated_config['training']['active_stage']
        for stage in updated_config['training']['stages']:
            if stage['name'] == active_stage:
                stage['learning_rate']['initial'] = args.learning_rate
    
    if args.seed:
        updated_config['seed'] = args.seed
    
    # Distributed training updates
    if args.distributed or args.deepspeed or args.fsdp:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        updated_config['distributed_training']['use_multi_gpu'] = True
    
    if args.strategy:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        updated_config['distributed_training']['strategy'] = args.strategy
    
    if args.deepspeed:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        if 'deepspeed' not in updated_config['distributed_training']:
            updated_config['distributed_training']['deepspeed'] = {}
        updated_config['distributed_training']['strategy'] = 'deepspeed'
        updated_config['distributed_training']['deepspeed']['enabled'] = True
    
    if args.zero_stage is not None:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        if 'deepspeed' not in updated_config['distributed_training']:
            updated_config['distributed_training']['deepspeed'] = {}
        updated_config['distributed_training']['deepspeed']['zero_stage'] = args.zero_stage
    
    if args.offload_optimizer:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        if 'deepspeed' not in updated_config['distributed_training']:
            updated_config['distributed_training']['deepspeed'] = {}
        updated_config['distributed_training']['deepspeed']['offload_optimizer'] = True
    
    if args.offload_parameters:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        if 'deepspeed' not in updated_config['distributed_training']:
            updated_config['distributed_training']['deepspeed'] = {}
        updated_config['distributed_training']['deepspeed']['offload_param'] = True
    
    if args.fsdp:
        if 'distributed_training' not in updated_config:
            updated_config['distributed_training'] = {}
        if 'fsdp' not in updated_config['distributed_training']:
            updated_config['distributed_training']['fsdp'] = {}
        updated_config['distributed_training']['strategy'] = 'fsdp'
        updated_config['distributed_training']['fsdp']['enabled'] = True
        updated_config['distributed_training']['fsdp']['sharding_strategy'] = args.fsdp_sharding_strategy
        updated_config['distributed_training']['fsdp']['cpu_offload'] = args.fsdp_cpu_offload
        updated_config['distributed_training']['fsdp']['auto_wrap_policy'] = args.fsdp_auto_wrap
    
    # Mixed precision and optimization updates
    if args.fp16:
        updated_config['training']['mixed_precision'] = "fp16"
    
    if args.bf16:
        updated_config['training']['mixed_precision'] = "bf16"
    
    if args.gradient_checkpointing:
        updated_config['model']['gradient_checkpointing'] = True
    
    if args.compile:
        if 'model' not in updated_config:
            updated_config['model'] = {}
        if 'compile' not in updated_config['model']:
            updated_config['model']['compile'] = {}
        updated_config['model']['compile']['enabled'] = True
        updated_config['model']['compile']['mode'] = args.compile_mode
    
    # PEFT updates
    if args.peft:
        if 'peft' not in updated_config:
            updated_config['peft'] = {}
        updated_config['peft']['enabled'] = True
        
        if args.peft_method:
            updated_config['peft']['method'] = args.peft_method
            
            # Configure LoRA/QLoRA parameters
            if args.peft_method in ['lora', 'qlora']:
                method_key = 'lora' if args.peft_method == 'lora' else 'qlora'
                if method_key not in updated_config['peft']:
                    updated_config['peft'][method_key] = {}
                
                updated_config['peft'][method_key]['r'] = args.lora_r
                updated_config['peft'][method_key]['alpha'] = args.lora_alpha
                updated_config['peft'][method_key]['dropout'] = args.lora_dropout
                
                if args.lora_target_modules:
                    updated_config['peft'][method_key]['target_modules'] = args.lora_target_modules.split(',')
                
                # QLoRA specific settings
                if args.peft_method == 'qlora':
                    updated_config['peft']['qlora']['bits'] = args.qlora_bits
            
            # Configure adapter parameters
            elif args.peft_method == 'adapter':
                if 'adapter' not in updated_config['peft']:
                    updated_config['peft']['adapter'] = {}
                updated_config['peft']['adapter']['dim'] = args.adapter_dim
            
            # Configure prefix tuning parameters
            elif args.peft_method == 'prefix_tuning':
                if 'prefix_tuning' not in updated_config['peft']:
                    updated_config['peft']['prefix_tuning'] = {}
                updated_config['peft']['prefix_tuning']['prefix_length'] = args.prefix_length
            
            # Add PEFT method to active training stage
            active_stage = updated_config['training']['active_stage']
            for stage in updated_config['training']['stages']:
                if stage['name'] == active_stage:
                    stage['peft_method'] = args.peft_method
        
        # Merge LORA weights option
        if args.merge_lora_weights:
            if 'training' not in updated_config:
                updated_config['training'] = {}
            if 'checkpointing' not in updated_config['training']:
                updated_config['training']['checkpointing'] = {}
            updated_config['training']['checkpointing']['merge_lora_weights'] = True
    
    # Hyperparameter tuning updates
    if args.hyperparameter_tuning:
        if 'hyperparameter_optimization' not in updated_config:
            updated_config['hyperparameter_optimization'] = {}
        updated_config['hyperparameter_optimization']['enabled'] = True
    
    # Model optimization updates
    if args.optimize_model:
        if 'model_optimization' not in updated_config:
            updated_config['model_optimization'] = {}
        updated_config['model_optimization']['enabled'] = True
        updated_config['model_optimization']['method'] = args.optimization_type
    
    # Inference optimization updates
    if args.inference_mode or args.continuous_batching or args.kv_cache_optimization or args.speculative_decoding:
        if 'inference_optimization' not in updated_config:
            updated_config['inference_optimization'] = {}
        updated_config['inference_optimization']['enabled'] = True
    
    if args.continuous_batching:
        if 'inference_optimization' not in updated_config:
            updated_config['inference_optimization'] = {}
        if 'continuous_batching' not in updated_config['inference_optimization']:
            updated_config['inference_optimization']['continuous_batching'] = {}
        updated_config['inference_optimization']['continuous_batching']['enabled'] = True
    
    if args.kv_cache_optimization:
        if 'inference_optimization' not in updated_config:
            updated_config['inference_optimization'] = {}
        if 'kv_cache' not in updated_config['inference_optimization']:
            updated_config['inference_optimization']['kv_cache'] = {}
        updated_config['inference_optimization']['kv_cache']['enabled'] = True
    
    if args.speculative_decoding:
        if 'inference_optimization' not in updated_config:
            updated_config['inference_optimization'] = {}
        if 'speculative_decoding' not in updated_config['inference_optimization']:
            updated_config['inference_optimization']['speculative_decoding'] = {}
        updated_config['inference_optimization']['speculative_decoding']['enabled'] = True
        
        if args.draft_model_path:
            updated_config['inference_optimization']['speculative_decoding']['draft_model'] = args.draft_model_path
    
    # HuggingFace Hub integration updates
    if args.push_to_hub:
        if 'huggingface_hub' not in updated_config:
            updated_config['huggingface_hub'] = {}
        updated_config['huggingface_hub']['push_to_hub'] = True
        
        if args.hub_model_id:
            updated_config['huggingface_hub']['model_id'] = args.hub_model_id
        
        if args.hub_private:
            updated_config['huggingface_hub']['private'] = True
    
    return updated_config


def main():
    """Main function for training AI models with advanced features."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    import numpy as np
    np.random.seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save updated configuration
    config_path = os.path.join(config['output_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Log into HuggingFace if token is provided
    if args.hf_token and HUGGINGFACE_HUB_AVAILABLE:
        logger.info("Logging into HuggingFace Hub")
        login(token=args.hf_token)
    
    # Get active stage
    active_stage = config['training']['active_stage']
    logger.info(f"Active training stage: {active_stage}")
    
    # Get stage configuration
    stage_config = None
    for stage in config['training']['stages']:
        if stage['name'] == active_stage:
            stage_config = stage
            break
    
    if stage_config is None:
        raise ValueError(f"Training stage {active_stage} not found in configuration")
    
    # Initialize dataset loader with HuggingFace token if provided
    dataset_loader = DatasetLoader(args.config, huggingface_token=args.hf_token)
    
    # Load datasets
    train_datasets = []
    for dataset_config in stage_config['datasets']:
        if dataset_config.get('split') == 'train':
            dataset = dataset_loader.load_dataset(
                dataset_config['name'],
                subset=dataset_config.get('subset'),
                split='train',
                streaming=dataset_config.get('streaming', False),
                max_samples=dataset_config.get('max_samples')
            )
            train_datasets.append(dataset)
    
    eval_datasets = []
    for dataset_config in stage_config['datasets']:
        if dataset_config.get('split') in ['validation', 'test']:
            dataset = dataset_loader.load_dataset(
                dataset_config['name'],
                subset=dataset_config.get('subset'),
                split=dataset_config.get('split'),
                streaming=dataset_config.get('streaming', False),
                max_samples=dataset_config.get('max_samples')
            )
            eval_datasets.append(dataset)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Preprocess datasets
    train_datasets = [preprocessor.process_dataset(ds) for ds in train_datasets]
    eval_datasets = [preprocessor.process_dataset(ds) for ds in eval_datasets]
    
    # Apply data augmentation if enabled
    if config.get('data_processing', {}).get('augmentation', {}).get('enabled', False):
        # Extract augmentation techniques and probabilities
        techniques = [t['name'] for t in config['data_processing']['augmentation']['techniques']]
        probabilities = [t['probability'] for t in config['data_processing']['augmentation']['techniques']]
        
        # Augment training datasets
        for i, dataset in enumerate(train_datasets):
            train_datasets[i] = augment_dataset(
                dataset=dataset,
                text_column='text' if 'text' in dataset.column_names else 'input',
                techniques=techniques,
                probabilities=probabilities,
                n_aug=1,
                keep_original=True
            )
    
    # Get tokenizer
    tokenizer = get_tokenizer(config['tokenizer'])
    
    # Run hyperparameter tuning if enabled
    if config.get('hyperparameter_optimization', {}).get('enabled', False):
        logger.info("Running hyperparameter tuning")
        
        # Get hyperparameter optimization configuration
        hpo_config = config['hyperparameter_optimization']
        
        # Run hyperparameter optimization
        best_params = optimize_hyperparameters(
            config_path=args.config,
            search_space=None,  # Use default search space from config
            num_samples=hpo_config['num_samples'],
            num_epochs=hpo_config['num_epochs'],
            search_alg=hpo_config['search_algorithm'],
            scheduler=hpo_config['scheduler'],
            metric=hpo_config['metric'],
            mode=hpo_config['mode']
        )
        
        logger.info(f"Best hyperparameters: {best_params['best_config']}")
        
        # Load updated configuration
        with open(os.path.join(config['output_dir'], 'best_config', 'best_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
    
    # Create model
    logger.info("Creating model")
    model = create_model_from_config(config)
    
    # Apply PEFT methods if enabled
    if config.get('peft', {}).get('enabled', False):
        peft_config = config['peft']
        peft_method = peft_config['method']
        logger.info(f"Applying PEFT method: {peft_method}")
        
        # Prepare model for QLoRA if needed
        if peft_method == 'qlora':
            logger.info(f"Preparing model for QLoRA with {peft_config['qlora']['bits']}-bit quantization")
            model = prepare_for_qlora(
                model,
                bits=peft_config['qlora']['bits'],
                groupsize=peft_config['qlora'].get('bnb_blocksize', 128),
                compute_dtype=peft_config['qlora'].get('bnb_compute_dtype', 'float16'),
                double_quant=peft_config['qlora'].get('double_quant', True)
            )
        
        # Apply PEFT method to model
        if peft_method in ['lora', 'qlora', 'adapter', 'prefix_tuning']:
            # Get the correct config for the PEFT method
            method_config = peft_config[peft_method]
            
            # Convert to appropriate objects for apply_peft_to_model
            if peft_method in ['lora', 'qlora']:
                target_modules = method_config.get('target_modules', None)
                
                logger.info(f"Applying {peft_method.upper()} with rank={method_config['r']}, " 
                            f"alpha={method_config['alpha']}, target_modules={target_modules}")
                
                model = apply_peft_to_model(
                    model=model,
                    peft_config=method_config,
                    peft_type=peft_method,
                    target_modules=target_modules,
                    adapter_name="default"
                )
            elif peft_method == 'adapter':
                logger.info(f"Applying Adapter tuning with bottleneck dim={method_config['dim']}")
                model = apply_peft_to_model(
                    model=model,
                    peft_config=method_config,
                    peft_type='adapter',
                    adapter_name="default"
                )
            elif peft_method == 'prefix_tuning':
                logger.info(f"Applying Prefix Tuning with prefix length={method_config['prefix_length']}")
                model = apply_peft_to_model(
                    model=model,
                    peft_config=method_config,
                    peft_type='prefix_tuning',
                    adapter_name="default"
                )
    
    # Apply torch.compile if enabled and available
    if config.get('model', {}).get('compile', {}).get('enabled', False) and TORCH_COMPILE_AVAILABLE:
        compile_mode = config['model']['compile'].get('mode', 'default')
        logger.info(f"Applying torch.compile with mode: {compile_mode}")
        model = torch.compile(model, mode=compile_mode)
    elif config.get('model', {}).get('compile', {}).get('enabled', False) and not TORCH_COMPILE_AVAILABLE:
        logger.warning("torch.compile is not available. Skipping model compilation.")
    
    # Evaluate only if requested
    if args.evaluate_only:
        logger.info("Evaluating model")
        
        # Get evaluation configuration
        eval_config = config['evaluation']
        
        # Evaluate model
        metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_datasets[0] if eval_datasets else None,
            task_type="text_generation",
            metrics=eval_config['metrics'],
            batch_size=config['data_processing']['batching']['eval_batch_size'],
            max_length=eval_config['generation']['max_length'],
            num_beams=eval_config['generation']['num_beams'],
            temperature=eval_config['generation']['temperature'],
            top_p=eval_config['generation']['top_p'],
            top_k=eval_config['generation']['top_k'],
            do_sample=eval_config['generation']['do_sample'],
            output_dir=os.path.join(config['output_dir'], 'evaluation'),
            visualize=True
        )
        
        logger.info(f"Evaluation metrics: {metrics}")
        return
    
    # Get training strategy from config or arguments
    strategy = config.get('distributed_training', {}).get('strategy', 'ddp')
    if args.strategy:
        strategy = args.strategy
    elif args.deepspeed:
        strategy = 'deepspeed'
    elif args.fsdp:
        strategy = 'fsdp'
    
    # Train model with the appropriate strategy
    if args.distributed:
        if strategy == 'fsdp':
            logger.info("Running distributed training with FSDP")
            
            # Get FSDP configuration
            fsdp_config = config['distributed_training'].get('fsdp', {})
            
            # Create FSDP config
            fsdp_config_obj = FSDPConfig(
                enabled=True,
                sharding_strategy=fsdp_config.get('sharding_strategy', 'full'),
                mixed_precision=config['training'].get('mixed_precision', 'fp16'),
                cpu_offload=fsdp_config.get('cpu_offload', False),
                auto_wrap_policy=fsdp_config.get('auto_wrap_policy', 'transformer'),
                transformer_layer_cls_names=fsdp_config.get('transformer_layer_cls_names'),
                backward_prefetch=fsdp_config.get('backward_prefetch', 'backward_pre'),
                forward_prefetch=fsdp_config.get('forward_prefetch', False),
                activation_checkpointing=fsdp_config.get('activation_checkpointing', False),
                state_dict_type=fsdp_config.get('state_dict_type', 'full')
            )
            
            # Run training with FSDP
            results = train_with_fsdp(
                model=model,
                train_dataset=train_datasets[0] if train_datasets else None,
                eval_dataset=eval_datasets[0] if eval_datasets else None,
                config=config,
                fsdp_config=fsdp_config_obj,
                output_dir=config['output_dir']
            )
        elif strategy == 'deepspeed':
            logger.info("Running distributed training with DeepSpeed")
            
            # Get DeepSpeed configuration
            ds_config = config['distributed_training'].get('deepspeed', {})
            
            # Create DeepSpeed configuration
            deepspeed_config = DeepSpeedConfig(
                zero_stage=ds_config.get('zero_stage', 2),
                offload_optimizer=ds_config.get('offload_optimizer', False),
                offload_param=ds_config.get('offload_param', False),
                fp16=config['training'].get('mixed_precision', '') == 'fp16',
                bf16=config['training'].get('mixed_precision', '') == 'bf16',
                gradient_accumulation_steps=config['data_processing']['batching'].get('gradient_accumulation_steps', 1),
                gradient_clipping=config['training'].get('gradient_clipping', 1.0),
                output_dir=config['output_dir']
            )
            
            # Run distributed training with DeepSpeed
            results = train_distributed(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_datasets[0] if train_datasets else None,
                eval_dataset=eval_datasets[0] if eval_datasets else None,
                config_path=args.config,
                output_dir=config['output_dir'],
                num_epochs=stage_config['epochs'],
                batch_size=config['data_processing']['batching']['train_batch_size'],
                learning_rate=stage_config['learning_rate']['initial'],
                weight_decay=config['training']['optimizer']['weight_decay'],
                warmup_steps=stage_config['learning_rate']['warmup_steps'],
                gradient_accumulation_steps=config['data_processing']['batching']['gradient_accumulation_steps'],
                gradient_clipping=config['training']['gradient_clipping'],
                fp16=config['training']['mixed_precision'] == 'fp16',
                bf16=config['training']['mixed_precision'] == 'bf16',
                zero_stage=ds_config.get('zero_stage', 2),
                offload_optimizer=ds_config.get('offload_optimizer', False),
                offload_param=ds_config.get('offload_param', False),
                use_deepspeed=True,
                local_rank=args.local_rank,
                seed=config['seed'],
                save_steps=config['training']['checkpointing']['save_steps'],
                eval_steps=config['training']['evaluation']['eval_steps'],
                logging_steps=config['logging'].get('log_steps', 10),
                use_wandb=config['logging'].get('use_wandb', False),
                wandb_project=config['logging'].get('wandb_project', ''),
                wandb_run_name=f"{config['project_name']}_{active_stage}"
            )
        else:  # Default to DDP
            logger.info("Running distributed training with PyTorch DDP")
            
            # Create training arguments
            training_args = TrainingArguments(config, active_stage)
            
            # Create distributed trainer
            trainer = DistributedTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_datasets[0] if train_datasets else None,
                eval_dataset=eval_datasets[0] if eval_datasets else None,
                args=training_args,
                local_rank=args.local_rank
            )
            
            # Train model
            results = trainer.train()
    else:
        logger.info("Running single-GPU training")
        
        # Create training arguments
        training_args = TrainingArguments(config, active_stage)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_datasets[0] if train_datasets else None,
            eval_dataset=eval_datasets[0] if eval_datasets else None,
            args=training_args
        )
        
        # Train model
        results = trainer.train()
    
    logger.info(f"Training results: {results}")
    
    # Merge PEFT weights if requested and applicable
    if config.get('peft', {}).get('enabled', False) and config.get('training', {}).get('checkpointing', {}).get('merge_lora_weights', False):
        if hasattr(model, 'merge_peft_weights'):
            logger.info("Merging PEFT weights into base model")
            model.merge_peft_weights()
    
    # Optimize model if requested
    if args.optimize_model:
        logger.info(f"Optimizing model with {args.optimization_type}")
        
        # Get optimization configuration
        opt_config = config.get('model_optimization', {})
        
        # Optimize model
        optimized_model = optimize_model(
            model=model,
            tokenizer=tokenizer,
            optimization_type=args.optimization_type,
            output_dir=os.path.join(config['output_dir'], 'optimized_model'),
            save_model=True,
            benchmark=True
        )
        
        # Use optimized model for subsequent steps
        model = optimized_model
        logger.info(f"Model optimization completed")
    
    # Set up inference optimization if requested
    if args.inference_mode or args.continuous_batching or args.kv_cache_optimization:
        logger.info("Setting up inference optimizations")
        
        inference_dir = os.path.join(config['output_dir'], 'inference_ready')
        os.makedirs(inference_dir, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(inference_dir)
        tokenizer.save_pretrained(inference_dir)
        
        # Create inference configuration
        inference_config = config.get('inference_optimization', {})
        with open(os.path.join(inference_dir, 'inference_config.yaml'), 'w') as f:
            yaml.dump(inference_config, f)
        
        # Test inference speed if continuous batching is enabled
        if args.continuous_batching:
            logger.info("Testing continuous batching inference performance")
            # Create continuous batching server
            server = ContinuousBatchingServer(
                model=model,
                tokenizer=tokenizer,
                max_batch_size=inference_config.get('continuous_batching', {}).get('max_batch_size', 32),
                device=inference_config.get('device', 'cuda'),
                precision=inference_config.get('dtype', 'float16')
            )
            
            # Perform a quick benchmark
            with server:  # Context manager to manage server lifecycle
                # Test a few sample requests
                test_requests = [
                    {"prompt": "Explain the concept of machine learning", "max_new_tokens": 64},
                    {"prompt": "Write a Python function to calculate Fibonacci numbers", "max_new_tokens": 64},
                    {"prompt": "What are the benefits of distributed training?", "max_new_tokens": 64}
                ]
                
                # Time the requests
                start_time = time.time()
                for i, req in enumerate(test_requests):
                    req_id = f"test_{i}"
                    server.add_request(GenerationRequest(
                        request_id=req_id,
                        prompt=req["prompt"],
                        max_new_tokens=req["max_new_tokens"]
                    ))
                
                # Wait for all requests to complete
                while len(server.completed_requests) < len(test_requests):
                    time.sleep(0.1)
                
                elapsed = time.time() - start_time
                
                # Log performance metrics
                stats = server.get_stats()
                logger.info(f"Continuous batching performance:")
                logger.info(f"  Processed {len(test_requests)} requests in {elapsed:.2f}s")
                logger.info(f"  Average batch size: {stats.get('avg_batch_size', 'N/A')}")
                logger.info(f"  Average tokens per second: {sum(len(server.completed_requests[f'test_{i}'].generated_tokens) for i in range(len(test_requests))) / elapsed:.2f}")
        
        # Test speculative decoding if enabled
        if args.speculative_decoding and args.draft_model_path:
            logger.info("Testing speculative decoding performance")
            # Load draft model
            draft_model = torch.load(args.draft_model_path)
            
            # Create speculative decoder
            decoder = SpeculativeDecoder(
                main_model=model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                num_speculative_tokens=inference_config.get('speculative_decoding', {}).get('num_speculative_tokens', 4)
            )
            
            # Test performance
            test_prompt = "Explain the benefits of transformer models in natural language processing."
            
            # Time generation with speculative decoding
            start_time = time.time()
            output_tokens, stats = decoder.generate(
                prompt=test_prompt,
                max_new_tokens=100
            )
            elapsed = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"Speculative decoding performance:")
            logger.info(f"  Generated {len(output_tokens)} tokens in {elapsed:.2f}s")
            logger.info(f"  Tokens per second: {len(output_tokens) / elapsed:.2f}")
            logger.info(f"  Acceptance rate: {stats['acceptance_rate']:.2f}")
            logger.info(f"  Speedup: {stats['speedup']:.2f}x")
    
    # Push model to HuggingFace Hub if requested
    if args.push_to_hub and HUGGINGFACE_HUB_AVAILABLE:
        logger.info("Pushing model to HuggingFace Hub")
        
        # Get HuggingFace Hub configuration
        hub_config = config.get('huggingface_hub', {})
        model_id = hub_config.get('model_id', f"{config['project_name']}-{active_stage}")
        if args.hub_model_id:
            model_id = args.hub_model_id
        
        # Make sure we have a token
        if not args.hf_token:
            logger.warning("No HuggingFace token provided. Skipping model upload.")
        else:
            # Verify repository exists or create it
            api = HfApi()
            try:
                repo_url = api.create_repo(
                    repo_id=model_id,
                    exist_ok=True,
                    private=hub_config.get('private', args.hub_private)
                )
                logger.info(f"Repository URL: {repo_url}")
                
                # Push model to hub
                model.push_to_hub(
                    repo_id=model_id,
                    use_auth_token=args.hf_token,
                    private=hub_config.get('private', args.hub_private)
                )
                
                # Push tokenizer to hub
                tokenizer.push_to_hub(
                    repo_id=model_id,
                    use_auth_token=args.hf_token
                )
                
                # Push configuration as well
                save_path = os.path.join(config['output_dir'], 'hub_ready_config.yaml')
                with open(save_path, 'w') as f:
                    yaml.dump(config, f)
                
                api.upload_file(
                    path_or_fileobj=save_path,
                    path_in_repo="config.yaml",
                    repo_id=model_id,
                    token=args.hf_token
                )
                
                logger.info(f"Successfully pushed model, tokenizer, and config to {model_id}")
            except Exception as e:
                logger.error(f"Error pushing to HuggingFace Hub: {e}")
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()