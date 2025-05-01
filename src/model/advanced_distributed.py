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
Advanced distributed training capabilities for large language models.

This module implements:
- FSDP (Fully Sharded Data Parallel) for memory-efficient training
- Hybrid parallelism (combining tensor, pipeline, and data parallelism)
- Advanced mixed precision training with the right casting policies
- Optimized GPU communication patterns for maximum throughput
"""

import os
import sys
import logging
import time
import math
import functools
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

# Import optional dependencies
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP, 
        MixedPrecision,
        StateDictType, 
        FullStateDictConfig,
        ShardedStateDictConfig
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    # Create stub classes for type checking
    class FSDP: pass
    class MixedPrecision: pass

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    import torch._dynamo
    import torch._inductor
    TORCH_COMPILE_AVAILABLE = True
except ImportError:
    TORCH_COMPILE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class FSDPConfig:
    """
    Configuration for Fully Sharded Data Parallel (FSDP) training.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        use_mixed_precision: bool = True,
        precision: str = "fp16",  # "fp16", "bf16", "tf32"
        sharding_strategy: str = "full",  # "full", "hybrid", "shard_grad_op"
        cpu_offload: bool = False,  # Whether to offload parameters to CPU
        auto_wrap_policy: str = "transformer",  # "transformer", "size_based", "custom"
        min_num_params: int = 1e6,  # Minimum number of parameters for auto-wrap with size-based policy
        # For transformer policy
        transformer_layer_cls_names: List[str] = None,  # Layer class names for transformer policy
        # Advanced settings
        backward_prefetch: str = "backward_pre",  # "backward_pre", "backward_post"
        forward_prefetch: bool = False,  # Whether to prefetch next module's parameters
        activation_checkpointing: bool = False,  # Whether to use activation checkpointing
        # Optimization settings
        limit_all_gathers: bool = False,  # Whether to limit all-gathers
        use_orig_params: bool = True,  # Whether to use original parameters
        sync_module_states: bool = False,  # Whether to synchronize module states
        state_dict_type: str = "full",  # "full", "sharded", "local"
    ):
        """
        Initialize FSDP configuration.
        
        Args:
            enabled: Whether to enable FSDP
            use_mixed_precision: Whether to use mixed precision
            precision: Precision to use for training
            sharding_strategy: FSDP sharding strategy
            cpu_offload: Whether to offload parameters to CPU
            auto_wrap_policy: Policy for auto-wrapping modules
            min_num_params: Minimum number of parameters for auto-wrap with size-based policy
            transformer_layer_cls_names: Layer class names for transformer policy
            backward_prefetch: When to prefetch parameters for backward pass
            forward_prefetch: Whether to prefetch next module's parameters
            activation_checkpointing: Whether to use activation checkpointing
            limit_all_gathers: Whether to limit all-gathers
            use_orig_params: Whether to use original parameters
            sync_module_states: Whether to synchronize module states
            state_dict_type: Type of state dict to use
        """
        self.enabled = enabled
        self.use_mixed_precision = use_mixed_precision
        self.precision = precision
        self.sharding_strategy = sharding_strategy
        self.cpu_offload = cpu_offload
        self.auto_wrap_policy = auto_wrap_policy
        self.min_num_params = min_num_params
        self.transformer_layer_cls_names = transformer_layer_cls_names or [
            "TransformerBlock", "TransformerLayer", "DecoderLayer", "EncoderLayer", 
            "DecoderBlock", "EncoderBlock", "GPTBlock", "LLaMABlock"
        ]
        self.backward_prefetch = backward_prefetch
        self.forward_prefetch = forward_prefetch
        self.activation_checkpointing = activation_checkpointing
        self.limit_all_gathers = limit_all_gathers
        self.use_orig_params = use_orig_params
        self.sync_module_states = sync_module_states
        self.state_dict_type = state_dict_type


def setup_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl"
) -> None:
    """
    Initialize the distributed environment with NCCL backend.
    
    Args:
        rank: Rank of the current process
        world_size: Number of processes
        master_addr: Address of the master node
        master_port: Port of the master node
        backend: Distributed backend
    """
    # Set environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(rank)
    
    logger.info(f"Initialized process {rank}/{world_size} using backend {backend}")


def get_sharded_optimizer(
    model: Union[nn.Module, FSDP],
    optimizer_class: type,
    optimizer_kwargs: Dict
) -> Optimizer:
    """
    Create an optimizer compatible with FSDP.
    
    Args:
        model: Model (possibly wrapped with FSDP)
        optimizer_class: Optimizer class
        optimizer_kwargs: Optimizer arguments
        
    Returns:
        Optimizer
    """
    if not FSDP_AVAILABLE:
        raise ImportError("FSDP is not available. Please upgrade to PyTorch 1.12+")
    
    # Check if model is already wrapped with FSDP
    if isinstance(model, FSDP):
        # Special handling for Adam/AdamW to avoid GPU memory overhead
        optimizer_name = optimizer_class.__name__.lower()
        if 'adam' in optimizer_name:
            # Use FSDP's flattened parameters
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        else:
            # For other optimizers, use regular initialization
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    else:
        # Regular optimizer for non-FSDP models
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    return optimizer


def get_mixed_precision_config(
    precision: str = "fp16",
    cpu_offload: bool = False
) -> Optional[MixedPrecision]:
    """
    Get mixed precision configuration for FSDP.
    
    Args:
        precision: Precision to use for training
        cpu_offload: Whether to offload parameters to CPU
        
    Returns:
        Mixed precision configuration or None
    """
    if not FSDP_AVAILABLE:
        return None
    
    # Determine compute, param, buffer, and reduce dtypes
    if precision == "fp16":
        compute_dtype = torch.float16
        param_dtype = torch.float16
        reduce_dtype = torch.float16
    elif precision == "bf16":
        # BF16 has better precision for training
        compute_dtype = torch.bfloat16
        param_dtype = torch.bfloat16
        reduce_dtype = torch.bfloat16
    elif precision == "tf32":
        # TF32 = compute in fp32, store in fp32, but internally use TF32 math on Ampere+ GPUs
        compute_dtype = torch.float32
        param_dtype = torch.float32
        reduce_dtype = torch.float32
        # Enable TF32 for cuBLAS and cuDNN operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        compute_dtype = torch.float32
        param_dtype = torch.float32
        reduce_dtype = torch.float32
    
    # Handle CPU offload case
    if cpu_offload:
        # Store parameters as fp32 to avoid precision loss during CPU/GPU transfers
        param_dtype = torch.float32
    
    # Create mixed precision config
    mp_config = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=param_dtype,
        cast_forward_inputs=True
    )
    
    return mp_config


def get_auto_wrap_policy(
    config: FSDPConfig,
    model_class: Optional[nn.Module] = None
) -> Optional[Callable]:
    """
    Get auto wrap policy for FSDP.
    
    Args:
        config: FSDP configuration
        model_class: Class of the model for custom policy
        
    Returns:
        Auto-wrap policy function or None
    """
    if not FSDP_AVAILABLE:
        return None
    
    if config.auto_wrap_policy == "transformer":
        # Import potential layer classes from common transformer libraries
        try:
            import transformers
            if hasattr(transformers, "PreTrainedModel"):
                # Add common class names from transformers
                if "GPTBlock" not in config.transformer_layer_cls_names:
                    config.transformer_layer_cls_names.extend([
                        "GPT2Block", "LlamaDecoderLayer", "T5Block", "BertLayer"
                    ])
        except ImportError:
            pass
        
        # Create transformer auto-wrap policy
        policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=tuple(
                get_module_class_from_name(class_name) 
                for class_name in config.transformer_layer_cls_names
                if get_module_class_from_name(class_name) is not None
            )
        )
        return policy
    
    elif config.auto_wrap_policy == "size_based":
        # Create size-based auto-wrap policy
        policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params
        )
        return policy
    
    elif config.auto_wrap_policy == "custom" and model_class is not None:
        # Create custom auto-wrap policy based on model architecture
        def custom_auto_wrap_policy(module, recurse, unwrapped_params, **kwargs):
            # Implement custom wrapping logic here
            return False
        
        return custom_auto_wrap_policy
    
    # Default: no auto-wrap policy
    return None


def get_module_class_from_name(class_name: str) -> Optional[type]:
    """
    Get module class from name, searching different modules.
    
    Args:
        class_name: Name of the class
        
    Returns:
        Class or None if not found
    """
    search_modules = [
        torch.nn,
        sys.modules.get('src.model.architecture', None),
        sys.modules.get('transformers.models.llama.modeling_llama', None),
        sys.modules.get('transformers.models.gpt2.modeling_gpt2', None),
    ]
    
    for module in search_modules:
        if module is not None and hasattr(module, class_name):
            return getattr(module, class_name)
    
    return None


def wrap_model_with_fsdp(
    model: nn.Module,
    config: FSDPConfig,
    model_class: Optional[nn.Module] = None
) -> nn.Module:
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: Model to wrap
        config: FSDP configuration
        model_class: Class of the model for custom auto-wrap policy
        
    Returns:
        Wrapped model
    """
    if not FSDP_AVAILABLE:
        logger.warning("FSDP is not available. Returning unwrapped model.")
        return model
    
    if not config.enabled:
        return model
    
    # Move model to device
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model.to(device)
    
    # Get mixed precision config
    mp_config = get_mixed_precision_config(config.precision, config.cpu_offload)
    
    # Get auto-wrap policy
    auto_wrap_policy = get_auto_wrap_policy(config, model_class)
    
    # Get sharding strategy
    from torch.distributed.fsdp import ShardingStrategy
    
    if config.sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif config.sharding_strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif config.sharding_strategy == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # Get CPU offload config
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
    
    cpu_offload = CPUOffload(offload_params=config.cpu_offload)
    
    # Get backward prefetch setting
    from torch.distributed.fsdp import BackwardPrefetch
    
    if config.backward_prefetch == "backward_pre":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif config.backward_prefetch == "backward_post":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    else:
        backward_prefetch = None
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_config if config.use_mixed_precision else None,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        forward_prefetch=config.forward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
        use_orig_params=config.use_orig_params,
        sync_module_states=config.sync_module_states
    )
    
    # Apply activation checkpointing if requested
    if config.activation_checkpointing:
        apply_activation_checkpointing(fsdp_model, config)
    
    logger.info(f"Wrapped model with FSDP using {config.sharding_strategy} sharding strategy")
    
    return fsdp_model


def apply_activation_checkpointing(
    model: nn.Module,
    config: FSDPConfig
) -> None:
    """
    Apply activation checkpointing to model.
    
    Args:
        model: Model to apply activation checkpointing to
        config: FSDP configuration
    """
    if not FSDP_AVAILABLE:
        return
    
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        apply_activation_checkpointing as apply_ac,
        CheckpointImpl
    )
    
    # Find the appropriate layer classes
    layer_classes = []
    for class_name in config.transformer_layer_cls_names:
        cls = get_module_class_from_name(class_name)
        if cls is not None:
            layer_classes.append(cls)
    
    if not layer_classes:
        logger.warning("No layer classes found for activation checkpointing")
        return
    
    # Apply activation checkpointing
    check_fn = lambda submodule: isinstance(submodule, tuple(layer_classes))
    
    apply_ac(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn
    )
    
    logger.info("Applied activation checkpointing to model")


def save_fsdp_model(
    model: Union[nn.Module, FSDP],
    optimizer: Optional[Optimizer],
    save_dir: str,
    epoch: int,
    step: int,
    config: FSDPConfig
) -> None:
    """
    Save model and optimizer state with FSDP.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        save_dir: Directory to save to
        epoch: Current epoch
        step: Current step
        config: FSDP configuration
    """
    if not FSDP_AVAILABLE or not isinstance(model, FSDP):
        # Regular saving for non-FSDP models
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None
        }, os.path.join(save_dir, f'checkpoint_{epoch}_{step}.pt'))
        return
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get rank to handle rank0-only full state dict saving
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Get appropriate state dict, based on config
    if config.state_dict_type == "full":
        # Full checkpoint (rank 0 only)
        full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config=full_config):
            # Get model state dict
            model_state = model.state_dict()
            
            # Save model if rank 0
            if rank == 0:
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model_state
                }, os.path.join(save_dir, f'full_checkpoint_{epoch}_{step}.pt'))
    
    elif config.state_dict_type == "sharded":
        # Sharded checkpoint
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            # Get model state dict
            model_state = model.state_dict()
            
            # Save model for all ranks
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model_state
            }, os.path.join(save_dir, f'sharded_checkpoint_{epoch}_{step}_rank{rank}.pt'))
    
    else:  # "local"
        # Local checkpoint - each rank saves its own part
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            # Get model state dict
            model_state = model.state_dict()
            
            # Save model for all ranks
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model_state
            }, os.path.join(save_dir, f'local_checkpoint_{epoch}_{step}_rank{rank}.pt'))
    
    # Optionally save optimizer state if needed
    # This is more complex with FSDP and may require specific handling
    if rank == 0:
        logger.info(f"Saved model checkpoint for epoch {epoch}, step {step}")
    
    # Synchronize to ensure all ranks have saved
    if dist.is_initialized():
        dist.barrier()


def load_fsdp_model(
    model: Union[nn.Module, FSDP],
    load_dir: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    config: FSDPConfig = None
) -> Tuple[Union[nn.Module, FSDP], int, int]:
    """
    Load model state with FSDP.
    
    Args:
        model: Model to load state into
        load_dir: Directory to load from
        epoch: Epoch to load (if None, loads latest)
        step: Step to load (if None, loads latest)
        config: FSDP configuration
        
    Returns:
        Tuple of (model, epoch, step)
    """
    if not FSDP_AVAILABLE or not isinstance(model, FSDP):
        # Regular loading for non-FSDP models
        if epoch is None or step is None:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(load_dir) if f.startswith('checkpoint_')]
            if not checkpoints:
                logger.warning(f"No checkpoints found in {load_dir}")
                return model, 0, 0
            
            checkpoints.sort(key=lambda x: [int(i) for i in x.split('_')[1:3]])
            checkpoint_file = checkpoints[-1]
            epoch, step = map(int, checkpoint_file.split('_')[1:3])
        else:
            checkpoint_file = f'checkpoint_{epoch}_{step}.pt'
        
        # Load checkpoint
        checkpoint = torch.load(os.path.join(load_dir, checkpoint_file), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded checkpoint for epoch {epoch}, step {step}")
        return model, epoch, step
    
    # Get rank for FSDP loading
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Handle different state dict types for FSDP
    if config is None:
        # Default to sharded state dict if no config provided
        config = FSDPConfig(state_dict_type="sharded")
    
    if config.state_dict_type == "full":
        # Full checkpoint (rank 0 loads and broadcasts)
        full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        if rank == 0:
            # Find appropriate checkpoint file
            if epoch is None or step is None:
                # Find latest checkpoint
                checkpoints = [f for f in os.listdir(load_dir) if f.startswith('full_checkpoint_')]
                if not checkpoints:
                    logger.warning(f"No full checkpoints found in {load_dir}")
                    return model, 0, 0
                
                checkpoints.sort(key=lambda x: [int(i) for i in x.split('_')[1:3]])
                checkpoint_file = checkpoints[-1]
                epoch, step = map(int, checkpoint_file.split('_')[1:3])
            else:
                checkpoint_file = f'full_checkpoint_{epoch}_{step}.pt'
            
            # Load checkpoint
            checkpoint = torch.load(os.path.join(load_dir, checkpoint_file), map_location='cpu')
            state_dict = checkpoint['model_state_dict']
        
            # Broadcast epoch and step
            if dist.is_initialized():
                epoch_tensor = torch.tensor([epoch], dtype=torch.int64, device='cuda')
                step_tensor = torch.tensor([step], dtype=torch.int64, device='cuda')
                dist.broadcast(epoch_tensor, src=0)
                dist.broadcast(step_tensor, src=0)
        else:
            # Other ranks receive epoch and step
            state_dict = None
            if dist.is_initialized():
                epoch_tensor = torch.tensor([0], dtype=torch.int64, device='cuda')
                step_tensor = torch.tensor([0], dtype=torch.int64, device='cuda')
                dist.broadcast(epoch_tensor, src=0)
                dist.broadcast(step_tensor, src=0)
                epoch, step = epoch_tensor.item(), step_tensor.item()
        
        # Load state dict
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config=full_config):
            model.load_state_dict(state_dict)
    
    elif config.state_dict_type == "sharded":
        # Sharded checkpoint
        # Find appropriate checkpoint file for this rank
        if epoch is None or step is None:
            # Find latest checkpoint for this rank
            checkpoints = [f for f in os.listdir(load_dir) if f.startswith(f'sharded_checkpoint_') and f.endswith(f'_rank{rank}.pt')]
            if not checkpoints:
                logger.warning(f"No sharded checkpoints found for rank {rank} in {load_dir}")
                return model, 0, 0
            
            checkpoints.sort(key=lambda x: [int(i) for i in x.split('_')[1:3]])
            checkpoint_file = checkpoints[-1]
            epoch, step = map(int, checkpoint_file.split('_')[1:3])
        else:
            checkpoint_file = f'sharded_checkpoint_{epoch}_{step}_rank{rank}.pt'
        
        # Load checkpoint
        checkpoint = torch.load(os.path.join(load_dir, checkpoint_file), map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Load state dict
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(state_dict)
    
    else:  # "local"
        # Local checkpoint - each rank loads its own part
        # Find appropriate checkpoint file for this rank
        if epoch is None or step is None:
            # Find latest checkpoint for this rank
            checkpoints = [f for f in os.listdir(load_dir) if f.startswith(f'local_checkpoint_') and f.endswith(f'_rank{rank}.pt')]
            if not checkpoints:
                logger.warning(f"No local checkpoints found for rank {rank} in {load_dir}")
                return model, 0, 0
            
            checkpoints.sort(key=lambda x: [int(i) for i in x.split('_')[1:3]])
            checkpoint_file = checkpoints[-1]
            epoch, step = map(int, checkpoint_file.split('_')[1:3])
        else:
            checkpoint_file = f'local_checkpoint_{epoch}_{step}_rank{rank}.pt'
        
        # Load checkpoint
        checkpoint = torch.load(os.path.join(load_dir, checkpoint_file), map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Load state dict
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            model.load_state_dict(state_dict)
    
    if rank == 0:
        logger.info(f"Loaded model checkpoint for epoch {epoch}, step {step}")
    
    # Synchronize to ensure all ranks have loaded
    if dist.is_initialized():
        dist.barrier()
    
    return model, epoch, step


def train_with_fsdp(
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: Dict,
    fsdp_config: FSDPConfig,
    output_dir: str,
    **kwargs
) -> Dict:
    """
    Train model with FSDP.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration
        fsdp_config: FSDP configuration
        output_dir: Output directory
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with training results
    """
    # Check if FSDP is available
    if not FSDP_AVAILABLE:
        raise ImportError("FSDP is not available. Please install PyTorch 1.12+")
    
    # Initialize distributed environment if not already initialized
    if not dist.is_initialized():
        if "LOCAL_RANK" not in os.environ:
            raise ValueError("LOCAL_RANK environment variable must be set for FSDP training")
        
        # Get rank and world size
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        # Initialize distributed environment
        setup_distributed(local_rank, world_size)
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Wrap model with FSDP
    fsdp_model = wrap_model_with_fsdp(model, fsdp_config)
    
    # Create optimizer
    optimizer_name = config.get("optimizer", {}).get("name", "adamw").lower()
    if optimizer_name == "adamw":
        from torch.optim import AdamW
        optimizer_class = AdamW
    elif optimizer_name == "adam":
        from torch.optim import Adam
        optimizer_class = Adam
    elif optimizer_name == "sgd":
        from torch.optim import SGD
        optimizer_class = SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Get optimizer kwargs
    optimizer_kwargs = {
        "lr": config.get("learning_rate", 5e-5),
        "weight_decay": config.get("weight_decay", 0.01)
    }
    
    if optimizer_name in ["adam", "adamw"]:
        optimizer_kwargs.update({
            "betas": (config.get("beta1", 0.9), config.get("beta2", 0.999)),
            "eps": config.get("epsilon", 1e-8)
        })
    
    # Create sharded optimizer
    optimizer = get_sharded_optimizer(fsdp_model, optimizer_class, optimizer_kwargs)
    
    # Create learning rate scheduler
    lr_scheduler_name = config.get("lr_scheduler", {}).get("name", "linear").lower()
    
    if lr_scheduler_name == "linear":
        from torch.optim.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.get("min_lr_ratio", 0.1),
            total_iters=config.get("num_epochs", 3) * len(train_dataset) // (config.get("batch_size", 8) * world_size)
        )
    elif lr_scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get("num_epochs", 3) * len(train_dataset) // (config.get("batch_size", 8) * world_size),
            eta_min=config.get("min_lr", 1e-6)
        )
    elif lr_scheduler_name == "constant":
        from torch.optim.lr_scheduler import ConstantLR
        lr_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=config.get("num_epochs", 3) * len(train_dataset) // (config.get("batch_size", 8) * world_size)
        )
    else:
        from torch.optim.lr_scheduler import LinearLR
        lr_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.get("min_lr_ratio", 0.1),
            total_iters=config.get("num_epochs", 3) * len(train_dataset) // (config.get("batch_size", 8) * world_size)
        )
    
    # Create data loaders
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 8),
        sampler=train_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True
    )
    
    eval_loader = None
    if eval_dataset is not None:
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.get("eval_batch_size", 8),
            sampler=eval_sampler,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )
    
    # Set up mixed precision
    use_mixed_precision = config.get("mixed_precision", False)
    scaler = None
    if use_mixed_precision and fsdp_config.precision == "fp16":
        scaler = GradScaler()
    
    # Training loop
    num_epochs = config.get("num_epochs", 3)
    
    # Get gradient accumulation steps
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    # Get other training parameters
    max_grad_norm = config.get("max_grad_norm", 1.0)
    logging_steps = config.get("logging_steps", 10)
    save_steps = config.get("save_steps", 500)
    eval_steps = config.get("eval_steps", 500)
    
    # Initialize step counters
    global_step = 0
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    
    # Initialize metrics
    train_loss = 0.0
    step_loss = 0.0
    epoch_loss = 0.0
    best_eval_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        if eval_loader is not None:
            eval_sampler.set_epoch(epoch)
        
        # Reset epoch metrics
        epoch_loss = 0.0
        step_count = 0
        
        # Create progress bar for master process
        if rank == 0:
            progress_bar = None
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
            except ImportError:
                pass
        
        # Set model to train mode
        fsdp_model.train()
        
        # Iterate over batches
        for step, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(torch.cuda.current_device()) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = [t.to(torch.cuda.current_device()) if isinstance(t, torch.Tensor) else t for t in batch]
            
            # Forward pass with mixed precision if needed
            if use_mixed_precision and fsdp_config.precision == "fp16":
                with autocast():
                    if isinstance(batch, dict):
                        outputs = fsdp_model(**batch)
                    else:
                        outputs = fsdp_model(*batch)
                    
                    # Get loss
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss = outputs["loss"]
                    elif isinstance(outputs, torch.Tensor):
                        loss = outputs
                    else:
                        loss = outputs[0]
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                # Regular forward pass
                if isinstance(batch, dict):
                    outputs = fsdp_model(**batch)
                else:
                    outputs = fsdp_model(*batch)
                
                # Get loss
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                elif isinstance(outputs, torch.Tensor):
                    loss = outputs
                else:
                    loss = outputs[0]
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            # Update metrics
            step_loss += loss.item() * gradient_accumulation_steps
            epoch_loss += loss.item() * gradient_accumulation_steps
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Update parameters after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                # Clip gradients
                if max_grad_norm > 0:
                    if use_mixed_precision and fsdp_config.precision == "fp16":
                        scaler.unscale_(optimizer)
                    
                    if use_mixed_precision and fsdp_config.precision == "fp16":
                        torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_grad_norm)
                
                # Update parameters
                if use_mixed_precision and fsdp_config.precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate
                lr_scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Increment global step
                global_step += 1
                step_count += 1
                
                # Log metrics
                if global_step % logging_steps == 0 and rank == 0:
                    avg_loss = step_loss / logging_steps
                    lr = optimizer.param_groups[0]['lr']
                    
                    logger.info(
                        f"Epoch: {epoch+1}/{num_epochs} | "
                        f"Step: {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.6f}"
                    )
                    
                    step_loss = 0.0
                
                # Update progress bar
                if rank == 0 and progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps, "lr": optimizer.param_groups[0]['lr']})
                
                # Evaluate model
                if eval_loader is not None and global_step % eval_steps == 0:
                    eval_loss = evaluate_with_fsdp(fsdp_model, eval_loader, use_mixed_precision, fsdp_config.precision)
                    
                    if rank == 0:
                        logger.info(f"Evaluation at step {global_step}: loss = {eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_fsdp_model(
                                fsdp_model,
                                optimizer,
                                os.path.join(output_dir, "best_model"),
                                epoch,
                                global_step,
                                fsdp_config
                            )
                    
                    # Set model back to train mode
                    fsdp_model.train()
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    save_fsdp_model(
                        fsdp_model,
                        optimizer,
                        os.path.join(output_dir, "checkpoints"),
                        epoch,
                        global_step,
                        fsdp_config
                    )
        
        # End of epoch processing
        if rank == 0:
            # Close progress bar
            if progress_bar is not None:
                progress_bar.close()
            
            # Log epoch metrics
            avg_epoch_loss = epoch_loss / step_count if step_count > 0 else 0
            logger.info(f"Epoch {epoch+1} completed: avg loss = {avg_epoch_loss:.4f}")
        
        # Evaluate at the end of each epoch
        if eval_loader is not None:
            eval_loss = evaluate_with_fsdp(fsdp_model, eval_loader, use_mixed_precision, fsdp_config.precision)
            
            if rank == 0:
                logger.info(f"Evaluation at epoch {epoch+1}: loss = {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_fsdp_model(
                        fsdp_model,
                        optimizer,
                        os.path.join(output_dir, "best_model"),
                        epoch,
                        global_step,
                        fsdp_config
                    )
        
        # Save checkpoint at the end of each epoch
        save_fsdp_model(
            fsdp_model,
            optimizer,
            os.path.join(output_dir, "checkpoints"),
            epoch,
            global_step,
            fsdp_config
        )
    
    # End of training
    if rank == 0:
        logger.info(f"Training completed: {num_epochs} epochs, {global_step} steps")
        logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    
    # Save final model
    save_fsdp_model(
        fsdp_model,
        optimizer,
        output_dir,
        num_epochs - 1,
        global_step,
        fsdp_config
    )
    
    # Return training results
    return {
        "loss": train_loss / global_step if global_step > 0 else 0,
        "epochs": num_epochs,
        "steps": global_step,
        "best_eval_loss": best_eval_loss
    }


def evaluate_with_fsdp(
    model: nn.Module,
    eval_loader: DataLoader,
    use_mixed_precision: bool = False,
    precision: str = "fp16"
) -> float:
    """
    Evaluate model with FSDP.
    
    Args:
        model: Model to evaluate
        eval_loader: Evaluation data loader
        use_mixed_precision: Whether to use mixed precision
        precision: Precision to use
        
    Returns:
        Evaluation loss
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize evaluation metrics
    total_loss = 0.0
    num_batches = 0
    
    # Evaluate
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(torch.cuda.current_device()) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = [t.to(torch.cuda.current_device()) if isinstance(t, torch.Tensor) else t for t in batch]
            
            # Forward pass with mixed precision if needed
            if use_mixed_precision and precision == "fp16":
                with autocast():
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                    else:
                        outputs = model(*batch)
            else:
                # Regular forward pass
                if isinstance(batch, dict):
                    outputs = model(**batch)
                else:
                    outputs = model(*batch)
            
            # Get loss
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = outputs[0]
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Aggregate loss across all processes
    if dist.is_initialized():
        world_size = dist.get_world_size()
        loss_tensor = torch.tensor([avg_loss], device=torch.cuda.current_device())
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
    
    return avg_loss
