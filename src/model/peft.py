"""
Parameter-Efficient Fine-Tuning (PEFT) methods for language models.

This module implements various PEFT techniques including:
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Adapters
- Prefix Tuning

These methods enable efficient fine-tuning of large language models by updating
only a small subset of parameters, dramatically reducing memory requirements.
"""

import os
import math
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Type

# Configure logging
logger = logging.getLogger(__name__)


class LoRAConfig:
    """
    Configuration class for LoRA (Low-Rank Adaptation).
    """
    
    def __init__(
        self,
        r: int = 8,  # Rank of the low-rank matrices
        alpha: float = 16.0,  # Scaling factor
        dropout: float = 0.0,  # Dropout probability for LoRA layers
        target_modules: List[str] = None,  # Modules to apply LoRA to
        bias: str = "none",  # Whether to add bias terms ('none', 'all', 'lora_only')
        use_rslora: bool = False,  # Whether to use rank-stabilized LoRA
        qweight_dtype: Optional[str] = None,  # Dtype for quantized weight ('int8', 'int4', etc.)
        use_8bit_qlora: bool = False,  # Whether to use 8-bit QLoRA
        use_4bit_qlora: bool = False,  # Whether to use 4-bit QLoRA
        qlora_double_quant: bool = False,  # Whether to use double quantization for QLoRA
        bnb_bits: int = 4,  # bitsandbytes quantization bits
        bnb_blocksize: int = 64,  # bitsandbytes block size
        bnb_compute_dtype: str = "float16",  # bitsandbytes compute dtype
    ):
        """
        Initialize LoRA configuration.
        
        Args:
            r: Rank of the low-rank matrices
            alpha: Scaling factor (controls strength of adaptation)
            dropout: Dropout probability for LoRA layers
            target_modules: Modules to apply LoRA to (e.g., ["q_proj", "v_proj"])
            bias: Whether to add bias terms
            use_rslora: Whether to use rank-stabilized LoRA
            qweight_dtype: Data type for quantized weights
            use_8bit_qlora: Whether to use 8-bit QLoRA
            use_4bit_qlora: Whether to use 4-bit QLoRA
            qlora_double_quant: Whether to use double quantization for QLoRA
            bnb_bits: bitsandbytes quantization bits (4 or 8)
            bnb_blocksize: bitsandbytes block size for quantization
            bnb_compute_dtype: Compute data type for bitsandbytes
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.bias = bias
        self.use_rslora = use_rslora
        
        # QLoRA settings
        self.qweight_dtype = qweight_dtype
        self.use_8bit_qlora = use_8bit_qlora
        self.use_4bit_qlora = use_4bit_qlora
        self.qlora_double_quant = qlora_double_quant
        self.bnb_bits = bnb_bits
        self.bnb_blocksize = bnb_blocksize
        self.bnb_compute_dtype = bnb_compute_dtype
        
        # Calculate actual scaling (as used in paper)
        self.scaling = alpha / r
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "LoRAConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig,
        adapter_name: str = "default"
    ):
        """
        Initialize LoRA adapted linear layer.
        
        Args:
            base_layer: Base linear layer to adapt
            config: LoRA configuration
            adapter_name: Name of the adapter (for multiple adaptation support)
        """
        super().__init__()
        
        # Save configuration
        self.config = config
        self.adapter_name = adapter_name
        
        # Save original layer and its shape
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros((config.r, self.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, config.r)))
        
        # Initialize dropout
        self.lora_dropout = nn.Dropout(p=config.dropout)
        
        # Initialize bias if needed
        if config.bias == "all":
            self.lora_bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.lora_bias = None
        
        # Use scaling factor from configuration
        self.scaling = config.scaling
        
        # Initialize weights
        self.reset_parameters()
        
        # Mark base layer as not trainable
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)
        
        # Track all parameters that should be updated
        self.active_adapter = True
    
    def reset_parameters(self):
        """Initialize LoRA weights."""
        if self.config.use_rslora:
            # Use rank-stabilized initialization
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            # Use standard initialization
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        
        # Initialize bias if used
        if self.lora_bias is not None:
            nn.init.zeros_(self.lora_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Get base layer output
        base_output = self.base_layer(x)
        
        # Apply LoRA if active
        if self.active_adapter:
            # Apply dropout to input
            lora_input = self.lora_dropout(x)
            
            # Calculate LoRA contribution: (x @ A^T) @ B
            lora_output = (lora_input @ self.lora_A.T) @ self.lora_B.T
            
            # Scale output
            lora_output = lora_output * self.scaling
            
            # Add bias if needed
            if self.lora_bias is not None:
                lora_output = lora_output + self.lora_bias
            
            # Combine with base output
            return base_output + lora_output
        else:
            return base_output
    
    def merge_lora_weights(self):
        """Merge LoRA weights into base layer weights for inference."""
        if not self.active_adapter:
            return
        
        # Calculate merged weight: W + scaling * B * A
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta_weight
        
        # Merge bias if present
        if self.lora_bias is not None and self.base_layer.bias is not None:
            self.base_layer.bias.data += self.lora_bias
        elif self.lora_bias is not None:
            self.base_layer.bias = nn.Parameter(self.lora_bias.clone())
        
        # Deactivate adapter
        self.active_adapter = False
    
    def unmerge_lora_weights(self):
        """Unmerge LoRA weights from base layer weights."""
        if self.active_adapter:
            return
        
        # Calculate and subtract delta weight
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data -= delta_weight
        
        # Unmerge bias if present
        if self.lora_bias is not None and self.base_layer.bias is not None:
            self.base_layer.bias.data -= self.lora_bias
        elif self.lora_bias is not None:
            self.base_layer.bias = None
        
        # Reactivate adapter
        self.active_adapter = True


class AdapterConfig:
    """
    Configuration class for Adapter tuning.
    """
    
    def __init__(
        self,
        dim: int = 64,  # Bottleneck dimension
        scaling: float = 1.0,  # Scaling factor
        use_gate: bool = True,  # Whether to use gating
        norm_position: str = "pre",  # Position of normalization ('pre', 'post', 'both', 'none')
        adapter_dropout: float = 0.0,  # Dropout rate
        target_modules: List[str] = None,  # Modules to add adapters to
        norm_type: str = "layer_norm",  # Type of normalization ('layer_norm', 'rms_norm')
        init_scale: float = 1e-3,  # Initialization scale
    ):
        """
        Initialize Adapter configuration.
        
        Args:
            dim: Bottleneck dimension of the adapter
            scaling: Scaling factor for the adapter outputs
            use_gate: Whether to use gating mechanism
            norm_position: Position of normalization layers
            adapter_dropout: Dropout probability for adapter layers
            target_modules: List of module types to add adapters to
            norm_type: Type of normalization to use
            init_scale: Initialization scale for adapter weights
        """
        self.dim = dim
        self.scaling = scaling
        self.use_gate = use_gate
        self.norm_position = norm_position
        self.adapter_dropout = adapter_dropout
        self.target_modules = target_modules or ["output", "attention.output", "ffn.output"]
        self.norm_type = norm_type
        self.init_scale = init_scale
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "AdapterConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class Adapter(nn.Module):
    """
    Adapter module for efficient fine-tuning.
    
    Implements adapters as described in "Parameter-Efficient Transfer Learning for NLP" 
    (Houlsby et al., 2019) and "AdapterFusion: Non-Destructive Task Composition for 
    Transfer Learning" (Pfeiffer et al., 2020).
    """
    
    def __init__(self, config: AdapterConfig, input_dim: int):
        """
        Initialize adapter module.
        
        Args:
            config: Adapter configuration
            input_dim: Dimension of input features
        """
        super().__init__()
        
        self.config = config
        self.input_dim = input_dim
        self.bottleneck_dim = config.dim
        
        # Create normalization layers
        if config.norm_type == "layer_norm":
            norm_class = nn.LayerNorm
        elif config.norm_type == "rms_norm":
            # Simple RMSNorm implementation
            class RMSNorm(nn.Module):
                def __init__(self, dim, eps=1e-6):
                    super().__init__()
                    self.eps = eps
                    self.weight = nn.Parameter(torch.ones(dim))
                
                def forward(self, x):
                    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                    return x * norm * self.weight
            
            norm_class = RMSNorm
        else:
            raise ValueError(f"Unknown normalization type: {config.norm_type}")
        
        # Create pre-normalization if needed
        if config.norm_position in ['pre', 'both']:
            self.pre_norm = norm_class(input_dim)
        else:
            self.pre_norm = nn.Identity()
        
        # Create down and up projections
        self.down_proj = nn.Linear(input_dim, self.bottleneck_dim)
        self.up_proj = nn.Linear(self.bottleneck_dim, input_dim)
        
        # Create post-normalization if needed
        if config.norm_position in ['post', 'both']:
            self.post_norm = norm_class(input_dim)
        else:
            self.post_norm = nn.Identity()
        
        # Create activation and dropout
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.adapter_dropout)
        
        # Create gate if needed
        if config.use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights."""
        # Initialize down projection with small weights
        nn.init.normal_(self.down_proj.weight, std=self.config.init_scale)
        nn.init.zeros_(self.down_proj.bias)
        
        # Initialize up projection with zeros for stable fine-tuning
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Adapted hidden states
        """
        # Apply pre-norm if needed
        residual = hidden_states
        x = self.pre_norm(hidden_states)
        
        # Down projection
        x = self.down_proj(x)
        
        # Activation
        x = self.act_fn(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Up projection
        x = self.up_proj(x)
        
        # Post-norm if needed
        x = self.post_norm(x)
        
        # Apply gate if needed
        if self.gate is not None:
            gate_factor = torch.sigmoid(self.gate)
            x = x * gate_factor
        
        # Apply scaling
        x = x * self.config.scaling
        
        # Add residual
        output = residual + x
        
        return output


class PrefixTuningConfig:
    """
    Configuration class for Prefix Tuning.
    """
    
    def __init__(
        self,
        prefix_length: int = 20,  # Length of the prefix
        num_virtual_tokens: int = 20,  # Number of virtual tokens
        encoder_hidden_size: Optional[int] = None,  # Encoder hidden size (if different)
        prefix_projection: bool = False,  # Whether to use prefix projection
        dropout: float = 0.0,  # Dropout probability
        prefix_dropout: float = 0.0,  # Dropout probability for prefix
        target_modules: List[str] = None,  # Modules to add prefix to
    ):
        """
        Initialize Prefix Tuning configuration.
        
        Args:
            prefix_length: Length of the prefix
            num_virtual_tokens: Number of virtual tokens
            encoder_hidden_size: Size of the encoder hidden states (for projection)
            prefix_projection: Whether to use prefix projection
            dropout: Dropout probability for layers
            prefix_dropout: Dropout probability for prefix
            target_modules: List of modules to add prefix to
        """
        self.prefix_length = prefix_length
        self.num_virtual_tokens = num_virtual_tokens
        self.encoder_hidden_size = encoder_hidden_size
        self.prefix_projection = prefix_projection
        self.dropout = dropout
        self.prefix_dropout = prefix_dropout
        self.target_modules = target_modules or ["attn"]
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "PrefixTuningConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


class PrefixEncoder(nn.Module):
    """
    Prefix Encoder module for Prefix Tuning.
    
    Implements the prefix encoder from "Prefix-Tuning: Optimizing Continuous Prompts
    for Generation" (Li & Liang, 2021).
    """
    
    def __init__(self, config: PrefixTuningConfig, hidden_size: int):
        """
        Initialize prefix encoder.
        
        Args:
            config: Prefix tuning configuration
            hidden_size: Hidden size of the model
        """
        super().__init__()
        
        self.config = config
        self.prefix_length = config.prefix_length
        self.hidden_size = hidden_size
        self.encoder_hidden_size = config.encoder_hidden_size or hidden_size
        
        # Create prefix embedding
        self.embedding = nn.Embedding(config.num_virtual_tokens, hidden_size)
        
        # Create projection if needed
        if config.prefix_projection:
            # Create MLP for prefix transformation
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, self.encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(self.encoder_hidden_size, 2 * hidden_size),  # For key and value
                nn.Dropout(config.prefix_dropout)
            )
        else:
            self.projection = nn.Identity()
            # If not using projection, directly learn prefix parameters
            self.prefix_tokens = nn.Parameter(
                torch.zeros(1, config.prefix_length, 2 * hidden_size)
            )
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Forward pass to get prefix key and value states.
        
        Args:
            batch_size: Batch size for prefix expansion
            
        Returns:
            Prefix key and value states
        """
        if self.config.prefix_projection:
            # Get embeddings and project them
            prefix_tokens = self.embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
            prefix_states = self.projection(prefix_tokens)
        else:
            # Use directly learned parameters
            prefix_states = self.prefix_tokens.expand(batch_size, -1, -1)
        
        # Reshape for key and value pairs
        # [batch_size, prefix_length, 2 * hidden_size] -> [batch_size, prefix_length, 2, hidden_size]
        prefix_states = prefix_states.view(batch_size, self.prefix_length, 2, self.hidden_size)
        
        # Split into key and value states
        # [batch_size, prefix_length, 2, hidden_size] -> 2 x [batch_size, prefix_length, hidden_size]
        prefix_key, prefix_value = prefix_states[:, :, 0], prefix_states[:, :, 1]
        
        return prefix_key, prefix_value


def apply_peft_to_model(
    model: nn.Module,
    peft_config: Dict,
    peft_type: str = "lora",
    target_modules: Optional[List[str]] = None,
    adapter_name: str = "default"
) -> nn.Module:
    """
    Apply PEFT technique to a PyTorch model.
    
    Args:
        model: The model to apply PEFT to
        peft_config: Configuration for the PEFT method
        peft_type: Type of PEFT to apply ('lora', 'adapter', 'prefix_tuning')
        target_modules: List of module names to target (overrides config)
        adapter_name: Name of the adapter for reference
        
    Returns:
        Model with PEFT applied
    """
    # Log PEFT application
    logger.info(f"Applying {peft_type} to model with name '{adapter_name}'")
    
    # Create appropriate config based on PEFT type
    if peft_type.lower() == "lora":
        config_class = LoRAConfig
        is_target = lambda name, module: (isinstance(module, nn.Linear) and 
                                         any(target in name for target in peft_config['target_modules']))
        wrapper_class = LoRALinear
    elif peft_type.lower() == "adapter":
        config_class = AdapterConfig
        is_target = lambda name, module: any(target in name for target in peft_config['target_modules'])
        wrapper_class = Adapter
    elif peft_type.lower() == "prefix_tuning":
        config_class = PrefixTuningConfig
        is_target = lambda name, module: any(target in name for target in peft_config['target_modules'])
        wrapper_class = None  # Prefix tuning requires special handling
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")
    
    # Create configuration
    config = config_class.from_dict(peft_config)
    
    # Override target modules if provided
    if target_modules is not None:
        config.target_modules = target_modules
    
    # Special handling for prefix tuning
    if peft_type.lower() == "prefix_tuning":
        # Implement prefix tuning directly on the model
        # This requires model-specific modifications
        raise NotImplementedError("Prefix tuning requires model-specific implementation")
    
    # Track original modules and their replacements
    peft_modules = {}
    
    # Apply PEFT to model by replacing modules
    for name, module in list(model.named_modules()):
        # Check if this module should be targeted
        if "." not in name and is_target(name, module):
            # Replace top-level module
            if peft_type.lower() == "lora" and isinstance(module, nn.Linear):
                # Replace linear with LoRA
                model._modules[name] = wrapper_class(module, config, adapter_name)
                peft_modules[name] = model._modules[name]
            elif peft_type.lower() == "adapter":
                # Add adapter after module
                adapter = wrapper_class(config, module.out_features if hasattr(module, 'out_features') else module.hidden_size)
                # Create sequential with module and adapter
                original_forward = module.forward
                
                def new_forward(self, x, *args, **kwargs):
                    outputs = original_forward(x, *args, **kwargs)
                    return adapter(outputs)
                
                module.forward = new_forward.__get__(module, type(module))
                peft_modules[name] = adapter
    
    # Recursively search for nested modules to replace
    for name, module in list(model.named_modules()):
        if "." in name:  # Nested module
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            
            # Check if this module should be targeted
            if is_target(name, module):
                if peft_type.lower() == "lora" and isinstance(module, nn.Linear):
                    # Replace linear with LoRA
                    setattr(parent, child_name, wrapper_class(module, config, adapter_name))
                    peft_modules[name] = getattr(parent, child_name)
                elif peft_type.lower() == "adapter":
                    # Add adapter after module
                    adapter = wrapper_class(config, module.out_features if hasattr(module, 'out_features') else module.hidden_size)
                    
                    # Create a new forward method that applies the adapter
                    original_module = module
                    original_forward = module.forward
                    
                    def new_forward(self, x, *args, **kwargs):
                        outputs = original_forward(x, *args, **kwargs)
                        return adapter(outputs)
                    
                    module.forward = new_forward.__get__(module, type(module))
                    peft_modules[name] = adapter
    
    # Add PEFT modules to model for tracking
    if not hasattr(model, "peft_modules"):
        model.peft_modules = {}
    
    model.peft_modules[adapter_name] = peft_modules
    
    # Add helper methods for PEFT management
    def set_peft_active(self, adapter_name=None, active=True):
        """Set whether PEFT modules are active."""
        adapter_names = [adapter_name] if adapter_name else list(self.peft_modules.keys())
        for name in adapter_names:
            if name in self.peft_modules:
                for module in self.peft_modules[name].values():
                    if hasattr(module, "active_adapter"):
                        module.active_adapter = active
    
    def merge_peft_weights(self, adapter_name=None):
        """Merge PEFT weights into base model weights."""
        adapter_names = [adapter_name] if adapter_name else list(self.peft_modules.keys())
        for name in adapter_names:
            if name in self.peft_modules:
                for module in self.peft_modules[name].values():
                    if hasattr(module, "merge_lora_weights"):
                        module.merge_lora_weights()
    
    def unmerge_peft_weights(self, adapter_name=None):
        """Unmerge PEFT weights from base model weights."""
        adapter_names = [adapter_name] if adapter_name else list(self.peft_modules.keys())
        for name in adapter_names:
            if name in self.peft_modules:
                for module in self.peft_modules[name].values():
                    if hasattr(module, "unmerge_lora_weights"):
                        module.unmerge_lora_weights()
    
    # Bind methods to model
    model.set_peft_active = set_peft_active.__get__(model, type(model))
    model.merge_peft_weights = merge_peft_weights.__get__(model, type(model))
    model.unmerge_peft_weights = unmerge_peft_weights.__get__(model, type(model))
    
    return model


def prepare_for_qlora(
    model: nn.Module,
    bits: int = 4,
    groupsize: int = 128,
    compute_dtype: str = "float16",
    double_quant: bool = True
) -> nn.Module:
    """
    Prepare a model for QLoRA fine-tuning by quantizing the model.
    
    Args:
        model: The model to prepare
        bits: Number of bits for quantization (4 or 8)
        groupsize: Size of quantization groups
        compute_dtype: Computation data type
        double_quant: Whether to use double quantization
    
    Returns:
        Quantized model ready for QLoRA
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import LinearFP4, Linear8bitLt
    except ImportError:
        raise ImportError("bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes")
    
    # Log quantization
    logger.info(f"Preparing model for QLoRA with {bits}-bit quantization")
    
    # Set compute dtype
    if compute_dtype == "float16":
        compute_dtype = torch.float16
    elif compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32
    
    # Replace linear layers with quantized versions
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not "lora" in name.lower():
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else model.get_submodule(parent_name)
            
            # Create quantized linear layer
            if bits == 4:
                new_module = LinearFP4(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    compress_statistics=double_quant,
                    quant_type="nf4"
                )
            elif bits == 8:
                new_module = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0
                )
            else:
                continue  # Skip unsupported bit sizes
            
            # Copy weights and bias
            if bits == 4:
                # For 4-bit, we need to use the special weight and bias props
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
            elif bits == 8:
                new_module.weight.data = module.weight.data
                if module.bias is not None:
                    new_module.bias.data = module.bias.data
            
            # Replace module
            setattr(parent, child_name, new_module)
    
    return model
