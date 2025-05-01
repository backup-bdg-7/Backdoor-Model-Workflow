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
Compatibility utilities for ensuring model can be used with Flask and CoreML.

This module provides utilities to check and ensure that models are compatible
with Flask applications and Apple's CoreML format without implementing the
full deployment pipeline.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_flask_compatibility(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Check if a PyTorch model is compatible with Flask applications.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        Dictionary with compatibility information
    """
    compatibility_info = {
        "is_compatible": True,
        "issues": [],
        "recommendations": []
    }
    
    # Check 1: Model should be in eval mode for inference
    if model.training:
        compatibility_info["is_compatible"] = False
        compatibility_info["issues"].append("Model is in training mode")
        compatibility_info["recommendations"].append("Set model to evaluation mode with model.eval()")
    
    # Check 2: Model should have a forward method with standard parameters
    try:
        signature = str(model.forward.__code__.co_varnames)
        if "self" not in signature or ("input_ids" not in signature and "inputs" not in signature):
            compatibility_info["issues"].append("Model forward method has non-standard parameters")
            compatibility_info["recommendations"].append("Ensure model.forward accepts standard parameters like input_ids")
    except AttributeError:
        compatibility_info["is_compatible"] = False
        compatibility_info["issues"].append("Model does not have a standard forward method")
        compatibility_info["recommendations"].append("Implement a standard forward method")
    
    # Check 3: Model should be serializable
    try:
        with torch.no_grad():
            # Try to serialize the model
            torch.save(model.state_dict(), "temp_model.pt")
            os.remove("temp_model.pt")
    except Exception as e:
        compatibility_info["is_compatible"] = False
        compatibility_info["issues"].append(f"Model is not serializable: {str(e)}")
        compatibility_info["recommendations"].append("Ensure model can be serialized with torch.save")
    
    # Check 4: Model should not have custom CUDA operations
    has_custom_cuda = False
    for module in model.modules():
        if hasattr(module, "_custom_cuda_kernel"):
            has_custom_cuda = True
            break
    
    if has_custom_cuda:
        compatibility_info["issues"].append("Model has custom CUDA operations")
        compatibility_info["recommendations"].append("Replace custom CUDA operations with standard PyTorch operations")
    
    # Check 5: Model should have a generate method for text generation
    if not hasattr(model, "generate"):
        compatibility_info["issues"].append("Model does not have a generate method")
        compatibility_info["recommendations"].append("Implement a generate method for text generation")
    
    return compatibility_info

def check_coreml_compatibility(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Check if a PyTorch model is compatible with Apple's CoreML format.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        Dictionary with compatibility information
    """
    compatibility_info = {
        "is_compatible": True,
        "issues": [],
        "recommendations": []
    }
    
    # Check 1: Model should be in eval mode for inference
    if model.training:
        compatibility_info["is_compatible"] = False
        compatibility_info["issues"].append("Model is in training mode")
        compatibility_info["recommendations"].append("Set model to evaluation mode with model.eval()")
    
    # Check 2: Model should not have unsupported operations
    unsupported_ops = [
        "aten::_th_bincount",
        "aten::_weight_norm",
        "aten::addbmm",
        "aten::addcdiv",
        "aten::addcmul",
        "aten::addmv",
        "aten::addr",
        "aten::baddbmm",
        "aten::bernoulli",
        "aten::col2im",
        "aten::cumprod",
        "aten::cumsum",
        "aten::digamma",
        "aten::elu",
        "aten::fft",
        "aten::gather",
        "aten::glu",
        "aten::histc",
        "aten::im2col",
        "aten::index_add",
        "aten::index_copy",
        "aten::index_fill",
        "aten::index_put",
        "aten::index_select",
        "aten::inverse",
        "aten::kthvalue",
        "aten::log_sigmoid",
        "aten::logcumsumexp",
        "aten::lstsq",
        "aten::margin_ranking_loss",
        "aten::masked_fill",
        "aten::masked_scatter",
        "aten::masked_select",
        "aten::matrix_power",
        "aten::median",
        "aten::mode",
        "aten::multinomial",
        "aten::mvlgamma",
        "aten::narrow",
        "aten::norm",
        "aten::normal",
        "aten::nuclear_norm",
        "aten::pdist",
        "aten::polygamma",
        "aten::prelu",
        "aten::qr",
        "aten::random",
        "aten::renorm",
        "aten::scatter",
        "aten::scatter_add",
        "aten::slogdet",
        "aten::smm",
        "aten::solve",
        "aten::svd",
        "aten::symeig",
        "aten::take",
        "aten::tensordot",
        "aten::topk",
        "aten::trace",
        "aten::triangular_solve",
        "aten::tril",
        "aten::triu",
        "aten::trunc",
        "aten::unfold",
        "aten::unique",
        "aten::unique_consecutive"
    ]
    
    # This is a simplified check - a real check would trace the model and analyze the graph
    # For demonstration purposes, we'll just check for common unsupported operations in the model's methods
    for op in unsupported_ops:
        op_name = op.split("::")[-1]
        if hasattr(torch, op_name) and any(op_name in str(m) for m in model.modules()):
            compatibility_info["issues"].append(f"Model may use unsupported operation: {op}")
            compatibility_info["recommendations"].append(f"Replace {op_name} with a supported alternative")
    
    # Check 3: Model should have fixed input shapes for CoreML
    if hasattr(model, "forward") and "input_ids" in str(model.forward.__code__.co_varnames):
        compatibility_info["recommendations"].append("Ensure model can handle fixed input shapes required by CoreML")
    
    # Check 4: Model size should be reasonable for mobile deployment
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    if model_size_mb > 100:
        compatibility_info["issues"].append(f"Model size ({model_size_mb:.2f} MB) may be too large for mobile deployment")
        compatibility_info["recommendations"].append("Consider quantization and pruning to reduce model size")
    
    # Check 5: Model should not use dynamic control flow
    # This is a simplified check - a real check would analyze the model's graph
    has_dynamic_control_flow = False
    for name, module in model.named_modules():
        if "if" in str(module.forward.__code__.co_consts) or "for" in str(module.forward.__code__.co_consts):
            has_dynamic_control_flow = True
            break
    
    if has_dynamic_control_flow:
        compatibility_info["issues"].append("Model may use dynamic control flow, which can be problematic for CoreML conversion")
        compatibility_info["recommendations"].append("Replace dynamic control flow with static operations where possible")
    
    return compatibility_info

def prepare_model_for_compatibility(
    model: torch.nn.Module,
    target_platforms: List[str] = ["flask", "coreml"]
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Prepare a PyTorch model for compatibility with target platforms.
    
    Args:
        model: PyTorch model to prepare
        target_platforms: List of target platforms (flask, coreml)
        
    Returns:
        Tuple of (prepared model, compatibility info)
    """
    # Set model to evaluation mode
    model.eval()
    
    compatibility_info = {}
    
    # Check compatibility with each target platform
    if "flask" in target_platforms:
        compatibility_info["flask"] = check_flask_compatibility(model)
    
    if "coreml" in target_platforms:
        compatibility_info["coreml"] = check_coreml_compatibility(model)
    
    # Apply basic compatibility fixes
    prepared_model = model
    
    # Fix 1: Ensure all parameters are on CPU for easier serialization
    prepared_model = prepared_model.cpu()
    
    # Fix 2: Freeze the model to prevent accidental updates
    for param in prepared_model.parameters():
        param.requires_grad = False
    
    # Fix 3: Add generate method if not present
    if not hasattr(prepared_model, "generate") and hasattr(prepared_model, "forward"):
        def generate(self, input_ids, attention_mask=None, max_new_tokens=20, **kwargs):
            """Simple generate method for text generation."""
            batch_size, seq_length = input_ids.shape
            
            # Initialize generated sequence with input_ids
            generated = input_ids.clone()
            
            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Get the last token
                inputs = generated[:, -seq_length:] if generated.size(1) > seq_length else generated
                
                # Create attention mask if not provided
                if attention_mask is None:
                    mask = torch.ones_like(inputs)
                else:
                    mask = attention_mask[:, -seq_length:] if attention_mask.size(1) > seq_length else attention_mask
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.forward(inputs, mask)
                
                # Get the next token
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the next token
                generated = torch.cat([generated, next_token], dim=1)
            
            return generated
        
        # Add the generate method to the model
        import types
        prepared_model.generate = types.MethodType(generate, prepared_model)
    
    return prepared_model, compatibility_info

def trace_model_for_export(
    model: torch.nn.Module,
    input_shapes: Dict[str, List[int]],
    output_path: Optional[str] = None
) -> torch.jit.ScriptModule:
    """
    Trace a PyTorch model for export to other formats.
    
    Args:
        model: PyTorch model to trace
        input_shapes: Dictionary mapping input names to their shapes
        output_path: Path to save the traced model
        
    Returns:
        Traced model
    """
    logger.info("Tracing model for export")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create example inputs
    example_inputs = []
    for name, shape in input_shapes.items():
        example_input = torch.randint(0, 100, shape)
        example_inputs.append(example_input)
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs)
    
    # Save the traced model if output_path is provided
    if output_path:
        traced_model.save(output_path)
        logger.info(f"Traced model saved to {output_path}")
    
    return traced_model

def get_compatibility_report(model: torch.nn.Module) -> str:
    """
    Generate a comprehensive compatibility report for a PyTorch model.
    
    Args:
        model: PyTorch model to check
        
    Returns:
        Compatibility report as a string
    """
    # Check compatibility with all platforms
    _, compatibility_info = prepare_model_for_compatibility(model)
    
    # Generate report
    report = "# Model Compatibility Report\n\n"
    
    # Flask compatibility
    report += "## Flask Compatibility\n\n"
    flask_info = compatibility_info.get("flask", {})
    report += f"**Compatible:** {flask_info.get('is_compatible', False)}\n\n"
    
    if flask_info.get("issues"):
        report += "### Issues\n\n"
        for issue in flask_info.get("issues", []):
            report += f"- {issue}\n"
        report += "\n"
    
    if flask_info.get("recommendations"):
        report += "### Recommendations\n\n"
        for rec in flask_info.get("recommendations", []):
            report += f"- {rec}\n"
        report += "\n"
    
    # CoreML compatibility
    report += "## CoreML Compatibility\n\n"
    coreml_info = compatibility_info.get("coreml", {})
    report += f"**Compatible:** {coreml_info.get('is_compatible', False)}\n\n"
    
    if coreml_info.get("issues"):
        report += "### Issues\n\n"
        for issue in coreml_info.get("issues", []):
            report += f"- {issue}\n"
        report += "\n"
    
    if coreml_info.get("recommendations"):
        report += "### Recommendations\n\n"
        for rec in coreml_info.get("recommendations", []):
            report += f"- {rec}\n"
        report += "\n"
    
    # Model information
    report += "## Model Information\n\n"
    
    # Model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    report += f"**Model Size:** {model_size_mb:.2f} MB\n\n"
    
    # Parameter count
    param_count = sum(p.numel() for p in model.parameters())
    report += f"**Parameter Count:** {param_count:,}\n\n"
    
    # Model architecture summary
    report += "**Model Architecture:**\n\n"
    report += "```\n"
    report += str(model)
    report += "\n```\n\n"
    
    return report