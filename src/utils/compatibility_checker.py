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
Compatibility checker for ensuring model compatibility with Flask and CoreML.

This module provides utilities to check if a model is compatible with Flask
applications and Apple's CoreML format.
"""

import os
import logging
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompatibilityChecker:
    """
    Checker for model compatibility with Flask and CoreML.
    """
    
    def __init__(self, model: nn.Module, output_dir: str = None):
        """
        Initialize compatibility checker.
        
        Args:
            model: PyTorch model to check
            output_dir: Directory to save compatibility reports
        """
        self.model = model
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def check_flask_compatibility(self) -> Dict[str, Any]:
        """
        Check if the model is compatible with Flask applications.
        
        Returns:
            Dictionary with compatibility information
        """
        logger.info("Checking Flask compatibility...")
        
        compatibility_info = {
            "is_compatible": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check 1: Model should be in eval mode for inference
        if self.model.training:
            compatibility_info["issues"].append("Model is in training mode")
            compatibility_info["recommendations"].append("Set model to evaluation mode with model.eval()")
        
        # Check 2: Model should have a forward method with standard parameters
        try:
            signature = str(self.model.forward.__code__.co_varnames)
            if "self" not in signature or ("input_ids" not in signature and "inputs" not in signature):
                compatibility_info["issues"].append("Model forward method has non-standard parameters")
                compatibility_info["recommendations"].append("Ensure model.forward accepts standard parameters like input_ids")
        except AttributeError:
            compatibility_info["issues"].append("Model does not have a standard forward method")
            compatibility_info["recommendations"].append("Implement a standard forward method")
        
        # Check 3: Model should be serializable
        try:
            with torch.no_grad():
                # Try to serialize the model
                torch.save(self.model.state_dict(), "temp_model.pt")
                os.remove("temp_model.pt")
        except Exception as e:
            compatibility_info["issues"].append(f"Model is not serializable: {str(e)}")
            compatibility_info["recommendations"].append("Ensure model can be serialized with torch.save")
        
        # Check 4: Model should have a generate method for text generation
        if not hasattr(self.model, "generate"):
            compatibility_info["issues"].append("Model does not have a generate method")
            compatibility_info["recommendations"].append("Implement a generate method for text generation")
        
        # Check 5: Model should not have custom CUDA operations
        has_custom_cuda = False
        for module in self.model.modules():
            if hasattr(module, "_custom_cuda_kernel"):
                has_custom_cuda = True
                break
        
        if has_custom_cuda:
            compatibility_info["issues"].append("Model has custom CUDA operations")
            compatibility_info["recommendations"].append("Replace custom CUDA operations with standard PyTorch operations")
        
        # Update compatibility status
        if compatibility_info["issues"]:
            compatibility_info["is_compatible"] = False
        
        # Save report if output_dir is provided
        if self.output_dir:
            report_path = os.path.join(self.output_dir, "flask_compatibility.json")
            with open(report_path, "w") as f:
                json.dump(compatibility_info, f, indent=2)
            
            logger.info(f"Flask compatibility report saved to {report_path}")
        
        return compatibility_info
    
    def check_coreml_compatibility(self) -> Dict[str, Any]:
        """
        Check if the model is compatible with Apple's CoreML format.
        
        Returns:
            Dictionary with compatibility information
        """
        logger.info("Checking CoreML compatibility...")
        
        compatibility_info = {
            "is_compatible": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check 1: Model should be in eval mode for inference
        if self.model.training:
            compatibility_info["issues"].append("Model is in training mode")
            compatibility_info["recommendations"].append("Set model to evaluation mode with model.eval()")
        
        # Check 2: Model should have fixed input shapes for CoreML
        if hasattr(self.model, "forward") and "input_ids" in str(self.model.forward.__code__.co_varnames):
            compatibility_info["recommendations"].append("Ensure model can handle fixed input shapes required by CoreML")
        
        # Check 3: Model size should be reasonable for mobile deployment
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        if model_size_mb > 100:
            compatibility_info["issues"].append(f"Model size ({model_size_mb:.2f} MB) may be too large for mobile deployment")
            compatibility_info["recommendations"].append("Consider quantization and pruning to reduce model size")
        
        # Check 4: Model should not use dynamic control flow
        # This is a simplified check - a real check would analyze the model's graph
        has_dynamic_control_flow = False
        for name, module in self.model.named_modules():
            if hasattr(module, "forward") and ("if" in str(module.forward.__code__.co_consts) or "for" in str(module.forward.__code__.co_consts)):
                has_dynamic_control_flow = True
                break
        
        if has_dynamic_control_flow:
            compatibility_info["issues"].append("Model may use dynamic control flow, which can be problematic for CoreML conversion")
            compatibility_info["recommendations"].append("Replace dynamic control flow with static operations where possible")
        
        # Check 5: Model should not use unsupported operations
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
        for op in unsupported_ops:
            op_name = op.split("::")[-1]
            if hasattr(torch, op_name) and any(op_name in str(m) for m in self.model.modules()):
                compatibility_info["issues"].append(f"Model may use unsupported operation: {op}")
                compatibility_info["recommendations"].append(f"Replace {op_name} with a supported alternative")
        
        # Update compatibility status
        if compatibility_info["issues"]:
            compatibility_info["is_compatible"] = False
        
        # Save report if output_dir is provided
        if self.output_dir:
            report_path = os.path.join(self.output_dir, "coreml_compatibility.json")
            with open(report_path, "w") as f:
                json.dump(compatibility_info, f, indent=2)
            
            logger.info(f"CoreML compatibility report saved to {report_path}")
        
        return compatibility_info
    
    def check_compatibility(self) -> Dict[str, Dict[str, Any]]:
        """
        Check model compatibility with both Flask and CoreML.
        
        Returns:
            Dictionary with compatibility information for both platforms
        """
        flask_compatibility = self.check_flask_compatibility()
        coreml_compatibility = self.check_coreml_compatibility()
        
        compatibility_info = {
            "flask": flask_compatibility,
            "coreml": coreml_compatibility,
            "overall": {
                "is_compatible": flask_compatibility["is_compatible"] and coreml_compatibility["is_compatible"],
                "issues_count": len(flask_compatibility["issues"]) + len(coreml_compatibility["issues"]),
                "recommendations_count": len(flask_compatibility["recommendations"]) + len(coreml_compatibility["recommendations"])
            }
        }
        
        # Save overall report if output_dir is provided
        if self.output_dir:
            report_path = os.path.join(self.output_dir, "compatibility_report.json")
            with open(report_path, "w") as f:
                json.dump(compatibility_info, f, indent=2)
            
            logger.info(f"Overall compatibility report saved to {report_path}")
            
            # Create a human-readable report
            markdown_report = self._generate_markdown_report(compatibility_info)
            markdown_path = os.path.join(self.output_dir, "compatibility_report.md")
            with open(markdown_path, "w") as f:
                f.write(markdown_report)
            
            logger.info(f"Human-readable compatibility report saved to {markdown_path}")
        
        return compatibility_info
    
    def _generate_markdown_report(self, compatibility_info: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a human-readable markdown report from compatibility information.
        
        Args:
            compatibility_info: Compatibility information
            
        Returns:
            Markdown report
        """
        flask_info = compatibility_info["flask"]
        coreml_info = compatibility_info["coreml"]
        overall_info = compatibility_info["overall"]
        
        report = "# Model Compatibility Report\n\n"
        
        # Overall compatibility
        report += "## Overall Compatibility\n\n"
        report += f"**Compatible with both platforms:** {'Yes' if overall_info['is_compatible'] else 'No'}\n\n"
        report += f"**Total issues:** {overall_info['issues_count']}\n\n"
        report += f"**Total recommendations:** {overall_info['recommendations_count']}\n\n"
        
        # Model information
        report += "## Model Information\n\n"
        
        # Model size
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        report += f"**Model size:** {model_size_mb:.2f} MB\n\n"
        
        # Parameter count
        param_count = sum(p.numel() for p in self.model.parameters())
        report += f"**Parameter count:** {param_count:,}\n\n"
        
        # Model architecture summary
        report += "**Model architecture:**\n\n"
        report += "```\n"
        report += str(self.model)
        report += "\n```\n\n"
        
        # Flask compatibility
        report += "## Flask Compatibility\n\n"
        report += f"**Compatible:** {'Yes' if flask_info['is_compatible'] else 'No'}\n\n"
        
        if flask_info["issues"]:
            report += "### Issues\n\n"
            for issue in flask_info["issues"]:
                report += f"- {issue}\n"
            report += "\n"
        
        if flask_info["recommendations"]:
            report += "### Recommendations\n\n"
            for rec in flask_info["recommendations"]:
                report += f"- {rec}\n"
            report += "\n"
        
        # CoreML compatibility
        report += "## CoreML Compatibility\n\n"
        report += f"**Compatible:** {'Yes' if coreml_info['is_compatible'] else 'No'}\n\n"
        
        if coreml_info["issues"]:
            report += "### Issues\n\n"
            for issue in coreml_info["issues"]:
                report += f"- {issue}\n"
            report += "\n"
        
        if coreml_info["recommendations"]:
            report += "### Recommendations\n\n"
            for rec in coreml_info["recommendations"]:
                report += f"- {rec}\n"
            report += "\n"
        
        return report


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from model.architecture import TransformerModel
    
    # Create a model
    model = TransformerModel(
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create compatibility checker
    checker = CompatibilityChecker(model, output_dir="outputs/compatibility")
    
    # Check compatibility
    compatibility_info = checker.check_compatibility()
    
    # Print results
    print(f"Flask compatibility: {compatibility_info['flask']['is_compatible']}")
    print(f"CoreML compatibility: {compatibility_info['coreml']['is_compatible']}")
    print(f"Overall compatibility: {compatibility_info['overall']['is_compatible']}")