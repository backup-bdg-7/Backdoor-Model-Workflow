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
Model optimization utilities for the AI model training workflow.
This module provides functions for model quantization, pruning, and optimization.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Try to import optional dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch.quantization
    from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
    TORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    TORCH_QUANTIZATION_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    A class to handle model optimization, quantization, and export.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the model optimizer.
        
        Args:
            model: Model to optimize
            tokenizer: Tokenizer for the model
            device: Device to use for optimization
            output_dir: Directory to save optimized models
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir or "./optimized_models"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def quantize_dynamic(
        self,
        quantization_config: Optional[Dict] = None,
        save_model: bool = True,
    ) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Args:
            quantization_config: Configuration for quantization
            save_model: Whether to save the quantized model
            
        Returns:
            Quantized model
        """
        if not TORCH_QUANTIZATION_AVAILABLE:
            raise ImportError("PyTorch quantization is not available. Please install PyTorch with quantization support.")
        
        logger.info("Applying dynamic quantization to the model")
        
        # Set default quantization config if not provided
        if quantization_config is None:
            quantization_config = {
                "dtype": torch.qint8,
                "qconfig_spec": {
                    nn.Linear: torch.quantization.default_dynamic_qconfig
                }
            }
        
        # Move model to CPU for quantization
        model_to_quantize = self.model.cpu()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize,
            qconfig_spec=quantization_config["qconfig_spec"],
            dtype=quantization_config["dtype"]
        )
        
        # Save quantized model if requested
        if save_model:
            model_path = os.path.join(self.output_dir, "quantized_dynamic_model.pt")
            torch.save(quantized_model.state_dict(), model_path)
            
            # Save tokenizer if available
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(os.path.join(self.output_dir, "quantized_dynamic_tokenizer"))
            
            logger.info(f"Saved dynamically quantized model to {model_path}")
        
        return quantized_model
    
    def quantize_static(
        self,
        calibration_dataset: Any,
        batch_size: int = 8,
        quantization_config: Optional[Dict] = None,
        save_model: bool = True,
    ) -> nn.Module:
        """
        Apply static quantization to the model.
        
        Args:
            calibration_dataset: Dataset for calibration
            batch_size: Batch size for calibration
            quantization_config: Configuration for quantization
            save_model: Whether to save the quantized model
            
        Returns:
            Quantized model
        """
        if not TORCH_QUANTIZATION_AVAILABLE:
            raise ImportError("PyTorch quantization is not available. Please install PyTorch with quantization support.")
        
        logger.info("Applying static quantization to the model")
        
        # Set default quantization config if not provided
        if quantization_config is None:
            quantization_config = {
                "qconfig": torch.quantization.get_default_qconfig("fbgemm"),
                "modules_to_fuse": []  # Add module names to fuse (e.g., ["conv", "bn", "relu"])
            }
        
        # Prepare model for static quantization
        model_to_quantize = self.model.cpu()
        
        # Add QuantStub and DeQuantStub
        if not hasattr(model_to_quantize, "quant"):
            model_to_quantize.quant = QuantStub()
        if not hasattr(model_to_quantize, "dequant"):
            model_to_quantize.dequant = DeQuantStub()
        
        # Fuse modules if specified
        if quantization_config["modules_to_fuse"]:
            torch.quantization.fuse_modules(
                model_to_quantize,
                quantization_config["modules_to_fuse"],
                inplace=True
            )
        
        # Set qconfig
        model_to_quantize.qconfig = quantization_config["qconfig"]
        torch.quantization.prepare(model_to_quantize, inplace=True)
        
        # Calibrate with dataset
        model_to_quantize.eval()
        with torch.no_grad():
            for i in range(0, len(calibration_dataset), batch_size):
                batch = calibration_dataset[i:i+batch_size]
                if isinstance(batch, dict):
                    # Forward pass for calibration
                    _ = model_to_quantize(**batch)
                else:
                    # Handle other dataset formats
                    inputs = self._prepare_calibration_inputs(batch)
                    _ = model_to_quantize(**inputs)
        
        # Convert to quantized model
        torch.quantization.convert(model_to_quantize, inplace=True)
        
        # Save quantized model if requested
        if save_model:
            model_path = os.path.join(self.output_dir, "quantized_static_model.pt")
            torch.save(model_to_quantize.state_dict(), model_path)
            
            # Save tokenizer if available
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(os.path.join(self.output_dir, "quantized_static_tokenizer"))
            
            logger.info(f"Saved statically quantized model to {model_path}")
        
        return model_to_quantize
    
    def quantize_with_bitsandbytes(
        self,
        quantization_type: str = "int8",
        save_model: bool = True,
    ) -> nn.Module:
        """
        Quantize model using bitsandbytes library.
        
        Args:
            quantization_type: Type of quantization (int8, int4, nf4)
            save_model: Whether to save the quantized model
            
        Returns:
            Quantized model
        """
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes is not available. Please install with: pip install bitsandbytes")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not available. Please install with: pip install transformers")
        
        logger.info(f"Applying {quantization_type} quantization with bitsandbytes")
        
        # Get model class and config
        model_config = self.model.config
        
        # Determine model type
        if hasattr(self.model, "model_type"):
            model_type = self.model.model_type
        else:
            model_type = "causal_lm"  # Default to causal language model
        
        # Load quantized model
        if model_type == "causal_lm" or model_type in ["gpt2", "gpt_neo", "gptj", "llama"]:
            if quantization_type == "int8":
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    self.model.name_or_path if hasattr(self.model, "name_or_path") else "gpt2",
                    load_in_8bit=True,
                    device_map="auto"
                )
            elif quantization_type == "int4":
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    self.model.name_or_path if hasattr(self.model, "name_or_path") else "gpt2",
                    load_in_4bit=True,
                    device_map="auto"
                )
            elif quantization_type == "nf4":
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    self.model.name_or_path if hasattr(self.model, "name_or_path") else "gpt2",
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    device_map="auto"
                )
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
        elif model_type in ["bert", "roberta", "distilbert"]:
            if quantization_type == "int8":
                quantized_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model.name_or_path if hasattr(self.model, "name_or_path") else "bert-base-uncased",
                    load_in_8bit=True,
                    device_map="auto"
                )
            elif quantization_type == "int4":
                quantized_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model.name_or_path if hasattr(self.model, "name_or_path") else "bert-base-uncased",
                    load_in_4bit=True,
                    device_map="auto"
                )
            elif quantization_type == "nf4":
                quantized_model = AutoModelForSequenceClassification.from_pretrained(
                    self.model.name_or_path if hasattr(self.model, "name_or_path") else "bert-base-uncased",
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    device_map="auto"
                )
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
        else:
            raise ValueError(f"Unsupported model type for bitsandbytes quantization: {model_type}")
        
        # Save quantized model if requested
        if save_model:
            model_dir = os.path.join(self.output_dir, f"quantized_{quantization_type}_model")
            os.makedirs(model_dir, exist_ok=True)
            
            quantized_model.save_pretrained(model_dir)
            
            # Save tokenizer if available
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(model_dir)
            
            logger.info(f"Saved {quantization_type} quantized model to {model_dir}")
        
        return quantized_model
    
    def export_to_onnx(
        self,
        input_shapes: Dict[str, List[int]],
        output_names: List[str],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 12,
        optimize: bool = True,
        quantize: bool = False,
        save_model: bool = True,
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            input_shapes: Dictionary of input names and shapes
            output_names: List of output names
            dynamic_axes: Dictionary of dynamic axes
            opset_version: ONNX opset version
            optimize: Whether to optimize the ONNX model
            quantize: Whether to quantize the ONNX model
            save_model: Whether to save the ONNX model
            
        Returns:
            Path to the exported ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is not available. Please install with: pip install onnx onnxruntime")
        
        logger.info("Exporting model to ONNX format")
        
        # Prepare model for export
        model_to_export = self.model.to(self.device)
        model_to_export.eval()
        
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.zeros(shape, dtype=torch.long, device=self.device)
        
        # Set dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {}
            for name in input_shapes:
                dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        
        # Export to ONNX
        onnx_path = os.path.join(self.output_dir, "model.onnx")
        with torch.no_grad():
            torch.onnx.export(
                model_to_export,
                tuple(dummy_inputs.values()),
                onnx_path,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
        
        logger.info(f"Exported model to ONNX format at {onnx_path}")
        
        # Optimize ONNX model if requested
        if optimize:
            optimized_path = self._optimize_onnx(onnx_path)
            onnx_path = optimized_path
        
        # Quantize ONNX model if requested
        if quantize:
            if not OPTIMUM_AVAILABLE:
                logger.warning("optimum-onnxruntime is not available. Skipping ONNX quantization.")
            else:
                quantized_path = self._quantize_onnx(onnx_path)
                onnx_path = quantized_path
        
        # Save tokenizer if available
        if self.tokenizer is not None and save_model:
            self.tokenizer.save_pretrained(os.path.join(self.output_dir, "onnx_tokenizer"))
        
        return onnx_path
    
    def _optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimize ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Path to optimized ONNX model
        """
        logger.info("Optimizing ONNX model")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Check model
        onnx.checker.check_model(onnx_model)
        
        # Optimize model
        from onnxruntime.transformers import optimizer
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type="gpt2",  # Use appropriate model type
            num_heads=12,  # Set appropriate number of attention heads
            hidden_size=768  # Set appropriate hidden size
        )
        
        # Save optimized model
        optimized_path = os.path.join(self.output_dir, "model_optimized.onnx")
        optimized_model.save_model_to_file(optimized_path)
        
        logger.info(f"Saved optimized ONNX model to {optimized_path}")
        
        return optimized_path
    
    def _quantize_onnx(self, onnx_path: str) -> str:
        """
        Quantize ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Path to quantized ONNX model
        """
        logger.info("Quantizing ONNX model")
        
        # Create quantizer
        quantizer = ORTQuantizer.from_pretrained(onnx_path)
        
        # Create quantization configuration
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        # Quantize model
        quantizer.quantize(
            save_dir=self.output_dir,
            quantization_config=qconfig
        )
        
        quantized_path = os.path.join(self.output_dir, "model_quantized.onnx")
        
        logger.info(f"Saved quantized ONNX model to {quantized_path}")
        
        return quantized_path
    
    def _prepare_calibration_inputs(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for calibration.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of inputs
        """
        if isinstance(batch, dict):
            return batch
        
        # Handle different batch formats
        if isinstance(batch, torch.Tensor):
            return {"input_ids": batch}
        
        if isinstance(batch, tuple) and len(batch) == 2:
            return {"input_ids": batch[0], "attention_mask": batch[1]}
        
        if isinstance(batch, list):
            if all(isinstance(item, torch.Tensor) for item in batch):
                return {"input_ids": batch[0]}
            
            if all(isinstance(item, dict) for item in batch):
                # Combine batch of dictionaries
                result = {}
                for key in batch[0].keys():
                    if all(key in item for item in batch):
                        result[key] = torch.stack([item[key] for item in batch])
                return result
        
        raise ValueError(f"Unsupported batch format for calibration: {type(batch)}")
    
    def prune_model(
        self,
        pruning_config: Dict,
        save_model: bool = True,
    ) -> nn.Module:
        """
        Apply pruning to the model.
        
        Args:
            pruning_config: Configuration for pruning
            save_model: Whether to save the pruned model
            
        Returns:
            Pruned model
        """
        logger.info("Applying pruning to the model")
        
        # Set default pruning config if not provided
        if pruning_config is None:
            pruning_config = {
                "method": "l1_unstructured",
                "amount": 0.3,
                "parameters_to_prune": ["weight"]
            }
        
        # Get pruning method
        method = pruning_config["method"]
        amount = pruning_config["amount"]
        parameters_to_prune = pruning_config["parameters_to_prune"]
        
        # Apply pruning
        model_to_prune = self.model.to(self.device)
        
        # Collect parameters to prune
        parameters_to_prune_list = []
        for name, module in model_to_prune.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                for param_name in parameters_to_prune:
                    if hasattr(module, param_name):
                        parameters_to_prune_list.append((module, param_name))
        
        # Apply pruning
        if method == "l1_unstructured":
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune_list,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=amount
            )
        elif method == "random_unstructured":
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune_list,
                pruning_method=torch.nn.utils.prune.RandomUnstructured,
                amount=amount
            )
        else:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune_list:
            torch.nn.utils.prune.remove(module, param_name)
        
        # Save pruned model if requested
        if save_model:
            model_path = os.path.join(self.output_dir, "pruned_model.pt")
            torch.save(model_to_prune.state_dict(), model_path)
            
            # Save tokenizer if available
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(os.path.join(self.output_dir, "pruned_tokenizer"))
            
            logger.info(f"Saved pruned model to {model_path}")
        
        return model_to_prune
    
    def benchmark_model(
        self,
        input_shapes: Dict[str, List[int]],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        compare_with_onnx: bool = False,
        onnx_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            input_shapes: Dictionary of input names and shapes
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            compare_with_onnx: Whether to compare with ONNX model
            onnx_path: Path to ONNX model
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Benchmarking model performance")
        
        # Prepare model for benchmarking
        model_to_benchmark = self.model.to(self.device)
        model_to_benchmark.eval()
        
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.zeros(shape, dtype=torch.long, device=self.device)
        
        # Benchmark PyTorch model
        torch_latencies = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model_to_benchmark(**dummy_inputs)
        
        # Benchmark
        with torch.no_grad():
            start_time = time.time()
            for _ in range(num_iterations):
                iter_start = time.time()
                _ = model_to_benchmark(**dummy_inputs)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                iter_end = time.time()
                torch_latencies.append((iter_end - iter_start) * 1000)  # ms
            end_time = time.time()
        
        torch_total_time = end_time - start_time
        torch_avg_latency = np.mean(torch_latencies)
        torch_p95_latency = np.percentile(torch_latencies, 95)
        torch_throughput = num_iterations / torch_total_time
        
        results = {
            "pytorch": {
                "avg_latency_ms": torch_avg_latency,
                "p95_latency_ms": torch_p95_latency,
                "throughput_ips": torch_throughput,
                "total_time_s": torch_total_time
            }
        }
        
        # Benchmark ONNX model if requested
        if compare_with_onnx:
            if not ONNX_AVAILABLE:
                logger.warning("ONNX is not available. Skipping ONNX benchmarking.")
            else:
                if onnx_path is None:
                    logger.warning("ONNX path not provided. Skipping ONNX benchmarking.")
                else:
                    # Create ONNX session
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session = ort.InferenceSession(
                        onnx_path,
                        sess_options=session_options,
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    
                    # Prepare inputs for ONNX
                    onnx_inputs = {}
                    for name, tensor in dummy_inputs.items():
                        onnx_inputs[name] = tensor.cpu().numpy()
                    
                    # Benchmark ONNX model
                    onnx_latencies = []
                    
                    # Warmup
                    for _ in range(warmup_iterations):
                        _ = session.run(None, onnx_inputs)
                    
                    # Benchmark
                    start_time = time.time()
                    for _ in range(num_iterations):
                        iter_start = time.time()
                        _ = session.run(None, onnx_inputs)
                        iter_end = time.time()
                        onnx_latencies.append((iter_end - iter_start) * 1000)  # ms
                    end_time = time.time()
                    
                    onnx_total_time = end_time - start_time
                    onnx_avg_latency = np.mean(onnx_latencies)
                    onnx_p95_latency = np.percentile(onnx_latencies, 95)
                    onnx_throughput = num_iterations / onnx_total_time
                    
                    results["onnx"] = {
                        "avg_latency_ms": onnx_avg_latency,
                        "p95_latency_ms": onnx_p95_latency,
                        "throughput_ips": onnx_throughput,
                        "total_time_s": onnx_total_time
                    }
        
        # Print results
        logger.info("Benchmark results:")
        logger.info(f"PyTorch - Avg Latency: {results['pytorch']['avg_latency_ms']:.2f} ms, "
                   f"P95 Latency: {results['pytorch']['p95_latency_ms']:.2f} ms, "
                   f"Throughput: {results['pytorch']['throughput_ips']:.2f} inferences/sec")
        
        if "onnx" in results:
            logger.info(f"ONNX - Avg Latency: {results['onnx']['avg_latency_ms']:.2f} ms, "
                       f"P95 Latency: {results['onnx']['p95_latency_ms']:.2f} ms, "
                       f"Throughput: {results['onnx']['throughput_ips']:.2f} inferences/sec")
            
            # Calculate speedup
            speedup = results['pytorch']['avg_latency_ms'] / results['onnx']['avg_latency_ms']
            logger.info(f"ONNX Speedup: {speedup:.2f}x")
            results["speedup"] = speedup
        
        return results


def optimize_model(
    model: nn.Module,
    tokenizer: Any = None,
    optimization_type: str = "dynamic_quantization",
    calibration_dataset: Optional[Any] = None,
    batch_size: int = 8,
    quantization_config: Optional[Dict] = None,
    pruning_config: Optional[Dict] = None,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 12,
    optimize_onnx: bool = True,
    quantize_onnx: bool = False,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    save_model: bool = True,
    benchmark: bool = False,
    num_benchmark_iterations: int = 100,
) -> Union[nn.Module, str, Dict[str, Any]]:
    """
    Optimize a model using various techniques.
    
    Args:
        model: Model to optimize
        tokenizer: Tokenizer for the model
        optimization_type: Type of optimization to apply
        calibration_dataset: Dataset for calibration (for static quantization)
        batch_size: Batch size for calibration
        quantization_config: Configuration for quantization
        pruning_config: Configuration for pruning
        input_shapes: Dictionary of input names and shapes (for ONNX export)
        output_names: List of output names (for ONNX export)
        dynamic_axes: Dictionary of dynamic axes (for ONNX export)
        opset_version: ONNX opset version
        optimize_onnx: Whether to optimize the ONNX model
        quantize_onnx: Whether to quantize the ONNX model
        device: Device to use for optimization
        output_dir: Directory to save optimized models
        save_model: Whether to save the optimized model
        benchmark: Whether to benchmark the optimized model
        num_benchmark_iterations: Number of iterations for benchmarking
        
    Returns:
        Optimized model, path to ONNX model, or benchmark results
    """
    # Create model optimizer
    optimizer = ModelOptimizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir
    )
    
    # Apply optimization based on type
    if optimization_type == "dynamic_quantization":
        optimized_model = optimizer.quantize_dynamic(
            quantization_config=quantization_config,
            save_model=save_model
        )
        
        if benchmark:
            if input_shapes is None:
                input_shapes = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            benchmark_results = optimizer.benchmark_model(
                input_shapes=input_shapes,
                num_iterations=num_benchmark_iterations
            )
            return benchmark_results
        
        return optimized_model
    
    elif optimization_type == "static_quantization":
        if calibration_dataset is None:
            raise ValueError("Calibration dataset is required for static quantization")
        
        optimized_model = optimizer.quantize_static(
            calibration_dataset=calibration_dataset,
            batch_size=batch_size,
            quantization_config=quantization_config,
            save_model=save_model
        )
        
        if benchmark:
            if input_shapes is None:
                input_shapes = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            benchmark_results = optimizer.benchmark_model(
                input_shapes=input_shapes,
                num_iterations=num_benchmark_iterations
            )
            return benchmark_results
        
        return optimized_model
    
    elif optimization_type in ["int8", "int4", "nf4"]:
        optimized_model = optimizer.quantize_with_bitsandbytes(
            quantization_type=optimization_type,
            save_model=save_model
        )
        
        if benchmark:
            if input_shapes is None:
                input_shapes = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            benchmark_results = optimizer.benchmark_model(
                input_shapes=input_shapes,
                num_iterations=num_benchmark_iterations
            )
            return benchmark_results
        
        return optimized_model
    
    elif optimization_type == "pruning":
        if pruning_config is None:
            pruning_config = {
                "method": "l1_unstructured",
                "amount": 0.3,
                "parameters_to_prune": ["weight"]
            }
        
        optimized_model = optimizer.prune_model(
            pruning_config=pruning_config,
            save_model=save_model
        )
        
        if benchmark:
            if input_shapes is None:
                input_shapes = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            benchmark_results = optimizer.benchmark_model(
                input_shapes=input_shapes,
                num_iterations=num_benchmark_iterations
            )
            return benchmark_results
        
        return optimized_model
    
    elif optimization_type == "onnx":
        if input_shapes is None:
            input_shapes = {"input_ids": [1, 128], "attention_mask": [1, 128]}
        
        if output_names is None:
            output_names = ["logits"]
        
        onnx_path = optimizer.export_to_onnx(
            input_shapes=input_shapes,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            optimize=optimize_onnx,
            quantize=quantize_onnx,
            save_model=save_model
        )
        
        if benchmark:
            benchmark_results = optimizer.benchmark_model(
                input_shapes=input_shapes,
                num_iterations=num_benchmark_iterations,
                compare_with_onnx=True,
                onnx_path=onnx_path
            )
            return benchmark_results
        
        return onnx_path
    
    else:
        raise ValueError(f"Unsupported optimization type: {optimization_type}")