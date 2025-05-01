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
Helper utilities for model exporting in various formats.
Provides simplified functions for exporting models to PyTorch, ONNX, CoreML, and other formats.
"""

import os
import logging
import tempfile
import platform
import subprocess
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import torch.nn as nn
import yaml
import json

# Configure logging
logger = logging.getLogger(__name__)

# Import optional dependencies
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Some export features will be limited.")

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. ONNX export will be disabled.")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    if platform.system() == "Darwin":  # Only warn on macOS
        logger.warning("CoreML tools not available. CoreML export will be disabled.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. TFLite export will be disabled.")

try:
    # Import our custom exporter
    from src.model.export import ModelExporter, export_model
    EXPORTER_AVAILABLE = True
except ImportError:
    EXPORTER_AVAILABLE = False
    logger.warning("Custom ModelExporter not available. Using simplified export.")


def export_to_all_formats(
    model: nn.Module,
    tokenizer: Any = None,
    output_dir: str = "./exported_models",
    input_shape: Tuple[int, int] = (1, 128),
    formats: List[str] = ["pytorch", "onnx", "coreml"],
    model_name: str = "transformer_model",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Export a model to multiple formats with a simple interface.
    
    Args:
        model: The PyTorch model to export
        tokenizer: Optional tokenizer (for HuggingFace models)
        output_dir: Directory to save exported models
        input_shape: Input shape for tracing (batch_size, seq_len)
        formats: List of formats to export to
        model_name: Name for the exported model
        metadata: Optional metadata to include with exported models
        
    Returns:
        Dictionary of export paths for each format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create default metadata if not provided
    if metadata is None:
        metadata = {
            "name": model_name,
            "description": "Transformer model exported from training pipeline",
            "input_shape": input_shape,
            "framework": "pytorch"
        }
    
    # Use our custom exporter if available
    if EXPORTER_AVAILABLE:
        logger.info("Using ModelExporter for comprehensive export")
        
        # Set up input shapes and output names
        input_shapes = {"input_ids": list(input_shape)}
        output_names = ["logits"]
        
        # Export using our full-featured exporter
        return export_model(
            model=model,
            tokenizer=tokenizer,
            formats=formats,
            input_shapes=input_shapes,
            output_names=output_names,
            output_dir=output_dir,
            metadata=metadata
        )
    
    # Otherwise fall back to simplified exports
    logger.info("Using simplified export functions")
    results = {}
    
    # Set model to evaluation mode
    model.eval()
    
    # Export to each requested format
    for fmt in formats:
        try:
            if fmt.lower() == "pytorch":
                path = export_to_pytorch(model, tokenizer, os.path.join(output_dir, "pytorch"), metadata)
                results["pytorch"] = path
            
            elif fmt.lower() == "onnx" and ONNX_AVAILABLE:
                path = export_to_onnx(model, input_shape, os.path.join(output_dir, "onnx"), metadata)
                results["onnx"] = path
            
            elif fmt.lower() == "torchscript":
                path = export_to_torchscript(model, input_shape, os.path.join(output_dir, "torchscript"), metadata)
                results["torchscript"] = path
            
            elif fmt.lower() == "coreml" and COREML_AVAILABLE:
                path = export_to_coreml(model, input_shape, os.path.join(output_dir, "coreml"), metadata)
                results["coreml"] = path
                
            elif fmt.lower() == "tflite" and TENSORFLOW_AVAILABLE:
                path = export_to_tflite(model, input_shape, os.path.join(output_dir, "tflite"), metadata)
                results["tflite"] = path
            
            else:
                logger.warning(f"Export format {fmt} not supported or dependencies missing")
        
        except Exception as e:
            logger.error(f"Error exporting to {fmt}: {e}")
    
    return results


def export_to_pytorch(
    model: nn.Module,
    tokenizer: Any = None,
    output_dir: str = "./pytorch_model",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export model to PyTorch format.
    
    Args:
        model: The model to export
        tokenizer: Optional tokenizer
        output_dir: Output directory
        metadata: Optional metadata
        
    Returns:
        Path to exported model
    """
    logger.info(f"Exporting model to PyTorch format at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HuggingFace model if it's a PreTrainedModel
    if TRANSFORMERS_AVAILABLE and isinstance(model, PreTrainedModel):
        logger.info("Saving as HuggingFace model")
        model.save_pretrained(output_dir)
        
        if tokenizer is not None and isinstance(tokenizer, PreTrainedTokenizer):
            tokenizer.save_pretrained(output_dir)
    
    # Otherwise save as standard PyTorch model
    else:
        logger.info("Saving as standard PyTorch model")
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save model definition if possible
        if hasattr(model, "__class__") and hasattr(model.__class__, "__name__"):
            model_info = {
                "class_name": model.__class__.__name__,
                "architecture": str(model),
            }
            with open(os.path.join(output_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
    
    # Save metadata if provided
    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, int] = (1, 128),
    output_dir: str = "./onnx_model",
    metadata: Optional[Dict[str, Any]] = None,
    opset_version: int = 15
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: The model to export
        input_shape: Input shape for tracing (batch_size, seq_len)
        output_dir: Output directory
        metadata: Optional metadata
        opset_version: ONNX opset version
        
    Returns:
        Path to exported model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available. Please install onnx and onnxruntime.")
    
    logger.info(f"Exporting model to ONNX format at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy input with the specified shape
    dummy_input = torch.zeros(input_shape, dtype=torch.long)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    # Verify the model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Save metadata if provided
    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir


def export_to_torchscript(
    model: nn.Module,
    input_shape: Tuple[int, int] = (1, 128),
    output_dir: str = "./torchscript_model",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: The model to export
        input_shape: Input shape for tracing (batch_size, seq_len)
        output_dir: Output directory
        metadata: Optional metadata
        
    Returns:
        Path to exported model
    """
    logger.info(f"Exporting model to TorchScript format at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input with the specified shape
    dummy_input = torch.zeros(input_shape, dtype=torch.long)
    
    # Try scripting first, fall back to tracing if that fails
    try:
        logger.info("Attempting to script the model")
        scripted_model = torch.jit.script(model)
        torchscript_path = os.path.join(output_dir, "model_scripted.pt")
        scripted_model.save(torchscript_path)
        logger.info("Successfully scripted the model")
    except Exception as e:
        logger.warning(f"Scripting failed: {e}. Falling back to tracing.")
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        torchscript_path = os.path.join(output_dir, "model_traced.pt")
        traced_model.save(torchscript_path)
        logger.info("Successfully traced the model")
    
    # Save metadata if provided
    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    return output_dir


def export_to_coreml(
    model: nn.Module,
    input_shape: Tuple[int, int] = (1, 128),
    output_dir: str = "./coreml_model",
    metadata: Optional[Dict[str, Any]] = None,
    compute_units: str = "ALL"
) -> str:
    """
    Export model to CoreML format.
    
    Args:
        model: The model to export
        input_shape: Input shape for tracing (batch_size, seq_len)
        output_dir: Output directory
        metadata: Optional metadata
        compute_units: CoreML compute units ("ALL", "CPU_ONLY", etc.)
        
    Returns:
        Path to exported model
    """
    if not COREML_AVAILABLE:
        raise ImportError("CoreML tools not available. Please install coremltools.")
    
    logger.info(f"Exporting model to CoreML format at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input with the specified shape
    dummy_input = torch.zeros(input_shape, dtype=torch.long)
    
    # First export to TorchScript as an intermediate step
    with tempfile.TemporaryDirectory() as tmpdir:
        torchscript_path = os.path.join(tmpdir, "model.pt")
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(torchscript_path)
        
        # Convert to CoreML using coremltools
        mlmodel_path = os.path.join(output_dir, "model.mlmodel")
        
        # Create input description
        inputs = [
            ct.TensorType(
                name="input_ids",
                shape=input_shape,
                dtype=ct.TensorType.INT32
            )
        ]
        
        # Convert the model
        coreml_model = ct.convert(
            model=torchscript_path,
            inputs=inputs,
            convert_to="mlprogram",
            compute_units=compute_units
        )
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    coreml_model.user_defined_metadata[key] = str(value)
        
        # Save the model
        coreml_model.save(mlmodel_path)
    
    # Save metadata to a separate file as well
    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Create a simple example Swift script for using the model
    with open(os.path.join(output_dir, "example_usage.swift"), "w") as f:
        f.write("""import CoreML

// Load the CoreML model
let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel")!
let compiledModelURL = try! MLModel.compileModel(at: modelURL)
let model = try! MLModel(contentsOf: compiledModelURL)

// Prepare input data
let inputShape = [1, 128] // [batch_size, sequence_length]
let inputArray = Array(repeating: Int32(0), count: inputShape.reduce(1, *))

// Create MLMultiArray for input
let inputMultiArray = try! MLMultiArray(shape: inputShape as [NSNumber], dataType: .int32)
for (index, value) in inputArray.enumerated() {
    inputMultiArray[index] = NSNumber(value: value)
}

// Create model input dictionary
let inputDict = ["input_ids": inputMultiArray]

// Run prediction
let output = try! model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))

// Extract and process output
let outputFeatures = output.featureValue(for: "logits")!
""")
    
    return output_dir


def export_to_tflite(
    model: nn.Module,
    input_shape: Tuple[int, int] = (1, 128),
    output_dir: str = "./tflite_model",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export model to TensorFlow Lite format (via ONNX).
    
    Args:
        model: The model to export
        input_shape: Input shape for tracing (batch_size, seq_len)
        output_dir: Output directory
        metadata: Optional metadata
        
    Returns:
        Path to exported model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available. Please install onnx and onnxruntime.")
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available. Please install tensorflow.")
    
    logger.info(f"Exporting model to TensorFlow Lite format at {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # First export to ONNX as an intermediate format
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to ONNX
        onnx_path = os.path.join(tmpdir, "model.onnx")
        dummy_input = torch.zeros(input_shape, dtype=torch.long)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=12,
            do_constant_folding=True
        )
        
        # Convert ONNX to TensorFlow
        import tensorflow as tf
        import tf2onnx.convert
        
        tf_model_path = os.path.join(tmpdir, "tf_model")
        
        # Convert ONNX to TensorFlow SavedModel
        tf_rep = tf2onnx.convert.from_onnx(onnx.load(onnx_path))
        tf_rep.export_graph(tf_model_path)
        
        # Convert TensorFlow SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert to TFLite model
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(output_dir, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    
    # Save metadata if provided
    if metadata:
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Create example Python code for using the TFLite model
    with open(os.path.join(output_dir, "example_usage.py"), "w") as f:
        f.write("""import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data
input_shape = input_details[0]['shape']
input_data = np.zeros(input_shape, dtype=np.int32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
""")
    
    return output_dir


def check_export_compatibility(model: nn.Module) -> Dict[str, bool]:
    """
    Check which export formats are compatible with the model.
    
    Args:
        model: The model to check
        
    Returns:
        Dictionary mapping format names to compatibility boolean
    """
    # Set up result dictionary
    compatibility = {
        "pytorch": True,  # PyTorch models are always compatible with PyTorch export
        "torchscript": False,
        "onnx": False,
        "coreml": False,
        "tflite": False
    }
    
    # Set model to evaluation mode for testing
    model.eval()
    
    # Create a small dummy input
    dummy_input = torch.zeros((1, 16), dtype=torch.long)
    
    # Test TorchScript compatibility
    try:
        # Test tracing
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model(dummy_input)  # Run a forward pass
        compatibility["torchscript"] = True
    except Exception as e:
        logger.warning(f"TorchScript tracing is not compatible: {e}")
        
        # Try scripting as a fallback
        try:
            scripted_model = torch.jit.script(model)
            scripted_model(dummy_input)  # Run a forward pass
            compatibility["torchscript"] = True
        except Exception as e:
            logger.warning(f"TorchScript scripting is also not compatible: {e}")
    
    # Test ONNX compatibility
    if ONNX_AVAILABLE:
        try:
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                torch.onnx.export(
                    model,
                    dummy_input,
                    tmp.name,
                    input_names=["input_ids"],
                    output_names=["logits"],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'logits': {0: 'batch_size', 1: 'sequence_length'}
                    },
                    opset_version=12
                )
                onnx_model = onnx.load(tmp.name)
                onnx.checker.check_model(onnx_model)
                compatibility["onnx"] = True
        except Exception as e:
            logger.warning(f"ONNX export is not compatible: {e}")
    else:
        logger.warning("ONNX is not available")
    
    # CoreML compatibility depends on both TorchScript and platform
    if COREML_AVAILABLE and compatibility["torchscript"]:
        try:
            if platform.system() == "Darwin":  # macOS only
                compatibility["coreml"] = True
        except Exception as e:
            logger.warning(f"CoreML conversion might not be compatible: {e}")
    
    # TFLite compatibility depends on ONNX compatibility
    if TENSORFLOW_AVAILABLE and compatibility["onnx"]:
        compatibility["tflite"] = True
    
    return compatibility


def get_export_size_estimates(model: nn.Module, input_shape: Tuple[int, int] = (1, 128)) -> Dict[str, str]:
    """
    Get estimated file sizes for each export format.
    
    Args:
        model: The model to check
        input_shape: Input shape for tracing
        
    Returns:
        Dictionary mapping format names to file size estimates
    """
    # Get model size in parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Rough estimates based on parameter count and empirical observations
    # These are very rough approximations
    formats = {
        "pytorch": f"{num_params * 4 / (1024*1024):.1f} MB",  # ~4 bytes per parameter
        "torchscript": f"{num_params * 4 / (1024*1024):.1f} MB",  # Similar to PyTorch
        "onnx": f"{num_params * 4.5 / (1024*1024):.1f} MB",  # Slightly larger due to graph info
        "coreml": f"{num_params * 5 / (1024*1024):.1f} MB",  # CoreML adds additional metadata
        "tflite": f"{num_params * 4.2 / (1024*1024):.1f} MB"  # TFLite is often similar to ONNX
    }
    
    return formats
