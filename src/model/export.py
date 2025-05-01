"""
Model export utilities for the AI model training workflow.
This module provides functions for exporting models to various formats including ONNX, TorchScript, and CoreML.
"""

import os
import logging
import json
import time
import platform
import shutil
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
import torch.nn as nn
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import onnx
    import onnxruntime as ort
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
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ModelExporter:
    """
    A class to handle exporting models to various formats.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any = None,
        device: Optional[torch.device] = None,
        output_dir: str = "./exported_models",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the model exporter.
        
        Args:
            model: Model to export
            tokenizer: Tokenizer for the model
            device: Device to use for export
            output_dir: Directory to save exported models
            config: Additional configuration options
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def export_all(
        self,
        input_shapes: Dict[str, List[int]],
        output_names: List[str],
        formats: List[str] = ["pytorch", "onnx", "torchscript", "coreml"],
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export model to multiple formats.
        
        Args:
            input_shapes: Dictionary of input names and shapes
            output_names: List of output names
            formats: List of formats to export to
            sample_inputs: Sample inputs for tracing (optional)
            metadata: Metadata to include with the models
            
        Returns:
            Dictionary mapping format names to export paths
        """
        logger.info(f"Exporting model to formats: {formats}")
        
        # Create sample inputs if not provided
        if sample_inputs is None:
            sample_inputs = {}
            for name, shape in input_shapes.items():
                sample_inputs[name] = torch.randint(0, 100, shape, dtype=torch.long, device=self.device)
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {
                "description": "Model exported with ModelExporter",
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_shapes": str(input_shapes),
                "output_names": output_names
            }
        
        # Export to each format
        export_paths = {}
        
        for fmt in formats:
            try:
                if fmt.lower() == "pytorch":
                    path = self.export_to_pytorch(metadata=metadata)
                    export_paths["pytorch"] = path
                
                elif fmt.lower() == "onnx":
                    if not ONNX_AVAILABLE:
                        logger.warning("ONNX not available. Skipping ONNX export.")
                        continue
                    
                    path = self.export_to_onnx(
                        input_shapes=input_shapes,
                        output_names=output_names,
                        sample_inputs=sample_inputs,
                        metadata=metadata
                    )
                    export_paths["onnx"] = path
                
                elif fmt.lower() == "torchscript":
                    path = self.export_to_torchscript(
                        sample_inputs=sample_inputs,
                        metadata=metadata
                    )
                    export_paths["torchscript"] = path
                
                elif fmt.lower() == "coreml":
                    if not COREML_AVAILABLE:
                        logger.warning("CoreML tools not available. Skipping CoreML export.")
                        continue
                    
                    path = self.export_to_coreml(
                        input_shapes=input_shapes,
                        output_names=output_names,
                        sample_inputs=sample_inputs,
                        metadata=metadata
                    )
                    export_paths["coreml"] = path
                
                else:
                    logger.warning(f"Unsupported export format: {fmt}")
            
            except Exception as e:
                logger.error(f"Error exporting to {fmt} format: {e}")
        
        # Print export summary
        logger.info("Export completed successfully")
        for fmt, path in export_paths.items():
            logger.info(f"  {fmt}: {path}")
        
        return export_paths
    
    def export_to_pytorch(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Export model to PyTorch format.
        
        Args:
            metadata: Metadata to include with the model
            
        Returns:
            Path to the exported model directory
        """
        logger.info("Exporting model to PyTorch format")
        
        # Create export directory
        export_dir = os.path.join(self.output_dir, "pytorch")
        os.makedirs(export_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(export_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save model architecture if available
        if hasattr(self.model, "config"):
            config_path = os.path.join(export_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(self.model.config.to_dict(), f, indent=2)
        
        # Save using save_pretrained if available
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(export_dir)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(export_dir)
            else:
                tokenizer_path = os.path.join(export_dir, "tokenizer.json")
                if hasattr(self.tokenizer, "to_json"):
                    with open(tokenizer_path, "w") as f:
                        f.write(self.tokenizer.to_json())
        
        # Save metadata
        if metadata:
            metadata_path = os.path.join(export_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Save readme with usage example
        readme_path = os.path.join(export_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# PyTorch Model

This directory contains a PyTorch model exported using ModelExporter.

## Loading the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{export_dir}")
tokenizer = AutoTokenizer.from_pretrained("{export_dir}")

# Example usage
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)
```
""")
        
        logger.info(f"Exported PyTorch model to {export_dir}")
        return export_dir
    
    def export_to_onnx(
        self,
        input_shapes: Dict[str, List[int]],
        output_names: List[str],
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 15,
        optimize: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            input_shapes: Dictionary of input names and shapes
            output_names: List of output names
            sample_inputs: Sample inputs for tracing
            dynamic_axes: Dictionary of dynamic axes
            opset_version: ONNX opset version
            optimize: Whether to optimize the ONNX model
            metadata: Metadata to include with the model
            
        Returns:
            Path to the exported ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is not available. Please install with: pip install onnx onnxruntime")
        
        logger.info("Exporting model to ONNX format")
        
        # Create export directory
        export_dir = os.path.join(self.output_dir, "onnx")
        os.makedirs(export_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create dummy inputs if not provided
        if sample_inputs is None:
            sample_inputs = {}
            for name, shape in input_shapes.items():
                sample_inputs[name] = torch.randint(0, 100, shape, dtype=torch.long, device=self.device)
        
        # Convert inputs to a tuple for onnx export
        input_tuple = tuple(sample_inputs.values())
        input_names = list(sample_inputs.keys())
        
        # Set dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {}
            for name in input_names:
                dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        
        # Export to ONNX
        onnx_path = os.path.join(export_dir, "model.onnx")
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                input_tuple,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
        
        # Optimize model if requested
        if optimize:
            try:
                from onnxruntime.transformers import optimizer
                from onnx import load_model, save_model
                
                # Load model
                onnx_model = load_model(onnx_path)
                
                # Optimize model
                model_type = "auto"
                if hasattr(self.model, "config"):
                    if hasattr(self.model.config, "model_type"):
                        model_type = self.model.config.model_type
                
                # Get number of attention heads and hidden size
                num_heads = 12
                hidden_size = 768
                if hasattr(self.model, "config"):
                    if hasattr(self.model.config, "n_head"):
                        num_heads = self.model.config.n_head
                    elif hasattr(self.model.config, "num_attention_heads"):
                        num_heads = self.model.config.num_attention_heads
                    
                    if hasattr(self.model.config, "hidden_size"):
                        hidden_size = self.model.config.hidden_size
                    elif hasattr(self.model.config, "n_embd"):
                        hidden_size = self.model.config.n_embd
                
                # Optimize for transformer models
                optimized_model = optimizer.optimize_model(
                    onnx_path,
                    model_type=model_type,
                    num_heads=num_heads,
                    hidden_size=hidden_size
                )
                
                # Save optimized model
                optimized_path = os.path.join(export_dir, "model_optimized.onnx")
                optimized_model.save_model_to_file(optimized_path)
                
                # Use optimized model
                onnx_path = optimized_path
                logger.info(f"ONNX model optimized and saved to {optimized_path}")
            except Exception as e:
                logger.warning(f"Failed to optimize ONNX model: {e}")
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(export_dir)
        
        # Save metadata
        if metadata:
            metadata_path = os.path.join(export_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Save readme with usage example
        readme_path = os.path.join(export_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# ONNX Model

This directory contains an ONNX model exported using ModelExporter.

## Loading the Model

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("{os.path.basename(onnx_path)}")

# Prepare inputs (example)
inputs = {{
    "{input_names[0]}": np.random.randint(0, 100, size={input_shapes[input_names[0]]}, dtype=np.int64)
}}

# Run inference
outputs = session.run(None, inputs)
```
""")
        
        logger.info(f"Exported ONNX model to {onnx_path}")
        return export_dir
    
    def export_to_torchscript(
        self,
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
        use_tracing: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export model to TorchScript format.
        
        Args:
            sample_inputs: Sample inputs for tracing
            use_tracing: Whether to use tracing (vs scripting)
            metadata: Metadata to include with the model
            
        Returns:
            Path to the exported TorchScript model
        """
        logger.info("Exporting model to TorchScript format")
        
        # Create export directory
        export_dir = os.path.join(self.output_dir, "torchscript")
        os.makedirs(export_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create traced/scripted model
        if use_tracing:
            # Ensure sample inputs is provided for tracing
            if sample_inputs is None:
                raise ValueError("sample_inputs must be provided for tracing")
            
            # Convert dictionary to kwargs
            sample_inputs_dict = {k: v for k, v in sample_inputs.items()}
            
            # Create a wrapper module if needed
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, *args, **kwargs):
                    return self.model(*args, **kwargs)
            
            wrapper = ModelWrapper(self.model)
            
            # Trace the model with dummy inputs
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    func=wrapper,
                    example_inputs=tuple(sample_inputs.values())
                )
                
                torchscript_model = traced_model
        else:
            # Use scripting (doesn't require sample inputs)
            # This might fail for complex models with control flow
            try:
                torchscript_model = torch.jit.script(self.model)
            except Exception as e:
                logger.error(f"Failed to script model: {e}")
                logger.info("Falling back to tracing...")
                
                # Fall back to tracing if scripting fails
                if sample_inputs is None:
                    raise ValueError("sample_inputs must be provided for tracing fallback")
                
                with torch.no_grad():
                    torchscript_model = torch.jit.trace(
                        func=self.model,
                        example_inputs=tuple(sample_inputs.values())
                    )
        
        # Save TorchScript model
        torchscript_path = os.path.join(export_dir, "model.pt")
        torch.jit.save(torchscript_model, torchscript_path)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(export_dir)
        
        # Save metadata
        if metadata:
            metadata_path = os.path.join(export_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Save readme with usage example
        readme_path = os.path.join(export_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# TorchScript Model

This directory contains a TorchScript model exported using ModelExporter.

## Loading the Model

```python
import torch

# Load TorchScript model
model = torch.jit.load("model.pt")

# Prepare inputs (example)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Run inference
with torch.no_grad():
    outputs = model(input_ids)
```
""")
        
        logger.info(f"Exported TorchScript model to {torchscript_path}")
        return export_dir
    
    def export_to_coreml(
        self,
        input_shapes: Dict[str, List[int]],
        output_names: List[str],
        sample_inputs: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        minimum_deployment_target: Optional[str] = None,
        compute_units: str = "ALL"
    ) -> str:
        """
        Export model to CoreML format for iOS/macOS deployment.
        
        Args:
            input_shapes: Dictionary of input names and shapes
            output_names: List of output names
            sample_inputs: Sample inputs for tracing
            metadata: Metadata to include with the model
            minimum_deployment_target: Minimum iOS/macOS version to target
            compute_units: CoreML compute units ("ALL", "CPU_ONLY", etc.)
            
        Returns:
            Path to the exported CoreML model
        """
        if not COREML_AVAILABLE:
            raise ImportError("CoreML tools not available. Please install with: pip install coremltools")
        
        logger.info("Exporting model to CoreML format")
        
        # Create export directory
        export_dir = os.path.join(self.output_dir, "coreml")
        os.makedirs(export_dir, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create sample inputs if not provided
        if sample_inputs is None:
            sample_inputs = {}
            for name, shape in input_shapes.items():
                sample_inputs[name] = torch.randint(0, 100, shape, dtype=torch.long, device=self.device)
        
        # Create a trace of the model using sample inputs
        try:
            # First, try to trace the model directly
            traced_model = torch.jit.trace(
                func=self.model, 
                example_inputs=tuple(sample_inputs.values())
            )
        except Exception as e:
            logger.warning(f"Direct tracing failed: {e}")
            logger.info("Trying to trace with a wrapper function...")
            
            # If direct tracing fails, try with a wrapper
            def model_forward(*inputs):
                return self.model(*inputs)
            
            traced_model = torch.jit.trace(
                func=model_forward,
                example_inputs=tuple(sample_inputs.values())
            )
        
        # Set up input and output descriptions
        input_descriptions = {}
        for name, shape in input_shapes.items():
            input_descriptions[name] = f"Input tensor with shape {shape}"
        
        output_descriptions = {}
        for name in output_names:
            output_descriptions[name] = f"Output tensor named {name}"
        
        # Prepare metadata for CoreML
        coreml_metadata = {
            "com.apple.coreml.model.description": "Model exported with ModelExporter",
            "com.apple.coreml.model.license": "Open source",
            "com.apple.coreml.model.author": "ModelExporter",
            "com.apple.coreml.model.version": "1.0",
        }
        
        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    coreml_metadata[f"com.apple.coreml.model.{key}"] = str(value)
        
        # Set deployment target based on model size and features
        if minimum_deployment_target is None:
            minimum_deployment_target = ct.target.iOS14
        
        # Convert to CoreML format
        try:
            # Convert using coremltools
            coreml_model = ct.convert(
                model=traced_model,
                inputs=[
                    ct.TensorType(name=name, shape=sample_inputs[name].shape, dtype=ct.TensorType.INT32)
                    for name in sample_inputs.keys()
                ],
                minimum_deployment_target=minimum_deployment_target,
                compute_units=compute_units,
                convert_to="mlprogram"  # Use mlprogram for newer deployment targets
            )
            
            # Set metadata
            for key, value in coreml_metadata.items():
                coreml_model.user_defined_metadata[key] = value
            
            # Set descriptions
            for name, desc in input_descriptions.items():
                input_spec = coreml_model.get_spec().description.input
                for input_desc in input_spec:
                    if input_desc.name == name:
                        input_desc.shortDescription = desc
            
            for i, name in enumerate(output_names):
                if i < len(coreml_model.get_spec().description.output):
                    output_spec = coreml_model.get_spec().description.output
                    output_spec[i].shortDescription = output_descriptions.get(name, "")
            
            # Save CoreML model
            mlmodel_path = os.path.join(export_dir, "model.mlmodel")
            mlpackage_path = os.path.join(export_dir, "model.mlpackage")
            
            # Save as .mlmodel
            coreml_model.save(mlmodel_path)
            
            # Also save as .mlpackage for newer systems
            try:
                coreml_model.save(mlpackage_path)
            except Exception as e:
                logger.warning(f"Failed to save as .mlpackage: {e}")
        
        except Exception as e:
            logger.error(f"CoreML conversion failed: {e}")
            logger.info("Trying alternative conversion approach...")
            
            # Alternative approach: ONNX -> CoreML
            try:
                # First export to ONNX
                onnx_path = os.path.join(export_dir, "temp_model.onnx")
                with torch.no_grad():
                    torch.onnx.export(
                        self.model,
                        tuple(sample_inputs.values()),
                        onnx_path,
                        input_names=list(sample_inputs.keys()),
                        output_names=output_names,
                        opset_version=12
                    )
                
                # Then convert ONNX to CoreML
                coreml_model = ct.convert(
                    model=onnx_path,
                    minimum_deployment_target=minimum_deployment_target,
                    compute_units=compute_units
                )
                
                # Save CoreML model
                mlmodel_path = os.path.join(export_dir, "model.mlmodel")
                coreml_model.save(mlmodel_path)
                
                # Clean up temporary ONNX file
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
                
            except Exception as e2:
                logger.error(f"Alternative CoreML conversion also failed: {e2}")
                raise ValueError(f"All CoreML conversion approaches failed: {e}, {e2}")
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(export_dir)
                
                # Also save a simple JSON vocabulary for iOS use
                try:
                    vocab_path = os.path.join(export_dir, "vocabulary.json")
                    with open(vocab_path, "w") as f:
                        json.dump(self.tokenizer.get_vocab(), f)
                except Exception as e:
                    logger.warning(f"Failed to save vocabulary JSON: {e}")
        
        # Save metadata
        if metadata:
            metadata_path = os.path.join(export_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        # Create README with usage instructions
        readme_path = os.path.join(export_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(f"""# CoreML Model

This directory contains a CoreML model exported using ModelExporter.

## iOS Integration

### Swift Usage Example

```swift
import CoreML

// Load the model
guard let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel") else {
    fatalError("Failed to find model in bundle")
}

let model: MLModel
do {{
    let compiledModelURL = try MLModel.compileModel(at: modelURL)
    model = try MLModel(contentsOf: compiledModelURL)
}} catch {{
    fatalError("Failed to load CoreML model")
}}

// Prepare input
let inputName = "{list(input_shapes.keys())[0]}"
let shape: [NSNumber] = {[int(x) for x in input_shapes[list(input_shapes.keys())[0]]]}
let inputArray = [Int32](repeating: 0, count: shape.reduce(1, *))

// Create input MLMultiArray
let inputMultiArray: MLMultiArray
do {{
    inputMultiArray = try MLMultiArray(shape: shape, dataType: .int32)
    
    // Fill with data
    for (index, value) in inputArray.enumerated() {{
        inputMultiArray[index] = NSNumber(value: value)
    }}
}} catch {{
    fatalError("Failed to create input MLMultiArray")
}}

// Create model input
let input = try! MLDictionaryFeatureProvider(dictionary: [inputName: inputMultiArray])

// Get prediction
guard let output = try? model.prediction(from: input) else {{
    fatalError("Failed to get prediction")
}}

// Process the output
let outputFeatures = output.featureValue(for: "{output_names[0]}")
// Handle outputFeatures based on your model's output format
```
""")
        
        logger.info(f"Exported CoreML model to {export_dir}")
        return export_dir
    
    def create_model_card(self, export_dir: str, formats: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a comprehensive model card.
        
        Args:
            export_dir: Directory where exports are stored
            formats: List of exported formats
            metadata: Metadata for the model
            
        Returns:
            Path to the model card file
        """
        logger.info("Creating model card")
        
        # Create or extract metadata
        if metadata is None:
            metadata = {
                "name": "Exported Model",
                "description": "Model exported with ModelExporter",
                "author": "ModelExporter",
                "license": "MIT",
                "date": time.strftime("%Y-%m-%d")
            }
        
        # Get model details if available
        model_details = {}
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "to_dict"):
                model_details = config.to_dict()
            else:
                # Extract commonly used attributes
                for attr in ["hidden_size", "num_hidden_layers", "num_attention_heads", 
                            "vocab_size", "model_type", "n_layer", "n_head", "n_embd"]:
                    if hasattr(config, attr):
                        model_details[attr] = getattr(config, attr)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Create the model card content
        model_card = f"""# {metadata.get('name', 'Exported Model')}

## Model Description

{metadata.get('description', 'A model exported with ModelExporter')}

### Model Details

- **Author:** {metadata.get('author', 'ModelExporter')}
- **Date:** {metadata.get('date', time.strftime("%Y-%m-%d"))}
- **License:** {metadata.get('license', 'MIT')}
- **Total Parameters:** {total_params:,}
- **Trainable Parameters:** {trainable_params:,} ({trainable_params/total_params:.2%})

"""
        
        # Add model architecture details if available
        if model_details:
            model_card += "### Architecture\n\n"
            for key, value in model_details.items():
                if isinstance(value, (int, float, str, bool)):
                    model_card += f"- **{key}:** {value}\n"
            model_card += "\n"
        
        # Add available formats
        model_card += "## Available Formats\n\n"
        for fmt in formats:
            if fmt == "pytorch":
                model_card += "### PyTorch\n\n"
                model_card += "```python\n"
                model_card += "import torch\n"
                model_card += "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
                model_card += "# Load model and tokenizer\n"
                model_card += "model = AutoModelForCausalLM.from_pretrained('pytorch/')\n"
                model_card += "tokenizer = AutoTokenizer.from_pretrained('pytorch/')\n\n"
                model_card += "# Example usage\n"
                model_card += "inputs = tokenizer('Hello, world!', return_tensors='pt')\n"
                model_card += "outputs = model(**inputs)\n"
                model_card += "```\n\n"
            
            elif fmt == "onnx":
                model_card += "### ONNX\n\n"
                model_card += "```python\n"
                model_card += "import onnxruntime as ort\n"
                model_card += "import numpy as np\n\n"
                model_card += "# Load ONNX model\n"
                model_card += "session = ort.InferenceSession('onnx/model.onnx')\n\n"
                model_card += "# Prepare inputs\n"
                model_card += "inputs = {'input_ids': np.array([[1, 2, 3, 4, 5]], dtype=np.int64)}\n\n"
                model_card += "# Run inference\n"
                model_card += "outputs = session.run(None, inputs)\n"
                model_card += "```\n\n"
            
            elif fmt == "torchscript":
                model_card += "### TorchScript\n\n"
                model_card += "```python\n"
                model_card += "import torch\n\n"
                model_card += "# Load TorchScript model\n"
                model_card += "model = torch.jit.load('torchscript/model.pt')\n\n"
                model_card += "# Prepare inputs\n"
                model_card += "inputs = torch.tensor([[1, 2, 3, 4, 5]])\n\n"
                model_card += "# Run inference\n"
                model_card += "with torch.no_grad():\n"
                model_card += "    outputs = model(inputs)\n"
                model_card += "```\n\n"
            
            elif fmt == "coreml":
                model_card += "### CoreML\n\n"
                model_card += "```swift\n"
                model_card += "import CoreML\n\n"
                model_card += "// Load the model\n"
                model_card += "guard let modelURL = Bundle.main.url(forResource: \"model\", withExtension: \"mlmodel\") else {\n"
                model_card += "    fatalError(\"Failed to find model in bundle\")\n"
                model_card += "}\n\n"
                model_card += "let model: MLModel\n"
                model_card += "do {\n"
                model_card += "    model = try MLModel(contentsOf: modelURL)\n"
                model_card += "} catch {\n"
                model_card += "    fatalError(\"Failed to load CoreML model\")\n"
                model_card += "}\n\n"
                model_card += "// Prepare input and run inference\n"
                model_card += "// See the README in the coreml/ directory for detailed examples\n"
                model_card += "```\n\n"
        
        # Write the model card to file
        model_card_path = os.path.join(export_dir, "MODEL_CARD.md")
        with open(model_card_path, "w") as f:
            f.write(model_card)
        
        return model_card_path


def export_model(
    model: nn.Module,
    tokenizer: Any = None,
    formats: List[str] = ["pytorch", "onnx", "torchscript", "coreml"],
    input_shapes: Optional[Dict[str, List[int]]] = None,
    output_names: Optional[List[str]] = None,
    output_dir: str = "./exported_models",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Convenience function to export a model to various formats.
    
    Args:
        model: Model to export
        tokenizer: Tokenizer for the model
        formats: Formats to export to
        input_shapes: Dictionary of input names and shapes
        output_names: List of output names
        output_dir: Directory to save exported models
        metadata: Metadata for the model
        
    Returns:
        Dictionary mapping format names to export paths
    """
    # Set default input shapes and output names if not provided
    if input_shapes is None:
        input_shapes = {"input_ids": [1, 32]}
    
    if output_names is None:
        output_names = ["logits"]
    
    # Create exporter and export
    exporter = ModelExporter(model, tokenizer, output_dir=output_dir)
    export_paths = exporter.export_all(
        input_shapes=input_shapes,
        output_names=output_names,
        formats=formats,
        metadata=metadata
    )
    
    # Create model card
    exporter.create_model_card(output_dir, formats, metadata)
    
    return export_paths
