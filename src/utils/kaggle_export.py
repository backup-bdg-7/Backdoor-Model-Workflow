"""
Utilities for exporting models to Kaggle datasets format.
"""

import os
import json
import shutil
import logging
import zipfile
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# Check if kaggle API is available
try:
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logger.warning("Kaggle API not available. Install with: pip install kaggle")


def prepare_for_kaggle(
    model_dir: str,
    output_dir: str,
    dataset_name: str,
    description: str = "Transformer model exported for Kaggle",
    keywords: List[str] = ["transformer", "nlp", "pytorch"],
    include_formats: List[str] = ["pytorch", "onnx", "torchscript", "coreml"],
    max_size_mb: int = 500
) -> str:
    """
    Prepare a model directory for upload to Kaggle Datasets.
    
    Args:
        model_dir: Directory containing exported model formats
        output_dir: Directory to create the Kaggle dataset
        dataset_name: Name for the Kaggle dataset (username/dataset-name)
        description: Description for the dataset
        keywords: List of keywords/tags for the dataset
        include_formats: List of model formats to include
        max_size_mb: Maximum size limit in MB (Kaggle has a 500MB limit for free accounts)
        
    Returns:
        Path to the prepared dataset directory
    """
    logger.info(f"Preparing model for Kaggle dataset: {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    # Create kaggle dataset metadata
    metadata = {
        "title": dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name,
        "id": dataset_name,
        "subtitle": "Transformer model exported with ModelExporter",
        "description": description,
        "keywords": keywords,
        "licenses": [{"name": "CC0-1.0"}],
        "resources": [],
        "version": 1
    }
    
    # Identify included formats
    format_dirs = {}
    for fmt in include_formats:
        fmt_dir = os.path.join(model_dir, fmt)
        if os.path.exists(fmt_dir):
            format_dirs[fmt] = fmt_dir
    
    if not format_dirs:
        raise ValueError(f"No valid model formats found in {model_dir}")
    
    # Create README.md
    readme_content = f"""# {metadata['title']}

{description}

## Available Model Formats

This dataset contains the model in various formats:

"""
    
    for fmt in format_dirs.keys():
        readme_content += f"### {fmt.upper()}\n\n"
        
        if fmt == "pytorch":
            readme_content += """```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model
model = AutoModelForCausalLM.from_pretrained("../input/{dataset_name}/pytorch")
tokenizer = AutoTokenizer.from_pretrained("../input/{dataset_name}/pytorch")

# Prepare input
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

"""
        
        elif fmt == "onnx":
            readme_content += """```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("../input/{dataset_name}/onnx/model.onnx")

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)  # Example input_ids

# Run inference
outputs = session.run(None, {input_name: input_data})
```

"""
        
        elif fmt == "torchscript":
            readme_content += """```python
import torch

# Load TorchScript model
model = torch.jit.load("../input/{dataset_name}/torchscript/model.pt")

# Prepare input
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Example input_ids

# Run inference
with torch.no_grad():
    outputs = model(input_ids)
```

"""
        
        elif fmt == "coreml":
            readme_content += """```python
# For macOS/iOS use only
import coremltools as ct

# Load CoreML model
model = ct.models.MLModel("../input/{dataset_name}/coreml/model.mlmodel")

# See the model inputs and outputs
print("Inputs:", model.input_description)
print("Outputs:", model.output_description)
```

For iOS/Swift integration, see the example_usage.swift file in the coreml directory.

"""
    
    readme_content += """## Citation

If you use this model in your research, please cite:

```
@misc{transformer_model,
  author = {Author},
  title = {Transformer Model},
  year = {2023},
  publisher = {Kaggle},
  howpublished = {\\url{https://kaggle.com/dataset/" + dataset_name + "}}
}
```
"""
    
    # Write README
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Copy model formats
    total_size = 0
    formats_copied = []
    
    for fmt, fmt_dir in format_dirs.items():
        # Get directory size
        dir_size = get_directory_size(fmt_dir) / (1024 * 1024)  # Convert to MB
        
        # Skip if adding this format would exceed the size limit
        if total_size + dir_size > max_size_mb:
            logger.warning(f"Skipping {fmt} format as it would exceed the {max_size_mb}MB limit")
            continue
        
        # Copy the directory
        dst_dir = os.path.join(output_dir, fmt)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        
        shutil.copytree(fmt_dir, dst_dir)
        total_size += dir_size
        formats_copied.append(fmt)
        
        logger.info(f"Copied {fmt} format ({dir_size:.1f}MB)")
    
    if not formats_copied:
        raise ValueError(f"No formats were copied. All formats exceed the {max_size_mb}MB limit.")
    
    # Update README with only the formats that were copied
    if set(formats_copied) != set(format_dirs.keys()):
        # Regenerate README with only the copied formats
        formats_section = "## Available Model Formats\n\nThis dataset contains the model in the following formats:\n\n"
        formats_section += ", ".join([f"**{fmt.upper()}**" for fmt in formats_copied])
        
        # Update the README
        with open(os.path.join(output_dir, "README.md"), "r") as f:
            content = f.read()
        
        # Replace the formats section
        start_marker = "## Available Model Formats"
        end_marker = "## Citation"
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            updated_content = content[:start_idx] + formats_section + "\n\n" + content[end_idx:]
            
            with open(os.path.join(output_dir, "README.md"), "w") as f:
                f.write(updated_content)
    
    # Write dataset-metadata.json for Kaggle
    metadata_path = os.path.join(output_dir, "dataset-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Prepared Kaggle dataset at {output_dir}")
    logger.info(f"Total size: {total_size:.1f}MB")
    logger.info(f"Formats included: {', '.join(formats_copied)}")
    
    return output_dir


def upload_to_kaggle(
    dataset_dir: str,
    dataset_name: Optional[str] = None,
    public: bool = False
) -> bool:
    """
    Upload a prepared dataset to Kaggle.
    
    Args:
        dataset_dir: Directory containing the prepared dataset
        dataset_name: Name for the Kaggle dataset (username/dataset-name)
                     If None, will read from dataset-metadata.json
        public: Whether to make the dataset public
        
    Returns:
        True if upload was successful
    """
    if not KAGGLE_AVAILABLE:
        raise ImportError("Kaggle API not available. Install with: pip install kaggle")
    
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Check for dataset-metadata.json
    metadata_path = os.path.join(dataset_dir, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        raise ValueError(f"dataset-metadata.json not found in {dataset_dir}")
    
    # Read dataset name from metadata if not provided
    if dataset_name is None:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "id" in metadata:
                dataset_name = metadata["id"]
            else:
                raise ValueError("Dataset name not provided and not found in metadata")
    
    logger.info(f"Uploading dataset to Kaggle: {dataset_name}")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Upload dataset
    try:
        api.dataset_create_new(
            folder=dataset_dir,
            public=public,
            convert_to_csv=False,
            dir_mode="zip"
        )
        logger.info(f"Successfully uploaded dataset to Kaggle: {dataset_name}")
        logger.info(f"View your dataset at: https://www.kaggle.com/datasets/{dataset_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload dataset to Kaggle: {e}")
        return False


def get_directory_size(path: str) -> int:
    """
    Get the total size of a directory in bytes.
    
    Args:
        path: Directory path
        
    Returns:
        Size in bytes
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def create_model_package(
    model_dir: str,
    output_path: str,
    readme_content: Optional[str] = None,
    include_formats: List[str] = ["pytorch", "onnx", "torchscript", "coreml"]
) -> str:
    """
    Create a compressed package (zip) of the model for easy sharing.
    
    Args:
        model_dir: Directory containing exported model formats
        output_path: Path to save the zip file
        readme_content: Optional content for README.md
        include_formats: List of model formats to include
        
    Returns:
        Path to the created zip file
    """
    logger.info(f"Creating model package at {output_path}")
    
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory does not exist: {model_dir}")
    
    # Create temporary directory to prepare the package
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy formats
        for fmt in include_formats:
            fmt_dir = os.path.join(model_dir, fmt)
            if os.path.exists(fmt_dir):
                dst_dir = os.path.join(tmpdir, fmt)
                shutil.copytree(fmt_dir, dst_dir)
                logger.info(f"Included {fmt} format in package")
        
        # Create README if content provided
        if readme_content:
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write(readme_content)
        
        # Create zip file
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmpdir)
                    zipf.write(file_path, arcname)
    
    logger.info(f"Created model package at {output_path}")
    return output_path
