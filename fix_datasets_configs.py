import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the dataset configuration
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '"datasets": [' in ''.join(cell['source']):
        # Get the current source
        source = ''.join(cell['source'])
        
        # Fix the dataset configurations with correct configs and splits
        # These are the changes we need to make:
        # 1. nvidia/OpenCodeReasoning needs a config parameter 'split_0'
        source = source.replace('{"name": "nvidia/OpenCodeReasoning", "split": "train"', '{"name": "nvidia/OpenCodeReasoning", "subset": "split_0", "split": "train"')
        
        # 2. openai/openai_humaneval uses 'test' split, not 'train'
        source = source.replace('{"name": "openai/openai_humaneval", "split": "train"', '{"name": "openai/openai_humaneval", "split": "test"')
        
        # 3. HuggingFaceH4/instruction-dataset uses 'test' split, not 'train'
        source = source.replace('{"name": "HuggingFaceH4/instruction-dataset", "split": "train"', '{"name": "HuggingFaceH4/instruction-dataset", "split": "test"')
        
        # 4. HuggingFaceH4/ultrachat_200k needs to specify the correct split 'train_sft'
        source = source.replace('{"name": "HuggingFaceH4/ultrachat_200k", "split": "train"', '{"name": "HuggingFaceH4/ultrachat_200k", "split": "train_sft"')
        
        # 5. For evaluation, use test_sft split
        source = source.replace('{"name": "HuggingFaceH4/ultrachat_200k", "split": "test"', '{"name": "HuggingFaceH4/ultrachat_200k", "split": "test_sft"')
        
        # 6. togethercomputer/RedPajama-Data-1T needs a config
        source = source.replace('{"name": "togethercomputer/RedPajama-Data-1T", "split": "train"', '{"name": "togethercomputer/RedPajama-Data-1T", "subset": "default", "split": "train"')
        
        # 7. allenai/c4 needs a config
        source = source.replace('{"name": "allenai/c4", "split": "train"', '{"name": "allenai/c4", "subset": "en", "split": "train"')
        
        # 8. bigscience/xP3 needs a config
        source = source.replace('{"name": "bigscience/xP3", "split": "train"', '{"name": "bigscience/xP3", "subset": "en", "split": "train"')
        
        # 9. HuggingFaceTB/cosmopedia needs a config
        source = source.replace('{"name": "HuggingFaceTB/cosmopedia", "split": "train"', '{"name": "HuggingFaceTB/cosmopedia", "subset": "web_samples_v2", "split": "train"')
        
        # Replace the cell source
        notebook['cells'][i]['source'] = [source]
        print(f"Updated dataset configurations in cell {i}")

# Update the dependency installation to handle fsspec conflict
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'pip install' in ''.join(cell['source']):
        # Add a better dependency installation approach
        new_source = [
            "# Clone repository and install dependencies\n",
            "!git clone https://github.com/backup-bdg-6/datasets.git\n",
            "%cd datasets\n",
            "\n",
            "# Install core dependencies\n",
            "!pip install rouge\n",
            "!pip install rouge_score\n",
            "# Install packages with specific versions to avoid conflicts\n",
            "!pip install -q fsspec==2025.3.0\n",  # Pin fsspec to a version datasets accepts
            "!pip install -q datasets>=2.10.0\n",  # Install datasets first with compatible fsspec
            "!pip install -q torch>=1.13.0 transformers>=4.30.0 accelerate>=0.20.0 bitsandbytes>=0.39.0\n",
            "!pip install -q optimum>=1.8.0 onnx>=1.13.0 onnxruntime>=1.14.0 huggingface_hub>=0.16.0 safetensors>=0.3.1\n",
            "!pip install -q pyyaml tqdm matplotlib numpy scikit-learn sentencepiece protobuf>=3.20.0,<4.0.0\n",
            "!pip install -q onnx2torch>=1.5.2 onnxsim>=0.4.21\n",
            "\n",
            "# Install gcsfs with compatible fsspec\n",
            "!pip install -q gcsfs\n",
            "\n",
            "# Install CoreML tools for macOS\n",
            "import platform\n",
            "IS_MACOS = platform.system() == \"Darwin\"\n",
            "if IS_MACOS:\n",
            "    !pip install -q coremltools\n",
            "    if platform.machine() == \"arm64\":  # Apple Silicon\n",
            "        !pip install -q tensorflow-macos tensorflow-metal\n",
            "    else:  # Intel Mac\n",
            "        !pip install -q tensorflow\n",
            "else:  # Linux/Windows\n",
            "    !pip install -q tensorflow\n",
            "\n",
            "print(\"✅ Required dependencies installed\")"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated dependency installation in cell {i}")

# Now, let's fix the load_datasets_from_config function to better handle streaming datasets
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'def load_datasets_from_config' in ''.join(cell['source']):
        # Update the function to better handle streaming datasets and trust_remote_code
        source = ''.join(cell['source'])
        
        # Add trust_remote_code parameter
        source = source.replace('def load_datasets_from_config(config, auth_token=None):', 
                                'def load_datasets_from_config(config, auth_token=None):')
        
        # Add trust_remote_code to load_dataset call
        source = source.replace('dataset = load_dataset(name, subset, split=split, streaming=streaming, **download_params)',
                               'dataset = load_dataset(name, subset, split=split, streaming=streaming, trust_remote_code=True, **download_params)')
        source = source.replace('dataset = load_dataset(name, split=split, streaming=streaming, **download_params)',
                               'dataset = load_dataset(name, split=split, streaming=streaming, trust_remote_code=True, **download_params)')
        
        # Fix handling of streaming datasets len() issue
        source = source.replace('if not streaming and max_samples and len(dataset) > max_samples:',
                               'if not streaming and max_samples and (not hasattr(dataset, "__iter__") and len(dataset) > max_samples):')
        
        source = source.replace('if not streaming:',
                               'if not streaming and not hasattr(dataset, "__iter__"):')
        
        # Replace the cell source
        notebook['cells'][i]['source'] = [source]
        print(f"Updated load_datasets_from_config function in cell {i}")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully")
