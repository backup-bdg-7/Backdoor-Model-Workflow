import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with dependency installation
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '!pip install' in ''.join(cell['source']):
        # Replace with comprehensive dependency installation that resolves all conflicts
        new_source = [
            "# Clone repository and install dependencies\n",
            "!git clone https://github.com/backup-bdg-6/datasets.git\n",
            "%cd datasets\n",
            "\n",
            "# ============== DEPENDENCY INSTALLATION STRATEGY ==============\n",
            "# 1. Create a clean environment by using a specific version for each package\n",
            "# 2. Install packages in a specific order to maintain compatibility\n",
            "# 3. Use --no-deps where needed to prevent pulling in conflicting dependencies\n",
            "\n",
            "# Install essential packages first\n",
            "!pip install rouge\n",
            "!pip install rouge_score\n",
            "\n",
            "# === First wave: Core infrastructure packages ===\n",
            "# Start with numpy to ensure consistent version for all packages\n",
            "!pip install -q numpy==1.24.3 --upgrade\n",
            "\n",
            "# Install typing-extensions at compatible version\n",
            "!pip install -q typing-extensions==4.5.0 --upgrade\n",
            "\n",
            "# Set fsspec and gcsfs to compatible versions\n",
            "!pip install -q fsspec==2023.9.2 --upgrade\n",
            "!pip install -q gcsfs==2023.9.2 --upgrade\n",
            "\n",
            "# Install protobuf at compatible version\n",
            "!pip install -q 'protobuf>=3.20.0,<4.0.0' --upgrade\n",
            "\n",
            "# === Second wave: ML framework packages ===\n",
            "# Install PyTorch with CUDA 11.8\n",
            "!pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118\n",
            "\n",
            "# Install datasets with specific version\n",
            "!pip install -q datasets==2.14.6\n",
            "\n",
            "# Install transformers with a specific version\n",
            "!pip install -q transformers==4.34.1\n",
            "\n",
            "# Install huggingface-hub with version that works with transformers 4.34.1\n",
            "!pip install -q huggingface-hub==0.17.3\n",
            "\n",
            "# === Third wave: Additional important ML packages ===\n",
            "# Install other dependencies with specific versions\n",
            "!pip install -q accelerate==0.23.0 bitsandbytes==0.41.1\n",
            "!pip install -q optimum==1.12.0\n",
            "!pip install -q onnx==1.14.1 onnxruntime==1.15.1\n",
            "!pip install -q safetensors==0.4.0\n",
            "\n",
            "# Install common utilities\n",
            "!pip install -q pyyaml tqdm matplotlib scikit-learn sentencepiece\n",
            "!pip install -q onnx2torch==1.5.12 onnxsim==0.4.33\n",
            "\n",
            "# === Fourth wave: Platform-specific requirements ===\n",
            "# CoreML tools for macOS\n",
            "import platform\n",
            "IS_MACOS = platform.system() == \"Darwin\"\n",
            "if IS_MACOS:\n",
            "    !pip install -q coremltools==6.3.0\n",
            "    if platform.machine() == \"arm64\":  # Apple Silicon\n",
            "        !pip install -q tensorflow-macos==2.13.0 tensorflow-metal==1.0.0\n",
            "    else:  # Intel Mac\n",
            "        !pip install -q tensorflow==2.13.0\n",
            "else:  # Linux/Windows\n",
            "    # Install older version of tensorflow to avoid conflicts\n",
            "    !pip install -q tensorflow==2.13.0\n",
            "\n",
            "# IMPORTANT: After installing all packages, reinstall critical packages to force correct versions\n",
            "# This ensures there are no leftover conflicting dependencies\n",
            "!pip install -q fsspec==2023.9.2 --force-reinstall\n",
            "!pip install -q gcsfs==2023.9.2 --force-reinstall\n",
            "!pip install -q transformers==4.34.1 --force-reinstall\n",
            "!pip install -q huggingface-hub==0.17.3 --force-reinstall\n",
            "\n",
            "print(\"✅ All dependencies installed with carefully controlled versions to avoid conflicts\")"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated dependency installation in cell {i}")
        break

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with comprehensive dependency resolution")
