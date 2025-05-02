import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with dependency installation
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '!pip install' in ''.join(cell['source']):
        # Replace with improved dependency installation that avoids circular conflicts
        new_source = [
            "# Clone repository and install dependencies\n",
            "!git clone https://github.com/backup-bdg-6/datasets.git\n",
            "%cd datasets\n",
            "\n",
            "# Install essential packages first\n",
            "!pip install rouge\n",
            "!pip install rouge_score\n",
            "\n",
            "# === DEPENDENCY RESOLUTION STRATEGY ===\n",
            "# 1. Use an older, stable PyTorch version to avoid CUDA conflicts\n",
            "# 2. Use specific versions for packages with circular dependencies\n",
            "# 3. Install packages in carefully ordered sequence to maintain compatibility\n",
            "\n",
            "# First, install an older PyTorch version with CUDA 11.8 to avoid conflicts\n",
            "!pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118\n",
            "\n",
            "# Install a specific version of datasets that works with older fsspec\n",
            "!pip install -q datasets==2.14.6\n",
            "\n",
            "# Install fsspec with a version that both datasets and gcsfs can work with\n",
            "!pip install -q fsspec==2023.9.2\n",
            "\n",
            "# Install gcsfs with a compatible version\n",
            "!pip install -q gcsfs==2023.9.2\n",
            "\n",
            "# Install remaining dependencies with version constraints\n",
            "!pip install -q transformers==4.34.1 accelerate==0.23.0 bitsandbytes==0.41.1\n",
            "!pip install -q optimum==1.12.0 onnx==1.14.1 onnxruntime==1.15.1 huggingface_hub==0.17.3 safetensors==0.4.0\n",
            "!pip install -q pyyaml tqdm matplotlib numpy scikit-learn sentencepiece 'protobuf>=3.20.0,<4.0.0'\n",
            "!pip install -q onnx2torch==1.5.12 onnxsim==0.4.33\n",
            "\n",
            "# Install CoreML tools for macOS\n",
            "import platform\n",
            "IS_MACOS = platform.system() == \"Darwin\"\n",
            "if IS_MACOS:\n",
            "    !pip install -q coremltools==6.3.0\n",
            "    if platform.machine() == \"arm64\":  # Apple Silicon\n",
            "        !pip install -q tensorflow-macos==2.13.0 tensorflow-metal==1.0.0\n",
            "    else:  # Intel Mac\n",
            "        !pip install -q tensorflow==2.13.0\n",
            "else:  # Linux/Windows\n",
            "    !pip install -q tensorflow==2.13.0\n",
            "\n",
            "print(\"✅ Required dependencies installed with carefully chosen versions to avoid conflicts\")"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated dependency installation in cell {i}")
        break

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with compatible dependency versions to avoid circular conflicts")
