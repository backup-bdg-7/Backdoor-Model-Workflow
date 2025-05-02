import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with dependency installation
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '!pip install' in ''.join(cell['source']):
        # Replace with improved dependency installation that avoids conflicts
        new_source = [
            "# Clone repository and install dependencies\n",
            "!git clone https://github.com/backup-bdg-6/datasets.git\n",
            "%cd datasets\n",
            "\n",
            "# Install core dependencies with careful versioning to avoid conflicts\n",
            "!pip install rouge\n",
            "!pip install rouge_score\n",
            "\n",
            "# Install fsspec first with the exact version required by gcsfs\n",
            "!pip install -q fsspec==2025.3.2\n",
            "\n",
            "# Install pytorch with compatible CUDA versions\n",
            "!pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118\n",
            "\n",
            "# Install other dependencies after torch to ensure compatibility\n",
            "!pip install -q transformers>=4.30.0 datasets>=2.10.0 accelerate>=0.20.0 bitsandbytes>=0.39.0\n",
            "!pip install -q optimum>=1.8.0 onnx>=1.13.0 onnxruntime>=1.14.0 huggingface_hub>=0.16.0 safetensors>=0.3.1\n",
            "!pip install -q pyyaml tqdm matplotlib numpy scikit-learn sentencepiece protobuf>=3.20.0,<4.0.0\n",
            "!pip install -q onnx2torch>=1.5.2 onnxsim>=0.4.21\n",
            "\n",
            "# Install gcsfs after fsspec to ensure version compatibility\n",
            "!pip install -q gcsfs==2025.3.2\n",
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
            "print(\"✅ Required dependencies installed with versioning to avoid conflicts\")"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated dependency installation in cell {i}")
        break

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with compatible dependency versions")
