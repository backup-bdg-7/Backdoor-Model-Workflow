import os
import shutil
import psutil
import subprocess
from huggingface_hub import snapshot_download, HfApi, login
from huggingface_hub.utils import HFValidationError

# Login to Hugging Face with the provided token
try:
    login(token="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")
    print("Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Failed to login to Hugging Face: {e}")

# List of datasets to download
datasets = [
    "nvidia/OpenCodeReasoning",
    "open-thoughts/OpenThoughts2-1M",
    "Anthropic/values-in-the-wild",
    "PrimeIntellect/verifiable-coding-problems",
    "DeepNLP/Coding-Agent-GitHub-2025-Feb",
    "aidando73/llama-coding-agent-evals",
    "openai/openai_humaneval",
    "HumanLLMs/Human-Like-DPO-Dataset",
    "HuggingFaceH4/ultrachat_200k",
    "Anthropic/hh-rlhf",
    "xAI/TruthfulQA",
    "Salesforce/dialogstudio",
    "bigcode/the-stack",
    "allenai/tool-augmented-dialogues",
    "google-research/toolbench",
    "codeparrot/github-code",
    "luahub/lua-code-dataset",
    "roblox/luau-code",
    "swift-code/swift-repos",
    "HuggingFaceH4/codeparrot-ds",
    "nuprl/MultiPL-E",
    "codeparrot/apps",
    "HuggingFaceH4/python-codes-25k",
    "HuggingFaceH4/instruction-dataset",
    "allenai/dolma",
    "openwebtext/openwebtext"
]

# Directory to store datasets
base_dir = "datasets"
os.makedirs(base_dir, exist_ok=True)

# Function to check available disk space
def get_free_disk_space(path):
    disk = psutil.disk_usage(path)
    return disk.free / (1024 ** 2)  # Free space in MB

# Function to estimate dataset size
def estimate_dataset_size(dataset):
    try:
        api = HfApi()
        dataset_info = api.dataset_info(dataset)
        total_size = 0
        for file in dataset_info.siblings:
            if file.size:
                total_size += file.size
        return total_size / (1024 ** 2)  # Size in MB
    except Exception as e:
        print(f"Could not estimate size for {dataset}: {e}")
        return float('inf')  # Assume it's too large if we can't estimate

# Function to clean up temporary files
def clean_temp_files(dataset_dir):
    cache_dir = os.path.join(dataset_dir, ".cache")
    if os.path.exists(cache_dir):
        print(f"Cleaning up {cache_dir}...")
        shutil.rmtree(cache_dir)

# Function to commit a dataset to Git LFS
def commit_dataset(dataset_name):
    try:
        # Clean up any temporary files first
        dataset_dir = f"{base_dir}/{dataset_name.replace('/', '_')}"
        clean_temp_files(dataset_dir)
        
        # Add the dataset to Git
        print(f"Adding {dataset_dir} to Git...")
        subprocess.run(["git", "add", dataset_dir], check=True)
        
        # Commit the dataset
        print(f"Committing {dataset_name}...")
        subprocess.run(["git", "commit", "-m", f"Add dataset: {dataset_name}"], check=True)
        
        print(f"Successfully committed {dataset_name}")
        return True
    except Exception as e:
        print(f"Failed to commit {dataset_name}: {e}")
        return False

# Download and commit each dataset one by one
for dataset in datasets:
    try:
        # Estimate dataset size
        dataset_size_mb = estimate_dataset_size(dataset)
        print(f"Estimated size of {dataset}: {dataset_size_mb:.2f} MB")

        # Check available disk space
        free_space_mb = get_free_disk_space(base_dir)
        print(f"Available disk space: {free_space_mb:.2f} MB")

        if free_space_mb < dataset_size_mb + 100:  # Add 100 MB buffer
            print(f"Skipping {dataset}: Not enough disk space (required: {dataset_size_mb:.2f} MB, available: {free_space_mb:.2f} MB)")
            continue

        # Define the target directory
        target_dir = f"{base_dir}/{dataset.replace('/', '_')}"
        
        print(f"Downloading {dataset}...")
        
        # For large datasets, we might want to limit what we download
        if dataset_size_mb > 500:
            print(f"{dataset} is large ({dataset_size_mb:.2f} MB). Downloading anyway...")
            
        snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            local_dir=target_dir,
            allow_patterns=["*.jsonl", "*.csv", "*.parquet", "*.txt"],
            ignore_patterns=["*.md", "*.ipynb"]
        )
        print(f"Successfully downloaded {dataset}")

        # Commit the dataset to Git LFS
        if commit_dataset(dataset):
            # Clean up the dataset directory to free up space for the next dataset
            print(f"Cleaning up {target_dir} to free up space...")
            shutil.rmtree(target_dir)
        
    except HFValidationError as e:
        print(f"Failed to download {dataset}: {e}. It might be a gated dataset or require authentication.")
    except Exception as e:
        print(f"Failed to download {dataset}: {e}. Skipping...")

print("Download and commit complete!")