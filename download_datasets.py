import os
import shutil
import psutil
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HFValidationError

# Read the list of datasets from the environment variable
datasets_input = os.getenv("DATASETS_TO_DOWNLOAD")
if not datasets_input:
    raise ValueError("No datasets specified. Set the DATASETS_TO_DOWNLOAD environment variable with a comma-separated list of datasets.")

# Split the comma-separated string into a list and strip whitespace
datasets = [dataset.strip() for dataset in datasets_input.split(",")]
if not datasets:
    raise ValueError("No valid datasets provided in DATASETS_TO_DOWNLOAD.")

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

# Download each dataset with space checks and cleanup
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
        
        # Try to download specific splits for large datasets
        splits = None
        if dataset_size_mb > 500:  # If dataset is larger than 500 MB, try downloading only the 'train' split
            print(f"{dataset} is large ({dataset_size_mb:.2f} MB). Attempting to download only 'train' split...")
            splits = ["train"]

        print(f"Downloading {dataset}...")
        snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            local_dir=target_dir,
            allow_patterns=["*.jsonl", "*.csv", "*.parquet", "*.txt"],
            ignore_patterns=["*.md", "*.ipynb"],
            dataset_splits=splits  # Download specific splits if specified
        )
        print(f"Successfully downloaded {dataset}")

        # Clean up temporary files
        clean_temp_files(target_dir)

    except HFValidationError as e:
        print(f"Failed to download {dataset}: {e}. It might be a gated dataset or require authentication.")
    except Exception as e:
        print(f"Failed to download {dataset}: {e}. Skipping...")

print("Download complete!")