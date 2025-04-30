import os
import shutil
import psutil
import subprocess
import time
from huggingface_hub import snapshot_download, login
from datasets import load_dataset

# Login to Hugging Face with the provided token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN environment variable not set. Some datasets may not be accessible.")

try:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Failed to login to Hugging Face: {e}")

# Function to get available disk space in MB
def get_available_disk_space():
    disk = psutil.disk_usage('.')
    return disk.free / (1024 * 1024)  # Convert to MB

# Function to download a dataset
def download_dataset(dataset_id, trust_remote_code=False):
    # Create a clean directory name
    dir_name = dataset_id.replace('/', '_')
    local_dir = f"datasets/{dir_name}"
    
    # Skip if already downloaded
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Dataset {dataset_id} already exists. Skipping...")
        return True
    
    # Check available disk space
    available_space = get_available_disk_space()
    print(f"Available disk space: {available_space:.2f} MB")
    
    if available_space < 500:  # Require at least 500MB free
        print(f"Not enough disk space to download {dataset_id}. Skipping...")
        return False
    
    # Download the dataset
    print(f"Downloading {dataset_id}...")
    try:
        os.makedirs(local_dir, exist_ok=True)
        
        # Try using datasets library first
        try:
            dataset = load_dataset(dataset_id, trust_remote_code=trust_remote_code)
            print(f"Successfully loaded {dataset_id} using datasets library")
            
            # Save the dataset to disk
            for split in dataset:
                split_dir = os.path.join(local_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                dataset[split].save_to_disk(split_dir)
            
            print(f"Successfully saved {dataset_id} to disk")
            return True
        except Exception as e:
            print(f"Failed to load {dataset_id} using datasets library: {e}")
            
            # Try using snapshot_download as fallback
            try:
                snapshot_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    local_dir=local_dir
                )
                print(f"Successfully downloaded {dataset_id} using snapshot_download")
                return True
            except Exception as e2:
                print(f"Failed to download {dataset_id} using snapshot_download: {e2}")
                return False
    except Exception as e:
        print(f"Failed to download {dataset_id}: {e}. Skipping...")
        # Clean up the directory if download failed
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        return False

# Function to add dataset to Git
def add_to_git(dataset_id):
    dir_name = dataset_id.replace('/', '_')
    local_dir = f"datasets/{dir_name}"
    
    print(f"Adding {local_dir} to Git...")
    try:
        subprocess.run(["git", "add", local_dir], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to add {local_dir} to Git: {e}")
        return False

# Function to commit dataset
def commit_dataset(dataset_id):
    print(f"Committing {dataset_id}...")
    try:
        subprocess.run(["git", "commit", "-m", f"Add dataset: {dataset_id}"], check=True)
        print(f"Successfully committed {dataset_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to commit {dataset_id}: {e}")
        return False

# Function to push changes
def push_changes():
    print("Pushing changes to remote repository...")
    try:
        subprocess.run(["git", "push", "origin", "clean-branch"], check=True)
        print("Successfully pushed changes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to push changes: {e}")
        return False

# List of datasets to try downloading
datasets_to_try = [
    "codeparrot/apps",
    "HuggingFaceH4/instruction-dataset",
    "Anthropic/hh-rlhf",
    "HuggingFaceH4/ultrachat_200k",
    "PrimeIntellect/verifiable-coding-problems",
    "DeepNLP/Coding-Agent-GitHub-2025-Feb",
    "aidando73/llama-coding-agent-evals",
    "HumanLLMs/Human-Like-DPO-Dataset",
    "Salesforce/dialogstudio"
]

# Process one dataset
def process_dataset(dataset_id, trust_remote_code=False):
    print(f"\n{'='*50}")
    print(f"Processing dataset: {dataset_id}")
    print(f"{'='*50}\n")
    
    # Download the dataset
    if download_dataset(dataset_id, trust_remote_code):
        # Add to Git
        if add_to_git(dataset_id):
            # Commit the dataset
            if commit_dataset(dataset_id):
                # Push changes
                push_changes()
                return True
    
    return False

# Main function
if __name__ == "__main__":
    # Process HuggingFaceH4/ultrachat_200k
    process_dataset("HuggingFaceH4/ultrachat_200k")
    
    # Process PrimeIntellect/verifiable-coding-problems
    process_dataset("PrimeIntellect/verifiable-coding-problems")
    
    # Process DeepNLP/Coding-Agent-GitHub-2025-Feb
    process_dataset("DeepNLP/Coding-Agent-GitHub-2025-Feb")