import os
import subprocess
import time

# Directory to add
base_dir = "datasets/nvidia_OpenCodeReasoning"

# Get all files in the directory
files = []
for root, dirs, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith('.parquet'):
            files.append(os.path.join(root, filename))

print(f"Found {len(files)} files to add")

# Add files one by one
for i, file in enumerate(files):
    print(f"Adding file {i+1}/{len(files)}: {file}")
    try:
        subprocess.run(["git", "add", file], check=True)
        print(f"Successfully added {file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to add {file}: {e}")
    
    # Sleep a bit to avoid overwhelming the system
    time.sleep(0.1)

print("All files added successfully!")