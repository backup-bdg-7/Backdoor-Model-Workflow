import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# First, look for and modify any cell that sets environment variables
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'os.environ' in ''.join(cell['source']) and 'HF_TOKEN' in ''.join(cell['source']):
        if 'Set Hugging Face token as environment variable' in ''.join(cell['source']):
            # Remove environment variable settings
            source = ''.join(cell['source'])
            # Extract just the imports without setting environment variables
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if 'import os' in line and not 'os.environ' in line:
                    # Keep line with import os
                    new_lines.append(line)
                elif 'os.environ' not in line:
                    # Keep any line that doesn't set environment variables
                    new_lines.append(line)
            
            new_source = '\n'.join(new_lines)
            notebook['cells'][i]['source'] = [new_source]
            print(f"Removed environment variable settings from cell {i}")

# Now update the HuggingFace authentication cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'Setup HuggingFace authentication' in ''.join(cell['source']):
        # Replace with simplified authentication that doesn't use environment variables
        new_source = [
            "# Setup HuggingFace authentication with hard-coded token (no environment variables)\n",
            "try:\n",
            "    from huggingface_hub import login, HfFolder, HfApi\n",
            "    import datasets\n",
            "    \n",
            "    # Define the token\n",
            "    HF_TOKEN = \"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\"  # Hard-coded token\n",
            "    \n",
            "    # Clear any existing environment variables to prevent conflicts\n",
            "    import os\n",
            "    if 'HF_TOKEN' in os.environ:\n",
            "        del os.environ['HF_TOKEN']\n",
            "    if 'HUGGING_FACE_HUB_TOKEN' in os.environ:\n",
            "        del os.environ['HUGGING_FACE_HUB_TOKEN']\n",
            "    \n",
            "    # Login directly with token (no environment variable, just direct API usage)\n",
            "    login(token=HF_TOKEN, write_permission=False, add_to_git_credential=False)\n",
            "    \n",
            "    # Set trust_remote_code to True globally to avoid prompts\n",
            "    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True\n",
            "    \n",
            "    # Also explicitly set token in HfFolder for maximum compatibility\n",
            "    token = HfFolder.get_token()\n",
            "    if token != HF_TOKEN:\n",
            "        HfFolder.save_token(HF_TOKEN)\n",
            "    \n",
            "    print(\"✅ Authenticated with HuggingFace using direct token method (no environment variables)\")\n",
            "        \n",
            "except ImportError as e:\n",
            "    print(f\"⚠️ huggingface_hub not available: {e}\")\n",
            "    # Still define token for direct use\n",
            "    HF_TOKEN = \"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\"\n",
            "except Exception as e:\n",
            "    print(f\"⚠️ Error setting up HuggingFace authentication: {e}\")\n",
            "    # Still define token for direct use\n",
            "    HF_TOKEN = \"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\"\n"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated HuggingFace authentication cell {i}")

# Also find and update any dataset loading to use direct token
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'load_datasets_from_config' in ''.join(cell['source']):
        # Update to use direct token parameter without environment variable fallbacks
        source = ''.join(cell['source'])
        
        # Remove environment variable checks
        if 'elif "HF_TOKEN" in os.environ:' in source:
            old_code = """            # Setup download parameters with authentication
            download_params = {}
            if auth_token:
                download_params["token"] = auth_token
            elif "HF_TOKEN" in os.environ:
                # Fall back to environment variable if not passed directly
                download_params["token"] = os.environ["HF_TOKEN"]
            elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
                # Check alternative env var name
                download_params["token"] = os.environ["HUGGING_FACE_HUB_TOKEN"]"""
                
            new_code = """            # Setup download parameters with authentication (direct token, no env vars)
            download_params = {}
            if auth_token:
                download_params["token"] = auth_token"""
                
            updated_source = source.replace(old_code, new_code)
            notebook['cells'][i]['source'] = [updated_source]
            print(f"Updated token usage in load_datasets_from_config in cell {i}")

# Update the token passing cell to be explicit about not using environment variables
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'auth_token = HF_TOKEN' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Update message to clarify direct token usage
        if "Authentication should already be set up globally" in source:
            updated_source = source.replace(
                "# Authentication should already be set up globally", 
                "# Pass token directly to the load function, not using environment variables"
            )
            notebook['cells'][i]['source'] = [updated_source]
            print(f"Updated token assignment explanation in cell {i}")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with simplified HuggingFace authentication")
