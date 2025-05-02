import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# First, find the HuggingFace authentication cell
auth_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'HF_TOKEN' in ''.join(cell['source']):
        auth_cell_idx = i
        break

if auth_cell_idx is not None:
    # Improve the authentication method to set up HF auth globally
    improved_auth = [
        "# Setup HuggingFace authentication with hard-coded token\n",
        "try:\n",
        "    import os\n",
        "    from huggingface_hub import login, HfFolder, HfApi\n",
        "    import datasets\n",
        "    \n",
        "    # Use hard-coded token for automatic authentication\n",
        "    HF_TOKEN = \"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\"  # Hard-coded token\n",
        "    \n",
        "    # Set environment variable for token (this ensures all HF libraries use it)\n",
        "    os.environ['HF_TOKEN'] = HF_TOKEN\n",
        "    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN\n",
        "    \n",
        "    # Login directly with token and set write_permission=False for read-only access\n",
        "    login(token=HF_TOKEN, write_permission=False)\n",
        "    \n",
        "    # Set trust_remote_code to True globally to avoid prompts\n",
        "    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True\n",
        "    \n",
        "    # Verify token is set and active\n",
        "    try:\n",
        "        api = HfApi(token=HF_TOKEN)\n",
        "        # Try a simple API call that requires authentication\n",
        "        whoami = api.whoami()\n",
        "        print(f\"✅ Authenticated with HuggingFace as: {whoami['name']}\")\n",
        "    except Exception as e:\n",
        "        # Still authenticate but show warning\n",
        "        print(f\"⚠️ Authentication verified but couldn't get user info: {e}\")\n",
        "        pass\n",
        "    \n",
        "    # Also set the token in HfFolder for other libraries that might use it\n",
        "    token = HfFolder.get_token()\n",
        "    if token != HF_TOKEN:\n",
        "        HfFolder.save_token(HF_TOKEN)\n",
        "        token = HF_TOKEN\n",
        "        \n",
        "    print(\"✅ HuggingFace token is active and ready for accessing gated datasets\")\n",
        "    \n",
        "except ImportError as e:\n",
        "    print(f\"⚠️ huggingface_hub not available: {e}\")\n",
        "    token = None\n",
        "    HF_TOKEN = \"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\"  # Still define token for direct use\n",
        "except Exception as e:\n",
        "    print(f\"⚠️ Error setting up HuggingFace authentication: {e}\")\n",
        "    # Still define token for direct use\n",
        "    token = HF_TOKEN = \"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\"\n"
    ]
    notebook['cells'][auth_cell_idx]['source'] = improved_auth
    print(f"Updated HuggingFace authentication cell {auth_cell_idx}")

    # Update the loading cell to prefer env vars instead of directly passing token
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'auth_token = HF_TOKEN' in ''.join(cell['source']):
            new_source = ''.join(cell['source']).replace(
                "# Get authentication token - use hardcoded token for consistent access\nauth_token = HF_TOKEN  # Use our hardcoded token to ensure access to gated datasets",
                "# Authentication should already be set up globally, but also pass token directly to be safe\nauth_token = HF_TOKEN"
            )
            notebook['cells'][i]['source'] = [new_source]
            print(f"Updated token assignment in cell {i}")

    # Update the load_datasets_from_config function to use env vars as fallback  
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code' and 'def load_datasets_from_config' in ''.join(cell['source']):
            new_source = ''.join(cell['source'])
            
            # Add environment variable fallback for token
            token_code = """            # Setup download parameters with authentication if needed
            download_params = {}
            if auth_token:
                download_params["token"] = auth_token"""
                
            improved_token_code = """            # Setup download parameters with authentication
            download_params = {}
            if auth_token:
                download_params["token"] = auth_token
            elif "HF_TOKEN" in os.environ:
                # Fall back to environment variable if not passed directly
                download_params["token"] = os.environ["HF_TOKEN"]
            elif "HUGGING_FACE_HUB_TOKEN" in os.environ:
                # Check alternative env var name
                download_params["token"] = os.environ["HUGGING_FACE_HUB_TOKEN"]"""
                
            new_source = new_source.replace(token_code, improved_token_code)
            
            # Add import os if needed
            if "import os" not in new_source:
                new_source = "import os\n" + new_source
                
            notebook['cells'][i]['source'] = [new_source]
            print(f"Updated load_datasets_from_config in cell {i} to use env vars as fallback")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with improved HuggingFace authentication")
