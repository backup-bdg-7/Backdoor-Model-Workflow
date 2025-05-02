import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the HuggingFace authentication
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '# Setup HuggingFace authentication' in ''.join(cell['source']):
        # Replace the authentication code to use hardcoded token
        new_source = [
            "# Setup HuggingFace authentication with hard-coded token\n",
            "try:\n",
            "    from huggingface_hub import login, HfFolder\n",
            "    \n",
            "    # Use hard-coded token for automatic authentication\n",
            "    HF_TOKEN = \"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\"  # Hard-coded token\n",
            "    \n",
            "    # Set token directly\n",
            "    login(token=HF_TOKEN, write_permission=False)\n",
            "    print(\"✅ Automatically authenticated with HuggingFace using provided token\")\n",
            "    \n",
            "    # Verify token is set\n",
            "    token = HfFolder.get_token()\n",
            "    if token is not None:\n",
            "        print(\"✅ HuggingFace token is active and ready for accessing gated datasets\")\n",
            "    else:\n",
            "        print(\"⚠️ Token could not be set. Using manual authentication as fallback.\")\n",
            "        login()  # Fallback to manual login if needed\n",
            "        token = HfFolder.get_token()\n",
            "except ImportError:\n",
            "    print(\"⚠️ huggingface_hub not available. Some datasets may not be accessible.\")\n",
            "    token = None\n",
            "    HF_TOKEN = \"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\"  # Still define token for direct use"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated cell {i} - replaced HuggingFace authentication with hardcoded token")

    # Also update the auth_token assignment in the dataset loading cell
    if cell['cell_type'] == 'code' and 'auth_token = HfFolder.get_token()' in ''.join(cell['source']):
        for j, line in enumerate(cell['source']):
            if 'auth_token = HfFolder.get_token()' in line:
                # Replace with direct token assignment
                cell['source'][j] = "# Get authentication token - use hardcoded token for consistent access\n" + \
                                   "auth_token = HF_TOKEN  # Use our hardcoded token to ensure access to gated datasets\n"
                print(f"Updated cell {i} - updated auth_token assignment to use hardcoded token")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully with hardcoded HuggingFace token")
