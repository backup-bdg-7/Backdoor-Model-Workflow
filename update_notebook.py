import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the demo configuration
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '# Limit to first 5 datasets for demo' in ''.join(cell['source']):
        # Replace the code in this cell
        new_source = [
            "# Load all datasets for production training\n",
            "print(\"Loading all configured datasets for production training workflow...\")\n",
            "\n",
            "# Get authentication token if available\n",
            "auth_token = HfFolder.get_token() if 'HfFolder' in globals() else None\n",
            "\n",
            "# Load the datasets with all configured datasets\n",
            "train_datasets, eval_datasets = load_datasets_from_config(config, auth_token)\n",
            "print(f\"\\n✅ Loaded {len(train_datasets)} training datasets and {len(eval_datasets)} evaluation datasets\")"
        ]
        notebook['cells'][i]['source'] = new_source
        print(f"Updated cell {i} - replaced demo configuration with production code")
        
    # Update the conclusion cell to remove mention of FULL_RUN
    if cell['cell_type'] == 'markdown' and 'Set `FULL_RUN = True`' in ''.join(cell['source']):
        # Find and update the line mentioning FULL_RUN
        for j, line in enumerate(cell['source']):
            if 'Set `FULL_RUN = True`' in line:
                cell['source'][j] = "All the code is ready for production use with the full dataset set.\n"
                print(f"Updated conclusion cell to remove mention of FULL_RUN")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully")
