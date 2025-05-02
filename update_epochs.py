import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the configuration
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and '"epochs": 1,' in ''.join(cell['source']):
        # Replace the epochs line
        for j, line in enumerate(cell['source']):
            if '"epochs": 1,' in line:
                # Replace with 3 epochs for production
                cell['source'][j] = '            "epochs": 3,  # Production training with multiple epochs\n'
                print(f"Updated epochs from 1 to 3 for production training")
                break

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook epochs updated successfully")
