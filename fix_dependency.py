import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the pip install commands
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'pip install' in ''.join(cell['source']):
        # Add fsspec installation at the end of the cell
        new_source = cell['source']
        new_source.append("\n# Fix dependency conflict between gcsfs and fsspec\n!pip install -q fsspec==2025.3.2 --upgrade")
        notebook['cells'][i]['source'] = new_source
        print(f"Updated cell {i} with fsspec installation command")
        break

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully")
