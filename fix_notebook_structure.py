import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Check if the notebook structure is valid
fixed_cells = 0
for i, cell in enumerate(notebook['cells']):
    # Check if 'outputs' property is missing
    if 'outputs' not in cell:
        cell['outputs'] = []
        fixed_cells += 1
        print(f"Added 'outputs' property to cell {i}")
    
    # Ensure other required properties are present
    if 'execution_count' not in cell:
        cell['execution_count'] = None
        print(f"Added 'execution_count' property to cell {i}")
    
    # Make sure metadata exists
    if 'metadata' not in cell:
        cell['metadata'] = {}
        print(f"Added 'metadata' property to cell {i}")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

if fixed_cells > 0:
    print(f"Fixed {fixed_cells} cells by adding 'outputs' property")
else:
    print("All cells already have 'outputs' property")
