import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Update any other dataset loading functions to include trust_remote_code
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'preprocess_dataset' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Add auto-trust_remote_code flag for notebook usability
        modified_source = source.replace('# Process datasets for different formats',
        '''# Process datasets for different formats
# Auto-answer 'Yes' to trust_remote_code prompts
import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
''')
        
        notebook['cells'][i]['source'] = [modified_source]
        print(f"Updated preprocess_dataset in cell {i} to auto-trust remote code")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with trust_remote_code handling")
