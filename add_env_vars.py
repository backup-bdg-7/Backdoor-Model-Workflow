import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the first cell to add the environment variables
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'import' in ''.join(cell['source']):
        # Add env vars to the beginning of the imports cell
        source = ''.join(cell['source'])
        new_source = "# Set Hugging Face token as environment variable for universal access\nimport os\nos.environ['HF_TOKEN'] = \"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\"\nos.environ['HUGGING_FACE_HUB_TOKEN'] = \"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\"\n\n" + source
        notebook['cells'][i]['source'] = [new_source]
        print(f"Added environment variables to cell {i}")
        break

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Added environment variables to the notebook")
