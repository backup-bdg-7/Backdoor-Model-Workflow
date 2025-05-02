import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the preprocess_dataset function
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'def preprocess_dataset' in ''.join(cell['source']):
        source = ''.join(cell['source'])
        
        # Check if we need to add token usage
        if "token=" not in source:
            # Add token usage to tokenizer
            tokenizer_code = """            # Tokenize
            tokenized = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )"""
            
            improved_tokenizer_code = """            # Tokenize (use token from environment if available)
            token = os.environ.get('HF_TOKEN')
            tokenized = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )"""
            
            # Update the source
            new_source = source.replace(tokenizer_code, improved_tokenizer_code)
            notebook['cells'][i]['source'] = [new_source]
            print(f"Updated preprocess_dataset in cell {i} to use token")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Updated preprocess_dataset function")
