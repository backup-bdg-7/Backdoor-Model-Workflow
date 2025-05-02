import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the function where we need to modify streaming dataset handling
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'def load_datasets_from_config' in ''.join(cell['source']):
        # Get the current source as a string
        source = ''.join(cell['source'])
        
        # Replace the problematic code for handling IterableDatasets
        # Adding a convert_to_regular_dataset function to handle streaming datasets
        modified_source = source.replace('def load_datasets_from_config(config, auth_token=None):',
        '''def load_datasets_from_config(config, auth_token=None):
    """Load datasets from configuration with proper error handling."""
    # Helper function to convert streaming datasets to regular datasets
    def convert_to_regular_dataset(streaming_dataset, max_samples):
        """Convert streaming dataset to regular dataset with specified number of samples."""
        try:
            # Take specified number of samples
            examples = []
            count = 0
            for example in streaming_dataset:
                examples.append(example)
                count += 1
                if count >= max_samples:
                    break
            
            # Convert to Dataset (no longer streaming)
            from datasets import Dataset
            return Dataset.from_list(examples)
        except Exception as e:
            print(f"  ⚠️ Error converting streaming dataset: {e}")
            return None''')

        # Fix how streaming datasets are processed
        old_streaming_code = """            # Limit samples from streaming dataset
            if streaming and max_samples:
                print(f"  Taking {max_samples} samples from streaming dataset")
                dataset = dataset.take(max_samples)
                streaming = False"""
                
        new_streaming_code = """            # Limit samples from streaming dataset
            if streaming and max_samples:
                print(f"  Taking {max_samples} samples from streaming dataset")
                # Convert streaming dataset to regular dataset
                regular_dataset = convert_to_regular_dataset(dataset, max_samples)
                if regular_dataset is not None:
                    dataset = regular_dataset
                    streaming = False
                else:
                    # If conversion failed, just keep as streaming but with take()
                    dataset = dataset.take(max_samples)"""
                    
        modified_source = modified_source.replace(old_streaming_code, new_streaming_code)
        
        # Update the cell with modified source
        notebook['cells'][i]['source'] = [modified_source]
        print(f"Updated load_datasets_from_config in cell {i} to better handle streaming datasets")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated with improved streaming dataset handling")
