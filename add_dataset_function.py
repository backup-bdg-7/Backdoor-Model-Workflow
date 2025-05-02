import json

# Read the notebook
with open('ml_workflow.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell right after HuggingFace authentication and before the use of load_datasets_from_config
auth_cell_idx = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'HF_TOKEN' in ''.join(cell['source']):
        auth_cell_idx = i
        break

if auth_cell_idx is not None:
    # Create a new cell with the load_datasets_from_config function
    load_datasets_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Load datasets from configuration\n",
            "def load_datasets_from_config(config, auth_token=None):\n",
            "    \"\"\"Load datasets from configuration with proper error handling.\"\"\"\n",
            "    # Get active stage configuration\n",
            "    active_stage = config['training']['active_stage']\n",
            "    stage_config = next((s for s in config['training']['stages'] if s['name'] == active_stage), None)\n",
            "    \n",
            "    if not stage_config:\n",
            "        raise ValueError(f\"Training stage {active_stage} not found\")\n",
            "        \n",
            "    train_datasets = []\n",
            "    eval_datasets = []\n",
            "    \n",
            "    total_datasets = len(stage_config['datasets'])\n",
            "    print(f\"Loading {total_datasets} datasets...\")\n",
            "    \n",
            "    # Process each dataset\n",
            "    for i, dataset_config in enumerate(stage_config['datasets']):\n",
            "        name = dataset_config['name']\n",
            "        split = dataset_config['split']\n",
            "        streaming = dataset_config.get('streaming', False)\n",
            "        max_samples = dataset_config.get('max_samples', None)\n",
            "        subset = dataset_config.get('subset', None)\n",
            "        \n",
            "        print(f\"\\n[{i+1}/{total_datasets}] Loading: {name} (split: {split})\")\n",
            "        \n",
            "        try:\n",
            "            # Setup download parameters with authentication if needed\n",
            "            download_params = {}\n",
            "            if auth_token:\n",
            "                download_params[\"token\"] = auth_token\n",
            "                \n",
            "            # Try to load the dataset\n",
            "            try:\n",
            "                if subset:\n",
            "                    dataset = load_dataset(name, subset, split=split, streaming=streaming, **download_params)\n",
            "                else:\n",
            "                    dataset = load_dataset(name, split=split, streaming=streaming, **download_params)\n",
            "                    \n",
            "                # For streaming datasets, verify we can access it\n",
            "                if streaming:\n",
            "                    iter_dataset = iter(dataset)\n",
            "                    next(iter_dataset)  # Try to get first item\n",
            "            except Exception as e:\n",
            "                print(f\"  ⚠️ Error loading dataset: {str(e)}\")\n",
            "                print(f\"  ⚠️ Skipping this dataset and continuing\")\n",
            "                continue\n",
            "            \n",
            "            # Limit samples from streaming dataset\n",
            "            if streaming and max_samples:\n",
            "                print(f\"  Taking {max_samples} samples from streaming dataset\")\n",
            "                dataset = dataset.take(max_samples)\n",
            "                streaming = False\n",
            "                \n",
            "            # Limit samples for non-streaming datasets\n",
            "            if not streaming and max_samples and len(dataset) > max_samples:\n",
            "                print(f\"  Limiting to {max_samples} samples\")\n",
            "                indices = np.random.choice(len(dataset), max_samples, replace=False)\n",
            "                dataset = dataset.select(indices)\n",
            "                \n",
            "            # Add to appropriate list\n",
            "            if split == 'train':\n",
            "                train_datasets.append(dataset)\n",
            "                print(f\"  ✅ Added to training datasets\")\n",
            "            elif split in ['test', 'validation']:\n",
            "                eval_datasets.append(dataset)\n",
            "                print(f\"  ✅ Added to evaluation datasets\")\n",
            "                \n",
            "            if not streaming:\n",
            "                print(f\"  Dataset size: {len(dataset)} examples\")\n",
            "                \n",
            "        except Exception as e:\n",
            "            print(f\"  ❌ Error processing dataset {name}: {e}\")\n",
            "    \n",
            "    return train_datasets, eval_datasets"
        ]
    }
    
    # Insert the new cell after the authentication cell
    notebook['cells'].insert(auth_cell_idx + 1, load_datasets_cell)
    print(f"Added load_datasets_from_config function after cell {auth_cell_idx}")
else:
    print("Couldn't find the HuggingFace authentication cell")

# Write the updated notebook
with open('ml_workflow.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully")
