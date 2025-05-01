"""
/**
 * Copyright (c) [2025] Backdoor Software Inc.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of Backdoor Software Inc.
 * You may not disclose, reproduce, or distribute this software without the express written
 * permission of Backdoor Software Inc.
 *
 * Created by: Backdoor Software Inc.
 * Purpose: Activity Maintenance
 */
"""

"""
Dataset loading utilities for the AI model training workflow.
This module provides functions to load, preprocess, and stream datasets from various sources.
"""

import os
import logging
import yaml
from typing import Dict, List, Optional, Union, Any, Tuple
import datasets
from datasets import load_dataset, Dataset, DatasetDict, IterableDataset, concatenate_datasets
from huggingface_hub import HfApi, HfFolder
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import tempfile
import shutil
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    A class to handle loading and preprocessing of datasets from various sources.
    """
    
    def __init__(self, config_path: str, huggingface_token: Optional[str] = None):
        """
        Initialize the DatasetLoader with configuration.
        
        Args:
            config_path: Path to the configuration file
            huggingface_token: HuggingFace API token for accessing gated datasets
        """
        self.config_path = config_path
        self.huggingface_token = huggingface_token
        self.config = self._load_config()
        self.dataset_cache = {}
        
        # Set up HuggingFace token if provided
        if huggingface_token:
            HfFolder.save_token(huggingface_token)
            self.hf_api = HfApi(token=huggingface_token)
            logger.info("HuggingFace token set successfully")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _get_dataset_config(self, dataset_name: str) -> Dict:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing dataset configuration
        """
        # Check core datasets
        for dataset in self.config['datasets']['core_datasets']:
            if dataset['name'] == dataset_name:
                return dataset
        
        # Check additional datasets
        for dataset in self.config['datasets']['additional_datasets']:
            if dataset['name'] == dataset_name:
                return dataset
        
        raise ValueError(f"Dataset {dataset_name} not found in configuration")
    
    def _create_cache_key(self, dataset_name: str, subset: Optional[str] = None, 
                         split: Optional[str] = None, streaming: bool = False) -> str:
        """
        Create a unique cache key for a dataset configuration.
        
        Args:
            dataset_name: Name of the dataset
            subset: Subset of the dataset
            split: Split of the dataset
            streaming: Whether the dataset is streamed
            
        Returns:
            A unique cache key
        """
        key_parts = [dataset_name]
        if subset:
            key_parts.append(subset)
        if split:
            key_parts.append(split)
        if streaming:
            key_parts.append("streaming")
        
        return "_".join(key_parts)
    
    def _get_dataset_from_cache(self, cache_key: str) -> Optional[Union[Dataset, IterableDataset, DatasetDict]]:
        """
        Retrieve a dataset from the cache if available.
        
        Args:
            cache_key: Cache key for the dataset
            
        Returns:
            Cached dataset or None if not found
        """
        return self.dataset_cache.get(cache_key)
    
    def _add_dataset_to_cache(self, cache_key: str, dataset: Union[Dataset, IterableDataset, DatasetDict]) -> None:
        """
        Add a dataset to the cache.
        
        Args:
            cache_key: Cache key for the dataset
            dataset: Dataset to cache
        """
        self.dataset_cache[cache_key] = dataset
    
    def load_dataset(self, dataset_name: str, subset: Optional[str] = None, 
                    split: Optional[str] = None, streaming: Optional[bool] = None,
                    max_samples: Optional[int] = None) -> Union[Dataset, IterableDataset, DatasetDict]:
        """
        Load a dataset from HuggingFace or other sources.
        
        Args:
            dataset_name: Name of the dataset
            subset: Subset of the dataset
            split: Split of the dataset (train, validation, test)
            streaming: Whether to stream the dataset (for large datasets)
            max_samples: Maximum number of samples to load
            
        Returns:
            Loaded dataset
        """
        # Get dataset configuration
        dataset_config = self._get_dataset_config(dataset_name)
        
        # Override streaming parameter if provided
        if streaming is None and 'streaming' in dataset_config:
            streaming = dataset_config['streaming']
        
        # Override max_samples parameter if provided
        if max_samples is None and 'max_samples' in dataset_config:
            max_samples = dataset_config['max_samples']
        
        # Create cache key
        cache_key = self._create_cache_key(dataset_name, subset, split, streaming)
        
        # Check if dataset is in cache
        cached_dataset = self._get_dataset_from_cache(cache_key)
        if cached_dataset is not None:
            logger.info(f"Using cached dataset: {cache_key}")
            return cached_dataset
        
        # Load dataset based on type
        dataset_type = dataset_config.get('type', 'huggingface')
        
        try:
            if dataset_type == 'huggingface':
                dataset = self._load_huggingface_dataset(
                    dataset_name, subset, split, streaming, max_samples
                )
            elif dataset_type == 'kaggle':
                dataset = self._load_kaggle_dataset(
                    dataset_config, subset, split, streaming, max_samples
                )
            elif dataset_type == 'local':
                dataset = self._load_local_dataset(
                    dataset_config, subset, split, streaming, max_samples
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            # Add to cache
            self._add_dataset_to_cache(cache_key, dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            # Try fallback if configured
            if self.config['error_handling']['fallbacks'].get('dataset_unavailable') == 'use_cached':
                logger.warning(f"Attempting to use cached version of {dataset_name}")
                return self._load_cached_fallback(dataset_name, subset, split)
            raise
    
    def _load_huggingface_dataset(self, dataset_name: str, subset: Optional[str] = None,
                                 split: Optional[str] = None, streaming: bool = False,
                                 max_samples: Optional[int] = None) -> Union[Dataset, IterableDataset, DatasetDict]:
        """
        Load a dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset
            subset: Subset of the dataset
            split: Split of the dataset
            streaming: Whether to stream the dataset
            max_samples: Maximum number of samples to load
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading HuggingFace dataset: {dataset_name}, subset: {subset}, split: {split}, streaming: {streaming}")
        
        try:
            # Load dataset
            dataset = load_dataset(
                dataset_name,
                name=subset,
                split=split,
                streaming=streaming,
                token=self.huggingface_token
            )
            
            # Limit samples if needed and not streaming
            if max_samples is not None:
                if streaming:
                    dataset = dataset.take(max_samples)
                else:
                    if isinstance(dataset, DatasetDict):
                        for key in dataset:
                            dataset[key] = dataset[key].select(range(min(len(dataset[key]), max_samples)))
                    else:
                        dataset = dataset.select(range(min(len(dataset), max_samples)))
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset {dataset_name}: {e}")
            raise
    
    def _load_kaggle_dataset(self, dataset_config: Dict, subset: Optional[str] = None,
                            split: Optional[str] = None, streaming: bool = False,
                            max_samples: Optional[int] = None) -> Union[Dataset, IterableDataset]:
        """
        Load a dataset from Kaggle.
        
        Args:
            dataset_config: Dataset configuration
            subset: Subset of the dataset
            split: Split of the dataset
            streaming: Whether to stream the dataset
            max_samples: Maximum number of samples to load
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading Kaggle dataset: {dataset_config['name']}")
        
        # Extract Kaggle dataset details
        kaggle_dataset = dataset_config['kaggle_dataset']
        file_name = dataset_config.get('file_name')
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download dataset using Kaggle API
                os.system(f"kaggle datasets download -d {kaggle_dataset} -p {temp_dir}")
                
                # Extract downloaded zip file
                zip_file = os.path.join(temp_dir, f"{kaggle_dataset.split('/')[-1]}.zip")
                shutil.unpack_archive(zip_file, temp_dir)
                
                # Find the file to load
                if file_name:
                    file_path = os.path.join(temp_dir, file_name)
                else:
                    # Try to find a suitable file
                    files = os.listdir(temp_dir)
                    csv_files = [f for f in files if f.endswith('.csv')]
                    json_files = [f for f in files if f.endswith('.json')]
                    parquet_files = [f for f in files if f.endswith('.parquet')]
                    
                    if csv_files:
                        file_path = os.path.join(temp_dir, csv_files[0])
                    elif json_files:
                        file_path = os.path.join(temp_dir, json_files[0])
                    elif parquet_files:
                        file_path = os.path.join(temp_dir, parquet_files[0])
                    else:
                        raise ValueError(f"No suitable file found in Kaggle dataset {kaggle_dataset}")
                
                # Load the file into a dataset
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    dataset = Dataset.from_pandas(df)
                elif file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    dataset = Dataset.from_dict(data)
                elif file_path.endswith('.parquet'):
                    dataset = Dataset.from_parquet(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
                
                # Apply split if needed
                if split:
                    if split == 'train':
                        dataset = dataset.train_test_split(test_size=0.2, seed=42)['train']
                    elif split == 'validation' or split == 'test':
                        dataset = dataset.train_test_split(test_size=0.2, seed=42)['test']
                
                # Limit samples if needed
                if max_samples is not None:
                    dataset = dataset.select(range(min(len(dataset), max_samples)))
                
                # Convert to iterable dataset if streaming is requested
                if streaming:
                    dataset = dataset.to_iterable_dataset(batch_size=1000)
                
                return dataset
                
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset {kaggle_dataset}: {e}")
            raise
    
    def _load_local_dataset(self, dataset_config: Dict, subset: Optional[str] = None,
                           split: Optional[str] = None, streaming: bool = False,
                           max_samples: Optional[int] = None) -> Union[Dataset, IterableDataset]:
        """
        Load a dataset from local files.
        
        Args:
            dataset_config: Dataset configuration
            subset: Subset of the dataset
            split: Split of the dataset
            streaming: Whether to stream the dataset
            max_samples: Maximum number of samples to load
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading local dataset: {dataset_config['name']}")
        
        # Get file path
        file_path = dataset_config['file_path']
        file_format = dataset_config.get('file_format', os.path.splitext(file_path)[1][1:])
        
        try:
            # Load dataset based on format
            if file_format == 'csv':
                dataset = Dataset.from_csv(file_path)
            elif file_format == 'json':
                dataset = Dataset.from_json(file_path)
            elif file_format == 'parquet':
                dataset = Dataset.from_parquet(file_path)
            elif file_format == 'text':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                dataset = Dataset.from_dict({'text': lines})
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Apply split if needed
            if split:
                if split == 'train':
                    dataset = dataset.train_test_split(test_size=0.2, seed=42)['train']
                elif split == 'validation' or split == 'test':
                    dataset = dataset.train_test_split(test_size=0.2, seed=42)['test']
            
            # Limit samples if needed
            if max_samples is not None:
                dataset = dataset.select(range(min(len(dataset), max_samples)))
            
            # Convert to iterable dataset if streaming is requested
            if streaming:
                dataset = dataset.to_iterable_dataset(batch_size=1000)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading local dataset {file_path}: {e}")
            raise
    
    def _load_cached_fallback(self, dataset_name: str, subset: Optional[str] = None,
                             split: Optional[str] = None) -> Dataset:
        """
        Load a cached version of a dataset as fallback.
        
        Args:
            dataset_name: Name of the dataset
            subset: Subset of the dataset
            split: Split of the dataset
            
        Returns:
            Cached dataset
        """
        # Create a unique identifier for the dataset
        dataset_id = f"{dataset_name}"
        if subset:
            dataset_id += f"_{subset}"
        if split:
            dataset_id += f"_{split}"
        
        # Create a hash of the dataset ID
        dataset_hash = hashlib.md5(dataset_id.encode()).hexdigest()
        
        # Check if cached version exists
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "datasets", dataset_hash)
        
        if os.path.exists(cache_dir):
            logger.info(f"Loading cached version of {dataset_name} from {cache_dir}")
            try:
                # Try to load from cache
                return Dataset.load_from_disk(cache_dir)
            except Exception as e:
                logger.error(f"Error loading cached dataset {dataset_name}: {e}")
                raise ValueError(f"Dataset {dataset_name} not available and no valid cache found")
        else:
            raise ValueError(f"Dataset {dataset_name} not available and no cache found")
    
    def load_multiple_datasets(self, dataset_names: List[str], split: str = 'train',
                              streaming: bool = False) -> Union[Dataset, IterableDataset]:
        """
        Load and combine multiple datasets.
        
        Args:
            dataset_names: List of dataset names to load
            split: Split to load for each dataset
            streaming: Whether to stream the datasets
            
        Returns:
            Combined dataset
        """
        logger.info(f"Loading multiple datasets: {dataset_names}")
        
        datasets_list = []
        
        for dataset_name in dataset_names:
            try:
                # Get dataset configuration
                dataset_config = self._get_dataset_config(dataset_name)
                
                # Determine if dataset should be streamed
                dataset_streaming = streaming
                if streaming is None and 'streaming' in dataset_config:
                    dataset_streaming = dataset_config['streaming']
                
                # Get max samples
                max_samples = dataset_config.get('max_samples')
                
                # Load dataset
                dataset = self.load_dataset(
                    dataset_name,
                    split=split,
                    streaming=dataset_streaming,
                    max_samples=max_samples
                )
                
                # Apply dataset-specific weight if needed
                weight = dataset_config.get('weight', 1.0)
                if weight != 1.0 and not dataset_streaming:
                    # For non-streaming datasets, we can duplicate or sample
                    if weight > 1.0:
                        # Duplicate examples
                        repeat_factor = int(weight)
                        remainder = weight - repeat_factor
                        
                        # Full duplications
                        for _ in range(repeat_factor - 1):
                            datasets_list.append(dataset)
                        
                        # Partial duplication for remainder
                        if remainder > 0:
                            sample_size = int(len(dataset) * remainder)
                            if sample_size > 0:
                                sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
                                datasets_list.append(dataset.select(sample_indices))
                    
                    elif weight < 1.0:
                        # Sample a subset
                        sample_size = int(len(dataset) * weight)
                        if sample_size > 0:
                            sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
                            dataset = dataset.select(sample_indices)
                
                datasets_list.append(dataset)
                
            except Exception as e:
                logger.warning(f"Error loading dataset {dataset_name}: {e}. Skipping.")
        
        if not datasets_list:
            raise ValueError("No datasets could be loaded")
        
        # Combine datasets
        if streaming:
            # For streaming datasets, we need to interleave them
            return self._interleave_datasets(datasets_list)
        else:
            # For regular datasets, we can concatenate them
            return concatenate_datasets(datasets_list)
    
    def _interleave_datasets(self, datasets_list: List[IterableDataset]) -> IterableDataset:
        """
        Interleave multiple streaming datasets.
        
        Args:
            datasets_list: List of streaming datasets to interleave
            
        Returns:
            Interleaved dataset
        """
        from itertools import cycle, islice
        
        def interleave_generator():
            # Create cyclic iterators for each dataset
            iterators = [cycle(dataset) for dataset in datasets_list]
            
            # Interleave the datasets
            while True:
                for iterator in iterators:
                    try:
                        yield next(iterator)
                    except StopIteration:
                        # This should not happen with cycle, but just in case
                        continue
        
        # Create an iterable dataset from the generator
        return IterableDataset.from_generator(interleave_generator)
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        try:
            # Get dataset configuration
            dataset_config = self._get_dataset_config(dataset_name)
            
            # Load a small sample to get schema
            dataset = self.load_dataset(dataset_name, max_samples=5)
            
            # Get features
            if isinstance(dataset, DatasetDict):
                features = next(iter(dataset.values())).features
                num_examples = {k: len(v) for k, v in dataset.items()}
            else:
                features = dataset.features
                num_examples = len(dataset)
            
            # Compile information
            info = {
                'name': dataset_name,
                'type': dataset_config.get('type', 'huggingface'),
                'task': dataset_config.get('task', 'unknown'),
                'features': str(features),
                'num_examples': num_examples,
                'streaming': dataset_config.get('streaming', False),
                'weight': dataset_config.get('weight', 1.0)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_name}: {e}")
            return {
                'name': dataset_name,
                'error': str(e)
            }
    
    def get_all_datasets_info(self) -> List[Dict]:
        """
        Get information about all configured datasets.
        
        Returns:
            List of dictionaries with dataset information
        """
        all_datasets = []
        
        # Add core datasets
        for dataset in self.config['datasets']['core_datasets']:
            all_datasets.append(dataset['name'])
        
        # Add additional datasets
        for dataset in self.config['datasets']['additional_datasets']:
            all_datasets.append(dataset['name'])
        
        # Get info for each dataset
        return [self.get_dataset_info(dataset) for dataset in all_datasets]
    
    def prepare_dataset_for_training(self, dataset_names: List[str], split: str = 'train',
                                    preprocessing_fn=None) -> Union[Dataset, IterableDataset]:
        """
        Prepare multiple datasets for training by loading, combining, and preprocessing them.
        
        Args:
            dataset_names: List of dataset names to load
            split: Split to load for each dataset
            preprocessing_fn: Function to apply for preprocessing
            
        Returns:
            Prepared dataset ready for training
        """
        # Load and combine datasets
        combined_dataset = self.load_multiple_datasets(dataset_names, split=split)
        
        # Apply preprocessing if provided
        if preprocessing_fn is not None:
            if isinstance(combined_dataset, IterableDataset):
                combined_dataset = combined_dataset.map(preprocessing_fn)
            else:
                combined_dataset = combined_dataset.map(
                    preprocessing_fn,
                    batched=True,
                    num_proc=os.cpu_count()
                )
        
        return combined_dataset


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize loader with config
    loader = DatasetLoader("configs/config.yaml")
    
    # Get information about all datasets
    datasets_info = loader.get_all_datasets_info()
    print(f"Found {len(datasets_info)} datasets")
    
    # Load a sample dataset
    sample_dataset = loader.load_dataset("openai/openai_humaneval", max_samples=5)
    print(f"Loaded sample dataset with {len(sample_dataset)} examples")
    print(f"Sample features: {sample_dataset.features}")
    
    # Prepare datasets for a training stage
    training_stage = loader.config['training']['stages'][0]
    training_datasets = training_stage['datasets']
    
    print(f"Preparing datasets for {training_stage['name']} stage: {training_datasets}")
    
    # Define a simple preprocessing function
    def preprocess_fn(examples):
        # Add a prefix based on the task
        return examples
    
    # Load and prepare datasets for the stage
    prepared_dataset = loader.prepare_dataset_for_training(
        training_datasets,
        split='train',
        preprocessing_fn=preprocess_fn
    )
    
    print(f"Prepared dataset ready for training")