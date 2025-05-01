"""
Dataset management utilities.
This module provides utilities for managing dataset configurations.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Class for managing dataset configurations.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize dataset manager.
        
        Args:
            config_path: Path to dataset configuration file
        """
        # Set config path
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'dataset_config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dataset configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded dataset configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading dataset configuration: {e}")
            # Return empty config
            return {
                'datasets': {
                    'core_datasets': [],
                    'additional_datasets': []
                },
                'error_handling': {
                    'fallbacks': {}
                },
                'training': {
                    'stages': []
                },
                'output_dir': '/tmp/model-trainer/models'
            }
    
    def save_config(self, config: Dict[str, Any] = None) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration to save (default: current configuration)
            
        Returns:
            True if successful, False otherwise
        """
        if config is None:
            config = self.config
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Saved dataset configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving dataset configuration: {e}")
            return False
    
    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """
        Get all datasets from configuration.
        
        Returns:
            List of dataset configurations
        """
        core_datasets = self.config.get('datasets', {}).get('core_datasets', [])
        additional_datasets = self.config.get('datasets', {}).get('additional_datasets', [])
        
        return core_datasets + additional_datasets
    
    def get_core_datasets(self) -> List[Dict[str, Any]]:
        """
        Get core datasets from configuration.
        
        Returns:
            List of core dataset configurations
        """
        return self.config.get('datasets', {}).get('core_datasets', [])
    
    def get_additional_datasets(self) -> List[Dict[str, Any]]:
        """
        Get additional datasets from configuration.
        
        Returns:
            List of additional dataset configurations
        """
        return self.config.get('datasets', {}).get('additional_datasets', [])
    
    def get_dataset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset configuration by name.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset configuration or None if not found
        """
        for dataset in self.get_all_datasets():
            if dataset.get('name') == name:
                return dataset
        
        return None
    
    def get_training_stages(self) -> List[Dict[str, Any]]:
        """
        Get training stages from configuration.
        
        Returns:
            List of training stage configurations
        """
        return self.config.get('training', {}).get('stages', [])
    
    def get_training_stage_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get training stage configuration by name.
        
        Args:
            name: Training stage name
            
        Returns:
            Training stage configuration or None if not found
        """
        for stage in self.get_training_stages():
            if stage.get('name') == name:
                return stage
        
        return None
    
    def add_dataset(self, dataset: Dict[str, Any], core: bool = False) -> bool:
        """
        Add a dataset to the configuration.
        
        Args:
            dataset: Dataset configuration
            core: Whether to add to core datasets (True) or additional datasets (False)
            
        Returns:
            True if successful, False otherwise
        """
        # Check if dataset already exists
        existing_dataset = self.get_dataset_by_name(dataset.get('name'))
        if existing_dataset:
            logger.warning(f"Dataset {dataset.get('name')} already exists")
            return False
        
        # Add dataset
        if core:
            self.config.get('datasets', {}).get('core_datasets', []).append(dataset)
        else:
            self.config.get('datasets', {}).get('additional_datasets', []).append(dataset)
        
        # Save configuration
        return self.save_config()
    
    def remove_dataset(self, name: str) -> bool:
        """
        Remove a dataset from the configuration.
        
        Args:
            name: Dataset name
            
        Returns:
            True if successful, False otherwise
        """
        # Check if dataset exists
        existing_dataset = self.get_dataset_by_name(name)
        if not existing_dataset:
            logger.warning(f"Dataset {name} not found")
            return False
        
        # Remove from core datasets
        core_datasets = self.config.get('datasets', {}).get('core_datasets', [])
        self.config['datasets']['core_datasets'] = [d for d in core_datasets if d.get('name') != name]
        
        # Remove from additional datasets
        additional_datasets = self.config.get('datasets', {}).get('additional_datasets', [])
        self.config['datasets']['additional_datasets'] = [d for d in additional_datasets if d.get('name') != name]
        
        # Save configuration
        return self.save_config()
    
    def add_training_stage(self, stage: Dict[str, Any]) -> bool:
        """
        Add a training stage to the configuration.
        
        Args:
            stage: Training stage configuration
            
        Returns:
            True if successful, False otherwise
        """
        # Check if stage already exists
        existing_stage = self.get_training_stage_by_name(stage.get('name'))
        if existing_stage:
            logger.warning(f"Training stage {stage.get('name')} already exists")
            return False
        
        # Add stage
        self.config.get('training', {}).get('stages', []).append(stage)
        
        # Save configuration
        return self.save_config()
    
    def remove_training_stage(self, name: str) -> bool:
        """
        Remove a training stage from the configuration.
        
        Args:
            name: Training stage name
            
        Returns:
            True if successful, False otherwise
        """
        # Check if stage exists
        existing_stage = self.get_training_stage_by_name(name)
        if not existing_stage:
            logger.warning(f"Training stage {name} not found")
            return False
        
        # Remove stage
        stages = self.config.get('training', {}).get('stages', [])
        self.config['training']['stages'] = [s for s in stages if s.get('name') != name]
        
        # Save configuration
        return self.save_config()

# Example usage
if __name__ == "__main__":
    manager = DatasetManager()
    
    # Get all datasets
    datasets = manager.get_all_datasets()
    print(f"Found {len(datasets)} datasets")
    
    # Get core datasets
    core_datasets = manager.get_core_datasets()
    print(f"Found {len(core_datasets)} core datasets")
    
    # Get additional datasets
    additional_datasets = manager.get_additional_datasets()
    print(f"Found {len(additional_datasets)} additional datasets")
    
    # Get training stages
    stages = manager.get_training_stages()
    print(f"Found {len(stages)} training stages")
