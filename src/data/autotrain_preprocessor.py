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
AutoTrain dataset preprocessor for fine-tuning language models.
This module provides utilities to prepare datasets for AutoTrain.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import Dataset, DatasetDict

from src.data.preprocessors import TextPreprocessor

# Configure logging
logger = logging.getLogger(__name__)

class AutoTrainPreprocessor:
    """
    Preprocessor for preparing datasets for AutoTrain fine-tuning.
    """
    
    def __init__(self, config: Dict, output_dir: str = "./"):
        """
        Initialize the AutoTrain preprocessor.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save processed datasets
        """
        self.config = config
        self.output_dir = output_dir
        self.text_preprocessor = TextPreprocessor(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_instruction_dataset(self, dataset: Dataset, 
                                   instruction_column: str = "instruction",
                                   response_column: str = "response",
                                   input_column: Optional[str] = None,
                                   output_file: str = "train.csv") -> str:
        """
        Prepare a dataset for instruction fine-tuning.
        
        Args:
            dataset: Input dataset
            instruction_column: Column containing instructions/questions
            response_column: Column containing responses/answers
            input_column: Optional column for additional context
            output_file: Name of the output CSV file
            
        Returns:
            Path to the prepared dataset
        """
        logger.info(f"Preparing instruction dataset with {len(dataset)} examples")
        
        # Define the formatting function
        def format_instruction(example):
            # Extract instruction and response
            instruction = example.get(instruction_column, "")
            response = example.get(response_column, "")
            
            # Add input context if available
            if input_column and input_column in example and example[input_column]:
                instruction = f"{instruction}\n\n{example[input_column]}"
            
            # Clean the text
            instruction = self.text_preprocessor.clean_text(instruction)
            response = self.text_preprocessor.clean_text(response)
            
            # Create the prompt format
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Question:
{instruction}

### Answer:
{response}"""
            
            return {"prompt": prompt}
        
        # Apply the formatting function
        formatted_dataset = dataset.map(format_instruction)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, output_file)
        formatted_dataset.to_csv(output_path)
        logger.info(f"Saved formatted dataset to {output_path}")
        
        return output_path
    
    def prepare_chat_dataset(self, dataset: Dataset,
                           messages_column: str = "messages",
                           output_file: str = "train.csv") -> str:
        """
        Prepare a dataset for chat fine-tuning.
        
        Args:
            dataset: Input dataset
            messages_column: Column containing chat messages
            output_file: Name of the output CSV file
            
        Returns:
            Path to the prepared dataset
        """
        logger.info(f"Preparing chat dataset with {len(dataset)} examples")
        
        # Define the formatting function
        def format_chat(example):
            # Extract messages
            messages = example.get(messages_column, [])
            
            # Format as a conversation
            conversation = ""
            for msg in messages:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                conversation += f"{role}: {content}\n\n"
            
            # Clean the text
            conversation = self.text_preprocessor.clean_text(conversation)
            
            return {"prompt": conversation}
        
        # Apply the formatting function
        formatted_dataset = dataset.map(format_chat)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, output_file)
        formatted_dataset.to_csv(output_path)
        logger.info(f"Saved formatted dataset to {output_path}")
        
        return output_path
    
    def prepare_completion_dataset(self, dataset: Dataset,
                                 text_column: str = "text",
                                 output_file: str = "train.csv") -> str:
        """
        Prepare a dataset for completion fine-tuning.
        
        Args:
            dataset: Input dataset
            text_column: Column containing text
            output_file: Name of the output CSV file
            
        Returns:
            Path to the prepared dataset
        """
        logger.info(f"Preparing completion dataset with {len(dataset)} examples")
        
        # Define the formatting function
        def format_completion(example):
            # Extract text
            text = example.get(text_column, "")
            
            # Clean the text
            text = self.text_preprocessor.clean_text(text)
            
            return {"prompt": text}
        
        # Apply the formatting function
        formatted_dataset = dataset.map(format_completion)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, output_file)
        formatted_dataset.to_csv(output_path)
        logger.info(f"Saved formatted dataset to {output_path}")
        
        return output_path
    
    def detect_dataset_format(self, dataset: Dataset) -> str:
        """
        Detect the format of a dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Detected format: 'instruction', 'chat', or 'completion'
        """
        # Get column names
        columns = dataset.column_names
        
        # Check for instruction format
        if any(col in columns for col in ['instruction', 'question']):
            if any(col in columns for col in ['response', 'answer', 'output']):
                return 'instruction'
        
        # Check for chat format
        if 'messages' in columns or 'conversation' in columns:
            return 'chat'
        
        # Default to completion format
        return 'completion'
    
    def prepare_dataset(self, dataset: Dataset, format_type: Optional[str] = None,
                      output_file: str = "train.csv") -> str:
        """
        Prepare a dataset for fine-tuning, automatically detecting the format if not specified.
        
        Args:
            dataset: Input dataset
            format_type: Format type ('instruction', 'chat', or 'completion')
            output_file: Name of the output CSV file
            
        Returns:
            Path to the prepared dataset
        """
        # Detect format if not specified
        if format_type is None:
            format_type = self.detect_dataset_format(dataset)
        
        logger.info(f"Preparing dataset with format: {format_type}")
        
        # Prepare based on format
        if format_type == 'instruction':
            # Try to find appropriate column names
            instruction_col = next((col for col in ['instruction', 'question', 'input'] 
                                  if col in dataset.column_names), None)
            response_col = next((col for col in ['response', 'answer', 'output'] 
                               if col in dataset.column_names), None)
            
            if not instruction_col or not response_col:
                logger.warning("Could not find appropriate columns for instruction format")
                # Fall back to first two columns
                columns = dataset.column_names
                instruction_col = columns[0]
                response_col = columns[1] if len(columns) > 1 else columns[0]
            
            return self.prepare_instruction_dataset(
                dataset, 
                instruction_column=instruction_col,
                response_column=response_col,
                output_file=output_file
            )
        
        elif format_type == 'chat':
            messages_col = next((col for col in ['messages', 'conversation'] 
                               if col in dataset.column_names), None)
            
            if not messages_col:
                logger.warning("Could not find appropriate columns for chat format")
                # Fall back to completion format
                return self.prepare_completion_dataset(
                    dataset,
                    output_file=output_file
                )
            
            return self.prepare_chat_dataset(
                dataset,
                messages_column=messages_col,
                output_file=output_file
            )
        
        else:  # completion format
            text_col = next((col for col in ['text', 'content', 'document'] 
                           if col in dataset.column_names), dataset.column_names[0])
            
            return self.prepare_completion_dataset(
                dataset,
                text_column=text_col,
                output_file=output_file
            )