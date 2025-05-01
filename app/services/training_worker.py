"""
Training worker service.
This module provides a service for training models in the background.
"""

import os
import sys
import time
import json
import logging
import threading
import psutil
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np

# Set the path to include the main app directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config.settings import (
    STORAGE_DIR, MAX_MEMORY_MB, TRAINING_CONFIG, 
    MODEL_SIZES, API_SERVICE_URL, MONITOR_SERVICE_URL
)
from app.utils.memory_tracker import MemoryTracker
from app.utils.task_queue import TaskQueue
from src.data.loaders import DatasetLoader
from src.data.preprocessors import DataPreprocessor
from src.model.architecture import TransformerModel, ModelConfig
from src.model.compatibility_wrapper import CompatibilityWrapper
from src.model.training import Trainer, TrainingArguments

# Configure logging
# On Render.com free tier, we only use stream handler to avoid issues with ephemeral storage
if os.environ.get('RENDER_SERVICE_TYPE', ''):
    # Running on Render.com - use only stream handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Running on Render.com - using stream logging only (no log files)")
else:
    # Local development - can use file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(STORAGE_DIR, 'logs', 'training_worker.log'))
        ]
    )
    logger = logging.getLogger(__name__)

class TrainingWorker:
    """
    Worker for training models in the background.
    """
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        """
        Initialize training worker.
        
        Args:
            storage_dir: Directory for storing model artifacts
        """
        self.storage_dir = storage_dir
        self.models_dir = os.path.join(storage_dir, 'models')
        self.exports_dir = os.path.join(storage_dir, 'exports')
        self.datasets_dir = os.path.join(storage_dir, 'datasets')
        self.logs_dir = os.path.join(storage_dir, 'logs')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize memory tracker
        self.memory_tracker = MemoryTracker()
        
        # Initialize task queue
        self.task_queue = TaskQueue()
        
        # Initialize active jobs
        self.active_jobs = {}
        self.job_locks = {}
        
        # Dataset configuration
        self.dataset_config_path = os.path.join(self.datasets_dir, 'config.yaml')
        self._ensure_dataset_config()
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader(self.dataset_config_path)
        
        logger.info("Training worker initialized")
    
    def _ensure_dataset_config(self) -> None:
        """
        Ensure dataset configuration file exists.
        """
        if not os.path.exists(self.dataset_config_path):
            from app.config.settings import DATASETS
            
            # Create basic dataset configuration
            config = {
                'datasets': {
                    'core_datasets': DATASETS['core'],
                    'additional_datasets': DATASETS['additional']
                },
                'error_handling': {
                    'fallbacks': {
                        'dataset_unavailable': 'use_cached'
                    }
                },
                'training': {
                    'stages': [
                        {
                            'name': 'pretrain',
                            'datasets': [d['name'] for d in DATASETS['core'][:3]],
                            'epochs': 3,
                            'learning_rate': {
                                'initial': 5e-5,
                                'min': 1e-5,
                                'schedule': 'linear',
                                'warmup_steps': 500
                            }
                        },
                        {
                            'name': 'finetune',
                            'datasets': [d['name'] for d in DATASETS['additional']],
                            'epochs': 2,
                            'learning_rate': {
                                'initial': 2e-5,
                                'min': 5e-6,
                                'schedule': 'cosine',
                                'warmup_steps': 200
                            }
                        }
                    ]
                },
                'output_dir': self.models_dir
            }
            
            # Save configuration
            with open(self.dataset_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Created dataset configuration at {self.dataset_config_path}")
    
    def start(self) -> None:
        """
        Start the training worker.
        """
        logger.info("Starting training worker")
        
        # Create Flask application for handling API requests
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            memory_info = self.memory_tracker.get_memory_info()
            return jsonify({
                'status': 'ok',
                'timestamp': time.time(),
                'memory': memory_info,
                'active_jobs': list(self.active_jobs.keys())
            })
        
        @app.route('/training/start', methods=['POST'])
        def start_training():
            """Start a training job"""
            data = request.json
            training_id = data.get('training_id')
            config = data.get('config')
            
            if not training_id or not config:
                return jsonify({
                    'success': False,
                    'message': 'Missing training_id or config'
                }), 400
            
            # Start training in a background thread
            threading.Thread(
                target=self.train_model,
                args=(training_id, config),
                daemon=True
            ).start()
            
            return jsonify({
                'success': True,
                'message': f"Training {training_id} started",
                'data': {
                    'training_id': training_id,
                    'status': 'starting'
                }
            })
        
        @app.route('/training/status/<training_id>', methods=['GET'])
        def training_status(training_id):
            """Get training job status"""
            # Check if job is active
            if training_id in self.active_jobs:
                job_info = self.active_jobs[training_id]
                return jsonify({
                    'success': True,
                    'data': job_info
                })
            
            # Check task queue for status
            task_info = self.task_queue.get_task_info(training_id)
            if task_info:
                return jsonify({
                    'success': True,
                    'data': task_info
                })
            
            return jsonify({
                'success': False,
                'message': f"Training job {training_id} not found"
            }), 404
        
        @app.route('/training/stop/<training_id>', methods=['POST'])
        def stop_training(training_id):
            """Stop a training job"""
            # Check if job is active
            if training_id in self.active_jobs:
                self.stop_training_job(training_id)
                return jsonify({
                    'success': True,
                    'message': f"Training job {training_id} stopping",
                    'data': {
                        'training_id': training_id,
                        'status': 'stopping'
                    }
                })
            
            return jsonify({
                'success': False,
                'message': f"Training job {training_id} not found or not active"
            }), 404
        
        @app.route('/export/request', methods=['POST'])
        def request_export():
            """Request model export"""
            data = request.json
            export_id = data.get('export_id')
            config = data.get('config')
            
            if not export_id or not config:
                return jsonify({
                    'success': False,
                    'message': 'Missing export_id or config'
                }), 400
            
            # Start export in a background thread
            threading.Thread(
                target=self.export_model,
                args=(export_id, config),
                daemon=True
            ).start()
            
            return jsonify({
                'success': True,
                'message': f"Export {export_id} started",
                'data': {
                    'export_id': export_id,
                    'status': 'starting'
                }
            })
        
        @app.route('/export/status/<export_id>', methods=['GET'])
        def export_status(export_id):
            """Get export job status"""
            # Check if job is active
            if export_id in self.active_jobs:
                job_info = self.active_jobs[export_id]
                return jsonify({
                    'success': True,
                    'data': job_info
                })
            
            # Check task queue for status
            task_info = self.task_queue.get_task_info(export_id)
            if task_info:
                return jsonify({
                    'success': True,
                    'data': task_info
                })
            
            return jsonify({
                'success': False,
                'message': f"Export job {export_id} not found"
            }), 404
        
        # Start Flask application
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port)
    
    def train_model(self, training_id: str, config: Dict[str, Any]) -> None:
        """
        Train a model with the given configuration.
        
        Args:
            training_id: Training job ID
            config: Training configuration
        """
        # Acquire lock for this job
        if training_id not in self.job_locks:
            self.job_locks[training_id] = threading.Lock()
        
        job_lock = self.job_locks[training_id]
        
        # Check if another thread is already handling this job
        if not job_lock.acquire(blocking=False):
            logger.warning(f"Training job {training_id} is already being processed")
            return
        
        try:
            # Update task status
            self.task_queue.update_task_status(
                training_id, 
                'running', 
                'Initializing training',
                0.0
            )
            
            # Update active jobs
            self.active_jobs[training_id] = {
                'id': training_id,
                'type': 'training',
                'config': config,
                'status': 'running',
                'progress': 0.0,
                'message': 'Initializing training',
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'metrics': {},
                'stop_requested': False
            }
            
            # Create model directory
            model_dir = os.path.join(self.models_dir, training_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Configure model
            model_size = config.get('model_size', 'small')
            model_config = self._create_model_config(model_size)
            
            # Update status
            self._update_job_status(training_id, 'Loading datasets', 5.0)
            
            # Load and prepare datasets
            try:
                train_dataset, eval_dataset = self._prepare_datasets(config.get('datasets', []))
            except Exception as e:
                logger.exception(f"Error preparing datasets: {e}")
                self._update_job_status(training_id, f"Error preparing datasets: {str(e)}", 0.0, 'failed')
                return
            
            # Update status
            self._update_job_status(training_id, 'Creating model', 10.0)
            
            # Create model
            try:
                model, tokenizer = self._create_model(model_config)
            except Exception as e:
                logger.exception(f"Error creating model: {e}")
                self._update_job_status(training_id, f"Error creating model: {str(e)}", 0.0, 'failed')
                return
            
            # Update status
            self._update_job_status(training_id, 'Preparing training', 15.0)
            
            # Create training arguments
            training_args = self._create_training_args(config, training_id)
            
            # Initialize trainer
            try:
                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    args=training_args
                )
            except Exception as e:
                logger.exception(f"Error initializing trainer: {e}")
                self._update_job_status(training_id, f"Error initializing trainer: {str(e)}", 0.0, 'failed')
                return
            
            # Update status
            self._update_job_status(training_id, 'Starting training', 20.0)
            
            # Start training
            try:
                # Register progress callback
                def progress_callback(metrics):
                    # Check if stop requested
                    if self.active_jobs[training_id].get('stop_requested', False):
                        logger.info(f"Stop requested for training job {training_id}")
                        return False
                    
                    # Calculate progress (20-90%)
                    if 'epoch' in metrics and 'num_epochs' in metrics:
                        progress = 20.0 + (metrics['epoch'] / metrics['num_epochs']) * 70.0
                    else:
                        progress = self.active_jobs[training_id].get('progress', 20.0)
                    
                    # Update status with metrics
                    self._update_job_status(
                        training_id,
                        f"Training epoch {metrics.get('epoch', 0)}/{metrics.get('num_epochs', 0)}",
                        progress,
                        'running',
                        metrics
                    )
                    
                    # Continue training unless stop requested
                    return not self.active_jobs[training_id].get('stop_requested', False)
                
                # Set callback
                trainer.set_progress_callback(progress_callback)
                
                # Train model
                training_result = trainer.train()
                
                # Save final model and tokenizer
                final_model_path = os.path.join(model_dir, 'final')
                trainer.save_model(final_model_path)
                tokenizer.save_pretrained(final_model_path)
                
                # Save training results
                with open(os.path.join(model_dir, 'training_results.json'), 'w') as f:
                    json.dump(training_result, f, indent=2)
                
                # Update status
                self._update_job_status(
                    training_id,
                    'Training completed',
                    100.0,
                    'completed',
                    training_result,
                    {
                        'model_path': final_model_path,
                        'training_time': training_result.get('training_time', 0)
                    }
                )
                
            except Exception as e:
                logger.exception(f"Error during training: {e}")
                self._update_job_status(training_id, f"Error during training: {str(e)}", 0.0, 'failed')
                return
            
        finally:
            # Remove from active jobs if it failed or completed
            status = self.active_jobs[training_id].get('status')
            if status in ['completed', 'failed', 'cancelled']:
                del self.active_jobs[training_id]
            
            # Free memory explicitly
            self.memory_tracker.free_memory()
            
            # Release lock
            job_lock.release()
            
            logger.info(f"Training job {training_id} finished with status: {status}")
    
    def export_model(self, export_id: str, config: Dict[str, Any]) -> None:
        """
        Export a trained model.
        
        Args:
            export_id: Export job ID
            config: Export configuration
        """
        # Acquire lock for this job
        if export_id not in self.job_locks:
            self.job_locks[export_id] = threading.Lock()
        
        job_lock = self.job_locks[export_id]
        
        # Check if another thread is already handling this job
        if not job_lock.acquire(blocking=False):
            logger.warning(f"Export job {export_id} is already being processed")
            return
        
        try:
            # Update task status
            self.task_queue.update_task_status(
                export_id, 
                'running', 
                'Initializing export',
                0.0
            )
            
            # Update active jobs
            self.active_jobs[export_id] = {
                'id': export_id,
                'type': 'export',
                'config': config,
                'status': 'running',
                'progress': 0.0,
                'message': 'Initializing export',
                'started_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Get export parameters
            model_id = config.get('model_id')
            export_format = config.get('format', 'flask')
            quantize = config.get('quantize', False)
            optimization_level = config.get('optimization_level', 1)
            
            # Check if model exists
            model_dir = os.path.join(self.models_dir, model_id)
            final_model_path = os.path.join(model_dir, 'final')
            
            if not os.path.exists(final_model_path):
                logger.error(f"Model {model_id} not found at {final_model_path}")
                self._update_job_status(export_id, f"Model {model_id} not found", 0.0, 'failed')
                return
            
            # Create export directory
            export_dir = os.path.join(self.exports_dir, export_id)
            os.makedirs(export_dir, exist_ok=True)
            
            # Update status
            self._update_job_status(export_id, 'Loading model', 10.0)
            
            try:
                # Load model and tokenizer
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                model = AutoModelForCausalLM.from_pretrained(final_model_path)
                tokenizer = AutoTokenizer.from_pretrained(final_model_path)
                
                # Make sure model is in evaluation mode
                model.eval()
                
                # Update status
                self._update_job_status(export_id, 'Wrapping model for compatibility', 20.0)
                
                # Create compatibility wrapper
                wrapped_model = CompatibilityWrapper(model)
                
                # Update status
                self._update_job_status(export_id, f'Exporting model to {export_format} format', 30.0)
                
                if export_format == 'flask':
                    # Export for Flask
                    self._export_for_flask(export_id, wrapped_model, tokenizer, export_dir, quantize)
                elif export_format == 'coreml':
                    # Export for CoreML
                    self._export_for_coreml(export_id, wrapped_model, tokenizer, export_dir, optimization_level)
                else:
                    logger.error(f"Unsupported export format: {export_format}")
                    self._update_job_status(export_id, f"Unsupported export format: {export_format}", 0.0, 'failed')
                    return
                
            except Exception as e:
                logger.exception(f"Error during export: {e}")
                self._update_job_status(export_id, f"Error during export: {str(e)}", 0.0, 'failed')
                return
            
        finally:
            # Remove from active jobs if it failed or completed
            status = self.active_jobs[export_id].get('status')
            if status in ['completed', 'failed', 'cancelled']:
                del self.active_jobs[export_id]
            
            # Free memory explicitly
            self.memory_tracker.free_memory()
            
            # Release lock
            job_lock.release()
            
            logger.info(f"Export job {export_id} finished with status: {status}")
    
    def _export_for_flask(self, export_id: str, model, tokenizer, export_dir: str, quantize: bool) -> None:
        """
        Export model for use in Flask applications.
        
        Args:
            export_id: Export job ID
            model: Model to export
            tokenizer: Tokenizer to export
            export_dir: Directory to save export
            quantize: Whether to quantize the model
        """
        # Update status
        self._update_job_status(export_id, 'Preparing model for Flask export', 40.0)
        
        # Create Flask export directory
        flask_dir = os.path.join(export_dir, 'flask')
        os.makedirs(flask_dir, exist_ok=True)
        
        try:
            # Quantize model if requested
            if quantize:
                self._update_job_status(export_id, 'Quantizing model', 50.0)
                
                # Quantize model to int8
                try:
                    from torch.quantization import quantize_dynamic
                    model = quantize_dynamic(
                        model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                    logger.info(f"Model quantized to int8")
                except Exception as e:
                    logger.warning(f"Error quantizing model: {e}, continuing with full precision")
            
            # Update status
            self._update_job_status(export_id, 'Saving model', 60.0)
            
            # Save model using TorchScript
            script_path = os.path.join(flask_dir, 'model.pt')
            
            # Create example inputs for tracing
            input_ids = torch.zeros((1, 10), dtype=torch.long)
            attention_mask = torch.ones((1, 10), dtype=torch.long)
            
            # Trace model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model,
                    (input_ids, attention_mask)
                )
            
            # Save traced model
            traced_model.save(script_path)
            
            # Save tokenizer
            tokenizer.save_pretrained(flask_dir)
            
            # Create model info
            model_info = {
                'export_id': export_id,
                'export_date': datetime.now().isoformat(),
                'format': 'flask',
                'quantized': quantize,
                'vocab_size': tokenizer.vocab_size,
                'model_type': type(model.base_model).__name__,
                'inputs': {
                    'input_ids': {'shape': [-1, -1], 'dtype': 'long'},
                    'attention_mask': {'shape': [-1, -1], 'dtype': 'long'}
                },
                'outputs': {
                    'logits': {'shape': [-1, -1, -1], 'dtype': 'float32'}
                }
            }
            
            # Save model info
            with open(os.path.join(flask_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Create example Flask application
            self._create_flask_example(flask_dir)
            
            # Create zip file
            self._update_job_status(export_id, 'Creating archive', 80.0)
            
            import shutil
            zip_path = os.path.join(export_dir, 'flask_model.zip')
            shutil.make_archive(
                os.path.splitext(zip_path)[0],
                'zip',
                flask_dir
            )
            
            # Update status
            self._update_job_status(
                export_id,
                'Flask export completed',
                100.0,
                'completed',
                None,
                {
                    'export_file': zip_path,
                    'export_size': os.path.getsize(zip_path),
                    'model_info': model_info
                }
            )
            
        except Exception as e:
            logger.exception(f"Error exporting model for Flask: {e}")
            self._update_job_status(export_id, f"Error exporting model for Flask: {str(e)}", 0.0, 'failed')
    
    def _export_for_coreml(self, export_id: str, model, tokenizer, export_dir: str, optimization_level: int) -> None:
        """
        Export model to Apple's CoreML format.
        
        Args:
            export_id: Export job ID
            model: Model to export
            tokenizer: Tokenizer to export
            export_dir: Directory to save export
            optimization_level: Optimization level (0-3)
        """
        # Update status
        self._update_job_status(export_id, 'Preparing model for CoreML export', 40.0)
        
        # Create CoreML export directory
        coreml_dir = os.path.join(export_dir, 'coreml')
        os.makedirs(coreml_dir, exist_ok=True)
        
        try:
            # Check if coremltools is available
            try:
                import coremltools as ct
            except ImportError:
                logger.error("coremltools package is not available for CoreML export")
                self._update_job_status(export_id, "coremltools package is not available", 0.0, 'failed')
                return
            
            # Update status
            self._update_job_status(export_id, 'Converting model to ONNX format', 50.0)
            
            # Convert model to ONNX first
            onnx_path = os.path.join(coreml_dir, 'model.onnx')
            
            # Create example inputs for tracing
            input_ids = torch.zeros((1, 10), dtype=torch.long)
            attention_mask = torch.ones((1, 10), dtype=torch.long)
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (input_ids, attention_mask),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'sequence'},
                        'attention_mask': {0: 'batch', 1: 'sequence'},
                        'logits': {0: 'batch', 1: 'sequence', 2: 'vocab'}
                    },
                    opset_version=12
                )
            
            # Update status
            self._update_job_status(export_id, 'Converting ONNX to CoreML', 70.0)
            
            # Load ONNX model
            onnx_model = ct.converters.onnx.load(onnx_path)
            
            # Set optimization parameters based on level
            if optimization_level >= 3:
                compute_precision = ct.precision.FLOAT16
                compute_units = ct.ComputeUnit.ALL
            elif optimization_level == 2:
                compute_precision = ct.precision.FLOAT16
                compute_units = ct.ComputeUnit.CPU_AND_GPU
            elif optimization_level == 1:
                compute_precision = ct.precision.FLOAT32
                compute_units = ct.ComputeUnit.CPU_ONLY
            else:
                compute_precision = ct.precision.FLOAT32
                compute_units = ct.ComputeUnit.CPU_ONLY
            
            # Convert to CoreML
            mlmodel = ct.convert(
                onnx_model,
                convert_to="mlprogram",
                compute_precision=compute_precision,
                compute_units=compute_units
            )
            
            # Add metadata
            mlmodel.author = "Model Trainer"
            mlmodel.license = "MIT"
            mlmodel.version = "1.0"
            mlmodel.short_description = f"Language model exported from Model Trainer"
            
            # Save CoreML model
            mlmodel_path = os.path.join(coreml_dir, 'model.mlmodel')
            mlmodel.save(mlmodel_path)
            
            # Save tokenizer configuration
            tokenizer_config = tokenizer.to_dict()
            with open(os.path.join(coreml_dir, 'tokenizer_config.json'), 'w') as f:
                json.dump(tokenizer_config, f, indent=2)
            
            # Save vocabulary
            vocab = tokenizer.get_vocab()
            with open(os.path.join(coreml_dir, 'vocab.json'), 'w') as f:
                json.dump(vocab, f, indent=2)
            
            # Create model info
            model_info = {
                'export_id': export_id,
                'export_date': datetime.now().isoformat(),
                'format': 'coreml',
                'optimization_level': optimization_level,
                'compute_precision': str(compute_precision),
                'compute_units': str(compute_units),
                'vocab_size': tokenizer.vocab_size,
                'model_type': type(model.base_model).__name__,
                'inputs': {
                    'input_ids': {'shape': [-1, -1], 'dtype': 'int32'},
                    'attention_mask': {'shape': [-1, -1], 'dtype': 'int32'}
                },
                'outputs': {
                    'logits': {'shape': [-1, -1, -1], 'dtype': 'float32'}
                }
            }
            
            # Save model info
            with open(os.path.join(coreml_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Create Swift example code
            self._create_swift_example(coreml_dir)
            
            # Create zip file
            self._update_job_status(export_id, 'Creating archive', 90.0)
            
            import shutil
            zip_path = os.path.join(export_dir, 'coreml_model.zip')
            shutil.make_archive(
                os.path.splitext(zip_path)[0],
                'zip',
                coreml_dir
            )
            
            # Update status
            self._update_job_status(
                export_id,
                'CoreML export completed',
                100.0,
                'completed',
                None,
                {
                    'export_file': zip_path,
                    'export_size': os.path.getsize(zip_path),
                    'model_info': model_info
                }
            )
            
        except Exception as e:
            logger.exception(f"Error exporting model to CoreML: {e}")
            self._update_job_status(export_id, f"Error exporting model to CoreML: {str(e)}", 0.0, 'failed')
    
    def _create_flask_example(self, export_dir: str) -> None:
        """
        Create example Flask application for the exported model.
        
        Args:
            export_dir: Directory where model is exported
        """
        app_code = """
from flask import Flask, request, jsonify
import torch
import os
import json
from transformers import AutoTokenizer

app = Flask(__name__)

# Load model and tokenizer
@app.before_first_request
def load_model():
    global model, tokenizer
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    model = torch.jit.load(model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__))

@app.route('/generate', methods=['POST'])
def generate():
    # Get request data
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 1.0)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=temperature > 0.0
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({
        'prompt': prompt,
        'response': response,
        'tokens': len(outputs[0])
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': 'model' in globals()
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
"""
        
        requirements = """
flask>=2.0.1
torch>=1.13.0
transformers>=4.30.0
"""
        
        readme = """
# Flask Model Server

This is an example Flask application for serving the exported model.

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python app.py
```

## API Endpoints

### Generate Text

```
POST /generate
```

Request body:
```json
{
    "prompt": "Once upon a time",
    "max_length": 50,
    "temperature": 0.7
}
```

Response:
```json
{
    "prompt": "Once upon a time",
    "response": "Once upon a time there was a great king who ruled...",
    "tokens": 62
}
```

### Health Check

```
GET /health
```

Response:
```json
{
    "status": "ok",
    "model_loaded": true
}
```
"""
        
        # Write files
        with open(os.path.join(export_dir, 'app.py'), 'w') as f:
            f.write(app_code.strip())
        
        with open(os.path.join(export_dir, 'requirements.txt'), 'w') as f:
            f.write(requirements.strip())
        
        with open(os.path.join(export_dir, 'README.md'), 'w') as f:
            f.write(readme.strip())
    
    def _create_swift_example(self, export_dir: str) -> None:
        """
        Create example Swift code for the exported CoreML model.
        
        Args:
            export_dir: Directory where model is exported
        """
        swift_code = """
import CoreML
import Foundation

class LanguageModel {
    private let model: MLModel
    private let tokenizer: Tokenizer
    
    init() throws {
        // Load the model
        let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel")!
        model = try MLModel(contentsOf: modelURL)
        
        // Initialize tokenizer
        tokenizer = try Tokenizer()
    }
    
    func generate(prompt: String, maxLength: Int = 50, temperature: Double = 1.0) throws -> String {
        // Tokenize the prompt
        let inputTokens = tokenizer.encode(text: prompt)
        
        // Create input array
        var inputIds = inputTokens
        var attentionMask = [Int](repeating: 1, count: inputTokens.count)
        
        // Generate tokens
        for _ in 0..<maxLength {
            // Create input dictionary
            let inputDict: [String: Any] = [
                "input_ids": MLMultiArray(inputIds),
                "attention_mask": MLMultiArray(attentionMask)
            ]
            
            // Run inference
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
            
            // Get logits
            let logits = output.featureValue(for: "logits")!.multiArrayValue!
            
            // Get next token (simplified - sampling logic would be more complex in real app)
            let nextToken = getNextToken(logits: logits, temperature: temperature)
            
            // Append to input
            inputIds.append(nextToken)
            attentionMask.append(1)
            
            // Check for end of sequence token
            if nextToken == tokenizer.eosToken {
                break
            }
        }
        
        // Decode the generated tokens
        return tokenizer.decode(tokens: inputIds)
    }
    
    private func getNextToken(logits: MLMultiArray, temperature: Double) -> Int {
        // Simplified token sampling logic
        // In a real app, you would need to implement proper temperature sampling
        
        // Get the last token logits
        let lastTokenLogits = logits[logits.shape[0] - 1, logits.shape[1] - 1]
        
        // Just return the argmax for this example
        return 0 // Placeholder - actual implementation needed
    }
}

// Placeholder for a tokenizer class
class Tokenizer {
    let eosToken: Int
    
    init() throws {
        // Load vocabulary from vocab.json
        eosToken = 50256 // Typical GPT-2 EOS token, adjust as needed
    }
    
    func encode(text: String) -> [Int] {
        // Placeholder for encoding logic
        return [0] // Placeholder - actual implementation needed
    }
    
    func decode(tokens: [Int]) -> String {
        // Placeholder for decoding logic
        return "" // Placeholder - actual implementation needed
    }
}
"""
        
        readme = """
# CoreML Model Usage

This is an example of how to use the exported CoreML model in a Swift application.

## Integration

1. Add the `model.mlmodel` file to your Xcode project
2. Implement the tokenizer based on the provided `vocab.json` and `tokenizer_config.json`
3. Use the sample code as a starting point for integrating the model

## Sample Code

The provided `ModelUsage.swift` file contains example code for:

1. Loading the CoreML model
2. Implementing a basic tokenizer
3. Text generation with the model

## Notes

- The actual tokenizer implementation is not included as it depends on your specific requirements
- You'll need to implement the token sampling logic based on the logits
- For more advanced usage, consider using Apple's CreateML framework as well
"""
        
        # Write files
        with open(os.path.join(export_dir, 'ModelUsage.swift'), 'w') as f:
            f.write(swift_code.strip())
        
        with open(os.path.join(export_dir, 'README.md'), 'w') as f:
            f.write(readme.strip())
    
    def _update_job_status(self, job_id: str, message: str, progress: float, 
                          status: str = 'running', metrics: Optional[Dict[str, Any]] = None,
                          result: Optional[Dict[str, Any]] = None) -> None:
        """
        Update job status in task queue and active jobs.
        
        Args:
            job_id: Job ID
            message: Status message
            progress: Progress value (0-100)
            status: Job status
            metrics: Optional metrics
            result: Optional result
        """
        # Update task queue
        self.task_queue.update_task_status(
            job_id,
            status,
            message,
            progress,
            result
        )
        
        # Update active jobs
        if job_id in self.active_jobs:
            self.active_jobs[job_id].update({
                'status': status,
                'message': message,
                'progress': progress,
                'updated_at': datetime.now().isoformat()
            })
            
            if metrics is not None:
                self.active_jobs[job_id]['metrics'] = metrics
            
            if result is not None:
                self.active_jobs[job_id]['result'] = result
            
            # Send metrics to monitoring service
            if metrics is not None:
                try:
                    requests.post(
                        f"{MONITOR_SERVICE_URL}/metrics/{job_id}",
                        json={
                            'job_id': job_id,
                            'status': status,
                            'message': message,
                            'progress': progress,
                            'metrics': metrics,
                            'timestamp': time.time()
                        },
                        timeout=2
                    )
                except Exception as e:
                    logger.warning(f"Error sending metrics to monitoring service: {e}")
    
    def stop_training_job(self, training_id: str) -> bool:
        """
        Stop a training job.
        
        Args:
            training_id: Training job ID
            
        Returns:
            True if job was found and stop was requested
        """
        if training_id in self.active_jobs:
            self.active_jobs[training_id]['stop_requested'] = True
            self._update_job_status(training_id, 'Stop requested', self.active_jobs[training_id].get('progress', 0.0), 'stopping')
            return True
        return False
    
    def _create_model_config(self, size: str) -> Dict[str, Any]:
        """
        Create model configuration.
        
        Args:
            size: Model size (small, medium, large)
            
        Returns:
            Model configuration
        """
        from app.config.settings import get_model_config
        return get_model_config(size)
    
    def _prepare_datasets(self, datasets_config: List[Dict[str, Any]]) -> Tuple[Any, Any]:
        """
        Prepare datasets for training.
        
        Args:
            datasets_config: Dataset configuration
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Get dataset names
        dataset_names = [dataset['name'] for dataset in datasets_config]
        
        # Load and prepare datasets
        try:
            # Create dataset preprocessor
            from app.config.settings import get_model_config
            config = get_model_config()
            preprocessor = DataPreprocessor(config)
            
            # Load and combine datasets
            train_datasets = []
            eval_datasets = []
            
            for dataset_config in datasets_config:
                dataset_name = dataset_config['name']
                streaming = dataset_config.get('streaming', False)
                max_samples = dataset_config.get('max_samples', TRAINING_CONFIG['max_train_samples'])
                
                logger.info(f"Loading dataset {dataset_name} (streaming={streaming}, max_samples={max_samples})")
                
                # Load dataset
                dataset = self.dataset_loader.load_dataset(
                    dataset_name,
                    streaming=streaming,
                    max_samples=max_samples
                )
                
                # Split dataset if it's not already split
                if isinstance(dataset, (list, tuple)):
                    train_dataset, eval_dataset = dataset
                else:
                    # Split with 90/10 ratio
                    split = dataset.train_test_split(test_size=0.1)
                    train_dataset = split['train']
                    eval_dataset = split['test']
                
                # Preprocess datasets
                train_dataset = preprocessor.process_dataset(train_dataset)
                eval_dataset = preprocessor.process_dataset(eval_dataset)
                
                # Append to lists
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
            
            # Combine datasets
            if len(train_datasets) > 1:
                from datasets import concatenate_datasets
                train_dataset = concatenate_datasets(train_datasets)
                eval_dataset = concatenate_datasets(eval_datasets)
            else:
                train_dataset = train_datasets[0]
                eval_dataset = eval_datasets[0]
            
            # Limit validation dataset size
            if not isinstance(eval_dataset, (list, tuple)) and hasattr(eval_dataset, '__len__') and len(eval_dataset) > TRAINING_CONFIG['max_validation_samples']:
                eval_dataset = eval_dataset.select(range(TRAINING_CONFIG['max_validation_samples']))
            
            logger.info(f"Prepared datasets with {len(train_dataset) if hasattr(train_dataset, '__len__') else 'unknown'} training examples and {len(eval_dataset) if hasattr(eval_dataset, '__len__') else 'unknown'} validation examples")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.exception(f"Error preparing datasets: {e}")
            raise
    
    def _create_model(self, config: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Create model and tokenizer.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            
            # Create tokenizer
            if os.path.exists(os.path.join(self.models_dir, 'tokenizer')):
                # Load existing tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    os.path.join(self.models_dir, 'tokenizer')
                )
                logger.info("Loaded existing tokenizer")
            else:
                # Create new tokenizer
                tokenizer_config = config['tokenizer']
                tokenizer = PreTrainedTokenizerFast(
                    vocab_size=tokenizer_config['vocab_size'],
                    bos_token=tokenizer_config['special_tokens'].get('bos_token'),
                    eos_token=tokenizer_config['special_tokens'].get('eos_token'),
                    unk_token=tokenizer_config['special_tokens'].get('unk_token'),
                    pad_token=tokenizer_config['special_tokens'].get('pad_token')
                )
                
                # Save tokenizer
                os.makedirs(os.path.join(self.models_dir, 'tokenizer'), exist_ok=True)
                tokenizer.save_pretrained(os.path.join(self.models_dir, 'tokenizer'))
                logger.info("Created new tokenizer")
            
            # Create model configuration
            model_config = ModelConfig(
                vocab_size=config['tokenizer']['vocab_size'],
                hidden_size=config['model']['sizes'][config['model']['size']]['d_model'],
                num_hidden_layers=config['model']['sizes'][config['model']['size']]['n_layers'],
                num_attention_heads=config['model']['sizes'][config['model']['size']]['n_heads'],
                intermediate_size=config['model']['sizes'][config['model']['size']]['d_ff'],
                max_position_embeddings=config['model']['sizes'][config['model']['size']]['max_seq_length'],
                hidden_dropout_prob=config['model']['dropout'],
                attention_probs_dropout_prob=config['model']['dropout'],
                use_rotary_embeddings=config['model']['attention']['rotary_embedding'],
                causal=config['model']['attention']['causal']
            )
            
            # Create model
            model = TransformerModel(
                vocab_size=config['tokenizer']['vocab_size'],
                hidden_size=config['model']['sizes'][config['model']['size']]['d_model'],
                num_hidden_layers=config['model']['sizes'][config['model']['size']]['n_layers'],
                num_attention_heads=config['model']['sizes'][config['model']['size']]['n_heads'],
                intermediate_size=config['model']['sizes'][config['model']['size']]['d_ff'],
                max_position_embeddings=config['model']['sizes'][config['model']['size']]['max_seq_length'],
                hidden_dropout_prob=config['model']['dropout'],
                attention_probs_dropout_prob=config['model']['dropout'],
                use_rotary_embeddings=config['model']['attention']['rotary_embedding'],
                causal=config['model']['attention']['causal']
            )
            
            logger.info(f"Created model with size {config['model']['size']}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.exception(f"Error creating model: {e}")
            raise
    
    def _create_training_args(self, config: Dict[str, Any], training_id: str) -> Any:
        """
        Create training arguments.
        
        Args:
            config: Training configuration
            training_id: Training job ID
            
        Returns:
            Training arguments
        """
        from app.config.settings import TRAINING_CONFIG
        
        # Get configuration values, using defaults if not provided
        epochs = config.get('epochs', TRAINING_CONFIG['max_epochs'])
        batch_size = config.get('batch_size', TRAINING_CONFIG['batch_size'])
        learning_rate = config.get('learning_rate', TRAINING_CONFIG['learning_rate'])
        save_steps = config.get('save_steps', TRAINING_CONFIG['save_steps'])
        eval_steps = config.get('eval_steps', TRAINING_CONFIG['eval_steps'])
        
        # Create output directory
        output_dir = os.path.join(self.models_dir, training_id)
        
        # Create training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=TRAINING_CONFIG['warmup_steps'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=100,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
            gradient_checkpointing=TRAINING_CONFIG['gradient_checkpointing'],
            fp16=TRAINING_CONFIG['mixed_precision'] == 'fp16',
            bf16=TRAINING_CONFIG['mixed_precision'] == 'bf16',
            optim='adamw_torch',
            learning_rate=learning_rate,
            lr_scheduler_type='linear',
            disable_tqdm=False,
            report_to='none',
            remove_unused_columns=False,
            run_name=f"training-{training_id}"
        )
        
        return args


if __name__ == "__main__":
    # Check available memory
    memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    logger.info(f"System memory: {memory_mb:.2f} MB")
    
    # Initialize worker
    worker = TrainingWorker()
    
    # Start worker
    worker.start()
