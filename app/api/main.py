"""
Main API server for the model training application.
This module defines the Flask API server that handles requests for training and exporting models.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Any
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
import requests

from app.config.settings import (
    API_TITLE, API_DESCRIPTION, API_VERSION, API_PREFIX,
    STORAGE_DIR, TRAINING_SERVICE_URL, MONITOR_SERVICE_URL
)
from app.utils.memory_tracker import MemoryTracker
from app.utils.task_queue import TaskQueue
from app.utils.response_utils import success_response, error_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Initialize API
api = Api(
    app,
    version=API_VERSION,
    title=API_TITLE,
    description=API_DESCRIPTION,
    prefix=API_PREFIX,
    doc="/docs"
)

# Initialize memory tracker
memory_tracker = MemoryTracker()

# Initialize task queue
task_queue = TaskQueue()

# Create namespaces
training_ns = Namespace("training", description="Training operations")
monitoring_ns = Namespace("monitoring", description="Monitoring operations")
export_ns = Namespace("export", description="Model export operations")
dataset_ns = Namespace("datasets", description="Dataset operations")
health_ns = Namespace("health", description="Health checks")

# Add namespaces to API
api.add_namespace(training_ns)
api.add_namespace(monitoring_ns)
api.add_namespace(export_ns)
api.add_namespace(dataset_ns)
api.add_namespace(health_ns)

# Define API models
dataset_model = api.model('Dataset', {
    'name': fields.String(required=True, description='Dataset name'),
    'streaming': fields.Boolean(required=False, description='Whether to stream the dataset'),
    'max_samples': fields.Integer(required=False, description='Maximum number of samples to use')
})

training_config_model = api.model('TrainingConfig', {
    'model_size': fields.String(required=True, description='Model size (small, medium, large)'),
    'datasets': fields.List(fields.Nested(dataset_model), required=True, description='Datasets to use for training'),
    'epochs': fields.Integer(required=False, description='Number of training epochs'),
    'batch_size': fields.Integer(required=False, description='Batch size for training'),
    'learning_rate': fields.Float(required=False, description='Learning rate'),
    'save_steps': fields.Integer(required=False, description='Steps between model checkpoints'),
    'eval_steps': fields.Integer(required=False, description='Steps between evaluations')
})

export_model = api.model('ExportConfig', {
    'model_id': fields.String(required=True, description='ID of the model to export'),
    'format': fields.String(required=True, description='Export format (flask, coreml)'),
    'quantize': fields.Boolean(required=False, description='Whether to quantize the model'),
    'optimization_level': fields.Integer(required=False, description='Optimization level (0-3)')
})

# Health check endpoint
@health_ns.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Check API health"""
        memory_info = memory_tracker.get_memory_info()
        return {
            'status': 'ok',
            'timestamp': time.time(),
            'memory': memory_info,
            'tasks': task_queue.get_queue_info()
        }

# Training endpoints
@training_ns.route('/start')
class StartTraining(Resource):
    @training_ns.expect(training_config_model)
    def post(self):
        """Start a training job"""
        try:
            # Get training configuration
            config = request.json
            
            # Validate configuration
            if not config.get('model_size') or not config.get('datasets'):
                return error_response('Invalid training configuration', 400)
            
            # Check if system has enough memory
            if not memory_tracker.has_sufficient_memory():
                return error_response('Insufficient memory to start training', 503)
            
            # Submit training job
            training_id = task_queue.submit_task('training', config)
            
            # Forward request to training service
            try:
                response = requests.post(
                    f"{TRAINING_SERVICE_URL}/training/start",
                    json={"training_id": training_id, "config": config},
                    timeout=5
                )
                if response.status_code != 200:
                    logger.error(f"Failed to start training job: {response.text}")
                    return error_response(f"Failed to start training job: {response.text}", 500)
            except requests.RequestException as e:
                logger.error(f"Error communicating with training service: {e}")
                # Continue anyway - the task is in the queue and will be processed
            
            return success_response({
                'training_id': training_id,
                'status': 'submitted',
                'config': config
            })
        except Exception as e:
            logger.exception(f"Error starting training: {e}")
            return error_response(f"Error starting training: {str(e)}", 500)

@training_ns.route('/status/<string:training_id>')
class TrainingStatus(Resource):
    def get(self, training_id):
        """Get training job status"""
        try:
            # Check if training job exists
            task_info = task_queue.get_task_info(training_id)
            if not task_info:
                return error_response(f"Training job {training_id} not found", 404)
            
            # Get additional status from training service
            try:
                response = requests.get(
                    f"{TRAINING_SERVICE_URL}/training/status/{training_id}",
                    timeout=5
                )
                if response.status_code == 200:
                    detailed_status = response.json().get('data', {})
                    task_info.update(detailed_status)
            except requests.RequestException as e:
                logger.warning(f"Error getting detailed status from training service: {e}")
            
            return success_response(task_info)
        except Exception as e:
            logger.exception(f"Error getting training status: {e}")
            return error_response(f"Error getting training status: {str(e)}", 500)

@training_ns.route('/stop/<string:training_id>')
class StopTraining(Resource):
    def post(self, training_id):
        """Stop a training job"""
        try:
            # Check if training job exists
            if not task_queue.get_task_info(training_id):
                return error_response(f"Training job {training_id} not found", 404)
            
            # Send stop signal to training service
            try:
                response = requests.post(
                    f"{TRAINING_SERVICE_URL}/training/stop/{training_id}",
                    timeout=5
                )
                if response.status_code != 200:
                    logger.error(f"Failed to stop training job: {response.text}")
                    return error_response(f"Failed to stop training job: {response.text}", 500)
            except requests.RequestException as e:
                logger.error(f"Error communicating with training service: {e}")
                # Continue anyway - update the local task state
            
            # Update task status
            task_queue.update_task_status(training_id, 'stopping')
            
            return success_response({
                'training_id': training_id,
                'status': 'stopping'
            })
        except Exception as e:
            logger.exception(f"Error stopping training: {e}")
            return error_response(f"Error stopping training: {str(e)}", 500)

@training_ns.route('/list')
class ListTrainings(Resource):
    def get(self):
        """List all training jobs"""
        try:
            # Get list of training jobs
            tasks = task_queue.list_tasks('training')
            
            return success_response(tasks)
        except Exception as e:
            logger.exception(f"Error listing training jobs: {e}")
            return error_response(f"Error listing training jobs: {str(e)}", 500)

# Export endpoints
@export_ns.route('/request')
class RequestExport(Resource):
    @export_ns.expect(export_model)
    def post(self):
        """Request model export"""
        try:
            # Get export configuration
            config = request.json
            
            # Validate configuration
            if not config.get('model_id') or not config.get('format'):
                return error_response('Invalid export configuration', 400)
            
            # Check if system has enough memory
            if not memory_tracker.has_sufficient_memory():
                return error_response('Insufficient memory to start export', 503)
            
            # Check if the model exists
            model_dir = os.path.join(STORAGE_DIR, 'models', config['model_id'])
            if not os.path.exists(model_dir):
                return error_response(f"Model {config['model_id']} not found", 404)
            
            # Submit export job
            export_id = task_queue.submit_task('export', config)
            
            # Forward request to training service (which handles exports)
            try:
                response = requests.post(
                    f"{TRAINING_SERVICE_URL}/export/request",
                    json={"export_id": export_id, "config": config},
                    timeout=5
                )
                if response.status_code != 200:
                    logger.error(f"Failed to start export job: {response.text}")
                    return error_response(f"Failed to start export job: {response.text}", 500)
            except requests.RequestException as e:
                logger.error(f"Error communicating with training service: {e}")
                # Continue anyway - the task is in the queue and will be processed
            
            return success_response({
                'export_id': export_id,
                'status': 'submitted',
                'config': config
            })
        except Exception as e:
            logger.exception(f"Error requesting export: {e}")
            return error_response(f"Error requesting export: {str(e)}", 500)

@export_ns.route('/status/<string:export_id>')
class ExportStatus(Resource):
    def get(self, export_id):
        """Get export job status"""
        try:
            # Check if export job exists
            task_info = task_queue.get_task_info(export_id)
            if not task_info:
                return error_response(f"Export job {export_id} not found", 404)
            
            # Get additional status from training service
            try:
                response = requests.get(
                    f"{TRAINING_SERVICE_URL}/export/status/{export_id}",
                    timeout=5
                )
                if response.status_code == 200:
                    detailed_status = response.json().get('data', {})
                    task_info.update(detailed_status)
            except requests.RequestException as e:
                logger.warning(f"Error getting detailed status from training service: {e}")
            
            return success_response(task_info)
        except Exception as e:
            logger.exception(f"Error getting export status: {e}")
            return error_response(f"Error getting export status: {str(e)}", 500)

@export_ns.route('/download/<string:export_id>')
class DownloadExport(Resource):
    def get(self, export_id):
        """Download exported model"""
        try:
            # Check if export job exists
            task_info = task_queue.get_task_info(export_id)
            if not task_info:
                return error_response(f"Export job {export_id} not found", 404)
            
            # Check if export is complete
            if task_info.get('status') != 'completed':
                return error_response(f"Export job {export_id} is not complete", 400)
            
            # Check if export file exists
            export_file = task_info.get('export_file')
            if not export_file or not os.path.exists(export_file):
                return error_response(f"Export file not found", 404)
            
            # Send file
            return send_file(
                export_file,
                as_attachment=True,
                download_name=os.path.basename(export_file)
            )
        except Exception as e:
            logger.exception(f"Error downloading export: {e}")
            return error_response(f"Error downloading export: {str(e)}", 500)

@export_ns.route('/list')
class ListExports(Resource):
    def get(self):
        """List all export jobs"""
        try:
            # Get list of export jobs
            tasks = task_queue.list_tasks('export')
            
            return success_response(tasks)
        except Exception as e:
            logger.exception(f"Error listing export jobs: {e}")
            return error_response(f"Error listing export jobs: {str(e)}", 500)

# Dataset endpoints
@dataset_ns.route('/list')
class ListDatasets(Resource):
    def get(self):
        """List available datasets"""
        from app.config.settings import DATASETS
        
        try:
            return success_response(DATASETS)
        except Exception as e:
            logger.exception(f"Error listing datasets: {e}")
            return error_response(f"Error listing datasets: {str(e)}", 500)

# Monitoring endpoints
@monitoring_ns.route('/metrics/<string:training_id>')
class TrainingMetrics(Resource):
    def get(self, training_id):
        """Get training metrics"""
        try:
            # Check if training job exists
            if not task_queue.get_task_info(training_id):
                return error_response(f"Training job {training_id} not found", 404)
            
            # Forward request to monitoring service
            try:
                response = requests.get(
                    f"{MONITOR_SERVICE_URL}/metrics/{training_id}",
                    timeout=5
                )
                if response.status_code != 200:
                    logger.error(f"Failed to get training metrics: {response.text}")
                    return error_response(f"Failed to get training metrics: {response.text}", 500)
                
                return success_response(response.json().get('data', {}))
            except requests.RequestException as e:
                logger.error(f"Error communicating with monitoring service: {e}")
                return error_response(f"Error communicating with monitoring service: {str(e)}", 503)
        except Exception as e:
            logger.exception(f"Error getting training metrics: {e}")
            return error_response(f"Error getting training metrics: {str(e)}", 500)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
