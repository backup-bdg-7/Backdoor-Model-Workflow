"""
Training monitoring service.
This module provides a service for monitoring training progress and metrics.
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Set the path to include the main app directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config.settings import (
    STORAGE_DIR, MONITOR_UPDATE_INTERVAL, MONITOR_RETENTION_PERIOD
)

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
            logging.FileHandler(os.path.join(STORAGE_DIR, 'logs', 'monitor.log'))
        ]
    )
    logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Training Monitor",
    description="Monitoring service for training jobs",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class MetricsData(BaseModel):
    job_id: str
    status: str
    message: str
    progress: float
    metrics: Dict[str, Any]
    timestamp: float

class MetricsResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    memory: Dict[str, Any]
    connections: int
    uptime_seconds: int

# Initialize metrics storage
metrics_dir = os.path.join(STORAGE_DIR, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)

# Store active websocket connections
active_connections: Dict[str, List[WebSocket]] = {}
metrics_cache: Dict[str, Dict[str, Any]] = {}
start_time = time.time()

# Detect if running on Render free tier
is_render_free_tier = os.environ.get('RENDER_SERVICE_TYPE', '') != ''
if is_render_free_tier:
    # On Render.com free tier, reduce retention period and use memory-centric approach
    logger.info("Running on Render.com free tier - using memory-centric metrics storage")
    MONITOR_RETENTION_PERIOD = min(MONITOR_RETENTION_PERIOD, 1)  # Maximum 1 day retention on free tier
    logger.info(f"Metrics retention period: {MONITOR_RETENTION_PERIOD} day(s)")

# Background cleanup task
def cleanup_old_metrics():
    """
    Clean up old metrics files.
    """
    try:
        # Get cutoff time
        cutoff_time = datetime.now() - timedelta(days=MONITOR_RETENTION_PERIOD)
        
        # Iterate through metric files
        for filename in os.listdir(metrics_dir):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(metrics_dir, filename)
            
            try:
                # Check file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if mod_time < cutoff_time:
                    # Remove old file
                    os.remove(file_path)
                    logger.info(f"Removed old metrics file: {filename}")
            except Exception as e:
                logger.error(f"Error checking file {filename}: {e}")
        
        # Also clean up memory cache
        for job_id in list(metrics_cache.keys()):
            last_update = metrics_cache[job_id].get('last_update', 0)
            if datetime.fromtimestamp(last_update) < cutoff_time:
                del metrics_cache[job_id]
                logger.info(f"Removed old metrics from cache: {job_id}")
                
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}")

# Background task to periodically clean up old metrics
@app.on_event("startup")
async def startup_event():
    # Run initial cleanup
    cleanup_old_metrics()
    
    # Create background task for periodic cleanup
    def periodic_cleanup():
        while True:
            time.sleep(86400)  # 24 hours
            cleanup_old_metrics()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

# Health check endpoint
@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Check service health"""
    # Get memory usage
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    
    # Count active connections
    connection_count = sum(len(connections) for connections in active_connections.values())
    
    # Calculate uptime
    uptime = int(time.time() - start_time)
    
    return {
        "status": "ok",
        "memory": {
            "used_mb": round(memory_mb, 2),
            "percent": round(process.memory_percent(), 2)
        },
        "connections": connection_count,
        "uptime_seconds": uptime
    }

# Post metrics endpoint
@app.post("/metrics/{job_id}", response_model=MetricsResponse)
async def post_metrics(job_id: str, metrics_data: MetricsData, background_tasks: BackgroundTasks):
    """
    Post metrics for a training job.
    """
    try:
        # Create metrics file path
        metrics_file = os.path.join(metrics_dir, f"{job_id}.json")
        
        # Load existing metrics if available
        metrics_history = []
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics_history = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading metrics file {metrics_file}, creating new file")
        
        # Append new metrics
        metrics_history.append(metrics_data.dict())
        
        # Limit history size to prevent files from getting too large
        max_history = 1000
        if len(metrics_history) > max_history:
            metrics_history = metrics_history[-max_history:]
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # Update cache
        metrics_cache[job_id] = {
            'last_update': time.time(),
            'status': metrics_data.status,
            'message': metrics_data.message,
            'progress': metrics_data.progress,
            'metrics': metrics_data.metrics,
            'timestamp': metrics_data.timestamp
        }
        
        # Broadcast to websocket clients
        if job_id in active_connections:
            for connection in active_connections[job_id]:
                try:
                    await connection.send_json({
                        'type': 'metrics_update',
                        'job_id': job_id,
                        'data': metrics_data.dict()
                    })
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")
        
        # Schedule cleanup in background
        background_tasks.add_task(cleanup_old_metrics)
        
        return {
            "success": True,
            "message": "Metrics saved successfully"
        }
    except Exception as e:
        logger.exception(f"Error saving metrics: {e}")
        return {
            "success": False,
            "message": f"Error saving metrics: {str(e)}"
        }

# Get metrics endpoint
@app.get("/metrics/{job_id}", response_model=MetricsResponse)
async def get_metrics(job_id: str, limit: int = 100):
    """
    Get metrics for a training job.
    """
    try:
        # Check if metrics exist in cache
        if job_id in metrics_cache:
            latest_metrics = metrics_cache[job_id]
        else:
            # Create metrics file path
            metrics_file = os.path.join(metrics_dir, f"{job_id}.json")
            
            # Check if metrics file exists
            if not os.path.exists(metrics_file):
                return {
                    "success": False,
                    "message": f"No metrics found for job {job_id}"
                }
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)
            
            # Check if there are any metrics
            if not metrics_history:
                return {
                    "success": False,
                    "message": f"No metrics found for job {job_id}"
                }
            
            # Get latest metrics
            latest_metrics = metrics_history[-1]
            
            # Update cache
            metrics_cache[job_id] = {
                'last_update': time.time(),
                'status': latest_metrics['status'],
                'message': latest_metrics['message'],
                'progress': latest_metrics['progress'],
                'metrics': latest_metrics['metrics'],
                'timestamp': latest_metrics['timestamp']
            }
        
        # Get metrics history (limited)
        metrics_file = os.path.join(metrics_dir, f"{job_id}.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)
            
            # Limit history and extract specific data points
            history_limit = min(limit, len(metrics_history))
            history = metrics_history[-history_limit:]
            
            # Extract time series for common metrics
            time_series = {
                'timestamps': [],
                'loss': [],
                'learning_rate': [],
                'progress': []
            }
            
            for entry in history:
                time_series['timestamps'].append(entry['timestamp'])
                time_series['progress'].append(entry['progress'])
                
                metrics = entry.get('metrics', {})
                if metrics:
                    time_series['loss'].append(metrics.get('loss', None))
                    time_series['learning_rate'].append(metrics.get('learning_rate', None))
            
            # Remove None values
            for key in time_series:
                if key != 'timestamps' and key != 'progress':
                    time_series[key] = [v for v in time_series[key] if v is not None]
        else:
            history = []
            time_series = {}
        
        return {
            "success": True,
            "message": "Metrics retrieved successfully",
            "data": {
                "latest": latest_metrics,
                "history": history if len(history) <= 10 else [],  # Only send history if small
                "time_series": time_series
            }
        }
    except Exception as e:
        logger.exception(f"Error retrieving metrics: {e}")
        return {
            "success": False,
            "message": f"Error retrieving metrics: {str(e)}"
        }

# Delete metrics endpoint
@app.delete("/metrics/{job_id}", response_model=MetricsResponse)
async def delete_metrics(job_id: str):
    """
    Delete metrics for a training job.
    """
    try:
        # Create metrics file path
        metrics_file = os.path.join(metrics_dir, f"{job_id}.json")
        
        # Check if metrics file exists
        if not os.path.exists(metrics_file):
            return {
                "success": False,
                "message": f"No metrics found for job {job_id}"
            }
        
        # Delete metrics file
        os.remove(metrics_file)
        
        # Remove from cache
        if job_id in metrics_cache:
            del metrics_cache[job_id]
        
        return {
            "success": True,
            "message": f"Metrics for job {job_id} deleted successfully"
        }
    except Exception as e:
        logger.exception(f"Error deleting metrics: {e}")
        return {
            "success": False,
            "message": f"Error deleting metrics: {str(e)}"
        }

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    
    # Register connection
    if job_id not in active_connections:
        active_connections[job_id] = []
    active_connections[job_id].append(websocket)
    
    try:
        # Send initial metrics if available
        metrics_file = os.path.join(metrics_dir, f"{job_id}.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)
            
            # Send the latest metrics
            if metrics_history:
                await websocket.send_json({
                    'type': 'initial_metrics',
                    'job_id': job_id,
                    'data': metrics_history[-1]
                })
        
        # Keep connection open until client disconnects
        while True:
            data = await websocket.receive_text()
            # Ping/pong to keep connection alive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        # Remove connection
        if job_id in active_connections:
            active_connections[job_id].remove(websocket)
            if not active_connections[job_id]:
                del active_connections[job_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Remove connection on error
        if job_id in active_connections and websocket in active_connections[job_id]:
            active_connections[job_id].remove(websocket)
            if not active_connections[job_id]:
                del active_connections[job_id]

# List active jobs with metrics
@app.get("/jobs", response_model=MetricsResponse)
async def list_jobs():
    """
    List all jobs with metrics.
    """
    try:
        jobs = []
        
        # Get all metrics files
        for filename in os.listdir(metrics_dir):
            if not filename.endswith('.json'):
                continue
            
            job_id = filename[:-5]  # Remove .json extension
            
            try:
                # Load metrics
                with open(os.path.join(metrics_dir, filename), 'r') as f:
                    metrics_history = json.load(f)
                
                # Get latest metrics
                if metrics_history:
                    latest = metrics_history[-1]
                    jobs.append({
                        'job_id': job_id,
                        'status': latest['status'],
                        'message': latest['message'],
                        'progress': latest['progress'],
                        'last_update': latest['timestamp'],
                        'has_metrics': bool(latest.get('metrics'))
                    })
            except Exception as e:
                logger.error(f"Error loading metrics for {job_id}: {e}")
        
        # Sort by last update time, most recent first
        jobs.sort(key=lambda x: x['last_update'], reverse=True)
        
        return {
            "success": True,
            "message": f"Found {len(jobs)} jobs with metrics",
            "data": {
                "jobs": jobs
            }
        }
    except Exception as e:
        logger.exception(f"Error listing jobs: {e}")
        return {
            "success": False,
            "message": f"Error listing jobs: {str(e)}"
        }

# Generate training charts
@app.get("/charts/{job_id}")
async def get_charts(job_id: str):
    """
    Generate training charts for a job.
    """
    try:
        # Import visualization libraries
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        import base64
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        # Get metrics
        metrics_file = os.path.join(metrics_dir, f"{job_id}.json")
        if not os.path.exists(metrics_file):
            raise HTTPException(status_code=404, detail=f"No metrics found for job {job_id}")
        
        with open(metrics_file, 'r') as f:
            metrics_history = json.load(f)
        
        if not metrics_history:
            raise HTTPException(status_code=404, detail=f"No metrics found for job {job_id}")
        
        # Extract time series
        timestamps = []
        loss_values = []
        lr_values = []
        
        for entry in metrics_history:
            timestamps.append(entry['timestamp'])
            
            metrics = entry.get('metrics', {})
            if metrics:
                if 'loss' in metrics:
                    loss_values.append(metrics['loss'])
                if 'learning_rate' in metrics:
                    lr_values.append(metrics['learning_rate'])
        
        # Convert timestamps to hours elapsed
        if timestamps:
            start_time = timestamps[0]
            timestamps = [(t - start_time) / 3600 for t in timestamps]  # Hours
        
        # Create charts
        charts = {}
        
        # Loss chart
        if loss_values:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(timestamps[:len(loss_values)], loss_values)
            ax.set_title('Training Loss')
            ax.set_xlabel('Hours Elapsed')
            ax.set_ylabel('Loss')
            ax.grid(True)
            
            # Save to bytes
            buf = io.BytesIO()
            FigureCanvas(fig).print_png(buf)
            loss_chart = base64.b64encode(buf.getbuffer()).decode('ascii')
            charts['loss'] = f"data:image/png;base64,{loss_chart}"
        
        # Learning rate chart
        if lr_values:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(timestamps[:len(lr_values)], lr_values)
            ax.set_title('Learning Rate')
            ax.set_xlabel('Hours Elapsed')
            ax.set_ylabel('Learning Rate')
            ax.grid(True)
            
            # Save to bytes
            buf = io.BytesIO()
            FigureCanvas(fig).print_png(buf)
            lr_chart = base64.b64encode(buf.getbuffer()).decode('ascii')
            charts['learning_rate'] = f"data:image/png;base64,{lr_chart}"
        
        return {
            "success": True,
            "message": "Charts generated successfully",
            "data": {
                "charts": charts
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error generating charts: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating charts: {str(e)}")

# Serve static files for simple dashboard
app.mount("/dashboard", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../frontend/build"), html=True), name="dashboard")

if __name__ == "__main__":
    # Start the server
    port = int(os.environ.get('PORT', 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
