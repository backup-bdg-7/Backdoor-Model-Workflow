"""
Task queue for managing background tasks.
This module provides a task queue for managing background tasks like training and export jobs.
"""

import os
import time
import json
import uuid
import logging
import threading
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from app.config.settings import STORAGE_DIR

# Configure logging
logger = logging.getLogger(__name__)

class TaskQueue:
    """
    Simple task queue for managing background tasks.
    """
    
    def __init__(self, storage_dir: str = STORAGE_DIR):
        """
        Initialize task queue.
        
        Args:
            storage_dir: Directory for storing task information
        """
        self.storage_dir = storage_dir
        self.tasks_dir = os.path.join(storage_dir, 'tasks')
        self.lock = threading.Lock()
        
        # Create tasks directory if it doesn't exist
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        # Load existing tasks
        self.tasks = self._load_tasks()
        
        logger.info(f"Task queue initialized with {len(self.tasks)} tasks")
    
    def _load_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Load tasks from disk.
        
        Returns:
            Dictionary of tasks
        """
        tasks = {}
        
        for filename in os.listdir(self.tasks_dir):
            if filename.endswith('.json'):
                try:
                    task_id = filename.rsplit('.', 1)[0]
                    with open(os.path.join(self.tasks_dir, filename), 'r') as f:
                        task_info = json.load(f)
                    
                    tasks[task_id] = task_info
                except Exception as e:
                    logger.error(f"Error loading task {filename}: {e}")
        
        return tasks
    
    def _save_task(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """
        Save task information to disk.
        
        Args:
            task_id: Task ID
            task_info: Task information
        """
        try:
            with open(os.path.join(self.tasks_dir, f"{task_id}.json"), 'w') as f:
                json.dump(task_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving task {task_id}: {e}")
    
    def submit_task(self, task_type: str, config: Dict[str, Any]) -> str:
        """
        Submit a new task.
        
        Args:
            task_type: Task type (e.g., 'training', 'export')
            config: Task configuration
            
        Returns:
            Task ID
        """
        with self.lock:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Create task information
            task_info = {
                'id': task_id,
                'type': task_type,
                'config': config,
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'progress': 0.0,
                'message': 'Task created'
            }
            
            # Save task
            self.tasks[task_id] = task_info
            self._save_task(task_id, task_info)
            
            logger.info(f"Task {task_id} ({task_type}) submitted")
            
            return task_id
    
    def update_task_status(self, task_id: str, status: str, 
                          message: Optional[str] = None, 
                          progress: Optional[float] = None,
                          result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update task status.
        
        Args:
            task_id: Task ID
            status: New status
            message: Optional status message
            progress: Optional progress (0-100)
            result: Optional task result
            
        Returns:
            Updated task information
        """
        with self.lock:
            # Check if task exists
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                return {}
            
            # Update task information
            task_info = self.tasks[task_id]
            task_info['status'] = status
            task_info['updated_at'] = datetime.now().isoformat()
            
            if message is not None:
                task_info['message'] = message
            
            if progress is not None:
                task_info['progress'] = max(0.0, min(100.0, float(progress)))
            
            if result is not None:
                task_info['result'] = result
            
            # Save task
            self._save_task(task_id, task_info)
            
            logger.info(f"Task {task_id} updated: status={status}, progress={task_info.get('progress')}")
            
            return task_info
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """
        Get task information.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information or empty dict if not found
        """
        with self.lock:
            return self.tasks.get(task_id, {})
    
    def list_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List tasks, optionally filtered by type.
        
        Args:
            task_type: Optional task type to filter by
            
        Returns:
            List of task information
        """
        with self.lock:
            tasks_list = list(self.tasks.values())
            
            if task_type:
                tasks_list = [task for task in tasks_list if task.get('type') == task_type]
            
            return sorted(tasks_list, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was deleted, False otherwise
        """
        with self.lock:
            # Check if task exists
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                return False
            
            # Remove task from memory
            del self.tasks[task_id]
            
            # Remove task file
            try:
                os.remove(os.path.join(self.tasks_dir, f"{task_id}.json"))
            except Exception as e:
                logger.error(f"Error removing task file for {task_id}: {e}")
                return False
            
            logger.info(f"Task {task_id} deleted")
            
            return True
    
    def clean_old_tasks(self, max_age_days: int = 7) -> int:
        """
        Clean up old completed tasks.
        
        Args:
            max_age_days: Maximum age of tasks to keep (in days)
            
        Returns:
            Number of tasks deleted
        """
        with self.lock:
            now = datetime.now()
            max_age_seconds = max_age_days * 24 * 60 * 60
            deleted_count = 0
            
            for task_id, task_info in list(self.tasks.items()):
                # Skip tasks that are not completed
                if task_info.get('status') not in ['completed', 'failed', 'cancelled']:
                    continue
                
                # Check task age
                try:
                    updated_at = datetime.fromisoformat(task_info.get('updated_at', ''))
                    age_seconds = (now - updated_at).total_seconds()
                    
                    if age_seconds > max_age_seconds:
                        # Delete task
                        if self.delete_task(task_id):
                            deleted_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning task {task_id}: {e}")
            
            logger.info(f"Cleaned {deleted_count} old tasks")
            
            return deleted_count
    
    def get_queue_info(self) -> Dict[str, Any]:
        """
        Get information about the task queue.
        
        Returns:
            Dictionary with queue information
        """
        with self.lock:
            tasks_by_type = {}
            tasks_by_status = {}
            
            for task in self.tasks.values():
                task_type = task.get('type')
                task_status = task.get('status')
                
                if task_type:
                    tasks_by_type[task_type] = tasks_by_type.get(task_type, 0) + 1
                
                if task_status:
                    tasks_by_status[task_status] = tasks_by_status.get(task_status, 0) + 1
            
            return {
                'total': len(self.tasks),
                'by_type': tasks_by_type,
                'by_status': tasks_by_status
            }
