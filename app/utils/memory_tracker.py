"""
Memory tracking utilities.
This module provides utilities for tracking and managing memory usage.
"""

import os
import time
import logging
import gc
import psutil
import threading
from typing import Dict, Any, Optional

from app.config.settings import MAX_MEMORY_MB, MEMORY_OPTIMIZATION

# Configure logging
logger = logging.getLogger(__name__)

class MemoryTracker:
    """
    Class for tracking and managing memory usage.
    """
    
    def __init__(self, memory_limit_mb: int = MAX_MEMORY_MB):
        """
        Initialize memory tracker.
        
        Args:
            memory_limit_mb: Memory limit in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_threshold = 0.9  # 90% of limit
        self.critical_threshold = 0.95  # 95% of limit
        self.poll_interval = MEMORY_OPTIMIZATION['garbage_collection_interval']
        self._stop_event = threading.Event()
        self._monitoring_thread = None
        
        # Start background memory monitoring
        self.start_monitoring()
        
        logger.info(f"Memory tracker initialized with limit: {memory_limit_mb} MB")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage information
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get virtual memory
        virtual_memory = psutil.virtual_memory()
        
        memory_usage_mb = memory_info.rss / (1024 * 1024)
        memory_percent = (memory_usage_mb / self.memory_limit_mb) * 100
        
        return {
            "used_mb": round(memory_usage_mb, 2),
            "limit_mb": self.memory_limit_mb,
            "percent": round(memory_percent, 2),
            "is_critical": memory_percent > (self.critical_threshold * 100),
            "system_total_mb": round(virtual_memory.total / (1024 * 1024), 2),
            "system_available_mb": round(virtual_memory.available / (1024 * 1024), 2),
            "system_percent": virtual_memory.percent
        }
    
    def has_sufficient_memory(self, required_mb: Optional[int] = None) -> bool:
        """
        Check if there is sufficient memory available.
        
        Args:
            required_mb: Required memory in MB (optional)
            
        Returns:
            True if sufficient memory is available, False otherwise
        """
        memory_info = self.get_memory_info()
        
        # If no specific requirement, check against threshold
        if required_mb is None:
            return memory_info["percent"] < (self.memory_threshold * 100)
        
        # Check if there's enough memory for the specific requirement
        available_mb = self.memory_limit_mb - memory_info["used_mb"]
        return available_mb >= required_mb
    
    def free_memory(self) -> Dict[str, Any]:
        """
        Free memory by garbage collection and other optimizations.
        
        Returns:
            Dictionary with memory usage information before and after
        """
        before = self.get_memory_info()
        
        # Force garbage collection
        gc.collect()
        
        # Try to release memory from caches
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass
        
        after = self.get_memory_info()
        freed_mb = before["used_mb"] - after["used_mb"]
        
        logger.info(f"Memory freed: {freed_mb:.2f} MB")
        
        return {
            "before": before,
            "after": after,
            "freed_mb": round(freed_mb, 2)
        }
    
    def _monitor_memory(self) -> None:
        """
        Background thread for monitoring memory usage.
        """
        logger.info("Starting memory monitoring thread")
        
        while not self._stop_event.is_set():
            try:
                memory_info = self.get_memory_info()
                
                # Log memory usage
                if memory_info["percent"] > (self.critical_threshold * 100):
                    logger.warning(f"Critical memory usage: {memory_info['percent']:.2f}% of limit")
                    self.free_memory()
                elif memory_info["percent"] > (self.memory_threshold * 100):
                    logger.info(f"High memory usage: {memory_info['percent']:.2f}% of limit")
                
                # Sleep for the specified interval
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.poll_interval)
    
    def start_monitoring(self) -> None:
        """
        Start background memory monitoring.
        """
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_memory)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """
        Stop background memory monitoring.
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5)
            self._monitoring_thread = None
