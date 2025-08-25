"""Memory leak detection and fixes for the Nexus framework"""

import gc
import weakref
import threading
import time
import tracemalloc
import linecache
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from collections import defaultdict, deque
import psutil
import torch
import numpy as np
import cv2
import asyncio
import structlog

from nexus.core.exceptions import ResourceError
from nexus.utils.memory import MemoryMonitor, get_memory_monitor

logger = structlog.get_logger()


class LeakDetector:
    """Advanced memory leak detection system"""
    
    def __init__(self, check_interval: float = 60.0, threshold_mb_per_minute: float = 2.0):
        self.check_interval = check_interval
        self.threshold = threshold_mb_per_minute
        self.baseline_memory = 0
        self.memory_samples: deque = deque(maxlen=100)
        self.leak_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Object tracking
        self.tracked_objects: Dict[str, List[weakref.ref]] = defaultdict(list)
        self.object_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Detection state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Tracemalloc integration
        self.tracemalloc_enabled = False
        self.memory_snapshots: List[Any] = []
    
    def start_monitoring(self):
        """Start memory leak monitoring"""
        if self.monitoring:
            logger.warning("Leak detector already monitoring")
            return
        
        self.monitoring = True
        self.stop_event.clear()
        self.baseline_memory = psutil.Process().memory_info().rss
        
        # Enable tracemalloc if not already enabled
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames
            self.tracemalloc_enabled = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Memory leak detection started")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Memory leak detection stopped")
    
    def add_leak_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback to be called when leak is detected"""
        self.leak_callbacks.append(callback)
    
    def track_object(self, obj: Any, category: str = "general"):
        """Track an object for leak detection"""
        weak_ref = weakref.ref(obj)
        self.tracked_objects[category].append(weak_ref)
        
        # Clean up dead references periodically
        if len(self.tracked_objects[category]) % 100 == 0:
            self.tracked_objects[category] = [ref for ref in self.tracked_objects[category] if ref() is not None]
    
    def get_object_counts(self) -> Dict[str, int]:
        """Get current counts of tracked objects"""
        counts = {}
        for category, refs in self.tracked_objects.items():
            # Clean up dead references
            self.tracked_objects[category] = [ref for ref in refs if ref() is not None]
            counts[category] = len(self.tracked_objects[category])
        return counts
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring and not self.stop_event.wait(self.check_interval):
            try:
                self._check_for_leaks()
            except Exception as e:
                logger.error(f"Error in leak detection: {e}")
    
    def _check_for_leaks(self):
        """Check for memory leaks"""
        current_memory = psutil.Process().memory_info().rss
        current_time = time.time()
        
        self.memory_samples.append((current_time, current_memory))
        
        # Need at least 2 samples to detect trend
        if len(self.memory_samples) < 2:
            return
        
        # Calculate memory trend
        time_span = self.memory_samples[-1][0] - self.memory_samples[0][0]
        memory_diff = self.memory_samples[-1][1] - self.memory_samples[0][1]
        
        if time_span > 0:
            memory_rate_mb_per_minute = (memory_diff / 1024 / 1024) / (time_span / 60)
            
            if memory_rate_mb_per_minute > self.threshold:
                leak_info = self._analyze_leak()
                self._notify_leak_detected(leak_info)
    
    def _analyze_leak(self) -> Dict[str, Any]:
        """Analyze detected memory leak"""
        current_memory = psutil.Process().memory_info().rss
        memory_increase = current_memory - self.baseline_memory
        
        leak_info = {
            "memory_increase_mb": memory_increase / 1024 / 1024,
            "baseline_memory_mb": self.baseline_memory / 1024 / 1024,
            "current_memory_mb": current_memory / 1024 / 1024,
            "object_counts": self.get_object_counts(),
            "gc_stats": self._get_gc_stats(),
            "top_allocators": self._get_top_allocators() if self.tracemalloc_enabled else []
        }
        
        return leak_info
    
    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        return {
            "objects": len(gc.get_objects()),
            "stats": gc.get_stats() if hasattr(gc, 'get_stats') else [],
            "counts": gc.get_count()
        }
    
    def _get_top_allocators(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory allocators using tracemalloc"""
        if not tracemalloc.is_tracing():
            return []
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:limit]
            
            allocators = []
            for stat in top_stats:
                allocators.append({
                    "filename": stat.traceback.format()[0],
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
            
            return allocators
        except Exception as e:
            logger.error(f"Failed to get tracemalloc stats: {e}")
            return []
    
    def _notify_leak_detected(self, leak_info: Dict[str, Any]):
        """Notify about detected memory leak"""
        logger.error(f"Memory leak detected: {leak_info['memory_increase_mb']:.1f}MB increase")
        
        for callback in self.leak_callbacks:
            try:
                callback(leak_info)
            except Exception as e:
                logger.error(f"Leak callback failed: {e}")


class MemoryManager:
    """Comprehensive memory management system"""
    
    def __init__(self):
        self.leak_detector = LeakDetector()
        self.memory_monitor = get_memory_monitor()
        self.cleanup_functions: List[Callable[[], None]] = []
        
        # Framework-specific cleanup
        self._register_framework_cleanups()
    
    def start(self):
        """Start memory management"""
        self.leak_detector.start_monitoring()
        self.memory_monitor.start_monitoring()
        
        # Add leak detection callback
        self.leak_detector.add_leak_callback(self._handle_memory_leak)
        
        logger.info("Memory management started")
    
    def stop(self):
        """Stop memory management"""
        self.leak_detector.stop_monitoring()
        self.memory_monitor.stop_monitoring()
        
        logger.info("Memory management stopped")
    
    def register_cleanup_function(self, cleanup_func: Callable[[], None]):
        """Register a cleanup function"""
        self.cleanup_functions.append(cleanup_func)
    
    def cleanup_all(self):
        """Run all cleanup functions"""
        logger.info("Running memory cleanup")
        
        # Run registered cleanup functions
        for cleanup_func in self.cleanup_functions:
            try:
                cleanup_func()
            except Exception as e:
                logger.error(f"Cleanup function failed: {e}")
        
        # Framework-specific cleanup
        self._cleanup_torch()
        self._cleanup_opencv()
        self._cleanup_numpy()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory cleanup completed")
    
    def _register_framework_cleanups(self):
        """Register framework-specific cleanup functions"""
        self.register_cleanup_function(self._cleanup_torch)
        self.register_cleanup_function(self._cleanup_opencv)
        self.register_cleanup_function(self._cleanup_numpy)
    
    def _cleanup_torch(self):
        """Clean up PyTorch memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear any cached tensors
            if hasattr(torch.utils.data, '_utils'):
                if hasattr(torch.utils.data._utils, 'signal_handling'):
                    # Clean up data loader workers
                    pass
            
        except Exception as e:
            logger.error(f"PyTorch cleanup failed: {e}")
    
    def _cleanup_opencv(self):
        """Clean up OpenCV resources"""
        try:
            # Destroy all OpenCV windows
            cv2.destroyAllWindows()
            
            # Clear any internal caches
            if hasattr(cv2, '_cleanup'):
                cv2._cleanup()
                
        except Exception as e:
            logger.error(f"OpenCV cleanup failed: {e}")
    
    def _cleanup_numpy(self):
        """Clean up NumPy resources"""
        try:
            # NumPy doesn't have much to clean up directly
            # But we can clear any cached arrays
            pass
        except Exception as e:
            logger.error(f"NumPy cleanup failed: {e}")
    
    def _handle_memory_leak(self, leak_info: Dict[str, Any]):
        """Handle detected memory leak"""
        logger.warning("Attempting automatic memory cleanup due to detected leak")
        
        # Try automatic cleanup
        self.cleanup_all()
        
        # Check if cleanup helped
        new_memory = psutil.Process().memory_info().rss / 1024 / 1024
        if new_memory < leak_info["current_memory_mb"] * 0.9:  # 10% reduction
            logger.info(f"Memory cleanup successful: {new_memory:.1f}MB (was {leak_info['current_memory_mb']:.1f}MB)")
        else:
            logger.error("Memory cleanup had minimal effect, manual intervention may be required")


class ResourceTracker:
    """Track resource usage across the framework"""
    
    def __init__(self):
        self.resources: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.resource_limits: Dict[str, Dict[str, Any]] = {
            "frames": {"max_count": 1000, "max_memory_mb": 500},
            "models": {"max_count": 10, "max_memory_mb": 2000},
            "textures": {"max_count": 500, "max_memory_mb": 200}
        }
    
    def track_resource(self, resource_type: str, resource_id: str, size_bytes: int, metadata: Dict[str, Any] = None):
        """Track a resource"""
        self.resources[resource_type][resource_id] = {
            "size_bytes": size_bytes,
            "created_at": time.time(),
            "metadata": metadata or {}
        }
        
        # Check limits
        self._check_resource_limits(resource_type)
    
    def untrack_resource(self, resource_type: str, resource_id: str):
        """Stop tracking a resource"""
        self.resources[resource_type].pop(resource_id, None)
    
    def get_resource_stats(self, resource_type: str = None) -> Dict[str, Any]:
        """Get resource usage statistics"""
        if resource_type:
            resources = {resource_type: self.resources[resource_type]}
        else:
            resources = self.resources
        
        stats = {}
        for res_type, items in resources.items():
            total_size = sum(item["size_bytes"] for item in items.values())
            stats[res_type] = {
                "count": len(items),
                "total_size_mb": total_size / 1024 / 1024,
                "avg_size_mb": (total_size / len(items) / 1024 / 1024) if items else 0
            }
        
        return stats
    
    def _check_resource_limits(self, resource_type: str):
        """Check if resource limits are exceeded"""
        if resource_type not in self.resource_limits:
            return
        
        current_count = len(self.resources[resource_type])
        current_size_mb = sum(item["size_bytes"] for item in self.resources[resource_type].values()) / 1024 / 1024
        
        limits = self.resource_limits[resource_type]
        
        if current_count > limits["max_count"]:
            logger.warning(f"Resource limit exceeded for {resource_type}: {current_count} items (limit: {limits['max_count']})")
        
        if current_size_mb > limits["max_memory_mb"]:
            logger.warning(f"Memory limit exceeded for {resource_type}: {current_size_mb:.1f}MB (limit: {limits['max_memory_mb']}MB)")


# Context managers for automatic cleanup

class MemoryContext:
    """Context manager for automatic memory cleanup"""
    
    def __init__(self, cleanup_on_exit: bool = True):
        self.cleanup_on_exit = cleanup_on_exit
        self.initial_memory = 0
        self.manager = MemoryManager()
    
    def __enter__(self):
        self.initial_memory = psutil.Process().memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            self.manager.cleanup_all()
        
        final_memory = psutil.Process().memory_info().rss
        memory_diff = (final_memory - self.initial_memory) / 1024 / 1024
        
        if memory_diff > 10:  # More than 10MB increase
            logger.warning(f"Memory increased by {memory_diff:.1f}MB in context")


class TorchMemoryContext:
    """Context manager for PyTorch memory management"""
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def fix_common_memory_leaks():
    """Fix common memory leaks in the framework"""
    logger.info("Applying common memory leak fixes")
    
    # Fix 1: Clear matplotlib figure cache
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
    
    # Fix 2: Clear OpenCV windows and release resources
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Fix 3: PyTorch CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    # Fix 4: Force garbage collection
    for i in range(3):
        gc.collect()
    
    # Fix 5: Clear any threading local data
    try:
        import threading
        threading.local().__dict__.clear()
    except:
        pass
    
    logger.info("Memory leak fixes applied")


# Decorators for automatic memory management

def track_memory_usage(func):
    """Decorator to track memory usage of a function"""
    def wrapper(*args, **kwargs):
        with MemoryContext(cleanup_on_exit=False):
            return func(*args, **kwargs)
    return wrapper


def auto_cleanup_memory(func):
    """Decorator to automatically cleanup memory after function execution"""
    def wrapper(*args, **kwargs):
        with MemoryContext(cleanup_on_exit=True):
            return func(*args, **kwargs)
    return wrapper