"""Memory monitoring and management utilities"""

import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import weakref
import tracemalloc
import structlog

from nexus.core.exceptions import ResourceError

logger = structlog.get_logger()


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    gc_objects: int
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None


class MemoryMonitor:
    """Monitor memory usage and detect leaks"""
    
    def __init__(self, check_interval: float = 30.0, warning_threshold_mb: float = 1024, 
                 critical_threshold_mb: float = 2048, enable_tracemalloc: bool = True):
        self.check_interval = check_interval
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.enable_tracemalloc = enable_tracemalloc
        
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Tracemalloc setup
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Memory tracing started")
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self._monitoring:
            logger.warning("Memory monitoring already started")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Memory monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        logger.info("Memory monitoring stopped")
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add callback to be called on each memory check"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        
        # Get garbage collection stats
        gc_objects = len(gc.get_objects())
        
        # Get tracemalloc stats if enabled
        tracemalloc_current_mb = None
        tracemalloc_peak_mb = None
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current_mb = current / 1024 / 1024
            tracemalloc_peak_mb = peak / 1024 / 1024
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=virtual_memory.available / 1024 / 1024,
            gc_objects=gc_objects,
            tracemalloc_current_mb=tracemalloc_current_mb,
            tracemalloc_peak_mb=tracemalloc_peak_mb
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]
        
        return snapshot
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring and not self._stop_event.wait(self.check_interval):
            try:
                snapshot = self.take_snapshot()
                
                # Check thresholds
                if snapshot.rss_mb > self.critical_threshold_mb:
                    logger.critical(f"Critical memory usage: {snapshot.rss_mb:.1f}MB (threshold: {self.critical_threshold_mb}MB)")
                elif snapshot.rss_mb > self.warning_threshold_mb:
                    logger.warning(f"High memory usage: {snapshot.rss_mb:.1f}MB (threshold: {self.warning_threshold_mb}MB)")
                
                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Memory monitor callback failed: {e}")
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def get_memory_trend(self, minutes: float = 10.0) -> Dict[str, float]:
        """Get memory usage trend over specified time period"""
        if not self.snapshots:
            return {"trend": 0.0, "avg_mb": 0.0, "peak_mb": 0.0}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {"trend": 0.0, "avg_mb": 0.0, "peak_mb": 0.0}
        
        # Calculate trend (MB per minute)
        first = recent_snapshots[0]
        last = recent_snapshots[-1]
        time_diff = (last.timestamp - first.timestamp) / 60  # minutes
        memory_diff = last.rss_mb - first.rss_mb
        
        trend = memory_diff / time_diff if time_diff > 0 else 0.0
        avg_mb = sum(s.rss_mb for s in recent_snapshots) / len(recent_snapshots)
        peak_mb = max(s.rss_mb for s in recent_snapshots)
        
        return {
            "trend": trend,
            "avg_mb": avg_mb,
            "peak_mb": peak_mb,
            "snapshots": len(recent_snapshots)
        }
    
    def detect_memory_leak(self, threshold_mb_per_minute: float = 1.0, 
                          observation_minutes: float = 30.0) -> bool:
        """Detect potential memory leaks"""
        trend_info = self.get_memory_trend(observation_minutes)
        
        if trend_info["trend"] > threshold_mb_per_minute:
            logger.warning(
                f"Potential memory leak detected: {trend_info['trend']:.2f}MB/min "
                f"(threshold: {threshold_mb_per_minute}MB/min)"
            )
            return True
        
        return False
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return stats"""
        before_objects = len(gc.get_objects())
        
        # Force collection of all generations
        collected = [gc.collect(i) for i in range(3)]
        
        after_objects = len(gc.get_objects())
        freed_objects = before_objects - after_objects
        
        logger.info(f"Garbage collection freed {freed_objects} objects")
        
        return {
            "before_objects": before_objects,
            "after_objects": after_objects,
            "freed_objects": freed_objects,
            "collected": collected
        }
    
    def get_top_memory_objects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory consuming objects (requires tracemalloc)"""
        if not self.enable_tracemalloc or not tracemalloc.is_tracing():
            logger.warning("Tracemalloc not enabled, cannot get top memory objects")
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        results = []
        for stat in top_stats[:limit]:
            results.append({
                "filename": stat.traceback.format()[0].split(", line ")[0],
                "line": stat.traceback.format()[0].split(", line ")[1] if ", line " in stat.traceback.format()[0] else "unknown",
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        current_snapshot = self.take_snapshot()
        trend_info = self.get_memory_trend()
        
        stats = {
            "current": {
                "rss_mb": current_snapshot.rss_mb,
                "vms_mb": current_snapshot.vms_mb,
                "percent": current_snapshot.percent,
                "available_mb": current_snapshot.available_mb,
                "gc_objects": current_snapshot.gc_objects
            },
            "trend": trend_info,
            "monitoring": {
                "active": self._monitoring,
                "snapshots_taken": len(self.snapshots),
                "tracemalloc_enabled": self.enable_tracemalloc and tracemalloc.is_tracing()
            },
            "thresholds": {
                "warning_mb": self.warning_threshold_mb,
                "critical_mb": self.critical_threshold_mb
            }
        }
        
        if current_snapshot.tracemalloc_current_mb:
            stats["tracemalloc"] = {
                "current_mb": current_snapshot.tracemalloc_current_mb,
                "peak_mb": current_snapshot.tracemalloc_peak_mb
            }
        
        return stats


class ObjectTracker:
    """Track object creation and deletion for leak detection"""
    
    def __init__(self):
        self.tracked_objects: Dict[str, List[weakref.ref]] = {}
        self._lock = threading.Lock()
    
    def track(self, obj: Any, category: str = "default"):
        """Start tracking an object"""
        with self._lock:
            if category not in self.tracked_objects:
                self.tracked_objects[category] = []
            
            # Clean up dead references
            self.tracked_objects[category] = [ref for ref in self.tracked_objects[category] if ref() is not None]
            
            # Add new reference
            self.tracked_objects[category].append(weakref.ref(obj))
    
    def get_count(self, category: str = None) -> Dict[str, int]:
        """Get count of tracked objects"""
        with self._lock:
            if category:
                if category in self.tracked_objects:
                    # Clean up dead references
                    self.tracked_objects[category] = [ref for ref in self.tracked_objects[category] if ref() is not None]
                    return {category: len(self.tracked_objects[category])}
                else:
                    return {category: 0}
            
            # Return all categories
            counts = {}
            for cat, refs in self.tracked_objects.items():
                # Clean up dead references
                self.tracked_objects[cat] = [ref for ref in refs if ref() is not None]
                counts[cat] = len(self.tracked_objects[cat])
            
            return counts
    
    def clear(self, category: str = None):
        """Clear tracked objects"""
        with self._lock:
            if category:
                self.tracked_objects.pop(category, None)
            else:
                self.tracked_objects.clear()


# Global instances
_memory_monitor: Optional[MemoryMonitor] = None
_object_tracker: Optional[ObjectTracker] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


def get_object_tracker() -> ObjectTracker:
    """Get global object tracker instance"""
    global _object_tracker
    if _object_tracker is None:
        _object_tracker = ObjectTracker()
    return _object_tracker


def cleanup_memory(force_gc: bool = True) -> Dict[str, Any]:
    """Perform comprehensive memory cleanup"""
    logger.info("Starting memory cleanup")
    
    results = {}
    
    # Force garbage collection
    if force_gc:
        monitor = get_memory_monitor()
        results["garbage_collection"] = monitor.force_garbage_collection()
    
    # Clear weak references
    import weakref
    weakref.getweakrefcount
    
    # Get memory stats before and after
    before_snapshot = get_memory_monitor().take_snapshot()
    
    # Additional cleanup can be added here
    
    after_snapshot = get_memory_monitor().take_snapshot()
    
    results["memory_freed_mb"] = before_snapshot.rss_mb - after_snapshot.rss_mb
    results["before_mb"] = before_snapshot.rss_mb
    results["after_mb"] = after_snapshot.rss_mb
    
    logger.info(f"Memory cleanup completed, freed {results['memory_freed_mb']:.1f}MB")
    
    return results