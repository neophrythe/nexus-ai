"""Real-time Performance Profiling System for Nexus Framework"""

import time
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import json
from pathlib import Path
import structlog
from collections import deque, defaultdict
import tracemalloc
import gc
import sys
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import asyncio
import torch
import tensorflow as tf

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of performance metrics"""
    FPS = "fps"
    LATENCY = "latency"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    GPU_MEMORY = "gpu_memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    POWER_USAGE = "power_usage"
    TEMPERATURE = "temperature"


class ProfileLevel(Enum):
    """Profiling detail levels"""
    MINIMAL = "minimal"      # Basic metrics only
    STANDARD = "standard"    # Standard profiling
    DETAILED = "detailed"    # Detailed profiling
    FULL = "full"           # Everything including traces


@dataclass
class PerformanceMetric:
    """Single performance metric"""
    timestamp: float
    metric_type: MetricType
    value: float
    unit: str
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileSnapshot:
    """Performance profile snapshot"""
    timestamp: float
    fps: float
    frame_time_ms: float
    cpu_percent: float
    memory_mb: float
    gpu_percent: Optional[float]
    gpu_memory_mb: Optional[float]
    active_threads: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComponentProfile:
    """Profile for a specific component"""
    component_name: str
    call_count: int
    total_time_ms: float
    average_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    memory_allocated_mb: float
    memory_peak_mb: float
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (calls per second)"""
        if self.total_time_ms > 0:
            return (self.call_count * 1000) / self.total_time_ms
        return 0.0


class PerformanceProfiler:
    """Real-time performance profiling system"""
    
    def __init__(self, profile_level: ProfileLevel = ProfileLevel.STANDARD,
                 sample_interval_ms: int = 100,
                 history_size: int = 1000):
        """
        Initialize performance profiler
        
        Args:
            profile_level: Level of profiling detail
            sample_interval_ms: Sampling interval in milliseconds
            history_size: Size of metric history buffer
        """
        self.profile_level = profile_level
        self.sample_interval_ms = sample_interval_ms
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.snapshots: deque = deque(maxlen=history_size)
        self.component_profiles: Dict[str, ComponentProfile] = {}
        
        # Timing data
        self.component_timings: defaultdict = defaultdict(list)
        self.frame_times: deque = deque(maxlen=100)
        self.last_frame_time = time.time()
        
        # Memory tracking
        self.memory_tracker_enabled = profile_level in [ProfileLevel.DETAILED, ProfileLevel.FULL]
        if self.memory_tracker_enabled:
            tracemalloc.start()
        
        # GPU tracking
        self.gpu_available = self._check_gpu_availability()
        self.gpu_handles = GPUtil.getGPUs() if self.gpu_available else []
        
        # Threading
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.metrics_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.stats = {
            "total_frames": 0,
            "total_time_s": 0.0,
            "average_fps": 0.0,
            "peak_memory_mb": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "total_components_profiled": 0,
            "profiling_overhead_ms": 0.0
        }
        
        # Alerts and thresholds
        self.alerts_enabled = True
        self.thresholds = {
            "max_frame_time_ms": 50,
            "max_memory_mb": 4096,
            "max_gpu_memory_mb": 8192,
            "min_fps": 20,
            "max_cpu_percent": 80,
            "max_gpu_percent": 90
        }
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        
        logger.info(f"Performance profiler initialized: level={profile_level.value}")
    
    def start(self):
        """Start performance monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        if self.memory_tracker_enabled:
            tracemalloc.stop()
        
        logger.info("Performance monitoring stopped")
    
    @contextmanager
    def profile_component(self, component_name: str):
        """
        Context manager for profiling a component
        
        Args:
            component_name: Name of component to profile
        """
        start_time = time.perf_counter()
        start_memory = 0
        
        if self.memory_tracker_enabled:
            start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Record timing
            self.component_timings[component_name].append(elapsed_ms)
            
            # Update component profile
            self._update_component_profile(component_name, elapsed_ms)
            
            # Memory tracking
            if self.memory_tracker_enabled:
                current_memory = tracemalloc.get_traced_memory()[0]
                memory_delta_mb = (current_memory - start_memory) / (1024 * 1024)
                
                if component_name in self.component_profiles:
                    self.component_profiles[component_name].memory_allocated_mb += memory_delta_mb
    
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator for profiling functions
        
        Args:
            func: Function to profile
        
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            with self.profile_component(func.__name__):
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        return wrapper
    
    async def profile_async_component(self, component_name: str):
        """Async version of profile_component"""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.component_timings[component_name].append(elapsed_ms)
            self._update_component_profile(component_name, elapsed_ms)
    
    def record_frame(self):
        """Record frame timing"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(frame_time)
        self.stats["total_frames"] += 1
        
        # Calculate FPS
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.stats["average_fps"] = fps
            
            # Check FPS threshold
            if self.alerts_enabled and fps < self.thresholds["min_fps"]:
                self._trigger_alert("low_fps", f"FPS dropped to {fps:.1f}")
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     component: str = "system", metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric
        
        Args:
            metric_type: Type of metric
            value: Metric value
            component: Component name
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            value=value,
            unit=self._get_metric_unit(metric_type),
            component=component,
            metadata=metadata or {}
        )
        
        try:
            self.metrics_queue.put_nowait(metric)
        except queue.Full:
            logger.warning("Metrics queue full, dropping metric")
    
    def get_current_snapshot(self) -> ProfileSnapshot:
        """Get current performance snapshot"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # GPU (if available)
        gpu_percent = None
        gpu_memory_mb = None
        
        if self.gpu_available and self.gpu_handles:
            try:
                gpu = self.gpu_handles[0]
                gpu_percent = gpu.load * 100
                gpu_memory_mb = gpu.memoryUsed
            except Exception as e:
                logger.debug(f"GPU metrics error: {e}")
        
        # FPS
        fps = self.stats["average_fps"]
        frame_time_ms = (1000.0 / fps) if fps > 0 else 0
        
        # Active threads
        active_threads = threading.active_count()
        
        snapshot = ProfileSnapshot(
            timestamp=time.time(),
            fps=fps,
            frame_time_ms=frame_time_ms,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_percent=gpu_percent,
            gpu_memory_mb=gpu_memory_mb,
            active_threads=active_threads
        )
        
        # Add to history
        self.snapshots.append(snapshot)
        
        # Check thresholds
        self._check_thresholds(snapshot)
        
        return snapshot
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        interval_s = self.sample_interval_ms / 1000.0
        
        while not self.stop_event.is_set():
            try:
                # Collect snapshot
                snapshot = self.get_current_snapshot()
                
                # Process queued metrics
                self._process_metrics_queue()
                
                # Update statistics
                self._update_statistics()
                
                # Sleep until next sample
                time.sleep(interval_s)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _process_metrics_queue(self):
        """Process metrics from queue"""
        processed = 0
        max_batch = 100
        
        while not self.metrics_queue.empty() and processed < max_batch:
            try:
                metric = self.metrics_queue.get_nowait()
                self.metrics_history.append(metric)
                processed += 1
            except queue.Empty:
                break
    
    def _update_component_profile(self, component_name: str, elapsed_ms: float):
        """Update component profile statistics"""
        if component_name not in self.component_profiles:
            self.component_profiles[component_name] = ComponentProfile(
                component_name=component_name,
                call_count=0,
                total_time_ms=0,
                average_time_ms=0,
                min_time_ms=float('inf'),
                max_time_ms=0,
                std_time_ms=0,
                memory_allocated_mb=0,
                memory_peak_mb=0
            )
        
        profile = self.component_profiles[component_name]
        profile.call_count += 1
        profile.total_time_ms += elapsed_ms
        profile.average_time_ms = profile.total_time_ms / profile.call_count
        profile.min_time_ms = min(profile.min_time_ms, elapsed_ms)
        profile.max_time_ms = max(profile.max_time_ms, elapsed_ms)
        
        # Calculate standard deviation
        if len(self.component_timings[component_name]) > 1:
            profile.std_time_ms = np.std(self.component_timings[component_name])
        
        # Update memory peak
        if self.memory_tracker_enabled:
            current, peak = tracemalloc.get_traced_memory()
            profile.memory_peak_mb = max(profile.memory_peak_mb, peak / (1024 * 1024))
    
    def _update_statistics(self):
        """Update global statistics"""
        self.stats["total_time_s"] = time.time() - (self.snapshots[0].timestamp if self.snapshots else time.time())
        
        if self.snapshots:
            # Memory peaks
            self.stats["peak_memory_mb"] = max(s.memory_mb for s in self.snapshots)
            
            if self.gpu_available:
                gpu_memories = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
                if gpu_memories:
                    self.stats["peak_gpu_memory_mb"] = max(gpu_memories)
        
        self.stats["total_components_profiled"] = len(self.component_profiles)
        
        # Calculate profiling overhead
        if self.component_profiles:
            overhead = sum(p.total_time_ms for p in self.component_profiles.values()) / len(self.component_profiles)
            self.stats["profiling_overhead_ms"] = overhead * 0.01  # Rough estimate
    
    def _check_thresholds(self, snapshot: ProfileSnapshot):
        """Check performance thresholds and trigger alerts"""
        if not self.alerts_enabled:
            return
        
        # Frame time
        if snapshot.frame_time_ms > self.thresholds["max_frame_time_ms"]:
            self._trigger_alert("high_frame_time", 
                              f"Frame time {snapshot.frame_time_ms:.1f}ms exceeds threshold")
        
        # Memory
        if snapshot.memory_mb > self.thresholds["max_memory_mb"]:
            self._trigger_alert("high_memory",
                              f"Memory usage {snapshot.memory_mb:.1f}MB exceeds threshold")
        
        # GPU memory
        if snapshot.gpu_memory_mb and snapshot.gpu_memory_mb > self.thresholds["max_gpu_memory_mb"]:
            self._trigger_alert("high_gpu_memory",
                              f"GPU memory {snapshot.gpu_memory_mb:.1f}MB exceeds threshold")
        
        # CPU
        if snapshot.cpu_percent > self.thresholds["max_cpu_percent"]:
            self._trigger_alert("high_cpu",
                              f"CPU usage {snapshot.cpu_percent:.1f}% exceeds threshold")
        
        # GPU
        if snapshot.gpu_percent and snapshot.gpu_percent > self.thresholds["max_gpu_percent"]:
            self._trigger_alert("high_gpu",
                              f"GPU usage {snapshot.gpu_percent:.1f}% exceeds threshold")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger performance alert"""
        logger.warning(f"Performance alert [{alert_type}]: {message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message, self.get_current_snapshot())
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except Exception:
            return False
    
    def _get_metric_unit(self, metric_type: MetricType) -> str:
        """Get unit for metric type"""
        units = {
            MetricType.FPS: "fps",
            MetricType.LATENCY: "ms",
            MetricType.CPU_USAGE: "%",
            MetricType.MEMORY_USAGE: "MB",
            MetricType.GPU_USAGE: "%",
            MetricType.GPU_MEMORY: "MB",
            MetricType.DISK_IO: "MB/s",
            MetricType.NETWORK_IO: "MB/s",
            MetricType.POWER_USAGE: "W",
            MetricType.TEMPERATURE: "°C"
        }
        return units.get(metric_type, "")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, threshold_name: str, value: float):
        """Set performance threshold"""
        if threshold_name in self.thresholds:
            self.thresholds[threshold_name] = value
            logger.info(f"Set threshold {threshold_name} = {value}")
    
    def get_component_report(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed component performance report"""
        report = {}
        
        for name, profile in self.component_profiles.items():
            report[name] = {
                "call_count": profile.call_count,
                "total_time_ms": profile.total_time_ms,
                "average_time_ms": profile.average_time_ms,
                "min_time_ms": profile.min_time_ms,
                "max_time_ms": profile.max_time_ms,
                "std_time_ms": profile.std_time_ms,
                "throughput": profile.throughput,
                "memory_allocated_mb": profile.memory_allocated_mb,
                "memory_peak_mb": profile.memory_peak_mb
            }
        
        return report
    
    def get_metrics_summary(self, time_window_s: Optional[float] = None) -> Dict[str, Any]:
        """
        Get metrics summary
        
        Args:
            time_window_s: Time window in seconds (None for all)
        
        Returns:
            Metrics summary dictionary
        """
        if not self.snapshots:
            return {}
        
        # Filter by time window
        if time_window_s:
            cutoff_time = time.time() - time_window_s
            filtered_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        else:
            filtered_snapshots = list(self.snapshots)
        
        if not filtered_snapshots:
            return {}
        
        # Calculate summary statistics
        summary = {
            "time_window_s": time_window_s or self.stats["total_time_s"],
            "num_samples": len(filtered_snapshots),
            "fps": {
                "mean": np.mean([s.fps for s in filtered_snapshots]),
                "min": np.min([s.fps for s in filtered_snapshots]),
                "max": np.max([s.fps for s in filtered_snapshots]),
                "std": np.std([s.fps for s in filtered_snapshots])
            },
            "frame_time_ms": {
                "mean": np.mean([s.frame_time_ms for s in filtered_snapshots]),
                "min": np.min([s.frame_time_ms for s in filtered_snapshots]),
                "max": np.max([s.frame_time_ms for s in filtered_snapshots]),
                "p95": np.percentile([s.frame_time_ms for s in filtered_snapshots], 95),
                "p99": np.percentile([s.frame_time_ms for s in filtered_snapshots], 99)
            },
            "cpu_percent": {
                "mean": np.mean([s.cpu_percent for s in filtered_snapshots]),
                "max": np.max([s.cpu_percent for s in filtered_snapshots])
            },
            "memory_mb": {
                "mean": np.mean([s.memory_mb for s in filtered_snapshots]),
                "max": np.max([s.memory_mb for s in filtered_snapshots])
            }
        }
        
        # Add GPU metrics if available
        gpu_snapshots = [s for s in filtered_snapshots if s.gpu_percent is not None]
        if gpu_snapshots:
            summary["gpu_percent"] = {
                "mean": np.mean([s.gpu_percent for s in gpu_snapshots]),
                "max": np.max([s.gpu_percent for s in gpu_snapshots])
            }
            summary["gpu_memory_mb"] = {
                "mean": np.mean([s.gpu_memory_mb for s in gpu_snapshots]),
                "max": np.max([s.gpu_memory_mb for s in gpu_snapshots])
            }
        
        return summary
    
    def export_metrics(self, output_path: str, format: str = "json"):
        """
        Export metrics to file
        
        Args:
            output_path: Output file path
            format: Export format ("json", "csv", "html")
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._export_json(output_path)
        elif format == "csv":
            self._export_csv(output_path)
        elif format == "html":
            self._export_html(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics exported to {output_path}")
    
    def _export_json(self, output_path: str):
        """Export metrics as JSON"""
        data = {
            "stats": self.stats,
            "thresholds": self.thresholds,
            "component_report": self.get_component_report(),
            "metrics_summary": self.get_metrics_summary(),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "fps": s.fps,
                    "frame_time_ms": s.frame_time_ms,
                    "cpu_percent": s.cpu_percent,
                    "memory_mb": s.memory_mb,
                    "gpu_percent": s.gpu_percent,
                    "gpu_memory_mb": s.gpu_memory_mb,
                    "active_threads": s.active_threads
                }
                for s in self.snapshots
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_csv(self, output_path: str):
        """Export metrics as CSV"""
        # Convert snapshots to DataFrame
        data = []
        for s in self.snapshots:
            data.append({
                "timestamp": s.timestamp,
                "fps": s.fps,
                "frame_time_ms": s.frame_time_ms,
                "cpu_percent": s.cpu_percent,
                "memory_mb": s.memory_mb,
                "gpu_percent": s.gpu_percent or 0,
                "gpu_memory_mb": s.gpu_memory_mb or 0,
                "active_threads": s.active_threads
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def _export_html(self, output_path: str):
        """Export metrics as HTML report with charts"""
        html_content = self._generate_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_html_report(self) -> str:
        """Generate HTML performance report"""
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # FPS over time
        times = [s.timestamp for s in self.snapshots]
        fps_values = [s.fps for s in self.snapshots]
        axes[0, 0].plot(times, fps_values)
        axes[0, 0].set_title("FPS Over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("FPS")
        
        # CPU usage
        cpu_values = [s.cpu_percent for s in self.snapshots]
        axes[0, 1].plot(times, cpu_values)
        axes[0, 1].set_title("CPU Usage")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("CPU %")
        
        # Memory usage
        memory_values = [s.memory_mb for s in self.snapshots]
        axes[1, 0].plot(times, memory_values)
        axes[1, 0].set_title("Memory Usage")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Memory (MB)")
        
        # Component performance
        if self.component_profiles:
            components = list(self.component_profiles.keys())[:10]  # Top 10
            avg_times = [self.component_profiles[c].average_time_ms for c in components]
            axes[1, 1].barh(components, avg_times)
            axes[1, 1].set_title("Component Average Time")
            axes[1, 1].set_xlabel("Time (ms)")
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        from io import BytesIO
        import base64
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nexus Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f0f0f0; }}
                .alert {{ color: red; font-weight: bold; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            <div class="metric">
                <h2>Summary Statistics</h2>
                <p>Total Frames: {self.stats['total_frames']}</p>
                <p>Average FPS: {self.stats['average_fps']:.2f}</p>
                <p>Peak Memory: {self.stats['peak_memory_mb']:.2f} MB</p>
                <p>Components Profiled: {self.stats['total_components_profiled']}</p>
            </div>
            
            <div class="metric">
                <h2>Performance Charts</h2>
                <img src="data:image/png;base64,{image_base64}" />
            </div>
            
            <div class="metric">
                <h2>Component Performance</h2>
                <table border="1">
                    <tr>
                        <th>Component</th>
                        <th>Calls</th>
                        <th>Avg Time (ms)</th>
                        <th>Throughput</th>
                    </tr>
        """
        
        for name, profile in self.component_profiles.items():
            html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{profile.call_count}</td>
                        <td>{profile.average_time_ms:.2f}</td>
                        <td>{profile.throughput:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </body>
        </html>
        """
        
        return html
    
    def reset(self):
        """Reset profiler statistics"""
        self.metrics_history.clear()
        self.snapshots.clear()
        self.component_profiles.clear()
        self.component_timings.clear()
        self.frame_times.clear()
        
        # Reset stats
        self.stats = {
            "total_frames": 0,
            "total_time_s": 0.0,
            "average_fps": 0.0,
            "peak_memory_mb": 0.0,
            "peak_gpu_memory_mb": 0.0,
            "total_components_profiled": 0,
            "profiling_overhead_ms": 0.0
        }
        
        logger.info("Profiler reset")
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Get detailed memory profile"""
        if not self.memory_tracker_enabled:
            return {"error": "Memory tracking not enabled"}
        
        current, peak = tracemalloc.get_traced_memory()
        
        # Get top memory consumers
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:10]
        
        top_consumers = []
        for stat in top_stats:
            top_consumers.append({
                "file": stat.traceback.format()[0],
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count
            })
        
        return {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "top_consumers": top_consumers
        }
    
    def benchmark_component(self, component: Callable, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark a component
        
        Args:
            component: Component to benchmark
            iterations: Number of iterations
        
        Returns:
            Benchmark results
        """
        times = []
        
        # Warmup
        for _ in range(min(10, iterations // 10)):
            component()
        
        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()
            component()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        return {
            "iterations": iterations,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "throughput": (iterations * 1000) / sum(times)
        }