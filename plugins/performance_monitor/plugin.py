"""
Performance Monitor Plugin for Nexus Game AI Framework

Monitors game and system performance metrics in real-time.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from collections import deque
import numpy as np
import structlog

from nexus.core.plugin_base import PluginBase
from nexus.core.game import Game

logger = structlog.get_logger()


class PerformanceMonitorPlugin(PluginBase):
    """
    Plugin that monitors performance metrics including:
    - FPS (Frames Per Second)
    - Frame time and consistency
    - CPU and GPU usage
    - Memory consumption
    - Network latency
    - Disk I/O
    - Temperature monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Performance Monitor"
        self.version = "1.0.0"
        self.description = "Real-time performance monitoring and optimization"
        
        # Metrics storage
        self.metrics_history = {
            'fps': deque(maxlen=300),
            'frame_time': deque(maxlen=300),
            'cpu_percent': deque(maxlen=300),
            'memory_percent': deque(maxlen=300),
            'gpu_percent': deque(maxlen=300),
            'network_latency': deque(maxlen=300),
            'disk_io': deque(maxlen=300)
        }
        
        # Performance thresholds
        self.thresholds = {
            'min_fps': 30,
            'max_frame_time': 33.33,  # ms (30 FPS)
            'max_cpu': 80,  # percent
            'max_memory': 80,  # percent
            'max_gpu': 90  # percent
        }
        
        # Monitoring state
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitor_interval = 0.1  # 100ms
        
        # Performance alerts
        self.alerts_enabled = True
        self.alert_callbacks: List[Callable] = []
        
        # Frame tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS every second
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # System info
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        
        # GPU monitoring (if available)
        self.gpu_available = self._check_gpu_monitoring()
        
    def on_load(self):
        """Called when plugin is loaded."""
        logger.info(f"Loading {self.name} v{self.version}")
        self.start_monitoring()
        
    def on_unload(self):
        """Called when plugin is unloaded."""
        self.stop_monitoring()
        logger.info(f"Unloaded {self.name}")
        
    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process each game frame."""
        # Track frame timing
        current_time = time.time()
        frame_time = (current_time - self.last_frame_time) * 1000  # Convert to ms
        self.last_frame_time = current_time
        
        # Update frame metrics
        self.metrics_history['frame_time'].append(frame_time)
        self.frame_count += 1
        
        # Update FPS calculation
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.metrics_history['fps'].append(self.current_fps)
            self.frame_count = 0
            self.last_fps_update = current_time
            
            # Check performance thresholds
            self._check_performance_alerts()
        
        # Add performance overlay if enabled
        if self.config.get('show_overlay', True):
            frame = self._add_performance_overlay(frame)
        
        return frame
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.metrics_history['cpu_percent'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics_history['memory_percent'].append(memory.percent)
                
                # GPU usage (if available)
                if self.gpu_available:
                    gpu_percent = self._get_gpu_usage()
                    if gpu_percent is not None:
                        self.metrics_history['gpu_percent'].append(gpu_percent)
                
                # Network latency (ping to common server)
                latency = self._measure_network_latency()
                if latency is not None:
                    self.metrics_history['network_latency'].append(latency)
                
                # Disk I/O
                disk_io = self._get_disk_io_rate()
                if disk_io is not None:
                    self.metrics_history['disk_io'].append(disk_io)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _check_performance_alerts(self):
        """Check for performance issues and trigger alerts."""
        if not self.alerts_enabled:
            return
        
        alerts = []
        
        # Check FPS
        if self.current_fps < self.thresholds['min_fps']:
            alerts.append({
                'type': 'low_fps',
                'message': f"Low FPS: {self.current_fps:.1f}",
                'severity': 'warning'
            })
        
        # Check frame time consistency
        if len(self.metrics_history['frame_time']) > 10:
            recent_frame_times = list(self.metrics_history['frame_time'])[-10:]
            std_dev = np.std(recent_frame_times)
            if std_dev > 10:  # High variance in frame times
                alerts.append({
                    'type': 'frame_stutter',
                    'message': f"Frame stutter detected (σ={std_dev:.1f}ms)",
                    'severity': 'warning'
                })
        
        # Check CPU usage
        if self.metrics_history['cpu_percent']:
            recent_cpu = list(self.metrics_history['cpu_percent'])[-10:]
            avg_cpu = np.mean(recent_cpu)
            if avg_cpu > self.thresholds['max_cpu']:
                alerts.append({
                    'type': 'high_cpu',
                    'message': f"High CPU usage: {avg_cpu:.1f}%",
                    'severity': 'warning'
                })
        
        # Check memory usage
        if self.metrics_history['memory_percent']:
            recent_memory = list(self.metrics_history['memory_percent'])[-10:]
            avg_memory = np.mean(recent_memory)
            if avg_memory > self.thresholds['max_memory']:
                alerts.append({
                    'type': 'high_memory',
                    'message': f"High memory usage: {avg_memory:.1f}%",
                    'severity': 'warning'
                })
        
        # Trigger alert callbacks
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger performance alert."""
        logger.warning(f"Performance alert: {alert['message']}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _add_performance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add performance metrics overlay to frame."""
        import cv2
        
        overlay = frame.copy()
        
        # Prepare metrics text
        metrics_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Frame: {self.metrics_history['frame_time'][-1]:.1f}ms" if self.metrics_history['frame_time'] else "Frame: --ms",
            f"CPU: {self.metrics_history['cpu_percent'][-1]:.1f}%" if self.metrics_history['cpu_percent'] else "CPU: --%",
            f"MEM: {self.metrics_history['memory_percent'][-1]:.1f}%" if self.metrics_history['memory_percent'] else "MEM: --%"
        ]
        
        if self.gpu_available and self.metrics_history['gpu_percent']:
            metrics_text.append(f"GPU: {self.metrics_history['gpu_percent'][-1]:.1f}%")
        
        # Draw background
        y_offset = 10
        for i, text in enumerate(metrics_text):
            y_pos = y_offset + (i * 25)
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            
            # Draw background rectangle
            cv2.rectangle(
                overlay,
                (5, y_pos - text_height - 2),
                (15 + text_width, y_pos + 5),
                (0, 0, 0),
                -1
            )
            
            # Determine color based on performance
            color = (0, 255, 0)  # Green by default
            if 'FPS' in text and self.current_fps < self.thresholds['min_fps']:
                color = (0, 0, 255)  # Red for low FPS
            elif 'CPU' in text and self.metrics_history['cpu_percent'] and \
                 self.metrics_history['cpu_percent'][-1] > self.thresholds['max_cpu']:
                color = (0, 165, 255)  # Orange for high CPU
            elif 'MEM' in text and self.metrics_history['memory_percent'] and \
                 self.metrics_history['memory_percent'][-1] > self.thresholds['max_memory']:
                color = (0, 165, 255)  # Orange for high memory
            
            # Draw text
            cv2.putText(
                overlay, text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 1,
                cv2.LINE_AA
            )
        
        # Blend overlay with original frame
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def _check_gpu_monitoring(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            # Try nvidia-ml-py for NVIDIA GPUs
            import pynvml
            pynvml.nvmlInit()
            self.gpu_type = 'nvidia'
            return True
        except:
            pass
        
        try:
            # Try GPUtil as fallback
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_type = 'gputil'
                return True
        except:
            pass
        
        return False
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        if not self.gpu_available:
            return None
        
        try:
            if self.gpu_type == 'nvidia':
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return util.gpu
            elif self.gpu_type == 'gputil':
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load * 100
        except:
            pass
        
        return None
    
    def _measure_network_latency(self) -> Optional[float]:
        """Measure network latency."""
        try:
            import socket
            import struct
            
            # Simple ping to Google DNS
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            
            start_time = time.time()
            result = sock.connect_ex(('8.8.8.8', 53))
            latency = (time.time() - start_time) * 1000  # ms
            
            sock.close()
            
            if result == 0:
                return latency
        except:
            pass
        
        return None
    
    def _get_disk_io_rate(self) -> Optional[float]:
        """Get disk I/O rate in MB/s."""
        try:
            io_counters = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                time_delta = time.time() - self._last_disk_io_time
                bytes_delta = (
                    io_counters.read_bytes + io_counters.write_bytes - 
                    self._last_disk_io
                )
                rate = bytes_delta / time_delta / (1024 * 1024)  # MB/s
                
                self._last_disk_io = io_counters.read_bytes + io_counters.write_bytes
                self._last_disk_io_time = time.time()
                
                return rate
            else:
                self._last_disk_io = io_counters.read_bytes + io_counters.write_bytes
                self._last_disk_io_time = time.time()
        except:
            pass
        
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                values_list = list(values)
                summary[metric_name] = {
                    'current': values_list[-1],
                    'avg': np.mean(values_list),
                    'min': np.min(values_list),
                    'max': np.max(values_list),
                    'std': np.std(values_list)
                }
        
        return summary
    
    def optimize_settings(self) -> Dict[str, Any]:
        """Suggest optimized settings based on performance."""
        suggestions = []
        
        # Analyze recent performance
        if self.current_fps < self.thresholds['min_fps']:
            suggestions.append({
                'setting': 'resolution',
                'action': 'reduce',
                'reason': 'Low FPS detected'
            })
        
        if self.metrics_history['cpu_percent']:
            avg_cpu = np.mean(list(self.metrics_history['cpu_percent'])[-30:])
            if avg_cpu > self.thresholds['max_cpu']:
                suggestions.append({
                    'setting': 'processing_threads',
                    'action': 'reduce',
                    'reason': 'High CPU usage'
                })
        
        if self.metrics_history['memory_percent']:
            avg_memory = np.mean(list(self.metrics_history['memory_percent'])[-30:])
            if avg_memory > self.thresholds['max_memory']:
                suggestions.append({
                    'setting': 'buffer_size',
                    'action': 'reduce',
                    'reason': 'High memory usage'
                })
        
        return {
            'suggestions': suggestions,
            'current_performance': self.get_metrics_summary()
        }
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to file."""
        import json
        
        data = {
            'timestamp': time.time(),
            'metrics': self.get_metrics_summary(),
            'thresholds': self.thresholds,
            'system_info': {
                'cpu_count': self.cpu_count,
                'total_memory': self.total_memory,
                'gpu_available': self.gpu_available
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric: str, value: float):
        """Set performance threshold."""
        if metric in self.thresholds:
            self.thresholds[metric] = value
            logger.info(f"Set {metric} threshold to {value}")


# Plugin registration
def create_plugin() -> PerformanceMonitorPlugin:
    """Create plugin instance."""
    return PerformanceMonitorPlugin()