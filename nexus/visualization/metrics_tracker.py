"""
Metrics Tracker for Real-time Training Monitoring

Collects, aggregates, and tracks metrics during training with
automatic TensorBoard integration.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricWindow:
    """Sliding window for metric aggregation."""
    size: int
    values: deque = field(init=False)
    
    def __post_init__(self):
        self.values = deque(maxlen=self.size)
    
    def add(self, value: float):
        """Add value to window."""
        self.values.append(value)
    
    def mean(self) -> float:
        """Get window mean."""
        return np.mean(self.values) if self.values else 0.0
    
    def std(self) -> float:
        """Get window standard deviation."""
        return np.std(self.values) if len(self.values) > 1 else 0.0
    
    def min(self) -> float:
        """Get window minimum."""
        return np.min(self.values) if self.values else 0.0
    
    def max(self) -> float:
        """Get window maximum."""
        return np.max(self.values) if self.values else 0.0


class MetricsTracker:
    """
    Advanced metrics tracking with aggregation and analysis.
    
    Features:
    - Real-time metric collection
    - Sliding window aggregation
    - Automatic rate calculation
    - Anomaly detection
    - Performance benchmarking
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 tensorboard_logger = None,
                 log_interval: int = 10,
                 enable_profiling: bool = True):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of sliding window for aggregation
            tensorboard_logger: TensorBoard logger instance
            log_interval: Steps between TensorBoard logs
            enable_profiling: Enable performance profiling
        """
        self.window_size = window_size
        self.tb_logger = tensorboard_logger
        self.log_interval = log_interval
        self.enable_profiling = enable_profiling
        
        # Metric storage
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.windows: Dict[str, MetricWindow] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Performance tracking
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        
        # Rate calculation
        self.rate_trackers: Dict[str, Tuple[float, int]] = {}  # (last_time, last_count)
        
        # Global step
        self.global_step = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start time for uptime tracking
        self.start_time = time.time()
        
        logger.info(f"Metrics tracker initialized with window size {window_size}")
    
    def track(self, name: str, value: float, step: Optional[int] = None, 
             metadata: Dict[str, Any] = None):
        """
        Track a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (uses global step if None)
            metadata: Additional metadata
        """
        with self.lock:
            step = step or self.global_step
            
            # Create metric
            metric = Metric(name, value, step, metadata=metadata or {})
            self.metrics[name].append(metric)
            
            # Update sliding window
            if name not in self.windows:
                self.windows[name] = MetricWindow(self.window_size)
            self.windows[name].add(value)
            
            # Update counter
            self.counters[name] += 1
            
            # Log to TensorBoard if interval reached
            if self.tb_logger and self.counters[name] % self.log_interval == 0:
                self.tb_logger.log_scalar(name, value, step)
                
                # Log aggregated metrics
                self.tb_logger.log_scalar(f"{name}/mean", self.windows[name].mean(), step)
                self.tb_logger.log_scalar(f"{name}/std", self.windows[name].std(), step)
    
    def track_rate(self, name: str, count: int = 1, step: Optional[int] = None):
        """
        Track a rate metric (e.g., FPS, steps/sec).
        
        Args:
            name: Metric name
            count: Count to add
            step: Step number
        """
        current_time = time.time()
        
        with self.lock:
            if name in self.rate_trackers:
                last_time, last_count = self.rate_trackers[name]
                time_diff = current_time - last_time
                
                if time_diff > 0:
                    rate = count / time_diff
                    self.track(f"{name}/rate", rate, step)
            
            self.rate_trackers[name] = (current_time, count)
    
    def start_timer(self, name: str):
        """Start a performance timer."""
        if self.enable_profiling:
            self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """
        End a performance timer and track the duration.
        
        Returns:
            Duration in seconds
        """
        if not self.enable_profiling or name not in self.start_times:
            return 0.0
        
        duration = time.perf_counter() - self.start_times[name]
        
        with self.lock:
            self.timers[name].append(duration)
            
            # Track as metric
            self.track(f"timing/{name}", duration * 1000)  # Convert to ms
        
        del self.start_times[name]
        return duration
    
    def track_episode(self, episode_num: int, reward: float, length: int, 
                     win: bool = False, additional: Dict[str, float] = None):
        """
        Track episode metrics.
        
        Args:
            episode_num: Episode number
            reward: Total episode reward
            length: Episode length in steps
            win: Whether episode was won
            additional: Additional metrics
        """
        self.track("episode/reward", reward, episode_num)
        self.track("episode/length", length, episode_num)
        self.track("episode/win_rate", float(win), episode_num)
        
        # Track reward per step
        if length > 0:
            self.track("episode/reward_per_step", reward / length, episode_num)
        
        # Track additional metrics
        if additional:
            for key, value in additional.items():
                self.track(f"episode/{key}", value, episode_num)
    
    def track_training(self, loss: float, learning_rate: float = None,
                      gradient_norm: float = None, step: Optional[int] = None):
        """
        Track training metrics.
        
        Args:
            loss: Training loss
            learning_rate: Current learning rate
            gradient_norm: Gradient norm
            step: Training step
        """
        self.track("training/loss", loss, step)
        
        if learning_rate is not None:
            self.track("training/learning_rate", learning_rate, step)
        
        if gradient_norm is not None:
            self.track("training/gradient_norm", gradient_norm, step)
    
    def track_model_metrics(self, model_name: str, metrics: Dict[str, float], 
                          step: Optional[int] = None):
        """Track model-specific metrics."""
        for key, value in metrics.items():
            self.track(f"model/{model_name}/{key}", value, step)
    
    def track_resource_usage(self, step: Optional[int] = None):
        """Track system resource usage."""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.track("resources/cpu_percent", cpu_percent, step)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.track("resources/memory_percent", memory.percent, step)
        self.track("resources/memory_gb", memory.used / (1024**3), step)
        
        # GPU usage if available
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.track(f"resources/gpu{i}_percent", util.gpu, step)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.track(f"resources/gpu{i}_memory_gb", 
                          mem_info.used / (1024**3), step)
        except:
            pass  # GPU monitoring not available
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.windows:
                return {}
            
            window = self.windows[name]
            return {
                'mean': window.mean(),
                'std': window.std(),
                'min': window.min(),
                'max': window.max(),
                'count': self.counters[name],
                'last': window.values[-1] if window.values else 0.0
            }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all tracked metrics."""
        summaries = {}
        
        with self.lock:
            for name in self.windows:
                summaries[name] = self.get_metric_summary(name)
        
        return summaries
    
    def detect_anomalies(self, name: str, threshold: float = 3.0) -> List[Metric]:
        """
        Detect anomalies in a metric using z-score.
        
        Args:
            name: Metric name
            threshold: Z-score threshold for anomaly
            
        Returns:
            List of anomalous metrics
        """
        with self.lock:
            if name not in self.windows or name not in self.metrics:
                return []
            
            window = self.windows[name]
            mean = window.mean()
            std = window.std()
            
            if std == 0:
                return []
            
            anomalies = []
            for metric in self.metrics[name][-self.window_size:]:
                z_score = abs((metric.value - mean) / std)
                if z_score > threshold:
                    anomalies.append(metric)
            
            return anomalies
    
    def create_report(self) -> str:
        """Create a text report of all metrics."""
        lines = ["=" * 60]
        lines.append("METRICS REPORT")
        lines.append("=" * 60)
        
        # Uptime
        uptime = time.time() - self.start_time
        lines.append(f"Uptime: {uptime:.1f} seconds")
        lines.append(f"Global Step: {self.global_step}")
        lines.append("")
        
        # Metrics summaries
        summaries = self.get_all_summaries()
        
        for category in ['episode', 'training', 'model', 'resources', 'timing']:
            category_metrics = {k: v for k, v in summaries.items() 
                              if k.startswith(category)}
            
            if category_metrics:
                lines.append(f"\n{category.upper()} METRICS:")
                lines.append("-" * 40)
                
                for name, summary in category_metrics.items():
                    lines.append(f"\n{name}:")
                    lines.append(f"  Mean: {summary.get('mean', 0):.4f}")
                    lines.append(f"  Std:  {summary.get('std', 0):.4f}")
                    lines.append(f"  Min:  {summary.get('min', 0):.4f}")
                    lines.append(f"  Max:  {summary.get('max', 0):.4f}")
                    lines.append(f"  Last: {summary.get('last', 0):.4f}")
        
        # Performance timings
        if self.timers:
            lines.append("\nPERFORMANCE TIMINGS:")
            lines.append("-" * 40)
            
            for name, durations in self.timers.items():
                if durations:
                    mean_ms = np.mean(durations) * 1000
                    lines.append(f"{name}: {mean_ms:.2f} ms (avg)")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def step(self):
        """Increment global step."""
        self.global_step += 1
        
        # Track FPS
        self.track_rate("global/fps", 1, self.global_step)
        
        # Periodically track resource usage
        if self.global_step % 100 == 0:
            self.track_resource_usage(self.global_step)
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.windows.clear()
            self.counters.clear()
            self.timers.clear()
            self.rate_trackers.clear()
            self.global_step = 0
            logger.info("Metrics tracker reset")
    
    def save_metrics(self, filepath: str):
        """Save metrics to file."""
        import pickle
        
        with self.lock:
            data = {
                'metrics': dict(self.metrics),
                'counters': dict(self.counters),
                'global_step': self.global_step,
                'summaries': self.get_all_summaries()
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load metrics from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        with self.lock:
            self.metrics = defaultdict(list, data['metrics'])
            self.counters = defaultdict(int, data['counters'])
            self.global_step = data['global_step']
        
        logger.info(f"Metrics loaded from {filepath}")