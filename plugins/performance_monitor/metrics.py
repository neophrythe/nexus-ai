"""
Metrics collection and analysis utilities.
"""

from typing import Dict, List, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class MetricSnapshot:
    """Single metric measurement."""
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary of a metric."""
    current: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    count: int
    
    @classmethod
    def from_values(cls, values: List[float]) -> 'MetricSummary':
        """Create summary from list of values."""
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        arr = np.array(values)
        return cls(
            current=values[-1],
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            percentile_25=float(np.percentile(arr, 25)),
            percentile_75=float(np.percentile(arr, 75)),
            percentile_95=float(np.percentile(arr, 95)),
            count=len(values)
        )


class MetricCollector:
    """Collects and analyzes metrics."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metric collector.
        
        Args:
            max_history: Maximum history size per metric
        """
        self.max_history = max_history
        self.metrics: Dict[str, Deque[MetricSnapshot]] = {}
        self.summaries: Dict[str, MetricSummary] = {}
        self.last_update: Dict[str, float] = {}
    
    def record(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.max_history)
        
        snapshot = MetricSnapshot(
            value=value,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.metrics[name].append(snapshot)
        self.last_update[name] = time.time()
        
        # Update summary
        self._update_summary(name)
    
    def get_values(self, name: str, time_window: Optional[float] = None) -> List[float]:
        """Get metric values within time window."""
        if name not in self.metrics:
            return []
        
        snapshots = list(self.metrics[name])
        
        if time_window:
            cutoff = time.time() - time_window
            snapshots = [s for s in snapshots if s.timestamp >= cutoff]
        
        return [s.value for s in snapshots]
    
    def get_summary(self, name: str) -> Optional[MetricSummary]:
        """Get metric summary."""
        return self.summaries.get(name)
    
    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get all metric summaries."""
        return self.summaries.copy()
    
    def _update_summary(self, name: str):
        """Update metric summary."""
        values = self.get_values(name)
        if values:
            self.summaries[name] = MetricSummary.from_values(values)
    
    def detect_anomalies(self, name: str, threshold: float = 3.0) -> List[float]:
        """Detect anomalies using z-score method."""
        values = self.get_values(name)
        if len(values) < 10:
            return []
        
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return []
        
        z_scores = np.abs((arr - mean) / std)
        anomalies = arr[z_scores > threshold].tolist()
        
        return anomalies
    
    def get_trend(self, name: str, window: int = 10) -> str:
        """Get metric trend (increasing, decreasing, stable)."""
        values = self.get_values(name)
        
        if len(values) < window:
            return 'unknown'
        
        recent = values[-window:]
        
        # Calculate linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Check if values are constant
        if np.std(y) < 0.01:
            return 'stable'
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope by mean value
        mean_val = np.mean(y)
        if mean_val != 0:
            normalized_slope = slope / mean_val
        else:
            normalized_slope = slope
        
        # Determine trend
        if normalized_slope > 0.01:
            return 'increasing'
        elif normalized_slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def calculate_rate(self, name: str) -> Optional[float]:
        """Calculate rate of change per second."""
        snapshots = list(self.metrics.get(name, []))
        
        if len(snapshots) < 2:
            return None
        
        # Get last two snapshots
        prev = snapshots[-2]
        curr = snapshots[-1]
        
        # Calculate rate
        time_diff = curr.timestamp - prev.timestamp
        if time_diff == 0:
            return None
        
        value_diff = curr.value - prev.value
        rate = value_diff / time_diff
        
        return rate
    
    def clear(self, name: Optional[str] = None):
        """Clear metrics."""
        if name:
            if name in self.metrics:
                del self.metrics[name]
            if name in self.summaries:
                del self.summaries[name]
            if name in self.last_update:
                del self.last_update[name]
        else:
            self.metrics.clear()
            self.summaries.clear()
            self.last_update.clear()


class PerformanceAnalyzer:
    """Analyzes performance metrics and provides insights."""
    
    def __init__(self, collector: MetricCollector):
        """
        Initialize performance analyzer.
        
        Args:
            collector: Metric collector instance
        """
        self.collector = collector
    
    def analyze_fps_stability(self) -> Dict[str, Any]:
        """Analyze FPS stability."""
        fps_values = self.collector.get_values('fps')
        
        if len(fps_values) < 10:
            return {'status': 'insufficient_data'}
        
        summary = self.collector.get_summary('fps')
        
        # Calculate stability metrics
        stability_score = 1.0 - min(summary.std / max(summary.mean, 1), 1.0)
        consistency = 1.0 - ((summary.max - summary.min) / max(summary.mean, 1))
        
        # Detect stutters (sudden drops)
        stutters = self._detect_stutters(fps_values)
        
        return {
            'status': 'analyzed',
            'stability_score': stability_score,
            'consistency': consistency,
            'stutter_count': len(stutters),
            'average_fps': summary.mean,
            'min_fps': summary.min,
            'percentile_1': summary.percentile_25,
            'recommendation': self._get_fps_recommendation(summary, stability_score)
        }
    
    def analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        cpu_summary = self.collector.get_summary('cpu_percent')
        mem_summary = self.collector.get_summary('memory_percent')
        gpu_summary = self.collector.get_summary('gpu_percent')
        
        analysis = {
            'cpu': self._analyze_resource(cpu_summary, 'CPU'),
            'memory': self._analyze_resource(mem_summary, 'Memory'),
            'gpu': self._analyze_resource(gpu_summary, 'GPU') if gpu_summary else None
        }
        
        # Identify bottleneck
        bottleneck = self._identify_bottleneck(cpu_summary, mem_summary, gpu_summary)
        analysis['bottleneck'] = bottleneck
        
        return analysis
    
    def _analyze_resource(self, summary: Optional[MetricSummary], name: str) -> Dict[str, Any]:
        """Analyze individual resource."""
        if not summary:
            return {'status': 'no_data'}
        
        return {
            'average': summary.mean,
            'peak': summary.max,
            'current': summary.current,
            'utilization': self._categorize_utilization(summary.mean),
            'trend': self.collector.get_trend(f"{name.lower()}_percent"),
            'headroom': max(0, 100 - summary.percentile_95)
        }
    
    def _detect_stutters(self, fps_values: List[float], threshold: float = 0.5) -> List[int]:
        """Detect frame stutters."""
        if len(fps_values) < 2:
            return []
        
        stutters = []
        avg_fps = np.mean(fps_values)
        
        for i in range(1, len(fps_values)):
            # Check for sudden drop
            if fps_values[i] < fps_values[i-1] * threshold:
                # And recovery
                if i < len(fps_values) - 1 and fps_values[i+1] > fps_values[i] * 1.5:
                    stutters.append(i)
        
        return stutters
    
    def _categorize_utilization(self, usage: float) -> str:
        """Categorize resource utilization level."""
        if usage < 30:
            return 'low'
        elif usage < 60:
            return 'moderate'
        elif usage < 80:
            return 'high'
        else:
            return 'critical'
    
    def _identify_bottleneck(self, cpu: Optional[MetricSummary], 
                            mem: Optional[MetricSummary],
                            gpu: Optional[MetricSummary]) -> str:
        """Identify performance bottleneck."""
        bottlenecks = []
        
        if cpu and cpu.percentile_95 > 90:
            bottlenecks.append('cpu')
        
        if mem and mem.percentile_95 > 85:
            bottlenecks.append('memory')
        
        if gpu and gpu.percentile_95 > 95:
            bottlenecks.append('gpu')
        
        if not bottlenecks:
            return 'none'
        elif len(bottlenecks) == 1:
            return bottlenecks[0]
        else:
            return 'multiple'
    
    def _get_fps_recommendation(self, summary: MetricSummary, stability: float) -> str:
        """Get FPS optimization recommendation."""
        if summary.min < 30:
            return "Reduce graphics settings or resolution"
        elif stability < 0.7:
            return "Enable V-Sync or frame rate limiting for stability"
        elif summary.percentile_95 > 120:
            return "Consider enabling frame rate cap to reduce power consumption"
        else:
            return "Performance is optimal"