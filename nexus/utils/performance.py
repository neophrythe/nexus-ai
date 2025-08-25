"""Performance monitoring and profiling utilities"""

import time
import cProfile
import pstats
import io
import threading
import functools
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    name: str
    duration: float
    iterations: int
    ops_per_second: float
    min_time: float
    max_time: float
    avg_time: float
    std_deviation: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Comprehensive performance profiler"""
    
    def __init__(self, max_metrics: int = 1000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
        
        # System monitoring
        self.process = psutil.Process()
        self._system_metrics = {}
        
    def record_metric(self, name: str, value: float, unit: str = "ms", 
                     category: str = "general", **metadata):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            category=category,
            metadata=metadata
        )
        
        with self._lock:
            self.metrics.append(metric)
    
    def start_timer(self, name: str):
        """Start a named timer"""
        with self._lock:
            self.timers[name] = time.time()
    
    def end_timer(self, name: str, record_metric: bool = True) -> float:
        """End a named timer and optionally record as metric"""
        with self._lock:
            if name not in self.timers:
                logger.warning(f"Timer '{name}' not found")
                return 0.0
            
            duration = (time.time() - self.timers[name]) * 1000  # Convert to ms
            del self.timers[name]
            
            if record_metric:
                self.record_metric(f"timer_{name}", duration, "ms", "timing")
            
            return duration
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter"""
        with self._lock:
            self.counters[name] += amount
    
    def record_rate(self, name: str, count: int = 1):
        """Record a rate (events per second)"""
        with self._lock:
            self.rates[name].append((time.time(), count))
    
    def get_rate(self, name: str, window_seconds: float = 60.0) -> float:
        """Get current rate for a metric"""
        with self._lock:
            if name not in self.rates:
                return 0.0
            
            now = time.time()
            cutoff = now - window_seconds
            
            # Filter recent events
            recent_events = [(ts, count) for ts, count in self.rates[name] if ts >= cutoff]
            
            if not recent_events:
                return 0.0
            
            total_count = sum(count for _, count in recent_events)
            time_span = now - recent_events[0][0]
            
            return total_count / time_span if time_span > 0 else 0.0
    
    def capture_system_metrics(self):
        """Capture system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            cpu_times = self.process.cpu_times()
            
            # Memory metrics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # I/O metrics
            try:
                io_counters = self.process.io_counters()
                read_bytes = io_counters.read_bytes
                write_bytes = io_counters.write_bytes
            except (AttributeError, psutil.AccessDenied):
                read_bytes = write_bytes = 0
            
            # Thread/process metrics
            num_threads = self.process.num_threads()
            
            # Record metrics
            self.record_metric("cpu_percent", cpu_percent, "%", "system")
            self.record_metric("memory_rss", memory_info.rss / 1024 / 1024, "MB", "system")
            self.record_metric("memory_percent", memory_percent, "%", "system")
            self.record_metric("threads", num_threads, "count", "system")
            self.record_metric("io_read", read_bytes / 1024 / 1024, "MB", "system")
            self.record_metric("io_write", write_bytes / 1024 / 1024, "MB", "system")
            
            # Store for comparison
            self._system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": memory_percent,
                "threads": num_threads,
                "io_read_mb": read_bytes / 1024 / 1024,
                "io_write_mb": write_bytes / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Failed to capture system metrics: {e}")
    
    def get_metrics_summary(self, category: str = None, 
                          time_window_seconds: float = None) -> Dict[str, Any]:
        """Get summary of recorded metrics"""
        with self._lock:
            metrics = list(self.metrics)
        
        # Filter by category
        if category:
            metrics = [m for m in metrics if m.category == category]
        
        # Filter by time window
        if time_window_seconds:
            cutoff = time.time() - time_window_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff]
        
        if not metrics:
            return {"count": 0}
        
        # Group by name
        by_name = defaultdict(list)
        for metric in metrics:
            by_name[metric.name].append(metric.value)
        
        # Calculate statistics
        summary = {
            "count": len(metrics),
            "time_span_seconds": max(m.timestamp for m in metrics) - min(m.timestamp for m in metrics),
            "metrics": {}
        }
        
        for name, values in by_name.items():
            summary["metrics"][name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1]
            }
        
        return summary
    
    def get_performance_report(self) -> str:
        """Generate a formatted performance report"""
        system_summary = self.get_metrics_summary("system")
        timing_summary = self.get_metrics_summary("timing")
        
        report_lines = [
            "=== PERFORMANCE REPORT ===",
            "",
            "System Metrics:",
        ]
        
        if system_summary.get("metrics"):
            for name, stats in system_summary["metrics"].items():
                report_lines.append(f"  {name}: {stats['latest']:.2f} (avg: {stats['avg']:.2f})")
        
        report_lines.extend([
            "",
            "Timing Metrics:",
        ])
        
        if timing_summary.get("metrics"):
            for name, stats in timing_summary["metrics"].items():
                report_lines.append(f"  {name}: {stats['latest']:.2f}ms (avg: {stats['avg']:.2f}ms)")
        
        report_lines.extend([
            "",
            "Counters:",
        ])
        
        for name, count in self.counters.items():
            report_lines.append(f"  {name}: {count}")
        
        report_lines.extend([
            "",
            "Rates (per second):",
        ])
        
        for name in self.rates:
            rate = self.get_rate(name)
            report_lines.append(f"  {name}: {rate:.2f}")
        
        return "\n".join(report_lines)
    
    def clear(self):
        """Clear all metrics and counters"""
        with self._lock:
            self.metrics.clear()
            self.timers.clear()
            self.counters.clear()
            self.rates.clear()


class CodeProfiler:
    """Code profiler using cProfile"""
    
    def __init__(self):
        self.profiler: Optional[cProfile.Profile] = None
        self.profiling_active = False
    
    def start(self):
        """Start profiling"""
        if self.profiling_active:
            logger.warning("Profiling already active")
            return
        
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profiling_active = True
        logger.info("Code profiling started")
    
    def stop(self) -> str:
        """Stop profiling and return results"""
        if not self.profiling_active or not self.profiler:
            logger.warning("Profiling not active")
            return ""
        
        self.profiler.disable()
        self.profiling_active = False
        
        # Generate report
        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        result = s.getvalue()
        logger.info("Code profiling stopped")
        return result
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a specific function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            local_profiler = cProfile.Profile()
            local_profiler.enable()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                local_profiler.disable()
                
                # Generate report
                s = io.StringIO()
                stats = pstats.Stats(local_profiler, stream=s)
                stats.sort_stats('cumulative')
                stats.print_stats(10)
                
                logger.debug(f"Profile for {func.__name__}:\\n{s.getvalue()}")
        
        return wrapper


def benchmark(func: Callable, iterations: int = 1000, warmup: int = 10, 
             name: str = None) -> BenchmarkResult:
    """Benchmark a function"""
    func_name = name or func.__name__
    
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    times = []
    start_total = time.time()
    
    for _ in range(iterations):
        start = time.time()
        func()
        end = time.time()
        times.append(end - start)
    
    end_total = time.time()
    total_duration = end_total - start_total
    
    # Calculate statistics
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    
    # Standard deviation
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_deviation = variance ** 0.5
    
    ops_per_second = iterations / total_duration
    
    result = BenchmarkResult(
        name=func_name,
        duration=total_duration,
        iterations=iterations,
        ops_per_second=ops_per_second,
        min_time=min_time,
        max_time=max_time,
        avg_time=avg_time,
        std_deviation=std_deviation,
        metadata={"warmup_iterations": warmup}
    )
    
    logger.info(f"Benchmark '{func_name}': {ops_per_second:.0f} ops/sec, avg {avg_time*1000:.2f}ms")
    
    return result


def time_function(func: Callable = None, *, name: str = None, log_slow: float = None):
    """Decorator to time function execution"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_name = name or f.__name__
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                
                if log_slow and duration > log_slow:
                    logger.warning(f"Slow function '{func_name}': {duration:.3f}s")
                else:
                    logger.debug(f"Function '{func_name}': {duration:.3f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Function '{func_name}' failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


class PerformanceContext:
    """Context manager for performance monitoring"""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str, 
                 capture_system: bool = False):
        self.profiler = profiler
        self.operation_name = operation_name
        self.capture_system = capture_system
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.capture_system:
            self.profiler.capture_system_metrics()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to ms
            self.profiler.record_metric(
                self.operation_name, 
                duration, 
                "ms", 
                "operation",
                success=exc_type is None
            )
            
            if self.capture_system:
                self.profiler.capture_system_metrics()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler