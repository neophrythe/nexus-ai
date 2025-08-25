"""Analytics client for tracking training metrics and events - adapted from SerpentAI"""

import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from collections import deque
from pathlib import Path
import structlog
import numpy as np

logger = structlog.get_logger()

# Try importing Redis for advanced features
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class AnalyticsClient:
    """Modern analytics client with multiple backends"""
    
    def __init__(self, project_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analytics client.
        
        Args:
            project_key: Unique project identifier
            config: Configuration dictionary
        """
        self.project_key = project_key
        self.config = config or {}
        
        # Backend selection
        self.use_redis = self.config.get('use_redis', False) and REDIS_AVAILABLE
        self.use_file = self.config.get('use_file', True)
        self.use_memory = self.config.get('use_memory', True)
        
        # Options
        self.broadcast = self.config.get('broadcast', True)
        self.debug = self.config.get('debug', False)
        self.event_whitelist: Optional[Set[str]] = None
        self.event_blacklist: Optional[Set[str]] = None
        
        # Set up whitelists/blacklists
        if 'event_whitelist' in self.config:
            self.event_whitelist = set(self.config['event_whitelist'])
        if 'event_blacklist' in self.config:
            self.event_blacklist = set(self.config['event_blacklist'])
        
        # Storage backends
        self.memory_buffer: deque = deque(maxlen=self.config.get('memory_buffer_size', 10000))
        self.redis_client: Optional[redis.StrictRedis] = None
        self.file_path: Optional[Path] = None
        
        # Metrics aggregation
        self.aggregated_metrics: Dict[str, List[float]] = {}
        self.event_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.total_events = 0
        
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize storage backends"""
        # Redis backend
        if self.use_redis and REDIS_AVAILABLE:
            try:
                redis_config = self.config.get('redis', {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0
                })
                self.redis_client = redis.StrictRedis(**redis_config)
                self.redis_client.ping()
                logger.info(f"Redis backend initialized for project '{self.project_key}'")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.use_redis = False
        
        # File backend
        if self.use_file:
            analytics_dir = Path(self.config.get('analytics_dir', 'analytics'))
            analytics_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.file_path = analytics_dir / f"{self.project_key}_{timestamp}.jsonl"
            logger.info(f"File backend initialized: {self.file_path}")
    
    @property
    def redis_key(self) -> str:
        """Get Redis key for events"""
        return f"NEXUS:{self.project_key}:EVENTS"
    
    @property
    def redis_metrics_key(self) -> str:
        """Get Redis key for metrics"""
        return f"NEXUS:{self.project_key}:METRICS"
    
    def track(self, event_key: str, data: Any = None, 
             timestamp: Optional[str] = None, 
             is_persistable: bool = True) -> None:
        """
        Track an analytics event.
        
        Args:
            event_key: Event identifier
            data: Event data
            timestamp: Event timestamp (auto-generated if None)
            is_persistable: Whether to persist the event
        """
        # Check filters
        if not self._should_track_event(event_key):
            return
        
        # Create event
        event = {
            "project_key": self.project_key,
            "event_key": event_key,
            "data": data,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "is_persistable": is_persistable
        }
        
        # Debug output
        if self.debug:
            logger.debug(f"Event: {event_key}", extra={"data": data})
        
        # Track event count
        self.event_counts[event_key] = self.event_counts.get(event_key, 0) + 1
        self.total_events += 1
        
        # Store event
        if self.broadcast and is_persistable:
            self._store_event(event)
        
        # Aggregate metrics if applicable
        if isinstance(data, dict):
            self._aggregate_metrics(event_key, data)
    
    def _should_track_event(self, event_key: str) -> bool:
        """Check if event should be tracked based on filters"""
        if self.event_whitelist and event_key not in self.event_whitelist:
            return False
        if self.event_blacklist and event_key in self.event_blacklist:
            return False
        return True
    
    def _store_event(self, event: Dict[str, Any]):
        """Store event in configured backends"""
        event_json = json.dumps(event)
        
        # Memory backend
        if self.use_memory:
            self.memory_buffer.append(event)
        
        # Redis backend
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.lpush(self.redis_key, event_json)
                # Set expiry if configured
                ttl = self.config.get('redis_ttl', 86400)  # 24 hours default
                self.redis_client.expire(self.redis_key, ttl)
            except Exception as e:
                logger.error(f"Failed to store event in Redis: {e}")
        
        # File backend
        if self.use_file and self.file_path:
            try:
                with open(self.file_path, 'a') as f:
                    f.write(event_json + '\n')
            except Exception as e:
                logger.error(f"Failed to store event in file: {e}")
    
    def _aggregate_metrics(self, event_key: str, data: Dict[str, Any]):
        """Aggregate numerical metrics for analysis"""
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metric_key = f"{event_key}:{key}"
                if metric_key not in self.aggregated_metrics:
                    self.aggregated_metrics[metric_key] = []
                self.aggregated_metrics[metric_key].append(value)
    
    def track_metric(self, metric_name: str, value: float, 
                    step: Optional[int] = None, **tags):
        """
        Track a numerical metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Training step or epoch
            **tags: Additional tags for the metric
        """
        data = {
            "value": value,
            "step": step,
            **tags
        }
        self.track(f"METRIC:{metric_name}", data)
    
    def track_reward(self, reward: float, episode: Optional[int] = None,
                    cumulative: Optional[float] = None):
        """Track reward metrics"""
        data = {
            "reward": reward,
            "episode": episode,
            "cumulative": cumulative
        }
        self.track("REWARD", data)
    
    def track_action(self, action: Any, state: Optional[Any] = None,
                    q_values: Optional[List[float]] = None):
        """Track agent action"""
        data = {
            "action": action,
            "q_values": q_values
        }
        if state is not None and hasattr(state, 'shape'):
            data["state_shape"] = state.shape
        self.track("ACTION", data)
    
    def track_loss(self, loss_type: str, value: float, **additional_losses):
        """Track training loss"""
        data = {
            loss_type: value,
            **additional_losses
        }
        self.track("LOSS", data)
    
    def track_episode(self, episode: int, total_reward: float,
                     steps: int, **additional_metrics):
        """Track episode completion"""
        data = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            **additional_metrics
        }
        self.track("EPISODE", data)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics"""
        summary = {
            "project_key": self.project_key,
            "runtime": time.time() - self.start_time,
            "total_events": self.total_events,
            "event_counts": self.event_counts,
            "metrics": {}
        }
        
        # Calculate statistics for aggregated metrics
        for metric_key, values in self.aggregated_metrics.items():
            if values:
                summary["metrics"][metric_key] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "last": values[-1]
                }
        
        return summary
    
    def get_recent_events(self, count: int = 100, 
                         event_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent events from memory buffer.
        
        Args:
            count: Number of events to retrieve
            event_key: Filter by event key
        
        Returns:
            List of recent events
        """
        events = list(self.memory_buffer)
        
        if event_key:
            events = [e for e in events if e.get('event_key') == event_key]
        
        return events[-count:]
    
    def clear_memory_buffer(self):
        """Clear the memory buffer"""
        self.memory_buffer.clear()
        logger.info("Memory buffer cleared")
    
    def export_to_file(self, output_path: str, format: str = 'jsonl'):
        """
        Export events to file.
        
        Args:
            output_path: Path to output file
            format: Export format ('jsonl', 'json', 'csv')
        """
        output_path = Path(output_path)
        
        if format == 'jsonl':
            with open(output_path, 'w') as f:
                for event in self.memory_buffer:
                    f.write(json.dumps(event) + '\n')
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(list(self.memory_buffer), f, indent=2)
        elif format == 'csv':
            import csv
            
            if self.memory_buffer:
                with open(output_path, 'w', newline='') as f:
                    # Use first event to determine fields
                    fieldnames = list(self.memory_buffer[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for event in self.memory_buffer:
                        # Flatten nested data
                        flat_event = {}
                        for key, value in event.items():
                            if isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    flat_event[f"{key}_{sub_key}"] = sub_value
                            else:
                                flat_event[key] = value
                        writer.writerow(flat_event)
        
        logger.info(f"Exported {len(self.memory_buffer)} events to {output_path}")
    
    async def flush_to_redis(self):
        """Flush memory buffer to Redis (async)"""
        if not self.redis_client or not self.memory_buffer:
            return
        
        pipeline = self.redis_client.pipeline()
        
        for event in self.memory_buffer:
            pipeline.lpush(self.redis_key, json.dumps(event))
        
        try:
            await asyncio.get_event_loop().run_in_executor(None, pipeline.execute)
            logger.info(f"Flushed {len(self.memory_buffer)} events to Redis")
        except Exception as e:
            logger.error(f"Failed to flush to Redis: {e}")
    
    def close(self):
        """Clean up resources"""
        # Export remaining events if configured
        if self.config.get('export_on_close'):
            export_path = self.config.get('export_path', f"{self.project_key}_final.jsonl")
            self.export_to_file(export_path)
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info(f"Analytics client closed. Total events: {self.total_events}")


class NoOpAnalyticsClient:
    """No-op analytics client for when analytics are disabled"""
    
    def __init__(self, *args, **kwargs):
        logger.debug("NoOpAnalyticsClient initialized - analytics disabled")
    
    def track(self, *args, **kwargs):
        logger.debug("NoOp: track() called but analytics disabled")
    
    def track_metric(self, *args, **kwargs):
        logger.debug("NoOp: track_metric() called but analytics disabled")
    
    def track_reward(self, *args, **kwargs):
        logger.debug("NoOp: track_reward() called but analytics disabled")
    
    def track_action(self, *args, **kwargs):
        logger.debug("NoOp: track_action() called but analytics disabled")
    
    def track_loss(self, *args, **kwargs):
        logger.debug("NoOp: track_loss() called but analytics disabled")
    
    def track_episode(self, *args, **kwargs):
        logger.debug("NoOp: track_episode() called but analytics disabled")
    
    def get_metrics_summary(self):
        return {}
    
    def get_recent_events(self, *args, **kwargs):
        return []
    
    def clear_memory_buffer(self):
        logger.debug("NoOp: clear_memory_buffer() called but analytics disabled")
    
    def export_to_file(self, *args, **kwargs):
        logger.debug("NoOp: export_to_file() called but analytics disabled")
    
    async def flush_to_redis(self):
        logger.debug("NoOp: flush_to_redis() called but analytics disabled")
    
    def close(self):
        logger.debug("NoOp: close() called but analytics disabled")