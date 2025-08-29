"""Data models for Nexus API"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class SystemStatus:
    """System status information"""
    status: str = "running"
    uptime_seconds: float = 0.0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_plugins: int = 0
    capture_fps: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PluginInfo:
    """Plugin information"""
    name: str
    version: str
    enabled: bool
    status: str
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CaptureStats:
    """Capture statistics"""
    fps: float
    frame_count: int
    dropped_frames: int
    capture_time_ms: float
    backend: str
    resolution: tuple[int, int]
    buffer_size: int
    buffer_usage: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()