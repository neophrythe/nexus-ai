"""
Configuration schema for Performance Monitor Plugin.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class PerformanceConfig:
    """Configuration for Performance Monitor Plugin."""
    
    # Monitoring settings
    enabled: bool = True
    monitor_interval: float = 0.1  # seconds
    
    # Display settings
    show_overlay: bool = True
    overlay_position: str = 'top_left'  # top_left, top_right, bottom_left, bottom_right
    overlay_opacity: float = 0.8
    overlay_scale: float = 1.0
    
    # Metrics to monitor
    monitor_fps: bool = True
    monitor_frame_time: bool = True
    monitor_cpu: bool = True
    monitor_memory: bool = True
    monitor_gpu: bool = True
    monitor_network: bool = True
    monitor_disk: bool = True
    monitor_temperature: bool = False
    
    # Performance thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_fps': 30,
        'max_frame_time': 33.33,
        'max_cpu': 80,
        'max_memory': 80,
        'max_gpu': 90,
        'max_temperature': 85,
        'max_network_latency': 100
    })
    
    # Alert settings
    alerts_enabled: bool = True
    alert_sound: bool = False
    alert_log_file: Optional[str] = None
    alert_cooldown: float = 5.0  # seconds between same alert
    
    # Export settings
    export_metrics: bool = False
    export_interval: float = 60.0  # seconds
    export_format: str = 'csv'  # csv, json, influxdb
    export_path: str = './performance_logs'
    
    # Advanced settings
    gpu_device_index: int = 0  # Which GPU to monitor
    cpu_per_core: bool = False  # Monitor per-core CPU usage
    memory_detailed: bool = False  # Detailed memory breakdown
    network_interface: Optional[str] = None  # Specific network interface
    
    # Optimization settings
    auto_optimize: bool = False
    optimization_profile: str = 'balanced'  # performance, balanced, quality
    dynamic_resolution: bool = False
    target_fps: int = 60
    
    @classmethod
    def from_file(cls, filepath: str) -> 'PerformanceConfig':
        """Load configuration from file."""
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def to_file(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.__dict__, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate monitor interval
        if self.monitor_interval <= 0 or self.monitor_interval > 10:
            errors.append("monitor_interval must be between 0 and 10 seconds")
        
        # Validate overlay settings
        valid_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        if self.overlay_position not in valid_positions:
            errors.append(f"overlay_position must be one of {valid_positions}")
        
        if not 0 <= self.overlay_opacity <= 1:
            errors.append("overlay_opacity must be between 0 and 1")
        
        if not 0.5 <= self.overlay_scale <= 2.0:
            errors.append("overlay_scale must be between 0.5 and 2.0")
        
        # Validate thresholds
        if self.thresholds['min_fps'] < 1:
            errors.append("min_fps must be at least 1")
        
        if self.thresholds['max_frame_time'] < 1:
            errors.append("max_frame_time must be at least 1ms")
        
        for key in ['max_cpu', 'max_memory', 'max_gpu']:
            if not 0 < self.thresholds[key] <= 100:
                errors.append(f"{key} must be between 0 and 100")
        
        # Validate export settings
        valid_formats = ['csv', 'json', 'influxdb']
        if self.export_format not in valid_formats:
            errors.append(f"export_format must be one of {valid_formats}")
        
        # Validate optimization settings
        valid_profiles = ['performance', 'balanced', 'quality']
        if self.optimization_profile not in valid_profiles:
            errors.append(f"optimization_profile must be one of {valid_profiles}")
        
        if self.target_fps < 1 or self.target_fps > 300:
            errors.append("target_fps must be between 1 and 300")
        
        return errors
    
    def apply_profile(self, profile: str):
        """Apply a preset configuration profile."""
        profiles = {
            'minimal': {
                'monitor_gpu': False,
                'monitor_network': False,
                'monitor_disk': False,
                'show_overlay': False,
                'export_metrics': False
            },
            'basic': {
                'monitor_gpu': True,
                'monitor_network': False,
                'monitor_disk': False,
                'show_overlay': True,
                'export_metrics': False
            },
            'full': {
                'monitor_gpu': True,
                'monitor_network': True,
                'monitor_disk': True,
                'monitor_temperature': True,
                'show_overlay': True,
                'export_metrics': True,
                'cpu_per_core': True,
                'memory_detailed': True
            },
            'competitive': {
                'show_overlay': True,
                'overlay_position': 'top_right',
                'overlay_scale': 0.8,
                'monitor_network': True,
                'thresholds': {
                    'min_fps': 60,
                    'max_frame_time': 16.67,
                    'max_network_latency': 50
                },
                'auto_optimize': True,
                'optimization_profile': 'performance',
                'target_fps': 144
            }
        }
        
        if profile in profiles:
            for key, value in profiles[profile].items():
                setattr(self, key, value)
        else:
            raise ValueError(f"Unknown profile: {profile}")