"""
Performance Monitor Plugin

Real-time performance monitoring and optimization for games.
"""

from plugins.performance_monitor.plugin import PerformanceMonitorPlugin, create_plugin
from plugins.performance_monitor.config import PerformanceConfig

__all__ = ['PerformanceMonitorPlugin', 'PerformanceConfig', 'create_plugin']
__version__ = '1.0.0'