"""
Nexus Visualization Module - TensorBoard Integration and Training Visualization

This module provides comprehensive training visualization and monitoring capabilities
for the Nexus Game AI Framework, including TensorBoard integration, real-time metrics,
and video logging.
"""

from nexus.visualization.tensorboard_logger import TensorBoardLogger
from nexus.visualization.metrics_tracker import MetricsTracker
from nexus.visualization.training_dashboard import TrainingDashboard
from nexus.visualization.video_logger import VideoLogger
from nexus.visualization.experiment_manager import ExperimentManager

__all__ = [
    'TensorBoardLogger',
    'MetricsTracker',
    'TrainingDashboard',
    'VideoLogger',
    'ExperimentManager'
]

# Version info
__version__ = '1.0.0'