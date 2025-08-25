"""
TensorBoard Logger for Nexus Game AI Framework

Provides comprehensive TensorBoard integration for training visualization,
metrics tracking, and experiment management.
"""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
import numpy as np
import structlog

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
try:
    import tensorflow as tf
    from tensorboard import program
    HAS_TF = True
except ImportError:
    HAS_TF = False

logger = structlog.get_logger()


class TensorBoardLogger:
    """
    Comprehensive TensorBoard logging for game AI training.
    
    Features:
    - Automatic TensorBoard server management
    - Real-time metric logging
    - Video and image logging
    - Model graph visualization
    - Hyperparameter tracking
    - Multi-experiment comparison
    """
    
    def __init__(self, 
                 log_dir: str = "./runs",
                 experiment_name: Optional[str] = None,
                 auto_start_server: bool = True,
                 server_port: int = 6006,
                 flush_interval: int = 30):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment run
            auto_start_server: Automatically start TensorBoard server
            server_port: Port for TensorBoard server
            flush_interval: Seconds between automatic flushes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_path = self.log_dir / experiment_name
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer based on available backend
        self.writer = None
        self.backend = None
        self._init_writer()
        
        # Server management
        self.server_port = server_port
        self.server_process = None
        if auto_start_server:
            self.start_tensorboard_server()
        
        # Metrics tracking
        self.global_step = 0
        self.flush_interval = flush_interval
        self.last_flush_time = time.time()
        
        # Metric history for aggregation
        self.metric_history: Dict[str, List[float]] = {}
        self.image_counter = 0
        self.video_counter = 0
        
        logger.info(f"TensorBoard logger initialized: {self.experiment_path}")
        logger.info(f"View at: http://localhost:{self.server_port}")
    
    def _init_writer(self):
        """Initialize the appropriate writer based on available backend."""
        if HAS_TORCH:
            self.writer = SummaryWriter(str(self.experiment_path))
            self.backend = 'torch'
            logger.info("Using PyTorch TensorBoard backend")
        elif HAS_TF:
            self.writer = tf.summary.create_file_writer(str(self.experiment_path))
            self.backend = 'tensorflow'
            logger.info("Using TensorFlow TensorBoard backend")
        else:
            logger.warning("No TensorBoard backend available (install torch or tensorflow)")
            self.backend = None
    
    def start_tensorboard_server(self) -> bool:
        """Start TensorBoard server in background."""
        try:
            # Check if server already running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.server_port))
            sock.close()
            
            if result == 0:
                logger.info(f"TensorBoard already running on port {self.server_port}")
                return True
            
            # Start TensorBoard server
            cmd = [
                'tensorboard',
                '--logdir', str(self.log_dir),
                '--port', str(self.server_port),
                '--reload_interval', '5'
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for server to start
            time.sleep(2)
            
            logger.info(f"TensorBoard server started at http://localhost:{self.server_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start TensorBoard server: {e}")
            return False
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value."""
        if self.writer is None:
            return
        
        step = step or self.global_step
        
        if self.backend == 'torch':
            self.writer.add_scalar(tag, value, step)
        elif self.backend == 'tensorflow':
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)
        
        # Track history
        if tag not in self.metric_history:
            self.metric_history[tag] = []
        self.metric_history[tag].append(value)
        
        self._check_flush()
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], 
                   step: Optional[int] = None):
        """Log multiple scalars under one tag."""
        if self.writer is None:
            return
        
        step = step or self.global_step
        
        if self.backend == 'torch':
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        else:
            for tag, value in tag_scalar_dict.items():
                self.log_scalar(f"{main_tag}/{tag}", value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log a histogram of values."""
        if self.writer is None:
            return
        
        step = step or self.global_step
        
        if self.backend == 'torch':
            self.writer.add_histogram(tag, values, step)
        elif self.backend == 'tensorflow':
            with self.writer.as_default():
                tf.summary.histogram(tag, values, step=step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """
        Log an image.
        
        Args:
            tag: Image tag
            image: Image array (HWC format, RGB)
            step: Global step
        """
        if self.writer is None:
            return
        
        step = step or self.global_step
        self.image_counter += 1
        
        if self.backend == 'torch':
            # Convert HWC to CHW for PyTorch
            if len(image.shape) == 3:
                image = np.transpose(image, (2, 0, 1))
            self.writer.add_image(tag, image, step, dataformats='CHW')
        elif self.backend == 'tensorflow':
            with self.writer.as_default():
                # TensorFlow expects HWC
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                image = np.expand_dims(image, axis=0)  # Add batch dimension
                tf.summary.image(tag, image, step=step)
    
    def log_video(self, tag: str, video: np.ndarray, fps: int = 30, 
                 step: Optional[int] = None):
        """
        Log a video.
        
        Args:
            tag: Video tag
            video: Video array (THWC format)
            fps: Frames per second
            step: Global step
        """
        if self.writer is None or self.backend != 'torch':
            logger.warning("Video logging only supported with PyTorch backend")
            return
        
        step = step or self.global_step
        self.video_counter += 1
        
        # Convert THWC to NTCHW for PyTorch (N=1, T=frames, C=channels, H, W)
        if len(video.shape) == 4:
            video = np.transpose(video, (0, 3, 1, 2))
            video = np.expand_dims(video, axis=0)
        
        self.writer.add_video(tag, video, step, fps=fps)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text data."""
        if self.writer is None:
            return
        
        step = step or self.global_step
        
        if self.backend == 'torch':
            self.writer.add_text(tag, text, step)
        elif self.backend == 'tensorflow':
            with self.writer.as_default():
                tf.summary.text(tag, text, step=step)
    
    def log_graph(self, model: Any, input_sample: Any = None):
        """Log model architecture graph."""
        if self.writer is None or self.backend != 'torch':
            logger.warning("Graph logging only supported with PyTorch backend")
            return
        
        try:
            if input_sample is not None:
                self.writer.add_graph(model, input_sample)
            else:
                # Try to create dummy input
                import torch
                if hasattr(model, 'observation_space'):
                    shape = model.observation_space.shape
                    dummy_input = torch.randn(1, *shape)
                    self.writer.add_graph(model, dummy_input)
        except Exception as e:
            logger.error(f"Failed to log model graph: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float] = None):
        """Log hyperparameters and optionally their metrics."""
        if self.writer is None:
            return
        
        if self.backend == 'torch':
            self.writer.add_hparams(hparams, metrics or {})
        else:
            # Log as text for TensorFlow
            hparam_str = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
            self.log_text("hyperparameters", hparam_str)
            
            if metrics:
                for key, value in metrics.items():
                    self.log_scalar(f"hparam/{key}", value)
    
    def log_agent_metrics(self, agent_name: str, metrics: Dict[str, float], 
                         step: Optional[int] = None):
        """Log agent-specific metrics."""
        step = step or self.global_step
        
        for metric_name, value in metrics.items():
            self.log_scalar(f"{agent_name}/{metric_name}", value, step)
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   additional_metrics: Dict[str, float] = None):
        """Log episode statistics."""
        self.log_scalar("episode/reward", reward, episode)
        self.log_scalar("episode/length", length, episode)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.log_scalar(f"episode/{key}", value, episode)
    
    def log_training_step(self, loss: float, learning_rate: float = None, 
                         gradients: Dict[str, np.ndarray] = None,
                         step: Optional[int] = None):
        """Log training step metrics."""
        step = step or self.global_step
        
        self.log_scalar("training/loss", loss, step)
        
        if learning_rate is not None:
            self.log_scalar("training/learning_rate", learning_rate, step)
        
        if gradients:
            for name, grad in gradients.items():
                self.log_histogram(f"gradients/{name}", grad, step)
                self.log_scalar(f"gradients/{name}_norm", np.linalg.norm(grad), step)
    
    def log_game_frame(self, frame: np.ndarray, overlays: Dict[str, Any] = None, 
                      step: Optional[int] = None):
        """
        Log a game frame with optional overlays.
        
        Args:
            frame: Game frame (HWC format)
            overlays: Dictionary of overlays to add (bounding boxes, etc.)
            step: Global step
        """
        step = step or self.global_step
        
        # Add overlays if provided
        if overlays:
            import cv2
            frame = frame.copy()
            
            # Draw bounding boxes
            if 'boxes' in overlays:
                for box in overlays['boxes']:
                    x1, y1, x2, y2 = box['coords']
                    color = box.get('color', (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if 'label' in box:
                        cv2.putText(frame, box['label'], (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw keypoints
            if 'keypoints' in overlays:
                for point in overlays['keypoints']:
                    x, y = point['coords']
                    color = point.get('color', (255, 0, 0))
                    cv2.circle(frame, (x, y), 5, color, -1)
        
        self.log_image("game/frame", frame, step)
    
    def create_comparison_plot(self, experiments: List[str], metric: str) -> np.ndarray:
        """Create comparison plot across multiple experiments."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp_name in experiments:
            exp_path = self.log_dir / exp_name
            # Load metrics from experiment
            # This would need actual implementation to read TensorBoard logs
            pass
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.legend()
        ax.grid(True)
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return image
    
    def step(self):
        """Increment global step."""
        self.global_step += 1
    
    def _check_flush(self):
        """Check if we should flush the writer."""
        current_time = time.time()
        if current_time - self.last_flush_time > self.flush_interval:
            self.flush()
            self.last_flush_time = current_time
    
    def flush(self):
        """Flush all pending logs."""
        if self.writer is None:
            return
        
        if self.backend == 'torch':
            self.writer.flush()
        elif self.backend == 'tensorflow':
            self.writer.flush()
    
    def close(self):
        """Close the logger and stop TensorBoard server."""
        if self.writer is not None:
            self.flush()
            
            if self.backend == 'torch':
                self.writer.close()
            elif self.backend == 'tensorflow':
                self.writer.close()
        
        # Stop TensorBoard server
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
            logger.info("TensorBoard server stopped")
        
        logger.info(f"TensorBoard logger closed. Logged {self.image_counter} images, "
                   f"{self.video_counter} videos")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all logged metrics."""
        summary = {}
        
        for metric_name, values in self.metric_history.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1],
                    'count': len(values)
                }
        
        return summary
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


# Convenience function for global logger
_global_logger: Optional[TensorBoardLogger] = None

def get_tensorboard_logger(log_dir: str = "./runs", **kwargs) -> TensorBoardLogger:
    """Get or create global TensorBoard logger."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = TensorBoardLogger(log_dir, **kwargs)
    
    return _global_logger