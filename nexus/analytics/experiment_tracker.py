"""Experiment Tracking and Analytics System for Nexus Framework"""

import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import structlog
import hashlib
import sqlite3
import threading
from queue import Queue

# Optional integrations
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics"""
    SCALAR = "scalar"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    CUSTOM = "custom"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Single metric data point"""
    name: str
    value: Any
    timestamp: float
    step: int
    metric_type: MetricType
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Experiment information"""
    experiment_id: str
    name: str
    project: str
    created_at: datetime
    config: Dict[str, Any]
    tags: List[str]
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class ExperimentTracker:
    """Main experiment tracking system"""
    
    def __init__(self, project_name: str = "nexus",
                 experiment_name: Optional[str] = None,
                 tracking_dir: Optional[Path] = None,
                 auto_save_interval: int = 60):
        """
        Initialize experiment tracker
        
        Args:
            project_name: Name of the project
            experiment_name: Name of the experiment
            tracking_dir: Directory for tracking data
            auto_save_interval: Auto-save interval in seconds
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.tracking_dir = tracking_dir or Path.home() / ".nexus" / "experiments"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Create experiment
        self.experiment = Experiment(
            experiment_id=self.experiment_id,
            name=self.experiment_name,
            project=self.project_name,
            created_at=datetime.now(),
            config={},
            tags=[],
            status="running"
        )
        
        # Metrics storage
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.current_step = 0
        
        # Database
        self.db_path = self.tracking_dir / f"{self.experiment_id}.db"
        self._init_database()
        
        # Background saving
        self.save_queue = Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        self.auto_save_interval = auto_save_interval
        self.last_save_time = time.time()
        
        # Integrations
        self.integrations = []
        self._init_integrations()
        
        logger.info(f"Initialized experiment tracker: {self.experiment_name} ({self.experiment_id})")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        data = f"{self.project_name}_{self.experiment_name}_{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT,
                project TEXT,
                created_at TEXT,
                config TEXT,
                tags TEXT,
                status TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                name TEXT,
                value REAL,
                timestamp REAL,
                step INTEGER,
                metric_type TEXT,
                tags TEXT,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics (step)
        """)
        
        # Save experiment
        cursor.execute("""
            INSERT OR REPLACE INTO experiments 
            (experiment_id, name, project, created_at, config, tags, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.experiment.experiment_id,
            self.experiment.name,
            self.experiment.project,
            self.experiment.created_at.isoformat(),
            json.dumps(self.experiment.config),
            json.dumps(self.experiment.tags),
            self.experiment.status,
            json.dumps(self.experiment.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _init_integrations(self):
        """Initialize third-party integrations"""
        # W&B
        if WANDB_AVAILABLE and self._should_use_wandb():
            try:
                wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    id=self.experiment_id,
                    config=self.experiment.config
                )
                self.integrations.append("wandb")
                logger.info("Weights & Biases integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
        
        # MLflow
        if MLFLOW_AVAILABLE and self._should_use_mlflow():
            try:
                mlflow.set_experiment(self.project_name)
                mlflow.start_run(run_name=self.experiment_name)
                self.integrations.append("mlflow")
                logger.info("MLflow integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
        
        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            try:
                tb_dir = self.tracking_dir / "tensorboard" / self.experiment_id
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(str(tb_dir))
                self.integrations.append("tensorboard")
                logger.info("TensorBoard integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
    
    def _should_use_wandb(self) -> bool:
        """Check if W&B should be used"""
        return os.environ.get("WANDB_API_KEY") is not None
    
    def _should_use_mlflow(self) -> bool:
        """Check if MLflow should be used"""
        return os.environ.get("MLFLOW_TRACKING_URI") is not None
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration
        
        Args:
            config: Configuration dictionary
        """
        self.experiment.config.update(config)
        
        # Update integrations
        if "wandb" in self.integrations:
            wandb.config.update(config)
        
        if "mlflow" in self.integrations:
            for key, value in config.items():
                mlflow.log_param(key, value)
        
        # Save to database
        self._save_experiment()
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None,
                  tags: Optional[Dict[str, Any]] = None):
        """
        Log a scalar metric
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            tags: Additional tags
        """
        step = step or self.current_step
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            step=step,
            metric_type=MetricType.SCALAR,
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        
        # Log to integrations
        if "wandb" in self.integrations:
            wandb.log({name: value}, step=step)
        
        if "mlflow" in self.integrations:
            mlflow.log_metric(name, value, step=step)
        
        if "tensorboard" in self.integrations:
            self.tb_writer.add_scalar(name, value, step)
        
        # Queue for saving
        self.save_queue.put(("metric", metric))
        
        # Auto-save check
        self._check_auto_save()
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log histogram
        
        Args:
            name: Histogram name
            values: Values array
            step: Training step
        """
        step = step or self.current_step
        
        metric = Metric(
            name=name,
            value=values.tolist(),
            timestamp=time.time(),
            step=step,
            metric_type=MetricType.HISTOGRAM,
            tags={}
        )
        
        self.metrics[name].append(metric)
        
        # Log to integrations
        if "wandb" in self.integrations:
            wandb.log({name: wandb.Histogram(values)}, step=step)
        
        if "tensorboard" in self.integrations:
            self.tb_writer.add_histogram(name, values, step)
        
        self.save_queue.put(("metric", metric))
    
    def log_image(self, name: str, image: np.ndarray, step: Optional[int] = None,
                 caption: Optional[str] = None):
        """
        Log image
        
        Args:
            name: Image name
            image: Image array
            step: Training step
            caption: Image caption
        """
        step = step or self.current_step
        
        # Save image to file
        image_dir = self.tracking_dir / "images" / self.experiment_id
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{name}_{step}.png"
        
        # Save image
        import cv2
        cv2.imwrite(str(image_path), image)
        
        metric = Metric(
            name=name,
            value=str(image_path),
            timestamp=time.time(),
            step=step,
            metric_type=MetricType.IMAGE,
            tags={},
            metadata={"caption": caption} if caption else {}
        )
        
        self.metrics[name].append(metric)
        
        # Log to integrations
        if "wandb" in self.integrations:
            wandb.log({name: wandb.Image(image, caption=caption)}, step=step)
        
        if "tensorboard" in self.integrations:
            # Convert to HWC format if needed
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))
            self.tb_writer.add_image(name, image, step, dataformats='HWC')
        
        self.save_queue.put(("metric", metric))
    
    def log_text(self, name: str, text: str, step: Optional[int] = None):
        """
        Log text
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        step = step or self.current_step
        
        metric = Metric(
            name=name,
            value=text,
            timestamp=time.time(),
            step=step,
            metric_type=MetricType.TEXT,
            tags={}
        )
        
        self.metrics[name].append(metric)
        
        # Log to integrations
        if "wandb" in self.integrations:
            wandb.log({name: wandb.Html(text)}, step=step)
        
        if "tensorboard" in self.integrations:
            self.tb_writer.add_text(name, text, step)
        
        self.save_queue.put(("metric", metric))
    
    def log_model(self, model: Any, name: str = "model"):
        """
        Log model checkpoint
        
        Args:
            model: Model to save
            name: Model name
        """
        model_dir = self.tracking_dir / "models" / self.experiment_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save based on framework
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                model_path = model_dir / f"{name}_{self.current_step}.pth"
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved PyTorch model to {model_path}")
        except ImportError:
            logger.debug("PyTorch not available for model saving")
        
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                model_path = model_dir / f"{name}_{self.current_step}"
                model.save(model_path)
                logger.info(f"Saved TensorFlow model to {model_path}")
        except ImportError:
            logger.debug("TensorFlow not available for model saving")
        
        # Fallback to pickle
        model_path = model_dir / f"{name}_{self.current_step}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_path}")
        
        # Log to MLflow
        if "mlflow" in self.integrations:
            mlflow.log_artifact(str(model_path))
    
    def step(self):
        """Increment training step"""
        self.current_step += 1
    
    def finish(self, status: str = "completed"):
        """
        Finish experiment
        
        Args:
            status: Final status
        """
        self.experiment.status = status
        
        # Finish integrations
        if "wandb" in self.integrations:
            wandb.finish()
        
        if "mlflow" in self.integrations:
            mlflow.end_run()
        
        if "tensorboard" in self.integrations:
            self.tb_writer.close()
        
        # Final save
        self._save_all()
        
        logger.info(f"Experiment finished: {self.experiment_name} ({status})")
    
    def _save_worker(self):
        """Background worker for saving metrics"""
        while True:
            try:
                item = self.save_queue.get()
                
                if item is None:
                    break
                
                item_type, data = item
                
                if item_type == "metric":
                    self._save_metric_to_db(data)
                
            except Exception as e:
                logger.error(f"Save worker error: {e}")
    
    def _save_metric_to_db(self, metric: Metric):
        """Save metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics 
            (experiment_id, name, value, timestamp, step, metric_type, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.experiment_id,
            metric.name,
            json.dumps(metric.value) if not isinstance(metric.value, (int, float)) else metric.value,
            metric.timestamp,
            metric.step,
            metric.metric_type.value,
            json.dumps(metric.tags),
            json.dumps(metric.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _save_experiment(self):
        """Save experiment metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE experiments 
            SET config = ?, tags = ?, status = ?, metadata = ?
            WHERE experiment_id = ?
        """, (
            json.dumps(self.experiment.config),
            json.dumps(self.experiment.tags),
            self.experiment.status,
            json.dumps(self.experiment.metadata),
            self.experiment_id
        ))
        
        conn.commit()
        conn.close()
    
    def _check_auto_save(self):
        """Check if auto-save is needed"""
        if time.time() - self.last_save_time > self.auto_save_interval:
            self._save_all()
            self.last_save_time = time.time()
    
    def _save_all(self):
        """Save all pending data"""
        # Process remaining queue items
        while not self.save_queue.empty():
            time.sleep(0.01)
        
        # Save experiment metadata
        self._save_experiment()
        
        # Save metrics summary
        self._save_metrics_summary()
    
    def _save_metrics_summary(self):
        """Save metrics summary to JSON"""
        summary_path = self.tracking_dir / f"{self.experiment_id}_summary.json"
        
        summary = {
            "experiment": self.experiment.to_dict(),
            "metrics": {}
        }
        
        for name, metrics in self.metrics.items():
            if metrics:
                values = [m.value for m in metrics if isinstance(m.value, (int, float))]
                if values:
                    summary["metrics"][name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "last": values[-1]
                    }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, List[Metric]]:
        """
        Get metrics
        
        Args:
            metric_name: Specific metric name
        
        Returns:
            Dictionary of metrics
        """
        if metric_name:
            return {metric_name: self.metrics.get(metric_name, [])}
        return dict(self.metrics)
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None,
                     save_path: Optional[str] = None):
        """
        Plot metrics
        
        Args:
            metric_names: List of metric names to plot
            save_path: Path to save plot
        """
        if not metric_names:
            metric_names = [name for name in self.metrics.keys() 
                          if self.metrics[name] and 
                          self.metrics[name][0].metric_type == MetricType.SCALAR]
        
        if not metric_names:
            logger.warning("No metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, name in enumerate(metric_names):
            if name not in self.metrics:
                continue
            
            metrics = self.metrics[name]
            steps = [m.step for m in metrics]
            values = [m.value for m in metrics if isinstance(m.value, (int, float))]
            
            if not values:
                continue
            
            ax = axes[idx] if n_metrics > 1 else axes[0]
            ax.plot(steps[:len(values)], values)
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved metrics plot to {save_path}")
        else:
            plt.show()


# Global tracker instance
_global_tracker: Optional[ExperimentTracker] = None


def init_tracker(project_name: str = "nexus", 
                experiment_name: Optional[str] = None,
                **kwargs) -> ExperimentTracker:
    """Initialize global experiment tracker"""
    global _global_tracker
    _global_tracker = ExperimentTracker(project_name, experiment_name, **kwargs)
    return _global_tracker


def get_tracker() -> Optional[ExperimentTracker]:
    """Get global experiment tracker"""
    return _global_tracker


# Convenience functions
def log_metric(name: str, value: float, **kwargs):
    """Log metric to global tracker"""
    if _global_tracker:
        _global_tracker.log_metric(name, value, **kwargs)


def log_metrics(metrics: Dict[str, float], **kwargs):
    """Log metrics to global tracker"""
    if _global_tracker:
        _global_tracker.log_metrics(metrics, **kwargs)


def log_config(config: Dict[str, Any]):
    """Log config to global tracker"""
    if _global_tracker:
        _global_tracker.log_config(config)