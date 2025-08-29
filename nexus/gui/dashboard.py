"""
Nexus AI Framework - Performance Dashboard
Real-time monitoring and analytics dashboard.
"""

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

import numpy as np
from collections import deque
import time
import json
from typing import Dict, List, Any


class Dashboard(QWidget if PYQT_AVAILABLE else object):
    """Performance monitoring dashboard."""
    
    def __init__(self):
        super().__init__()
        self.metrics_history = {
            'fps': deque(maxlen=100),
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'gpu': deque(maxlen=100),
            'reward': deque(maxlen=100),
            'loss': deque(maxlen=100),
        }
        self.start_time = time.time()
        self.episode_count = 0
        self.total_steps = 0
        
        self.init_ui()
        self.setup_update_timer()
        
    def init_ui(self):
        """Initialize the dashboard UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QWidget { background-color: #2d2d30; color: #cccccc; }
            QGroupBox {
                border: 2px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 2px;
            }
        """)
        
        # Title
        title = QLabel("Performance Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # System Performance
        system_group = QGroupBox("System Performance")
        system_layout = QGridLayout()
        
        # FPS
        system_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_bar = QProgressBar()
        self.fps_bar.setRange(0, 120)
        system_layout.addWidget(self.fps_bar, 0, 1)
        self.fps_label = QLabel("0")
        system_layout.addWidget(self.fps_label, 0, 2)
        
        # CPU
        system_layout.addWidget(QLabel("CPU:"), 1, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        system_layout.addWidget(self.cpu_bar, 1, 1)
        self.cpu_label = QLabel("0%")
        system_layout.addWidget(self.cpu_label, 1, 2)
        
        # Memory
        system_layout.addWidget(QLabel("Memory:"), 2, 0)
        self.memory_bar = QProgressBar()
        self.memory_bar.setRange(0, 100)
        system_layout.addWidget(self.memory_bar, 2, 1)
        self.memory_label = QLabel("0 MB")
        system_layout.addWidget(self.memory_label, 2, 2)
        
        # GPU
        system_layout.addWidget(QLabel("GPU:"), 3, 0)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, 100)
        system_layout.addWidget(self.gpu_bar, 3, 1)
        self.gpu_label = QLabel("0%")
        system_layout.addWidget(self.gpu_label, 3, 2)
        
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        # Training Performance
        training_group = QGroupBox("Training Performance")
        training_layout = QGridLayout()
        
        # Episodes
        training_layout.addWidget(QLabel("Episodes:"), 0, 0)
        self.episodes_label = QLabel("0")
        self.episodes_label.setStyleSheet("font-weight: bold;")
        training_layout.addWidget(self.episodes_label, 0, 1)
        
        # Total Steps
        training_layout.addWidget(QLabel("Total Steps:"), 1, 0)
        self.steps_label = QLabel("0")
        self.steps_label.setStyleSheet("font-weight: bold;")
        training_layout.addWidget(self.steps_label, 1, 1)
        
        # Average Reward
        training_layout.addWidget(QLabel("Avg Reward:"), 2, 0)
        self.reward_label = QLabel("0.0")
        self.reward_label.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        training_layout.addWidget(self.reward_label, 2, 1)
        
        # Current Loss
        training_layout.addWidget(QLabel("Loss:"), 3, 0)
        self.loss_label = QLabel("0.0")
        self.loss_label.setStyleSheet("font-weight: bold; color: #f14c4c;")
        training_layout.addWidget(self.loss_label, 3, 1)
        
        # Learning Rate
        training_layout.addWidget(QLabel("Learning Rate:"), 4, 0)
        self.lr_label = QLabel("0.001")
        training_layout.addWidget(self.lr_label, 4, 1)
        
        # Epsilon
        training_layout.addWidget(QLabel("Epsilon:"), 5, 0)
        self.epsilon_label = QLabel("1.0")
        training_layout.addWidget(self.epsilon_label, 5, 1)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # Session Statistics
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QGridLayout()
        
        # Uptime
        stats_layout.addWidget(QLabel("Uptime:"), 0, 0)
        self.uptime_label = QLabel("00:00:00")
        stats_layout.addWidget(self.uptime_label, 0, 1)
        
        # Best Reward
        stats_layout.addWidget(QLabel("Best Reward:"), 1, 0)
        self.best_reward_label = QLabel("0.0")
        self.best_reward_label.setStyleSheet("color: #4ec9b0;")
        stats_layout.addWidget(self.best_reward_label, 1, 1)
        
        # Success Rate
        stats_layout.addWidget(QLabel("Success Rate:"), 2, 0)
        self.success_rate_label = QLabel("0%")
        stats_layout.addWidget(self.success_rate_label, 2, 1)
        
        # Avg Episode Length
        stats_layout.addWidget(QLabel("Avg Episode Length:"), 3, 0)
        self.avg_length_label = QLabel("0")
        stats_layout.addWidget(self.avg_length_label, 3, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Charts placeholder
        charts_group = QGroupBox("Performance Trends")
        charts_layout = QVBoxLayout()
        
        # Simple text representation of trends
        self.trend_display = QTextEdit()
        self.trend_display.setReadOnly(True)
        self.trend_display.setMaximumHeight(100)
        self.trend_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        charts_layout.addWidget(self.trend_display)
        
        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)
        
        layout.addStretch()
        
    def setup_update_timer(self):
        """Setup timer for periodic updates."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(1000)  # Update every second
        
    def update_metrics(self):
        """Update all dashboard metrics."""
        # Simulate metrics (replace with actual data in production)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            memory_percent = memory_info.percent
        except ImportError:
            # Fallback values if psutil not installed
            cpu_percent = np.random.randint(10, 50)
            memory_mb = np.random.randint(1000, 4000)
            memory_percent = np.random.randint(20, 60)
        
        # Update system metrics
        fps = np.random.randint(30, 60)
        self.fps_bar.setValue(fps)
        self.fps_label.setText(str(fps))
        self.metrics_history['fps'].append(fps)
        
        self.cpu_bar.setValue(int(cpu_percent))
        self.cpu_label.setText(f"{cpu_percent:.1f}%")
        self.metrics_history['cpu'].append(cpu_percent)
        
        self.memory_bar.setValue(int(memory_percent))
        self.memory_label.setText(f"{memory_mb:.0f} MB")
        self.metrics_history['memory'].append(memory_percent)
        
        # GPU (simulated)
        gpu_percent = np.random.randint(20, 80)
        self.gpu_bar.setValue(gpu_percent)
        self.gpu_label.setText(f"{gpu_percent}%")
        self.metrics_history['gpu'].append(gpu_percent)
        
        # Update uptime
        uptime_seconds = int(time.time() - self.start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        self.uptime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update trends
        self.update_trends()
        
    def update_trends(self):
        """Update trend display."""
        trends = []
        
        if len(self.metrics_history['fps']) > 0:
            avg_fps = np.mean(list(self.metrics_history['fps']))
            trends.append(f"FPS: {'━' * int(avg_fps/2)} {avg_fps:.0f}")
        
        if len(self.metrics_history['cpu']) > 0:
            avg_cpu = np.mean(list(self.metrics_history['cpu']))
            trends.append(f"CPU: {'━' * int(avg_cpu/2)} {avg_cpu:.0f}%")
        
        if len(self.metrics_history['reward']) > 0:
            recent_rewards = list(self.metrics_history['reward'])[-10:]
            trend = "↑" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "↓"
            trends.append(f"Reward Trend: {trend}")
        
        self.trend_display.setPlainText("\n".join(trends))
        
    def update_training_metrics(self, metrics: Dict[str, Any]):
        """Update training-specific metrics."""
        if 'episode' in metrics:
            self.episode_count = metrics['episode']
            self.episodes_label.setText(str(self.episode_count))
        
        if 'steps' in metrics:
            self.total_steps = metrics['steps']
            self.steps_label.setText(str(self.total_steps))
        
        if 'reward' in metrics:
            reward = metrics['reward']
            self.metrics_history['reward'].append(reward)
            avg_reward = np.mean(list(self.metrics_history['reward']))
            self.reward_label.setText(f"{avg_reward:.2f}")
            
            # Update best reward
            if hasattr(self, 'best_reward'):
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_reward_label.setText(f"{reward:.2f}")
            else:
                self.best_reward = reward
                self.best_reward_label.setText(f"{reward:.2f}")
        
        if 'loss' in metrics:
            loss = metrics['loss']
            self.metrics_history['loss'].append(loss)
            self.loss_label.setText(f"{loss:.4f}")
        
        if 'epsilon' in metrics:
            self.epsilon_label.setText(f"{metrics['epsilon']:.3f}")
        
        if 'learning_rate' in metrics:
            self.lr_label.setText(f"{metrics['learning_rate']:.4f}")
        
        if 'success_rate' in metrics:
            self.success_rate_label.setText(f"{metrics['success_rate']:.1f}%")
        
        if 'avg_episode_length' in metrics:
            self.avg_length_label.setText(str(metrics['avg_episode_length']))


class VisualDebugger(Dashboard):
    """Alias for Dashboard for compatibility."""
    pass