"""
Training Dashboard for Real-time Monitoring

Provides a comprehensive dashboard for monitoring training progress
with real-time updates and visualizations.
"""

import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import structlog
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = structlog.get_logger()


class TrainingDashboard:
    """
    Real-time training dashboard with live plotting and monitoring.
    
    Features:
    - Live metric plotting
    - Multi-experiment comparison
    - Performance monitoring
    - Training progress tracking
    - Alert system for anomalies
    """
    
    def __init__(self,
                 metrics_tracker = None,
                 tensorboard_logger = None,
                 update_interval: float = 1.0,
                 plot_window: int = 1000):
        """
        Initialize training dashboard.
        
        Args:
            metrics_tracker: MetricsTracker instance
            tensorboard_logger: TensorBoardLogger instance
            update_interval: Seconds between dashboard updates
            plot_window: Number of points to show in plots
        """
        self.metrics_tracker = metrics_tracker
        self.tb_logger = tensorboard_logger
        self.update_interval = update_interval
        self.plot_window = plot_window
        
        # Dashboard state
        self.is_running = False
        self.update_thread = None
        
        # Plotting data
        self.plot_data: Dict[str, List[float]] = {}
        self.plot_steps: Dict[str, List[int]] = {}
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds: Dict[str, Tuple[float, float]] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        logger.info("Training dashboard initialized")
    
    def start(self):
        """Start the dashboard update loop."""
        if self.is_running:
            logger.warning("Dashboard already running")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Training dashboard started")
    
    def stop(self):
        """Stop the dashboard."""
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("Training dashboard stopped")
    
    def _update_loop(self):
        """Main update loop for the dashboard."""
        while self.is_running:
            try:
                self.update()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
    
    def update(self):
        """Update dashboard with latest metrics."""
        if not self.metrics_tracker:
            return
        
        # Get latest metrics
        summaries = self.metrics_tracker.get_all_summaries()
        
        # Update plot data
        for metric_name, summary in summaries.items():
            if metric_name not in self.plot_data:
                self.plot_data[metric_name] = []
                self.plot_steps[metric_name] = []
            
            # Add latest value
            if 'last' in summary:
                self.plot_data[metric_name].append(summary['last'])
                self.plot_steps[metric_name].append(self.metrics_tracker.global_step)
                
                # Trim to window size
                if len(self.plot_data[metric_name]) > self.plot_window:
                    self.plot_data[metric_name] = self.plot_data[metric_name][-self.plot_window:]
                    self.plot_steps[metric_name] = self.plot_steps[metric_name][-self.plot_window:]
        
        # Check for alerts
        self._check_alerts(summaries)
        
        # Generate and log dashboard image
        if HAS_MATPLOTLIB and self.tb_logger:
            dashboard_image = self.create_dashboard_image()
            if dashboard_image is not None:
                self.tb_logger.log_image("dashboard/overview", dashboard_image,
                                        self.metrics_tracker.global_step)
        
        self.last_update_time = time.time()
    
    def create_dashboard_image(self) -> Optional[np.ndarray]:
        """
        Create dashboard visualization image.
        
        Returns:
            Dashboard image as numpy array
        """
        if not HAS_MATPLOTLIB or not self.plot_data:
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_metric(ax1, 'training/loss', 'Training Loss', 'Loss')
        
        # Plot 2: Reward
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_metric(ax2, 'episode/reward', 'Episode Reward', 'Reward')
        
        # Plot 3: Learning Rate
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_metric(ax3, 'training/learning_rate', 'Learning Rate', 'LR')
        
        # Plot 4: Episode Length
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_metric(ax4, 'episode/length', 'Episode Length', 'Steps')
        
        # Plot 5: Win Rate
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_metric(ax5, 'episode/win_rate', 'Win Rate', 'Rate')
        
        # Plot 6: FPS
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_metric(ax6, 'global/fps/rate', 'FPS', 'FPS')
        
        # Plot 7: Resource Usage
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_resource_usage(ax7)
        
        # Plot 8: Training Progress
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_progress(ax8)
        
        # Plot 9: Alerts
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_alerts(ax9)
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return image
    
    def _plot_metric(self, ax, metric_name: str, title: str, ylabel: str):
        """Plot a single metric."""
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        if metric_name in self.plot_data and self.plot_data[metric_name]:
            steps = self.plot_steps[metric_name]
            values = self.plot_data[metric_name]
            
            # Plot line
            ax.plot(steps, values, 'b-', alpha=0.7)
            
            # Plot moving average
            if len(values) > 10:
                window = min(50, len(values) // 4)
                ma = np.convolve(values, np.ones(window)/window, mode='valid')
                ma_steps = steps[window-1:]
                ax.plot(ma_steps, ma, 'r-', linewidth=2, label=f'MA({window})')
            
            # Add latest value text
            if values:
                ax.text(0.02, 0.98, f'Latest: {values[-1]:.4f}',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
    
    def _plot_resource_usage(self, ax):
        """Plot resource usage."""
        ax.set_title('Resource Usage')
        ax.set_xlabel('Step')
        ax.set_ylabel('Usage %')
        ax.grid(True, alpha=0.3)
        
        # Plot CPU and memory usage
        cpu_metric = 'resources/cpu_percent'
        mem_metric = 'resources/memory_percent'
        
        plotted = False
        for metric, label, color in [(cpu_metric, 'CPU', 'blue'),
                                     (mem_metric, 'Memory', 'green')]:
            if metric in self.plot_data and self.plot_data[metric]:
                steps = self.plot_steps[metric]
                values = self.plot_data[metric]
                ax.plot(steps, values, color=color, label=label, alpha=0.7)
                plotted = True
        
        # Plot GPU if available
        for i in range(4):  # Check up to 4 GPUs
            gpu_metric = f'resources/gpu{i}_percent'
            if gpu_metric in self.plot_data and self.plot_data[gpu_metric]:
                steps = self.plot_steps[gpu_metric]
                values = self.plot_data[gpu_metric]
                ax.plot(steps, values, label=f'GPU{i}', alpha=0.7)
                plotted = True
        
        if plotted:
            ax.legend(loc='upper left')
            ax.set_ylim([0, 100])
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
    
    def _plot_progress(self, ax):
        """Plot training progress."""
        ax.set_title('Training Progress')
        ax.axis('off')
        
        # Calculate progress metrics
        elapsed_time = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        if self.metrics_tracker:
            step = self.metrics_tracker.global_step
            
            # Get episode count
            episode_count = self.metrics_tracker.counters.get('episode/reward', 0)
            
            # Calculate rates
            steps_per_sec = step / max(1, elapsed_time)
            episodes_per_hour = (episode_count / max(1, elapsed_time)) * 3600
            
            # Create progress text
            text_lines = [
                f'Elapsed Time: {elapsed_str}',
                f'Global Step: {step:,}',
                f'Episodes: {episode_count:,}',
                f'Steps/sec: {steps_per_sec:.1f}',
                f'Episodes/hour: {episodes_per_hour:.1f}',
            ]
            
            # Add best scores
            if 'episode/reward' in self.plot_data and self.plot_data['episode/reward']:
                best_reward = max(self.plot_data['episode/reward'])
                text_lines.append(f'Best Reward: {best_reward:.2f}')
            
            # Display text
            y_pos = 0.9
            for line in text_lines:
                ax.text(0.1, y_pos, line, transform=ax.transAxes,
                       fontsize=11, verticalalignment='top')
                y_pos -= 0.15
        else:
            ax.text(0.5, 0.5, 'No tracker connected', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
    
    def _plot_alerts(self, ax):
        """Plot recent alerts."""
        ax.set_title('Alerts')
        ax.axis('off')
        
        if self.alerts:
            # Show last 5 alerts
            recent_alerts = self.alerts[-5:]
            y_pos = 0.9
            
            for alert in recent_alerts:
                # Color based on severity
                color = {'warning': 'orange', 
                        'error': 'red',
                        'info': 'blue'}.get(alert['severity'], 'black')
                
                # Format time
                alert_time = datetime.fromtimestamp(alert['timestamp'])
                time_str = alert_time.strftime('%H:%M:%S')
                
                # Display alert
                text = f"[{time_str}] {alert['message']}"
                ax.text(0.05, y_pos, text, transform=ax.transAxes,
                       fontsize=9, color=color, verticalalignment='top')
                y_pos -= 0.18
        else:
            ax.text(0.5, 0.5, 'No alerts', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
    
    def set_alert_threshold(self, metric_name: str, min_val: float = None,
                           max_val: float = None):
        """
        Set alert thresholds for a metric.
        
        Args:
            metric_name: Name of the metric
            min_val: Minimum threshold (alert if below)
            max_val: Maximum threshold (alert if above)
        """
        self.alert_thresholds[metric_name] = (min_val, max_val)
    
    def _check_alerts(self, summaries: Dict[str, Dict[str, float]]):
        """Check for alert conditions."""
        for metric_name, (min_val, max_val) in self.alert_thresholds.items():
            if metric_name in summaries:
                value = summaries[metric_name].get('last', 0)
                
                if min_val is not None and value < min_val:
                    self.add_alert(f"{metric_name} below threshold: {value:.4f} < {min_val}",
                                 'warning')
                
                if max_val is not None and value > max_val:
                    self.add_alert(f"{metric_name} above threshold: {value:.4f} > {max_val}",
                                 'warning')
    
    def add_alert(self, message: str, severity: str = 'info'):
        """
        Add an alert to the dashboard.
        
        Args:
            message: Alert message
            severity: Alert severity (info, warning, error)
        """
        alert = {
            'message': message,
            'severity': severity,
            'timestamp': time.time()
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log alert
        log_func = {'info': logger.info,
                   'warning': logger.warning,
                   'error': logger.error}.get(severity, logger.info)
        log_func(f"Dashboard alert: {message}")
    
    def generate_report(self) -> str:
        """Generate text report of training progress."""
        lines = ["=" * 60]
        lines.append("TRAINING DASHBOARD REPORT")
        lines.append("=" * 60)
        
        # Time information
        elapsed_time = time.time() - self.start_time
        lines.append(f"Training Duration: {str(timedelta(seconds=int(elapsed_time)))}")
        lines.append(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Metrics summary
        if self.metrics_tracker:
            lines.append("METRICS SUMMARY:")
            lines.append("-" * 40)
            
            summaries = self.metrics_tracker.get_all_summaries()
            
            for metric_name in sorted(summaries.keys()):
                summary = summaries[metric_name]
                lines.append(f"\n{metric_name}:")
                lines.append(f"  Current: {summary.get('last', 0):.4f}")
                lines.append(f"  Mean:    {summary.get('mean', 0):.4f}")
                lines.append(f"  Std:     {summary.get('std', 0):.4f}")
                lines.append(f"  Min:     {summary.get('min', 0):.4f}")
                lines.append(f"  Max:     {summary.get('max', 0):.4f}")
        
        # Recent alerts
        if self.alerts:
            lines.append("\nRECENT ALERTS:")
            lines.append("-" * 40)
            
            for alert in self.alerts[-10:]:
                alert_time = datetime.fromtimestamp(alert['timestamp'])
                lines.append(f"[{alert_time.strftime('%H:%M:%S')}] "
                           f"[{alert['severity'].upper()}] {alert['message']}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def save_dashboard(self, filepath: str):
        """Save dashboard state to file."""
        import pickle
        
        state = {
            'plot_data': self.plot_data,
            'plot_steps': self.plot_steps,
            'alerts': self.alerts,
            'alert_thresholds': self.alert_thresholds,
            'start_time': self.start_time
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Dashboard saved to {filepath}")