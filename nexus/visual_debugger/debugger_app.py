"""Visual Debugger Application - SerpentAI Compatible

Standalone visual debugging application with game overlay capabilities.
"""

import sys
import time
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import cv2
import structlog

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                                QVBoxLayout, QHBoxLayout, QPushButton,
                                QLabel, QTextEdit, QTreeWidget, QTreeWidgetItem,
                                QSplitter, QTabWidget, QMenuBar, QMenu,
                                QAction, QFileDialog, QMessageBox)
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
    HAS_QT = True
except ImportError:
    HAS_QT = False

logger = structlog.get_logger()


@dataclass
class DebugOverlay:
    """Debug overlay configuration"""
    enabled: bool = True
    show_fps: bool = True
    show_regions: bool = True
    show_detections: bool = True
    show_inputs: bool = True
    show_metrics: bool = True
    show_grid: bool = False
    grid_size: int = 32
    alpha: float = 0.7
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'fps': (0, 255, 0),
        'region': (255, 255, 0),
        'detection': (0, 255, 255),
        'input': (255, 0, 255),
        'metric': (255, 255, 255),
        'grid': (128, 128, 128)
    })


class FrameViewer(QWidget if HAS_QT else object):
    """Frame viewer widget for Qt"""
    
    def __init__(self, parent=None):
        super().__init__(parent) if HAS_QT else None
        
        if HAS_QT:
            self.image_label = QLabel()
            self.image_label.setScaledContents(True)
            self.image_label.setMinimumSize(640, 480)
            
            layout = QVBoxLayout()
            layout.addWidget(self.image_label)
            self.setLayout(layout)
            
            self.current_frame = None
            self.overlay_config = DebugOverlay()
    
    def update_frame(self, frame: np.ndarray, overlays: Optional[Dict] = None):
        """Update displayed frame with overlays"""
        if not HAS_QT:
            return
        
        self.current_frame = frame.copy()
        
        # Apply overlays
        if overlays and self.overlay_config.enabled:
            frame = self._apply_overlays(frame, overlays)
        
        # Convert to QImage
        height, width = frame.shape[:2]
        if len(frame.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, 
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = width
            q_image = QImage(frame.data, width, height,
                           bytes_per_line, QImage.Format_Grayscale8)
        
        # Display
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
    
    def _apply_overlays(self, frame: np.ndarray, overlays: Dict) -> np.ndarray:
        """Apply debug overlays to frame"""
        frame = frame.copy()
        
        # FPS overlay
        if self.overlay_config.show_fps and 'fps' in overlays:
            fps = overlays['fps']
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       self.overlay_config.colors['fps'], 2)
        
        # Region overlays
        if self.overlay_config.show_regions and 'regions' in overlays:
            for region in overlays['regions']:
                x, y, w, h = region['bbox']
                label = region.get('label', '')
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                            self.overlay_config.colors['region'], 2)
                if label:
                    cv2.putText(frame, label, (x, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              self.overlay_config.colors['region'], 1)
        
        # Detection overlays
        if self.overlay_config.show_detections and 'detections' in overlays:
            for detection in overlays['detections']:
                x, y, w, h = detection['bbox']
                label = detection.get('label', '')
                confidence = detection.get('confidence', 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                            self.overlay_config.colors['detection'], 2)
                
                text = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"
                cv2.putText(frame, text, (x, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          self.overlay_config.colors['detection'], 1)
        
        # Input overlays
        if self.overlay_config.show_inputs and 'inputs' in overlays:
            y_offset = 60
            for input_event in overlays['inputs'][-5:]:  # Show last 5 inputs
                text = f"{input_event['type']}: {input_event['data']}"
                cv2.putText(frame, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          self.overlay_config.colors['input'], 1)
                y_offset += 20
        
        # Metrics overlay
        if self.overlay_config.show_metrics and 'metrics' in overlays:
            y_offset = frame.shape[0] - 100
            for key, value in overlays['metrics'].items():
                text = f"{key}: {value}"
                cv2.putText(frame, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          self.overlay_config.colors['metric'], 1)
                y_offset += 20
        
        # Grid overlay
        if self.overlay_config.show_grid:
            grid_size = self.overlay_config.grid_size
            h, w = frame.shape[:2]
            
            # Vertical lines
            for x in range(0, w, grid_size):
                cv2.line(frame, (x, 0), (x, h),
                        self.overlay_config.colors['grid'], 1)
            
            # Horizontal lines
            for y in range(0, h, grid_size):
                cv2.line(frame, (0, y), (w, y),
                        self.overlay_config.colors['grid'], 1)
        
        return frame


class VisualDebuggerApp(QMainWindow if HAS_QT else object):
    """Main visual debugger application"""
    
    def __init__(self):
        super().__init__() if HAS_QT else None
        
        if not HAS_QT:
            logger.error("Qt not available, cannot create GUI")
            return
        
        self.setWindowTitle("Nexus Visual Debugger - SerpentAI Compatible")
        self.setGeometry(100, 100, 1400, 900)
        
        # State
        self.is_connected = False
        self.is_recording = False
        self.current_game = None
        self.current_agent = None
        
        # Components
        self.frame_viewer = None
        self.metrics_tree = None
        self.log_viewer = None
        self.control_panel = None
        
        # Data
        self.frame_buffer = []
        self.metrics_history = []
        self.detection_history = []
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_connections()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(33)  # ~30 FPS
    
    def _setup_ui(self):
        """Setup UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Control panel
        self.control_panel = self._create_control_panel()
        main_layout.addWidget(self.control_panel)
        
        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Frame viewer
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        self.frame_viewer = FrameViewer()
        left_layout.addWidget(QLabel("Frame Viewer"))
        left_layout.addWidget(self.frame_viewer)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Tabs
        right_panel = QTabWidget()
        splitter.addWidget(right_panel)
        
        # Metrics tab
        self.metrics_tree = QTreeWidget()
        self.metrics_tree.setHeaderLabels(["Metric", "Value"])
        right_panel.addTab(self.metrics_tree, "Metrics")
        
        # Detections tab
        self.detections_tree = QTreeWidget()
        self.detections_tree.setHeaderLabels(["Type", "Label", "Confidence", "Location"])
        right_panel.addTab(self.detections_tree, "Detections")
        
        # Regions tab
        self.regions_tree = QTreeWidget()
        self.regions_tree.setHeaderLabels(["Name", "Location", "Size"])
        right_panel.addTab(self.regions_tree, "Regions")
        
        # Log tab
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        right_panel.addTab(self.log_viewer, "Logs")
        
        # Set splitter sizes
        splitter.setSizes([900, 500])
    
    def _create_control_panel(self) -> QWidget:
        """Create control panel"""
        panel = QWidget()
        layout = QHBoxLayout()
        panel.setLayout(layout)
        
        # Connection controls
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._toggle_connection)
        layout.addWidget(self.connect_btn)
        
        self.game_label = QLabel("Game: Not connected")
        layout.addWidget(self.game_label)
        
        self.agent_label = QLabel("Agent: Not connected")
        layout.addWidget(self.agent_label)
        
        layout.addStretch()
        
        # Recording controls
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setEnabled(False)
        layout.addWidget(self.record_btn)
        
        # Overlay controls
        self.overlay_btn = QPushButton("Configure Overlays")
        self.overlay_btn.clicked.connect(self._configure_overlays)
        layout.addWidget(self.overlay_btn)
        
        # Capture controls
        self.capture_btn = QPushButton("Capture Frame")
        self.capture_btn.clicked.connect(self._capture_frame)
        self.capture_btn.setEnabled(False)
        layout.addWidget(self.capture_btn)
        
        return panel
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Recording", self)
        open_action.triggered.connect(self._open_recording)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Frame", self)
        save_action.triggered.connect(self._save_frame)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        fps_action = QAction("Show FPS", self)
        fps_action.setCheckable(True)
        fps_action.setChecked(True)
        fps_action.triggered.connect(lambda checked: self._toggle_overlay('fps', checked))
        view_menu.addAction(fps_action)
        
        grid_action = QAction("Show Grid", self)
        grid_action.setCheckable(True)
        grid_action.triggered.connect(lambda checked: self._toggle_overlay('grid', checked))
        view_menu.addAction(grid_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        region_action = QAction("Define Region", self)
        region_action.triggered.connect(self._define_region)
        tools_menu.addAction(region_action)
        
        calibrate_action = QAction("Calibrate", self)
        calibrate_action.triggered.connect(self._calibrate)
        tools_menu.addAction(calibrate_action)
    
    def _setup_connections(self):
        """Setup WebSocket/API connections"""
        # This would connect to the Nexus API/WebSocket server
        pass
    
    def _toggle_connection(self):
        """Toggle connection to game/agent"""
        if not self.is_connected:
            # Connect to Nexus
            self._connect_to_nexus()
        else:
            # Disconnect
            self._disconnect_from_nexus()
    
    def _connect_to_nexus(self):
        """Connect to Nexus framework"""
        try:
            # TODO: Implement WebSocket connection to Nexus
            self.is_connected = True
            self.connect_btn.setText("Disconnect")
            self.game_label.setText("Game: Connected")
            self.agent_label.setText("Agent: Active")
            self.record_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            
            self._log("Connected to Nexus framework")
            
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self._log(f"Connection failed: {e}")
    
    def _disconnect_from_nexus(self):
        """Disconnect from Nexus framework"""
        self.is_connected = False
        self.connect_btn.setText("Connect")
        self.game_label.setText("Game: Not connected")
        self.agent_label.setText("Agent: Not connected")
        self.record_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        
        self._log("Disconnected from Nexus framework")
    
    def _toggle_recording(self):
        """Toggle recording"""
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.setText("Stop Recording")
            self._log("Recording started")
        else:
            self.is_recording = False
            self.record_btn.setText("Start Recording")
            self._log("Recording stopped")
    
    def _configure_overlays(self):
        """Open overlay configuration dialog"""
        # TODO: Implement overlay configuration dialog
        pass
    
    def _capture_frame(self):
        """Capture current frame"""
        if self.frame_viewer.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"
            cv2.imwrite(filename, self.frame_viewer.current_frame)
            self._log(f"Frame captured: {filename}")
    
    def _open_recording(self):
        """Open a recording file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Recording", "",
            "HDF5 Files (*.h5);;Pickle Files (*.pkl);;All Files (*.*)"
        )
        
        if filename:
            # TODO: Load and play recording
            self._log(f"Opened recording: {filename}")
    
    def _save_frame(self):
        """Save current frame"""
        if self.frame_viewer.current_frame is not None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Frame", "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
            )
            
            if filename:
                cv2.imwrite(filename, self.frame_viewer.current_frame)
                self._log(f"Frame saved: {filename}")
    
    def _toggle_overlay(self, overlay_type: str, enabled: bool):
        """Toggle overlay visibility"""
        if overlay_type == 'fps':
            self.frame_viewer.overlay_config.show_fps = enabled
        elif overlay_type == 'grid':
            self.frame_viewer.overlay_config.show_grid = enabled
    
    def _define_region(self):
        """Define a region of interest"""
        # TODO: Implement region selection tool
        pass
    
    def _calibrate(self):
        """Open calibration tool"""
        # TODO: Implement calibration tool
        pass
    
    def _update_display(self):
        """Update display with latest data"""
        if not self.is_connected:
            return
        
        # TODO: Get latest frame and data from Nexus
        # For now, generate test data
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        overlays = {
            'fps': 30.0,
            'regions': [],
            'detections': [],
            'inputs': [],
            'metrics': {
                'frames': 1000,
                'reward': 42.5
            }
        }
        
        # Update frame viewer
        self.frame_viewer.update_frame(test_frame, overlays)
        
        # Update metrics
        self._update_metrics(overlays.get('metrics', {}))
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics display"""
        self.metrics_tree.clear()
        
        for key, value in metrics.items():
            item = QTreeWidgetItem([str(key), str(value)])
            self.metrics_tree.addTopLevelItem(item)
    
    def _log(self, message: str):
        """Add message to log viewer"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_viewer.append(f"[{timestamp}] {message}")


class VisualDebuggerCLI:
    """CLI interface for visual debugger"""
    
    def __init__(self):
        self.app = None
        self.window = None
    
    def run(self):
        """Run the visual debugger"""
        if HAS_QT:
            self.app = QApplication(sys.argv)
            self.window = VisualDebuggerApp()
            self.window.show()
            sys.exit(self.app.exec_())
        else:
            logger.error("Qt not available. Install PyQt5: pip install PyQt5")
            print("Visual debugger requires PyQt5. Install with: pip install PyQt5")
            sys.exit(1)


def main():
    """Main entry point"""
    debugger = VisualDebuggerCLI()
    debugger.run()


if __name__ == "__main__":
    main()