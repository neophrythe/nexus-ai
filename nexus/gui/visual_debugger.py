"""
Visual Debugger Window - Advanced game state visualization.
Similar to SerpentAI's visual debugger but with more features.
"""

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

import numpy as np
import cv2
from collections import deque


class VisualDebugger:
    """Main Visual Debugger class - delegates to appropriate implementation"""
    
    def __init__(self, config=None):
        self.config = config or {}
        if PYQT_AVAILABLE:
            self.window = VisualDebuggerWindow(config)
        else:
            raise ImportError("PyQt5 not available. Install with: pip install PyQt5")
    
    def show(self):
        if hasattr(self.window, 'show'):
            self.window.show()
    
    def run(self):
        if hasattr(self.window, 'run'):
            self.window.run()


class VisualDebuggerWindow(QMainWindow if PYQT_AVAILABLE else object):
    """Advanced visual debugger for game analysis."""
    
    def __init__(self):
        super().__init__()
        self.frame_buffer = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.detection_history = deque(maxlen=100)
        self.current_frame = None
        self.is_paused = False
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the debugger UI."""
        self.setWindowTitle("Nexus Visual Debugger")
        self.setGeometry(150, 150, 1400, 800)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #2d2d30; color: #cccccc; }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QGroupBox {
                border: 2px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
        """)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)
        
        # Left panel - Frame Analysis
        left_panel = self.create_frame_panel()
        layout.addWidget(left_panel, 2)
        
        # Right panel - Detection & Analysis
        right_panel = self.create_analysis_panel()
        layout.addWidget(right_panel, 1)
        
        # Create toolbar
        self.create_toolbar()
    
    def create_toolbar(self):
        """Create the toolbar."""
        toolbar = self.addToolBar('Debug Tools')
        
        # Play/Pause
        self.play_action = QAction('▶ Play', self)
        self.play_action.triggered.connect(self.toggle_playback)
        toolbar.addAction(self.play_action)
        
        # Step forward
        step_forward = QAction('⏭ Step', self)
        step_forward.triggered.connect(self.step_forward)
        toolbar.addAction(step_forward)
        
        # Step backward  
        step_back = QAction('⏮ Back', self)
        step_back.triggered.connect(self.step_backward)
        toolbar.addAction(step_back)
        
        toolbar.addSeparator()
        
        # Detection toggles
        self.show_bbox_action = QAction('📦 Bounding Boxes', self)
        self.show_bbox_action.setCheckable(True)
        self.show_bbox_action.setChecked(True)
        toolbar.addAction(self.show_bbox_action)
        
        self.show_mask_action = QAction('🎭 Masks', self)
        self.show_mask_action.setCheckable(True)
        toolbar.addAction(self.show_mask_action)
        
        self.show_keypoints_action = QAction('📍 Keypoints', self)
        self.show_keypoints_action.setCheckable(True)
        toolbar.addAction(self.show_keypoints_action)
        
        self.show_trajectory_action = QAction('📈 Trajectory', self)
        self.show_trajectory_action.setCheckable(True)
        toolbar.addAction(self.show_trajectory_action)
        
        toolbar.addSeparator()
        
        # Analysis tools
        measure_tool = QAction('📏 Measure', self)
        measure_tool.triggered.connect(self.activate_measure)
        toolbar.addAction(measure_tool)
        
        roi_tool = QAction('🔲 ROI', self)
        roi_tool.triggered.connect(self.activate_roi)
        toolbar.addAction(roi_tool)
        
        export_action = QAction('💾 Export', self)
        export_action.triggered.connect(self.export_frame)
        toolbar.addAction(export_action)
    
    def create_frame_panel(self):
        """Create the frame analysis panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Tab widget for different views
        self.view_tabs = QTabWidget()
        
        # Original frame
        self.original_view = QLabel()
        self.original_view.setMinimumSize(600, 400)
        self.original_view.setStyleSheet("background-color: #000000;")
        self.original_view.setScaledContents(True)
        self.view_tabs.addTab(self.original_view, "Original")
        
        # Processed frame
        self.processed_view = QLabel()
        self.processed_view.setMinimumSize(600, 400)
        self.processed_view.setStyleSheet("background-color: #000000;")
        self.processed_view.setScaledContents(True)
        self.view_tabs.addTab(self.processed_view, "Processed")
        
        # Heatmap view
        self.heatmap_view = QLabel()
        self.heatmap_view.setMinimumSize(600, 400)
        self.heatmap_view.setStyleSheet("background-color: #000000;")
        self.heatmap_view.setScaledContents(True)
        self.view_tabs.addTab(self.heatmap_view, "Heatmap")
        
        # Edge detection
        self.edge_view = QLabel()
        self.edge_view.setMinimumSize(600, 400)
        self.edge_view.setStyleSheet("background-color: #000000;")
        self.edge_view.setScaledContents(True)
        self.view_tabs.addTab(self.edge_view, "Edges")
        
        layout.addWidget(self.view_tabs)
        
        # Frame timeline
        timeline_group = QGroupBox("Timeline")
        timeline_layout = QVBoxLayout()
        
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(299)
        self.timeline_slider.valueChanged.connect(self.seek_frame)
        timeline_layout.addWidget(self.timeline_slider)
        
        # Frame info
        frame_info_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0/0")
        frame_info_layout.addWidget(self.frame_label)
        
        self.time_label = QLabel("Time: 00:00.000")
        frame_info_layout.addWidget(self.time_label)
        
        self.fps_label = QLabel("FPS: 0")
        frame_info_layout.addWidget(self.fps_label)
        
        frame_info_layout.addStretch()
        timeline_layout.addLayout(frame_info_layout)
        
        timeline_group.setLayout(timeline_layout)
        layout.addWidget(timeline_group)
        
        return panel
    
    def create_analysis_panel(self):
        """Create the analysis panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Detection results
        detection_group = QGroupBox("Detections")
        detection_layout = QVBoxLayout()
        
        self.detection_tree = QTreeWidget()
        self.detection_tree.setHeaderLabels(["Object", "Confidence", "Position"])
        detection_layout.addWidget(self.detection_tree)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Frame analysis
        analysis_group = QGroupBox("Frame Analysis")
        analysis_layout = QFormLayout()
        
        self.brightness_label = QLabel("0")
        analysis_layout.addRow("Brightness:", self.brightness_label)
        
        self.contrast_label = QLabel("0")
        analysis_layout.addRow("Contrast:", self.contrast_label)
        
        self.sharpness_label = QLabel("0")
        analysis_layout.addRow("Sharpness:", self.sharpness_label)
        
        self.motion_label = QLabel("0")
        analysis_layout.addRow("Motion:", self.motion_label)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Color analysis
        color_group = QGroupBox("Color Analysis")
        color_layout = QVBoxLayout()
        
        # Histogram display widget
        self.histogram_widget = QWidget()
        self.histogram_widget.setMinimumHeight(150)
        self.histogram_widget.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3c3c3c;")
        
        # Create histogram bars (simple visualization)
        histogram_layout = QHBoxLayout()
        self.histogram_widget.setLayout(histogram_layout)
        
        # RGB channel bars
        self.r_bar = QProgressBar()
        self.r_bar.setOrientation(Qt.Vertical)
        self.r_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff0000; }")
        histogram_layout.addWidget(self.r_bar)
        
        self.g_bar = QProgressBar()
        self.g_bar.setOrientation(Qt.Vertical)
        self.g_bar.setStyleSheet("QProgressBar::chunk { background-color: #00ff00; }")
        histogram_layout.addWidget(self.g_bar)
        
        self.b_bar = QProgressBar()
        self.b_bar.setOrientation(Qt.Vertical)
        self.b_bar.setStyleSheet("QProgressBar::chunk { background-color: #0000ff; }")
        histogram_layout.addWidget(self.b_bar)
        
        color_layout.addWidget(QLabel("Color Histogram:"))
        color_layout.addWidget(self.histogram_widget)
        
        # Dominant colors
        self.colors_layout = QHBoxLayout()
        for i in range(5):
            color_label = QLabel()
            color_label.setFixedSize(40, 40)
            color_label.setStyleSheet(f"background-color: #333333;")
            self.colors_layout.addWidget(color_label)
        color_layout.addLayout(self.colors_layout)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # Console
        console_group = QGroupBox("Debug Console")
        console_layout = QVBoxLayout()
        
        self.debug_console = QTextEdit()
        self.debug_console.setReadOnly(True)
        self.debug_console.setMaximumHeight(150)
        self.debug_console.setStyleSheet("font-family: monospace; font-size: 10px;")
        console_layout.addWidget(self.debug_console)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        layout.addStretch()
        
        return panel
    
    def process_frame(self, frame):
        """Process and analyze a frame."""
        if frame is None:
            return
        
        self.current_frame = frame
        self.frame_buffer.append(frame)
        
        # Update original view
        self.update_view(self.original_view, frame)
        
        # Generate processed views
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.update_view(self.edge_view, edge_colored)
        
        # Heatmap (motion or attention)
        heatmap = self.generate_heatmap(frame)
        self.update_view(self.heatmap_view, heatmap)
        
        # Processed with detections
        processed = frame.copy()
        if self.show_bbox_action.isChecked():
            processed = self.draw_detections(processed)
        self.update_view(self.processed_view, processed)
        
        # Update analysis
        self.analyze_frame(frame)
        
    def update_view(self, view_widget, frame):
        """Update a view widget with a frame."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, 
                        bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(view_widget.size(), Qt.KeepAspectRatio, 
                              Qt.SmoothTransformation)
        view_widget.setPixmap(scaled)
    
    def generate_heatmap(self, frame):
        """Generate a heatmap visualization."""
        # Simple motion heatmap simulation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return heatmap
    
    def draw_detections(self, frame):
        """Draw detection overlays on frame."""
        # Simulate some detections
        detections = [
            {"class": "Player", "bbox": [100, 100, 200, 300], "conf": 0.95},
            {"class": "Enemy", "bbox": [400, 150, 480, 280], "conf": 0.87},
            {"class": "Item", "bbox": [300, 350, 340, 390], "conf": 0.72}
        ]
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0) if det["class"] == "Player" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class']}: {det['conf']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def analyze_frame(self, frame):
        """Analyze frame properties."""
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.brightness_label.setText(f"{brightness:.1f}")
        
        # Calculate contrast
        contrast = np.std(gray)
        self.contrast_label.setText(f"{contrast:.1f}")
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        self.sharpness_label.setText(f"{sharpness:.1f}")
        
        # Update detections tree
        self.update_detections()
        
        # Update color histogram
        self.update_histogram(frame)
        
        # Update dominant colors
        self.update_dominant_colors(frame)
        
        # Log to console
        self.debug_console.append(f"Frame analyzed - B:{brightness:.1f} C:{contrast:.1f} S:{sharpness:.1f}")
    
    def update_detections(self):
        """Update detection results tree."""
        self.detection_tree.clear()
        
        # Simulate detection results
        detections = [
            ("Player", "0.95", "(100, 100)"),
            ("Enemy", "0.87", "(400, 150)"),
            ("Item", "0.72", "(300, 350)")
        ]
        
        for class_name, conf, pos in detections:
            item = QTreeWidgetItem([class_name, conf, pos])
            self.detection_tree.addTopLevelItem(item)
    
    def toggle_playback(self):
        """Toggle playback pause/resume."""
        self.is_paused = not self.is_paused
        self.play_action.setText('⏸ Pause' if not self.is_paused else '▶ Play')
    
    def step_forward(self):
        """Step forward one frame."""
        current = self.timeline_slider.value()
        if current < self.timeline_slider.maximum():
            self.timeline_slider.setValue(current + 1)
    
    def step_backward(self):
        """Step backward one frame."""
        current = self.timeline_slider.value()
        if current > 0:
            self.timeline_slider.setValue(current - 1)
    
    def seek_frame(self, value):
        """Seek to specific frame."""
        if 0 <= value < len(self.frame_buffer):
            frame = self.frame_buffer[value]
            self.process_frame(frame)
            self.frame_label.setText(f"Frame: {value}/{len(self.frame_buffer)}")
            
            # Update time
            time_sec = value / 60.0  # Assuming 60 FPS
            time_str = f"{int(time_sec//60):02d}:{int(time_sec%60):02d}.{int((time_sec%1)*1000):03d}"
            self.time_label.setText(f"Time: {time_str}")
    
    def activate_measure(self):
        """Activate measurement tool."""
        self.debug_console.append("Measurement tool activated - Click two points to measure")
    
    def activate_roi(self):
        """Activate ROI selection tool."""
        self.debug_console.append("ROI tool activated - Draw rectangle to select region")
    
    def export_frame(self):
        """Export current frame."""
        if self.current_frame is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Export Frame", "", 
                                                       "PNG (*.png);;JPEG (*.jpg)")
            if file_name:
                cv2.imwrite(file_name, self.current_frame)
                self.debug_console.append(f"Frame exported to {file_name}")
    
    def update_histogram(self, frame):
        """Update the color histogram display."""
        if frame is None:
            return
        
        # Calculate histogram for each channel
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        
        # Normalize to 0-100 for progress bars
        r_val = int(np.mean(hist_r) / np.max(hist_r) * 100) if np.max(hist_r) > 0 else 0
        g_val = int(np.mean(hist_g) / np.max(hist_g) * 100) if np.max(hist_g) > 0 else 0
        b_val = int(np.mean(hist_b) / np.max(hist_b) * 100) if np.max(hist_b) > 0 else 0
        
        # Update progress bars
        self.r_bar.setValue(r_val)
        self.g_bar.setValue(g_val)
        self.b_bar.setValue(b_val)
    
    def update_dominant_colors(self, frame):
        """Update the dominant colors display."""
        if frame is None:
            return
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (50, 50))
        
        # Reshape to list of pixels
        pixels = small_frame.reshape((-1, 3))
        
        # Use K-means to find dominant colors
        from sklearn.cluster import KMeans
        import warnings
        warnings.filterwarnings('ignore')
        
        try:
            # Find 5 dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Update color display
            for i, color in enumerate(colors):
                if i < len(self.colors_layout.children()):
                    color_widget = self.colors_layout.itemAt(i).widget()
                    if color_widget:
                        # Convert BGR to RGB for display
                        r, g, b = color[2], color[1], color[0]
                        color_widget.setStyleSheet(f"background-color: rgb({r},{g},{b});")
        except Exception:
            # sklearn not installed or error in clustering
            # Use simple color sampling instead
            for i in range(5):
                y = int(frame.shape[0] * (i + 1) / 6)
                x = int(frame.shape[1] / 2)
                if y < frame.shape[0] and x < frame.shape[1]:
                    b, g, r = frame[y, x]
                    if i < len(self.colors_layout.children()):
                        color_widget = self.colors_layout.itemAt(i).widget()
                        if color_widget:
                            color_widget.setStyleSheet(f"background-color: rgb({r},{g},{b});")