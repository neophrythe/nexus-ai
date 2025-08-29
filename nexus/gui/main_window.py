"""
Nexus AI Framework - Main GUI Application
Complete visual interface similar to SerpentAI but with modern features.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Fallback to tkinter
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

import numpy as np
import cv2


class NexusGUI(QMainWindow if PYQT_AVAILABLE else object):
    """Main GUI window for Nexus AI Framework."""
    
    def __init__(self):
        super().__init__()
        self.current_game = None
        self.current_agent = None
        self.is_recording = False
        self.is_training = False
        self.frame_count = 0
        self.fps = 0
        self.last_frame = None
        
        self.init_ui()
        self.setup_timers()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Nexus AI Framework - Visual Interface")
        self.setGeometry(100, 100, 1600, 900)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2d2d30;
                color: #cccccc;
                font-family: 'Segoe UI', Consolas, monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #666666;
            }
            QGroupBox {
                border: 2px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
            QLineEdit, QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3c3c3c;
                padding: 5px;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #2d2d30;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #007acc;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #3c3c3c;
                selection-background-color: #094771;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                padding: 5px;
                border: none;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Control Panel
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Center panel - Game View
        center_panel = self.create_game_view()
        main_layout.addWidget(center_panel, 2)
        
        # Right panel - Information Panel
        right_panel = self.create_info_panel()
        main_layout.addWidget(right_panel, 1)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
        
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_project = QAction('New Project', self)
        new_project.setShortcut('Ctrl+N')
        new_project.triggered.connect(self.new_project)
        file_menu.addAction(new_project)
        
        open_project = QAction('Open Project', self)
        open_project.setShortcut('Ctrl+O')
        open_project.triggered.connect(self.open_project)
        file_menu.addAction(open_project)
        
        save_project = QAction('Save Project', self)
        save_project.setShortcut('Ctrl+S')
        save_project.triggered.connect(self.save_project)
        file_menu.addAction(save_project)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Game menu
        game_menu = menubar.addMenu('Game')
        
        register_game = QAction('Register Game', self)
        register_game.triggered.connect(self.register_game)
        game_menu.addAction(register_game)
        
        launch_game = QAction('Launch Game', self)
        launch_game.setShortcut('F5')
        launch_game.triggered.connect(self.launch_game)
        game_menu.addAction(launch_game)
        
        # Agent menu
        agent_menu = menubar.addMenu('Agent')
        
        new_agent = QAction('New Agent', self)
        new_agent.triggered.connect(self.new_agent)
        agent_menu.addAction(new_agent)
        
        load_agent = QAction('Load Agent', self)
        load_agent.triggered.connect(self.load_agent)
        agent_menu.addAction(load_agent)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        visual_debugger = QAction('Visual Debugger', self)
        visual_debugger.setShortcut('F12')
        visual_debugger.triggered.connect(self.open_visual_debugger)
        tools_menu.addAction(visual_debugger)
        
        dataset_manager = QAction('Dataset Manager', self)
        dataset_manager.triggered.connect(self.open_dataset_manager)
        tools_menu.addAction(dataset_manager)
        
        plugin_manager = QAction('Plugin Manager', self)
        plugin_manager.triggered.connect(self.open_plugin_manager)
        tools_menu.addAction(plugin_manager)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        documentation = QAction('Documentation', self)
        documentation.triggered.connect(self.open_documentation)
        help_menu.addAction(documentation)
        
        about = QAction('About', self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)
        
    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Game Selection
        game_group = QGroupBox("Game Selection")
        game_layout = QVBoxLayout()
        
        self.game_combo = QComboBox()
        self.game_combo.addItems(["Select Game...", "Counter-Strike", "Fortnite", "League of Legends", "Minecraft"])
        game_layout.addWidget(QLabel("Game:"))
        game_layout.addWidget(self.game_combo)
        
        self.game_path_input = QLineEdit()
        self.game_path_input.setPlaceholderText("Game executable path...")
        game_layout.addWidget(QLabel("Path:"))
        game_layout.addWidget(self.game_path_input)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_game)
        game_layout.addWidget(self.browse_button)
        
        self.launch_button = QPushButton("Launch Game")
        self.launch_button.clicked.connect(self.launch_game)
        game_layout.addWidget(self.launch_button)
        
        game_group.setLayout(game_layout)
        layout.addWidget(game_group)
        
        # Agent Configuration
        agent_group = QGroupBox("Agent Configuration")
        agent_layout = QVBoxLayout()
        
        self.agent_type_combo = QComboBox()
        self.agent_type_combo.addItems(["DQN", "PPO", "Rainbow", "Scripted", "Random"])
        agent_layout.addWidget(QLabel("Agent Type:"))
        agent_layout.addWidget(self.agent_type_combo)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 1.0)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setDecimals(4)
        agent_layout.addWidget(QLabel("Learning Rate:"))
        agent_layout.addWidget(self.learning_rate)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 512)
        self.batch_size.setValue(32)
        agent_layout.addWidget(QLabel("Batch Size:"))
        agent_layout.addWidget(self.batch_size)
        
        self.load_agent_button = QPushButton("Load Agent")
        self.load_agent_button.clicked.connect(self.load_agent)
        agent_layout.addWidget(self.load_agent_button)
        
        self.save_agent_button = QPushButton("Save Agent")
        self.save_agent_button.clicked.connect(self.save_agent)
        agent_layout.addWidget(self.save_agent_button)
        
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)
        
        # Control Buttons
        control_group = QGroupBox("Control")
        control_layout = QVBoxLayout()
        
        # Row 1: Training controls
        train_row = QHBoxLayout()
        self.train_button = QPushButton("▶ Train")
        self.train_button.clicked.connect(self.toggle_training)
        train_row.addWidget(self.train_button)
        
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self.pause_training)
        self.pause_button.setEnabled(False)
        train_row.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        train_row.addWidget(self.stop_button)
        
        control_layout.addLayout(train_row)
        
        # Row 2: Recording controls
        record_row = QHBoxLayout()
        self.record_button = QPushButton("⏺ Record")
        self.record_button.clicked.connect(self.toggle_recording)
        record_row.addWidget(self.record_button)
        
        self.capture_button = QPushButton("📷 Capture")
        self.capture_button.clicked.connect(self.capture_frame)
        record_row.addWidget(self.capture_button)
        
        control_layout.addLayout(record_row)
        
        # Row 3: Play controls
        play_row = QHBoxLayout()
        self.play_button = QPushButton("🎮 Play")
        self.play_button.clicked.connect(self.play_game)
        play_row.addWidget(self.play_button)
        
        self.evaluate_button = QPushButton("📊 Evaluate")
        self.evaluate_button.clicked.connect(self.evaluate_agent)
        play_row.addWidget(self.evaluate_button)
        
        control_layout.addLayout(play_row)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Training Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0 / 0")
        progress_layout.addWidget(self.episode_label)
        
        self.episode_progress = QProgressBar()
        progress_layout.addWidget(self.episode_progress)
        
        self.reward_label = QLabel("Reward: 0.00")
        progress_layout.addWidget(self.reward_label)
        
        self.loss_label = QLabel("Loss: 0.0000")
        progress_layout.addWidget(self.loss_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_game_view(self):
        """Create the center game view panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Tab widget for different views
        self.view_tabs = QTabWidget()
        
        # Game View Tab
        game_tab = QWidget()
        game_layout = QVBoxLayout()
        
        # Game display
        self.game_display = QLabel()
        self.game_display.setMinimumSize(800, 600)
        self.game_display.setStyleSheet("background-color: #000000; border: 2px solid #3c3c3c;")
        self.game_display.setAlignment(Qt.AlignCenter)
        self.game_display.setText("No game loaded\nClick 'Launch Game' to start")
        self.game_display.setScaledContents(True)
        game_layout.addWidget(self.game_display)
        
        # Overlay controls
        overlay_row = QHBoxLayout()
        
        self.show_overlay = QCheckBox("Show Overlay")
        self.show_overlay.setChecked(True)
        overlay_row.addWidget(self.show_overlay)
        
        self.show_detections = QCheckBox("Show Detections")
        overlay_row.addWidget(self.show_detections)
        
        self.show_heatmap = QCheckBox("Show Heatmap")
        overlay_row.addWidget(self.show_heatmap)
        
        self.show_inputs = QCheckBox("Show Inputs")
        overlay_row.addWidget(self.show_inputs)
        
        overlay_row.addStretch()
        game_layout.addLayout(overlay_row)
        
        game_tab.setLayout(game_layout)
        self.view_tabs.addTab(game_tab, "Game View")
        
        # Agent Vision Tab
        vision_tab = QWidget()
        vision_layout = QGridLayout()
        
        # Create 4 vision displays
        self.vision_displays = []
        vision_labels = ["Raw Input", "Preprocessed", "Feature Map", "Action Heatmap"]
        
        for i in range(4):
            display = QLabel()
            display.setMinimumSize(380, 280)
            display.setStyleSheet("background-color: #000000; border: 1px solid #3c3c3c;")
            display.setAlignment(Qt.AlignCenter)
            display.setText(vision_labels[i])
            display.setScaledContents(True)
            self.vision_displays.append(display)
            
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(QLabel(vision_labels[i]))
            container_layout.addWidget(display)
            container.setLayout(container_layout)
            
            vision_layout.addWidget(container, i // 2, i % 2)
        
        vision_tab.setLayout(vision_layout)
        self.view_tabs.addTab(vision_tab, "Agent Vision")
        
        # Metrics Tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout()
        
        # Metrics display with real-time chart capability
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setPlaceholderText("Training metrics will appear here...\n\nFormat:\nEpisode | Reward | Loss | Epsilon | Steps")
        metrics_layout.addWidget(self.metrics_display)
        
        metrics_tab.setLayout(metrics_layout)
        self.view_tabs.addTab(metrics_tab, "Metrics")
        
        layout.addWidget(self.view_tabs)
        
        return panel
    
    def create_info_panel(self):
        """Create the right information panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # System Status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self.fps_label)
        
        self.cpu_label = QLabel("CPU: 0%")
        status_layout.addWidget(self.cpu_label)
        
        self.memory_label = QLabel("Memory: 0 MB")
        status_layout.addWidget(self.memory_label)
        
        self.gpu_label = QLabel("GPU: 0%")
        status_layout.addWidget(self.gpu_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Game State
        state_group = QGroupBox("Game State")
        state_layout = QVBoxLayout()
        
        self.state_text = QTextEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setMaximumHeight(150)
        state_layout.addWidget(self.state_text)
        
        state_group.setLayout(state_layout)
        layout.addWidget(state_group)
        
        # Action History
        action_group = QGroupBox("Action History")
        action_layout = QVBoxLayout()
        
        self.action_list = QListWidget()
        self.action_list.setMaximumHeight(200)
        action_layout.addWidget(self.action_list)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Console Output
        console_group = QGroupBox("Console")
        console_layout = QVBoxLayout()
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 10px;")
        console_layout.addWidget(self.console_output)
        
        # Console input
        self.console_input = QLineEdit()
        self.console_input.setPlaceholderText("Enter command...")
        self.console_input.returnPressed.connect(self.execute_command)
        console_layout.addWidget(self.console_input)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        return panel
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        
        # Create status widgets
        self.status_message = QLabel("Ready")
        self.status_bar.addWidget(self.status_message)
        
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        self.frame_counter = QLabel("Frame: 0")
        self.status_bar.addPermanentWidget(self.frame_counter)
        
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        self.time_label = QLabel("00:00:00")
        self.status_bar.addPermanentWidget(self.time_label)
        
        self.status_bar.addPermanentWidget(QLabel(" | "))
        
        self.connection_status = QLabel("● Connected")
        self.connection_status.setStyleSheet("color: #00ff00;")
        self.status_bar.addPermanentWidget(self.connection_status)
    
    def setup_timers(self):
        """Setup update timers."""
        # UI update timer (30 FPS)
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(33)  # ~30 FPS
        
        # System status timer (1 Hz)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(1000)
        
        # Time update timer
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)
    
    def update_ui(self):
        """Update the UI with latest frame."""
        # Update frame counter
        self.frame_count += 1
        self.frame_counter.setText(f"Frame: {self.frame_count}")
        
        # Update game display if we have a frame
        if self.last_frame is not None:
            # Convert numpy array to QImage
            height, width, channel = self.last_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.last_frame.data, width, height, 
                           bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale to fit display
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.game_display.size(), 
                                         Qt.KeepAspectRatio, 
                                         Qt.SmoothTransformation)
            self.game_display.setPixmap(scaled_pixmap)
    
    def update_system_status(self):
        """Update system status information."""
        try:
            import psutil
            
            # Update CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
            
            # Update Memory
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self.memory_label.setText(f"Memory: {memory_mb:.0f} MB")
            
            # Update GPU (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_label.setText(f"GPU: {gpu.load * 100:.1f}%")
            except Exception:
                # GPUtil not available or GPU not accessible
                self.gpu_label.setText("GPU: N/A")
                
        except ImportError:
            # psutil not installed - show static values
            self.cpu_label.setText("CPU: Install psutil")
            self.memory_label.setText("Memory: Install psutil")
    
    def update_time(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
    
    # Action handlers
    def new_project(self):
        """Create a new project."""
        dialog = QInputDialog()
        name, ok = dialog.getText(self, "New Project", "Project name:")
        if ok and name:
            self.log_console(f"Created new project: {name}")
            self.status_message.setText(f"Project: {name}")
    
    def open_project(self):
        """Open an existing project."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Project", "", 
                                                   "Nexus Project (*.nxp);;All Files (*)")
        if file_name:
            self.log_console(f"Opened project: {file_name}")
    
    def save_project(self):
        """Save the current project."""
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Project", "", 
                                                   "Nexus Project (*.nxp);;All Files (*)")
        if file_name:
            self.log_console(f"Saved project: {file_name}")
    
    def register_game(self):
        """Register a new game."""
        dialog = GameRegistrationDialog(self)
        if dialog.exec_():
            game_info = dialog.get_game_info()
            self.log_console(f"Registered game: {game_info['name']}")
            self.game_combo.addItem(game_info['name'])
    
    def browse_game(self):
        """Browse for game executable."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Game Executable", "", 
                                                   "Executable (*.exe);;All Files (*)")
        if file_name:
            self.game_path_input.setText(file_name)
    
    def launch_game(self):
        """Launch the selected game."""
        game = self.game_combo.currentText()
        if game != "Select Game...":
            self.log_console(f"Launching {game}...")
            self.status_message.setText(f"Game: {game} (Running)")
            
            # Simulate game launch
            self.connection_status.setText("● Connected")
            self.connection_status.setStyleSheet("color: #00ff00;")
            
            # Start capturing frames (simulated)
            self.start_frame_capture()
    
    def start_frame_capture(self):
        """Start capturing frames from the game."""
        # This would connect to actual frame grabber
        # For now, generate test frames
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_game_frame)
        self.capture_timer.start(33)  # 30 FPS
    
    def capture_game_frame(self):
        """Capture a frame from the game."""
        # Generate test frame
        frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        # Add some visual elements
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, (400, 300), 50, (255, 0, 0), -1)
        
        self.last_frame = frame
        self.fps = 30  # Simulated FPS
        self.fps_label.setText(f"FPS: {self.fps}")
    
    def new_agent(self):
        """Create a new agent."""
        dialog = AgentCreationDialog(self)
        if dialog.exec_():
            agent_config = dialog.get_agent_config()
            self.log_console(f"Created agent: {agent_config['name']}")
    
    def load_agent(self):
        """Load an existing agent."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Agent", "", 
                                                   "Agent Model (*.pth *.h5);;All Files (*)")
        if file_name:
            self.log_console(f"Loaded agent: {file_name}")
            self.current_agent = Path(file_name).stem
            self.status_message.setText(f"Agent: {self.current_agent}")
    
    def save_agent(self):
        """Save the current agent."""
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Agent", "", 
                                                   "PyTorch Model (*.pth);;TensorFlow Model (*.h5)")
        if file_name:
            self.log_console(f"Saved agent: {file_name}")
    
    def toggle_training(self):
        """Toggle training on/off."""
        if not self.is_training:
            self.start_training()
        else:
            self.pause_training()
    
    def start_training(self):
        """Start training the agent."""
        self.is_training = True
        self.train_button.setText("⏸ Pause")
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        
        self.log_console("Training started...")
        self.status_message.setText("Training in progress...")
        
        # Start training loop (simulated)
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self.training_step)
        self.training_timer.start(100)  # Update every 100ms
    
    def training_step(self):
        """Perform one training step."""
        if self.is_training:
            # Simulate training progress
            current = self.episode_progress.value()
            if current < 100:
                self.episode_progress.setValue(current + 1)
                self.episode_label.setText(f"Episode: {current + 1} / 100")
                
                # Update metrics
                reward = np.random.uniform(-1, 1)
                loss = np.random.uniform(0, 0.1)
                self.reward_label.setText(f"Reward: {reward:.2f}")
                self.loss_label.setText(f"Loss: {loss:.4f}")
                
                # Add to metrics display
                self.metrics_display.append(f"Episode {current + 1}: Reward={reward:.2f}, Loss={loss:.4f}")
            else:
                self.stop_training()
    
    def pause_training(self):
        """Pause training."""
        self.is_training = False
        self.train_button.setText("▶ Resume")
        self.log_console("Training paused")
        self.status_message.setText("Training paused")
    
    def stop_training(self):
        """Stop training."""
        self.is_training = False
        self.train_button.setText("▶ Train")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        
        if hasattr(self, 'training_timer'):
            self.training_timer.stop()
        
        self.episode_progress.setValue(0)
        self.episode_label.setText("Episode: 0 / 0")
        
        self.log_console("Training stopped")
        self.status_message.setText("Ready")
    
    def toggle_recording(self):
        """Toggle recording on/off."""
        if not self.is_recording:
            self.is_recording = True
            self.record_button.setText("⏹ Stop Recording")
            self.record_button.setStyleSheet("QPushButton { background-color: #cc0000; }")
            self.log_console("Recording started...")
        else:
            self.is_recording = False
            self.record_button.setText("⏺ Record")
            self.record_button.setStyleSheet("")
            self.log_console("Recording stopped")
    
    def capture_frame(self):
        """Capture a single frame."""
        if self.last_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"
            cv2.imwrite(filename, self.last_frame)
            self.log_console(f"Frame captured: {filename}")
    
    def play_game(self):
        """Let the agent play the game."""
        if self.current_agent:
            self.log_console(f"Agent {self.current_agent} is playing...")
            self.status_message.setText("Agent playing...")
            
            # Simulate agent actions
            actions = ["Move Left", "Move Right", "Jump", "Attack", "Defend"]
            for _ in range(5):
                action = np.random.choice(actions)
                self.action_list.addItem(f"{datetime.now().strftime('%H:%M:%S')} - {action}")
        else:
            QMessageBox.warning(self, "No Agent", "Please load an agent first")
    
    def evaluate_agent(self):
        """Evaluate the agent's performance."""
        if self.current_agent:
            self.log_console(f"Evaluating agent {self.current_agent}...")
            
            # Show evaluation dialog
            dialog = EvaluationDialog(self)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "No Agent", "Please load an agent first")
    
    def open_visual_debugger(self):
        """Open the visual debugger."""
        from nexus.gui.visual_debugger import VisualDebuggerWindow
        self.debugger = VisualDebuggerWindow()
        self.debugger.show()
    
    def open_dataset_manager(self):
        """Open the dataset manager."""
        from nexus.gui.dataset_manager import DatasetManagerWindow
        self.dataset_manager = DatasetManagerWindow()
        self.dataset_manager.show()
    
    def open_plugin_manager(self):
        """Open the plugin manager."""
        from nexus.gui.plugin_manager import PluginManagerWindow
        self.plugin_manager = PluginManagerWindow()
        self.plugin_manager.show()
    
    def open_documentation(self):
        """Open documentation."""
        import webbrowser
        webbrowser.open("https://github.com/neophrythe/Nexus-AI-Framework")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Nexus AI Framework",
                         "Nexus AI Framework v1.0.0\n\n"
                         "A comprehensive game automation and AI training platform.\n\n"
                         "© 2025 Nexus AI Team")
    
    def execute_command(self):
        """Execute a console command."""
        command = self.console_input.text()
        if command:
            self.log_console(f"> {command}")
            
            # Process command
            if command.startswith("nexus "):
                parts = command.split()
                if len(parts) > 1:
                    if parts[1] == "train":
                        self.start_training()
                    elif parts[1] == "stop":
                        self.stop_training()
                    elif parts[1] == "capture":
                        self.capture_frame()
                    else:
                        self.log_console(f"Unknown command: {parts[1]}")
            
            self.console_input.clear()
    
    def log_console(self, message):
        """Log a message to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_output.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(self, 'Exit', 'Are you sure you want to exit?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Cleanup
            if hasattr(self, 'capture_timer'):
                self.capture_timer.stop()
            if hasattr(self, 'training_timer'):
                self.training_timer.stop()
            event.accept()
        else:
            event.ignore()


class GameRegistrationDialog(QDialog):
    """Dialog for registering a new game."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Register Game")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_input = QLineEdit()
        layout.addRow("Game Name:", self.name_input)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Executable", "Steam", "Epic", "Android (BlueStacks)"])
        layout.addRow("Type:", self.type_combo)
        
        self.path_input = QLineEdit()
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_button)
        layout.addRow("Path/Package:", path_layout)
        
        self.window_input = QLineEdit()
        layout.addRow("Window Name:", self.window_input)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
    
    def browse_path(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Game", "", 
                                                   "Executable (*.exe);;All Files (*)")
        if file_name:
            self.path_input.setText(file_name)
    
    def get_game_info(self):
        return {
            'name': self.name_input.text(),
            'type': self.type_combo.currentText(),
            'path': self.path_input.text(),
            'window': self.window_input.text()
        }


class AgentCreationDialog(QDialog):
    """Dialog for creating a new agent."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Agent")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QFormLayout()
        
        self.name_input = QLineEdit()
        layout.addRow("Agent Name:", self.name_input)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["DQN", "PPO", "Rainbow", "A3C", "SAC"])
        layout.addRow("Algorithm:", self.type_combo)
        
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.0001, 1.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(0.001)
        self.lr_input.setDecimals(4)
        layout.addRow("Learning Rate:", self.lr_input)
        
        self.batch_input = QSpinBox()
        self.batch_input.setRange(1, 512)
        self.batch_input.setValue(32)
        layout.addRow("Batch Size:", self.batch_input)
        
        self.epsilon_input = QDoubleSpinBox()
        self.epsilon_input.setRange(0.0, 1.0)
        self.epsilon_input.setSingleStep(0.01)
        self.epsilon_input.setValue(1.0)
        layout.addRow("Initial Epsilon:", self.epsilon_input)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
    
    def get_agent_config(self):
        return {
            'name': self.name_input.text(),
            'type': self.type_combo.currentText(),
            'learning_rate': self.lr_input.value(),
            'batch_size': self.batch_input.value(),
            'epsilon': self.epsilon_input.value()
        }


class EvaluationDialog(QDialog):
    """Dialog for agent evaluation results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Evaluation")
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value", "Baseline"])
        
        # Add sample results
        metrics = [
            ("Average Reward", "156.3", "100.0"),
            ("Win Rate", "78.5%", "50.0%"),
            ("Average Episode Length", "234", "300"),
            ("Actions Per Minute", "145", "120"),
            ("Success Rate", "82.3%", "60.0%")
        ]
        
        self.results_table.setRowCount(len(metrics))
        for i, (metric, value, baseline) in enumerate(metrics):
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(i, 1, QTableWidgetItem(value))
            self.results_table.setItem(i, 2, QTableWidgetItem(baseline))
        
        layout.addWidget(self.results_table)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.resize(400, 300)


def main():
    """Main entry point for GUI."""
    if not PYQT_AVAILABLE:
        print("PyQt5 not installed. Please install with: pip install PyQt5")
        print("Falling back to tkinter interface...")
        # Could implement tkinter version here
        return
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon
    app.setWindowIcon(QIcon('nexus_icon.png'))
    
    window = NexusGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()