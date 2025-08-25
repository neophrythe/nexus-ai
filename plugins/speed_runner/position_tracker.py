"""
Position tracking module for Speed Runner Plugin.
"""

import numpy as np
import cv2
import time
import math
import struct
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
from threading import Thread, Lock
import logging

try:
    import pymem
    import pymem.process
    HAS_PYMEM = True
except ImportError:
    HAS_PYMEM = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

from filterpy.kalman import KalmanFilter


@dataclass
class Position3D:
    """3D position with timestamp."""
    x: float
    y: float
    z: float
    timestamp: float = field(default_factory=time.time)
    
    def distance_to(self, other: 'Position3D') -> float:
        """Calculate distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Get position as tuple."""
        return (self.x, self.y, self.z)
    
    def __sub__(self, other: 'Position3D') -> 'Position3D':
        """Subtract positions to get displacement."""
        return Position3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )


@dataclass
class Velocity3D:
    """3D velocity vector."""
    vx: float
    vy: float
    vz: float
    
    @property
    def speed(self) -> float:
        """Get scalar speed."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
    
    @property
    def horizontal_speed(self) -> float:
        """Get horizontal speed (XY plane)."""
        return math.sqrt(self.vx ** 2 + self.vy ** 2)
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Get velocity as tuple."""
        return (self.vx, self.vy, self.vz)


class PositionTracker:
    """Tracks player position using various methods."""
    
    def __init__(self, method: str = "memory", update_rate: float = 60.0):
        """
        Initialize position tracker.
        
        Args:
            method: Tracking method (memory, ocr, image, api)
            update_rate: Update frequency in Hz
        """
        self.method = method
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        
        # Current state
        self.current_position = Position3D(0, 0, 0)
        self.current_velocity = Velocity3D(0, 0, 0)
        self.position_history = deque(maxlen=int(update_rate * 5))  # 5 seconds
        
        # Kalman filter for smoothing
        self.kalman_filter = self._init_kalman_filter()
        
        # Memory reading
        self.process = None
        self.base_address = None
        self.position_offsets = None
        
        # OCR region
        self.ocr_region = None
        
        # Threading
        self.tracking_thread = None
        self.is_tracking = False
        self.lock = Lock()
        
        # Callbacks
        self.position_callbacks = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize Kalman filter for position smoothing."""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        
        # State: [x, vx, y, vy, z, vz]
        # Measurement: [x, y, z]
        
        dt = self.update_interval
        
        # State transition matrix
        kf.F = np.array([
            [1, dt, 0, 0,  0, 0],
            [0, 1,  0, 0,  0, 0],
            [0, 0,  1, dt, 0, 0],
            [0, 0,  0, 1,  0, 0],
            [0, 0,  0, 0,  1, dt],
            [0, 0,  0, 0,  0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        # Process noise
        kf.Q *= 0.1
        
        # Measurement noise
        kf.R *= 1.0
        
        # Initial state
        kf.x = np.zeros((6, 1))
        
        # Initial covariance
        kf.P *= 100
        
        return kf
    
    def start_tracking(self, process_name: Optional[str] = None,
                      position_offsets: Optional[List[int]] = None):
        """Start position tracking."""
        if self.is_tracking:
            return
        
        # Initialize based on method
        if self.method == "memory" and HAS_PYMEM:
            if not self._init_memory_reading(process_name, position_offsets):
                self.logger.error("Failed to initialize memory reading")
                return
        elif self.method == "ocr" and not HAS_TESSERACT:
            self.logger.error("Tesseract not available for OCR")
            return
        
        # Start tracking thread
        self.is_tracking = True
        self.tracking_thread = Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        
        self.logger.info(f"Started position tracking using {self.method} method")
    
    def stop_tracking(self):
        """Stop position tracking."""
        self.is_tracking = False
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        
        if self.process:
            self.process.close_handle()
            self.process = None
        
        self.logger.info("Stopped position tracking")
    
    def _init_memory_reading(self, process_name: str, 
                           position_offsets: List[int]) -> bool:
        """Initialize memory reading for position."""
        if not HAS_PYMEM:
            return False
        
        try:
            # Open process
            self.process = pymem.Pymem(process_name)
            
            # Get base address
            module = pymem.process.module_from_name(
                self.process.process_handle,
                process_name
            )
            self.base_address = module.lpBaseOfDll
            
            # Store offsets
            self.position_offsets = position_offsets
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open process: {e}")
            return False
    
    def _tracking_loop(self):
        """Main tracking loop."""
        last_update = time.time()
        
        while self.is_tracking:
            current_time = time.time()
            
            if current_time - last_update >= self.update_interval:
                # Read position
                position = self._read_position()
                
                if position:
                    # Update state
                    self._update_position(position)
                
                last_update = current_time
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def _read_position(self) -> Optional[Position3D]:
        """Read position based on configured method."""
        if self.method == "memory":
            return self._read_position_from_memory()
        elif self.method == "ocr":
            return self._read_position_from_ocr()
        elif self.method == "image":
            return self._read_position_from_image()
        elif self.method == "api":
            return self._read_position_from_api()
        else:
            return None
    
    def _read_position_from_memory(self) -> Optional[Position3D]:
        """Read position from game memory."""
        if not self.process or not self.position_offsets:
            return None
        
        try:
            # Follow pointer chain
            address = self.base_address
            
            for offset in self.position_offsets[:-3]:
                address = self.process.read_int(address + offset)
            
            # Read position values (assuming floats)
            x = self.process.read_float(address + self.position_offsets[-3])
            y = self.process.read_float(address + self.position_offsets[-2])
            z = self.process.read_float(address + self.position_offsets[-1])
            
            return Position3D(x, y, z)
            
        except Exception as e:
            self.logger.debug(f"Memory read error: {e}")
            return None
    
    def _read_position_from_ocr(self) -> Optional[Position3D]:
        """Read position from screen using OCR."""
        if not HAS_TESSERACT or not self.ocr_region:
            return None
        
        try:
            # Capture screen region
            import pyautogui
            screenshot = pyautogui.screenshot(region=self.ocr_region)
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            
            # Apply preprocessing
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(
                thresh,
                config='--psm 6 -c tessedit_char_whitelist=0123456789.-,'
            )
            
            # Parse coordinates
            numbers = [float(n) for n in text.replace(',', '').split() if n]
            
            if len(numbers) >= 2:
                x, y = numbers[0], numbers[1]
                z = numbers[2] if len(numbers) >= 3 else 0
                return Position3D(x, y, z)
            
        except Exception as e:
            self.logger.debug(f"OCR error: {e}")
        
        return None
    
    def _read_position_from_image(self) -> Optional[Position3D]:
        """Read position from image recognition."""
        # Placeholder for image-based position detection
        # Could use feature matching, marker detection, etc.
        return None
    
    def _read_position_from_api(self) -> Optional[Position3D]:
        """Read position from game API."""
        # Placeholder for API-based position reading
        # Would connect to game's API or mod interface
        return None
    
    def _update_position(self, new_position: Position3D):
        """Update position and calculate velocity."""
        with self.lock:
            # Apply Kalman filter
            measurement = np.array([
                [new_position.x],
                [new_position.y],
                [new_position.z]
            ])
            
            self.kalman_filter.predict()
            self.kalman_filter.update(measurement)
            
            # Extract filtered position and velocity
            state = self.kalman_filter.x
            
            filtered_position = Position3D(
                state[0, 0],  # x
                state[2, 0],  # y
                state[4, 0]   # z
            )
            
            filtered_velocity = Velocity3D(
                state[1, 0],  # vx
                state[3, 0],  # vy
                state[5, 0]   # vz
            )
            
            # Update state
            self.current_position = filtered_position
            self.current_velocity = filtered_velocity
            
            # Add to history
            self.position_history.append(filtered_position)
            
            # Trigger callbacks
            for callback in self.position_callbacks:
                callback(filtered_position, filtered_velocity)
    
    def get_position(self) -> Position3D:
        """Get current position."""
        with self.lock:
            return self.current_position
    
    def get_velocity(self) -> Velocity3D:
        """Get current velocity."""
        with self.lock:
            return self.current_velocity
    
    def get_speed(self) -> float:
        """Get current speed."""
        return self.current_velocity.speed
    
    def get_trajectory(self, duration: float = 1.0) -> List[Position3D]:
        """Get recent trajectory."""
        with self.lock:
            cutoff_time = time.time() - duration
            return [
                pos for pos in self.position_history
                if pos.timestamp >= cutoff_time
            ]
    
    def predict_position(self, time_ahead: float) -> Position3D:
        """Predict future position based on current velocity."""
        with self.lock:
            return Position3D(
                self.current_position.x + self.current_velocity.vx * time_ahead,
                self.current_position.y + self.current_velocity.vy * time_ahead,
                self.current_position.z + self.current_velocity.vz * time_ahead
            )
    
    def add_position_callback(self, callback):
        """Add callback for position updates."""
        self.position_callbacks.append(callback)
    
    def set_ocr_region(self, x: int, y: int, width: int, height: int):
        """Set screen region for OCR."""
        self.ocr_region = (x, y, width, height)
    
    def calibrate_coordinates(self, reference_points: List[Tuple[Position3D, Position3D]]):
        """Calibrate coordinate transformation."""
        # Would implement coordinate system calibration
        # using reference points (game coords -> real coords)
        pass


class CheckpointDetector:
    """Detects and manages speedrun checkpoints."""
    
    def __init__(self, position_tracker: PositionTracker):
        """
        Initialize checkpoint detector.
        
        Args:
            position_tracker: Position tracker instance
        """
        self.position_tracker = position_tracker
        self.checkpoints = []
        self.current_checkpoint = 0
        
        # Detection settings
        self.detection_radius = 50.0
        self.detection_height = 10.0
        self.require_stop = False
        self.stop_duration = 1.0  # seconds
        
        # Detection state
        self.checkpoint_timers = {}
        self.checkpoint_callbacks = []
        
        # Register position callback
        position_tracker.add_position_callback(self._on_position_update)
    
    def add_checkpoint(self, position: Position3D, radius: float = 50.0,
                      name: Optional[str] = None):
        """Add a checkpoint."""
        checkpoint = {
            'position': position,
            'radius': radius,
            'name': name or f"Checkpoint {len(self.checkpoints) + 1}",
            'reached': False,
            'split_time': None
        }
        
        self.checkpoints.append(checkpoint)
        return checkpoint
    
    def _on_position_update(self, position: Position3D, velocity: Velocity3D):
        """Handle position update from tracker."""
        # Check current checkpoint
        if self.current_checkpoint >= len(self.checkpoints):
            return
        
        checkpoint = self.checkpoints[self.current_checkpoint]
        
        # Calculate distance to checkpoint
        distance = position.distance_to(checkpoint['position'])
        
        # Check if within radius
        if distance <= checkpoint['radius']:
            # Check height if applicable
            height_diff = abs(position.z - checkpoint['position'].z)
            if height_diff > self.detection_height:
                return
            
            # Handle stop requirement
            if self.require_stop:
                checkpoint_id = id(checkpoint)
                
                if velocity.speed < 1.0:  # Nearly stopped
                    if checkpoint_id not in self.checkpoint_timers:
                        self.checkpoint_timers[checkpoint_id] = time.time()
                    elif time.time() - self.checkpoint_timers[checkpoint_id] >= self.stop_duration:
                        self._checkpoint_reached(checkpoint)
                else:
                    # Moving, reset timer
                    if checkpoint_id in self.checkpoint_timers:
                        del self.checkpoint_timers[checkpoint_id]
            else:
                # No stop required
                self._checkpoint_reached(checkpoint)
    
    def _checkpoint_reached(self, checkpoint: Dict[str, Any]):
        """Handle checkpoint reached."""
        if checkpoint['reached']:
            return  # Already reached
        
        checkpoint['reached'] = True
        checkpoint['split_time'] = time.time()
        
        # Trigger callbacks
        for callback in self.checkpoint_callbacks:
            callback(checkpoint)
        
        # Move to next checkpoint
        self.current_checkpoint += 1
    
    def add_checkpoint_callback(self, callback):
        """Add callback for checkpoint events."""
        self.checkpoint_callbacks.append(callback)
    
    def reset(self):
        """Reset checkpoint progress."""
        self.current_checkpoint = 0
        self.checkpoint_timers.clear()
        
        for checkpoint in self.checkpoints:
            checkpoint['reached'] = False
            checkpoint['split_time'] = None
    
    def get_progress(self) -> Tuple[int, int]:
        """Get checkpoint progress (current, total)."""
        return (self.current_checkpoint, len(self.checkpoints))
    
    def get_next_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get next checkpoint."""
        if self.current_checkpoint < len(self.checkpoints):
            return self.checkpoints[self.current_checkpoint]
        return None
    
    def get_distance_to_next(self) -> Optional[float]:
        """Get distance to next checkpoint."""
        next_checkpoint = self.get_next_checkpoint()
        if next_checkpoint:
            current_pos = self.position_tracker.get_position()
            return current_pos.distance_to(next_checkpoint['position'])
        return None