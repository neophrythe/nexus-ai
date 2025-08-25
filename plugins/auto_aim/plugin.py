"""
Auto-Aim Assistant Plugin for Nexus Game AI Framework

Provides aim assistance for FPS games (for AI training and research only).
"""

import cv2
import numpy as np
import time
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import structlog

from nexus.core.plugin_base import PluginBase
from nexus.core.input_controller import InputController

logger = structlog.get_logger()


@dataclass
class Target:
    """Represents a detected target."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    distance: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    target_type: str = "enemy"
    priority: float = 1.0
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def head_position(self) -> Tuple[int, int]:
        """Estimate head position (top 20% of target)."""
        return (self.x + self.width // 2, self.y + int(self.height * 0.2))


class AutoAimPlugin(PluginBase):
    """
    Plugin for aim assistance in FPS games.
    
    Features:
    - Target detection and tracking
    - Aim smoothing and prediction
    - Recoil compensation
    - Target prioritization
    - Configurable aim zones
    - Training mode for AI agents
    
    NOTE: This plugin is for AI training and research purposes only.
    Using aim assistance in online games violates terms of service.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Auto-Aim Assistant"
        self.version = "1.0.0"
        self.description = "Aim assistance for FPS AI training"
        
        # Detection settings
        self.detection_model = None
        self.detection_confidence = 0.5
        self.target_colors = {  # HSV ranges for color-based detection
            'red': [(0, 100, 100), (10, 255, 255)],
            'red2': [(170, 100, 100), (180, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)],
            'blue': [(100, 100, 100), (130, 255, 255)]
        }
        
        # Aim settings
        self.aim_enabled = False
        self.aim_mode = 'assist'  # 'assist', 'lock', 'training'
        self.aim_smoothness = 5.0  # Higher = smoother
        self.aim_fov = 100  # Field of view in pixels
        self.aim_zone = 'head'  # 'head', 'body', 'center'
        self.aim_prediction = True
        self.aim_key = 'right_mouse'  # Aim activation key
        
        # Recoil compensation
        self.recoil_enabled = False
        self.recoil_pattern = []
        self.recoil_multiplier = 1.0
        self.current_spray_index = 0
        
        # Target tracking
        self.targets: List[Target] = []
        self.target_history = deque(maxlen=30)
        self.locked_target: Optional[Target] = None
        
        # Screen info
        self.screen_width = 1920
        self.screen_height = 1080
        self.crosshair_x = self.screen_width // 2
        self.crosshair_y = self.screen_height // 2
        
        # Performance
        self.last_aim_time = 0
        self.aim_delay = 0.01  # Minimum delay between aim adjustments
        
        # Statistics
        self.stats = {
            'targets_detected': 0,
            'aim_adjustments': 0,
            'headshots': 0,
            'accuracy': 0.0
        }
        
        logger.warning("Auto-Aim Plugin loaded - FOR AI TRAINING ONLY")
        logger.warning("Do not use in online games - violates TOS")
    
    def on_load(self):
        """Called when plugin is loaded."""
        logger.info(f"Loading {self.name} v{self.version}")
        
        # Load configuration
        self.aim_enabled = self.config.get('enabled', False)
        self.aim_mode = self.config.get('mode', 'assist')
        self.aim_smoothness = self.config.get('smoothness', 5.0)
        self.aim_fov = self.config.get('fov', 100)
        self.aim_zone = self.config.get('zone', 'head')
        
        # Initialize detection model if specified
        model_path = self.config.get('model_path')
        if model_path:
            self.load_detection_model(model_path)
        
        # Update screen dimensions
        self.screen_width = self.config.get('screen_width', 1920)
        self.screen_height = self.config.get('screen_height', 1080)
        self.crosshair_x = self.screen_width // 2
        self.crosshair_y = self.screen_height // 2
    
    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process game frame for target detection and aiming."""
        if not self.aim_enabled:
            return frame
        
        # Detect targets
        self.targets = self.detect_targets(frame)
        self.stats['targets_detected'] = len(self.targets)
        
        # Track targets
        self.update_target_tracking()
        
        # Perform aim assistance
        if self.targets and self.should_aim():
            self.perform_aim_assist()
        
        # Handle recoil compensation
        if self.recoil_enabled and self.is_firing():
            self.compensate_recoil()
        
        # Draw debug overlay
        if self.config.get('show_overlay', True):
            frame = self.draw_overlay(frame)
        
        return frame
    
    def detect_targets(self, frame: np.ndarray) -> List[Target]:
        """Detect targets in frame."""
        targets = []
        
        # Use ML model if available
        if self.detection_model:
            targets = self.detect_with_model(frame)
        else:
            # Fallback to color-based detection
            targets = self.detect_by_color(frame)
        
        # Filter targets by FOV
        targets = self.filter_by_fov(targets)
        
        # Calculate distances and priorities
        for target in targets:
            target.distance = self.calculate_distance_to_crosshair(target)
            target.priority = self.calculate_priority(target)
        
        # Sort by priority
        targets.sort(key=lambda t: t.priority, reverse=True)
        
        return targets
    
    def detect_with_model(self, frame: np.ndarray) -> List[Target]:
        """Detect targets using ML model."""
        # This would use YOLO or similar model
        # Placeholder implementation
        return []
    
    def detect_by_color(self, frame: np.ndarray) -> List[Target]:
        """Detect targets by color (enemy outlines, healthbars, etc)."""
        targets = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for target colors
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for color_name, (lower, upper) in self.target_colors.items():
            if color_name in self.config.get('target_colors', ['red']):
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.bitwise_or(mask, color_mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (human-like)
                aspect_ratio = h / w if w > 0 else 0
                if 1.5 < aspect_ratio < 4.0:
                    targets.append(Target(
                        x=x, y=y, width=w, height=h,
                        confidence=min(area / 10000, 1.0)  # Simple confidence
                    ))
        
        return targets
    
    def filter_by_fov(self, targets: List[Target]) -> List[Target]:
        """Filter targets within aim FOV."""
        filtered = []
        
        for target in targets:
            distance = self.calculate_distance_to_crosshair(target)
            if distance <= self.aim_fov:
                filtered.append(target)
        
        return filtered
    
    def calculate_distance_to_crosshair(self, target: Target) -> float:
        """Calculate distance from target to crosshair."""
        target_x, target_y = self.get_aim_point(target)
        dx = target_x - self.crosshair_x
        dy = target_y - self.crosshair_y
        return math.sqrt(dx * dx + dy * dy)
    
    def get_aim_point(self, target: Target) -> Tuple[int, int]:
        """Get the point to aim at on target."""
        if self.aim_zone == 'head':
            return target.head_position
        elif self.aim_zone == 'body':
            return target.center
        else:
            return target.center
    
    def calculate_priority(self, target: Target) -> float:
        """Calculate target priority for aiming."""
        priority = 1.0
        
        # Distance factor (closer = higher priority)
        distance_factor = 1.0 - (target.distance / self.aim_fov)
        priority *= distance_factor
        
        # Confidence factor
        priority *= target.confidence
        
        # Size factor (larger = higher priority)
        size_factor = (target.width * target.height) / (self.screen_width * self.screen_height)
        priority *= (1.0 + size_factor * 10)
        
        return priority
    
    def update_target_tracking(self):
        """Update target velocity and prediction."""
        self.target_history.append(self.targets)
        
        if len(self.target_history) < 2:
            return
        
        # Calculate velocities
        current_targets = self.target_history[-1]
        previous_targets = self.target_history[-2]
        
        for current in current_targets:
            # Find matching target in previous frame
            best_match = None
            min_distance = float('inf')
            
            for previous in previous_targets:
                dx = current.center[0] - previous.center[0]
                dy = current.center[1] - previous.center[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < min_distance and distance < 50:  # Max movement threshold
                    min_distance = distance
                    best_match = previous
            
            if best_match:
                # Calculate velocity
                dt = 1.0 / 60  # Assume 60 FPS
                current.velocity = (
                    (current.center[0] - best_match.center[0]) / dt,
                    (current.center[1] - best_match.center[1]) / dt
                )
    
    def should_aim(self) -> bool:
        """Check if aim assistance should be active."""
        # Check aim key (would check actual input in real implementation)
        # For now, always active in training mode
        if self.aim_mode == 'training':
            return True
        
        # Check time delay
        current_time = time.time()
        if current_time - self.last_aim_time < self.aim_delay:
            return False
        
        return True
    
    def perform_aim_assist(self):
        """Perform aim assistance."""
        # Select target
        if self.aim_mode == 'lock' and self.locked_target:
            target = self.locked_target
        else:
            target = self.targets[0]  # Highest priority
            if self.aim_mode == 'lock':
                self.locked_target = target
        
        # Get aim point
        aim_x, aim_y = self.get_aim_point(target)
        
        # Apply prediction if enabled
        if self.aim_prediction and target.velocity != (0.0, 0.0):
            # Simple linear prediction
            prediction_time = 0.1  # Predict 100ms ahead
            aim_x += int(target.velocity[0] * prediction_time)
            aim_y += int(target.velocity[1] * prediction_time)
        
        # Calculate adjustment
        dx = aim_x - self.crosshair_x
        dy = aim_y - self.crosshair_y
        
        # Apply smoothing
        if self.aim_mode == 'assist':
            dx = dx / self.aim_smoothness
            dy = dy / self.aim_smoothness
        
        # Move mouse (would use actual input in real implementation)
        self.move_aim(dx, dy)
        
        # Update stats
        self.stats['aim_adjustments'] += 1
        self.last_aim_time = time.time()
    
    def move_aim(self, dx: float, dy: float):
        """Move aim by specified amount."""
        # In real implementation, this would move the mouse
        # For training mode, we just log the adjustment
        if self.aim_mode == 'training':
            logger.debug(f"Aim adjustment: ({dx:.2f}, {dy:.2f})")
        else:
            # Would use InputController to move mouse
            pass
    
    def compensate_recoil(self):
        """Compensate for weapon recoil."""
        if not self.recoil_pattern:
            return
        
        if self.current_spray_index < len(self.recoil_pattern):
            recoil = self.recoil_pattern[self.current_spray_index]
            compensation_x = -recoil[0] * self.recoil_multiplier
            compensation_y = -recoil[1] * self.recoil_multiplier
            
            self.move_aim(compensation_x, compensation_y)
            self.current_spray_index += 1
    
    def is_firing(self) -> bool:
        """Check if weapon is firing."""
        # Would check actual game state
        return False
    
    def reset_recoil(self):
        """Reset recoil compensation."""
        self.current_spray_index = 0
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw debug overlay on frame."""
        overlay = frame.copy()
        
        # Draw FOV circle
        cv2.circle(
            overlay,
            (self.crosshair_x, self.crosshair_y),
            self.aim_fov,
            (0, 255, 0),
            1
        )
        
        # Draw targets
        for i, target in enumerate(self.targets):
            color = (0, 0, 255) if i == 0 else (0, 255, 255)  # Red for primary
            
            # Draw bounding box
            cv2.rectangle(
                overlay,
                (target.x, target.y),
                (target.x + target.width, target.y + target.height),
                color,
                2
            )
            
            # Draw aim point
            aim_x, aim_y = self.get_aim_point(target)
            cv2.circle(overlay, (aim_x, aim_y), 5, color, -1)
            
            # Draw info
            info_text = f"P:{target.priority:.2f} D:{target.distance:.0f}"
            cv2.putText(
                overlay, info_text,
                (target.x, target.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
            
            # Draw velocity vector if available
            if target.velocity != (0.0, 0.0):
                end_x = aim_x + int(target.velocity[0] * 0.1)
                end_y = aim_y + int(target.velocity[1] * 0.1)
                cv2.arrowedLine(
                    overlay,
                    (aim_x, aim_y),
                    (end_x, end_y),
                    (255, 0, 0),
                    2
                )
        
        # Draw crosshair
        cv2.drawMarker(
            overlay,
            (self.crosshair_x, self.crosshair_y),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2
        )
        
        # Draw stats
        stats_text = [
            f"Targets: {self.stats['targets_detected']}",
            f"Adjustments: {self.stats['aim_adjustments']}",
            f"Mode: {self.aim_mode}",
            f"Zone: {self.aim_zone}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(
                overlay, text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            y_offset += 25
        
        # Blend overlay
        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    def load_detection_model(self, model_path: str):
        """Load target detection model."""
        try:
            # Would load YOLO or similar model
            logger.info(f"Loading detection model from {model_path}")
            # self.detection_model = load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
    
    def load_recoil_pattern(self, weapon: str):
        """Load recoil pattern for weapon."""
        patterns = {
            'ak47': [(0, -3), (0, -5), (0, -7), (0, -8), (1, -8), (2, -7)],
            'm4a4': [(0, -2), (0, -3), (0, -4), (0, -4), (1, -4), (1, -3)],
            'mp9': [(0, -1), (0, -2), (1, -2), (1, -2), (0, -2), (-1, -2)]
        }
        
        self.recoil_pattern = patterns.get(weapon, [])
        self.current_spray_index = 0
        logger.info(f"Loaded recoil pattern for {weapon}")
    
    def set_aim_mode(self, mode: str):
        """Set aim assistance mode."""
        if mode in ['assist', 'lock', 'training']:
            self.aim_mode = mode
            self.locked_target = None
            logger.info(f"Aim mode set to: {mode}")
    
    def toggle_aim(self):
        """Toggle aim assistance on/off."""
        self.aim_enabled = not self.aim_enabled
        logger.info(f"Aim assistance {'enabled' if self.aim_enabled else 'disabled'}")
    
    def calibrate_crosshair(self, x: int, y: int):
        """Calibrate crosshair position."""
        self.crosshair_x = x
        self.crosshair_y = y
        logger.info(f"Crosshair calibrated to ({x}, {y})")
    
    def get_training_data(self) -> Dict[str, Any]:
        """Get data for AI training."""
        return {
            'targets': [
                {
                    'position': target.center,
                    'size': (target.width, target.height),
                    'distance': target.distance,
                    'velocity': target.velocity,
                    'priority': target.priority
                }
                for target in self.targets
            ],
            'crosshair': (self.crosshair_x, self.crosshair_y),
            'stats': self.stats
        }


# Plugin registration
def create_plugin() -> AutoAimPlugin:
    """Create plugin instance."""
    return AutoAimPlugin()