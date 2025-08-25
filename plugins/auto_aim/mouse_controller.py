"""
Mouse control module with humanization for Auto-Aim Assistant.

IMPORTANT: For AI training purposes only.
"""

import time
import math
import random
import numpy as np
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass
from collections import deque
import threading
from queue import Queue

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable fail-safe
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import win32api
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


@dataclass
class MouseMovement:
    """Represents a mouse movement command."""
    dx: float
    dy: float
    duration: float = 0.0
    curve_type: str = "linear"  # linear, bezier, sine
    humanize: bool = True


class HumanMouseController:
    """Controls mouse with human-like movements."""
    
    def __init__(self, training_mode: bool = True):
        """
        Initialize mouse controller.
        
        Args:
            training_mode: If True, only logs movements without executing
        """
        self.training_mode = training_mode
        
        # Movement settings
        self.sensitivity = 1.0
        self.acceleration = 1.0
        self.smoothing = 5.0
        
        # Humanization settings
        self.reaction_time = 0.15  # Average human reaction time
        self.overshoot_amount = 0.1  # How much to overshoot target
        self.correction_speed = 0.8  # Speed of overshoot correction
        self.shake_amount = 2.0  # Natural hand shake
        self.fatigue_factor = 0.0  # Increases over time
        
        # Movement history
        self.movement_history = deque(maxlen=100)
        self.movement_queue = Queue()
        
        # Threading for smooth movement
        self.is_running = False
        self.movement_thread = None
        
        # Movement curves
        self.curve_functions = {
            'linear': self._linear_curve,
            'bezier': self._bezier_curve,
            'sine': self._sine_curve,
            'ease_in_out': self._ease_in_out_curve
        }
        
        # Start movement thread if not in training mode
        if not training_mode:
            self._start_movement_thread()
    
    def move(self, dx: float, dy: float, duration: float = 0.0, 
             humanize: bool = True, immediate: bool = False):
        """Move mouse by relative amount."""
        if self.training_mode:
            # Just log the movement
            self._log_movement(dx, dy, duration)
            return
        
        movement = MouseMovement(
            dx=dx * self.sensitivity,
            dy=dy * self.sensitivity,
            duration=duration,
            humanize=humanize
        )
        
        if immediate:
            self._execute_movement(movement)
        else:
            self.movement_queue.put(movement)
    
    def move_to(self, x: int, y: int, duration: float = 0.0):
        """Move mouse to absolute position."""
        if self.training_mode:
            self._log_movement(x, y, duration, absolute=True)
            return
        
        current_x, current_y = self.get_position()
        dx = x - current_x
        dy = y - current_y
        
        self.move(dx, dy, duration)
    
    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        if HAS_PYAUTOGUI:
            return pyautogui.position()
        elif HAS_WIN32:
            return win32api.GetCursorPos()
        else:
            return (0, 0)
    
    def click(self, button: str = 'left', clicks: int = 1):
        """Perform mouse click."""
        if self.training_mode:
            self._log_click(button, clicks)
            return
        
        if HAS_PYAUTOGUI:
            pyautogui.click(button=button, clicks=clicks)
        elif HAS_WIN32:
            if button == 'left':
                for _ in range(clicks):
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    
    def _execute_movement(self, movement: MouseMovement):
        """Execute a single movement."""
        if movement.humanize:
            # Add reaction time
            time.sleep(random.uniform(
                self.reaction_time * 0.8,
                self.reaction_time * 1.2
            ))
            
            # Humanize the movement
            movement = self._humanize_movement(movement)
        
        if movement.duration > 0:
            # Smooth movement over time
            self._smooth_move(movement)
        else:
            # Instant movement
            self._instant_move(movement.dx, movement.dy)
        
        # Record movement
        self.movement_history.append((
            movement.dx, movement.dy, time.time()
        ))
    
    def _humanize_movement(self, movement: MouseMovement) -> MouseMovement:
        """Add human-like characteristics to movement."""
        # Add overshoot
        if random.random() < 0.3:  # 30% chance of overshoot
            overshoot_x = movement.dx * self.overshoot_amount * random.uniform(-1, 1)
            overshoot_y = movement.dy * self.overshoot_amount * random.uniform(-1, 1)
            
            # Create overshoot movement
            self.movement_queue.put(MouseMovement(
                dx=overshoot_x,
                dy=overshoot_y,
                duration=0.05,
                humanize=False
            ))
            
            # Create correction movement
            self.movement_queue.put(MouseMovement(
                dx=-overshoot_x * self.correction_speed,
                dy=-overshoot_y * self.correction_speed,
                duration=0.05,
                humanize=False
            ))
        
        # Add shake
        shake_x = random.gauss(0, self.shake_amount)
        shake_y = random.gauss(0, self.shake_amount)
        
        movement.dx += shake_x
        movement.dy += shake_y
        
        # Apply fatigue
        if self.fatigue_factor > 0:
            movement.dx *= (1 - self.fatigue_factor * 0.1)
            movement.dy *= (1 - self.fatigue_factor * 0.1)
        
        return movement
    
    def _smooth_move(self, movement: MouseMovement):
        """Execute smooth movement over duration."""
        steps = max(int(movement.duration * 60), 1)  # 60 FPS
        
        # Get curve function
        curve_func = self.curve_functions.get(
            movement.curve_type, 
            self._linear_curve
        )
        
        # Generate path points
        path_points = curve_func(movement.dx, movement.dy, steps)
        
        # Execute movement along path
        for i, (x, y) in enumerate(path_points):
            if i > 0:
                prev_x, prev_y = path_points[i-1]
                step_dx = x - prev_x
                step_dy = y - prev_y
                
                self._instant_move(step_dx, step_dy)
                time.sleep(movement.duration / steps)
    
    def _instant_move(self, dx: float, dy: float):
        """Execute instant relative mouse movement."""
        if HAS_WIN32:
            # Use raw input for better precision
            win32api.mouse_event(
                win32con.MOUSEEVENTF_MOVE,
                int(dx), int(dy), 0, 0
            )
        elif HAS_PYAUTOGUI:
            pyautogui.moveRel(int(dx), int(dy), duration=0)
    
    def _linear_curve(self, dx: float, dy: float, steps: int) -> List[Tuple[float, float]]:
        """Generate linear movement path."""
        points = []
        for i in range(steps):
            t = (i + 1) / steps
            points.append((dx * t, dy * t))
        return points
    
    def _bezier_curve(self, dx: float, dy: float, steps: int) -> List[Tuple[float, float]]:
        """Generate bezier curve movement path."""
        # Control points for bezier curve
        p0 = (0, 0)
        p1 = (dx * 0.3, dy * 0.7)  # First control point
        p2 = (dx * 0.7, dy * 0.3)  # Second control point
        p3 = (dx, dy)
        
        points = []
        for i in range(steps):
            t = (i + 1) / steps
            
            # Cubic bezier formula
            x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + \
                3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
            y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + \
                3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
            
            points.append((x, y))
        
        return points
    
    def _sine_curve(self, dx: float, dy: float, steps: int) -> List[Tuple[float, float]]:
        """Generate sine wave movement path."""
        points = []
        amplitude = min(abs(dx), abs(dy)) * 0.1  # 10% amplitude
        
        for i in range(steps):
            t = (i + 1) / steps
            
            # Linear progress
            x = dx * t
            y = dy * t
            
            # Add sine wave perpendicular to movement
            angle = math.atan2(dy, dx)
            perpendicular = angle + math.pi / 2
            
            sine_offset = amplitude * math.sin(t * math.pi * 2)
            x += math.cos(perpendicular) * sine_offset
            y += math.sin(perpendicular) * sine_offset
            
            points.append((x, y))
        
        return points
    
    def _ease_in_out_curve(self, dx: float, dy: float, steps: int) -> List[Tuple[float, float]]:
        """Generate ease-in-out movement path."""
        points = []
        
        for i in range(steps):
            t = (i + 1) / steps
            
            # Ease-in-out formula
            if t < 0.5:
                progress = 2 * t * t
            else:
                progress = 1 - pow(-2 * t + 2, 2) / 2
            
            points.append((dx * progress, dy * progress))
        
        return points
    
    def _start_movement_thread(self):
        """Start background thread for movement processing."""
        self.is_running = True
        self.movement_thread = threading.Thread(
            target=self._movement_loop,
            daemon=True
        )
        self.movement_thread.start()
    
    def _movement_loop(self):
        """Background loop for processing movement queue."""
        while self.is_running:
            try:
                movement = self.movement_queue.get(timeout=0.1)
                self._execute_movement(movement)
            except:
                continue
    
    def _log_movement(self, dx: float, dy: float, duration: float, absolute: bool = False):
        """Log movement for training mode."""
        action = "move_to" if absolute else "move"
        print(f"[TRAINING] {action}: dx={dx:.2f}, dy={dy:.2f}, duration={duration:.3f}")
    
    def _log_click(self, button: str, clicks: int):
        """Log click for training mode."""
        print(f"[TRAINING] click: button={button}, clicks={clicks}")
    
    def set_sensitivity(self, sensitivity: float):
        """Set mouse sensitivity multiplier."""
        self.sensitivity = max(0.1, min(10.0, sensitivity))
    
    def set_humanization(self, reaction_time: float, overshoot: float, shake: float):
        """Configure humanization parameters."""
        self.reaction_time = reaction_time
        self.overshoot_amount = overshoot
        self.shake_amount = shake
    
    def update_fatigue(self, delta: float):
        """Update fatigue factor over time."""
        self.fatigue_factor = min(1.0, self.fatigue_factor + delta)
    
    def reset_fatigue(self):
        """Reset fatigue factor."""
        self.fatigue_factor = 0.0
    
    def stop(self):
        """Stop mouse controller."""
        self.is_running = False
        if self.movement_thread:
            self.movement_thread.join(timeout=1.0)