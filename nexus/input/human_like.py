"""Human-like input patterns and anti-detection measures"""

import random
import time
import math
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class MousePattern:
    """Human mouse movement pattern"""
    speed: float  # pixels per second
    acceleration: float
    jitter: float
    curve_intensity: float
    pause_probability: float
    overshoot_probability: float


class HumanLikeInput:
    """Generate human-like input patterns for anti-detection"""
    
    # Typing patterns based on skill level
    TYPING_PATTERNS = {
        "beginner": {
            "wpm": 30,
            "accuracy": 0.92,
            "pause_probability": 0.15,
            "burst_probability": 0.05,
            "fatigue_factor": 1.2
        },
        "intermediate": {
            "wpm": 60,
            "accuracy": 0.96,
            "pause_probability": 0.08,
            "burst_probability": 0.15,
            "fatigue_factor": 1.1
        },
        "advanced": {
            "wpm": 90,
            "accuracy": 0.98,
            "pause_probability": 0.05,
            "burst_probability": 0.25,
            "fatigue_factor": 1.05
        }
    }
    
    # Mouse movement patterns
    MOUSE_PATTERNS = {
        "relaxed": MousePattern(
            speed=500, acceleration=1.2, jitter=2.0,
            curve_intensity=0.3, pause_probability=0.1,
            overshoot_probability=0.05
        ),
        "normal": MousePattern(
            speed=800, acceleration=1.5, jitter=1.5,
            curve_intensity=0.2, pause_probability=0.05,
            overshoot_probability=0.1
        ),
        "fast": MousePattern(
            speed=1200, acceleration=2.0, jitter=1.0,
            curve_intensity=0.1, pause_probability=0.02,
            overshoot_probability=0.15
        ),
        "precise": MousePattern(
            speed=400, acceleration=1.1, jitter=0.5,
            curve_intensity=0.05, pause_probability=0.15,
            overshoot_probability=0.02
        )
    }
    
    def __init__(self, skill_level: str = "intermediate", mouse_style: str = "normal"):
        self.typing_pattern = self.TYPING_PATTERNS.get(skill_level, self.TYPING_PATTERNS["intermediate"])
        self.mouse_pattern = self.MOUSE_PATTERNS.get(mouse_style, self.MOUSE_PATTERNS["normal"])
        
        # Session variables for realistic behavior
        self.session_start = time.time()
        self.total_keystrokes = 0
        self.total_mouse_distance = 0
        self.last_action_time = time.time()
        
        # Fatigue simulation
        self.fatigue_level = 0.0
        
    def get_typing_delay(self, char: str, prev_char: Optional[str] = None) -> float:
        """Calculate realistic typing delay for a character"""
        
        # Base delay from WPM
        base_delay = 60.0 / (self.typing_pattern["wpm"] * 5)  # 5 chars per word average
        
        # Add variation
        delay = random.gauss(base_delay, base_delay * 0.2)
        
        # Adjust for character difficulty
        if char in "qwaszx":  # Easy left hand
            delay *= 0.9
        elif char in "plo;/.":  # Harder right hand
            delay *= 1.1
        
        # Adjust for character transitions
        if prev_char:
            if self._same_hand(prev_char, char):
                delay *= 1.05  # Slightly slower for same hand
            if self._same_finger(prev_char, char):
                delay *= 1.2  # Much slower for same finger
        
        # Add pauses
        if random.random() < self.typing_pattern["pause_probability"]:
            delay += random.uniform(0.2, 0.8)  # Thinking pause
        
        # Burst typing (fast sequences)
        if random.random() < self.typing_pattern["burst_probability"]:
            delay *= 0.7
        
        # Apply fatigue
        delay *= (1 + self.fatigue_level * self.typing_pattern["fatigue_factor"])
        
        # Update fatigue
        self._update_fatigue()
        
        # Ensure minimum delay
        return max(delay, 0.03)
    
    def generate_mouse_path(self, start: Tuple[int, int], end: Tuple[int, int],
                          duration: Optional[float] = None) -> List[Tuple[float, float, float]]:
        """Generate human-like mouse movement path"""
        
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        if duration is None:
            # Calculate duration based on Fitts' Law
            duration = self._fitts_law_duration(distance)
        
        points = []
        pattern = self.mouse_pattern
        
        # Decide on movement type
        if distance < 50:
            # Short, direct movement
            points = self._generate_direct_path(start, end, duration)
        elif distance < 200:
            # Medium movement with slight curve
            points = self._generate_curved_path(start, end, duration, intensity=0.1)
        else:
            # Long movement with curve and possible overshoot
            points = self._generate_complex_path(start, end, duration)
        
        # Add jitter to all points
        points = self._add_jitter(points, pattern.jitter)
        
        # Add micro-pauses
        if random.random() < pattern.pause_probability:
            points = self._add_pause(points, random.uniform(0.05, 0.15))
        
        # Update session stats
        self.total_mouse_distance += distance
        self._update_fatigue()
        
        return points
    
    def generate_click_pattern(self) -> Tuple[float, float]:
        """Generate human-like click timing (down duration, up delay)"""
        
        # Normal click duration varies between 50-150ms
        down_duration = random.gauss(0.08, 0.02)
        down_duration = max(0.03, min(0.15, down_duration))
        
        # Sometimes double-click accidentally
        if random.random() < 0.02:  # 2% chance
            # Accidental double click
            up_delay = random.uniform(0.05, 0.15)
        else:
            up_delay = 0
        
        return down_duration, up_delay
    
    def generate_scroll_pattern(self, direction: int, amount: int) -> List[Tuple[int, float]]:
        """Generate human-like scroll pattern"""
        
        scrolls = []
        remaining = abs(amount)
        
        while remaining > 0:
            # Humans don't scroll uniformly
            if remaining > 5:
                # Large scrolls with variation
                scroll_amount = random.randint(3, min(7, remaining))
            else:
                scroll_amount = 1
            
            # Delay between scrolls
            if len(scrolls) > 0:
                delay = random.gauss(0.05, 0.02)
                delay = max(0.02, min(0.1, delay))
            else:
                delay = 0
            
            scrolls.append((scroll_amount * direction, delay))
            remaining -= scroll_amount
        
        return scrolls
    
    def should_make_mistake(self) -> bool:
        """Determine if a typing mistake should occur"""
        
        # Base accuracy
        if random.random() > self.typing_pattern["accuracy"]:
            return True
        
        # Fatigue increases mistakes
        if random.random() < self.fatigue_level * 0.1:
            return True
        
        return False
    
    def generate_mistake_correction(self, intended: str) -> List[str]:
        """Generate a mistake and correction sequence"""
        
        mistakes = []
        
        # Common mistake types
        mistake_type = random.choice(["adjacent", "swap", "double", "miss"])
        
        if mistake_type == "adjacent":
            # Hit adjacent key
            adjacent_keys = self._get_adjacent_keys(intended)
            if adjacent_keys:
                wrong_key = random.choice(adjacent_keys)
                mistakes = [wrong_key, '\b', intended]  # Wrong, backspace, correct
        
        elif mistake_type == "swap":
            # Swap with next character (caught immediately)
            mistakes = [intended, '\b', intended]
        
        elif mistake_type == "double":
            # Double tap
            mistakes = [intended, intended, '\b']
        
        elif mistake_type == "miss":
            # Miss the key (no input), realize, then correct
            mistakes = ['', intended]
        
        return mistakes
    
    def _fitts_law_duration(self, distance: float, target_size: float = 20) -> float:
        """Calculate movement duration using Fitts' Law"""
        
        # Fitts' Law: T = a + b * log2(D/W + 1)
        # Where T = time, D = distance, W = target width
        
        a = 0.1  # Start/stop time
        b = 0.1  # Speed factor
        
        # Calculate base duration
        duration = a + b * math.log2(distance / target_size + 1)
        
        # Adjust for mouse pattern speed
        speed_factor = self.mouse_pattern.speed / 800  # Normalize to "normal" speed
        duration /= speed_factor
        
        # Add variation
        duration *= random.uniform(0.9, 1.1)
        
        return max(duration, 0.1)
    
    def _generate_direct_path(self, start: Tuple[int, int], end: Tuple[int, int],
                             duration: float) -> List[Tuple[float, float, float]]:
        """Generate a mostly direct path with slight variation"""
        
        points = []
        steps = max(int(duration * 60), 10)
        
        for i in range(steps):
            t = i / steps
            
            # Linear interpolation with slight sine wave
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            
            # Add very slight curve
            perpendicular_x = -(end[1] - start[1]) / (math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) + 0.001)
            perpendicular_y = (end[0] - start[0]) / (math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) + 0.001)
            
            curve = math.sin(t * math.pi) * 5  # Small curve
            x += perpendicular_x * curve
            y += perpendicular_y * curve
            
            delay = duration / steps
            points.append((x, y, delay))
        
        return points
    
    def _generate_curved_path(self, start: Tuple[int, int], end: Tuple[int, int],
                            duration: float, intensity: float = 0.2) -> List[Tuple[float, float, float]]:
        """Generate a curved path using bezier curves"""
        
        points = []
        steps = max(int(duration * 60), 20)
        
        # Generate control points
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Offset control points perpendicular to the line
        perpendicular_x = -(end[1] - start[1])
        perpendicular_y = end[0] - start[0]
        
        # Normalize and scale
        length = math.sqrt(perpendicular_x**2 + perpendicular_y**2) + 0.001
        perpendicular_x = perpendicular_x / length * intensity * length
        perpendicular_y = perpendicular_y / length * intensity * length
        
        # Control point with randomization
        control_x = mid_x + perpendicular_x * random.uniform(-1, 1)
        control_y = mid_y + perpendicular_y * random.uniform(-1, 1)
        
        for i in range(steps):
            t = i / steps
            
            # Quadratic bezier
            x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]
            
            # Variable speed (slower at start/end)
            speed_factor = 1 - abs(2 * t - 1) ** 2
            delay = (duration / steps) * (0.7 + 0.6 * speed_factor)
            
            points.append((x, y, delay))
        
        return points
    
    def _generate_complex_path(self, start: Tuple[int, int], end: Tuple[int, int],
                              duration: float) -> List[Tuple[float, float, float]]:
        """Generate complex path with possible overshoot"""
        
        pattern = self.mouse_pattern
        
        # Possibly overshoot target
        if random.random() < pattern.overshoot_probability:
            # Calculate overshoot point
            overshoot_distance = random.uniform(10, 30)
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            overshoot_x = end[0] + math.cos(angle) * overshoot_distance
            overshoot_y = end[1] + math.sin(angle) * overshoot_distance
            
            # Generate path to overshoot, then back
            duration1 = duration * 0.8
            duration2 = duration * 0.2
            
            path1 = self._generate_curved_path(start, (overshoot_x, overshoot_y), duration1, 0.15)
            path2 = self._generate_direct_path((overshoot_x, overshoot_y), end, duration2)
            
            return path1 + path2
        else:
            # Normal curved path
            return self._generate_curved_path(start, end, duration, pattern.curve_intensity)
    
    def _add_jitter(self, points: List[Tuple[float, float, float]],
                   jitter_amount: float) -> List[Tuple[float, float, float]]:
        """Add random jitter to path points"""
        
        jittered = []
        for i, (x, y, delay) in enumerate(points):
            # Don't jitter start and end points
            if 0 < i < len(points) - 1:
                x += random.gauss(0, jitter_amount)
                y += random.gauss(0, jitter_amount)
            jittered.append((x, y, delay))
        
        return jittered
    
    def _add_pause(self, points: List[Tuple[float, float, float]],
                  pause_duration: float) -> List[Tuple[float, float, float]]:
        """Add a pause in the middle of movement"""
        
        if len(points) < 4:
            return points
        
        # Add pause at random point (not start/end)
        pause_index = random.randint(len(points)//3, 2*len(points)//3)
        
        modified = points[:pause_index]
        x, y, delay = points[pause_index]
        modified.append((x, y, delay + pause_duration))
        modified.extend(points[pause_index+1:])
        
        return modified
    
    def _update_fatigue(self):
        """Update fatigue level based on session duration"""
        
        session_duration = time.time() - self.session_start
        
        # Fatigue increases over time
        if session_duration < 300:  # First 5 minutes
            self.fatigue_level = 0.0
        elif session_duration < 1200:  # 5-20 minutes
            self.fatigue_level = (session_duration - 300) / 900 * 0.1
        elif session_duration < 3600:  # 20-60 minutes
            self.fatigue_level = 0.1 + (session_duration - 1200) / 2400 * 0.2
        else:  # Over 1 hour
            self.fatigue_level = min(0.5, 0.3 + (session_duration - 3600) / 7200 * 0.2)
    
    def _same_hand(self, char1: str, char2: str) -> bool:
        """Check if two characters are typed with same hand"""
        
        left_hand = "qwertasdfgzxcvb12345"
        right_hand = "yuiophjklnm67890"
        
        return (char1.lower() in left_hand and char2.lower() in left_hand) or \
               (char1.lower() in right_hand and char2.lower() in right_hand)
    
    def _same_finger(self, char1: str, char2: str) -> bool:
        """Check if two characters use the same finger"""
        
        finger_map = {
            'q': 'left_pinky', 'a': 'left_pinky', 'z': 'left_pinky',
            'w': 'left_ring', 's': 'left_ring', 'x': 'left_ring',
            'e': 'left_middle', 'd': 'left_middle', 'c': 'left_middle',
            'r': 'left_index', 'f': 'left_index', 'v': 'left_index',
            't': 'left_index', 'g': 'left_index', 'b': 'left_index',
            'y': 'right_index', 'h': 'right_index', 'n': 'right_index',
            'u': 'right_index', 'j': 'right_index', 'm': 'right_index',
            'i': 'right_middle', 'k': 'right_middle',
            'o': 'right_ring', 'l': 'right_ring',
            'p': 'right_pinky',
        }
        
        return finger_map.get(char1.lower()) == finger_map.get(char2.lower())
    
    def _get_adjacent_keys(self, char: str) -> List[str]:
        """Get keys adjacent to the given character on keyboard"""
        
        keyboard_layout = [
            "1234567890",
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm"
        ]
        
        adjacent = []
        
        for row_idx, row in enumerate(keyboard_layout):
            if char.lower() in row:
                col_idx = row.index(char.lower())
                
                # Same row adjacents
                if col_idx > 0:
                    adjacent.append(row[col_idx - 1])
                if col_idx < len(row) - 1:
                    adjacent.append(row[col_idx + 1])
                
                # Adjacent rows
                if row_idx > 0:
                    prev_row = keyboard_layout[row_idx - 1]
                    if col_idx < len(prev_row):
                        adjacent.append(prev_row[col_idx])
                
                if row_idx < len(keyboard_layout) - 1:
                    next_row = keyboard_layout[row_idx + 1]
                    if col_idx < len(next_row):
                        adjacent.append(next_row[col_idx])
                
                break
        
        return adjacent