"""
Speed Runner Plugin for Nexus Game AI Framework

Optimizes speedrun routes and tracks split times.
"""

import time
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import structlog

from nexus.core.plugin_base import PluginBase

logger = structlog.get_logger()


@dataclass
class Checkpoint:
    """Represents a speedrun checkpoint."""
    name: str
    x: float
    y: float
    z: float = 0.0
    radius: float = 50.0
    split_name: str = ""
    is_mandatory: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Split:
    """Represents a speedrun split."""
    name: str
    pb_time: float = 0.0  # Personal best
    gold_time: float = 0.0  # Best ever for this split
    current_time: float = 0.0
    comparison_time: float = 0.0  # WR or comparison run
    checkpoint_index: int = -1


@dataclass
class Route:
    """Represents a speedrun route."""
    name: str
    checkpoints: List[Checkpoint]
    category: str = "Any%"
    estimated_time: float = 0.0
    difficulty: str = "Medium"
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunAttempt:
    """Represents a speedrun attempt."""
    start_time: float
    end_time: Optional[float] = None
    splits: List[Split] = field(default_factory=list)
    route: Optional[Route] = None
    completed: bool = False
    reset: bool = False
    final_time: float = 0.0
    category: str = "Any%"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpeedRunnerPlugin(PluginBase):
    """
    Plugin for speedrun optimization and tracking.
    
    Features:
    - Route planning and optimization
    - Split timing and comparison
    - Automatic route learning
    - Ghost/comparison runs
    - Practice mode with save states
    - Statistics and analysis
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Speed Runner"
        self.version = "1.0.0"
        self.description = "Speedrun optimization and tracking"
        
        # Current run state
        self.current_run: Optional[RunAttempt] = None
        self.current_route: Optional[Route] = None
        self.current_checkpoint_index = 0
        self.timer_running = False
        self.run_start_time = 0.0
        
        # Position tracking
        self.current_position = (0.0, 0.0, 0.0)
        self.position_history = deque(maxlen=300)  # 5 seconds at 60fps
        self.velocity = (0.0, 0.0, 0.0)
        
        # Routes and records
        self.routes: Dict[str, Route] = {}
        self.personal_bests: Dict[str, RunAttempt] = {}
        self.world_records: Dict[str, RunAttempt] = {}
        self.run_history: List[RunAttempt] = []
        
        # Ghost/comparison
        self.ghost_run: Optional[RunAttempt] = None
        self.ghost_position = (0.0, 0.0, 0.0)
        self.comparison_mode = 'pb'  # 'pb', 'wr', 'average', 'custom'
        
        # Practice mode
        self.practice_mode = False
        self.save_states: Dict[str, Dict[str, Any]] = {}
        self.current_save_slot = 1
        
        # Route learning
        self.learning_mode = False
        self.learned_checkpoints: List[Checkpoint] = []
        self.checkpoint_detection_threshold = 5.0  # seconds
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'completed_runs': 0,
            'total_time_played': 0.0,
            'best_pace': float('inf'),
            'average_reset_time': 0.0
        }
        
        # Data directory
        self.data_dir = Path.home() / '.nexus' / 'speedrun_data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def on_load(self):
        """Called when plugin is loaded."""
        logger.info(f"Loading {self.name} v{self.version}")
        
        # Load saved data
        self.load_routes()
        self.load_records()
        self.load_statistics()
        
        # Load configuration
        self.practice_mode = self.config.get('practice_mode', False)
        self.comparison_mode = self.config.get('comparison_mode', 'pb')
        
        # Set default route if specified
        default_route = self.config.get('default_route')
        if default_route and default_route in self.routes:
            self.set_route(default_route)
    
    def on_unload(self):
        """Called when plugin is unloaded."""
        # Save current data
        self.save_routes()
        self.save_records()
        self.save_statistics()
        
        logger.info(f"Unloaded {self.name}")
    
    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process game frame."""
        # Update position (would get from game state)
        self.update_position()
        
        # Check checkpoints
        if self.timer_running and self.current_route:
            self.check_checkpoints()
        
        # Update ghost position
        if self.ghost_run:
            self.update_ghost()
        
        # Learning mode
        if self.learning_mode:
            self.detect_checkpoints()
        
        # Draw overlay
        if self.config.get('show_overlay', True):
            frame = self.draw_overlay(frame)
        
        return frame
    
    def start_run(self, route_name: str = None, category: str = "Any%"):
        """Start a new speedrun attempt."""
        if route_name:
            if route_name not in self.routes:
                logger.error(f"Route '{route_name}' not found")
                return
            self.current_route = self.routes[route_name]
        
        if not self.current_route:
            logger.error("No route selected")
            return
        
        # Create new run attempt
        self.current_run = RunAttempt(
            start_time=time.time(),
            route=self.current_route,
            category=category
        )
        
        # Initialize splits
        for checkpoint in self.current_route.checkpoints:
            if checkpoint.split_name:
                split = Split(name=checkpoint.split_name)
                
                # Load comparison times
                if self.comparison_mode == 'pb' and category in self.personal_bests:
                    pb = self.personal_bests[category]
                    # Find matching split
                    for pb_split in pb.splits:
                        if pb_split.name == split.name:
                            split.pb_time = pb_split.current_time
                            split.gold_time = pb_split.gold_time
                            break
                
                self.current_run.splits.append(split)
        
        # Reset state
        self.current_checkpoint_index = 0
        self.timer_running = True
        self.run_start_time = time.time()
        
        # Load ghost run
        self.load_ghost_run()
        
        # Update stats
        self.stats['total_attempts'] += 1
        
        logger.info(f"Started speedrun: {self.current_route.name} ({category})")
    
    def reset_run(self):
        """Reset current run."""
        if not self.current_run:
            return
        
        self.current_run.reset = True
        self.current_run.end_time = time.time()
        
        # Save to history
        self.run_history.append(self.current_run)
        
        # Update average reset time
        if self.current_checkpoint_index > 0:
            reset_time = time.time() - self.run_start_time
            self.stats['average_reset_time'] = (
                (self.stats['average_reset_time'] * (self.stats['total_attempts'] - 1) + reset_time) /
                self.stats['total_attempts']
            )
        
        self.timer_running = False
        self.current_run = None
        
        logger.info("Run reset")
    
    def finish_run(self):
        """Finish current run."""
        if not self.current_run or not self.timer_running:
            return
        
        self.current_run.end_time = time.time()
        self.current_run.final_time = self.current_run.end_time - self.current_run.start_time
        self.current_run.completed = True
        
        # Check for personal best
        category = self.current_run.category
        is_pb = False
        
        if category not in self.personal_bests or \
           self.current_run.final_time < self.personal_bests[category].final_time:
            self.personal_bests[category] = self.current_run
            is_pb = True
            logger.info(f"NEW PERSONAL BEST: {self.format_time(self.current_run.final_time)}")
        
        # Save to history
        self.run_history.append(self.current_run)
        
        # Update stats
        self.stats['completed_runs'] += 1
        self.stats['total_time_played'] += self.current_run.final_time
        
        self.timer_running = False
        
        # Save records
        self.save_records()
        
        return {
            'time': self.current_run.final_time,
            'is_pb': is_pb,
            'splits': self.current_run.splits
        }
    
    def split(self):
        """Manual split."""
        if not self.timer_running or not self.current_run:
            return
        
        current_time = time.time() - self.run_start_time
        
        # Find next split
        for split in self.current_run.splits:
            if split.current_time == 0.0:
                split.current_time = current_time
                
                # Check for gold split
                if split.gold_time == 0.0 or current_time < split.gold_time:
                    split.gold_time = current_time
                    logger.info(f"GOLD SPLIT: {split.name}")
                
                logger.info(f"Split: {split.name} - {self.format_time(current_time)}")
                break
    
    def check_checkpoints(self):
        """Check if player reached checkpoint."""
        if self.current_checkpoint_index >= len(self.current_route.checkpoints):
            # All checkpoints reached
            self.finish_run()
            return
        
        checkpoint = self.current_route.checkpoints[self.current_checkpoint_index]
        
        # Calculate distance to checkpoint
        distance = self.calculate_distance(
            self.current_position,
            (checkpoint.x, checkpoint.y, checkpoint.z)
        )
        
        if distance <= checkpoint.radius:
            # Checkpoint reached
            current_time = time.time() - self.run_start_time
            
            logger.info(f"Checkpoint reached: {checkpoint.name} at {self.format_time(current_time)}")
            
            # Auto-split if configured
            if checkpoint.split_name:
                self.split()
            
            self.current_checkpoint_index += 1
    
    def update_position(self):
        """Update player position and velocity."""
        # In real implementation, would get from game state
        # For now, simulate movement
        if self.timer_running:
            # Simulate progress through route
            progress = (time.time() - self.run_start_time) / 300  # 5 minute run
            if self.current_route and self.current_checkpoint_index < len(self.current_route.checkpoints):
                checkpoint = self.current_route.checkpoints[self.current_checkpoint_index]
                self.current_position = (
                    self.current_position[0] * 0.95 + checkpoint.x * 0.05,
                    self.current_position[1] * 0.95 + checkpoint.y * 0.05,
                    self.current_position[2] * 0.95 + checkpoint.z * 0.05
                )
        
        # Track position history
        self.position_history.append(self.current_position)
        
        # Calculate velocity
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            dt = 1.0 / 60  # Assume 60 FPS
            self.velocity = (
                (self.current_position[0] - prev_pos[0]) / dt,
                (self.current_position[1] - prev_pos[1]) / dt,
                (self.current_position[2] - prev_pos[2]) / dt
            )
    
    def update_ghost(self):
        """Update ghost position."""
        if not self.ghost_run or not self.timer_running:
            return
        
        current_time = time.time() - self.run_start_time
        
        # Find ghost position at current time
        # (Would interpolate from recorded positions)
        # For now, simulate
        if current_time < self.ghost_run.final_time:
            progress = current_time / self.ghost_run.final_time
            if self.current_route and self.current_route.checkpoints:
                # Move ghost through checkpoints
                checkpoint_index = int(progress * len(self.current_route.checkpoints))
                checkpoint_index = min(checkpoint_index, len(self.current_route.checkpoints) - 1)
                checkpoint = self.current_route.checkpoints[checkpoint_index]
                self.ghost_position = (checkpoint.x, checkpoint.y, checkpoint.z)
    
    def detect_checkpoints(self):
        """Detect checkpoints in learning mode."""
        # Detect when player stays in area for threshold time
        if len(self.position_history) < self.checkpoint_detection_threshold * 60:
            return
        
        # Check if position is stable
        recent_positions = list(self.position_history)[-int(self.checkpoint_detection_threshold * 60):]
        
        # Calculate position variance
        positions_array = np.array(recent_positions)
        variance = np.var(positions_array, axis=0)
        
        if np.all(variance < 100):  # Low variance = stationary
            # Create checkpoint
            avg_position = np.mean(positions_array, axis=0)
            checkpoint = Checkpoint(
                name=f"Checkpoint {len(self.learned_checkpoints) + 1}",
                x=avg_position[0],
                y=avg_position[1],
                z=avg_position[2] if len(avg_position) > 2 else 0
            )
            
            # Check if not duplicate
            is_duplicate = False
            for existing in self.learned_checkpoints:
                distance = self.calculate_distance(
                    (existing.x, existing.y, existing.z),
                    (checkpoint.x, checkpoint.y, checkpoint.z)
                )
                if distance < 100:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.learned_checkpoints.append(checkpoint)
                logger.info(f"Learned checkpoint: {checkpoint.name} at ({checkpoint.x:.1f}, {checkpoint.y:.1f}, {checkpoint.z:.1f})")
    
    def create_route(self, name: str, checkpoints: List[Checkpoint], category: str = "Any%") -> Route:
        """Create a new route."""
        route = Route(
            name=name,
            checkpoints=checkpoints,
            category=category
        )
        
        # Estimate time based on distances
        total_distance = 0
        for i in range(1, len(checkpoints)):
            distance = self.calculate_distance(
                (checkpoints[i-1].x, checkpoints[i-1].y, checkpoints[i-1].z),
                (checkpoints[i].x, checkpoints[i].y, checkpoints[i].z)
            )
            total_distance += distance
        
        # Assume average speed
        avg_speed = 10.0  # units per second
        route.estimated_time = total_distance / avg_speed
        
        self.routes[name] = route
        self.save_routes()
        
        logger.info(f"Created route: {name} with {len(checkpoints)} checkpoints")
        return route
    
    def optimize_route(self, route_name: str) -> Route:
        """Optimize route using TSP solver."""
        if route_name not in self.routes:
            return None
        
        route = self.routes[route_name]
        checkpoints = route.checkpoints
        
        if len(checkpoints) <= 2:
            return route  # No optimization needed
        
        # Simple nearest neighbor optimization
        # (In real implementation, would use proper TSP solver)
        optimized = [checkpoints[0]]  # Start checkpoint
        remaining = checkpoints[1:-1]  # Middle checkpoints
        current = checkpoints[0]
        
        while remaining:
            # Find nearest checkpoint
            nearest = None
            min_distance = float('inf')
            
            for checkpoint in remaining:
                distance = self.calculate_distance(
                    (current.x, current.y, current.z),
                    (checkpoint.x, checkpoint.y, checkpoint.z)
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest = checkpoint
            
            optimized.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        optimized.append(checkpoints[-1])  # End checkpoint
        
        # Create optimized route
        optimized_route = Route(
            name=f"{route_name}_optimized",
            checkpoints=optimized,
            category=route.category
        )
        
        self.routes[optimized_route.name] = optimized_route
        logger.info(f"Optimized route: {route_name}")
        
        return optimized_route
    
    def save_state(self, slot: int = None):
        """Save current game state (practice mode)."""
        if not self.practice_mode:
            logger.warning("Save states only available in practice mode")
            return
        
        slot = slot or self.current_save_slot
        
        self.save_states[str(slot)] = {
            'position': self.current_position,
            'checkpoint_index': self.current_checkpoint_index,
            'time': time.time() - self.run_start_time if self.timer_running else 0,
            'splits': self.current_run.splits if self.current_run else []
        }
        
        logger.info(f"State saved to slot {slot}")
    
    def load_state(self, slot: int = None):
        """Load saved game state (practice mode)."""
        if not self.practice_mode:
            logger.warning("Save states only available in practice mode")
            return
        
        slot = slot or self.current_save_slot
        slot_key = str(slot)
        
        if slot_key not in self.save_states:
            logger.warning(f"No save state in slot {slot}")
            return
        
        state = self.save_states[slot_key]
        self.current_position = state['position']
        self.current_checkpoint_index = state['checkpoint_index']
        
        if self.current_run:
            self.current_run.splits = state['splits']
        
        logger.info(f"State loaded from slot {slot}")
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw speedrun overlay."""
        import cv2
        overlay = frame.copy()
        
        # Draw timer
        if self.timer_running:
            current_time = time.time() - self.run_start_time
            timer_text = self.format_time(current_time)
            
            # Color based on pace
            color = (0, 255, 0)  # Green
            if self.current_run and self.comparison_mode == 'pb' and self.current_run.category in self.personal_bests:
                pb = self.personal_bests[self.current_run.category]
                if current_time > pb.final_time * (self.current_checkpoint_index + 1) / len(self.current_route.checkpoints):
                    color = (0, 0, 255)  # Red - behind pace
            
            cv2.putText(
                overlay, timer_text,
                (frame.shape[1] - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5, color, 3
            )
        
        # Draw splits
        if self.current_run:
            y_offset = 100
            for split in self.current_run.splits[-5:]:  # Show last 5 splits
                split_text = f"{split.name}: "
                
                if split.current_time > 0:
                    split_text += self.format_time(split.current_time)
                    
                    # Show comparison
                    if split.pb_time > 0:
                        diff = split.current_time - split.pb_time
                        diff_text = f" ({'+' if diff > 0 else ''}{diff:.2f})"
                        split_text += diff_text
                else:
                    split_text += "--:--"
                
                cv2.putText(
                    overlay, split_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1
                )
                y_offset += 30
        
        # Draw route info
        if self.current_route:
            route_text = f"Route: {self.current_route.name}"
            checkpoint_text = f"Checkpoint: {self.current_checkpoint_index + 1}/{len(self.current_route.checkpoints)}"
            
            cv2.putText(
                overlay, route_text,
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
            
            cv2.putText(
                overlay, checkpoint_text,
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )
        
        # Draw speed
        speed = math.sqrt(sum(v*v for v in self.velocity))
        speed_text = f"Speed: {speed:.1f}"
        cv2.putText(
            overlay, speed_text,
            (frame.shape[1] - 150, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1
        )
        
        return overlay
    
    def calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between positions."""
        return math.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2 +
            (pos1[2] - pos2[2])**2
        )
    
    def format_time(self, seconds: float) -> str:
        """Format time as MM:SS.ms."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"
    
    def set_route(self, route_name: str):
        """Set active route."""
        if route_name in self.routes:
            self.current_route = self.routes[route_name]
            logger.info(f"Set route: {route_name}")
    
    def load_ghost_run(self):
        """Load ghost run for comparison."""
        if not self.current_run:
            return
        
        category = self.current_run.category
        
        if self.comparison_mode == 'pb' and category in self.personal_bests:
            self.ghost_run = self.personal_bests[category]
        elif self.comparison_mode == 'wr' and category in self.world_records:
            self.ghost_run = self.world_records[category]
    
    def load_routes(self):
        """Load saved routes."""
        routes_file = self.data_dir / 'routes.json'
        if routes_file.exists():
            with open(routes_file, 'r') as f:
                data = json.load(f)
                for name, route_data in data.items():
                    checkpoints = [
                        Checkpoint(**cp) for cp in route_data['checkpoints']
                    ]
                    self.routes[name] = Route(
                        name=name,
                        checkpoints=checkpoints,
                        category=route_data.get('category', 'Any%')
                    )
    
    def save_routes(self):
        """Save routes to disk."""
        routes_file = self.data_dir / 'routes.json'
        data = {}
        
        for name, route in self.routes.items():
            data[name] = {
                'checkpoints': [asdict(cp) for cp in route.checkpoints],
                'category': route.category,
                'estimated_time': route.estimated_time
            }
        
        with open(routes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_records(self):
        """Load personal bests and records."""
        records_file = self.data_dir / 'records.json'
        if records_file.exists():
            with open(records_file, 'r') as f:
                data = json.load(f)
                # Simplified loading for example
                # In real implementation would properly deserialize
    
    def save_records(self):
        """Save records to disk."""
        records_file = self.data_dir / 'records.json'
        data = {
            'personal_bests': {},
            'stats': self.stats
        }
        
        # Simplified saving for example
        with open(records_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_statistics(self):
        """Load statistics."""
        # Would load from file
        pass
    
    def save_statistics(self):
        """Save statistics."""
        # Would save to file
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get speedrun statistics."""
        return {
            'stats': self.stats,
            'total_routes': len(self.routes),
            'pb_count': len(self.personal_bests),
            'attempts_today': sum(
                1 for run in self.run_history
                if datetime.fromtimestamp(run.start_time).date() == datetime.now().date()
            )
        }


# Plugin registration
def create_plugin() -> SpeedRunnerPlugin:
    """Create plugin instance."""
    return SpeedRunnerPlugin()