"""
Game State Logger Plugin for Nexus Game AI Framework

Logs game states, events, and actions for analysis and training.
"""

import json
import time
import csv
import gzip
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from datetime import datetime
import numpy as np
import structlog

from nexus.core.plugin_base import PluginBase

logger = structlog.get_logger()


@dataclass
class GameEvent:
    """Represents a game event."""
    timestamp: float
    event_type: str
    data: Dict[str, Any]
    frame_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    """Represents a complete game state."""
    timestamp: float
    frame_number: int
    player_state: Dict[str, Any]
    world_state: Dict[str, Any]
    ui_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    """Records player actions."""
    timestamp: float
    action_type: str
    inputs: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    success: bool = True


class GameStateLoggerPlugin(PluginBase):
    """
    Plugin for comprehensive game state logging.
    
    Features:
    - Event logging with categorization
    - State snapshots at intervals
    - Action recording and replay
    - Data export in multiple formats
    - Real-time analysis and statistics
    - Training data generation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Game State Logger"
        self.version = "1.0.0"
        self.description = "Comprehensive game state and event logging"
        
        # Logging configuration
        self.log_events = True
        self.log_states = True
        self.log_actions = True
        self.log_frames = False  # Frame logging (storage intensive)
        
        # Storage
        self.events_buffer = deque(maxlen=10000)
        self.states_buffer = deque(maxlen=1000)
        self.actions_buffer = deque(maxlen=5000)
        self.frames_buffer = deque(maxlen=300)  # 5 seconds at 60fps
        
        # Current session
        self.session_id = None
        self.session_start_time = None
        self.frame_count = 0
        self.event_count = 0
        
        # State tracking
        self.current_state = {}
        self.previous_state = {}
        self.state_snapshot_interval = 1.0  # seconds
        self.last_snapshot_time = 0
        
        # Event categorization
        self.event_categories = {
            'combat': ['damage', 'kill', 'death', 'heal'],
            'movement': ['jump', 'dash', 'teleport', 'fall'],
            'interaction': ['pickup', 'use', 'talk', 'buy', 'sell'],
            'progression': ['level_up', 'achievement', 'quest_complete'],
            'system': ['save', 'load', 'pause', 'menu']
        }
        
        # Statistics
        self.event_stats = defaultdict(int)
        self.action_stats = defaultdict(int)
        self.state_transitions = defaultdict(lambda: defaultdict(int))
        
        # Export settings
        self.export_format = 'json'  # 'json', 'csv', 'binary', 'sql'
        self.compression = True
        self.export_interval = 60.0  # Auto-export interval
        self.last_export_time = 0
        
        # Data directory
        self.data_dir = Path.home() / '.nexus' / 'game_logs'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis callbacks
        self.event_analyzers: List[Callable] = []
        self.state_analyzers: List[Callable] = []
        
        # Replay system
        self.replay_mode = False
        self.replay_data = None
        self.replay_index = 0
        self.replay_speed = 1.0
        
    def on_load(self):
        """Called when plugin is loaded."""
        logger.info(f"Loading {self.name} v{self.version}")
        
        # Start new session
        self.start_session()
        
        # Load configuration
        self.log_events = self.config.get('log_events', True)
        self.log_states = self.config.get('log_states', True)
        self.log_actions = self.config.get('log_actions', True)
        self.log_frames = self.config.get('log_frames', False)
        self.export_format = self.config.get('export_format', 'json')
        self.compression = self.config.get('compression', True)
        
        # Register default analyzers
        self.register_default_analyzers()
        
    def on_unload(self):
        """Called when plugin is unloaded."""
        # Export remaining data
        self.export_session()
        
        # Generate session summary
        self.generate_summary()
        
        logger.info(f"Unloaded {self.name}")
    
    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process game frame."""
        self.frame_count += 1
        
        # Log frame if enabled
        if self.log_frames:
            self.log_frame(frame)
        
        # Take state snapshot at intervals
        current_time = time.time()
        if current_time - self.last_snapshot_time >= self.state_snapshot_interval:
            self.take_state_snapshot()
            self.last_snapshot_time = current_time
        
        # Auto-export at intervals
        if current_time - self.last_export_time >= self.export_interval:
            self.export_buffered_data()
            self.last_export_time = current_time
        
        # Replay mode
        if self.replay_mode:
            frame = self.apply_replay_overlay(frame)
        
        # Draw overlay if enabled
        if self.config.get('show_overlay', False):
            frame = self.draw_overlay(frame)
        
        return frame
    
    def log_event(self, event_type: str, data: Dict[str, Any] = None, 
                  metadata: Dict[str, Any] = None):
        """Log a game event."""
        if not self.log_events:
            return
        
        event = GameEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data or {},
            frame_number=self.frame_count,
            metadata=metadata or {}
        )
        
        self.events_buffer.append(event)
        self.event_count += 1
        
        # Update statistics
        self.event_stats[event_type] += 1
        
        # Run analyzers
        for analyzer in self.event_analyzers:
            try:
                analyzer(event)
            except Exception as e:
                logger.error(f"Event analyzer error: {e}")
        
        logger.debug(f"Logged event: {event_type}")
    
    def log_state(self, player_state: Dict[str, Any] = None,
                  world_state: Dict[str, Any] = None,
                  ui_state: Dict[str, Any] = None):
        """Log current game state."""
        if not self.log_states:
            return
        
        state = GameState(
            timestamp=time.time(),
            frame_number=self.frame_count,
            player_state=player_state or {},
            world_state=world_state or {},
            ui_state=ui_state or {}
        )
        
        self.states_buffer.append(state)
        
        # Track state transitions
        self.track_state_transition(state)
        
        # Update current state
        self.previous_state = self.current_state.copy()
        self.current_state = {
            'player': player_state,
            'world': world_state,
            'ui': ui_state
        }
        
        # Run analyzers
        for analyzer in self.state_analyzers:
            try:
                analyzer(state)
            except Exception as e:
                logger.error(f"State analyzer error: {e}")
    
    def log_action(self, action_type: str, inputs: Dict[str, Any],
                   result: Dict[str, Any] = None, success: bool = True):
        """Log player action."""
        if not self.log_actions:
            return
        
        action = ActionRecord(
            timestamp=time.time(),
            action_type=action_type,
            inputs=inputs,
            result=result,
            success=success
        )
        
        self.actions_buffer.append(action)
        
        # Update statistics
        self.action_stats[action_type] += 1
        
        logger.debug(f"Logged action: {action_type}")
    
    def log_frame(self, frame: np.ndarray):
        """Log game frame for replay."""
        # Compress frame
        import cv2
        compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
        
        frame_data = {
            'timestamp': time.time(),
            'frame_number': self.frame_count,
            'data': compressed.tobytes(),
            'shape': frame.shape
        }
        
        self.frames_buffer.append(frame_data)
    
    def take_state_snapshot(self):
        """Take a comprehensive state snapshot."""
        # Gather all available state information
        # In real implementation, would query game for state
        player_state = {
            'position': (0, 0, 0),
            'health': 100,
            'inventory': [],
            'stats': {}
        }
        
        world_state = {
            'time': self.frame_count,
            'entities': [],
            'environment': {}
        }
        
        ui_state = {
            'menu': None,
            'dialog': None,
            'notifications': []
        }
        
        self.log_state(player_state, world_state, ui_state)
    
    def track_state_transition(self, new_state: GameState):
        """Track state transitions for pattern analysis."""
        if not self.previous_state:
            return
        
        # Example: Track health transitions
        if 'player' in self.previous_state and 'player' in self.current_state:
            prev_health = self.previous_state['player'].get('health', 0)
            curr_health = self.current_state['player'].get('health', 0)
            
            if prev_health != curr_health:
                transition = f"health_{prev_health}_to_{curr_health}"
                self.state_transitions['health'][transition] += 1
    
    def start_session(self, session_id: str = None):
        """Start a new logging session."""
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = time.time()
        self.frame_count = 0
        self.event_count = 0
        
        # Clear buffers
        self.events_buffer.clear()
        self.states_buffer.clear()
        self.actions_buffer.clear()
        self.frames_buffer.clear()
        
        # Reset statistics
        self.event_stats.clear()
        self.action_stats.clear()
        self.state_transitions.clear()
        
        logger.info(f"Started logging session: {self.session_id}")
    
    def export_session(self, filepath: str = None):
        """Export current session data."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{self.session_id}_{timestamp}.{self.export_format}"
            if self.compression:
                filename += '.gz'
            filepath = self.data_dir / filename
        
        data = self.prepare_export_data()
        
        if self.export_format == 'json':
            self.export_json(data, filepath)
        elif self.export_format == 'csv':
            self.export_csv(data, filepath)
        elif self.export_format == 'binary':
            self.export_binary(data, filepath)
        else:
            logger.error(f"Unknown export format: {self.export_format}")
            return
        
        logger.info(f"Exported session to {filepath}")
    
    def prepare_export_data(self) -> Dict[str, Any]:
        """Prepare data for export."""
        return {
            'session': {
                'id': self.session_id,
                'start_time': self.session_start_time,
                'duration': time.time() - self.session_start_time,
                'frame_count': self.frame_count,
                'event_count': self.event_count
            },
            'events': [asdict(e) for e in self.events_buffer],
            'states': [asdict(s) for s in self.states_buffer],
            'actions': [asdict(a) for a in self.actions_buffer],
            'statistics': {
                'event_stats': dict(self.event_stats),
                'action_stats': dict(self.action_stats),
                'state_transitions': {
                    k: dict(v) for k, v in self.state_transitions.items()
                }
            }
        }
    
    def export_json(self, data: Dict[str, Any], filepath: Path):
        """Export data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        
        if self.compression:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(filepath, 'w') as f:
                f.write(json_str)
    
    def export_csv(self, data: Dict[str, Any], filepath: Path):
        """Export data as CSV."""
        # Export events to CSV
        events_file = filepath.with_suffix('.events.csv')
        
        with open(events_file, 'w', newline='') as f:
            if data['events']:
                writer = csv.DictWriter(f, fieldnames=data['events'][0].keys())
                writer.writeheader()
                writer.writerows(data['events'])
        
        # Export actions to CSV
        actions_file = filepath.with_suffix('.actions.csv')
        
        with open(actions_file, 'w', newline='') as f:
            if data['actions']:
                writer = csv.DictWriter(f, fieldnames=data['actions'][0].keys())
                writer.writeheader()
                writer.writerows(data['actions'])
    
    def export_binary(self, data: Dict[str, Any], filepath: Path):
        """Export data as binary (pickle)."""
        if self.compression:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
    
    def export_buffered_data(self):
        """Export and clear buffered data."""
        if not any([self.events_buffer, self.states_buffer, self.actions_buffer]):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"buffer_{self.session_id}_{timestamp}.json"
        if self.compression:
            filename += '.gz'
        
        filepath = self.data_dir / filename
        data = self.prepare_export_data()
        self.export_json(data, filepath)
        
        # Clear buffers after export
        self.events_buffer.clear()
        self.states_buffer.clear()
        self.actions_buffer.clear()
        
        logger.debug(f"Exported buffered data to {filepath}")
    
    def load_session(self, filepath: str) -> Dict[str, Any]:
        """Load a logged session."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        logger.info(f"Loaded session from {filepath}")
        return data
    
    def start_replay(self, session_data: Dict[str, Any]):
        """Start replaying a logged session."""
        self.replay_mode = True
        self.replay_data = session_data
        self.replay_index = 0
        
        logger.info("Started replay mode")
    
    def stop_replay(self):
        """Stop replay mode."""
        self.replay_mode = False
        self.replay_data = None
        self.replay_index = 0
        
        logger.info("Stopped replay mode")
    
    def apply_replay_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply replay overlay to frame."""
        if not self.replay_data:
            return frame
        
        import cv2
        overlay = frame.copy()
        
        # Show replay progress
        events = self.replay_data.get('events', [])
        if self.replay_index < len(events):
            event = events[self.replay_index]
            
            # Display event info
            event_text = f"Event: {event['event_type']}"
            cv2.putText(
                overlay, event_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )
            
            # Advance replay
            self.replay_index += int(self.replay_speed)
        
        # Show replay controls
        controls_text = f"Replay: {self.replay_index}/{len(events)} Speed: {self.replay_speed}x"
        cv2.putText(
            overlay, controls_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
        
        return overlay
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw logging statistics overlay."""
        import cv2
        overlay = frame.copy()
        
        # Prepare statistics text
        stats_text = [
            f"Session: {self.session_id}",
            f"Events: {self.event_count}",
            f"Frames: {self.frame_count}",
            f"Buffer: {len(self.events_buffer)}/{self.events_buffer.maxlen}"
        ]
        
        # Draw statistics
        y_offset = frame.shape[0] - 100
        for text in stats_text:
            cv2.putText(
                overlay, text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )
            y_offset += 20
        
        return overlay
    
    def register_event_analyzer(self, analyzer: Callable):
        """Register event analyzer callback."""
        self.event_analyzers.append(analyzer)
    
    def register_state_analyzer(self, analyzer: Callable):
        """Register state analyzer callback."""
        self.state_analyzers.append(analyzer)
    
    def register_default_analyzers(self):
        """Register default analyzers."""
        # Example: Detect combat patterns
        def combat_analyzer(event: GameEvent):
            if event.event_type in ['damage', 'kill', 'death']:
                # Analyze combat patterns
                pass
        
        # Example: Detect speedrun splits
        def speedrun_analyzer(state: GameState):
            # Check for checkpoint reached
            pass
        
        self.register_event_analyzer(combat_analyzer)
        self.register_state_analyzer(speedrun_analyzer)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate session summary."""
        duration = time.time() - self.session_start_time
        
        summary = {
            'session_id': self.session_id,
            'duration': duration,
            'total_events': self.event_count,
            'total_frames': self.frame_count,
            'events_per_second': self.event_count / duration if duration > 0 else 0,
            'top_events': sorted(
                self.event_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'top_actions': sorted(
                self.action_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        # Save summary
        summary_file = self.data_dir / f"summary_{self.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated session summary: {summary_file}")
        return summary
    
    def query_events(self, event_type: str = None, time_range: Tuple[float, float] = None) -> List[GameEvent]:
        """Query logged events."""
        results = []
        
        for event in self.events_buffer:
            # Filter by type
            if event_type and event.event_type != event_type:
                continue
            
            # Filter by time
            if time_range:
                if not (time_range[0] <= event.timestamp <= time_range[1]):
                    continue
            
            results.append(event)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current logging statistics."""
        return {
            'session': {
                'id': self.session_id,
                'duration': time.time() - self.session_start_time,
                'frames': self.frame_count,
                'events': self.event_count
            },
            'buffers': {
                'events': len(self.events_buffer),
                'states': len(self.states_buffer),
                'actions': len(self.actions_buffer),
                'frames': len(self.frames_buffer)
            },
            'statistics': {
                'event_types': dict(self.event_stats),
                'action_types': dict(self.action_stats)
            }
        }


# Plugin registration
def create_plugin() -> GameStateLoggerPlugin:
    """Create plugin instance."""
    return GameStateLoggerPlugin()


# Import Tuple for type hints
from typing import Tuple