"""Input Recording System for Nexus Framework"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import weakref
import threading
import h5py
import numpy as np

logger = structlog.get_logger()


class EventType(Enum):
    """Input event types"""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GAMEPAD = "gamepad"


class KeyboardAction(Enum):
    """Keyboard event actions"""
    DOWN = "down"
    UP = "up"


class MouseAction(Enum):
    """Mouse event actions"""
    MOVE = "move"
    CLICK = "click"
    RELEASE = "release"
    SCROLL = "scroll"
    DOUBLE_CLICK = "double_click"


@dataclass
class InputEvent:
    """Input event data structure"""
    timestamp: float
    event_type: EventType
    data: Dict[str, Any]
    frame_id: Optional[str] = None
    window_bounds: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputEvent':
        """Create from dictionary"""
        data['event_type'] = EventType(data['event_type'])
        return cls(**data)


class RecordingState(Enum):
    """Recording state management"""
    STOPPED = "stopped"
    RECORDING = "recording"
    PAUSED = "paused"


class InputRecorder:
    """Records user input events for playback and analysis"""
    
    def __init__(self, 
                 storage_backend: str = "memory",
                 redis_config: Optional[Dict[str, Any]] = None,
                 hdf5_path: Optional[str] = None,
                 max_events: int = 10000):
        """
        Initialize input recorder
        
        Args:
            storage_backend: "memory", "redis", "hdf5", or "hybrid"
            redis_config: Redis configuration for real-time storage
            hdf5_path: Path for HDF5 dataset storage
            max_events: Maximum events to keep in memory
        """
        self.storage_backend = storage_backend
        self.redis_config = redis_config or {}
        self.hdf5_path = hdf5_path
        self.max_events = max_events
        
        # Recording state
        self.state = RecordingState.STOPPED
        self.events: List[InputEvent] = []
        self.active_keys = set()
        self.last_mouse_pos = (0, 0)
        
        # Event callbacks
        self.event_callbacks: List[weakref.ref] = []
        
        # Storage backends
        self.redis_client = None
        self.hdf5_file = None
        
        # Input capture
        self.capture_thread = None
        self.stop_capture = threading.Event()
        
        # Frame synchronization
        self.frame_sync_enabled = False
        self.current_frame_id = None
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "keyboard_events": 0,
            "mouse_events": 0,
            "recording_start_time": None,
            "recording_duration": 0
        }
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage backends"""
        if self.storage_backend in ["redis", "hybrid"]:
            self._initialize_redis()
        
        if self.storage_backend in ["hdf5", "hybrid"]:
            self._initialize_hdf5()
    
    def _initialize_redis(self):
        """Initialize Redis storage"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.redis_config.get("host", "127.0.0.1"),
                port=self.redis_config.get("port", 6379),
                db=self.redis_config.get("db", 0),
                decode_responses=True
            )
            logger.info("Redis storage initialized")
        except ImportError:
            logger.warning("Redis not available, falling back to memory storage")
            self.redis_client = None
    
    def _initialize_hdf5(self):
        """Initialize HDF5 storage"""
        if not self.hdf5_path:
            return
        
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, 'a')
            logger.info(f"HDF5 storage initialized: {self.hdf5_path}")
        except Exception as e:
            logger.error(f"Failed to initialize HDF5 storage: {e}")
            self.hdf5_file = None
    
    def start_recording(self, enable_frame_sync: bool = False) -> bool:
        """Start input recording"""
        if self.state == RecordingState.RECORDING:
            logger.warning("Recording already in progress")
            return False
        
        try:
            self.state = RecordingState.RECORDING
            self.frame_sync_enabled = enable_frame_sync
            self.events.clear()
            self.active_keys.clear()
            
            # Reset stats
            self.stats["recording_start_time"] = time.time()
            self.stats["total_events"] = 0
            self.stats["keyboard_events"] = 0
            self.stats["mouse_events"] = 0
            
            # Start input capture
            self._start_input_capture()
            
            logger.info("Input recording started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.state = RecordingState.STOPPED
            return False
    
    def stop_recording(self) -> bool:
        """Stop input recording"""
        if self.state == RecordingState.STOPPED:
            return False
        
        try:
            self.state = RecordingState.STOPPED
            self.stop_capture.set()
            
            # Calculate recording duration
            if self.stats["recording_start_time"]:
                self.stats["recording_duration"] = time.time() - self.stats["recording_start_time"]
            
            # Stop input capture
            self._stop_input_capture()
            
            # Save final dataset
            if self.storage_backend in ["hdf5", "hybrid"]:
                self._save_hdf5_dataset()
            
            logger.info(f"Recording stopped. Captured {len(self.events)} events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            return False
    
    def pause_recording(self) -> bool:
        """Pause input recording"""
        if self.state == RecordingState.RECORDING:
            self.state = RecordingState.PAUSED
            logger.info("Recording paused")
            return True
        return False
    
    def resume_recording(self) -> bool:
        """Resume input recording"""
        if self.state == RecordingState.PAUSED:
            self.state = RecordingState.RECORDING
            logger.info("Recording resumed")
            return True
        return False
    
    def _start_input_capture(self):
        """Start input capture thread"""
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=self._capture_input_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def _stop_input_capture(self):
        """Stop input capture thread"""
        if self.capture_thread and self.capture_thread.is_alive():
            self.stop_capture.set()
            self.capture_thread.join(timeout=5.0)
    
    def _capture_input_loop(self):
        """Main input capture loop"""
        try:
            # Import input capture library
            from pynput import keyboard, mouse
            
            def on_key_press(key):
                if self.state == RecordingState.RECORDING:
                    self._handle_keyboard_event(key, KeyboardAction.DOWN)
            
            def on_key_release(key):
                if self.state == RecordingState.RECORDING:
                    self._handle_keyboard_event(key, KeyboardAction.UP)
            
            def on_mouse_move(x, y):
                if self.state == RecordingState.RECORDING:
                    self._handle_mouse_event(MouseAction.MOVE, x, y)
            
            def on_mouse_click(x, y, button, pressed):
                if self.state == RecordingState.RECORDING:
                    action = MouseAction.CLICK if pressed else MouseAction.RELEASE
                    self._handle_mouse_event(action, x, y, button=button.name)
            
            def on_mouse_scroll(x, y, dx, dy):
                if self.state == RecordingState.RECORDING:
                    self._handle_mouse_event(MouseAction.SCROLL, x, y, scroll_x=dx, scroll_y=dy)
            
            # Start listeners
            keyboard_listener = keyboard.Listener(
                on_press=on_key_press,
                on_release=on_key_release
            )
            
            mouse_listener = mouse.Listener(
                on_move=on_mouse_move,
                on_click=on_mouse_click,
                on_scroll=on_mouse_scroll
            )
            
            keyboard_listener.start()
            mouse_listener.start()
            
            # Wait for stop signal
            while not self.stop_capture.is_set():
                time.sleep(0.1)
            
            # Stop listeners
            keyboard_listener.stop()
            mouse_listener.stop()
            
        except ImportError:
            logger.error("pynput library not available for input capture")
        except Exception as e:
            logger.error(f"Input capture error: {e}")
    
    def _handle_keyboard_event(self, key, action: KeyboardAction):
        """Handle keyboard input event"""
        try:
            # Get key name
            if hasattr(key, 'char') and key.char:
                key_name = key.char
            else:
                key_name = str(key).replace('Key.', '')
            
            # Check for duplicate key-down events
            if action == KeyboardAction.DOWN:
                if key_name in self.active_keys:
                    return  # Skip duplicate
                self.active_keys.add(key_name)
            elif action == KeyboardAction.UP:
                self.active_keys.discard(key_name)
            
            # Create event
            event = InputEvent(
                timestamp=time.time(),
                event_type=EventType.KEYBOARD,
                data={
                    "key": key_name,
                    "action": action.value,
                    "active_keys": list(self.active_keys)
                },
                frame_id=self.current_frame_id
            )
            
            self._store_event(event)
            self.stats["keyboard_events"] += 1
            
        except Exception as e:
            logger.error(f"Error handling keyboard event: {e}")
    
    def _handle_mouse_event(self, action: MouseAction, x: int, y: int, **kwargs):
        """Handle mouse input event"""
        try:
            # Filter duplicate mouse moves
            if action == MouseAction.MOVE:
                if (x, y) == self.last_mouse_pos:
                    return
                self.last_mouse_pos = (x, y)
            
            # Create event
            event_data = {
                "action": action.value,
                "x": x,
                "y": y
            }
            event_data.update(kwargs)
            
            event = InputEvent(
                timestamp=time.time(),
                event_type=EventType.MOUSE,
                data=event_data,
                frame_id=self.current_frame_id
            )
            
            self._store_event(event)
            self.stats["mouse_events"] += 1
            
        except Exception as e:
            logger.error(f"Error handling mouse event: {e}")
    
    def _store_event(self, event: InputEvent):
        """Store event in appropriate backend"""
        # Memory storage
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        # Redis storage
        if self.redis_client and self.storage_backend in ["redis", "hybrid"]:
            try:
                self.redis_client.lpush(
                    "nexus:input_events",
                    json.dumps(event.to_dict())
                )
                self.redis_client.ltrim("nexus:input_events", 0, self.max_events - 1)
            except Exception as e:
                logger.error(f"Redis storage error: {e}")
        
        # Notify callbacks
        self._notify_callbacks(event)
        
        self.stats["total_events"] += 1
    
    def _notify_callbacks(self, event: InputEvent):
        """Notify event callbacks"""
        dead_refs = []
        for callback_ref in self.event_callbacks:
            callback = callback_ref()
            if callback is None:
                dead_refs.append(callback_ref)
            else:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # Clean up dead references
        for ref in dead_refs:
            self.event_callbacks.remove(ref)
    
    def add_event_callback(self, callback: Callable[[InputEvent], None]):
        """Add callback for real-time event processing"""
        self.event_callbacks.append(weakref.ref(callback))
    
    def get_events(self, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  event_type: Optional[EventType] = None) -> List[InputEvent]:
        """Get recorded events with optional filtering"""
        events = self.events
        
        # Time filtering
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Type filtering
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events
    
    def save_recording(self, filepath: str, format: str = "json") -> bool:
        """Save recording to file"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                data = {
                    "metadata": {
                        "recording_duration": self.stats["recording_duration"],
                        "total_events": len(self.events),
                        "recording_start_time": self.stats["recording_start_time"]
                    },
                    "events": [event.to_dict() for event in self.events]
                }
                
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == "hdf5":
                with h5py.File(path, 'w') as f:
                    # Metadata
                    meta_group = f.create_group("metadata")
                    meta_group.attrs["recording_duration"] = self.stats["recording_duration"]
                    meta_group.attrs["total_events"] = len(self.events)
                    
                    # Events
                    events_group = f.create_group("events")
                    
                    for i, event in enumerate(self.events):
                        event_group = events_group.create_group(f"event_{i}")
                        event_group.attrs["timestamp"] = event.timestamp
                        event_group.attrs["event_type"] = event.event_type.value
                        
                        # Store event data
                        for key, value in event.data.items():
                            event_group.attrs[key] = value
            
            logger.info(f"Recording saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return False
    
    def load_recording(self, filepath: str) -> bool:
        """Load recording from file"""
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"Recording file not found: {filepath}")
                return False
            
            if filepath.endswith('.json'):
                with open(path, 'r') as f:
                    data = json.load(f)
                
                self.events = [InputEvent.from_dict(event_data) for event_data in data["events"]]
                
            elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
                with h5py.File(path, 'r') as f:
                    events_group = f["events"]
                    self.events = []
                    
                    for event_key in sorted(events_group.keys()):
                        event_group = events_group[event_key]
                        
                        # Reconstruct event data
                        data = {}
                        for attr_name, attr_value in event_group.attrs.items():
                            if attr_name not in ["timestamp", "event_type"]:
                                data[attr_name] = attr_value
                        
                        event = InputEvent(
                            timestamp=event_group.attrs["timestamp"],
                            event_type=EventType(event_group.attrs["event_type"]),
                            data=data
                        )
                        self.events.append(event)
            
            logger.info(f"Recording loaded: {len(self.events)} events from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return False
    
    def _save_hdf5_dataset(self):
        """Save events to HDF5 dataset format"""
        if not self.hdf5_file:
            return
        
        try:
            timestamp = str(int(time.time()))
            
            # Create datasets for this recording session
            events_group = self.hdf5_file.create_group(f"recording_{timestamp}")
            
            # Separate events by type
            keyboard_events = [e for e in self.events if e.event_type == EventType.KEYBOARD]
            mouse_events = [e for e in self.events if e.event_type == EventType.MOUSE]
            
            if keyboard_events:
                kb_group = events_group.create_group("keyboard")
                kb_timestamps = [e.timestamp for e in keyboard_events]
                kb_keys = [e.data.get("key", "") for e in keyboard_events]
                kb_actions = [e.data.get("action", "") for e in keyboard_events]
                
                kb_group.create_dataset("timestamps", data=kb_timestamps)
                kb_group.create_dataset("keys", data=kb_keys, dtype=h5py.string_dtype())
                kb_group.create_dataset("actions", data=kb_actions, dtype=h5py.string_dtype())
            
            if mouse_events:
                mouse_group = events_group.create_group("mouse")
                mouse_timestamps = [e.timestamp for e in mouse_events]
                mouse_x = [e.data.get("x", 0) for e in mouse_events]
                mouse_y = [e.data.get("y", 0) for e in mouse_events]
                mouse_actions = [e.data.get("action", "") for e in mouse_events]
                
                mouse_group.create_dataset("timestamps", data=mouse_timestamps)
                mouse_group.create_dataset("x_positions", data=mouse_x)
                mouse_group.create_dataset("y_positions", data=mouse_y)
                mouse_group.create_dataset("actions", data=mouse_actions, dtype=h5py.string_dtype())
            
            # Save metadata
            events_group.attrs["recording_duration"] = self.stats["recording_duration"]
            events_group.attrs["total_events"] = len(self.events)
            events_group.attrs["keyboard_events"] = len(keyboard_events)
            events_group.attrs["mouse_events"] = len(mouse_events)
            
            self.hdf5_file.flush()
            logger.info(f"HDF5 dataset saved for recording session {timestamp}")
            
        except Exception as e:
            logger.error(f"Failed to save HDF5 dataset: {e}")
    
    def set_frame_sync(self, frame_id: str):
        """Set current frame ID for synchronization"""
        self.current_frame_id = frame_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics"""
        stats = self.stats.copy()
        stats["current_state"] = self.state.value
        stats["events_in_memory"] = len(self.events)
        stats["active_keys"] = list(self.active_keys)
        return stats
    
    def clear_recording(self):
        """Clear recorded events"""
        self.events.clear()
        self.active_keys.clear()
        self.stats["total_events"] = 0
        self.stats["keyboard_events"] = 0
        self.stats["mouse_events"] = 0
        logger.info("Recording data cleared")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_recording()
        
        if self.hdf5_file:
            self.hdf5_file.close()
        
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close redis client: {e}")