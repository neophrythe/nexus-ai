"""Enhanced Input Recording and Playback System - SerpentAI Compatible with Improvements

Provides comprehensive input recording and playback with multiple storage backends,
frame synchronization, and human-like patterns.
"""

import time
import json
import pickle
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import numpy as np
import structlog
from datetime import datetime
import asyncio
import h5py

# Input backend imports
try:
    from pynput import keyboard, mouse
    from pynput.keyboard import Key, Controller as KeyboardController
    from pynput.mouse import Button, Controller as MouseController
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import ctypes
    from ctypes import wintypes
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# Optional Redis support
try:
    from redis import StrictRedis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = structlog.get_logger()


class EventType(Enum):
    """Input event types"""
    KEYBOARD_DOWN = "keyboard_down"
    KEYBOARD_UP = "keyboard_up"
    MOUSE_MOVE = "mouse_move"
    MOUSE_CLICK = "mouse_click"
    MOUSE_SCROLL = "mouse_scroll"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    GAMEPAD_BUTTON = "gamepad_button"
    GAMEPAD_AXIS = "gamepad_axis"


class StorageBackend(Enum):
    """Storage backend options"""
    MEMORY = "memory"
    REDIS = "redis"
    HDF5 = "hdf5"
    JSON = "json"
    HYBRID = "hybrid"  # Memory + persistent storage


@dataclass
class InputEvent:
    """Structured input event"""
    timestamp: float
    event_type: EventType
    data: Dict[str, Any]
    frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type.value,
            'data': self.data,
            'frame_id': self.frame_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'InputEvent':
        """Create from dictionary"""
        return cls(
            timestamp=data['timestamp'],
            event_type=EventType(data['event_type']),
            data=data['data'],
            frame_id=data.get('frame_id'),
            metadata=data.get('metadata', {})
        )


@dataclass
class RecordingConfig:
    """Recording configuration"""
    storage_backend: StorageBackend = StorageBackend.MEMORY
    max_events: int = 10000
    record_mouse_moves: bool = True
    mouse_move_interval: float = 0.01  # Minimum time between mouse move events
    deduplicate_keys: bool = True
    frame_sync: bool = False
    compress: bool = False
    redis_config: Optional[Dict] = None
    hdf5_path: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class PlaybackConfig:
    """Playback configuration"""
    speed: float = 1.0
    loop: bool = False
    precise_timing: bool = True
    filter_event_types: Optional[List[EventType]] = None
    filter_mouse_moves: bool = False
    min_mouse_move_distance: int = 5
    human_like: bool = False
    variation_percent: float = 0.1  # Timing variation for human-like playback


class InputRecorder:
    """Enhanced input recorder with multiple storage backends"""
    
    def __init__(self, config: Optional[RecordingConfig] = None):
        self.config = config or RecordingConfig()
        
        # Storage
        self.events: deque = deque(maxlen=self.config.max_events)
        self.redis_client = None
        self.hdf5_file = None
        self.output_file = None
        
        # Recording state
        self.recording = False
        self.paused = False
        self.start_time = None
        self.last_mouse_move_time = 0
        self.active_keys = set()
        self.frame_counter = 0
        
        # Listeners
        self.keyboard_listener = None
        self.mouse_listener = None
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'keyboard_events': 0,
            'mouse_events': 0,
            'dropped_events': 0
        }
        
        # Initialize storage
        self._init_storage()
        
    def _init_storage(self):
        """Initialize storage backend"""
        if self.config.storage_backend in [StorageBackend.REDIS, StorageBackend.HYBRID]:
            if HAS_REDIS and self.config.redis_config:
                self.redis_client = StrictRedis(**self.config.redis_config)
                # Clear old recordings
                self.redis_client.delete("nexus:input:events")
                self.redis_client.delete("nexus:input:metadata")
            else:
                logger.warning("Redis not available, falling back to memory storage")
                
        if self.config.storage_backend in [StorageBackend.HDF5, StorageBackend.HYBRID]:
            if self.config.hdf5_path:
                self.hdf5_file = h5py.File(self.config.hdf5_path, 'w')
                self._init_hdf5_structure()
                
        if self.config.output_path:
            self.output_file = Path(self.config.output_path)
            
    def _init_hdf5_structure(self):
        """Initialize HDF5 file structure"""
        # Create groups
        self.hdf5_file.create_group('metadata')
        self.hdf5_file.create_group('events')
        self.hdf5_file.create_group('frames')
        
        # Metadata
        self.hdf5_file['metadata'].attrs['start_time'] = time.time()
        self.hdf5_file['metadata'].attrs['config'] = json.dumps(asdict(self.config))
        
    def start(self):
        """Start recording"""
        if self.recording:
            return
            
        logger.info("Starting input recording")
        self.recording = True
        self.paused = False
        self.start_time = time.time()
        
        # Start listeners
        if HAS_PYNPUT:
            self._start_pynput_listeners()
        elif HAS_PYAUTOGUI:
            self._start_pyautogui_listeners()
        else:
            logger.error("No input backend available")
            
    def _start_pynput_listeners(self):
        """Start pynput listeners"""
        # Keyboard listener
        def on_key_press(key):
            if not self.paused:
                self._handle_keyboard_event(key, True)
                
        def on_key_release(key):
            if not self.paused:
                self._handle_keyboard_event(key, False)
                
        self.keyboard_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release
        )
        
        # Mouse listener
        def on_move(x, y):
            if not self.paused:
                self._handle_mouse_move(x, y)
                
        def on_click(x, y, button, pressed):
            if not self.paused:
                self._handle_mouse_click(x, y, button, pressed)
                
        def on_scroll(x, y, dx, dy):
            if not self.paused:
                self._handle_mouse_scroll(x, y, dx, dy)
                
        self.mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll
        )
        
        self.keyboard_listener.start()
        self.mouse_listener.start()
        
    def _handle_keyboard_event(self, key, pressed: bool):
        """Handle keyboard event"""
        # Get key name
        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key).replace("Key.", "")
            
        # Deduplicate if configured
        if self.config.deduplicate_keys:
            if pressed and key_name in self.active_keys:
                return
            elif not pressed and key_name not in self.active_keys:
                return
                
        if pressed:
            self.active_keys.add(key_name)
            event_type = EventType.KEYBOARD_DOWN
        else:
            self.active_keys.discard(key_name)
            event_type = EventType.KEYBOARD_UP
            
        # Create event
        event = InputEvent(
            timestamp=time.time(),
            event_type=event_type,
            data={
                'key': key_name,
                'action': 'down' if pressed else 'up',
                'active_keys': list(self.active_keys)
            },
            frame_id=f"frame_{self.frame_counter}" if self.config.frame_sync else None
        )
        
        self._store_event(event)
        self.stats['keyboard_events'] += 1
        
    def _handle_mouse_move(self, x: int, y: int):
        """Handle mouse move event"""
        if not self.config.record_mouse_moves:
            return
            
        # Rate limit mouse moves
        current_time = time.time()
        if current_time - self.last_mouse_move_time < self.config.mouse_move_interval:
            return
            
        self.last_mouse_move_time = current_time
        
        event = InputEvent(
            timestamp=current_time,
            event_type=EventType.MOUSE_MOVE,
            data={'x': x, 'y': y},
            frame_id=f"frame_{self.frame_counter}" if self.config.frame_sync else None
        )
        
        self._store_event(event)
        self.stats['mouse_events'] += 1
        
    def _handle_mouse_click(self, x: int, y: int, button, pressed: bool):
        """Handle mouse click event"""
        event_type = EventType.MOUSE_DOWN if pressed else EventType.MOUSE_UP
        
        event = InputEvent(
            timestamp=time.time(),
            event_type=event_type,
            data={
                'x': x,
                'y': y,
                'button': str(button).replace("Button.", ""),
                'action': 'down' if pressed else 'up'
            },
            frame_id=f"frame_{self.frame_counter}" if self.config.frame_sync else None
        )
        
        self._store_event(event)
        self.stats['mouse_events'] += 1
        
    def _handle_mouse_scroll(self, x: int, y: int, dx: int, dy: int):
        """Handle mouse scroll event"""
        event = InputEvent(
            timestamp=time.time(),
            event_type=EventType.MOUSE_SCROLL,
            data={
                'x': x,
                'y': y,
                'dx': dx,
                'dy': dy
            },
            frame_id=f"frame_{self.frame_counter}" if self.config.frame_sync else None
        )
        
        self._store_event(event)
        self.stats['mouse_events'] += 1
        
    def _store_event(self, event: InputEvent):
        """Store event in configured backend"""
        # Memory storage
        self.events.append(event)
        self.stats['total_events'] += 1
        
        # Redis storage
        if self.redis_client and self.config.storage_backend in [StorageBackend.REDIS, StorageBackend.HYBRID]:
            try:
                # Store as JSON for compatibility
                self.redis_client.lpush("nexus:input:events", json.dumps(event.to_dict()))
                self.redis_client.ltrim("nexus:input:events", 0, self.config.max_events)
            except Exception as e:
                logger.error(f"Failed to store event in Redis: {e}")
                
        # HDF5 storage
        if self.hdf5_file and self.config.storage_backend in [StorageBackend.HDF5, StorageBackend.HYBRID]:
            self._store_hdf5_event(event)
            
    def _store_hdf5_event(self, event: InputEvent):
        """Store event in HDF5"""
        try:
            event_group = self.hdf5_file['events']
            
            # Create dataset for this event type if not exists
            event_type_name = event.event_type.value
            if event_type_name not in event_group:
                event_group.create_group(event_type_name)
                
            # Store event data
            type_group = event_group[event_type_name]
            event_idx = len(type_group.keys())
            event_data = type_group.create_group(str(event_idx))
            
            event_data.attrs['timestamp'] = event.timestamp
            event_data.attrs['data'] = json.dumps(event.data)
            if event.frame_id:
                event_data.attrs['frame_id'] = event.frame_id
                
        except Exception as e:
            logger.error(f"Failed to store event in HDF5: {e}")
            
    def pause(self):
        """Pause recording"""
        self.paused = True
        logger.info("Recording paused")
        
    def resume(self):
        """Resume recording"""
        self.paused = False
        logger.info("Recording resumed")
        
    def stop(self):
        """Stop recording"""
        if not self.recording:
            return
            
        self.recording = False
        
        # Stop listeners
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
            
        # Save recording
        self.save()
        
        # Close files
        if self.hdf5_file:
            self.hdf5_file.close()
            
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Recording stopped. Duration: {duration:.2f}s, Events: {self.stats['total_events']}")
        
    def save(self, path: Optional[str] = None):
        """Save recording to file"""
        save_path = path or self.config.output_path
        if not save_path:
            save_path = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        save_path = Path(save_path)
        
        # Prepare data
        data = {
            'metadata': {
                'start_time': self.start_time,
                'duration': time.time() - self.start_time if self.start_time else 0,
                'total_events': self.stats['total_events'],
                'keyboard_events': self.stats['keyboard_events'],
                'mouse_events': self.stats['mouse_events'],
                'config': asdict(self.config)
            },
            'events': [event.to_dict() for event in self.events]
        }
        
        # Save based on extension
        if save_path.suffix == '.pkl':
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            # Default to JSON
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        logger.info(f"Recording saved to {save_path}")
        
    def load(self, path: str):
        """Load recording from file"""
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Recording file not found: {path}")
            
        # Load based on extension
        if load_path.suffix == '.pkl':
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(load_path, 'r') as f:
                data = json.load(f)
                
        # Restore events
        self.events.clear()
        for event_data in data['events']:
            self.events.append(InputEvent.from_dict(event_data))
            
        # Restore metadata
        self.start_time = data['metadata'].get('start_time')
        self.stats = {
            'total_events': data['metadata'].get('total_events', len(self.events)),
            'keyboard_events': data['metadata'].get('keyboard_events', 0),
            'mouse_events': data['metadata'].get('mouse_events', 0)
        }
        
        logger.info(f"Recording loaded from {load_path}. Events: {len(self.events)}")
        
    def get_statistics(self) -> Dict:
        """Get recording statistics"""
        duration = time.time() - self.start_time if self.start_time and self.recording else 0
        
        return {
            'recording': self.recording,
            'paused': self.paused,
            'duration': duration,
            'total_events': self.stats['total_events'],
            'keyboard_events': self.stats['keyboard_events'],
            'mouse_events': self.stats['mouse_events'],
            'events_per_second': self.stats['total_events'] / duration if duration > 0 else 0,
            'active_keys': list(self.active_keys),
            'storage_backend': self.config.storage_backend.value
        }


class InputPlayback:
    """Enhanced input playback with precise timing and filtering"""
    
    def __init__(self, config: Optional[PlaybackConfig] = None):
        self.config = config or PlaybackConfig()
        
        # Playback state
        self.events: List[InputEvent] = []
        self.current_index = 0
        self.playing = False
        self.paused = False
        self.stop_event = threading.Event()
        
        # Controllers
        self.keyboard_controller = None
        self.mouse_controller = None
        
        # Timing
        self.start_timestamp = None
        self.playback_start_time = None
        
        # Initialize controllers
        self._init_controllers()
        
    def _init_controllers(self):
        """Initialize input controllers"""
        if HAS_PYNPUT:
            self.keyboard_controller = KeyboardController()
            self.mouse_controller = MouseController()
        elif HAS_PYAUTOGUI:
            # PyAutoGUI as fallback
            import pyautogui
            self.mouse_controller = pyautogui
            self.keyboard_controller = pyautogui
            
    def load(self, events: Union[List[InputEvent], str]):
        """Load events for playback"""
        if isinstance(events, str):
            # Load from file
            recorder = InputRecorder()
            recorder.load(events)
            self.events = list(recorder.events)
        else:
            self.events = events
            
        if self.events:
            self.start_timestamp = self.events[0].timestamp
            
        logger.info(f"Loaded {len(self.events)} events for playback")
        
    def play(self):
        """Start playback"""
        if self.playing or not self.events:
            return
            
        self.playing = True
        self.paused = False
        self.current_index = 0
        self.stop_event.clear()
        self.playback_start_time = time.time()
        
        # Start playback thread
        playback_thread = threading.Thread(target=self._playback_loop)
        playback_thread.start()
        
        logger.info("Playback started")
        
    def _playback_loop(self):
        """Main playback loop"""
        while self.current_index < len(self.events) and not self.stop_event.is_set():
            if self.paused:
                time.sleep(0.1)
                continue
                
            event = self.events[self.current_index]
            
            # Filter events if configured
            if self._should_filter_event(event):
                self.current_index += 1
                continue
                
            # Calculate timing
            if self.config.precise_timing and self.current_index > 0:
                relative_time = (event.timestamp - self.start_timestamp) / self.config.speed
                
                # Add human-like variation
                if self.config.human_like:
                    variation = np.random.normal(0, self.config.variation_percent * relative_time)
                    relative_time += variation
                    
                expected_time = self.playback_start_time + relative_time
                sleep_time = expected_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            # Execute event
            self._execute_event(event)
            
            self.current_index += 1
            
            # Loop if configured
            if self.config.loop and self.current_index >= len(self.events):
                self.current_index = 0
                self.playback_start_time = time.time()
                
        self.playing = False
        logger.info("Playback completed")
        
    def _should_filter_event(self, event: InputEvent) -> bool:
        """Check if event should be filtered"""
        # Filter by event type
        if self.config.filter_event_types:
            if event.event_type not in self.config.filter_event_types:
                return True
                
        # Filter mouse moves
        if self.config.filter_mouse_moves and event.event_type == EventType.MOUSE_MOVE:
            if self.current_index > 0:
                prev_event = self.events[self.current_index - 1]
                if prev_event.event_type == EventType.MOUSE_MOVE:
                    # Calculate distance
                    dx = event.data['x'] - prev_event.data['x']
                    dy = event.data['y'] - prev_event.data['y']
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance < self.config.min_mouse_move_distance:
                        return True
                        
        return False
        
    def _execute_event(self, event: InputEvent):
        """Execute an input event"""
        try:
            if event.event_type == EventType.KEYBOARD_DOWN:
                self._press_key(event.data['key'])
            elif event.event_type == EventType.KEYBOARD_UP:
                self._release_key(event.data['key'])
            elif event.event_type == EventType.MOUSE_MOVE:
                self._move_mouse(event.data['x'], event.data['y'])
            elif event.event_type == EventType.MOUSE_DOWN:
                self._press_mouse(event.data['x'], event.data['y'], event.data['button'])
            elif event.event_type == EventType.MOUSE_UP:
                self._release_mouse(event.data['x'], event.data['y'], event.data['button'])
            elif event.event_type == EventType.MOUSE_SCROLL:
                self._scroll_mouse(event.data['dx'], event.data['dy'])
        except Exception as e:
            logger.error(f"Failed to execute event: {e}")
            
    def _press_key(self, key: str):
        """Press a key"""
        if self.keyboard_controller:
            # Convert key string to pynput Key
            if hasattr(Key, key):
                key_obj = getattr(Key, key)
            else:
                key_obj = key
            self.keyboard_controller.press(key_obj)
        elif HAS_PYAUTOGUI:
            pyautogui.keyDown(key)
            
    def _release_key(self, key: str):
        """Release a key"""
        if self.keyboard_controller:
            if hasattr(Key, key):
                key_obj = getattr(Key, key)
            else:
                key_obj = key
            self.keyboard_controller.release(key_obj)
        elif HAS_PYAUTOGUI:
            pyautogui.keyUp(key)
            
    def _move_mouse(self, x: int, y: int):
        """Move mouse to position"""
        if self.mouse_controller:
            if self.config.human_like:
                # Smooth movement
                current_pos = self.mouse_controller.position
                steps = 10
                for i in range(steps):
                    progress = (i + 1) / steps
                    new_x = current_pos[0] + (x - current_pos[0]) * progress
                    new_y = current_pos[1] + (y - current_pos[1]) * progress
                    self.mouse_controller.position = (new_x, new_y)
                    time.sleep(0.01)
            else:
                self.mouse_controller.position = (x, y)
        elif HAS_PYAUTOGUI:
            pyautogui.moveTo(x, y)
            
    def _press_mouse(self, x: int, y: int, button: str):
        """Press mouse button"""
        if self.mouse_controller:
            self.mouse_controller.position = (x, y)
            button_obj = getattr(Button, button.lower(), Button.left)
            self.mouse_controller.press(button_obj)
        elif HAS_PYAUTOGUI:
            pyautogui.mouseDown(x, y, button=button.lower())
            
    def _release_mouse(self, x: int, y: int, button: str):
        """Release mouse button"""
        if self.mouse_controller:
            self.mouse_controller.position = (x, y)
            button_obj = getattr(Button, button.lower(), Button.left)
            self.mouse_controller.release(button_obj)
        elif HAS_PYAUTOGUI:
            pyautogui.mouseUp(x, y, button=button.lower())
            
    def _scroll_mouse(self, dx: int, dy: int):
        """Scroll mouse wheel"""
        if self.mouse_controller:
            self.mouse_controller.scroll(dx, dy)
        elif HAS_PYAUTOGUI:
            pyautogui.scroll(dy)
            
    def pause(self):
        """Pause playback"""
        self.paused = True
        logger.info("Playback paused")
        
    def resume(self):
        """Resume playback"""
        self.paused = False
        logger.info("Playback resumed")
        
    def stop(self):
        """Stop playback"""
        self.stop_event.set()
        self.playing = False
        logger.info("Playback stopped")
        
    def seek(self, index: int):
        """Seek to specific event index"""
        if 0 <= index < len(self.events):
            self.current_index = index
            logger.info(f"Seeked to event {index}")
            
    def get_progress(self) -> Dict:
        """Get playback progress"""
        return {
            'playing': self.playing,
            'paused': self.paused,
            'current_index': self.current_index,
            'total_events': len(self.events),
            'progress_percent': (self.current_index / len(self.events)) * 100 if self.events else 0
        }


# SerpentAI compatibility functions
def record_inputs():
    """Start input recording - SerpentAI compatible"""
    recorder = InputRecorder(RecordingConfig(
        storage_backend=StorageBackend.HYBRID if HAS_REDIS else StorageBackend.MEMORY
    ))
    recorder.start()
    return recorder


def play_inputs(recording_path: str, speed: float = 1.0):
    """Play recorded inputs - SerpentAI compatible"""
    playback = InputPlayback(PlaybackConfig(speed=speed))
    playback.load(recording_path)
    playback.play()
    return playback


# Convenience class combining recording and playback
class InputManager:
    """Combined input recording and playback manager"""
    
    def __init__(self):
        self.recorder = None
        self.playback = None
        
    def start_recording(self, config: Optional[RecordingConfig] = None):
        """Start recording inputs"""
        self.recorder = InputRecorder(config)
        self.recorder.start()
        
    def stop_recording(self, save_path: Optional[str] = None):
        """Stop recording and save"""
        if self.recorder:
            self.recorder.stop()
            if save_path:
                self.recorder.save(save_path)
                
    def start_playback(self, recording_path: str, config: Optional[PlaybackConfig] = None):
        """Start playing recorded inputs"""
        self.playback = InputPlayback(config)
        self.playback.load(recording_path)
        self.playback.play()
        
    def stop_playback(self):
        """Stop playback"""
        if self.playback:
            self.playback.stop()