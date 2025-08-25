"""Input Playback System for Nexus Framework"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from enum import Enum
import structlog
import weakref
from dataclasses import dataclass

from .recorder import InputEvent, EventType, KeyboardAction, MouseAction

logger = structlog.get_logger()


class PlaybackState(Enum):
    """Playback state management"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"


class PlaybackSpeed(Enum):
    """Playback speed options"""
    QUARTER = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    QUADRUPLE = 4.0


@dataclass
class PlaybackConfig:
    """Playback configuration"""
    speed: float = 1.0
    loop: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    filter_event_types: Optional[List[EventType]] = None
    sync_with_frames: bool = False
    precise_timing: bool = True
    
    # Input filtering
    filter_keys: Optional[List[str]] = None
    filter_mouse_moves: bool = False
    min_mouse_move_distance: int = 5


class InputPlayback:
    """Plays back recorded input events"""
    
    def __init__(self, config: Optional[PlaybackConfig] = None):
        """Initialize input playback system"""
        self.config = config or PlaybackConfig()
        
        # Playback state
        self.state = PlaybackState.STOPPED
        self.events: List[InputEvent] = []
        self.current_index = 0
        self.start_timestamp = None
        self.playback_start_time = None
        
        # Threading
        self.playback_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Callbacks
        self.event_callbacks: List[weakref.ref] = []
        self.state_callbacks: List[weakref.ref] = []
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_played": 0,
            "keyboard_events_played": 0,
            "mouse_events_played": 0,
            "playback_duration": 0,
            "skipped_events": 0
        }
        
        # Input backends
        self.input_backend = None
        self._initialize_input_backend()
    
    def _initialize_input_backend(self):
        """Initialize input simulation backend"""
        try:
            from pynput.keyboard import Key, KeyCode, Controller as KeyboardController
            from pynput.mouse import Button, Controller as MouseController
            
            self.keyboard_controller = KeyboardController()
            self.mouse_controller = MouseController()
            self.input_backend = "pynput"
            
            logger.info("Pynput input backend initialized")
            
        except ImportError:
            logger.warning("Pynput not available, input simulation disabled")
            self.input_backend = None
    
    def load_events(self, events: List[InputEvent]) -> bool:
        """Load events for playback"""
        try:
            if not events:
                logger.error("No events to load")
                return False
            
            # Sort events by timestamp
            self.events = sorted(events, key=lambda e: e.timestamp)
            
            # Apply filtering
            if self.config.filter_event_types:
                self.events = [e for e in self.events if e.event_type in self.config.filter_event_types]
            
            # Apply time range filtering
            if self.config.start_time:
                self.events = [e for e in self.events if e.timestamp >= self.config.start_time]
            
            if self.config.end_time:
                self.events = [e for e in self.events if e.timestamp <= self.config.end_time]
            
            # Apply additional filtering
            self.events = self._apply_event_filters(self.events)
            
            self.current_index = 0
            self.start_timestamp = self.events[0].timestamp if self.events else None
            
            self.stats["total_events"] = len(self.events)
            self.stats["events_played"] = 0
            self.stats["keyboard_events_played"] = 0
            self.stats["mouse_events_played"] = 0
            self.stats["skipped_events"] = 0
            
            logger.info(f"Loaded {len(self.events)} events for playback")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load events: {e}")
            return False
    
    def _apply_event_filters(self, events: List[InputEvent]) -> List[InputEvent]:
        """Apply additional event filters"""
        filtered_events = []
        last_mouse_pos = None
        
        for event in events:
            skip_event = False
            
            # Filter specific keys
            if (event.event_type == EventType.KEYBOARD and 
                self.config.filter_keys and 
                event.data.get("key") in self.config.filter_keys):
                skip_event = True
            
            # Filter mouse movements
            if (event.event_type == EventType.MOUSE and 
                event.data.get("action") == MouseAction.MOVE.value and
                self.config.filter_mouse_moves):
                
                current_pos = (event.data.get("x", 0), event.data.get("y", 0))
                
                if last_mouse_pos:
                    distance = ((current_pos[0] - last_mouse_pos[0]) ** 2 + 
                              (current_pos[1] - last_mouse_pos[1]) ** 2) ** 0.5
                    
                    if distance < self.config.min_mouse_move_distance:
                        skip_event = True
                
                last_mouse_pos = current_pos
            
            if not skip_event:
                filtered_events.append(event)
            else:
                self.stats["skipped_events"] += 1
        
        return filtered_events
    
    def start_playback(self) -> bool:
        """Start event playback"""
        if not self.events:
            logger.error("No events loaded for playback")
            return False
        
        if self.state == PlaybackState.PLAYING:
            logger.warning("Playback already in progress")
            return False
        
        if not self.input_backend:
            logger.error("No input backend available for playback")
            return False
        
        try:
            self.state = PlaybackState.PLAYING
            self.playback_start_time = time.time()
            self.stop_event.clear()
            self.pause_event.clear()
            
            # Start playback thread
            self.playback_thread = threading.Thread(target=self._playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            self._notify_state_callbacks()
            logger.info("Playback started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start playback: {e}")
            self.state = PlaybackState.STOPPED
            return False
    
    def stop_playback(self) -> bool:
        """Stop event playback"""
        if self.state == PlaybackState.STOPPED:
            return False
        
        try:
            self.stop_event.set()
            self.pause_event.set()  # Unpause if paused
            
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=5.0)
            
            self.state = PlaybackState.STOPPED
            self.current_index = 0
            
            # Update stats
            if self.playback_start_time:
                self.stats["playback_duration"] = time.time() - self.playback_start_time
            
            self._notify_state_callbacks()
            logger.info("Playback stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop playback: {e}")
            return False
    
    def pause_playback(self) -> bool:
        """Pause event playback"""
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED
            self.pause_event.set()
            self._notify_state_callbacks()
            logger.info("Playback paused")
            return True
        return False
    
    def resume_playback(self) -> bool:
        """Resume event playback"""
        if self.state == PlaybackState.PAUSED:
            self.state = PlaybackState.PLAYING
            self.pause_event.clear()
            self._notify_state_callbacks()
            logger.info("Playback resumed")
            return True
        return False
    
    def seek_to_position(self, position: float) -> bool:
        """Seek to specific position (0.0 - 1.0)"""
        if not self.events:
            return False
        
        try:
            target_index = int(position * len(self.events))
            self.current_index = max(0, min(target_index, len(self.events) - 1))
            logger.info(f"Seeked to position {position:.2%} (event {self.current_index})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to seek: {e}")
            return False
    
    def seek_to_time(self, target_time: float) -> bool:
        """Seek to specific timestamp"""
        if not self.events or not self.start_timestamp:
            return False
        
        try:
            relative_time = target_time - self.start_timestamp
            
            for i, event in enumerate(self.events):
                event_relative_time = event.timestamp - self.start_timestamp
                if event_relative_time >= relative_time:
                    self.current_index = i
                    logger.info(f"Seeked to time {relative_time:.2f}s (event {i})")
                    return True
            
            # If not found, go to end
            self.current_index = len(self.events) - 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to seek to time: {e}")
            return False
    
    def set_speed(self, speed: float) -> bool:
        """Set playback speed"""
        try:
            self.config.speed = max(0.1, min(speed, 10.0))  # Clamp between 0.1x and 10x
            logger.info(f"Playback speed set to {self.config.speed}x")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set speed: {e}")
            return False
    
    def _playback_loop(self):
        """Main playback loop"""
        try:
            logger.info("Starting playback loop")
            
            while (self.current_index < len(self.events) and 
                   not self.stop_event.is_set()):
                
                # Handle pause
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                
                event = self.events[self.current_index]
                
                # Calculate timing
                if self.config.precise_timing and self.start_timestamp:
                    # Calculate expected time based on original timing
                    relative_time = (event.timestamp - self.start_timestamp) / self.config.speed
                    
                    if self.playback_start_time:
                        expected_time = self.playback_start_time + relative_time
                        current_time = time.time()
                        sleep_time = expected_time - current_time
                        
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                
                # Execute event
                if self._execute_event(event):
                    self.stats["events_played"] += 1
                    
                    if event.event_type == EventType.KEYBOARD:
                        self.stats["keyboard_events_played"] += 1
                    elif event.event_type == EventType.MOUSE:
                        self.stats["mouse_events_played"] += 1
                
                # Notify callbacks
                self._notify_event_callbacks(event)
                
                self.current_index += 1
            
            # Handle completion
            if self.current_index >= len(self.events):
                if self.config.loop:
                    logger.info("Restarting playback loop")
                    self.current_index = 0
                    self.playback_start_time = time.time()
                else:
                    logger.info("Playback completed")
                    self.state = PlaybackState.COMPLETED
                    self._notify_state_callbacks()
            
        except Exception as e:
            logger.error(f"Playback loop error: {e}")
            self.state = PlaybackState.STOPPED
            self._notify_state_callbacks()
    
    def _execute_event(self, event: InputEvent) -> bool:
        """Execute a single input event"""
        try:
            if event.event_type == EventType.KEYBOARD:
                return self._execute_keyboard_event(event)
            elif event.event_type == EventType.MOUSE:
                return self._execute_mouse_event(event)
            else:
                logger.warning(f"Unknown event type: {event.event_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute event: {e}")
            return False
    
    def _execute_keyboard_event(self, event: InputEvent) -> bool:
        """Execute keyboard event"""
        if not self.keyboard_controller:
            return False
        
        try:
            key_name = event.data.get("key")
            action = event.data.get("action")
            
            if not key_name or not action:
                return False
            
            # Convert key name to pynput key
            key = self._get_pynput_key(key_name)
            if not key:
                return False
            
            # Execute action
            if action == KeyboardAction.DOWN.value:
                self.keyboard_controller.press(key)
            elif action == KeyboardAction.UP.value:
                self.keyboard_controller.release(key)
            else:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Keyboard event execution error: {e}")
            return False
    
    def _execute_mouse_event(self, event: InputEvent) -> bool:
        """Execute mouse event"""
        if not self.mouse_controller:
            return False
        
        try:
            action = event.data.get("action")
            x = event.data.get("x", 0)
            y = event.data.get("y", 0)
            
            if action == MouseAction.MOVE.value:
                self.mouse_controller.position = (x, y)
                
            elif action == MouseAction.CLICK.value:
                button_name = event.data.get("button", "left")
                button = self._get_mouse_button(button_name)
                if button:
                    self.mouse_controller.position = (x, y)
                    self.mouse_controller.press(button)
                    
            elif action == MouseAction.RELEASE.value:
                button_name = event.data.get("button", "left")
                button = self._get_mouse_button(button_name)
                if button:
                    self.mouse_controller.release(button)
                    
            elif action == MouseAction.SCROLL.value:
                scroll_x = event.data.get("scroll_x", 0)
                scroll_y = event.data.get("scroll_y", 0)
                self.mouse_controller.position = (x, y)
                self.mouse_controller.scroll(scroll_x, scroll_y)
                
            elif action == MouseAction.DOUBLE_CLICK.value:
                button_name = event.data.get("button", "left")
                button = self._get_mouse_button(button_name)
                if button:
                    self.mouse_controller.position = (x, y)
                    self.mouse_controller.click(button, 2)
            else:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Mouse event execution error: {e}")
            return False
    
    def _get_pynput_key(self, key_name: str):
        """Convert key name to pynput key"""
        try:
            from pynput.keyboard import Key, KeyCode
            
            # Handle special keys
            special_keys = {
                'space': Key.space,
                'enter': Key.enter,
                'tab': Key.tab,
                'backspace': Key.backspace,
                'delete': Key.delete,
                'esc': Key.esc,
                'escape': Key.esc,
                'shift': Key.shift,
                'ctrl': Key.ctrl,
                'alt': Key.alt,
                'cmd': Key.cmd,
                'up': Key.up,
                'down': Key.down,
                'left': Key.left,
                'right': Key.right,
                'home': Key.home,
                'end': Key.end,
                'page_up': Key.page_up,
                'page_down': Key.page_down,
                'caps_lock': Key.caps_lock,
                'num_lock': Key.num_lock,
                'scroll_lock': Key.scroll_lock,
                'insert': Key.insert,
                'pause': Key.pause,
                'print_screen': Key.print_screen,
                'menu': Key.menu,
            }
            
            # Function keys
            for i in range(1, 25):  # F1-F24
                special_keys[f'f{i}'] = getattr(Key, f'f{i}')
            
            key_lower = key_name.lower()
            if key_lower in special_keys:
                return special_keys[key_lower]
            
            # Handle single characters
            if len(key_name) == 1:
                return KeyCode.from_char(key_name)
            
            # Handle key codes
            try:
                return KeyCode.from_vk(int(key_name))
            except (ValueError, TypeError):
                logger.debug(f"Could not convert key '{key_name}' to KeyCode")
            
            logger.warning(f"Unknown key: {key_name}")
            return None
            
        except Exception as e:
            logger.error(f"Key conversion error: {e}")
            return None
    
    def _get_mouse_button(self, button_name: str):
        """Convert button name to pynput button"""
        try:
            from pynput.mouse import Button
            
            button_map = {
                'left': Button.left,
                'right': Button.right,
                'middle': Button.middle,
                'x1': Button.x1,
                'x2': Button.x2
            }
            
            return button_map.get(button_name.lower())
            
        except Exception as e:
            logger.error(f"Button conversion error: {e}")
            return None
    
    def add_event_callback(self, callback: Callable[[InputEvent], None]):
        """Add callback for event execution"""
        self.event_callbacks.append(weakref.ref(callback))
    
    def add_state_callback(self, callback: Callable[[PlaybackState], None]):
        """Add callback for state changes"""
        self.state_callbacks.append(weakref.ref(callback))
    
    def _notify_event_callbacks(self, event: InputEvent):
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
                    logger.error(f"Event callback error: {e}")
        
        for ref in dead_refs:
            self.event_callbacks.remove(ref)
    
    def _notify_state_callbacks(self):
        """Notify state change callbacks"""
        dead_refs = []
        for callback_ref in self.state_callbacks:
            callback = callback_ref()
            if callback is None:
                dead_refs.append(callback_ref)
            else:
                try:
                    callback(self.state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
        
        for ref in dead_refs:
            self.state_callbacks.remove(ref)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get playback progress information"""
        if not self.events:
            return {
                "current_position": 0.0,
                "current_index": 0,
                "total_events": 0,
                "events_played": 0,
                "time_remaining": 0.0
            }
        
        progress = self.current_index / len(self.events)
        
        # Calculate time remaining
        time_remaining = 0.0
        if (self.start_timestamp and 
            self.current_index < len(self.events) and 
            self.playback_start_time):
            
            current_event = self.events[self.current_index]
            last_event = self.events[-1]
            
            remaining_original_time = last_event.timestamp - current_event.timestamp
            time_remaining = remaining_original_time / self.config.speed
        
        return {
            "current_position": progress,
            "current_index": self.current_index,
            "total_events": len(self.events),
            "events_played": self.stats["events_played"],
            "time_remaining": time_remaining,
            "playback_speed": self.config.speed,
            "state": self.state.value
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get playback statistics"""
        stats = self.stats.copy()
        stats.update(self.get_progress())
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_playback()
        self.events.clear()
        self.event_callbacks.clear()
        self.state_callbacks.clear()