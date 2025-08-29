"""
Game Event System - Event-driven game interaction framework.
Similar to SerpentAI's event system but with modern improvements.
"""

import time
import threading
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json


class EventType(Enum):
    """Standard game event types."""
    # Game lifecycle events
    GAME_STARTED = "game_started"
    GAME_STOPPED = "game_stopped"
    GAME_PAUSED = "game_paused"
    GAME_RESUMED = "game_resumed"
    
    # State events
    MENU_ENTERED = "menu_entered"
    MENU_EXITED = "menu_exited"
    LEVEL_STARTED = "level_started"
    LEVEL_COMPLETED = "level_completed"
    LEVEL_FAILED = "level_failed"
    
    # Gameplay events
    PLAYER_SPAWN = "player_spawn"
    PLAYER_DEATH = "player_death"
    ENEMY_SPAWN = "enemy_spawn"
    ENEMY_DEATH = "enemy_death"
    ITEM_COLLECTED = "item_collected"
    OBJECTIVE_COMPLETED = "objective_completed"
    CHECKPOINT_REACHED = "checkpoint_reached"
    
    # UI events
    DIALOG_OPENED = "dialog_opened"
    DIALOG_CLOSED = "dialog_closed"
    BUTTON_CLICKED = "button_clicked"
    TEXT_ENTERED = "text_entered"
    
    # Performance events
    FPS_DROP = "fps_drop"
    LAG_DETECTED = "lag_detected"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class GameEvent:
    """Represents a game event."""
    type: Union[EventType, str]
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    priority: int = 0  # Higher priority events processed first
    
    def __str__(self) -> str:
        return f"GameEvent({self.type}, data={self.data})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'type': self.type.value if isinstance(self.type, EventType) else self.type,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameEvent':
        """Create event from dictionary."""
        return cls(
            type=data['type'],
            timestamp=data.get('timestamp', time.time()),
            data=data.get('data', {}),
            source=data.get('source'),
            priority=data.get('priority', 0)
        )


class EventHandler:
    """Wrapper for event handler functions."""
    
    def __init__(self, callback: Callable, 
                 filter_func: Optional[Callable] = None,
                 priority: int = 0,
                 once: bool = False):
        self.callback = callback
        self.filter_func = filter_func
        self.priority = priority
        self.once = once
        self.call_count = 0
    
    def can_handle(self, event: GameEvent) -> bool:
        """Check if handler can process event."""
        if self.once and self.call_count > 0:
            return False
        if self.filter_func:
            return self.filter_func(event)
        return True
    
    def handle(self, event: GameEvent) -> Any:
        """Process the event."""
        self.call_count += 1
        return self.callback(event)


class GameEventSystem:
    """Centralized game event management system."""
    
    def __init__(self, max_history: int = 1000):
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_queue: deque = deque()
        self.event_history: deque = deque(maxlen=max_history)
        self.processing = False
        self.thread = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Event statistics
        self.stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'handler_calls': 0,
            'errors': 0
        }
    
    def register_handler(self, event_type: Union[EventType, str],
                         callback: Callable,
                         filter_func: Optional[Callable] = None,
                         priority: int = 0,
                         once: bool = False) -> EventHandler:
        """Register an event handler."""
        handler = EventHandler(callback, filter_func, priority, once)
        
        with self.lock:
            event_key = event_type.value if isinstance(event_type, EventType) else event_type
            self.handlers[event_key].append(handler)
            # Sort by priority
            self.handlers[event_key].sort(key=lambda h: h.priority, reverse=True)
        
        self.logger.debug(f"Registered handler for {event_key}")
        return handler
    
    def unregister_handler(self, event_type: Union[EventType, str],
                          handler: EventHandler) -> bool:
        """Unregister an event handler."""
        with self.lock:
            event_key = event_type.value if isinstance(event_type, EventType) else event_type
            if event_key in self.handlers and handler in self.handlers[event_key]:
                self.handlers[event_key].remove(handler)
                self.logger.debug(f"Unregistered handler for {event_key}")
                return True
        return False
    
    def emit(self, event_type: Union[EventType, str],
             data: Optional[Dict[str, Any]] = None,
             source: Optional[str] = None,
             priority: int = 0) -> GameEvent:
        """Emit a game event."""
        event = GameEvent(
            type=event_type,
            data=data or {},
            source=source,
            priority=priority
        )
        
        with self.lock:
            self.event_queue.append(event)
            self.stats['total_events'] += 1
            event_key = event_type.value if isinstance(event_type, EventType) else event_type
            self.stats['events_by_type'][event_key] += 1
        
        self.logger.debug(f"Emitted event: {event}")
        
        # Process immediately if not running async
        if not self.processing:
            self._process_event(event)
        
        return event
    
    def _process_event(self, event: GameEvent):
        """Process a single event."""
        event_key = event.type.value if isinstance(event.type, EventType) else event.type
        
        # Add to history
        self.event_history.append(event)
        
        # Get handlers
        handlers = self.handlers.get(event_key, [])
        handlers.extend(self.handlers.get('*', []))  # Global handlers
        
        # Process handlers
        for handler in handlers:
            if handler.can_handle(event):
                try:
                    handler.handle(event)
                    self.stats['handler_calls'] += 1
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
                    self.stats['errors'] += 1
    
    def start_async_processing(self):
        """Start asynchronous event processing."""
        if self.processing:
            return
        
        self.processing = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        self.logger.info("Started async event processing")
    
    def stop_async_processing(self):
        """Stop asynchronous event processing."""
        self.processing = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.logger.info("Stopped async event processing")
    
    def _process_loop(self):
        """Main event processing loop."""
        while self.processing:
            if self.event_queue:
                with self.lock:
                    if self.event_queue:
                        event = self.event_queue.popleft()
                        self._process_event(event)
            else:
                time.sleep(0.001)  # Small delay when idle
    
    def wait_for_event(self, event_type: Union[EventType, str],
                       timeout: float = 30.0,
                       filter_func: Optional[Callable] = None) -> Optional[GameEvent]:
        """Wait for a specific event to occur."""
        received_event = None
        event_received = threading.Event()
        
        def handler(event):
            nonlocal received_event
            received_event = event
            event_received.set()
        
        handler_obj = self.register_handler(
            event_type, handler, filter_func, once=True
        )
        
        if event_received.wait(timeout):
            return received_event
        else:
            self.unregister_handler(event_type, handler_obj)
            return None
    
    def get_history(self, event_type: Optional[Union[EventType, str]] = None,
                   limit: int = 100) -> List[GameEvent]:
        """Get event history."""
        history = list(self.event_history)
        
        if event_type:
            event_key = event_type.value if isinstance(event_type, EventType) else event_type
            history = [e for e in history if 
                      (e.type.value if isinstance(e.type, EventType) else e.type) == event_key]
        
        return history[-limit:]
    
    def clear_handlers(self, event_type: Optional[Union[EventType, str]] = None):
        """Clear event handlers."""
        with self.lock:
            if event_type:
                event_key = event_type.value if isinstance(event_type, EventType) else event_type
                self.handlers[event_key] = []
            else:
                self.handlers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        return {
            'total_events': self.stats['total_events'],
            'events_by_type': dict(self.stats['events_by_type']),
            'handler_calls': self.stats['handler_calls'],
            'errors': self.stats['errors'],
            'queue_size': len(self.event_queue),
            'history_size': len(self.event_history),
            'handler_count': sum(len(h) for h in self.handlers.values())
        }
    
    def export_history(self, filepath: str):
        """Export event history to file."""
        history_data = [event.to_dict() for event in self.event_history]
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)


class GameEventRecorder:
    """Records game events for replay and analysis."""
    
    def __init__(self, event_system: GameEventSystem):
        self.event_system = event_system
        self.recording = False
        self.recorded_events: List[GameEvent] = []
        self.start_time = None
        
    def start_recording(self):
        """Start recording events."""
        self.recording = True
        self.recorded_events = []
        self.start_time = time.time()
        
        # Register global handler
        self.event_system.register_handler(
            '*', self._record_event, priority=100
        )
    
    def stop_recording(self) -> List[GameEvent]:
        """Stop recording and return events."""
        self.recording = False
        return self.recorded_events
    
    def _record_event(self, event: GameEvent):
        """Record an event."""
        if self.recording:
            # Adjust timestamp to be relative to recording start
            adjusted_event = GameEvent(
                type=event.type,
                timestamp=event.timestamp - self.start_time,
                data=event.data,
                source=event.source,
                priority=event.priority
            )
            self.recorded_events.append(adjusted_event)
    
    def replay_events(self, events: List[GameEvent], speed: float = 1.0):
        """Replay recorded events."""
        if not events:
            return
        
        start_time = time.time()
        
        for event in events:
            # Wait until event time
            target_time = start_time + (event.timestamp / speed)
            wait_time = target_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Re-emit event
            self.event_system.emit(
                event.type,
                event.data,
                source=f"replay_{event.source}" if event.source else "replay",
                priority=event.priority
            )


# Global event system instance
_global_event_system = None


def get_event_system() -> GameEventSystem:
    """Get the global event system instance."""
    global _global_event_system
    if _global_event_system is None:
        _global_event_system = GameEventSystem()
        _global_event_system.start_async_processing()
    return _global_event_system


def emit_event(event_type: Union[EventType, str], 
               data: Optional[Dict[str, Any]] = None,
               source: Optional[str] = None,
               priority: int = 0) -> GameEvent:
    """Convenience function to emit an event."""
    return get_event_system().emit(event_type, data, source, priority)


def on_event(event_type: Union[EventType, str],
             filter_func: Optional[Callable] = None,
             priority: int = 0,
             once: bool = False):
    """Decorator for event handlers."""
    def decorator(func):
        get_event_system().register_handler(
            event_type, func, filter_func, priority, once
        )
        return func
    return decorator