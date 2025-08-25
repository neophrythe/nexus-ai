"""
Base Classes for Gamepad/Controller Support

Provides abstract base classes and common data structures for
gamepad implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import threading
import numpy as np
import structlog

logger = structlog.get_logger()


class Button(IntEnum):
    """Standard gamepad buttons."""
    # Face buttons
    A = 0
    B = 1
    X = 2
    Y = 3
    
    # Shoulder buttons
    LB = 4  # Left bumper
    RB = 5  # Right bumper
    
    # Control buttons
    BACK = 6
    START = 7
    GUIDE = 8  # Xbox/PS button
    
    # Stick buttons
    LEFT_STICK = 9
    RIGHT_STICK = 10
    
    # D-Pad
    DPAD_UP = 11
    DPAD_DOWN = 12
    DPAD_LEFT = 13
    DPAD_RIGHT = 14


class Axis(IntEnum):
    """Standard gamepad axes."""
    LEFT_X = 0
    LEFT_Y = 1
    RIGHT_X = 2
    RIGHT_Y = 3
    LEFT_TRIGGER = 4
    RIGHT_TRIGGER = 5


class ControllerType(Enum):
    """Supported controller types."""
    XBOX = "xbox"
    XBOX_360 = "xbox360"
    XBOX_ONE = "xboxone"
    XBOX_SERIES = "xboxseries"
    PS3 = "ps3"
    PS4 = "ps4"
    PS5 = "ps5"
    GENERIC = "generic"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


@dataclass
class ControllerState:
    """
    Complete state of a game controller.
    """
    # Button states (pressed=True)
    buttons: Dict[Button, bool] = field(default_factory=dict)
    
    # Axis values (-1.0 to 1.0 for sticks, 0.0 to 1.0 for triggers)
    axes: Dict[Axis, float] = field(default_factory=dict)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    # Controller info
    controller_id: int = 0
    controller_type: ControllerType = ControllerType.UNKNOWN
    
    # Additional features
    battery_level: Optional[float] = None  # 0.0 to 1.0
    is_connected: bool = True
    
    def copy(self) -> 'ControllerState':
        """Create a deep copy of the state."""
        return ControllerState(
            buttons=self.buttons.copy(),
            axes=self.axes.copy(),
            timestamp=self.timestamp,
            controller_id=self.controller_id,
            controller_type=self.controller_type,
            battery_level=self.battery_level,
            is_connected=self.is_connected
        )
    
    def get_pressed_buttons(self) -> List[Button]:
        """Get list of currently pressed buttons."""
        return [btn for btn, pressed in self.buttons.items() if pressed]
    
    def get_active_axes(self, deadzone: float = 0.1) -> Dict[Axis, float]:
        """Get axes with values beyond deadzone."""
        return {axis: value for axis, value in self.axes.items() 
                if abs(value) > deadzone}
    
    def to_vector(self) -> np.ndarray:
        """Convert state to numerical vector for ML models."""
        # Create fixed-size vector
        vector = np.zeros(21)  # 15 buttons + 6 axes
        
        # Add button states
        for i in range(15):
            if Button(i) in self.buttons:
                vector[i] = float(self.buttons[Button(i)])
        
        # Add axis values
        for i in range(6):
            if Axis(i) in self.axes:
                vector[15 + i] = self.axes[Axis(i)]
        
        return vector
    
    @staticmethod
    def from_vector(vector: np.ndarray) -> 'ControllerState':
        """Create state from numerical vector."""
        state = ControllerState()
        
        # Parse button states
        for i in range(15):
            state.buttons[Button(i)] = bool(vector[i] > 0.5)
        
        # Parse axis values
        for i in range(6):
            state.axes[Axis(i)] = float(vector[15 + i])
        
        return state


@dataclass
class ControllerEvent:
    """Controller input event."""
    event_type: str  # 'button_press', 'button_release', 'axis_move'
    control: Any  # Button or Axis enum
    value: Any  # bool for buttons, float for axes
    timestamp: float = field(default_factory=time.time)
    controller_id: int = 0


class GamepadBase(ABC):
    """
    Abstract base class for gamepad implementations.
    """
    
    def __init__(self, controller_id: int = 0):
        """
        Initialize gamepad.
        
        Args:
            controller_id: Controller identifier (0-3 typically)
        """
        self.controller_id = controller_id
        self.controller_type = ControllerType.UNKNOWN
        
        # Current state
        self.state = ControllerState(controller_id=controller_id)
        self.previous_state = self.state.copy()
        
        # Polling thread
        self.polling_thread = None
        self.polling_rate = 60  # Hz
        self.is_polling = False
        
        # Event callbacks
        self.event_callbacks: List[Callable[[ControllerEvent], None]] = []
        
        # Deadzone settings
        self.stick_deadzone = 0.1
        self.trigger_deadzone = 0.05
        
        # Vibration/haptics
        self.vibration_enabled = True
        
        logger.info(f"Gamepad {controller_id} initialized")
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the controller."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the controller."""
        pass
    
    @abstractmethod
    def poll(self) -> ControllerState:
        """Poll current controller state."""
        pass
    
    @abstractmethod
    def vibrate(self, left_motor: float = 0.0, right_motor: float = 0.0,
                duration_ms: int = 100):
        """Trigger controller vibration."""
        pass
    
    def start_polling(self, rate: int = 60):
        """Start automatic polling."""
        if self.is_polling:
            logger.warning(f"Controller {self.controller_id} already polling")
            return
        
        self.polling_rate = rate
        self.is_polling = True
        self.polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self.polling_thread.start()
        
        logger.info(f"Started polling controller {self.controller_id} at {rate} Hz")
    
    def stop_polling(self):
        """Stop automatic polling."""
        self.is_polling = False
        
        if self.polling_thread:
            self.polling_thread.join(timeout=1.0)
            self.polling_thread = None
        
        logger.info(f"Stopped polling controller {self.controller_id}")
    
    def _polling_loop(self):
        """Main polling loop."""
        poll_interval = 1.0 / self.polling_rate
        
        while self.is_polling:
            try:
                # Poll controller
                new_state = self.poll()
                
                if new_state:
                    # Check for changes and fire events
                    self._process_state_change(new_state)
                    
                    # Update states
                    self.previous_state = self.state
                    self.state = new_state
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Polling error on controller {self.controller_id}: {e}")
                time.sleep(0.1)
    
    def _process_state_change(self, new_state: ControllerState):
        """Process state changes and fire events."""
        # Check button changes
        for button in Button:
            old_pressed = self.state.buttons.get(button, False)
            new_pressed = new_state.buttons.get(button, False)
            
            if old_pressed != new_pressed:
                event = ControllerEvent(
                    event_type='button_press' if new_pressed else 'button_release',
                    control=button,
                    value=new_pressed,
                    controller_id=self.controller_id
                )
                self._fire_event(event)
        
        # Check axis changes
        for axis in Axis:
            old_value = self.state.axes.get(axis, 0.0)
            new_value = new_state.axes.get(axis, 0.0)
            
            # Apply deadzone
            deadzone = self.trigger_deadzone if axis in [Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER] else self.stick_deadzone
            
            if abs(new_value - old_value) > deadzone:
                event = ControllerEvent(
                    event_type='axis_move',
                    control=axis,
                    value=new_value,
                    controller_id=self.controller_id
                )
                self._fire_event(event)
    
    def _fire_event(self, event: ControllerEvent):
        """Fire event to all registered callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def add_event_listener(self, callback: Callable[[ControllerEvent], None]):
        """Add event listener."""
        self.event_callbacks.append(callback)
    
    def remove_event_listener(self, callback: Callable[[ControllerEvent], None]):
        """Remove event listener."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    def get_state(self) -> ControllerState:
        """Get current controller state."""
        return self.state.copy()
    
    def is_button_pressed(self, button: Button) -> bool:
        """Check if button is pressed."""
        return self.state.buttons.get(button, False)
    
    def get_axis_value(self, axis: Axis) -> float:
        """Get axis value."""
        return self.state.axes.get(axis, 0.0)
    
    def get_left_stick(self) -> Tuple[float, float]:
        """Get left stick position."""
        x = self.get_axis_value(Axis.LEFT_X)
        y = self.get_axis_value(Axis.LEFT_Y)
        return (x, y)
    
    def get_right_stick(self) -> Tuple[float, float]:
        """Get right stick position."""
        x = self.get_axis_value(Axis.RIGHT_X)
        y = self.get_axis_value(Axis.RIGHT_Y)
        return (x, y)
    
    def get_triggers(self) -> Tuple[float, float]:
        """Get trigger values."""
        left = self.get_axis_value(Axis.LEFT_TRIGGER)
        right = self.get_axis_value(Axis.RIGHT_TRIGGER)
        return (left, right)
    
    def wait_for_button(self, timeout: float = None) -> Optional[Button]:
        """
        Wait for any button press.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Pressed button or None if timeout
        """
        pressed_button = None
        event_received = threading.Event()
        
        def button_listener(event: ControllerEvent):
            nonlocal pressed_button
            if event.event_type == 'button_press':
                pressed_button = event.control
                event_received.set()
        
        self.add_event_listener(button_listener)
        
        try:
            if event_received.wait(timeout):
                return pressed_button
            return None
        finally:
            self.remove_event_listener(button_listener)
    
    def calibrate(self):
        """
        Calibrate controller (center sticks, etc.).
        Override in subclasses if needed.
        """
        logger.info(f"Calibrating controller {self.controller_id}...")
        
        # Get current state as center position
        state = self.poll()
        if state:
            # Store center positions for analog sticks
            self._calibration = {
                'left_x_center': state.axes.get(Axis.LEFT_X, 0.0),
                'left_y_center': state.axes.get(Axis.LEFT_Y, 0.0),
                'right_x_center': state.axes.get(Axis.RIGHT_X, 0.0),
                'right_y_center': state.axes.get(Axis.RIGHT_Y, 0.0)
            }
            logger.info(f"Controller {self.controller_id} calibrated")
        else:
            logger.warning(f"Failed to calibrate controller {self.controller_id}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        self.start_polling()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_polling()
        self.disconnect()