"""
Virtual Gamepad for Testing and Emulation

Provides a software-emulated gamepad for testing and automation.
"""

import time
import threading
import queue
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import structlog

from nexus.input.gamepad.gamepad_base import (
    GamepadBase, Button, Axis, ControllerState, ControllerType, ControllerEvent
)

logger = structlog.get_logger()


class VirtualGamepad(GamepadBase):
    """
    Virtual gamepad implementation for testing and automation.
    
    Features:
    - Programmatic control simulation
    - Action recording and playback
    - Macro support
    - Input sequence generation
    - AI agent integration
    """
    
    def __init__(self, controller_id: int = 0):
        """
        Initialize virtual gamepad.
        
        Args:
            controller_id: Controller identifier
        """
        super().__init__(controller_id)
        
        self.controller_type = ControllerType.VIRTUAL
        
        # Action queue for programmatic control
        self.action_queue = queue.Queue()
        
        # Macro system
        self.macros: Dict[str, List[Dict[str, Any]]] = {}
        self.active_macros: List[threading.Thread] = []
        
        # Recording system
        self.is_recording = False
        self.recorded_actions: List[Dict[str, Any]] = []
        self.recording_start_time = 0
        
        # Playback system
        self.is_playing = False
        self.playback_thread = None
        self.playback_speed = 1.0
        
        # AI control
        self.ai_controller: Optional[Callable] = None
        
        logger.info(f"Virtual gamepad {controller_id} initialized")
    
    def connect(self) -> bool:
        """
        Connect virtual controller (always succeeds).
        
        Returns:
            True
        """
        self.state.is_connected = True
        
        # Initialize default state
        for button in Button:
            self.state.buttons[button] = False
        
        for axis in Axis:
            self.state.axes[axis] = 0.0
        
        logger.info(f"Virtual gamepad {self.controller_id} connected")
        return True
    
    def disconnect(self):
        """Disconnect virtual controller."""
        self.stop_polling()
        self.stop_recording()
        self.stop_playback()
        self.state.is_connected = False
        logger.info(f"Virtual gamepad {self.controller_id} disconnected")
    
    def poll(self) -> Optional[ControllerState]:
        """
        Poll virtual controller state.
        
        Returns:
            Current controller state
        """
        if not self.state.is_connected:
            return None
        
        # Process queued actions
        while not self.action_queue.empty():
            try:
                action = self.action_queue.get_nowait()
                self._apply_action(action)
            except queue.Empty:
                break
        
        # Apply AI control if configured
        if self.ai_controller:
            ai_state = self.ai_controller(self.state)
            if ai_state:
                self.state = ai_state
        
        # Record if needed
        if self.is_recording:
            self._record_state()
        
        return self.state.copy()
    
    def vibrate(self, left_motor: float = 0.0, right_motor: float = 0.0,
                duration_ms: int = 100):
        """
        Simulate vibration (logs only for virtual controller).
        
        Args:
            left_motor: Left motor strength (0.0-1.0)
            right_motor: Right motor strength (0.0-1.0)
            duration_ms: Duration in milliseconds
        """
        logger.debug(f"Virtual vibration: L={left_motor:.2f}, R={right_motor:.2f}, {duration_ms}ms")
    
    # Virtual control methods
    
    def press_button(self, button: Button, duration_ms: int = 0):
        """
        Press a button.
        
        Args:
            button: Button to press
            duration_ms: Hold duration (0 = instant press/release)
        """
        self.action_queue.put({
            'type': 'button_press',
            'button': button,
            'duration': duration_ms
        })
    
    def release_button(self, button: Button):
        """
        Release a button.
        
        Args:
            button: Button to release
        """
        self.action_queue.put({
            'type': 'button_release',
            'button': button
        })
    
    def move_stick(self, stick: str, x: float, y: float):
        """
        Move an analog stick.
        
        Args:
            stick: 'left' or 'right'
            x: X position (-1.0 to 1.0)
            y: Y position (-1.0 to 1.0)
        """
        if stick == 'left':
            self.action_queue.put({
                'type': 'axis_move',
                'axis': Axis.LEFT_X,
                'value': max(-1.0, min(1.0, x))
            })
            self.action_queue.put({
                'type': 'axis_move',
                'axis': Axis.LEFT_Y,
                'value': max(-1.0, min(1.0, y))
            })
        elif stick == 'right':
            self.action_queue.put({
                'type': 'axis_move',
                'axis': Axis.RIGHT_X,
                'value': max(-1.0, min(1.0, x))
            })
            self.action_queue.put({
                'type': 'axis_move',
                'axis': Axis.RIGHT_Y,
                'value': max(-1.0, min(1.0, y))
            })
    
    def pull_trigger(self, trigger: str, value: float):
        """
        Pull a trigger.
        
        Args:
            trigger: 'left' or 'right'
            value: Trigger value (0.0 to 1.0)
        """
        axis = Axis.LEFT_TRIGGER if trigger == 'left' else Axis.RIGHT_TRIGGER
        self.action_queue.put({
            'type': 'axis_move',
            'axis': axis,
            'value': max(0.0, min(1.0, value))
        })
    
    def reset_state(self):
        """Reset controller to default state."""
        for button in Button:
            self.state.buttons[button] = False
        
        for axis in Axis:
            self.state.axes[axis] = 0.0
    
    # Macro system
    
    def create_macro(self, name: str, actions: List[Dict[str, Any]]):
        """
        Create a macro sequence.
        
        Args:
            name: Macro name
            actions: List of actions with timing
        """
        self.macros[name] = actions
        logger.info(f"Created macro '{name}' with {len(actions)} actions")
    
    def execute_macro(self, name: str, repeat: int = 1):
        """
        Execute a macro.
        
        Args:
            name: Macro name
            repeat: Number of repetitions
        """
        if name not in self.macros:
            logger.warning(f"Macro '{name}' not found")
            return
        
        def run_macro():
            for _ in range(repeat):
                for action in self.macros[name]:
                    self._apply_action(action)
                    if 'delay' in action:
                        time.sleep(action['delay'] / 1000.0)
        
        thread = threading.Thread(target=run_macro, daemon=True)
        thread.start()
        self.active_macros.append(thread)
    
    def create_combo(self, buttons: List[Button], timing_ms: int = 50):
        """
        Create a button combo.
        
        Args:
            buttons: Sequence of buttons
            timing_ms: Timing between presses
        """
        actions = []
        for button in buttons:
            actions.append({
                'type': 'button_press',
                'button': button,
                'duration': timing_ms
            })
            actions.append({'delay': timing_ms})
        
        return actions
    
    # Recording and playback
    
    def start_recording(self):
        """Start recording controller actions."""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        self.recorded_actions = []
        self.recording_start_time = time.time()
        self.is_recording = True
        logger.info("Started recording virtual gamepad actions")
    
    def stop_recording(self) -> List[Dict[str, Any]]:
        """
        Stop recording and return recorded actions.
        
        Returns:
            List of recorded actions
        """
        if not self.is_recording:
            return []
        
        self.is_recording = False
        logger.info(f"Stopped recording. Captured {len(self.recorded_actions)} actions")
        return self.recorded_actions
    
    def start_playback(self, actions: List[Dict[str, Any]], loop: bool = False):
        """
        Start playing back recorded actions.
        
        Args:
            actions: Actions to play back
            loop: Whether to loop playback
        """
        if self.is_playing:
            logger.warning("Already playing")
            return
        
        self.is_playing = True
        
        def playback_loop():
            while self.is_playing:
                for action in actions:
                    if not self.is_playing:
                        break
                    
                    # Apply action
                    self._apply_action(action)
                    
                    # Wait for next action
                    if 'timestamp' in action and actions.index(action) < len(actions) - 1:
                        next_action = actions[actions.index(action) + 1]
                        if 'timestamp' in next_action:
                            delay = (next_action['timestamp'] - action['timestamp']) / self.playback_speed
                            time.sleep(max(0, delay))
                
                if not loop:
                    break
            
            self.is_playing = False
        
        self.playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self.playback_thread.start()
        logger.info("Started playback")
    
    def stop_playback(self):
        """Stop playback."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        logger.info("Stopped playback")
    
    # AI integration
    
    def set_ai_controller(self, controller: Callable[[ControllerState], ControllerState]):
        """
        Set AI controller function.
        
        Args:
            controller: Function that takes current state and returns new state
        """
        self.ai_controller = controller
        logger.info("AI controller configured")
    
    def generate_random_input(self, intensity: float = 0.5) -> ControllerState:
        """
        Generate random input for testing.
        
        Args:
            intensity: Input intensity (0.0-1.0)
        
        Returns:
            Random controller state
        """
        state = ControllerState(
            controller_id=self.controller_id,
            controller_type=self.controller_type
        )
        
        # Random buttons
        for button in Button:
            if np.random.random() < intensity * 0.2:
                state.buttons[button] = True
        
        # Random axes
        for axis in [Axis.LEFT_X, Axis.LEFT_Y, Axis.RIGHT_X, Axis.RIGHT_Y]:
            if np.random.random() < intensity:
                state.axes[axis] = np.random.uniform(-intensity, intensity)
        
        # Random triggers
        for axis in [Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER]:
            if np.random.random() < intensity * 0.5:
                state.axes[axis] = np.random.uniform(0, intensity)
        
        return state
    
    # Private methods
    
    def _apply_action(self, action: Dict[str, Any]):
        """Apply an action to the controller state."""
        action_type = action.get('type')
        
        if action_type == 'button_press':
            button = action['button']
            self.state.buttons[button] = True
            
            # Fire event
            event = ControllerEvent(
                event_type='button_press',
                control=button,
                value=True,
                controller_id=self.controller_id
            )
            self._fire_event(event)
            
            # Handle duration
            duration = action.get('duration', 0)
            if duration > 0:
                def release():
                    time.sleep(duration / 1000.0)
                    self.release_button(button)
                
                threading.Thread(target=release, daemon=True).start()
        
        elif action_type == 'button_release':
            button = action['button']
            self.state.buttons[button] = False
            
            # Fire event
            event = ControllerEvent(
                event_type='button_release',
                control=button,
                value=False,
                controller_id=self.controller_id
            )
            self._fire_event(event)
        
        elif action_type == 'axis_move':
            axis = action['axis']
            value = action['value']
            self.state.axes[axis] = value
            
            # Fire event
            event = ControllerEvent(
                event_type='axis_move',
                control=axis,
                value=value,
                controller_id=self.controller_id
            )
            self._fire_event(event)
    
    def _record_state(self):
        """Record current state for playback."""
        timestamp = time.time() - self.recording_start_time
        
        # Record button changes
        for button, pressed in self.state.buttons.items():
            prev_pressed = self.previous_state.buttons.get(button, False)
            if pressed != prev_pressed:
                self.recorded_actions.append({
                    'type': 'button_press' if pressed else 'button_release',
                    'button': button,
                    'timestamp': timestamp
                })
        
        # Record axis changes
        for axis, value in self.state.axes.items():
            prev_value = self.previous_state.axes.get(axis, 0.0)
            if abs(value - prev_value) > 0.01:
                self.recorded_actions.append({
                    'type': 'axis_move',
                    'axis': axis,
                    'value': value,
                    'timestamp': timestamp
                })
    
    def save_recording(self, filepath: str):
        """
        Save recorded actions to file.
        
        Args:
            filepath: Path to save file
        """
        import json
        
        # Convert enums to strings for JSON serialization
        serializable_actions = []
        for action in self.recorded_actions:
            s_action = action.copy()
            if 'button' in s_action:
                s_action['button'] = s_action['button'].name
            if 'axis' in s_action:
                s_action['axis'] = s_action['axis'].name
            serializable_actions.append(s_action)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_actions, f, indent=2)
        
        logger.info(f"Saved recording to {filepath}")
    
    def load_recording(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load recorded actions from file.
        
        Args:
            filepath: Path to recording file
        
        Returns:
            List of actions
        """
        import json
        
        with open(filepath, 'r') as f:
            serializable_actions = json.load(f)
        
        # Convert strings back to enums
        actions = []
        for s_action in serializable_actions:
            action = s_action.copy()
            if 'button' in action:
                action['button'] = Button[action['button']]
            if 'axis' in action:
                action['axis'] = Axis[action['axis']]
            actions.append(action)
        
        logger.info(f"Loaded recording from {filepath}")
        return actions