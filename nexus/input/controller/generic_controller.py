"""
Generic Gamepad Controller Support

Provides support for generic gamepads and unknown controller types.
"""

import time
from typing import Optional, Dict, Any
import structlog

from nexus.input.controller.gamepad_base import (
    GamepadBase, Button, Axis, ControllerState, ControllerType
)

logger = structlog.get_logger()

# Try to import pygame for generic controller support
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class GenericController(GamepadBase):
    """
    Generic gamepad implementation for unknown controller types.
    
    Features:
    - Automatic button mapping
    - Configurable button layouts
    - Profile support for known generic controllers
    - Basic vibration support
    """
    
    # Common generic controller profiles
    PROFILES = {
        'logitech_f310': {
            'buttons': {
                0: Button.A, 1: Button.B, 2: Button.X, 3: Button.Y,
                4: Button.LB, 5: Button.RB,
                6: Button.BACK, 7: Button.START, 8: Button.GUIDE,
                9: Button.LEFT_STICK, 10: Button.RIGHT_STICK
            },
            'axes': {
                0: Axis.LEFT_X, 1: Axis.LEFT_Y,
                2: Axis.RIGHT_X, 3: Axis.RIGHT_Y
            },
            'triggers_as_axes': False,
            'dpad_as_hat': True
        },
        'generic_xinput': {
            'buttons': {
                0: Button.A, 1: Button.B, 2: Button.X, 3: Button.Y,
                4: Button.LB, 5: Button.RB,
                6: Button.BACK, 7: Button.START, 8: Button.GUIDE,
                9: Button.LEFT_STICK, 10: Button.RIGHT_STICK
            },
            'axes': {
                0: Axis.LEFT_X, 1: Axis.LEFT_Y,
                2: Axis.LEFT_TRIGGER, 3: Axis.RIGHT_X,
                4: Axis.RIGHT_Y, 5: Axis.RIGHT_TRIGGER
            },
            'triggers_as_axes': True,
            'dpad_as_hat': True
        },
        'generic_directinput': {
            'buttons': {
                0: Button.Y, 1: Button.B, 2: Button.A, 3: Button.X,
                4: Button.LB, 5: Button.RB,
                6: Button.BACK, 7: Button.START,
                8: Button.LEFT_STICK, 9: Button.RIGHT_STICK
            },
            'axes': {
                0: Axis.LEFT_X, 1: Axis.LEFT_Y,
                2: Axis.RIGHT_Y, 3: Axis.RIGHT_X,
                4: Axis.LEFT_TRIGGER, 5: Axis.RIGHT_TRIGGER
            },
            'triggers_as_axes': True,
            'dpad_as_hat': True
        }
    }
    
    def __init__(self, controller_id: int = 0, profile: str = None):
        """
        Initialize generic controller.
        
        Args:
            controller_id: Controller index
            profile: Optional profile name for known controllers
        """
        super().__init__(controller_id)
        
        self.controller_type = ControllerType.GENERIC
        self.pygame_joystick = None
        
        # Load profile or use auto-detection
        if profile and profile in self.PROFILES:
            self.mapping = self.PROFILES[profile]
            logger.info(f"Using profile '{profile}' for controller {controller_id}")
        else:
            self.mapping = None  # Will auto-detect
        
        # Auto-detected mapping
        self.button_map = {}
        self.axis_map = {}
        self.has_hat = False
        self.triggers_as_axes = False
        
        if not HAS_PYGAME:
            logger.error("pygame not available for generic controller support")
            raise RuntimeError("Generic controller support requires pygame")
    
    def connect(self) -> bool:
        """
        Connect to generic controller.
        
        Returns:
            True if connected successfully
        """
        try:
            pygame.init()
            pygame.joystick.init()
            
            if self.controller_id >= pygame.joystick.get_count():
                logger.warning(f"Controller {self.controller_id} not found")
                return False
            
            self.pygame_joystick = pygame.joystick.Joystick(self.controller_id)
            self.pygame_joystick.init()
            
            # Get controller info
            name = self.pygame_joystick.get_name()
            num_buttons = self.pygame_joystick.get_numbuttons()
            num_axes = self.pygame_joystick.get_numaxes()
            num_hats = self.pygame_joystick.get_numhats()
            
            logger.info(f"Connected to '{name}' - "
                       f"Buttons: {num_buttons}, Axes: {num_axes}, Hats: {num_hats}")
            
            # Auto-detect mapping if not provided
            if not self.mapping:
                self._auto_detect_mapping(name, num_buttons, num_axes, num_hats)
            else:
                self._apply_mapping(self.mapping)
            
            self.state.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect generic controller: {e}")
            return False
    
    def _auto_detect_mapping(self, name: str, num_buttons: int, 
                            num_axes: int, num_hats: int):
        """Auto-detect controller mapping based on capabilities."""
        name_lower = name.lower()
        
        # Try to detect known controllers
        if 'logitech' in name_lower and 'f310' in name_lower:
            self.mapping = self.PROFILES['logitech_f310']
        elif 'xinput' in name_lower or 'xbox' in name_lower:
            self.mapping = self.PROFILES['generic_xinput']
        elif 'directinput' in name_lower or 'direct' in name_lower:
            self.mapping = self.PROFILES['generic_directinput']
        else:
            # Create default mapping based on number of controls
            self._create_default_mapping(num_buttons, num_axes, num_hats)
            return
        
        self._apply_mapping(self.mapping)
    
    def _create_default_mapping(self, num_buttons: int, num_axes: int, num_hats: int):
        """Create default mapping for unknown controller."""
        logger.info("Creating default mapping for unknown controller")
        
        # Map buttons (assuming standard layout)
        standard_buttons = [
            Button.A, Button.B, Button.X, Button.Y,
            Button.LB, Button.RB,
            Button.BACK, Button.START, Button.GUIDE,
            Button.LEFT_STICK, Button.RIGHT_STICK
        ]
        
        for i in range(min(num_buttons, len(standard_buttons))):
            self.button_map[i] = standard_buttons[i]
        
        # Map axes
        if num_axes >= 2:
            self.axis_map[0] = Axis.LEFT_X
            self.axis_map[1] = Axis.LEFT_Y
        
        if num_axes >= 4:
            self.axis_map[2] = Axis.RIGHT_X
            self.axis_map[3] = Axis.RIGHT_Y
        
        if num_axes >= 6:
            # Assume triggers are axes
            self.axis_map[4] = Axis.LEFT_TRIGGER
            self.axis_map[5] = Axis.RIGHT_TRIGGER
            self.triggers_as_axes = True
        elif num_buttons >= 13:
            # Triggers might be buttons
            self.button_map[11] = Button.LB  # Remap if needed
            self.button_map[12] = Button.RB
        
        # Check for D-pad
        self.has_hat = num_hats > 0
    
    def _apply_mapping(self, mapping: Dict[str, Any]):
        """Apply a controller mapping profile."""
        self.button_map = mapping.get('buttons', {})
        self.axis_map = mapping.get('axes', {})
        self.triggers_as_axes = mapping.get('triggers_as_axes', False)
        self.has_hat = mapping.get('dpad_as_hat', True)
    
    def disconnect(self):
        """Disconnect from controller."""
        self.stop_polling()
        
        if self.pygame_joystick:
            self.pygame_joystick.quit()
            self.pygame_joystick = None
        
        self.state.is_connected = False
        logger.info(f"Generic controller {self.controller_id} disconnected")
    
    def poll(self) -> Optional[ControllerState]:
        """
        Poll current controller state.
        
        Returns:
            Current controller state or None if disconnected
        """
        if not self.pygame_joystick:
            return None
        
        try:
            pygame.event.pump()
            
            new_state = ControllerState(
                controller_id=self.controller_id,
                controller_type=self.controller_type,
                is_connected=True
            )
            
            # Map buttons
            for pygame_btn, button in self.button_map.items():
                if pygame_btn < self.pygame_joystick.get_numbuttons():
                    new_state.buttons[button] = bool(
                        self.pygame_joystick.get_button(pygame_btn)
                    )
            
            # Map axes
            for pygame_axis, axis in self.axis_map.items():
                if pygame_axis < self.pygame_joystick.get_numaxes():
                    value = self.pygame_joystick.get_axis(pygame_axis)
                    
                    # Special handling for triggers
                    if axis in [Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER]:
                        if self.triggers_as_axes:
                            # Convert from -1,1 to 0,1 if needed
                            if value < 0:
                                value = (value + 1.0) / 2.0
                            else:
                                # Already in 0,1 range
                                pass
                    elif axis in [Axis.LEFT_Y, Axis.RIGHT_Y]:
                        # Invert Y axes
                        value = -value
                    
                    new_state.axes[axis] = value
            
            # Map D-pad from hat
            if self.has_hat and self.pygame_joystick.get_numhats() > 0:
                hat_x, hat_y = self.pygame_joystick.get_hat(0)
                new_state.buttons[Button.DPAD_UP] = hat_y > 0
                new_state.buttons[Button.DPAD_DOWN] = hat_y < 0
                new_state.buttons[Button.DPAD_LEFT] = hat_x < 0
                new_state.buttons[Button.DPAD_RIGHT] = hat_x > 0
            
            # Apply deadzones
            new_state = self._apply_deadzones(new_state)
            
            return new_state
            
        except Exception as e:
            logger.error(f"Polling error: {e}")
            return None
    
    def _apply_deadzones(self, state: ControllerState) -> ControllerState:
        """Apply deadzones to analog inputs."""
        # Apply stick deadzones
        for axis in [Axis.LEFT_X, Axis.LEFT_Y, Axis.RIGHT_X, Axis.RIGHT_Y]:
            if axis in state.axes:
                value = state.axes[axis]
                if abs(value) < self.stick_deadzone:
                    state.axes[axis] = 0.0
                else:
                    # Rescale
                    sign = 1 if value > 0 else -1
                    state.axes[axis] = sign * (abs(value) - self.stick_deadzone) / (1.0 - self.stick_deadzone)
        
        # Apply trigger deadzones
        for axis in [Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER]:
            if axis in state.axes:
                value = state.axes[axis]
                if value < self.trigger_deadzone:
                    state.axes[axis] = 0.0
                else:
                    state.axes[axis] = (value - self.trigger_deadzone) / (1.0 - self.trigger_deadzone)
        
        return state
    
    def vibrate(self, left_motor: float = 0.0, right_motor: float = 0.0,
                duration_ms: int = 100):
        """
        Trigger controller vibration.
        
        Args:
            left_motor: Left motor strength (0.0-1.0)
            right_motor: Right motor strength (0.0-1.0)
            duration_ms: Duration in milliseconds
        """
        if not self.vibration_enabled or not self.pygame_joystick:
            return
        
        try:
            # Try pygame rumble if available
            if hasattr(self.pygame_joystick, 'rumble'):
                left_motor = max(0.0, min(1.0, left_motor))
                right_motor = max(0.0, min(1.0, right_motor))
                self.pygame_joystick.rumble(left_motor, right_motor, duration_ms)
            else:
                logger.debug("Vibration not supported for this controller")
        except Exception as e:
            logger.debug(f"Vibration error: {e}")
    
    def calibrate_mapping(self) -> Dict[str, Any]:
        """
        Interactive calibration to create custom mapping.
        
        Returns:
            Custom mapping dictionary
        """
        if not self.pygame_joystick:
            logger.error("Controller not connected")
            return {}
        
        logger.info("Starting interactive calibration...")
        custom_map = {
            'buttons': {},
            'axes': {},
            'triggers_as_axes': False,
            'dpad_as_hat': False
        }
        
        # Calibrate buttons
        button_prompts = [
            (Button.A, "Press A/Cross button"),
            (Button.B, "Press B/Circle button"),
            (Button.X, "Press X/Square button"),
            (Button.Y, "Press Y/Triangle button"),
            (Button.LB, "Press Left Bumper/L1"),
            (Button.RB, "Press Right Bumper/R1"),
            (Button.BACK, "Press Back/Select/Share button"),
            (Button.START, "Press Start/Options button"),
            (Button.GUIDE, "Press Guide/Home/PS button"),
            (Button.LEFT_STICK, "Press Left Stick (L3)"),
            (Button.RIGHT_STICK, "Press Right Stick (R3)")
        ]
        
        for button, prompt in button_prompts:
            logger.info(prompt)
            pressed = self.wait_for_button(timeout=5.0)
            if pressed is not None:
                # Find pygame button index
                for i in range(self.pygame_joystick.get_numbuttons()):
                    if self.pygame_joystick.get_button(i):
                        custom_map['buttons'][i] = button
                        logger.info(f"Mapped pygame button {i} to {button.name}")
                        break
        
        # Calibrate axes
        logger.info("Move left stick to calibrate axes...")
        time.sleep(2)
        # Would continue with axis calibration
        
        return custom_map
    
    def save_profile(self, name: str, mapping: Dict[str, Any]):
        """
        Save a custom controller profile.
        
        Args:
            name: Profile name
            mapping: Mapping dictionary
        """
        # In a real implementation, this would save to a config file
        self.PROFILES[name] = mapping
        logger.info(f"Saved profile '{name}'")
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller information."""
        if not self.pygame_joystick:
            return {}
        
        return {
            'name': self.pygame_joystick.get_name(),
            'id': self.pygame_joystick.get_id(),
            'guid': self.pygame_joystick.get_guid() if hasattr(self.pygame_joystick, 'get_guid') else None,
            'num_buttons': self.pygame_joystick.get_numbuttons(),
            'num_axes': self.pygame_joystick.get_numaxes(),
            'num_hats': self.pygame_joystick.get_numhats(),
            'mapping': self.mapping or 'auto-detected'
        }