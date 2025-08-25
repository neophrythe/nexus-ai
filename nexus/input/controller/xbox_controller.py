"""
Xbox Controller Support via XInput

Provides support for Xbox 360, Xbox One, and Xbox Series controllers
using the XInput API on Windows or alternative libraries on other platforms.
"""

import sys
import time
import struct
from typing import Optional, Tuple
import structlog

from nexus.input.controller.gamepad_base import (
    GamepadBase, Button, Axis, ControllerState, ControllerType
)

logger = structlog.get_logger()

# Try to import platform-specific libraries
try:
    import XInput  # Windows XInput wrapper
    HAS_XINPUT = True
except ImportError:
    HAS_XINPUT = False

try:
    import evdev  # Linux support
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False

try:
    import pygame  # Cross-platform fallback
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class XboxController(GamepadBase):
    """
    Xbox controller implementation with XInput support.
    
    Features:
    - Native XInput support on Windows
    - Cross-platform fallback using pygame
    - Vibration/haptic feedback
    - Battery level monitoring
    - Automatic dead zone handling
    """
    
    # XInput button mapping
    XINPUT_BUTTON_MAP = {
        0x0001: Button.DPAD_UP,
        0x0002: Button.DPAD_DOWN,
        0x0004: Button.DPAD_LEFT,
        0x0008: Button.DPAD_RIGHT,
        0x0010: Button.START,
        0x0020: Button.BACK,
        0x0040: Button.LEFT_STICK,
        0x0080: Button.RIGHT_STICK,
        0x0100: Button.LB,
        0x0200: Button.RB,
        0x1000: Button.A,
        0x2000: Button.B,
        0x4000: Button.X,
        0x8000: Button.Y
    }
    
    def __init__(self, controller_id: int = 0):
        """
        Initialize Xbox controller.
        
        Args:
            controller_id: Controller index (0-3)
        """
        super().__init__(controller_id)
        
        self.controller_type = ControllerType.XBOX
        self.backend = None
        self.xinput_state = None
        self.pygame_joystick = None
        
        # XInput specific
        self.packet_number = 0
        
        # Detect available backend
        if HAS_XINPUT and sys.platform == 'win32':
            self.backend = 'xinput'
            logger.info(f"Using XInput backend for controller {controller_id}")
        elif HAS_PYGAME:
            self.backend = 'pygame'
            logger.info(f"Using pygame backend for controller {controller_id}")
        else:
            logger.error("No suitable backend found for Xbox controller")
            raise RuntimeError("Xbox controller support not available")
    
    def connect(self) -> bool:
        """
        Connect to Xbox controller.
        
        Returns:
            True if connected successfully
        """
        try:
            if self.backend == 'xinput':
                return self._connect_xinput()
            elif self.backend == 'pygame':
                return self._connect_pygame()
            return False
        except Exception as e:
            logger.error(f"Failed to connect Xbox controller {self.controller_id}: {e}")
            return False
    
    def _connect_xinput(self) -> bool:
        """Connect using XInput."""
        if not HAS_XINPUT:
            return False
        
        try:
            # Test if controller is connected
            state = XInput.State()
            result = XInput.XInputGetState(self.controller_id, state)
            
            if result == 0:  # ERROR_SUCCESS
                self.xinput_state = state
                self.state.is_connected = True
                
                # Get battery info if available
                try:
                    battery = XInput.BatteryInformation()
                    XInput.XInputGetBatteryInformation(
                        self.controller_id,
                        XInput.BATTERY_DEVTYPE_GAMEPAD,
                        battery
                    )
                    self.state.battery_level = battery.BatteryLevel / 3.0  # Convert to 0-1
                except:
                    pass
                
                logger.info(f"Xbox controller {self.controller_id} connected via XInput")
                return True
            
        except Exception as e:
            logger.error(f"XInput connection error: {e}")
        
        return False
    
    def _connect_pygame(self) -> bool:
        """Connect using pygame."""
        if not HAS_PYGAME:
            return False
        
        try:
            # Initialize pygame joystick module
            pygame.init()
            pygame.joystick.init()
            
            # Check if controller exists
            if self.controller_id >= pygame.joystick.get_count():
                logger.warning(f"Controller {self.controller_id} not found")
                return False
            
            # Open joystick
            self.pygame_joystick = pygame.joystick.Joystick(self.controller_id)
            self.pygame_joystick.init()
            
            # Detect Xbox controller
            name = self.pygame_joystick.get_name().lower()
            if 'xbox' in name:
                if '360' in name:
                    self.controller_type = ControllerType.XBOX_360
                elif 'one' in name:
                    self.controller_type = ControllerType.XBOX_ONE
                elif 'series' in name:
                    self.controller_type = ControllerType.XBOX_SERIES
            
            self.state.is_connected = True
            logger.info(f"Xbox controller {self.controller_id} connected via pygame")
            return True
            
        except Exception as e:
            logger.error(f"Pygame connection error: {e}")
        
        return False
    
    def disconnect(self):
        """Disconnect from controller."""
        self.stop_polling()
        
        if self.backend == 'pygame' and self.pygame_joystick:
            self.pygame_joystick.quit()
            self.pygame_joystick = None
        
        self.state.is_connected = False
        logger.info(f"Xbox controller {self.controller_id} disconnected")
    
    def poll(self) -> Optional[ControllerState]:
        """
        Poll current controller state.
        
        Returns:
            Current controller state or None if disconnected
        """
        if self.backend == 'xinput':
            return self._poll_xinput()
        elif self.backend == 'pygame':
            return self._poll_pygame()
        return None
    
    def _poll_xinput(self) -> Optional[ControllerState]:
        """Poll using XInput."""
        if not HAS_XINPUT:
            return None
        
        try:
            state = XInput.State()
            result = XInput.XInputGetState(self.controller_id, state)
            
            if result != 0:  # Controller disconnected
                self.state.is_connected = False
                return None
            
            # Check if state changed
            if state.dwPacketNumber == self.packet_number:
                return self.state  # No change
            
            self.packet_number = state.dwPacketNumber
            gamepad = state.Gamepad
            
            # Create new state
            new_state = ControllerState(
                controller_id=self.controller_id,
                controller_type=self.controller_type,
                is_connected=True
            )
            
            # Parse buttons
            for mask, button in self.XINPUT_BUTTON_MAP.items():
                new_state.buttons[button] = bool(gamepad.wButtons & mask)
            
            # Parse axes (normalize to -1.0 to 1.0)
            new_state.axes[Axis.LEFT_X] = gamepad.sThumbLX / 32767.0
            new_state.axes[Axis.LEFT_Y] = gamepad.sThumbLY / 32767.0
            new_state.axes[Axis.RIGHT_X] = gamepad.sThumbRX / 32767.0
            new_state.axes[Axis.RIGHT_Y] = gamepad.sThumbRY / 32767.0
            
            # Triggers (normalize to 0.0 to 1.0)
            new_state.axes[Axis.LEFT_TRIGGER] = gamepad.bLeftTrigger / 255.0
            new_state.axes[Axis.RIGHT_TRIGGER] = gamepad.bRightTrigger / 255.0
            
            # Apply deadzones
            new_state = self._apply_deadzones(new_state)
            
            return new_state
            
        except Exception as e:
            logger.error(f"XInput polling error: {e}")
            return None
    
    def _poll_pygame(self) -> Optional[ControllerState]:
        """Poll using pygame."""
        if not self.pygame_joystick:
            return None
        
        try:
            # Process pygame events
            pygame.event.pump()
            
            # Create new state
            new_state = ControllerState(
                controller_id=self.controller_id,
                controller_type=self.controller_type,
                is_connected=True
            )
            
            # Map buttons (pygame button indices to our Button enum)
            button_map = {
                0: Button.A,
                1: Button.B,
                2: Button.X,
                3: Button.Y,
                4: Button.LB,
                5: Button.RB,
                6: Button.BACK,
                7: Button.START,
                8: Button.GUIDE,
                9: Button.LEFT_STICK,
                10: Button.RIGHT_STICK
            }
            
            for pygame_btn, button in button_map.items():
                if pygame_btn < self.pygame_joystick.get_numbuttons():
                    new_state.buttons[button] = bool(
                        self.pygame_joystick.get_button(pygame_btn)
                    )
            
            # Map axes
            if self.pygame_joystick.get_numaxes() >= 6:
                new_state.axes[Axis.LEFT_X] = self.pygame_joystick.get_axis(0)
                new_state.axes[Axis.LEFT_Y] = -self.pygame_joystick.get_axis(1)  # Invert Y
                new_state.axes[Axis.RIGHT_X] = self.pygame_joystick.get_axis(3)
                new_state.axes[Axis.RIGHT_Y] = -self.pygame_joystick.get_axis(4)  # Invert Y
                
                # Triggers (convert from -1,1 to 0,1)
                left_trigger = self.pygame_joystick.get_axis(2)
                right_trigger = self.pygame_joystick.get_axis(5)
                new_state.axes[Axis.LEFT_TRIGGER] = (left_trigger + 1.0) / 2.0
                new_state.axes[Axis.RIGHT_TRIGGER] = (right_trigger + 1.0) / 2.0
            
            # Map D-pad from hat
            if self.pygame_joystick.get_numhats() > 0:
                hat_x, hat_y = self.pygame_joystick.get_hat(0)
                new_state.buttons[Button.DPAD_UP] = hat_y > 0
                new_state.buttons[Button.DPAD_DOWN] = hat_y < 0
                new_state.buttons[Button.DPAD_LEFT] = hat_x < 0
                new_state.buttons[Button.DPAD_RIGHT] = hat_x > 0
            
            # Apply deadzones
            new_state = self._apply_deadzones(new_state)
            
            return new_state
            
        except Exception as e:
            logger.error(f"Pygame polling error: {e}")
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
                    # Rescale to maintain full range
                    sign = 1 if value > 0 else -1
                    state.axes[axis] = sign * (abs(value) - self.stick_deadzone) / (1.0 - self.stick_deadzone)
        
        # Apply trigger deadzones
        for axis in [Axis.LEFT_TRIGGER, Axis.RIGHT_TRIGGER]:
            if axis in state.axes:
                value = state.axes[axis]
                if value < self.trigger_deadzone:
                    state.axes[axis] = 0.0
                else:
                    # Rescale to maintain full range
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
        if not self.vibration_enabled:
            return
        
        # Clamp values
        left_motor = max(0.0, min(1.0, left_motor))
        right_motor = max(0.0, min(1.0, right_motor))
        
        if self.backend == 'xinput':
            self._vibrate_xinput(left_motor, right_motor, duration_ms)
        elif self.backend == 'pygame':
            self._vibrate_pygame(left_motor, right_motor, duration_ms)
    
    def _vibrate_xinput(self, left: float, right: float, duration_ms: int):
        """Vibrate using XInput."""
        if not HAS_XINPUT:
            return
        
        try:
            vibration = XInput.Vibration()
            vibration.wLeftMotorSpeed = int(left * 65535)
            vibration.wRightMotorSpeed = int(right * 65535)
            
            # Set vibration
            XInput.XInputSetState(self.controller_id, vibration)
            
            # Schedule stop
            if duration_ms > 0:
                import threading
                def stop_vibration():
                    time.sleep(duration_ms / 1000.0)
                    vibration.wLeftMotorSpeed = 0
                    vibration.wRightMotorSpeed = 0
                    XInput.XInputSetState(self.controller_id, vibration)
                
                threading.Thread(target=stop_vibration, daemon=True).start()
                
        except Exception as e:
            logger.error(f"XInput vibration error: {e}")
    
    def _vibrate_pygame(self, left: float, right: float, duration_ms: int):
        """Vibrate using pygame."""
        if not self.pygame_joystick:
            return
        
        try:
            # Pygame rumble support (if available)
            if hasattr(self.pygame_joystick, 'rumble'):
                self.pygame_joystick.rumble(left, right, duration_ms)
        except Exception as e:
            logger.debug(f"Pygame vibration not supported: {e}")
    
    def get_battery_level(self) -> Optional[float]:
        """
        Get battery level.
        
        Returns:
            Battery level (0.0-1.0) or None if not available
        """
        if self.backend == 'xinput' and HAS_XINPUT:
            try:
                battery = XInput.BatteryInformation()
                result = XInput.XInputGetBatteryInformation(
                    self.controller_id,
                    XInput.BATTERY_DEVTYPE_GAMEPAD,
                    battery
                )
                if result == 0:
                    return battery.BatteryLevel / 3.0  # Convert to 0-1
            except:
                pass
        
        return self.state.battery_level
    
    def set_led(self, pattern: int):
        """
        Set controller LED pattern (Xbox 360 only).
        
        Args:
            pattern: LED pattern (0-14)
        """
        # This would require additional native code
        # Not implemented in basic XInput
        pass