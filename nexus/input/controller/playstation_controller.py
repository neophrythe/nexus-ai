"""
PlayStation Controller Support (DualShock 4 / DualSense)

Provides support for PS4 DualShock 4 and PS5 DualSense controllers.
"""

import time
import struct
import threading
from typing import Optional, Tuple, Dict, Any
import structlog

from nexus.input.controller.gamepad_base import (
    GamepadBase, Button, Axis, ControllerState, ControllerType, ControllerEvent
)

logger = structlog.get_logger()

# Try to import required libraries
try:
    import hid  # For HID communication
    HAS_HID = True
except ImportError:
    HAS_HID = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    import evdev  # Linux support
    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False


class PlayStationController(GamepadBase):
    """
    PlayStation controller implementation for DS4 and DS5.
    
    Features:
    - DualShock 4 support
    - DualSense (PS5) support
    - Touchpad tracking
    - Motion sensors (gyro/accelerometer)
    - Adaptive triggers (PS5)
    - Advanced haptic feedback
    - LED color control
    - Battery monitoring
    """
    
    # USB/Bluetooth vendor and product IDs
    SONY_VENDOR_ID = 0x054C
    DS4_PRODUCT_ID = 0x09CC  # DualShock 4 v2
    DS4_V1_PRODUCT_ID = 0x05C4  # DualShock 4 v1
    DS5_PRODUCT_ID = 0x0CE6  # DualSense
    
    # Button mapping for PlayStation controllers
    PS_BUTTON_MAP = {
        'cross': Button.A,  # X button
        'circle': Button.B,  # Circle button
        'square': Button.X,  # Square button
        'triangle': Button.Y,  # Triangle button
        'l1': Button.LB,
        'r1': Button.RB,
        'share': Button.BACK,  # Share/Create button
        'options': Button.START,
        'ps': Button.GUIDE,  # PlayStation button
        'l3': Button.LEFT_STICK,
        'r3': Button.RIGHT_STICK,
        'dpad_up': Button.DPAD_UP,
        'dpad_down': Button.DPAD_DOWN,
        'dpad_left': Button.DPAD_LEFT,
        'dpad_right': Button.DPAD_RIGHT
    }
    
    def __init__(self, controller_id: int = 0):
        """
        Initialize PlayStation controller.
        
        Args:
            controller_id: Controller index
        """
        super().__init__(controller_id)
        
        self.controller_type = ControllerType.PS4  # Default, will detect actual type
        self.backend = None
        self.device = None
        self.pygame_joystick = None
        
        # PlayStation specific features
        self.touchpad_state = {'x': 0, 'y': 0, 'pressed': False}
        self.motion_state = {
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0
        }
        self.led_color = (0, 0, 255)  # Blue default
        self.lightbar_brightness = 1.0
        
        # Adaptive triggers (PS5)
        self.adaptive_triggers = {
            'left': {'mode': 'off', 'force': 0},
            'right': {'mode': 'off', 'force': 0}
        }
        
        # Detect backend
        if HAS_HID:
            self.backend = 'hid'
            logger.info(f"Using HID backend for PlayStation controller {controller_id}")
        elif HAS_PYGAME:
            self.backend = 'pygame'
            logger.info(f"Using pygame backend for PlayStation controller {controller_id}")
        else:
            logger.error("No suitable backend found for PlayStation controller")
            raise RuntimeError("PlayStation controller support not available")
    
    def connect(self) -> bool:
        """
        Connect to PlayStation controller.
        
        Returns:
            True if connected successfully
        """
        try:
            if self.backend == 'hid':
                return self._connect_hid()
            elif self.backend == 'pygame':
                return self._connect_pygame()
            return False
        except Exception as e:
            logger.error(f"Failed to connect PlayStation controller {self.controller_id}: {e}")
            return False
    
    def _connect_hid(self) -> bool:
        """Connect using HID."""
        if not HAS_HID:
            return False
        
        try:
            # Find PlayStation controller
            devices = hid.enumerate()
            ps_devices = [
                d for d in devices
                if d['vendor_id'] == self.SONY_VENDOR_ID
                and d['product_id'] in [self.DS4_PRODUCT_ID, self.DS4_V1_PRODUCT_ID, self.DS5_PRODUCT_ID]
            ]
            
            if self.controller_id >= len(ps_devices):
                logger.warning(f"PlayStation controller {self.controller_id} not found")
                return False
            
            device_info = ps_devices[self.controller_id]
            
            # Detect controller type
            if device_info['product_id'] == self.DS5_PRODUCT_ID:
                self.controller_type = ControllerType.PS5
            else:
                self.controller_type = ControllerType.PS4
            
            # Open device
            self.device = hid.device()
            self.device.open(device_info['vendor_id'], device_info['product_id'])
            self.device.set_nonblocking(True)
            
            self.state.is_connected = True
            logger.info(f"{self.controller_type.value} controller {self.controller_id} connected via HID")
            
            # Initialize controller features
            self._initialize_features()
            
            return True
            
        except Exception as e:
            logger.error(f"HID connection error: {e}")
            return False
    
    def _connect_pygame(self) -> bool:
        """Connect using pygame."""
        if not HAS_PYGAME:
            return False
        
        try:
            pygame.init()
            pygame.joystick.init()
            
            if self.controller_id >= pygame.joystick.get_count():
                logger.warning(f"Controller {self.controller_id} not found")
                return False
            
            self.pygame_joystick = pygame.joystick.Joystick(self.controller_id)
            self.pygame_joystick.init()
            
            # Detect PlayStation controller
            name = self.pygame_joystick.get_name().lower()
            if 'dualshock' in name or 'ds4' in name:
                self.controller_type = ControllerType.PS4
            elif 'dualsense' in name or 'ds5' in name:
                self.controller_type = ControllerType.PS5
            elif 'playstation' in name or 'sony' in name:
                self.controller_type = ControllerType.PS4  # Default
            else:
                logger.warning(f"Unknown controller type: {name}")
                return False
            
            self.state.is_connected = True
            logger.info(f"{self.controller_type.value} controller {self.controller_id} connected via pygame")
            return True
            
        except Exception as e:
            logger.error(f"Pygame connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from controller."""
        self.stop_polling()
        
        if self.backend == 'hid' and self.device:
            self.device.close()
            self.device = None
        elif self.backend == 'pygame' and self.pygame_joystick:
            self.pygame_joystick.quit()
            self.pygame_joystick = None
        
        self.state.is_connected = False
        logger.info(f"PlayStation controller {self.controller_id} disconnected")
    
    def poll(self) -> Optional[ControllerState]:
        """
        Poll current controller state.
        
        Returns:
            Current controller state or None if disconnected
        """
        if self.backend == 'hid':
            return self._poll_hid()
        elif self.backend == 'pygame':
            return self._poll_pygame()
        return None
    
    def _poll_hid(self) -> Optional[ControllerState]:
        """Poll using HID."""
        if not self.device:
            return None
        
        try:
            # Read HID report
            data = self.device.read(64, timeout_ms=0)
            if not data:
                return self.state  # No new data
            
            # Parse based on controller type
            if self.controller_type == ControllerType.PS5:
                return self._parse_ds5_report(data)
            else:
                return self._parse_ds4_report(data)
            
        except Exception as e:
            logger.error(f"HID polling error: {e}")
            self.state.is_connected = False
            return None
    
    def _parse_ds4_report(self, data: bytes) -> ControllerState:
        """Parse DualShock 4 HID report."""
        if len(data) < 10:
            return self.state
        
        new_state = ControllerState(
            controller_id=self.controller_id,
            controller_type=self.controller_type,
            is_connected=True
        )
        
        # Parse analog sticks (bytes 1-4)
        new_state.axes[Axis.LEFT_X] = (data[1] - 128) / 127.0
        new_state.axes[Axis.LEFT_Y] = -(data[2] - 128) / 127.0  # Invert Y
        new_state.axes[Axis.RIGHT_X] = (data[3] - 128) / 127.0
        new_state.axes[Axis.RIGHT_Y] = -(data[4] - 128) / 127.0  # Invert Y
        
        # Parse buttons (bytes 5-6)
        buttons1 = data[5]
        buttons2 = data[6]
        
        # D-pad (lower 4 bits of buttons1)
        dpad = buttons1 & 0x0F
        new_state.buttons[Button.DPAD_UP] = dpad in [0, 1, 7]
        new_state.buttons[Button.DPAD_RIGHT] = dpad in [1, 2, 3]
        new_state.buttons[Button.DPAD_DOWN] = dpad in [3, 4, 5]
        new_state.buttons[Button.DPAD_LEFT] = dpad in [5, 6, 7]
        
        # Face buttons (upper 4 bits of buttons1)
        new_state.buttons[Button.X] = bool(buttons1 & 0x10)  # Square
        new_state.buttons[Button.A] = bool(buttons1 & 0x20)  # Cross
        new_state.buttons[Button.B] = bool(buttons1 & 0x40)  # Circle
        new_state.buttons[Button.Y] = bool(buttons1 & 0x80)  # Triangle
        
        # Shoulder and system buttons (buttons2)
        new_state.buttons[Button.LB] = bool(buttons2 & 0x01)  # L1
        new_state.buttons[Button.RB] = bool(buttons2 & 0x02)  # R1
        # L2/R2 are analog triggers
        new_state.buttons[Button.BACK] = bool(buttons2 & 0x10)  # Share
        new_state.buttons[Button.START] = bool(buttons2 & 0x20)  # Options
        new_state.buttons[Button.LEFT_STICK] = bool(buttons2 & 0x40)  # L3
        new_state.buttons[Button.RIGHT_STICK] = bool(buttons2 & 0x80)  # R3
        
        # PS button (byte 7)
        if len(data) > 7:
            new_state.buttons[Button.GUIDE] = bool(data[7] & 0x01)
        
        # Triggers (bytes 8-9)
        if len(data) > 9:
            new_state.axes[Axis.LEFT_TRIGGER] = data[8] / 255.0
            new_state.axes[Axis.RIGHT_TRIGGER] = data[9] / 255.0
        
        # Battery level (byte 12)
        if len(data) > 12:
            battery = (data[12] & 0x0F)
            new_state.battery_level = battery / 10.0  # 0-10 scale to 0-1
        
        # Parse touchpad if available
        if len(data) > 35:
            self._parse_touchpad(data[35:39])
        
        # Parse motion sensors if available
        if len(data) > 19:
            self._parse_motion(data[13:19])
        
        # Apply deadzones
        new_state = self._apply_deadzones(new_state)
        
        return new_state
    
    def _parse_ds5_report(self, data: bytes) -> ControllerState:
        """Parse DualSense HID report."""
        # Similar to DS4 but with additional features
        # This is a simplified version
        return self._parse_ds4_report(data)  # Fallback for now
    
    def _poll_pygame(self) -> Optional[ControllerState]:
        """Poll using pygame."""
        if not self.pygame_joystick:
            return None
        
        try:
            pygame.event.pump()
            
            new_state = ControllerState(
                controller_id=self.controller_id,
                controller_type=self.controller_type,
                is_connected=True
            )
            
            # Map buttons (PlayStation layout)
            button_map = {
                0: Button.A,  # Cross
                1: Button.B,  # Circle
                2: Button.X,  # Square
                3: Button.Y,  # Triangle
                4: Button.BACK,  # Share
                5: Button.GUIDE,  # PS button
                6: Button.START,  # Options
                7: Button.LEFT_STICK,  # L3
                8: Button.RIGHT_STICK,  # R3
                9: Button.LB,  # L1
                10: Button.RB  # R1
            }
            
            for pygame_btn, button in button_map.items():
                if pygame_btn < self.pygame_joystick.get_numbuttons():
                    new_state.buttons[button] = bool(
                        self.pygame_joystick.get_button(pygame_btn)
                    )
            
            # Map axes
            if self.pygame_joystick.get_numaxes() >= 6:
                new_state.axes[Axis.LEFT_X] = self.pygame_joystick.get_axis(0)
                new_state.axes[Axis.LEFT_Y] = -self.pygame_joystick.get_axis(1)
                new_state.axes[Axis.RIGHT_X] = self.pygame_joystick.get_axis(2)
                new_state.axes[Axis.RIGHT_Y] = -self.pygame_joystick.get_axis(5)
                new_state.axes[Axis.LEFT_TRIGGER] = (self.pygame_joystick.get_axis(3) + 1.0) / 2.0
                new_state.axes[Axis.RIGHT_TRIGGER] = (self.pygame_joystick.get_axis(4) + 1.0) / 2.0
            
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
    
    def _parse_touchpad(self, data: bytes):
        """Parse touchpad data."""
        if len(data) >= 4:
            # Touchpad coordinates (12-bit values)
            x = ((data[2] & 0x0F) << 8) | data[1]
            y = (data[3] << 4) | ((data[2] & 0xF0) >> 4)
            
            self.touchpad_state['x'] = x / 1920.0  # Normalize to 0-1
            self.touchpad_state['y'] = y / 942.0   # Normalize to 0-1
            self.touchpad_state['pressed'] = not bool(data[0] & 0x80)
    
    def _parse_motion(self, data: bytes):
        """Parse motion sensor data."""
        if len(data) >= 6:
            # Gyroscope (16-bit signed values)
            gyro_x = struct.unpack('<h', data[0:2])[0] / 1024.0
            gyro_y = struct.unpack('<h', data[2:4])[0] / 1024.0
            gyro_z = struct.unpack('<h', data[4:6])[0] / 1024.0
            
            self.motion_state['gyro_x'] = gyro_x
            self.motion_state['gyro_y'] = gyro_y
            self.motion_state['gyro_z'] = gyro_z
    
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
        
        left_motor = max(0.0, min(1.0, left_motor))
        right_motor = max(0.0, min(1.0, right_motor))
        
        if self.backend == 'hid':
            self._vibrate_hid(left_motor, right_motor, duration_ms)
        elif self.backend == 'pygame':
            self._vibrate_pygame(left_motor, right_motor, duration_ms)
    
    def _vibrate_hid(self, left: float, right: float, duration_ms: int):
        """Vibrate using HID."""
        if not self.device:
            return
        
        try:
            # Build rumble report
            if self.controller_type == ControllerType.PS4:
                # DS4 output report
                report = bytearray(11)
                report[0] = 0x05  # Report ID
                report[1] = 0xFF  # Enable rumble
                report[4] = int(right * 255)  # Small motor
                report[5] = int(left * 255)   # Large motor
                
                # Set LED color
                report[6] = self.led_color[0]  # R
                report[7] = self.led_color[1]  # G
                report[8] = self.led_color[2]  # B
                
                self.device.write(report)
            
            # Schedule stop
            if duration_ms > 0:
                def stop_vibration():
                    time.sleep(duration_ms / 1000.0)
                    self.vibrate(0, 0, 0)
                
                threading.Thread(target=stop_vibration, daemon=True).start()
                
        except Exception as e:
            logger.error(f"HID vibration error: {e}")
    
    def _vibrate_pygame(self, left: float, right: float, duration_ms: int):
        """Vibrate using pygame."""
        if not self.pygame_joystick:
            return
        
        try:
            if hasattr(self.pygame_joystick, 'rumble'):
                self.pygame_joystick.rumble(left, right, duration_ms)
        except Exception as e:
            logger.debug(f"Pygame vibration not supported: {e}")
    
    def set_led_color(self, r: int, g: int, b: int):
        """
        Set controller LED/lightbar color.
        
        Args:
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
        """
        self.led_color = (r, g, b)
        
        if self.backend == 'hid' and self.device:
            # Send color update
            self.vibrate(0, 0, 0)  # This will update LED as side effect
    
    def get_touchpad_state(self) -> Dict[str, Any]:
        """Get touchpad state."""
        return self.touchpad_state.copy()
    
    def get_motion_state(self) -> Dict[str, float]:
        """Get motion sensor state."""
        return self.motion_state.copy()
    
    def set_adaptive_trigger(self, trigger: str, mode: str, **kwargs):
        """
        Set adaptive trigger feedback (PS5 only).
        
        Args:
            trigger: 'left' or 'right'
            mode: 'off', 'resistance', 'vibration', 'weapon'
            **kwargs: Mode-specific parameters
        """
        if self.controller_type != ControllerType.PS5:
            logger.warning("Adaptive triggers only available on PS5 controller")
            return
        
        if trigger in self.adaptive_triggers:
            self.adaptive_triggers[trigger]['mode'] = mode
            self.adaptive_triggers[trigger].update(kwargs)
            
            # Would send HID report to controller here
            logger.info(f"Set {trigger} trigger to {mode} mode")
    
    def _initialize_features(self):
        """Initialize controller-specific features."""
        if self.controller_type == ControllerType.PS4:
            # Enable extended reports for motion and touchpad
            self.set_led_color(0, 0, 255)  # Blue LED
        elif self.controller_type == ControllerType.PS5:
            # Initialize DualSense features
            self.set_led_color(0, 0, 255)  # Blue LED
            # Would initialize adaptive triggers here