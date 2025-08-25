"""Native Windows Input Controller with anti-detection"""

import asyncio
import time
import random
import math
from typing import Optional, Tuple, List
import numpy as np
import structlog

from nexus.input.base import InputController, InputAction, MouseButton

logger = structlog.get_logger()

try:
    import win32api
    import win32con
    import win32gui
    import ctypes
    from ctypes import wintypes
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False
    logger.warning("Windows API not available, using fallback")


class NativeInputController(InputController):
    """Native Windows input controller with anti-detection measures"""
    
    # Virtual key codes
    VK_CODES = {
        'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
        'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
        'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
        'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
        'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
        'z': 0x5A, '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33,
        '4': 0x34, '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38,
        '9': 0x39, 'space': 0x20, 'enter': 0x0D, 'tab': 0x09,
        'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12, 'esc': 0x1B,
        'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
        'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False
        
        if WINDOWS_AVAILABLE:
            # Setup SendInput structures
            self.PUL = ctypes.POINTER(ctypes.c_ulong)
            
            class KeyBdInput(ctypes.Structure):
                _fields_ = [("wVk", ctypes.c_ushort),
                          ("wScan", ctypes.c_ushort),
                          ("dwFlags", ctypes.c_ulong),
                          ("time", ctypes.c_ulong),
                          ("dwExtraInfo", self.PUL)]
            
            class HardwareInput(ctypes.Structure):
                _fields_ = [("uMsg", ctypes.c_ulong),
                          ("wParamL", ctypes.c_short),
                          ("wParamH", ctypes.c_ushort)]
            
            class MouseInput(ctypes.Structure):
                _fields_ = [("dx", ctypes.c_long),
                          ("dy", ctypes.c_long),
                          ("mouseData", ctypes.c_ulong),
                          ("dwFlags", ctypes.c_ulong),
                          ("time", ctypes.c_ulong),
                          ("dwExtraInfo", self.PUL)]
            
            class Input_I(ctypes.Union):
                _fields_ = [("ki", KeyBdInput),
                          ("mi", MouseInput),
                          ("hi", HardwareInput)]
            
            class Input(ctypes.Structure):
                _fields_ = [("type", ctypes.c_ulong),
                          ("ii", Input_I)]
            
            self.KeyBdInput = KeyBdInput
            self.MouseInput = MouseInput
            self.Input = Input
            self.Input_I = Input_I
    
    async def initialize(self) -> None:
        """Initialize the controller"""
        if not WINDOWS_AVAILABLE:
            logger.warning("Windows API not available, functionality limited")
        self._initialized = True
        logger.info("Native input controller initialized")
    
    async def key_press(self, key: str) -> None:
        """Press a key using SendInput"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if WINDOWS_AVAILABLE:
            vk_code = self._get_vk_code(key)
            if vk_code:
                self._send_key_event(vk_code, 0)  # Key down
        
        self._add_to_history(InputAction(
            action_type="key_press",
            data={"key": key}
        ))
    
    async def key_release(self, key: str) -> None:
        """Release a key using SendInput"""
        if not self._initialized:
            await self.initialize()
        
        if WINDOWS_AVAILABLE:
            vk_code = self._get_vk_code(key)
            if vk_code:
                self._send_key_event(vk_code, 0x0002)  # Key up
        
        self._add_to_history(InputAction(
            action_type="key_release",
            data={"key": key}
        ))
    
    async def key_tap(self, key: str, duration: Optional[float] = None) -> None:
        """Press and release a key with human-like timing"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        # Add random variation to duration
        if self.human_like:
            duration = duration or random.uniform(0.05, 0.15)
        else:
            duration = duration or 0.05
        
        await self.key_press(key)
        await asyncio.sleep(duration)
        await self.key_release(key)
        
        self._add_to_history(InputAction(
            action_type="key_tap",
            data={"key": key},
            duration=duration
        ))
    
    async def type_text(self, text: str, interval: Optional[float] = None) -> None:
        """Type text with human-like patterns"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        for char in text:
            if self.human_like:
                # Vary typing speed
                interval = random.gauss(0.08, 0.02)
                interval = max(0.03, min(0.15, interval))
                
                # Occasionally pause (thinking)
                if random.random() < 0.05:
                    await asyncio.sleep(random.uniform(0.3, 0.8))
            else:
                interval = interval or 0.05
            
            await self.key_tap(char)
            await asyncio.sleep(interval)
        
        self._add_to_history(InputAction(
            action_type="type_text",
            data={"text": text, "interval": interval}
        ))
    
    async def mouse_move(self, x: int, y: int, duration: Optional[float] = None) -> None:
        """Move mouse with human-like curve"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if WINDOWS_AVAILABLE:
            current_x, current_y = self.get_mouse_position()
            
            if self.human_like:
                # Generate human-like mouse path
                points = self._generate_bezier_curve(
                    (current_x, current_y), (x, y),
                    duration or 0.5
                )
                
                for px, py, delay in points:
                    win32api.SetCursorPos((int(px), int(py)))
                    await asyncio.sleep(delay)
            else:
                win32api.SetCursorPos((x, y))
        
        self._add_to_history(InputAction(
            action_type="mouse_move",
            data={"x": x, "y": y},
            duration=duration or 0
        ))
    
    async def mouse_click(self, button: MouseButton = MouseButton.LEFT,
                         x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Click mouse with human-like timing"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if x is not None and y is not None:
            await self.mouse_move(x, y)
        
        if WINDOWS_AVAILABLE:
            # Human-like click duration
            if self.human_like:
                click_duration = random.gauss(0.08, 0.02)
                click_duration = max(0.03, min(0.15, click_duration))
            else:
                click_duration = 0.05
            
            await self.mouse_down(button)
            await asyncio.sleep(click_duration)
            await self.mouse_up(button)
        
        self._add_to_history(InputAction(
            action_type="mouse_click",
            data={"button": button.value, "x": x, "y": y}
        ))
    
    async def mouse_down(self, button: MouseButton = MouseButton.LEFT) -> None:
        """Press mouse button"""
        if not self._initialized:
            await self.initialize()
        
        if WINDOWS_AVAILABLE:
            x, y = self.get_mouse_position()
            
            if button == MouseButton.LEFT:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            elif button == MouseButton.RIGHT:
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
            elif button == MouseButton.MIDDLE:
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, x, y, 0, 0)
        
        self._add_to_history(InputAction(
            action_type="mouse_down",
            data={"button": button.value}
        ))
    
    async def mouse_up(self, button: MouseButton = MouseButton.LEFT) -> None:
        """Release mouse button"""
        if not self._initialized:
            await self.initialize()
        
        if WINDOWS_AVAILABLE:
            x, y = self.get_mouse_position()
            
            if button == MouseButton.LEFT:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
            elif button == MouseButton.RIGHT:
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
            elif button == MouseButton.MIDDLE:
                win32api.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, x, y, 0, 0)
        
        self._add_to_history(InputAction(
            action_type="mouse_up",
            data={"button": button.value}
        ))
    
    async def mouse_scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll mouse wheel"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        if x is not None and y is not None:
            await self.mouse_move(x, y)
        
        if WINDOWS_AVAILABLE:
            # Each click is 120 units
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, clicks * 120, 0)
        
        self._add_to_history(InputAction(
            action_type="mouse_scroll",
            data={"clicks": clicks, "x": x, "y": y}
        ))
    
    async def mouse_drag(self, start_x: int, start_y: int, end_x: int, end_y: int,
                        button: MouseButton = MouseButton.LEFT, duration: float = 1.0) -> None:
        """Drag mouse with human-like path"""
        if not self._initialized:
            await self.initialize()
        
        await self._add_human_delay()
        
        # Move to start
        await self.mouse_move(start_x, start_y, duration=0.2)
        
        # Press button
        await self.mouse_down(button)
        
        # Drag to end with curve
        if self.human_like:
            points = self._generate_bezier_curve(
                (start_x, start_y), (end_x, end_y), duration
            )
            
            for px, py, delay in points:
                if WINDOWS_AVAILABLE:
                    win32api.SetCursorPos((int(px), int(py)))
                await asyncio.sleep(delay)
        else:
            await self.mouse_move(end_x, end_y, duration)
        
        # Release button
        await self.mouse_up(button)
        
        self._add_to_history(InputAction(
            action_type="mouse_drag",
            data={
                "start_x": start_x, "start_y": start_y,
                "end_x": end_x, "end_y": end_y,
                "button": button.value
            },
            duration=duration
        ))
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        if WINDOWS_AVAILABLE:
            return win32api.GetCursorPos()
        return (0, 0)
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        if WINDOWS_AVAILABLE:
            return (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1))
        return (1920, 1080)
    
    def _get_vk_code(self, key: str) -> Optional[int]:
        """Get virtual key code"""
        return self.VK_CODES.get(key.lower())
    
    def _send_key_event(self, vk_code: int, flags: int) -> None:
        """Send key event using SendInput"""
        if not WINDOWS_AVAILABLE:
            return
        
        extra = ctypes.c_ulong(0)
        ii_ = self.Input_I()
        ii_.ki = self.KeyBdInput(vk_code, 0, flags, 0, ctypes.pointer(extra))
        
        x = self.Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    def _generate_bezier_curve(self, start: Tuple[int, int], end: Tuple[int, int],
                              duration: float) -> List[Tuple[float, float, float]]:
        """Generate human-like bezier curve for mouse movement"""
        points = []
        steps = max(int(duration * 60), 20)  # 60 FPS
        
        # Add control points for curve
        control1_x = start[0] + (end[0] - start[0]) * 0.3 + random.uniform(-50, 50)
        control1_y = start[1] + (end[1] - start[1]) * 0.3 + random.uniform(-50, 50)
        
        control2_x = start[0] + (end[0] - start[0]) * 0.7 + random.uniform(-50, 50)
        control2_y = start[1] + (end[1] - start[1]) * 0.7 + random.uniform(-50, 50)
        
        for i in range(steps):
            t = i / steps
            
            # Bezier curve formula
            x = ((1-t)**3 * start[0] +
                 3*(1-t)**2*t * control1_x +
                 3*(1-t)*t**2 * control2_x +
                 t**3 * end[0])
            
            y = ((1-t)**3 * start[1] +
                 3*(1-t)**2*t * control1_y +
                 3*(1-t)*t**2 * control2_y +
                 t**3 * end[1])
            
            # Add small random jitter
            if self.human_like and i > 0 and i < steps - 1:
                x += random.uniform(-1, 1)
                y += random.uniform(-1, 1)
            
            # Variable speed (slower at start/end)
            speed_factor = 1 - abs(2 * t - 1) ** 2
            delay = (duration / steps) * (0.5 + speed_factor)
            
            points.append((x, y, delay))
        
        return points