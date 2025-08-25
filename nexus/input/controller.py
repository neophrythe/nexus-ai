"""Input Controller for Nexus Framework"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import structlog

from .recorder import InputRecorder, InputEvent, EventType
from .playback import InputPlayback, PlaybackConfig

logger = structlog.get_logger()


class InputBackend(Enum):
    """Available input backends"""
    PYNPUT = "pynput"
    PYAUTOGUI = "pyautogui"
    WIN32 = "win32"
    CLIENT = "client"  # Redis-based remote control


class InputController:
    """Main input controller for recording, playback, and real-time input"""
    
    def __init__(self, 
                 backend: InputBackend = InputBackend.PYNPUT,
                 recorder_config: Optional[Dict[str, Any]] = None,
                 game=None):
        """
        Initialize input controller
        
        Args:
            backend: Input backend to use
            recorder_config: Configuration for input recorder
            game: Game instance for window-aware input
        """
        self.backend = backend
        self.game = game
        
        # Initialize components
        self.recorder = InputRecorder(**(recorder_config or {}))
        self.playback = InputPlayback()
        
        # Backend-specific controllers
        self.keyboard_controller = None
        self.mouse_controller = None
        self.win32_available = False
        self.pyautogui_available = False
        
        self._initialize_backend()
        
        # State
        self.active_keys = set()
        self.last_mouse_position = (0, 0)
        
        logger.info(f"InputController initialized with {backend.value} backend")
    
    def _initialize_backend(self):
        """Initialize the selected input backend"""
        if self.backend == InputBackend.PYNPUT:
            self._initialize_pynput()
        elif self.backend == InputBackend.PYAUTOGUI:
            self._initialize_pyautogui()
        elif self.backend == InputBackend.WIN32:
            self._initialize_win32()
        elif self.backend == InputBackend.CLIENT:
            self._initialize_client()
    
    def _initialize_pynput(self):
        """Initialize pynput backend"""
        try:
            from pynput.keyboard import Controller as KeyboardController
            from pynput.mouse import Controller as MouseController
            
            self.keyboard_controller = KeyboardController()
            self.mouse_controller = MouseController()
            logger.info("Pynput backend initialized")
            
        except ImportError:
            logger.error("Pynput not available")
            raise
    
    def _initialize_pyautogui(self):
        """Initialize PyAutoGUI backend"""
        try:
            import pyautogui
            self.pyautogui = pyautogui
            self.pyautogui_available = True
            
            # Configure PyAutoGUI
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.01
            
            logger.info("PyAutoGUI backend initialized")
            
        except ImportError:
            logger.error("PyAutoGUI not available")
            raise
    
    def _initialize_win32(self):
        """Initialize Win32 backend"""
        try:
            import win32api
            import win32con
            import win32gui
            
            self.win32api = win32api
            self.win32con = win32con
            self.win32gui = win32gui
            self.win32_available = True
            
            logger.info("Win32 backend initialized")
            
        except ImportError:
            logger.error("Win32 API not available")
            raise
    
    def _initialize_client(self):
        """Initialize client (Redis) backend"""
        try:
            import redis
            
            self.redis_client = redis.Redis(
                host='127.0.0.1',
                port=6379,
                db=0,
                decode_responses=True
            )
            
            logger.info("Client (Redis) backend initialized")
            
        except ImportError:
            logger.error("Redis not available")
            raise
    
    # Recording Methods
    def start_recording(self, enable_frame_sync: bool = False) -> bool:
        """Start input recording"""
        return self.recorder.start_recording(enable_frame_sync)
    
    def stop_recording(self) -> bool:
        """Stop input recording"""
        return self.recorder.stop_recording()
    
    def pause_recording(self) -> bool:
        """Pause input recording"""
        return self.recorder.pause_recording()
    
    def resume_recording(self) -> bool:
        """Resume input recording"""
        return self.recorder.resume_recording()
    
    def save_recording(self, filepath: str, format: str = "json") -> bool:
        """Save current recording"""
        return self.recorder.save_recording(filepath, format)
    
    def load_recording(self, filepath: str) -> bool:
        """Load recording for playback"""
        if not self.recorder.load_recording(filepath):
            return False
        
        # Load events into playback system
        return self.playback.load_events(self.recorder.get_events())
    
    # Playback Methods
    def start_playback(self, config: Optional[PlaybackConfig] = None) -> bool:
        """Start input playback"""
        if config:
            self.playback.config = config
        return self.playback.start_playback()
    
    def stop_playback(self) -> bool:
        """Stop input playback"""
        return self.playback.stop_playback()
    
    def pause_playback(self) -> bool:
        """Pause input playback"""
        return self.playback.pause_playback()
    
    def resume_playback(self) -> bool:
        """Resume input playback"""
        return self.playback.resume_playback()
    
    def set_playback_speed(self, speed: float) -> bool:
        """Set playback speed"""
        return self.playback.set_speed(speed)
    
    def seek_playback(self, position: float) -> bool:
        """Seek to position in playback (0.0-1.0)"""
        return self.playback.seek_to_position(position)
    
    # Real-time Input Methods
    async def press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press a key"""
        try:
            if self.backend == InputBackend.PYNPUT:
                return self._pynput_press_key(key, duration)
            elif self.backend == InputBackend.PYAUTOGUI:
                return self._pyautogui_press_key(key, duration)
            elif self.backend == InputBackend.WIN32:
                return self._win32_press_key(key, duration)
            elif self.backend == InputBackend.CLIENT:
                return self._client_press_key(key, duration)
            return False
            
        except Exception as e:
            logger.error(f"Failed to press key {key}: {e}")
            return False
    
    async def release_key(self, key: str) -> bool:
        """Release a key"""
        try:
            if self.backend == InputBackend.PYNPUT:
                return self._pynput_release_key(key)
            elif self.backend == InputBackend.WIN32:
                return self._win32_release_key(key)
            return True
            
        except Exception as e:
            logger.error(f"Failed to release key {key}: {e}")
            return False
    
    async def type_text(self, text: str, interval: float = 0.01) -> bool:
        """Type text"""
        try:
            if self.backend == InputBackend.PYNPUT:
                return self._pynput_type_text(text, interval)
            elif self.backend == InputBackend.PYAUTOGUI:
                return self._pyautogui_type_text(text, interval)
            elif self.backend == InputBackend.WIN32:
                return self._win32_type_text(text, interval)
            return False
            
        except Exception as e:
            logger.error(f"Failed to type text: {e}")
            return False
    
    async def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to position"""
        try:
            if self.backend == InputBackend.PYNPUT:
                return self._pynput_move_mouse(x, y, duration)
            elif self.backend == InputBackend.PYAUTOGUI:
                return self._pyautogui_move_mouse(x, y, duration)
            elif self.backend == InputBackend.WIN32:
                return self._win32_move_mouse(x, y)
            return False
            
        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            return False
    
    async def click_mouse(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click mouse at position"""
        try:
            if self.backend == InputBackend.PYNPUT:
                return self._pynput_click_mouse(x, y, button, clicks)
            elif self.backend == InputBackend.PYAUTOGUI:
                return self._pyautogui_click_mouse(x, y, button, clicks)
            elif self.backend == InputBackend.WIN32:
                return self._win32_click_mouse(x, y, button, clicks)
            return False
            
        except Exception as e:
            logger.error(f"Failed to click mouse: {e}")
            return False
    
    async def scroll_mouse(self, x: int, y: int, scroll_x: int = 0, scroll_y: int = 0) -> bool:
        """Scroll mouse wheel"""
        try:
            if self.backend == InputBackend.PYNPUT:
                return self._pynput_scroll_mouse(x, y, scroll_x, scroll_y)
            elif self.backend == InputBackend.PYAUTOGUI:
                return self._pyautogui_scroll_mouse(x, y, scroll_y)
            elif self.backend == InputBackend.WIN32:
                return self._win32_scroll_mouse(x, y, scroll_y)
            return False
            
        except Exception as e:
            logger.error(f"Failed to scroll mouse: {e}")
            return False
    
    # Pynput Implementation
    def _pynput_press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press key using pynput"""
        if not self.keyboard_controller:
            return False
        
        pynput_key = self._get_pynput_key(key)
        if not pynput_key:
            return False
        
        self.keyboard_controller.press(pynput_key)
        self.active_keys.add(key)
        
        if duration:
            time.sleep(duration)
            self.keyboard_controller.release(pynput_key)
            self.active_keys.discard(key)
        
        return True
    
    def _pynput_release_key(self, key: str) -> bool:
        """Release key using pynput"""
        if not self.keyboard_controller:
            return False
        
        pynput_key = self._get_pynput_key(key)
        if not pynput_key:
            return False
        
        self.keyboard_controller.release(pynput_key)
        self.active_keys.discard(key)
        return True
    
    def _pynput_type_text(self, text: str, interval: float) -> bool:
        """Type text using pynput"""
        if not self.keyboard_controller:
            return False
        
        for char in text:
            self.keyboard_controller.type(char)
            if interval > 0:
                time.sleep(interval)
        
        return True
    
    def _pynput_move_mouse(self, x: int, y: int, duration: float) -> bool:
        """Move mouse using pynput"""
        if not self.mouse_controller:
            return False
        
        if duration > 0:
            # Smooth movement
            start_pos = self.mouse_controller.position
            steps = max(10, int(duration * 60))  # 60 FPS
            
            for i in range(steps + 1):
                progress = i / steps
                current_x = int(start_pos[0] + (x - start_pos[0]) * progress)
                current_y = int(start_pos[1] + (y - start_pos[1]) * progress)
                
                self.mouse_controller.position = (current_x, current_y)
                time.sleep(duration / steps)
        else:
            self.mouse_controller.position = (x, y)
        
        self.last_mouse_position = (x, y)
        return True
    
    def _pynput_click_mouse(self, x: int, y: int, button: str, clicks: int) -> bool:
        """Click mouse using pynput"""
        if not self.mouse_controller:
            return False
        
        from pynput.mouse import Button
        
        button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        
        mouse_button = button_map.get(button.lower())
        if not mouse_button:
            return False
        
        self.mouse_controller.position = (x, y)
        self.mouse_controller.click(mouse_button, clicks)
        return True
    
    def _pynput_scroll_mouse(self, x: int, y: int, scroll_x: int, scroll_y: int) -> bool:
        """Scroll mouse using pynput"""
        if not self.mouse_controller:
            return False
        
        self.mouse_controller.position = (x, y)
        self.mouse_controller.scroll(scroll_x, scroll_y)
        return True
    
    # PyAutoGUI Implementation
    def _pyautogui_press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press key using PyAutoGUI"""
        if not self.pyautogui_available:
            return False
        
        if duration:
            self.pyautogui.keyDown(key)
            time.sleep(duration)
            self.pyautogui.keyUp(key)
        else:
            self.pyautogui.press(key)
        
        return True
    
    def _pyautogui_type_text(self, text: str, interval: float) -> bool:
        """Type text using PyAutoGUI"""
        if not self.pyautogui_available:
            return False
        
        self.pyautogui.write(text, interval=interval)
        return True
    
    def _pyautogui_move_mouse(self, x: int, y: int, duration: float) -> bool:
        """Move mouse using PyAutoGUI"""
        if not self.pyautogui_available:
            return False
        
        self.pyautogui.moveTo(x, y, duration=duration)
        return True
    
    def _pyautogui_click_mouse(self, x: int, y: int, button: str, clicks: int) -> bool:
        """Click mouse using PyAutoGUI"""
        if not self.pyautogui_available:
            return False
        
        self.pyautogui.click(x, y, clicks=clicks, button=button)
        return True
    
    def _pyautogui_scroll_mouse(self, x: int, y: int, scroll_y: int) -> bool:
        """Scroll mouse using PyAutoGUI"""
        if not self.pyautogui_available:
            return False
        
        self.pyautogui.moveTo(x, y)
        self.pyautogui.scroll(scroll_y)
        return True
    
    # Win32 Implementation (Windows only)
    def _win32_press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press key using Win32 API"""
        if not self.win32_available:
            return False
        
        vk_code = self._get_win32_vk_code(key)
        if not vk_code:
            return False
        
        self.win32api.keybd_event(vk_code, 0, 0, 0)  # Key down
        
        if duration:
            time.sleep(duration)
            self.win32api.keybd_event(vk_code, 0, self.win32con.KEYEVENTF_KEYUP, 0)  # Key up
        
        return True
    
    def _win32_release_key(self, key: str) -> bool:
        """Release key using Win32 API"""
        if not self.win32_available:
            return False
        
        vk_code = self._get_win32_vk_code(key)
        if not vk_code:
            return False
        
        self.win32api.keybd_event(vk_code, 0, self.win32con.KEYEVENTF_KEYUP, 0)
        return True
    
    def _win32_type_text(self, text: str, interval: float) -> bool:
        """Type text using Win32 API"""
        for char in text:
            vk_code = ord(char.upper())
            self.win32api.keybd_event(vk_code, 0, 0, 0)
            self.win32api.keybd_event(vk_code, 0, self.win32con.KEYEVENTF_KEYUP, 0)
            
            if interval > 0:
                time.sleep(interval)
        
        return True
    
    def _win32_move_mouse(self, x: int, y: int) -> bool:
        """Move mouse using Win32 API"""
        if not self.win32_available:
            return False
        
        self.win32api.SetCursorPos((x, y))
        return True
    
    def _win32_click_mouse(self, x: int, y: int, button: str, clicks: int) -> bool:
        """Click mouse using Win32 API"""
        if not self.win32_available:
            return False
        
        self.win32api.SetCursorPos((x, y))
        
        button_down_map = {
            'left': self.win32con.MOUSEEVENTF_LEFTDOWN,
            'right': self.win32con.MOUSEEVENTF_RIGHTDOWN,
            'middle': self.win32con.MOUSEEVENTF_MIDDLEDOWN
        }
        
        button_up_map = {
            'left': self.win32con.MOUSEEVENTF_LEFTUP,
            'right': self.win32con.MOUSEEVENTF_RIGHTUP,
            'middle': self.win32con.MOUSEEVENTF_MIDDLEUP
        }
        
        down_event = button_down_map.get(button.lower())
        up_event = button_up_map.get(button.lower())
        
        if not down_event or not up_event:
            return False
        
        for _ in range(clicks):
            self.win32api.mouse_event(down_event, x, y, 0, 0)
            self.win32api.mouse_event(up_event, x, y, 0, 0)
            time.sleep(0.01)
        
        return True
    
    def _win32_scroll_mouse(self, x: int, y: int, scroll_y: int) -> bool:
        """Scroll mouse using Win32 API"""
        if not self.win32_available:
            return False
        
        self.win32api.SetCursorPos((x, y))
        self.win32api.mouse_event(self.win32con.MOUSEEVENTF_WHEEL, x, y, scroll_y * 120, 0)
        return True
    
    # Client (Redis) Implementation
    def _client_press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press key using client backend"""
        command = {
            'type': 'key_press',
            'key': key,
            'duration': duration
        }
        
        self.redis_client.lpush('nexus:input_commands', str(command))
        return True
    
    # Utility Methods
    def _get_pynput_key(self, key: str):
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
                'shift': Key.shift,
                'ctrl': Key.ctrl,
                'alt': Key.alt,
                'cmd': Key.cmd,
                'up': Key.up,
                'down': Key.down,
                'left': Key.left,
                'right': Key.right,
            }
            
            if key.lower() in special_keys:
                return special_keys[key.lower()]
            
            if len(key) == 1:
                return KeyCode.from_char(key)
            
            return None
            
        except Exception:
            return None
    
    def _get_win32_vk_code(self, key: str) -> Optional[int]:
        """Get Win32 virtual key code"""
        vk_map = {
            'space': 0x20,
            'enter': 0x0D,
            'tab': 0x09,
            'backspace': 0x08,
            'delete': 0x2E,
            'esc': 0x1B,
            'shift': 0x10,
            'ctrl': 0x11,
            'alt': 0x12,
            'up': 0x26,
            'down': 0x28,
            'left': 0x25,
            'right': 0x27,
        }
        
        if key.lower() in vk_map:
            return vk_map[key.lower()]
        
        if len(key) == 1:
            return ord(key.upper())
        
        return None
    
    # Status and Statistics
    def get_recorder_stats(self) -> Dict[str, Any]:
        """Get recorder statistics"""
        return self.recorder.get_stats()
    
    def get_playback_stats(self) -> Dict[str, Any]:
        """Get playback statistics"""
        return self.playback.get_stats()
    
    def is_recording(self) -> bool:
        """Check if recording is active"""
        return self.recorder.state.value in ['recording', 'paused']
    
    def is_playing(self) -> bool:
        """Check if playback is active"""
        return self.playback.state.value in ['playing', 'paused']
    
    def get_active_keys(self) -> List[str]:
        """Get currently active keys"""
        return list(self.active_keys)
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return self.last_mouse_position
    
    def cleanup(self):
        """Cleanup resources"""
        self.recorder.cleanup()
        self.playback.cleanup()
        
        if hasattr(self, 'redis_client'):
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"Failed to close redis client: {e}")