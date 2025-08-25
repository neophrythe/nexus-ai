"""Windows Window Controller Implementation for Nexus Framework"""

import ctypes
import ctypes.wintypes
from typing import Dict, List, Optional, Any
import win32gui
import win32con
import win32api
import win32process
import psutil
import numpy as np
from PIL import ImageGrab
import structlog

from nexus.window.window_controller import BaseWindowAdapter, WindowInfo

logger = structlog.get_logger()


class Win32WindowController(BaseWindowAdapter):
    """Windows-specific window controller"""
    
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        
    def locate_window(self, name: str) -> Optional[WindowInfo]:
        """Locate window by name/title"""
        name_lower = name.lower()
        
        def enum_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title and name_lower in window_title.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            hwnd = windows[0]
            return self._get_window_info(hwnd)
        
        return None
    
    def list_windows(self) -> List[WindowInfo]:
        """List all windows"""
        windows = []
        
        def enum_callback(hwnd, window_list):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:
                    info = self._get_window_info(hwnd)
                    if info:
                        window_list.append(info)
            return True
        
        win32gui.EnumWindows(enum_callback, windows)
        return windows
    
    def move_window(self, window_id: Any, x: int, y: int) -> bool:
        """Move window to position"""
        try:
            rect = win32gui.GetWindowRect(window_id)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            win32gui.SetWindowPos(
                window_id,
                win32con.HWND_TOP,
                x, y, width, height,
                win32con.SWP_SHOWWINDOW
            )
            return True
        except Exception as e:
            logger.error(f"Failed to move window: {e}")
            return False
    
    def resize_window(self, window_id: Any, width: int, height: int) -> bool:
        """Resize window"""
        try:
            rect = win32gui.GetWindowRect(window_id)
            x = rect[0]
            y = rect[1]
            
            win32gui.SetWindowPos(
                window_id,
                win32con.HWND_TOP,
                x, y, width, height,
                win32con.SWP_SHOWWINDOW
            )
            return True
        except Exception as e:
            logger.error(f"Failed to resize window: {e}")
            return False
    
    def focus_window(self, window_id: Any) -> bool:
        """Focus window"""
        try:
            # Restore window if minimized
            if win32gui.IsIconic(window_id):
                win32gui.ShowWindow(window_id, win32con.SW_RESTORE)
            
            # Bring to foreground
            win32gui.SetForegroundWindow(window_id)
            
            # Focus
            win32gui.SetFocus(window_id)
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False
    
    def bring_window_to_top(self, window_id: Any) -> bool:
        """Bring window to top"""
        try:
            win32gui.BringWindowToTop(window_id)
            return True
        except Exception as e:
            logger.error(f"Failed to bring window to top: {e}")
            return False
    
    def is_window_focused(self, window_id: Any) -> bool:
        """Check if window is focused"""
        try:
            return win32gui.GetForegroundWindow() == window_id
        except Exception:
            return False
    
    def get_focused_window(self) -> Optional[WindowInfo]:
        """Get currently focused window"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            return self._get_window_info(hwnd)
        except Exception as e:
            logger.error(f"Failed to get focused window: {e}")
            return None
    
    def get_window_geometry(self, window_id: Any) -> Optional[Dict[str, int]]:
        """Get window geometry"""
        try:
            rect = win32gui.GetWindowRect(window_id)
            return {
                "x": rect[0],
                "y": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1]
            }
        except Exception as e:
            logger.error(f"Failed to get window geometry: {e}")
            return None
    
    def capture_window(self, window_id: Any) -> Optional[np.ndarray]:
        """Capture window content"""
        try:
            # Get window rectangle
            rect = win32gui.GetWindowRect(window_id)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
            
            # Capture window area
            img = ImageGrab.grab(bbox=(x, y, right, bottom))
            
            # Convert to numpy array
            return np.array(img)
        except Exception as e:
            logger.error(f"Failed to capture window: {e}")
            return None
    
    def _get_window_info(self, hwnd: int) -> Optional[WindowInfo]:
        """Get window information from handle"""
        try:
            # Get window title
            title = win32gui.GetWindowText(hwnd)
            
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            x, y, right, bottom = rect
            
            # Get process info
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            
            try:
                process = psutil.Process(process_id)
                process_name = process.name()
            except Exception:
                process_name = None
            
            # Check visibility and focus
            is_visible = win32gui.IsWindowVisible(hwnd)
            is_focused = win32gui.GetForegroundWindow() == hwnd
            
            return WindowInfo(
                window_id=hwnd,
                title=title,
                x=x,
                y=y,
                width=right - x,
                height=bottom - y,
                is_visible=is_visible,
                is_focused=is_focused,
                process_name=process_name,
                process_id=process_id
            )
        except Exception as e:
            logger.debug(f"Failed to get window info: {e}")
            return None