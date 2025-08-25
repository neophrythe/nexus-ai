"""Cross-platform Window Controller for Nexus Framework - Production Ready"""

import sys
import platform
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import structlog
import numpy as np

logger = structlog.get_logger()


@dataclass
class WindowInfo:
    """Window information"""
    window_id: Any
    title: str
    x: int
    y: int
    width: int
    height: int
    is_visible: bool
    is_focused: bool
    process_name: Optional[str] = None
    process_id: Optional[int] = None


class WindowControllerError(Exception):
    """Window controller exception"""
    pass


class WindowController:
    """Cross-platform window controller - Production Ready"""
    
    def __init__(self):
        self.adapter = self._load_adapter()
        logger.info(f"Initialized window controller for {platform.system()}")
    
    def locate_window(self, name: str) -> Optional[WindowInfo]:
        """Locate window by name/title"""
        return self.adapter.locate_window(name)
    
    def list_windows(self) -> List[WindowInfo]:
        """List all windows"""
        return self.adapter.list_windows()
    
    def move_window(self, window_id: Any, x: int, y: int) -> bool:
        """Move window to position"""
        return self.adapter.move_window(window_id, x, y)
    
    def resize_window(self, window_id: Any, width: int, height: int) -> bool:
        """Resize window"""
        return self.adapter.resize_window(window_id, width, height)
    
    def focus_window(self, window_id: Any) -> bool:
        """Focus window"""
        return self.adapter.focus_window(window_id)
    
    def bring_window_to_top(self, window_id: Any) -> bool:
        """Bring window to top"""
        return self.adapter.bring_window_to_top(window_id)
    
    def is_window_focused(self, window_id: Any) -> bool:
        """Check if window is focused"""
        return self.adapter.is_window_focused(window_id)
    
    def get_focused_window(self) -> Optional[WindowInfo]:
        """Get currently focused window"""
        return self.adapter.get_focused_window()
    
    def get_window_bounds(self, window_id: Any) -> Optional[Tuple[int, int, int, int]]:
        """Get window bounds (x, y, width, height)"""
        geometry = self.adapter.get_window_geometry(window_id)
        if geometry:
            return (geometry['x'], geometry['y'], geometry['width'], geometry['height'])
        return None
    
    def capture_window(self, window_id: Any) -> Optional[np.ndarray]:
        """Capture window screenshot"""
        return self.adapter.capture_window(window_id)
    
    def wait_for_window(self, name: str, timeout: float = 30.0) -> Optional[WindowInfo]:
        """Wait for window to appear"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            window = self.locate_window(name)
            if window:
                logger.info(f"Window '{name}' found after {time.time() - start_time:.1f}s")
                return window
            time.sleep(0.5)
        
        logger.warning(f"Window '{name}' not found after {timeout}s")
        return None
    
    def _load_adapter(self):
        """Load platform-specific adapter"""
        system = platform.system()
        
        if system == "Windows":
            return WindowsWindowAdapter()
        elif system == "Linux":
            return LinuxWindowAdapter()
        else:
            # Fallback adapter with basic functionality
            logger.warning(f"No specific adapter for {system}, using fallback")
            return FallbackWindowAdapter()


class BaseWindowAdapter:
    """Base window adapter interface"""
    
    def locate_window(self, name: str) -> Optional[WindowInfo]:
        """Locate window by name"""
        windows = self.list_windows()
        name_lower = name.lower()
        
        for window in windows:
            if name_lower in window.title.lower():
                return window
        return None
    
    def list_windows(self) -> List[WindowInfo]:
        """List all windows"""
        return []
    
    def move_window(self, window_id: Any, x: int, y: int) -> bool:
        """Move window"""
        return False
    
    def resize_window(self, window_id: Any, width: int, height: int) -> bool:
        """Resize window"""
        return False
    
    def focus_window(self, window_id: Any) -> bool:
        """Focus window"""
        return False
    
    def bring_window_to_top(self, window_id: Any) -> bool:
        """Bring window to top"""
        return self.focus_window(window_id)
    
    def is_window_focused(self, window_id: Any) -> bool:
        """Check if window is focused"""
        focused = self.get_focused_window()
        return focused and focused.window_id == window_id
    
    def get_focused_window(self) -> Optional[WindowInfo]:
        """Get focused window"""
        return None
    
    def get_window_geometry(self, window_id: Any) -> Optional[Dict[str, int]]:
        """Get window geometry"""
        return None
    
    def capture_window(self, window_id: Any) -> Optional[np.ndarray]:
        """Capture window screenshot"""
        try:
            geometry = self.get_window_geometry(window_id)
            if geometry:
                import mss
                with mss.mss() as sct:
                    monitor = {
                        "left": geometry['x'],
                        "top": geometry['y'],
                        "width": geometry['width'],
                        "height": geometry['height']
                    }
                    screenshot = sct.grab(monitor)
                    return np.array(screenshot)[:, :, :3]  # Remove alpha channel
        except Exception as e:
            logger.error(f"Failed to capture window: {e}")
        return None


class WindowsWindowAdapter(BaseWindowAdapter):
    """Windows-specific window adapter"""
    
    def __init__(self):
        try:
            import win32gui
            import win32con
            import win32process
            self.win32gui = win32gui
            self.win32con = win32con
            self.win32process = win32process
        except ImportError:
            logger.warning("pywin32 not available, Windows functionality limited")
            self.win32gui = None
    
    def list_windows(self) -> List[WindowInfo]:
        """List all windows"""
        if not self.win32gui:
            return []
        
        windows = []
        
        def callback(hwnd, windows_list):
            if self.win32gui.IsWindowVisible(hwnd):
                title = self.win32gui.GetWindowText(hwnd)
                if title:
                    rect = self.win32gui.GetWindowRect(hwnd)
                    
                    # Get process info
                    try:
                        _, pid = self.win32process.GetWindowThreadProcessId(hwnd)
                        process_name = None  # Could get from pid if needed
                    except:
                        pid = None
                        process_name = None
                    
                    windows_list.append(WindowInfo(
                        window_id=hwnd,
                        title=title,
                        x=rect[0],
                        y=rect[1],
                        width=rect[2] - rect[0],
                        height=rect[3] - rect[1],
                        is_visible=True,
                        is_focused=(hwnd == self.win32gui.GetForegroundWindow()),
                        process_id=pid,
                        process_name=process_name
                    ))
            return True
        
        self.win32gui.EnumWindows(callback, windows)
        return windows
    
    def move_window(self, window_id: Any, x: int, y: int) -> bool:
        """Move window"""
        if not self.win32gui:
            return False
        
        try:
            self.win32gui.SetWindowPos(window_id, 0, x, y, 0, 0, 
                                      self.win32con.SWP_NOSIZE | self.win32con.SWP_NOZORDER)
            return True
        except Exception as e:
            logger.error(f"Failed to move window: {e}")
            return False
    
    def resize_window(self, window_id: Any, width: int, height: int) -> bool:
        """Resize window"""
        if not self.win32gui:
            return False
        
        try:
            rect = self.win32gui.GetWindowRect(window_id)
            self.win32gui.SetWindowPos(window_id, 0, rect[0], rect[1], width, height,
                                      self.win32con.SWP_NOZORDER)
            return True
        except Exception as e:
            logger.error(f"Failed to resize window: {e}")
            return False
    
    def focus_window(self, window_id: Any) -> bool:
        """Focus window"""
        if not self.win32gui:
            return False
        
        try:
            self.win32gui.SetForegroundWindow(window_id)
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False
    
    def get_focused_window(self) -> Optional[WindowInfo]:
        """Get focused window"""
        if not self.win32gui:
            return None
        
        try:
            hwnd = self.win32gui.GetForegroundWindow()
            if hwnd:
                title = self.win32gui.GetWindowText(hwnd)
                rect = self.win32gui.GetWindowRect(hwnd)
                return WindowInfo(
                    window_id=hwnd,
                    title=title,
                    x=rect[0],
                    y=rect[1],
                    width=rect[2] - rect[0],
                    height=rect[3] - rect[1],
                    is_visible=True,
                    is_focused=True
                )
        except Exception as e:
            logger.error(f"Failed to get focused window: {e}")
        return None
    
    def get_window_geometry(self, window_id: Any) -> Optional[Dict[str, int]]:
        """Get window geometry"""
        if not self.win32gui:
            return None
        
        try:
            rect = self.win32gui.GetWindowRect(window_id)
            return {
                'x': rect[0],
                'y': rect[1],
                'width': rect[2] - rect[0],
                'height': rect[3] - rect[1]
            }
        except Exception as e:
            logger.error(f"Failed to get window geometry: {e}")
            return None


class LinuxWindowAdapter(BaseWindowAdapter):
    """Linux-specific window adapter"""
    
    def __init__(self):
        self.has_wmctrl = self._check_command("wmctrl")
        self.has_xdotool = self._check_command("xdotool")
        self.has_xwininfo = self._check_command("xwininfo")
        
        if not self.has_wmctrl:
            logger.warning("wmctrl not found, window management limited")
    
    def _check_command(self, command: str) -> bool:
        """Check if command is available"""
        try:
            subprocess.run([command, "--help"], capture_output=True)
            return True
        except:
            return False
    
    def list_windows(self) -> List[WindowInfo]:
        """List all windows"""
        if not self.has_wmctrl:
            return []
        
        windows = []
        try:
            result = subprocess.run(['wmctrl', '-l', '-G'], capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(None, 8)
                    if len(parts) >= 8:
                        windows.append(WindowInfo(
                            window_id=parts[0],
                            title=parts[8] if len(parts) > 8 else "",
                            x=int(parts[2]),
                            y=int(parts[3]),
                            width=int(parts[4]),
                            height=int(parts[5]),
                            is_visible=True,
                            is_focused=False  # Would need xdotool to check
                        ))
        except Exception as e:
            logger.error(f"Failed to list windows: {e}")
        
        return windows
    
    def move_window(self, window_id: Any, x: int, y: int) -> bool:
        """Move window"""
        if not self.has_wmctrl:
            return False
        
        try:
            subprocess.run(['wmctrl', '-i', '-r', str(window_id), '-e', f'0,{x},{y},-1,-1'])
            return True
        except Exception as e:
            logger.error(f"Failed to move window: {e}")
            return False
    
    def resize_window(self, window_id: Any, width: int, height: int) -> bool:
        """Resize window"""
        if not self.has_wmctrl:
            return False
        
        try:
            # Get current position
            geometry = self.get_window_geometry(window_id)
            if geometry:
                x, y = geometry['x'], geometry['y']
                subprocess.run(['wmctrl', '-i', '-r', str(window_id), '-e', f'0,{x},{y},{width},{height}'])
                return True
        except Exception as e:
            logger.error(f"Failed to resize window: {e}")
        return False
    
    def focus_window(self, window_id: Any) -> bool:
        """Focus window"""
        if not self.has_wmctrl:
            return False
        
        try:
            subprocess.run(['wmctrl', '-i', '-a', str(window_id)])
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False
    
    def get_focused_window(self) -> Optional[WindowInfo]:
        """Get focused window"""
        if not self.has_xdotool:
            return None
        
        try:
            result = subprocess.run(['xdotool', 'getactivewindow'], capture_output=True, text=True)
            window_id = result.stdout.strip()
            
            if window_id:
                # Get window info
                result = subprocess.run(['xdotool', 'getwindowname', window_id], capture_output=True, text=True)
                title = result.stdout.strip()
                
                geometry = self.get_window_geometry(window_id)
                if geometry:
                    return WindowInfo(
                        window_id=window_id,
                        title=title,
                        x=geometry['x'],
                        y=geometry['y'],
                        width=geometry['width'],
                        height=geometry['height'],
                        is_visible=True,
                        is_focused=True
                    )
        except Exception as e:
            logger.error(f"Failed to get focused window: {e}")
        return None
    
    def get_window_geometry(self, window_id: Any) -> Optional[Dict[str, int]]:
        """Get window geometry"""
        if self.has_xwininfo:
            try:
                result = subprocess.run(['xwininfo', '-id', str(window_id)], 
                                      capture_output=True, text=True)
                
                geometry = {}
                for line in result.stdout.split('\n'):
                    if 'Absolute upper-left X:' in line:
                        geometry['x'] = int(line.split(':')[1].strip())
                    elif 'Absolute upper-left Y:' in line:
                        geometry['y'] = int(line.split(':')[1].strip())
                    elif 'Width:' in line:
                        geometry['width'] = int(line.split(':')[1].strip())
                    elif 'Height:' in line:
                        geometry['height'] = int(line.split(':')[1].strip())
                
                if len(geometry) == 4:
                    return geometry
            except Exception as e:
                logger.error(f"Failed to get window geometry: {e}")
        
        return None


class FallbackWindowAdapter(BaseWindowAdapter):
    """Fallback adapter with minimal functionality"""
    
    def __init__(self):
        logger.warning("Using fallback window adapter - limited functionality")
    
    def list_windows(self) -> List[WindowInfo]:
        """List windows - returns empty list in fallback"""
        logger.warning("Window listing not available in fallback mode")
        return []
    
    def capture_window(self, window_id: Any) -> Optional[np.ndarray]:
        """Capture full screen as fallback"""
        try:
            import mss
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[1])
                return np.array(screenshot)[:, :, :3]
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
        return None


# Compatibility exports
WindowManager = WindowController