"""Window management and control for games"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import structlog

logger = structlog.get_logger()

try:
    if os.name == 'nt':
        import win32gui
        import win32con
        import win32process
        import win32api
        WINDOWS_AVAILABLE = True
    else:
        WINDOWS_AVAILABLE = False
except ImportError:
    WINDOWS_AVAILABLE = False

try:
    import subprocess
    LINUX_TOOLS_AVAILABLE = True
except ImportError:
    LINUX_TOOLS_AVAILABLE = False


class WindowController:
    """Cross-platform window management"""
    
    def __init__(self):
        self.platform = os.name
        self.current_window = None
        
    def find_window_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Find window by title (partial match)"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            return self._find_window_windows(title)
        elif LINUX_TOOLS_AVAILABLE:
            return self._find_window_linux(title)
        
        return None
    
    def find_window_by_pid(self, pid: int) -> Optional[Dict[str, Any]]:
        """Find window by process ID"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            return self._find_window_by_pid_windows(pid)
        elif LINUX_TOOLS_AVAILABLE:
            return self._find_window_by_pid_linux(pid)
        
        return None
    
    def get_window_rect(self, window_handle: Any) -> Optional[Tuple[int, int, int, int]]:
        """Get window rectangle (x, y, width, height)"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                rect = win32gui.GetWindowRect(window_handle)
                return (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return None
    
    def move_window(self, window_handle: Any, x: int, y: int) -> bool:
        """Move window to position"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                rect = win32gui.GetWindowRect(window_handle)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                win32gui.MoveWindow(window_handle, x, y, width, height, True)
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                subprocess.run(["xdotool", "windowmove", str(window_handle), str(x), str(y)])
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return False
    
    def resize_window(self, window_handle: Any, width: int, height: int) -> bool:
        """Resize window"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                rect = win32gui.GetWindowRect(window_handle)
                x, y = rect[0], rect[1]
                win32gui.MoveWindow(window_handle, x, y, width, height, True)
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                subprocess.run(["xdotool", "windowsize", str(window_handle), str(width), str(height)])
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return False
    
    def focus_window(self, window_handle: Any) -> bool:
        """Bring window to foreground"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                # Restore if minimized
                if win32gui.IsIconic(window_handle):
                    win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
                
                # Bring to foreground
                win32gui.SetForegroundWindow(window_handle)
                
                # Also try alternative method
                win32gui.BringWindowToTop(window_handle)
                
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                subprocess.run(["xdotool", "windowactivate", str(window_handle)])
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return False
    
    def minimize_window(self, window_handle: Any) -> bool:
        """Minimize window"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                win32gui.ShowWindow(window_handle, win32con.SW_MINIMIZE)
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                subprocess.run(["xdotool", "windowminimize", str(window_handle)])
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return False
    
    def maximize_window(self, window_handle: Any) -> bool:
        """Maximize window"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                win32gui.ShowWindow(window_handle, win32con.SW_MAXIMIZE)
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                subprocess.run(["xdotool", "windowmaximize", str(window_handle)])
                return True
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return False
    
    def is_window_visible(self, window_handle: Any) -> bool:
        """Check if window is visible"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                return win32gui.IsWindowVisible(window_handle)
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return False
    
    def get_foreground_window(self) -> Optional[Dict[str, Any]]:
        """Get currently focused window"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                
                return {
                    "handle": hwnd,
                    "title": title,
                    "x": rect[0],
                    "y": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                }
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                result = subprocess.run(
                    ["xdotool", "getactivewindow"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    window_id = result.stdout.strip()
                    return {"handle": window_id}
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return None
    
    def list_windows(self) -> List[Dict[str, Any]]:
        """List all visible windows"""
        windows = []
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            def callback(hwnd, windows_list):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        rect = win32gui.GetWindowRect(hwnd)
                        windows_list.append({
                            "handle": hwnd,
                            "title": title,
                            "x": rect[0],
                            "y": rect[1],
                            "width": rect[2] - rect[0],
                            "height": rect[3] - rect[1]
                        })
            
            win32gui.EnumWindows(callback, windows)
        
        elif LINUX_TOOLS_AVAILABLE:
            try:
                result = subprocess.run(
                    ["xdotool", "search", "--onlyvisible", "--name", ".*"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    window_ids = result.stdout.strip().split('\n')
                    for window_id in window_ids:
                        if window_id:
                            windows.append({"handle": window_id})
            except Exception as e:
                logger.debug(f"Failed to get window geometry: {e}")
        
        return windows
    
    def capture_window(self, window_handle: Any) -> Optional[Any]:
        """Capture window screenshot"""
        
        if self.platform == 'nt' and WINDOWS_AVAILABLE:
            try:
                import win32ui
                import win32con
                from PIL import Image
                
                # Get window dimensions
                rect = win32gui.GetWindowRect(window_handle)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                
                # Get window device context
                hwnd_dc = win32gui.GetWindowDC(window_handle)
                mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                save_dc = mfc_dc.CreateCompatibleDC()
                
                # Create bitmap
                save_bitmap = win32ui.CreateBitmap()
                save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                save_dc.SelectObject(save_bitmap)
                
                # Copy window to bitmap
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
                
                # Convert to PIL Image
                bmpinfo = save_bitmap.GetInfo()
                bmpstr = save_bitmap.GetBitmapBits(True)
                
                image = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1
                )
                
                # Cleanup
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(window_handle, hwnd_dc)
                
                return image
                
            except Exception as e:
                logger.error(f"Failed to capture window: {e}")
        
        return None
    
    def _find_window_windows(self, title: str) -> Optional[Dict[str, Any]]:
        """Find window on Windows"""
        
        found_windows = []
        
        def callback(hwnd, pattern):
            window_title = win32gui.GetWindowText(hwnd)
            if pattern.lower() in window_title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                found_windows.append({
                    "handle": hwnd,
                    "title": window_title,
                    "x": rect[0],
                    "y": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                })
        
        win32gui.EnumWindows(callback, title)
        
        if found_windows:
            # Return best match (exact match or largest window)
            for window in found_windows:
                if window["title"].lower() == title.lower():
                    return window
            
            return max(found_windows, key=lambda w: w["width"] * w["height"])
        
        return None
    
    def _find_window_linux(self, title: str) -> Optional[Dict[str, Any]]:
        """Find window on Linux"""
        
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                window_id = result.stdout.strip().split()[0]
                return {"handle": window_id, "title": title}
        except:
            pass
        
        return None
    
    def _find_window_by_pid_windows(self, pid: int) -> Optional[Dict[str, Any]]:
        """Find window by PID on Windows"""
        
        found_windows = []
        
        def callback(hwnd, target_pid):
            _, window_pid = win32process.GetWindowThreadProcessId(hwnd)
            if window_pid == target_pid:
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    rect = win32gui.GetWindowRect(hwnd)
                    found_windows.append({
                        "handle": hwnd,
                        "title": title,
                        "pid": window_pid,
                        "x": rect[0],
                        "y": rect[1],
                        "width": rect[2] - rect[0],
                        "height": rect[3] - rect[1]
                    })
        
        win32gui.EnumWindows(callback, pid)
        
        if found_windows:
            # Return largest window
            return max(found_windows, key=lambda w: w["width"] * w["height"])
        
        return None
    
    def _find_window_by_pid_linux(self, pid: int) -> Optional[Dict[str, Any]]:
        """Find window by PID on Linux"""
        
        try:
            result = subprocess.run(
                ["xdotool", "search", "--pid", str(pid)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                window_id = result.stdout.strip().split()[0]
                return {"handle": window_id, "pid": pid}
        except:
            pass
        
        return None