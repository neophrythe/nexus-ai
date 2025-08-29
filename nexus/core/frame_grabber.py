"""
Frame grabbing utilities for Nexus Game AI Framework.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass
import structlog
import platform

logger = structlog.get_logger()


@dataclass
class WindowInfo:
    """Window information."""
    title: str
    x: int
    y: int
    width: int
    height: int
    process_name: Optional[str] = None
    is_focused: bool = False


class FrameGrabber:
    """Cross-platform frame grabbing."""
    
    def __init__(self):
        self.platform = platform.system()
        self.last_frame = None
        self.frame_count = 0
        
    def grab_frame(self, window: WindowInfo) -> np.ndarray:
        """Grab a frame from the specified window.
        
        Args:
            window: Window information
            
        Returns:
            Captured frame as numpy array
        """
        if self.platform == "Windows":
            return self._grab_windows(window)
        elif self.platform == "Linux":
            return self._grab_linux(window)
        elif self.platform == "Darwin":  # macOS
            return self._grab_macos(window)
        else:
            raise NotImplementedError(f"Platform {self.platform} not supported")
    
    def _grab_windows(self, window: WindowInfo) -> np.ndarray:
        """Grab frame on Windows."""
        try:
            import win32gui
            import win32ui
            import win32con
            from ctypes import windll
            
            # Get window handle
            hwnd = win32gui.FindWindow(None, window.title)
            if not hwnd:
                logger.warning(f"Window not found: {window.title}")
                return self._get_fallback_frame()
            
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            # Get device contexts
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Copy window contents
            result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            frame = np.frombuffer(bmpstr, dtype='uint8')
            frame = frame.reshape((height, width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            
            self.frame_count += 1
            self.last_frame = frame
            return frame
            
        except Exception as e:
            logger.error(f"Windows frame grab error: {e}")
            return self._get_fallback_frame()
    
    def _grab_linux(self, window: WindowInfo) -> np.ndarray:
        """Grab frame on Linux."""
        try:
            import mss
            
            # Use mss for screen capture
            with mss.mss() as sct:
                monitor = {
                    "top": window.y,
                    "left": window.x,
                    "width": window.width,
                    "height": window.height
                }
                
                # Capture
                img = sct.grab(monitor)
                
                # Convert to numpy array
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                self.frame_count += 1
                self.last_frame = frame
                return frame
                
        except Exception as e:
            logger.error(f"Linux frame grab error: {e}")
            return self._get_fallback_frame()
    
    def _grab_macos(self, window: WindowInfo) -> np.ndarray:
        """Grab frame on macOS."""
        try:
            import Quartz
            import Quartz.CoreGraphics as CG
            
            # Get window list
            window_list = CG.CGWindowListCopyWindowInfo(
                CG.kCGWindowListOptionAll,
                CG.kCGNullWindowID
            )
            
            # Find target window
            target_window = None
            for window_dict in window_list:
                if window.title in window_dict.get(CG.kCGWindowName, ""):
                    target_window = window_dict
                    break
            
            if not target_window:
                logger.warning(f"Window not found: {window.title}")
                return self._get_fallback_frame()
            
            # Get window ID
            window_id = target_window[CG.kCGWindowNumber]
            
            # Capture window
            image = CG.CGWindowListCreateImage(
                CG.CGRectNull,
                CG.kCGWindowListOptionIncludingWindow,
                window_id,
                CG.kCGWindowImageDefault
            )
            
            # Convert to numpy array
            width = CG.CGImageGetWidth(image)
            height = CG.CGImageGetHeight(image)
            bytes_per_row = CG.CGImageGetBytesPerRow(image)
            
            pixel_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
            frame = np.frombuffer(pixel_data, dtype=np.uint8)
            frame = frame.reshape((height, bytes_per_row // 4, 4))
            frame = frame[:, :width, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            self.frame_count += 1
            self.last_frame = frame
            return frame
            
        except Exception as e:
            logger.error(f"macOS frame grab error: {e}")
            return self._get_fallback_frame()
    
    def _get_fallback_frame(self) -> np.ndarray:
        """Get a fallback frame when capture fails."""
        if self.last_frame is not None:
            return self.last_frame
        
        # Return a blank frame
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def grab_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Grab a specific screen region.
        
        Args:
            x: X coordinate
            y: Y coordinate
            width: Region width
            height: Region height
            
        Returns:
            Captured region as numpy array
        """
        window = WindowInfo(
            title="Region",
            x=x,
            y=y,
            width=width,
            height=height
        )
        
        return self.grab_frame(window)
    
    def grab_fullscreen(self) -> np.ndarray:
        """Grab the entire screen.
        
        Returns:
            Full screen capture as numpy array
        """
        try:
            import mss
            
            with mss.mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1]
                
                # Capture
                img = sct.grab(monitor)
                
                # Convert to numpy array
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                return frame
                
        except Exception as e:
            logger.error(f"Fullscreen grab error: {e}")
            return self._get_fallback_frame()
    
    def get_frame_count(self) -> int:
        """Get total frames captured.
        
        Returns:
            Number of frames captured
        """
        return self.frame_count
    
    def save_frame(self, frame: np.ndarray, path: str):
        """Save a frame to file.
        
        Args:
            frame: Frame to save
            path: Output file path
        """
        cv2.imwrite(path, frame)
        logger.info(f"Frame saved to {path}")