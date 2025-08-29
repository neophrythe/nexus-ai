"""
BlueStacks emulator controller for Nexus Game AI Framework.
Full support for Android game automation through BlueStacks.
"""

import subprocess
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import cv2
import structlog

logger = structlog.get_logger()


@dataclass
class BlueStacksConfig:
    """BlueStacks configuration."""
    instance_name: str = "BlueStacks"  # Or "BlueStacks_nxt" for BlueStacks 5
    adb_port: int = 5555
    window_title_pattern: str = "BlueStacks"
    install_path: str = r"C:\Program Files\BlueStacks_nxt"  # Default for BlueStacks 5
    
    # Performance settings
    use_advanced_graphics: bool = True
    enable_virtualization: bool = True
    cpu_cores: int = 4
    ram_mb: int = 4096
    
    # Input settings
    use_adb_input: bool = True  # Use ADB for input (more reliable)
    use_window_input: bool = False  # Use Windows API as fallback
    input_delay_ms: int = 50
    
    # Screen settings
    resolution: Tuple[int, int] = (1920, 1080)
    dpi: int = 240
    fps_cap: int = 60


class BlueStacksController:
    """Controller for BlueStacks Android emulator."""
    
    def __init__(self, config: Optional[BlueStacksConfig] = None):
        self.config = config or BlueStacksConfig()
        self.adb_connected = False
        self.window_handle = None
        self.current_package = None
        self.current_activity = None
        
        # Initialize ADB connection
        self._init_adb()
        
    def _init_adb(self):
        """Initialize ADB connection to BlueStacks."""
        try:
            # Check if ADB is available
            result = subprocess.run(['adb', 'version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("ADB not found in PATH")
                return
            
            # Connect to BlueStacks
            adb_address = f"127.0.0.1:{self.config.adb_port}"
            result = subprocess.run(
                ['adb', 'connect', adb_address],
                capture_output=True,
                text=True
            )
            
            if "connected" in result.stdout.lower():
                self.adb_connected = True
                logger.info(f"Connected to BlueStacks via ADB at {adb_address}")
            else:
                logger.warning(f"Failed to connect to BlueStacks: {result.stdout}")
                
        except Exception as e:
            logger.error(f"ADB initialization error: {e}")
    
    def start_bluestacks(self) -> bool:
        """Start BlueStacks if not running."""
        try:
            # Check if already running
            if self.is_running():
                logger.info("BlueStacks is already running")
                return True
            
            # Find BlueStacks executable
            exe_paths = [
                Path(self.config.install_path) / "HD-Player.exe",
                Path(self.config.install_path) / "BlueStacks.exe",
                Path(r"C:\Program Files\BlueStacks") / "HD-Player.exe",
                Path(r"C:\Program Files\BlueStacks_nxt") / "HD-Player.exe"
            ]
            
            exe_path = None
            for path in exe_paths:
                if path.exists():
                    exe_path = path
                    break
            
            if not exe_path:
                logger.error("BlueStacks executable not found")
                return False
            
            # Start BlueStacks
            subprocess.Popen([str(exe_path)])
            logger.info(f"Starting BlueStacks from {exe_path}")
            
            # Wait for BlueStacks to start
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.is_running():
                    time.sleep(5)  # Extra time for full initialization
                    self._init_adb()  # Reconnect ADB
                    return True
            
            logger.error("BlueStacks failed to start")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start BlueStacks: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if BlueStacks is running."""
        try:
            result = subprocess.run(
                ['tasklist', '/FI', f'IMAGENAME eq HD-Player.exe'],
                capture_output=True,
                text=True
            )
            return "HD-Player.exe" in result.stdout
        except:
            return False
    
    def launch_app(self, package_name: str, activity: Optional[str] = None) -> bool:
        """Launch an Android app in BlueStacks.
        
        Args:
            package_name: Android package name (e.g., 'com.supercell.clashofclans')
            activity: Main activity name (optional)
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            logger.error("ADB not connected")
            return False
        
        try:
            if activity:
                # Launch specific activity
                cmd = ['adb', 'shell', 'am', 'start', 
                       f'{package_name}/{activity}']
            else:
                # Launch default activity
                cmd = ['adb', 'shell', 'monkey', '-p', package_name, 
                       '-c', 'android.intent.category.LAUNCHER', '1']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.current_package = package_name
                self.current_activity = activity
                logger.info(f"Launched app: {package_name}")
                return True
            else:
                logger.error(f"Failed to launch app: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"App launch error: {e}")
            return False
    
    def install_app(self, apk_path: str) -> bool:
        """Install an APK file to BlueStacks.
        
        Args:
            apk_path: Path to APK file
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            logger.error("ADB not connected")
            return False
        
        try:
            logger.info(f"Installing APK: {apk_path}")
            result = subprocess.run(
                ['adb', 'install', '-r', apk_path],
                capture_output=True,
                text=True
            )
            
            if "Success" in result.stdout:
                logger.info("APK installed successfully")
                return True
            else:
                logger.error(f"APK installation failed: {result.stdout}")
                return False
                
        except Exception as e:
            logger.error(f"APK installation error: {e}")
            return False
    
    def tap(self, x: int, y: int) -> bool:
        """Tap at coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Success status
        """
        if self.config.use_adb_input and self.adb_connected:
            return self._adb_tap(x, y)
        elif self.config.use_window_input:
            return self._window_tap(x, y)
        else:
            logger.error("No input method available")
            return False
    
    def _adb_tap(self, x: int, y: int) -> bool:
        """Tap using ADB."""
        try:
            result = subprocess.run(
                ['adb', 'shell', 'input', 'tap', str(x), str(y)],
                capture_output=True,
                text=True
            )
            
            if self.config.input_delay_ms > 0:
                time.sleep(self.config.input_delay_ms / 1000)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"ADB tap error: {e}")
            return False
    
    def _window_tap(self, x: int, y: int) -> bool:
        """Tap using Windows API."""
        try:
            import win32api
            import win32con
            import win32gui
            
            # Find BlueStacks window
            hwnd = win32gui.FindWindow(None, self.config.window_title_pattern)
            if not hwnd:
                logger.error("BlueStacks window not found")
                return False
            
            # Convert to window coordinates
            lParam = win32api.MAKELONG(x, y)
            
            # Send click
            win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, 
                                win32con.MK_LBUTTON, lParam)
            win32api.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, lParam)
            
            if self.config.input_delay_ms > 0:
                time.sleep(self.config.input_delay_ms / 1000)
            
            return True
            
        except Exception as e:
            logger.error(f"Window tap error: {e}")
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, 
              duration_ms: int = 300) -> bool:
        """Swipe from one point to another.
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            duration_ms: Swipe duration in milliseconds
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            return False
        
        try:
            result = subprocess.run(
                ['adb', 'shell', 'input', 'swipe', 
                 str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Swipe error: {e}")
            return False
    
    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> bool:
        """Long press at coordinates.
        
        Args:
            x, y: Coordinates
            duration_ms: Press duration in milliseconds
            
        Returns:
            Success status
        """
        return self.swipe(x, y, x, y, duration_ms)
    
    def multi_touch(self, points: List[Tuple[int, int]], 
                   duration_ms: int = 100) -> bool:
        """Perform multi-touch gesture (pinch, zoom, etc).
        
        Args:
            points: List of (x, y) coordinates
            duration_ms: Gesture duration
            
        Returns:
            Success status
        """
        if not self.adb_connected or len(points) < 2:
            return False
        
        try:
            # Create sendevent script for multi-touch
            script = "#!/system/bin/sh\n"
            
            # Touch down events
            for i, (x, y) in enumerate(points):
                script += f"sendevent /dev/input/event0 3 57 {i}\n"  # Tracking ID
                script += f"sendevent /dev/input/event0 3 53 {x}\n"  # X position
                script += f"sendevent /dev/input/event0 3 54 {y}\n"  # Y position
                script += f"sendevent /dev/input/event0 3 48 5\n"    # Touch major
                script += f"sendevent /dev/input/event0 3 58 50\n"   # Pressure
                script += "sendevent /dev/input/event0 0 0 0\n"      # Sync
            
            # Hold
            script += f"sleep {duration_ms / 1000}\n"
            
            # Touch up events
            for i in range(len(points)):
                script += f"sendevent /dev/input/event0 3 57 {i}\n"
                script += "sendevent /dev/input/event0 3 57 -1\n"    # Release
                script += "sendevent /dev/input/event0 0 0 0\n"
            
            # Execute script
            result = subprocess.run(
                ['adb', 'shell', script],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Multi-touch error: {e}")
            return False
    
    def send_text(self, text: str) -> bool:
        """Send text input.
        
        Args:
            text: Text to send
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            return False
        
        try:
            # Escape special characters
            text = text.replace(' ', '%s')
            text = text.replace('&', '\\&')
            text = text.replace('|', '\\|')
            text = text.replace('<', '\\<')
            text = text.replace('>', '\\>')
            text = text.replace('(', '\\(')
            text = text.replace(')', '\\)')
            text = text.replace('$', '\\$')
            
            result = subprocess.run(
                ['adb', 'shell', 'input', 'text', text],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Text input error: {e}")
            return False
    
    def press_key(self, keycode: str) -> bool:
        """Press a key.
        
        Args:
            keycode: Android keycode (e.g., 'KEYCODE_BACK', 'KEYCODE_HOME')
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            return False
        
        try:
            result = subprocess.run(
                ['adb', 'shell', 'input', 'keyevent', keycode],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Key press error: {e}")
            return False
    
    def get_screenshot(self) -> Optional[np.ndarray]:
        """Get screenshot from BlueStacks.
        
        Returns:
            Screenshot as numpy array or None
        """
        if self.adb_connected:
            return self._adb_screenshot()
        else:
            return self._window_screenshot()
    
    def _adb_screenshot(self) -> Optional[np.ndarray]:
        """Get screenshot using ADB."""
        try:
            # Capture screenshot to device
            subprocess.run(
                ['adb', 'shell', 'screencap', '-p', '/sdcard/screen.png'],
                capture_output=True
            )
            
            # Pull screenshot to local
            temp_path = Path("temp_screen.png")
            subprocess.run(
                ['adb', 'pull', '/sdcard/screen.png', str(temp_path)],
                capture_output=True
            )
            
            # Load image
            if temp_path.exists():
                img = cv2.imread(str(temp_path))
                temp_path.unlink()  # Delete temp file
                return img
            
            return None
            
        except Exception as e:
            logger.error(f"ADB screenshot error: {e}")
            return None
    
    def _window_screenshot(self) -> Optional[np.ndarray]:
        """Get screenshot using window capture."""
        try:
            from nexus.core.frame_grabber import FrameGrabber, WindowInfo
            
            # Find BlueStacks window
            import win32gui
            
            def callback(hwnd, windows):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if self.config.window_title_pattern in title:
                        windows.append(hwnd)
            
            windows = []
            win32gui.EnumWindows(callback, windows)
            
            if not windows:
                logger.error("BlueStacks window not found")
                return None
            
            # Get window info
            hwnd = windows[0]
            rect = win32gui.GetWindowRect(hwnd)
            
            window_info = WindowInfo(
                title=win32gui.GetWindowText(hwnd),
                x=rect[0],
                y=rect[1],
                width=rect[2] - rect[0],
                height=rect[3] - rect[1]
            )
            
            # Capture frame
            grabber = FrameGrabber()
            return grabber.grab_frame(window_info)
            
        except Exception as e:
            logger.error(f"Window screenshot error: {e}")
            return None
    
    def get_ui_hierarchy(self) -> Optional[Dict[str, Any]]:
        """Get UI hierarchy dump for analysis.
        
        Returns:
            UI hierarchy as dict or None
        """
        if not self.adb_connected:
            return None
        
        try:
            # Dump UI hierarchy
            result = subprocess.run(
                ['adb', 'shell', 'uiautomator', 'dump', '/sdcard/ui.xml'],
                capture_output=True,
                text=True
            )
            
            # Pull XML file
            temp_path = Path("temp_ui.xml")
            subprocess.run(
                ['adb', 'pull', '/sdcard/ui.xml', str(temp_path)],
                capture_output=True
            )
            
            if temp_path.exists():
                # Parse XML
                import xml.etree.ElementTree as ET
                tree = ET.parse(temp_path)
                root = tree.getroot()
                
                # Convert to dict
                hierarchy = self._xml_to_dict(root)
                
                temp_path.unlink()  # Delete temp file
                return hierarchy
            
            return None
            
        except Exception as e:
            logger.error(f"UI hierarchy error: {e}")
            return None
    
    def _xml_to_dict(self, element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {
            'tag': element.tag,
            'attrib': element.attrib,
            'children': []
        }
        
        for child in element:
            result['children'].append(self._xml_to_dict(child))
        
        return result
    
    def find_element(self, text: Optional[str] = None,
                    resource_id: Optional[str] = None,
                    class_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find UI element by attributes.
        
        Args:
            text: Element text
            resource_id: Resource ID
            class_name: Class name
            
        Returns:
            Element info with bounds or None
        """
        hierarchy = self.get_ui_hierarchy()
        if not hierarchy:
            return None
        
        def search_element(node):
            """Recursively search for element."""
            # Check current node
            attrib = node.get('attrib', {})
            
            match = True
            if text and text not in attrib.get('text', ''):
                match = False
            if resource_id and resource_id != attrib.get('resource-id', ''):
                match = False
            if class_name and class_name != attrib.get('class', ''):
                match = False
            
            if match and attrib:
                # Parse bounds
                bounds_str = attrib.get('bounds', '')
                if bounds_str:
                    # Format: [x1,y1][x2,y2]
                    import re
                    coords = re.findall(r'\d+', bounds_str)
                    if len(coords) == 4:
                        return {
                            'text': attrib.get('text', ''),
                            'resource_id': attrib.get('resource-id', ''),
                            'class': attrib.get('class', ''),
                            'bounds': {
                                'x1': int(coords[0]),
                                'y1': int(coords[1]),
                                'x2': int(coords[2]),
                                'y2': int(coords[3]),
                                'center_x': (int(coords[0]) + int(coords[2])) // 2,
                                'center_y': (int(coords[1]) + int(coords[3])) // 2
                            }
                        }
            
            # Search children
            for child in node.get('children', []):
                result = search_element(child)
                if result:
                    return result
            
            return None
        
        return search_element(hierarchy)
    
    def tap_element(self, text: Optional[str] = None,
                   resource_id: Optional[str] = None,
                   class_name: Optional[str] = None) -> bool:
        """Find and tap UI element.
        
        Args:
            text: Element text
            resource_id: Resource ID
            class_name: Class name
            
        Returns:
            Success status
        """
        element = self.find_element(text, resource_id, class_name)
        if element:
            bounds = element['bounds']
            return self.tap(bounds['center_x'], bounds['center_y'])
        
        logger.warning(f"Element not found: text={text}, id={resource_id}")
        return False
    
    def wait_for_element(self, text: Optional[str] = None,
                        resource_id: Optional[str] = None,
                        class_name: Optional[str] = None,
                        timeout: float = 10) -> bool:
        """Wait for UI element to appear.
        
        Args:
            text: Element text
            resource_id: Resource ID
            class_name: Class name
            timeout: Timeout in seconds
            
        Returns:
            True if element found, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            element = self.find_element(text, resource_id, class_name)
            if element:
                return True
            time.sleep(0.5)
        
        return False
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state information.
        
        Returns:
            Game state dict
        """
        state = {
            'connected': self.adb_connected,
            'running': self.is_running(),
            'current_app': self.current_package,
            'timestamp': time.time()
        }
        
        if self.adb_connected:
            # Get additional info
            try:
                # Get current activity
                result = subprocess.run(
                    ['adb', 'shell', 'dumpsys', 'window', 'windows'],
                    capture_output=True,
                    text=True
                )
                
                # Parse current focus
                for line in result.stdout.split('\n'):
                    if 'mCurrentFocus' in line:
                        state['current_focus'] = line.strip()
                        break
                
                # Get memory info
                result = subprocess.run(
                    ['adb', 'shell', 'dumpsys', 'meminfo', '-s'],
                    capture_output=True,
                    text=True
                )
                
                state['memory_info'] = result.stdout[:500]  # First 500 chars
                
            except Exception as e:
                logger.error(f"Failed to get game state: {e}")
        
        return state
    
    def record_gameplay(self, output_path: str, duration: int = 60) -> bool:
        """Record gameplay video.
        
        Args:
            output_path: Output video path
            duration: Recording duration in seconds
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            return False
        
        try:
            # Start recording
            subprocess.Popen(
                ['adb', 'shell', 'screenrecord', '--time-limit', str(duration),
                 '/sdcard/gameplay.mp4']
            )
            
            logger.info(f"Recording for {duration} seconds...")
            time.sleep(duration + 2)  # Extra time for processing
            
            # Pull video file
            result = subprocess.run(
                ['adb', 'pull', '/sdcard/gameplay.mp4', output_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Gameplay saved to {output_path}")
                return True
            else:
                logger.error(f"Failed to save recording: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return False
    
    def enable_show_taps(self, enable: bool = True) -> bool:
        """Enable/disable show taps for debugging.
        
        Args:
            enable: True to show taps, False to hide
            
        Returns:
            Success status
        """
        if not self.adb_connected:
            return False
        
        try:
            value = "1" if enable else "0"
            result = subprocess.run(
                ['adb', 'shell', 'settings', 'put', 'system', 
                 'show_touches', value],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Show taps error: {e}")
            return False
    
    def get_installed_apps(self) -> List[str]:
        """Get list of installed apps.
        
        Returns:
            List of package names
        """
        if not self.adb_connected:
            return []
        
        try:
            result = subprocess.run(
                ['adb', 'shell', 'pm', 'list', 'packages'],
                capture_output=True,
                text=True
            )
            
            apps = []
            for line in result.stdout.split('\n'):
                if line.startswith('package:'):
                    apps.append(line.replace('package:', '').strip())
            
            return apps
            
        except Exception as e:
            logger.error(f"Failed to get installed apps: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources."""
        if self.adb_connected:
            try:
                subprocess.run(['adb', 'disconnect'], capture_output=True)
                logger.info("Disconnected from BlueStacks")
            except Exception as e:
                logger.debug(f"Cleanup error (non-critical): {e}")