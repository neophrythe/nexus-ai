"""
ADB Client for Android Device Communication

Provides interface to Android Debug Bridge for device control.
"""

import subprocess
import re
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class ADBDevice:
    """Represents an Android device/emulator."""
    device_id: str
    status: str
    product: str = ""
    model: str = ""
    device_type: str = ""  # 'device' or 'emulator'
    transport_id: str = ""
    android_version: str = ""
    screen_resolution: Tuple[int, int] = (0, 0)
    
    @property
    def is_emulator(self) -> bool:
        return self.device_type == 'emulator' or self.device_id.startswith('emulator')
    
    @property
    def is_online(self) -> bool:
        return self.status == 'device'


class ADBClient:
    """
    Client for Android Debug Bridge communication.
    
    Features:
    - Device discovery and management
    - App installation and launching
    - Screen capture and recording
    - Input event injection
    - File transfer
    - Shell command execution
    """
    
    def __init__(self, adb_path: str = None):
        """
        Initialize ADB client.
        
        Args:
            adb_path: Path to adb executable (auto-detect if None)
        """
        self.adb_path = adb_path or self._find_adb()
        
        if not self.adb_path:
            raise RuntimeError("ADB not found. Please install Android SDK.")
        
        self.current_device: Optional[ADBDevice] = None
        
        # Start ADB server
        self._start_server()
        
        logger.info(f"ADB client initialized with: {self.adb_path}")
    
    def _find_adb(self) -> Optional[str]:
        """Find ADB executable in system."""
        # Check if adb is in PATH
        adb = shutil.which('adb')
        if adb:
            return adb
        
        # Check common locations
        common_paths = [
            Path.home() / 'Android' / 'Sdk' / 'platform-tools' / 'adb',
            Path.home() / 'AppData' / 'Local' / 'Android' / 'Sdk' / 'platform-tools' / 'adb.exe',
            Path('/usr/local/bin/adb'),
            Path('C:/android-sdk/platform-tools/adb.exe')
        ]
        
        for path in common_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _start_server(self):
        """Start ADB server."""
        try:
            self._run_adb(['start-server'])
            logger.info("ADB server started")
        except Exception as e:
            logger.error(f"Failed to start ADB server: {e}")
    
    def _run_adb(self, args: List[str], device_id: str = None) -> str:
        """Run ADB command."""
        cmd = [self.adb_path]
        
        if device_id:
            cmd.extend(['-s', device_id])
        elif self.current_device:
            cmd.extend(['-s', self.current_device.device_id])
        
        cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"ADB command failed: {' '.join(cmd)}")
                logger.error(f"Error: {result.stderr}")
                return result.stderr
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timed out: {' '.join(cmd)}")
            return ""
        except Exception as e:
            logger.error(f"ADB command error: {e}")
            return ""
    
    def get_devices(self) -> List[ADBDevice]:
        """
        Get list of connected devices/emulators.
        
        Returns:
            List of ADB devices
        """
        output = self._run_adb(['devices', '-l'])
        devices = []
        
        for line in output.strip().split('\n')[1:]:
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            device_id = parts[0]
            status = parts[1]
            
            # Parse additional info
            device = ADBDevice(device_id=device_id, status=status)
            
            # Extract properties from -l output
            for part in parts[2:]:
                if ':' in part:
                    key, value = part.split(':', 1)
                    if key == 'product':
                        device.product = value
                    elif key == 'model':
                        device.model = value
                    elif key == 'device':
                        device.device_type = value
                    elif key == 'transport_id':
                        device.transport_id = value
            
            # Determine if emulator
            if device_id.startswith('emulator') or 'emulator' in device.product.lower():
                device.device_type = 'emulator'
            else:
                device.device_type = 'device'
            
            # Get additional info if device is online
            if device.is_online:
                # Get Android version
                version = self._run_adb(
                    ['shell', 'getprop', 'ro.build.version.release'],
                    device_id
                ).strip()
                device.android_version = version
                
                # Get screen resolution
                resolution = self._run_adb(
                    ['shell', 'wm', 'size'],
                    device_id
                ).strip()
                
                match = re.search(r'(\d+)x(\d+)', resolution)
                if match:
                    device.screen_resolution = (
                        int(match.group(1)),
                        int(match.group(2))
                    )
            
            devices.append(device)
        
        logger.info(f"Found {len(devices)} devices")
        return devices
    
    def connect_device(self, device_id: str = None) -> bool:
        """
        Connect to a specific device.
        
        Args:
            device_id: Device ID (uses first available if None)
        
        Returns:
            True if connected successfully
        """
        devices = self.get_devices()
        
        if not devices:
            logger.error("No devices found")
            return False
        
        if device_id:
            # Find specific device
            for device in devices:
                if device.device_id == device_id:
                    self.current_device = device
                    logger.info(f"Connected to device: {device_id}")
                    return True
            
            logger.error(f"Device not found: {device_id}")
            return False
        else:
            # Use first available device
            self.current_device = devices[0]
            logger.info(f"Connected to device: {self.current_device.device_id}")
            return True
    
    def install_app(self, apk_path: str) -> bool:
        """
        Install APK on device.
        
        Args:
            apk_path: Path to APK file
        
        Returns:
            True if installed successfully
        """
        if not self.current_device:
            logger.error("No device connected")
            return False
        
        logger.info(f"Installing APK: {apk_path}")
        output = self._run_adb(['install', '-r', apk_path])
        
        if 'Success' in output:
            logger.info("APK installed successfully")
            return True
        else:
            logger.error(f"APK installation failed: {output}")
            return False
    
    def uninstall_app(self, package_name: str) -> bool:
        """
        Uninstall app from device.
        
        Args:
            package_name: Package name (e.g., com.example.app)
        
        Returns:
            True if uninstalled successfully
        """
        if not self.current_device:
            logger.error("No device connected")
            return False
        
        output = self._run_adb(['uninstall', package_name])
        
        if 'Success' in output:
            logger.info(f"Uninstalled: {package_name}")
            return True
        else:
            logger.error(f"Uninstall failed: {output}")
            return False
    
    def launch_app(self, package_name: str, activity: str = None) -> bool:
        """
        Launch app on device.
        
        Args:
            package_name: Package name
            activity: Activity name (auto-detect if None)
        
        Returns:
            True if launched successfully
        """
        if not self.current_device:
            logger.error("No device connected")
            return False
        
        if not activity:
            # Try to get main activity
            activity = self._get_main_activity(package_name)
            if not activity:
                logger.error(f"Could not find main activity for {package_name}")
                return False
        
        # Launch app
        cmd = [
            'shell', 'am', 'start',
            '-n', f"{package_name}/{activity}"
        ]
        
        output = self._run_adb(cmd)
        
        if 'Error' not in output:
            logger.info(f"Launched: {package_name}")
            return True
        else:
            logger.error(f"Launch failed: {output}")
            return False
    
    def stop_app(self, package_name: str) -> bool:
        """
        Stop app on device.
        
        Args:
            package_name: Package name
        
        Returns:
            True if stopped successfully
        """
        if not self.current_device:
            return False
        
        output = self._run_adb(['shell', 'am', 'force-stop', package_name])
        logger.info(f"Stopped: {package_name}")
        return True
    
    def capture_screenshot(self, output_path: str = None) -> Optional[bytes]:
        """
        Capture screenshot from device.
        
        Args:
            output_path: Path to save screenshot (returns bytes if None)
        
        Returns:
            Screenshot data as bytes (if output_path is None)
        """
        if not self.current_device:
            logger.error("No device connected")
            return None
        
        # Capture to device
        device_path = '/sdcard/screenshot.png'
        self._run_adb(['shell', 'screencap', '-p', device_path])
        
        if output_path:
            # Pull to local file
            self._run_adb(['pull', device_path, output_path])
            self._run_adb(['shell', 'rm', device_path])
            logger.info(f"Screenshot saved to: {output_path}")
            return None
        else:
            # Get screenshot data
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                self._run_adb(['pull', device_path, tmp.name])
                self._run_adb(['shell', 'rm', device_path])
                
                with open(tmp.name, 'rb') as f:
                    data = f.read()
                
                Path(tmp.name).unlink()
                return data
    
    def record_screen(self, output_path: str, duration: int = 180):
        """
        Record screen video.
        
        Args:
            output_path: Path to save video
            duration: Recording duration in seconds (max 180)
        """
        if not self.current_device:
            logger.error("No device connected")
            return
        
        device_path = '/sdcard/recording.mp4'
        
        # Start recording
        logger.info(f"Recording screen for {duration} seconds...")
        self._run_adb(
            ['shell', 'screenrecord', '--time-limit', str(duration), device_path]
        )
        
        # Pull video
        self._run_adb(['pull', device_path, output_path])
        self._run_adb(['shell', 'rm', device_path])
        
        logger.info(f"Recording saved to: {output_path}")
    
    def tap(self, x: int, y: int):
        """
        Send tap event.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if not self.current_device:
            return
        
        self._run_adb(['shell', 'input', 'tap', str(x), str(y)])
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """
        Send swipe gesture.
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            duration_ms: Swipe duration in milliseconds
        """
        if not self.current_device:
            return
        
        self._run_adb([
            'shell', 'input', 'swipe',
            str(x1), str(y1), str(x2), str(y2), str(duration_ms)
        ])
    
    def send_key(self, keycode: str):
        """
        Send key event.
        
        Args:
            keycode: Android keycode (e.g., 'KEYCODE_HOME')
        """
        if not self.current_device:
            return
        
        self._run_adb(['shell', 'input', 'keyevent', keycode])
    
    def send_text(self, text: str):
        """
        Send text input.
        
        Args:
            text: Text to input
        """
        if not self.current_device:
            return
        
        # Escape special characters
        text = text.replace(' ', '%s')
        text = text.replace('"', '\\"')
        
        self._run_adb(['shell', 'input', 'text', f'"{text}"'])
    
    def get_current_activity(self) -> Optional[str]:
        """
        Get current foreground activity.
        
        Returns:
            Current activity name
        """
        if not self.current_device:
            return None
        
        output = self._run_adb([
            'shell', 'dumpsys', 'window', 'windows',
            '|', 'grep', '-E', 'mCurrentFocus|mFocusedApp'
        ])
        
        # Parse activity from output
        match = re.search(r'([\w\.]+)/([\w\.]+)}', output)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
        
        return None
    
    def get_installed_packages(self) -> List[str]:
        """
        Get list of installed packages.
        
        Returns:
            List of package names
        """
        if not self.current_device:
            return []
        
        output = self._run_adb(['shell', 'pm', 'list', 'packages'])
        packages = []
        
        for line in output.strip().split('\n'):
            if line.startswith('package:'):
                packages.append(line.replace('package:', ''))
        
        return packages
    
    def push_file(self, local_path: str, device_path: str) -> bool:
        """
        Push file to device.
        
        Args:
            local_path: Local file path
            device_path: Device destination path
        
        Returns:
            True if successful
        """
        if not self.current_device:
            return False
        
        output = self._run_adb(['push', local_path, device_path])
        return 'pushed' in output.lower()
    
    def pull_file(self, device_path: str, local_path: str) -> bool:
        """
        Pull file from device.
        
        Args:
            device_path: Device file path
            local_path: Local destination path
        
        Returns:
            True if successful
        """
        if not self.current_device:
            return False
        
        output = self._run_adb(['pull', device_path, local_path])
        return 'pulled' in output.lower()
    
    def shell(self, command: str) -> str:
        """
        Execute shell command on device.
        
        Args:
            command: Shell command
        
        Returns:
            Command output
        """
        if not self.current_device:
            return ""
        
        return self._run_adb(['shell', command])
    
    def _get_main_activity(self, package_name: str) -> Optional[str]:
        """
        Get main activity of an app.
        
        Args:
            package_name: Package name
        
        Returns:
            Main activity name
        """
        output = self._run_adb([
            'shell', 'cmd', 'package', 'resolve-activity',
            '--brief', package_name
        ])
        
        # Parse activity from output
        for line in output.split('\n'):
            if '/' in line and package_name in line:
                return line.split('/')[-1].strip()
        
        return None
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.
        
        Returns:
            Device information dictionary
        """
        if not self.current_device:
            return {}
        
        info = {
            'device_id': self.current_device.device_id,
            'android_version': self.current_device.android_version,
            'screen_resolution': self.current_device.screen_resolution,
            'device_type': self.current_device.device_type
        }
        
        # Get additional properties
        props = [
            ('manufacturer', 'ro.product.manufacturer'),
            ('model', 'ro.product.model'),
            ('sdk_version', 'ro.build.version.sdk'),
            ('cpu_abi', 'ro.product.cpu.abi'),
            ('ram_size', 'ro.config.ram_size')
        ]
        
        for key, prop in props:
            value = self._run_adb(['shell', 'getprop', prop]).strip()
            if value:
                info[key] = value
        
        return info