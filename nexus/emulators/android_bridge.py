"""
Android Debug Bridge (ADB) wrapper for game automation.
Works with BlueStacks, NoxPlayer, MEmu, and other Android emulators.
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import structlog

logger = structlog.get_logger()


class AndroidBridge:
    """Universal Android emulator controller via ADB."""
    
    # Common emulator ADB ports
    EMULATOR_PORTS = {
        'bluestacks': 5555,
        'bluestacks_nxt': 5555,
        'nox': 62001,
        'memu': 21503,
        'ldplayer': 5555,
        'mumu': 7555,
        'gameloop': 5555
    }
    
    def __init__(self, device: Optional[str] = None):
        """Initialize Android Bridge.
        
        Args:
            device: Device identifier (e.g., '127.0.0.1:5555' or 'emulator-5554')
        """
        self.device = device
        self.connected = False
        self.screen_size = None
        
        # Auto-detect if no device specified
        if not device:
            self.auto_detect()
        else:
            self.connect(device)
    
    def auto_detect(self) -> bool:
        """Auto-detect running emulator."""
        logger.info("Auto-detecting Android emulator...")
        
        # Try common ports
        for emulator, port in self.EMULATOR_PORTS.items():
            device_id = f"127.0.0.1:{port}"
            if self.connect(device_id):
                logger.info(f"Detected {emulator} on port {port}")
                return True
        
        # Check for physical devices or other emulators
        devices = self.list_devices()
        if devices:
            if self.connect(devices[0]):
                logger.info(f"Connected to device: {devices[0]}")
                return True
        
        logger.warning("No Android device detected")
        return False
    
    def connect(self, device: str) -> bool:
        """Connect to Android device.
        
        Args:
            device: Device identifier
            
        Returns:
            Success status
        """
        try:
            # Connect if it's a network device
            if ':' in device:
                result = subprocess.run(
                    ['adb', 'connect', device],
                    capture_output=True,
                    text=True
                )
                
                if 'connected' not in result.stdout.lower() and 'already' not in result.stdout.lower():
                    logger.error(f"Failed to connect: {result.stdout}")
                    return False
            
            self.device = device
            self.connected = True
            
            # Get screen size
            self.screen_size = self.get_screen_size()
            
            logger.info(f"Connected to {device}, screen: {self.screen_size}")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from device."""
        if self.device and ':' in self.device:
            try:
                subprocess.run(['adb', 'disconnect', self.device], 
                             capture_output=True)
                logger.info(f"Disconnected from {self.device}")
            except Exception as e:
                logger.debug(f"Disconnect error (non-critical): {e}")
        
        self.connected = False
        self.device = None
    
    def list_devices(self) -> List[str]:
        """List all connected devices.
        
        Returns:
            List of device identifiers
        """
        try:
            result = subprocess.run(
                ['adb', 'devices'],
                capture_output=True,
                text=True
            )
            
            devices = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if '\t' in line:
                    device_id = line.split('\t')[0]
                    if device_id:
                        devices.append(device_id)
            
            return devices
            
        except Exception as e:
            logger.error(f"Failed to list devices: {e}")
            return []
    
    def _run_adb(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run ADB command with device selection."""
        cmd = ['adb']
        
        if self.device:
            cmd.extend(['-s', self.device])
        
        cmd.extend(args)
        
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    
    def shell(self, command: str) -> str:
        """Execute shell command on device.
        
        Args:
            command: Shell command
            
        Returns:
            Command output
        """
        result = self._run_adb(['shell', command])
        return result.stdout
    
    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """Get screen size.
        
        Returns:
            (width, height) or None
        """
        try:
            output = self.shell('wm size')
            
            # Parse output: "Physical size: 1920x1080"
            import re
            match = re.search(r'(\d+)x(\d+)', output)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get screen size: {e}")
            return None
    
    def tap(self, x: int, y: int) -> bool:
        """Tap at coordinates."""
        try:
            self.shell(f'input tap {x} {y}')
            return True
        except:
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, 
              duration_ms: int = 300) -> bool:
        """Swipe gesture."""
        try:
            self.shell(f'input swipe {x1} {y1} {x2} {y2} {duration_ms}')
            return True
        except:
            return False
    
    def text(self, text: str) -> bool:
        """Input text."""
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
            text = text.replace('"', '\\"')
            text = text.replace("'", "\\'")
            
            self.shell(f'input text "{text}"')
            return True
        except:
            return False
    
    def key(self, keycode: str) -> bool:
        """Send key event.
        
        Common keycodes:
        - KEYCODE_HOME, KEYCODE_BACK, KEYCODE_MENU
        - KEYCODE_ENTER, KEYCODE_TAB, KEYCODE_ESCAPE
        - KEYCODE_DPAD_UP, KEYCODE_DPAD_DOWN, etc.
        """
        try:
            self.shell(f'input keyevent {keycode}')
            return True
        except:
            return False
    
    def screenshot(self) -> Optional[np.ndarray]:
        """Capture screenshot.
        
        Returns:
            Screenshot as numpy array or None
        """
        try:
            # Use screencap with raw output for speed
            result = self._run_adb(['exec-out', 'screencap', '-p'])
            
            if result.returncode == 0:
                # Convert bytes to numpy array
                nparr = np.frombuffer(result.stdout.encode('latin-1'), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            
            # Fallback method
            temp_path = Path('temp_screen.png')
            
            # Capture on device
            self.shell('screencap -p /sdcard/screen.png')
            
            # Pull to local
            self._run_adb(['pull', '/sdcard/screen.png', str(temp_path)])
            
            if temp_path.exists():
                img = cv2.imread(str(temp_path))
                temp_path.unlink()
                return img
            
            return None
            
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return None
    
    def start_app(self, package: str, activity: Optional[str] = None) -> bool:
        """Start application.
        
        Args:
            package: Package name (e.g., 'com.supercell.brawlstars')
            activity: Activity name (optional)
        """
        try:
            if activity:
                self.shell(f'am start -n {package}/{activity}')
            else:
                # Use monkey to launch default activity
                self.shell(f'monkey -p {package} -c android.intent.category.LAUNCHER 1')
            
            return True
        except:
            return False
    
    def stop_app(self, package: str) -> bool:
        """Stop application."""
        try:
            self.shell(f'am force-stop {package}')
            return True
        except:
            return False
    
    def install_apk(self, apk_path: str) -> bool:
        """Install APK file."""
        try:
            result = self._run_adb(['install', '-r', apk_path])
            return 'Success' in result.stdout
        except:
            return False
    
    def uninstall_app(self, package: str) -> bool:
        """Uninstall application."""
        try:
            result = self._run_adb(['uninstall', package])
            return 'Success' in result.stdout
        except:
            return False
    
    def list_packages(self) -> List[str]:
        """List installed packages."""
        try:
            output = self.shell('pm list packages')
            packages = []
            
            for line in output.split('\n'):
                if line.startswith('package:'):
                    packages.append(line.replace('package:', '').strip())
            
            return packages
        except:
            return []
    
    def get_current_activity(self) -> Optional[str]:
        """Get current foreground activity."""
        try:
            output = self.shell('dumpsys window windows | grep -E "mCurrentFocus"')
            
            # Parse output
            import re
            match = re.search(r'(\S+)/(\S+)}', output)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
            
            return None
        except:
            return None
    
    def pull_file(self, device_path: str, local_path: str) -> bool:
        """Pull file from device."""
        try:
            result = self._run_adb(['pull', device_path, local_path])
            return result.returncode == 0
        except:
            return False
    
    def push_file(self, local_path: str, device_path: str) -> bool:
        """Push file to device."""
        try:
            result = self._run_adb(['push', local_path, device_path])
            return result.returncode == 0
        except:
            return False
    
    def record_screen(self, output_path: str, duration: int = 60) -> bool:
        """Record screen video."""
        try:
            device_path = '/sdcard/recording.mp4'
            
            # Start recording (runs in background)
            self.shell(f'screenrecord --time-limit {duration} {device_path} &')
            
            # Wait for recording
            time.sleep(duration + 2)
            
            # Pull video
            return self.pull_file(device_path, output_path)
            
        except:
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        try:
            # CPU usage
            cpu_output = self.shell('top -n 1 -d 1')
            stats['cpu'] = cpu_output[:200]  # First 200 chars
            
            # Memory info
            mem_output = self.shell('dumpsys meminfo -s')
            stats['memory'] = mem_output[:200]
            
            # FPS (if gfxinfo available)
            fps_output = self.shell('dumpsys gfxinfo')
            if 'Total frames rendered' in fps_output:
                stats['fps_available'] = True
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
        
        return stats