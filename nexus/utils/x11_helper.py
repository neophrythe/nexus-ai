"""X11 Helper for cross-platform compatibility"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class X11Manager:
    """Manages X11 display configuration across different platforms"""
    
    def __init__(self):
        self.platform = self.detect_platform()
        self.display = None
        self.x11_available = False
        self.xauthority_path = None
        self.initialize()
    
    def detect_platform(self) -> str:
        """Detect the current platform"""
        if sys.platform == "win32":
            return "windows"
        elif sys.platform == "darwin":
            return "macos"
        else:
            # Check if we're in WSL
            try:
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        # Check WSL version
                        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
                        kernel = result.stdout.strip().lower()
                        if 'microsoft' in kernel:
                            return "wsl2" if any(v in kernel for v in ['wsl2', '5.']) else "wsl1"
            except:
                pass
            return "linux"
    
    def initialize(self):
        """Initialize X11 configuration based on platform"""
        if self.platform == "windows":
            self._setup_windows()
        elif self.platform == "wsl2":
            self._setup_wsl2()
        elif self.platform == "wsl1":
            self._setup_wsl1()
        elif self.platform == "linux":
            self._setup_linux()
        elif self.platform == "macos":
            self._setup_macos()
    
    def _setup_windows(self):
        """Setup for native Windows"""
        # Windows doesn't use X11
        self.x11_available = False
        logger.info("Running on Windows - X11 not required")
    
    def _setup_wsl2(self):
        """Setup for WSL2"""
        # Try to detect X server on Windows host
        self.display = self._detect_wsl2_display()
        
        # Create .Xauthority if it doesn't exist
        self._ensure_xauthority()
        
        if self.display:
            os.environ['DISPLAY'] = self.display
            self.x11_available = self._test_x11_connection()
            
        if self.x11_available:
            logger.info(f"WSL2 X11 configured: DISPLAY={self.display}")
        else:
            logger.warning("WSL2 detected but X11 server not available - GUI features limited")
    
    def _setup_wsl1(self):
        """Setup for WSL1"""
        # WSL1 typically uses localhost:0
        self.display = "localhost:0"
        self._ensure_xauthority()
        
        os.environ['DISPLAY'] = self.display
        self.x11_available = self._test_x11_connection()
        
        if self.x11_available:
            logger.info(f"WSL1 X11 configured: DISPLAY={self.display}")
        else:
            logger.warning("WSL1 detected but X11 server not available")
    
    def _setup_linux(self):
        """Setup for native Linux"""
        # Check if DISPLAY is already set
        self.display = os.environ.get('DISPLAY', ':0')
        
        # Ensure .Xauthority exists
        self._ensure_xauthority()
        
        self.x11_available = self._test_x11_connection()
        
        if self.x11_available:
            logger.info(f"Linux X11 configured: DISPLAY={self.display}")
        else:
            # Check if we're in a headless environment
            if not os.environ.get('DISPLAY'):
                logger.info("Running in headless Linux environment")
            else:
                logger.warning("X11 display set but connection failed")
    
    def _setup_macos(self):
        """Setup for macOS"""
        # macOS uses different display system
        self.x11_available = False
        logger.info("Running on macOS - using native display system")
    
    def _detect_wsl2_display(self) -> Optional[str]:
        """Detect the display for WSL2"""
        # Try multiple methods to find the Windows host IP
        
        # Method 1: Check /etc/resolv.conf for nameserver
        try:
            with open("/etc/resolv.conf", "r") as f:
                for line in f:
                    if "nameserver" in line:
                        ip = line.split()[1]
                        if ip and ip != "127.0.0.1":
                            return f"{ip}:0"
        except:
            pass
        
        # Method 2: Check if DISPLAY is already set
        if os.environ.get('DISPLAY'):
            return os.environ['DISPLAY']
        
        # Method 3: Try common WSL2 host IPs
        for ip in ["172.0.0.1", "192.168.0.1"]:
            if self._test_x11_connection(f"{ip}:0"):
                return f"{ip}:0"
        
        return None
    
    def _ensure_xauthority(self):
        """Ensure .Xauthority file exists"""
        home = Path.home()
        xauth_path = home / ".Xauthority"
        
        if not xauth_path.exists():
            try:
                # Create empty .Xauthority file
                xauth_path.touch(mode=0o600)
                self.xauthority_path = str(xauth_path)
                logger.debug(f"Created .Xauthority at {xauth_path}")
                
                # Try to generate a basic entry
                if self.display:
                    try:
                        subprocess.run(
                            ["xauth", "add", self.display, ".", "0" * 32],
                            capture_output=True,
                            timeout=1
                        )
                    except:
                        # xauth might not be installed
                        pass
            except Exception as e:
                logger.debug(f"Could not create .Xauthority: {e}")
        else:
            self.xauthority_path = str(xauth_path)
    
    def _test_x11_connection(self, display: Optional[str] = None) -> bool:
        """Test if X11 connection is available"""
        test_display = display or self.display
        
        if not test_display:
            return False
        
        # Set DISPLAY temporarily for test
        old_display = os.environ.get('DISPLAY')
        os.environ['DISPLAY'] = test_display
        
        try:
            # Try to import and test Xlib
            try:
                from Xlib import display as xdisplay
                from Xlib.error import DisplayConnectionError
                
                try:
                    d = xdisplay.Display(test_display)
                    d.close()
                    return True
                except (DisplayConnectionError, Exception):
                    return False
            except ImportError:
                # Xlib not installed, try alternative test
                try:
                    result = subprocess.run(
                        ["xset", "q"],
                        capture_output=True,
                        timeout=1,
                        env={**os.environ, 'DISPLAY': test_display}
                    )
                    return result.returncode == 0
                except:
                    return False
        except Exception:
            return False
        finally:
            # Restore original DISPLAY
            if old_display:
                os.environ['DISPLAY'] = old_display
            elif 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']
    
    def get_display_info(self) -> dict:
        """Get display configuration information"""
        return {
            "platform": self.platform,
            "x11_available": self.x11_available,
            "display": self.display,
            "xauthority": self.xauthority_path
        }
    
    def configure_environment(self):
        """Configure environment variables for X11"""
        if self.display:
            os.environ['DISPLAY'] = self.display
        
        if self.xauthority_path:
            os.environ['XAUTHORITY'] = self.xauthority_path
        
        # For WSL2, also set these for better compatibility
        if self.platform == "wsl2" and self.x11_available:
            # Helps with scaling issues
            os.environ.setdefault('GDK_SCALE', '1')
            os.environ.setdefault('QT_SCALE_FACTOR', '1')


# Global instance
_x11_manager = None


def get_x11_manager() -> X11Manager:
    """Get or create the global X11 manager"""
    global _x11_manager
    if _x11_manager is None:
        _x11_manager = X11Manager()
    return _x11_manager


def setup_x11_environment():
    """Setup X11 environment for the current platform"""
    manager = get_x11_manager()
    manager.configure_environment()
    return manager.x11_available


def is_gui_available() -> bool:
    """Check if GUI operations are available"""
    manager = get_x11_manager()
    
    if manager.platform == "windows":
        return True  # Windows always has GUI
    elif manager.platform == "macos":
        return True  # macOS always has GUI
    else:
        return manager.x11_available