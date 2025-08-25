"""Enhanced Game Launcher Implementations - SerpentAI Compatible with Modern Features

Provides comprehensive game launching for all platforms and game stores.
"""

import os
import sys
import subprocess
import webbrowser
import time
import json
import shlex
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog
import psutil
import platform

# Platform-specific imports
if sys.platform == "win32":
    import winreg
    import win32gui
    import win32con
    import win32process
    import ctypes
    from ctypes import wintypes
elif sys.platform.startswith("linux"):
    import pwd
    import grp

logger = structlog.get_logger()


class LauncherType(Enum):
    """Supported launcher types"""
    STEAM = "steam"
    EPIC = "epic"
    GOG = "gog"
    ORIGIN = "origin"
    UPLAY = "uplay"
    BATTLENET = "battlenet"
    XBOX = "xbox"
    EXECUTABLE = "executable"
    WEB_BROWSER = "web_browser"
    ANDROID = "android"
    CUSTOM = "custom"


@dataclass
class LaunchConfig:
    """Game launch configuration"""
    launcher_type: LauncherType
    game_name: Optional[str] = None
    game_path: Optional[str] = None
    app_id: Optional[str] = None
    url: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[str] = None
    
    # Window management
    wait_for_window: bool = True
    window_name: Optional[str] = None
    window_timeout: float = 30.0
    focus_window: bool = True
    
    # Browser specific
    browser: str = "chrome"
    fullscreen: bool = False
    incognito: bool = False
    browser_profile: Optional[str] = None
    disable_gpu: bool = False
    
    # Process management
    startup_delay: float = 5.0
    kill_on_exit: bool = True
    monitor_process: bool = True
    
    # Performance
    priority: str = "normal"  # low, normal, high, realtime
    cpu_affinity: Optional[List[int]] = None
    
    # Debugging
    debug_mode: bool = False
    log_output: bool = False


class GameLauncherException(Exception):
    """Game launcher exception"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class GameLauncher(ABC):
    """Base game launcher - SerpentAI compatible with enhancements"""
    
    def __init__(self, config: LaunchConfig):
        self.config = config
        self.process = None
        self.pid = None
        self.window_handle = None
        self.is_running = False
        self.launch_time = None
        
    @abstractmethod
    def launch(self, **kwargs) -> bool:
        """Launch the game - SerpentAI compatible"""
        self.launch_time = time.time()
        self.is_running = True
        logger.info(f"Game launch initiated")
        return True
        
    def close(self) -> bool:
        """Close the game"""
        if not self.process:
            return False
            
        try:
            # Try graceful termination first
            self.process.terminate()
            time.sleep(2)
            
            # Force kill if still running
            if self.process.poll() is None:
                self.process.kill()
                
            # Kill child processes
            if self.pid:
                self._kill_process_tree(self.pid)
                
            self.is_running = False
            logger.info(f"Game closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close game: {e}")
            return False
            
    def _kill_process_tree(self, pid: int):
        """Kill process and all children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    logger.debug("Process already terminated")
                    
            # Wait for termination
            gone, alive = psutil.wait_procs(children, timeout=3)
            
            # Force kill remaining
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    logger.debug("Process already terminated")
                    
        except psutil.NoSuchProcess:
            logger.debug("Parent process not found")
            
    def wait_for_window(self, timeout: Optional[float] = None) -> bool:
        """Wait for game window to appear"""
        timeout = timeout or self.config.window_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            window = self._find_window(self.config.window_name)
            
            if window:
                self.window_handle = window
                
                if self.config.focus_window:
                    self._focus_window(window)
                    
                logger.info(f"Game window found: {self.config.window_name}")
                return True
                
            time.sleep(0.5)
            
        logger.warning(f"Game window not found after {timeout}s")
        return False
        
    def _find_window(self, title: str):
        """Find window by title"""
        if sys.platform == "win32":
            return self._find_window_win32(title)
        elif sys.platform.startswith("linux"):
            return self._find_window_linux(title)
        elif sys.platform == "darwin":
            return self._find_window_macos(title)
        return None
        
    def _find_window_win32(self, title: str):
        """Find window on Windows"""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if title.lower() in window_title.lower():
                    windows.append(hwnd)
            return True
            
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        return windows[0] if windows else None
        
    def _find_window_linux(self, title: str):
        """Find window on Linux"""
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", title],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and result.stdout:
                window_ids = result.stdout.strip().split('\n')
                return int(window_ids[0]) if window_ids else None
                
        except FileNotFoundError:
            logger.warning("xdotool not found")
            
        return None
        
    def _find_window_macos(self, title: str):
        """Find window on macOS"""
        # Would use AppleScript or Quartz
        return None
        
    def _focus_window(self, window_handle):
        """Bring window to foreground"""
        if sys.platform == "win32":
            try:
                win32gui.SetForegroundWindow(window_handle)
                win32gui.ShowWindow(window_handle, win32con.SW_RESTORE)
            except Exception as e:
                logger.warning(f"Failed to focus window: {e}")
        elif sys.platform.startswith("linux"):
            try:
                subprocess.run(["xdotool", "windowactivate", str(window_handle)])
            except Exception as e:
                logger.warning(f"Failed to focus window: {e}")
                
    def set_process_priority(self):
        """Set process priority"""
        if not self.process or not self.pid:
            return
            
        try:
            p = psutil.Process(self.pid)
            
            # Set priority
            priority_map = {
                "low": psutil.IDLE_PRIORITY_CLASS if sys.platform == "win32" else 19,
                "normal": psutil.NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 0,
                "high": psutil.HIGH_PRIORITY_CLASS if sys.platform == "win32" else -10,
                "realtime": psutil.REALTIME_PRIORITY_CLASS if sys.platform == "win32" else -20
            }
            
            if self.config.priority in priority_map:
                if sys.platform == "win32":
                    p.nice(priority_map[self.config.priority])
                else:
                    p.nice(priority_map[self.config.priority])
                    
            # Set CPU affinity
            if self.config.cpu_affinity:
                p.cpu_affinity(self.config.cpu_affinity)
                
        except Exception as e:
            logger.warning(f"Failed to set process priority: {e}")
            
    def get_process_info(self) -> Dict:
        """Get process information"""
        if not self.pid:
            return {}
            
        try:
            p = psutil.Process(self.pid)
            
            return {
                'pid': self.pid,
                'name': p.name(),
                'status': p.status(),
                'cpu_percent': p.cpu_percent(),
                'memory_info': p.memory_info()._asdict(),
                'create_time': p.create_time(),
                'num_threads': p.num_threads()
            }
        except psutil.NoSuchProcess:
            return {}


class SteamGameLauncher(GameLauncher):
    """Steam game launcher - SerpentAI compatible with enhancements"""
    
    def __init__(self, config: LaunchConfig):
        super().__init__(config)
        self.steam_path = self._find_steam()
        self.steam_apps = self._find_steam_apps()
        
    def _find_steam(self) -> Optional[str]:
        """Find Steam installation"""
        if sys.platform == "win32":
            # Check registry
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam")
                steam_path = winreg.QueryValueEx(key, "SteamPath")[0]
                winreg.CloseKey(key)
                return steam_path
            except Exception as e:
                logger.warning(f"Failed to focus window: {e}")
                
            # Check common paths
            paths = [
                r"C:\Program Files (x86)\Steam",
                r"C:\Program Files\Steam"
            ]
            
        elif sys.platform.startswith("linux"):
            paths = [
                os.path.expanduser("~/.steam/steam"),
                os.path.expanduser("~/.local/share/Steam"),
                "/usr/games/steam",
                "/usr/local/games/steam"
            ]
            
        elif sys.platform == "darwin":
            paths = [
                os.path.expanduser("~/Library/Application Support/Steam"),
                "/Applications/Steam.app"
            ]
        else:
            paths = []
            
        for path in paths:
            if os.path.exists(path):
                return path
                
        return None
        
    def _find_steam_apps(self) -> Dict[str, str]:
        """Find installed Steam games"""
        apps = {}
        
        if not self.steam_path:
            return apps
            
        # Parse libraryfolders.vdf
        library_file = Path(self.steam_path) / "steamapps" / "libraryfolders.vdf"
        
        if library_file.exists():
            try:
                # Simple VDF parser
                with open(library_file, 'r') as f:
                    content = f.read()
                    
                # Extract library paths
                import re
                paths = re.findall(r'"path"\s+"([^"]+)"', content)
                
                for library_path in paths:
                    steamapps = Path(library_path) / "steamapps"
                    
                    # Find app manifests
                    for manifest in steamapps.glob("appmanifest_*.acf"):
                        app_id = manifest.stem.replace("appmanifest_", "")
                        
                        # Parse manifest for game name
                        with open(manifest, 'r') as f:
                            manifest_content = f.read()
                            name_match = re.search(r'"name"\s+"([^"]+)"', manifest_content)
                            
                            if name_match:
                                apps[app_id] = name_match.group(1)
                                
            except Exception as e:
                logger.warning(f"Failed to parse Steam library: {e}")
                
        return apps
        
    def launch(self, **kwargs) -> bool:
        """Launch Steam game - SerpentAI compatible"""
        app_id = kwargs.get("app_id") or self.config.app_id
        
        if not app_id:
            raise GameLauncherException("No Steam app ID provided")
            
        # Build Steam URL
        protocol_string = f"steam://run/{app_id}"
        
        # Add launch options
        if self.config.arguments:
            args = " ".join(self.config.arguments)
            protocol_string += f"//{args}"
            
        logger.info(f"Launching Steam game: {app_id}")
        
        # Launch based on platform
        if sys.platform == "win32":
            # Use Steam executable directly for better control
            if self.steam_path:
                steam_exe = Path(self.steam_path) / "steam.exe"
                cmd = [str(steam_exe), "-applaunch", app_id] + self.config.arguments
                
                self.process = subprocess.Popen(
                    cmd,
                    env={**os.environ, **self.config.environment},
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                self.pid = self.process.pid
            else:
                # Fallback to protocol
                webbrowser.open(protocol_string)
                
        elif sys.platform.startswith("linux"):
            # Try xdg-open first
            try:
                self.process = subprocess.Popen(
                    ["xdg-open", protocol_string],
                    env={**os.environ, **self.config.environment}
                )
                self.pid = self.process.pid
            except FileNotFoundError:
                # Fallback to steam command
                self.process = subprocess.Popen(
                    ["steam", f"steam://run/{app_id}"],
                    env={**os.environ, **self.config.environment}
                )
                self.pid = self.process.pid
                
        elif sys.platform == "darwin":
            self.process = subprocess.Popen(
                ["open", protocol_string],
                env={**os.environ, **self.config.environment}
            )
            self.pid = self.process.pid
            
        # Wait for startup
        time.sleep(self.config.startup_delay)
        
        # Set process priority
        self.set_process_priority()
        
        # Wait for window if configured
        if self.config.wait_for_window:
            self.is_running = self.wait_for_window()
        else:
            self.is_running = True
            
        self.launch_time = time.time()
        return self.is_running
        
    def validate_game(self, app_id: str) -> bool:
        """Validate if game is installed"""
        return app_id in self.steam_apps


class EpicGamesLauncher(GameLauncher):
    """Epic Games launcher"""
    
    def __init__(self, config: LaunchConfig):
        super().__init__(config)
        self.epic_path = self._find_epic()
        
    def _find_epic(self) -> Optional[str]:
        """Find Epic Games Launcher"""
        if sys.platform == "win32":
            paths = [
                r"C:\Program Files (x86)\Epic Games\Launcher",
                r"C:\Program Files\Epic Games\Launcher"
            ]
            
            for path in paths:
                launcher = Path(path) / "Portal" / "Binaries" / "Win64" / "EpicGamesLauncher.exe"
                if launcher.exists():
                    return str(launcher)
                    
        return None
        
    def launch(self, **kwargs) -> bool:
        """Launch Epic Games game"""
        app_id = kwargs.get("app_id") or self.config.app_id
        
        if not app_id:
            raise GameLauncherException("No Epic Games app ID provided")
            
        # Epic Games URL scheme
        epic_url = f"com.epicgames.launcher://apps/{app_id}?action=launch"
        
        logger.info(f"Launching Epic Games: {app_id}")
        
        if sys.platform == "win32":
            if self.epic_path:
                # Launch Epic Games Launcher with URL
                cmd = [self.epic_path, epic_url]
                
                self.process = subprocess.Popen(
                    cmd,
                    env={**os.environ, **self.config.environment},
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                self.pid = self.process.pid
            else:
                # Try URL protocol
                webbrowser.open(epic_url)
                
        # Wait for startup
        time.sleep(self.config.startup_delay)
        
        # Wait for window
        if self.config.wait_for_window:
            self.is_running = self.wait_for_window()
        else:
            self.is_running = True
            
        self.launch_time = time.time()
        return self.is_running


class ExecutableGameLauncher(GameLauncher):
    """Executable game launcher - SerpentAI compatible with enhancements"""
    
    def launch(self, **kwargs) -> bool:
        """Launch executable game - SerpentAI compatible"""
        executable_path = kwargs.get("executable_path") or self.config.game_path
        
        if not executable_path:
            raise GameLauncherException("No executable path provided")
            
        executable_path = Path(executable_path)
        
        if not executable_path.exists():
            raise GameLauncherException(f"Executable not found: {executable_path}")
            
        # Build command
        if sys.platform == "win32":
            # Handle spaces in path
            cmd = [str(executable_path)]
        else:
            cmd = shlex.split(str(executable_path))
            
        cmd.extend(self.config.arguments)
        
        # Set working directory
        cwd = self.config.working_directory or executable_path.parent
        
        logger.info(f"Launching executable: {executable_path}")
        
        # Launch process
        if sys.platform == "win32":
            # Windows-specific flags
            self.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env={**os.environ, **self.config.environment},
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.PIPE if self.config.log_output else None,
                stderr=subprocess.PIPE if self.config.log_output else None
            )
        else:
            # Unix-like systems
            self.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env={**os.environ, **self.config.environment},
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE if self.config.log_output else None,
                stderr=subprocess.PIPE if self.config.log_output else None
            )
            
        self.pid = self.process.pid
        
        # Log output if configured
        if self.config.log_output:
            self._start_output_logging()
            
        # Wait for startup
        time.sleep(self.config.startup_delay)
        
        # Set process priority
        self.set_process_priority()
        
        # Wait for window
        if self.config.wait_for_window:
            self.is_running = self.wait_for_window()
        else:
            self.is_running = True
            
        self.launch_time = time.time()
        return self.is_running
        
    def _start_output_logging(self):
        """Start logging process output"""
        import threading
        
        def log_output(pipe, prefix):
            for line in iter(pipe.readline, b''):
                logger.debug(f"{prefix}: {line.decode().strip()}")
                
        if self.process.stdout:
            stdout_thread = threading.Thread(
                target=log_output,
                args=(self.process.stdout, "STDOUT"),
                daemon=True
            )
            stdout_thread.start()
            
        if self.process.stderr:
            stderr_thread = threading.Thread(
                target=log_output,
                args=(self.process.stderr, "STDERR"),
                daemon=True
            )
            stderr_thread.start()


class WebBrowserGameLauncher(GameLauncher):
    """Web browser game launcher - SerpentAI compatible with enhancements"""
    
    def __init__(self, config: LaunchConfig):
        super().__init__(config)
        self.browser_path = self._find_browser()
        
    def _find_browser(self) -> Optional[str]:
        """Find browser executable"""
        browser = self.config.browser.lower()
        
        if sys.platform == "win32":
            browsers = {
                "chrome": [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                ],
                "firefox": [
                    r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
                ],
                "edge": [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                ]
            }
        elif sys.platform.startswith("linux"):
            browsers = {
                "chrome": [
                    "/usr/bin/google-chrome",
                    "/usr/bin/google-chrome-stable",
                    "/usr/bin/chromium",
                    "/usr/bin/chromium-browser"
                ],
                "firefox": [
                    "/usr/bin/firefox"
                ],
                "edge": [
                    "/usr/bin/microsoft-edge"
                ]
            }
        elif sys.platform == "darwin":
            browsers = {
                "chrome": [
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                ],
                "firefox": [
                    "/Applications/Firefox.app/Contents/MacOS/firefox"
                ],
                "safari": [
                    "/Applications/Safari.app/Contents/MacOS/Safari"
                ]
            }
        else:
            browsers = {}
            
        paths = browsers.get(browser, [])
        
        for path in paths:
            if os.path.exists(path):
                return path
                
        return None
        
    def launch(self, **kwargs) -> bool:
        """Launch web browser game - SerpentAI compatible"""
        url = kwargs.get("url") or self.config.url
        
        if not url:
            raise GameLauncherException("No URL provided")
            
        browser_name = kwargs.get("browser", self.config.browser)
        
        logger.info(f"Launching browser game: {url}")
        
        # Build browser arguments
        args = []
        
        if self.browser_path:
            args.append(self.browser_path)
            
            # Browser-specific arguments
            if "chrome" in self.config.browser.lower() or "chromium" in self.config.browser.lower():
                # Chrome/Chromium flags
                if self.config.fullscreen:
                    args.append("--start-fullscreen")
                else:
                    args.append(f"--window-size=1920,1080")
                    
                if self.config.incognito:
                    args.append("--incognito")
                    
                if self.config.browser_profile:
                    args.append(f"--user-data-dir={self.config.browser_profile}")
                    
                if self.config.disable_gpu:
                    args.append("--disable-gpu")
                    
                # Gaming optimizations
                args.extend([
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-backgrounding-occluded-windows",
                    "--high-dpi-support=1",
                    "--force-device-scale-factor=1"
                ])
                
            elif "firefox" in self.config.browser.lower():
                # Firefox flags
                if self.config.incognito:
                    args.append("--private-window")
                    
                if self.config.browser_profile:
                    args.extend(["-profile", self.config.browser_profile])
                    
            # Add custom arguments
            args.extend(self.config.arguments)
            
            # Add URL
            args.append(url)
            
            # Launch browser
            self.process = subprocess.Popen(
                args,
                env={**os.environ, **self.config.environment}
            )
            self.pid = self.process.pid
            
        else:
            # Fallback to webbrowser module
            try:
                controller = webbrowser.get(browser_name)
                controller.open_new(url)
            except:
                webbrowser.open(url)
                
        # Wait for startup
        time.sleep(self.config.startup_delay)
        
        # Wait for window
        if self.config.wait_for_window:
            self.is_running = self.wait_for_window()
        else:
            self.is_running = True
            
        self.launch_time = time.time()
        return self.is_running


class GameLauncherFactory:
    """Factory for creating game launchers"""
    
    LAUNCHER_CLASSES = {
        LauncherType.STEAM: SteamGameLauncher,
        LauncherType.EPIC: EpicGamesLauncher,
        LauncherType.EXECUTABLE: ExecutableGameLauncher,
        LauncherType.WEB_BROWSER: WebBrowserGameLauncher,
    }
    
    @classmethod
    def create_launcher(cls, config: LaunchConfig) -> GameLauncher:
        """Create appropriate launcher"""
        launcher_class = cls.LAUNCHER_CLASSES.get(config.launcher_type)
        
        if not launcher_class:
            raise GameLauncherException(f"Unsupported launcher type: {config.launcher_type}")
            
        return launcher_class(config)
        
    @classmethod
    def launch_game(cls, config: LaunchConfig) -> GameLauncher:
        """Create launcher and launch game"""
        launcher = cls.create_launcher(config)
        
        success = launcher.launch()
        
        if not success:
            raise GameLauncherException(f"Failed to launch game")
            
        return launcher


# SerpentAI compatibility functions
def launch_steam_game(app_id: str, **kwargs) -> GameLauncher:
    """Launch Steam game - SerpentAI compatible"""
    config = LaunchConfig(
        launcher_type=LauncherType.STEAM,
        app_id=app_id,
        **kwargs
    )
    return GameLauncherFactory.launch_game(config)


def launch_executable_game(executable_path: str, **kwargs) -> GameLauncher:
    """Launch executable game - SerpentAI compatible"""
    config = LaunchConfig(
        launcher_type=LauncherType.EXECUTABLE,
        game_path=executable_path,
        **kwargs
    )
    return GameLauncherFactory.launch_game(config)


def launch_web_browser_game(url: str, browser: str = "chrome", **kwargs) -> GameLauncher:
    """Launch web browser game - SerpentAI compatible"""
    config = LaunchConfig(
        launcher_type=LauncherType.WEB_BROWSER,
        url=url,
        browser=browser,
        **kwargs
    )
    return GameLauncherFactory.launch_game(config)