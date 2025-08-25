"""Executable game launcher implementation"""

import asyncio
import subprocess
import os
import time
from typing import Optional, Dict, Any, List
import psutil
import structlog
from pathlib import Path

from nexus.launchers.base import GameLauncher, LauncherType

logger = structlog.get_logger()


class ExecutableLauncher(GameLauncher):
    """Launch games directly from executable files"""
    
    def __init__(self, game_name: str, executable_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(game_name, config)
        self.executable_path = Path(executable_path)
        self.working_directory = config.get("working_directory") if config else None
        self.arguments = config.get("arguments", []) if config else []
        self.environment = config.get("environment", {}) if config else {}
        
        if not self.executable_path.exists():
            logger.warning(f"Executable not found: {self.executable_path}")
    
    async def launch(self) -> bool:
        """Launch the game executable"""
        if not self.executable_path.exists():
            logger.error(f"Executable not found: {self.executable_path}")
            return False
        
        try:
            # Prepare launch environment
            env = os.environ.copy()
            env.update(self.environment)
            
            # Prepare command
            cmd = [str(self.executable_path)]
            if self.arguments:
                if isinstance(self.arguments, str):
                    cmd.extend(self.arguments.split())
                else:
                    cmd.extend(self.arguments)
            
            # Determine working directory
            cwd = self.working_directory or self.executable_path.parent
            
            logger.info(f"Launching {self.game_name}: {' '.join(cmd)}")
            logger.debug(f"Working directory: {cwd}")
            
            # Launch process
            if os.name == 'nt':  # Windows
                # Use CREATE_NEW_PROCESS_GROUP to detach from parent
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Linux/Mac
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid  # Create new session
                )
            
            self.pid = self.process.pid
            
            # Wait a bit to check if process started successfully
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is None:
                self.is_running = True
                logger.info(f"Game {self.game_name} launched successfully (PID: {self.pid})")
                
                # Wait for window to appear
                window_found = await self._wait_for_window()
                if not window_found:
                    logger.warning("Game window not detected, but process is running")
                
                return True
            else:
                logger.error(f"Game {self.game_name} exited immediately with code: {self.process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")
            return False
    
    async def terminate(self) -> bool:
        """Terminate the game process"""
        try:
            if self.process and self.process.poll() is None:
                # Try graceful termination first
                self.process.terminate()
                
                # Wait for termination
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.process.kill()
                    self.process.wait(timeout=5)
                
                self.process = None
                self.pid = None
                self.is_running = False
                logger.info(f"Game {self.game_name} terminated")
                return True
            
            # Try using psutil if we have PID
            if self.pid:
                try:
                    process = psutil.Process(self.pid)
                    
                    # Terminate child processes first
                    children = process.children(recursive=True)
                    for child in children:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            logger.debug(f"Child process already terminated")
                    
                    # Terminate main process
                    process.terminate()
                    
                    # Wait for termination
                    gone, alive = psutil.wait_procs([process] + children, timeout=10)
                    
                    # Force kill if still alive
                    for p in alive:
                        p.kill()
                    
                    self.pid = None
                    self.is_running = False
                    logger.info(f"Game {self.game_name} terminated via psutil")
                    return True
                    
                except psutil.NoSuchProcess:
                    self.pid = None
                    self.is_running = False
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to terminate game: {e}")
            return False
    
    def is_game_running(self) -> bool:
        """Check if game is running"""
        # Check subprocess first
        if self.process and self.process.poll() is None:
            return True
        
        # Check by PID
        if self.pid:
            try:
                process = psutil.Process(self.pid)
                if process.is_running():
                    return True
                else:
                    self.pid = None
            except psutil.NoSuchProcess:
                self.pid = None
        
        # Search by executable name
        exe_name = self.executable_path.name.lower()
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                if proc.info['name'] and exe_name in proc.info['name'].lower():
                    self.pid = proc.info['pid']
                    return True
                
                if proc.info.get('exe') and self.executable_path.samefile(proc.info['exe']):
                    self.pid = proc.info['pid']
                    return True
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        
        return False
    
    def get_window_info(self) -> Optional[Dict[str, Any]]:
        """Get game window information"""
        if not self.is_game_running():
            return None
        
        try:
            if os.name == 'nt':  # Windows
                import win32gui
                import win32process
                
                def callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if pid == self.pid:
                            rect = win32gui.GetWindowRect(hwnd)
                            windows.append({
                                "hwnd": hwnd,
                                "title": win32gui.GetWindowText(hwnd),
                                "x": rect[0],
                                "y": rect[1],
                                "width": rect[2] - rect[0],
                                "height": rect[3] - rect[1],
                                "is_foreground": hwnd == win32gui.GetForegroundWindow()
                            })
                
                windows = []
                win32gui.EnumWindows(callback, windows)
                
                if windows:
                    # Prefer foreground window
                    for window in windows:
                        if window["is_foreground"]:
                            return window
                    
                    # Otherwise return largest window
                    return max(windows, key=lambda w: w["width"] * w["height"])
            
            else:  # Linux/Mac
                # Use xwininfo or similar tools
                try:
                    # Get window ID for PID
                    result = subprocess.run(
                        ["xdotool", "search", "--pid", str(self.pid)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0 and result.stdout:
                        window_id = result.stdout.strip().split()[0]
                        
                        # Get window info
                        result = subprocess.run(
                            ["xwininfo", "-id", window_id],
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0:
                            # Parse xwininfo output
                            lines = result.stdout.split('\n')
                            info = {"window_id": window_id}
                            
                            for line in lines:
                                if "Width:" in line:
                                    info["width"] = int(line.split()[1])
                                elif "Height:" in line:
                                    info["height"] = int(line.split()[1])
                                elif "Absolute upper-left X:" in line:
                                    info["x"] = int(line.split()[-1])
                                elif "Absolute upper-left Y:" in line:
                                    info["y"] = int(line.split()[-1])
                            
                            return info
                except Exception as e:
                    logger.debug(f"Failed to get window info via xtools: {e}")
            
            # Fallback
            return {
                "pid": self.pid,
                "title": self.game_name,
                "executable": str(self.executable_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get window info: {e}")
            return None
    
    async def _wait_for_window(self, timeout: int = 10) -> bool:
        """Wait for game window to appear"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            window_info = self.get_window_info()
            if window_info and window_info.get("width", 0) > 0:
                logger.info(f"Game window detected: {window_info}")
                return True
            
            await asyncio.sleep(0.5)
        
        return False
    
    def focus_window(self) -> bool:
        """Bring game window to foreground"""
        window_info = self.get_window_info()
        if not window_info:
            return False
        
        try:
            if os.name == 'nt' and 'hwnd' in window_info:
                import win32gui
                win32gui.SetForegroundWindow(window_info['hwnd'])
                return True
            
            elif 'window_id' in window_info:
                # Linux - use xdotool
                subprocess.run(["xdotool", "windowactivate", window_info['window_id']])
                return True
                
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
        
        return False