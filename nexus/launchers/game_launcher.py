"""Game Launcher System for Nexus Framework"""

import subprocess
import webbrowser
import shlex
import time
import platform
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog
import psutil

logger = structlog.get_logger()


class LauncherType(Enum):
    """Game launcher types"""
    EXECUTABLE = "executable"
    STEAM = "steam"
    EPIC = "epic"
    ORIGIN = "origin"
    UPLAY = "uplay"
    GOG = "gog"
    BATTLE_NET = "battle_net"
    WEB_BROWSER = "web_browser"
    CUSTOM = "custom"


@dataclass
class LaunchConfig:
    """Game launch configuration"""
    launcher_type: LauncherType
    game_path: Optional[str] = None
    app_id: Optional[str] = None
    url: Optional[str] = None
    arguments: List[str] = None
    environment: Dict[str, str] = None
    working_directory: Optional[str] = None
    wait_for_window: bool = True
    window_name: Optional[str] = None
    startup_delay: float = 5.0
    
    def __post_init__(self):
        if self.arguments is None:
            self.arguments = []
        if self.environment is None:
            self.environment = {}


class GameLauncherException(Exception):
    """Game launcher exception"""
    pass


class GameLauncher:
    """Base game launcher"""
    
    def __init__(self, config: LaunchConfig):
        self.config = config
        self.process = None
        self.window_info = None
        
    def launch(self) -> bool:
        """
        Launch the game
        
        Returns:
            True if successful
        """
        import subprocess
        import os
        
        try:
            # Build command
            cmd = [self.config.executable_path]
            if self.config.arguments:
                if isinstance(self.config.arguments, str):
                    cmd.extend(self.config.arguments.split())
                else:
                    cmd.extend(self.config.arguments)
            
            # Set working directory
            working_dir = self.config.working_directory or os.path.dirname(self.config.executable_path)
            
            # Launch process
            self.process = subprocess.Popen(
                cmd,
                cwd=working_dir,
                env=dict(os.environ, **self.config.environment) if self.config.environment else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Game launched with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")
            return False
    
    def is_running(self) -> bool:
        """
        Check if game is running
        
        Returns:
            True if running
        """
        if self.process:
            return self.process.poll() is None
        return False
    
    def terminate(self) -> bool:
        """
        Terminate the game
        
        Returns:
            True if successful
        """
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                return True
            except Exception as e:
                logger.error(f"Failed to terminate game: {e}")
                return False
        return True
    
    def kill(self) -> bool:
        """
        Force kill the game
        
        Returns:
            True if successful
        """
        if self.process:
            try:
                self.process.kill()
                return True
            except Exception as e:
                logger.error(f"Failed to kill game: {e}")
                return False
        return True
    
    def wait_for_window(self, timeout: float = 30) -> bool:
        """
        Wait for game window to appear
        
        Args:
            timeout: Maximum wait time
        
        Returns:
            True if window found
        """
        if not self.config.wait_for_window:
            return True
        
        if not self.config.window_name:
            logger.warning("No window name specified for waiting")
            return True
        
        from nexus.window.window_controller import WindowController
        window_controller = WindowController()
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            window = window_controller.locate_window(self.config.window_name)
            if window:
                self.window_info = window
                logger.info(f"Found game window: {window.title}")
                return True
            time.sleep(0.5)
        
        logger.warning(f"Game window not found after {timeout} seconds")
        return False


class ExecutableGameLauncher(GameLauncher):
    """Launcher for executable games"""
    
    def launch(self) -> bool:
        """Launch executable game"""
        if not self.config.game_path:
            raise GameLauncherException("game_path is required for executable launcher")
        
        game_path = Path(self.config.game_path)
        if not game_path.exists():
            raise GameLauncherException(f"Game executable not found: {game_path}")
        
        # Build command
        cmd = [str(game_path)] + self.config.arguments
        
        # Set up environment
        env = os.environ.copy()
        env.update(self.config.environment)
        
        # Set working directory
        cwd = self.config.working_directory or str(game_path.parent)
        
        try:
            logger.info(f"Launching executable: {game_path}")
            
            # Launch process
            self.process = subprocess.Popen(
                cmd,
                env=env,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for startup
            time.sleep(self.config.startup_delay)
            
            # Wait for window
            if self.config.wait_for_window:
                self.wait_for_window()
            
            logger.info(f"Game launched successfully (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")
            return False


class SteamGameLauncher(GameLauncher):
    """Launcher for Steam games"""
    
    def launch(self) -> bool:
        """Launch Steam game"""
        if not self.config.app_id:
            raise GameLauncherException("app_id is required for Steam launcher")
        
        # Build Steam URL
        steam_url = f"steam://run/{self.config.app_id}"
        
        # Add arguments if provided
        if self.config.arguments:
            args_str = " ".join(self.config.arguments)
            steam_url += f"//{args_str}/"
        
        try:
            logger.info(f"Launching Steam game: {self.config.app_id}")
            
            # Launch via Steam protocol
            if platform.system() == "Windows":
                webbrowser.open(steam_url)
            elif platform.system() == "Linux":
                subprocess.call(shlex.split(f"xdg-open '{steam_url}'"))
            elif platform.system() == "Darwin":
                subprocess.call(["open", steam_url])
            else:
                raise GameLauncherException(f"Unsupported platform: {platform.system()}")
            
            # Wait for startup
            time.sleep(self.config.startup_delay)
            
            # Find Steam process
            self._find_steam_process()
            
            # Wait for window
            if self.config.wait_for_window:
                self.wait_for_window()
            
            logger.info(f"Steam game launched successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Steam game: {e}")
            return False
    
    def _find_steam_process(self):
        """Find the Steam game process"""
        # This is a simplified version - in practice would need more robust detection
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if self.config.window_name and self.config.window_name.lower() in proc.info['name'].lower():
                    self.process = psutil.Process(proc.info['pid'])
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


class EpicGamesLauncher(GameLauncher):
    """Launcher for Epic Games"""
    
    def launch(self) -> bool:
        """Launch Epic Games game"""
        if not self.config.app_id:
            raise GameLauncherException("app_id is required for Epic Games launcher")
        
        # Epic Games launcher URL format
        epic_url = f"com.epicgames.launcher://apps/{self.config.app_id}?action=launch"
        
        try:
            logger.info(f"Launching Epic Games: {self.config.app_id}")
            
            if platform.system() == "Windows":
                webbrowser.open(epic_url)
            else:
                logger.warning("Epic Games launcher only supported on Windows")
                return False
            
            # Wait for startup
            time.sleep(self.config.startup_delay)
            
            # Wait for window
            if self.config.wait_for_window:
                self.wait_for_window()
            
            logger.info("Epic Games launched successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch Epic Games: {e}")
            return False


class WebBrowserGameLauncher(GameLauncher):
    """Launcher for web browser games"""
    
    def launch(self) -> bool:
        """Launch web browser game"""
        if not self.config.url:
            raise GameLauncherException("url is required for web browser launcher")
        
        try:
            logger.info(f"Launching web game: {self.config.url}")
            
            # Open in default browser
            webbrowser.open(self.config.url)
            
            # Wait for startup
            time.sleep(self.config.startup_delay)
            
            # For web games, window detection is browser-specific
            if self.config.wait_for_window and self.config.window_name:
                self.wait_for_window()
            
            logger.info("Web game launched successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch web game: {e}")
            return False


class GameLauncherFactory:
    """Factory for creating game launchers"""
    
    LAUNCHER_CLASSES = {
        LauncherType.EXECUTABLE: ExecutableGameLauncher,
        LauncherType.STEAM: SteamGameLauncher,
        LauncherType.EPIC: EpicGamesLauncher,
        LauncherType.WEB_BROWSER: WebBrowserGameLauncher,
    }
    
    @classmethod
    def create(cls, config: LaunchConfig) -> GameLauncher:
        """
        Create appropriate game launcher
        
        Args:
            config: Launch configuration
        
        Returns:
            Game launcher instance
        """
        launcher_class = cls.LAUNCHER_CLASSES.get(config.launcher_type)
        
        if not launcher_class:
            raise GameLauncherException(f"Unsupported launcher type: {config.launcher_type}")
        
        return launcher_class(config)
    
    @classmethod
    def launch_game(cls, config: LaunchConfig) -> GameLauncher:
        """
        Convenience method to create and launch game
        
        Args:
            config: Launch configuration
        
        Returns:
            Game launcher instance
        """
        launcher = cls.create(config)
        
        if launcher.launch():
            return launcher
        else:
            raise GameLauncherException("Failed to launch game")


# Convenience functions
def launch_executable(path: str, **kwargs) -> GameLauncher:
    """Launch executable game"""
    config = LaunchConfig(
        launcher_type=LauncherType.EXECUTABLE,
        game_path=path,
        **kwargs
    )
    return GameLauncherFactory.launch_game(config)


def launch_steam(app_id: str, **kwargs) -> GameLauncher:
    """Launch Steam game"""
    config = LaunchConfig(
        launcher_type=LauncherType.STEAM,
        app_id=app_id,
        **kwargs
    )
    return GameLauncherFactory.launch_game(config)


def launch_web(url: str, **kwargs) -> GameLauncher:
    """Launch web game"""
    config = LaunchConfig(
        launcher_type=LauncherType.WEB_BROWSER,
        url=url,
        **kwargs
    )
    return GameLauncherFactory.launch_game(config)