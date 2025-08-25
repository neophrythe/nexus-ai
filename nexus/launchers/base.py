from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
import asyncio
import psutil
import structlog

logger = structlog.get_logger()


class LauncherType(Enum):
    EXECUTABLE = "executable"
    STEAM = "steam"
    BROWSER = "browser"
    EPIC = "epic"
    CUSTOM = "custom"


class GameLauncher(ABC):
    """Base class for game launchers"""
    
    def __init__(self, game_name: str, config: Optional[Dict[str, Any]] = None):
        self.game_name = game_name
        self.config = config or {}
        self.process = None
        self.pid = None
        self.is_running = False
        
    @abstractmethod
    async def launch(self) -> bool:
        """Launch the game"""
        logger.info(f"Launching game: {self.game_name}")
        self.is_running = True
        return True
    
    @abstractmethod
    async def terminate(self) -> bool:
        """Terminate the game"""
        logger.info(f"Terminating game: {self.game_name}")
        if self.pid:
            try:
                proc = psutil.Process(self.pid)
                proc.terminate()
                proc.wait(timeout=5)
                self.is_running = False
                self.pid = None
                return True
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                self.is_running = False
                self.pid = None
        return False
    
    @abstractmethod
    def is_game_running(self) -> bool:
        """Check if game is running"""
        if self.pid:
            try:
                proc = psutil.Process(self.pid)
                return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                self.pid = None
        return False
    
    @abstractmethod
    def get_window_info(self) -> Optional[Dict[str, Any]]:
        """Get game window information"""
        if not self.is_game_running():
            return None
        
        # Basic window info - can be extended by subclasses
        return {
            'title': self.game_name,
            'pid': self.pid,
            'process_name': self.config.get('process_name', 'unknown'),
            'running': True
        }
    
    async def wait_for_game(self, timeout: int = 30) -> bool:
        """Wait for game to start"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.is_game_running():
                logger.info(f"Game {self.game_name} started successfully")
                return True
            await asyncio.sleep(1)
        
        logger.error(f"Game {self.game_name} failed to start within {timeout} seconds")
        return False
    
    def find_process_by_name(self, process_name: str) -> Optional[psutil.Process]:
        """Find process by name"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if process_name.lower() in proc.info['name'].lower():
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return None
    
    def get_process_info(self) -> Optional[Dict[str, Any]]:
        """Get process information"""
        if self.pid:
            try:
                proc = psutil.Process(self.pid)
                return {
                    "pid": self.pid,
                    "name": proc.name(),
                    "status": proc.status(),
                    "cpu_percent": proc.cpu_percent(),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "create_time": proc.create_time()
                }
            except psutil.NoSuchProcess:
                self.pid = None
        return None