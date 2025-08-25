"""Steam game launcher implementation"""

import asyncio
import subprocess
import os
import time
from typing import Optional, Dict, Any
import psutil
import structlog
from pathlib import Path

from nexus.launchers.base import GameLauncher

logger = structlog.get_logger()


class SteamLauncher(GameLauncher):
    """Launch games through Steam"""
    
    STEAM_PATHS = {
        "win32": [
            "C:\\Program Files (x86)\\Steam\\steam.exe",
            "C:\\Program Files\\Steam\\steam.exe",
            "D:\\Steam\\steam.exe",
            "E:\\Steam\\steam.exe"
        ],
        "linux": [
            os.path.expanduser("~/.steam/steam.sh"),
            "/usr/bin/steam",
            "/usr/games/steam"
        ],
        "darwin": [
            "/Applications/Steam.app/Contents/MacOS/steam_osx"
        ]
    }
    
    def __init__(self, game_name: str, app_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(game_name, config)
        self.app_id = app_id
        self.steam_path = self._find_steam()
        self.launch_options = config.get("launch_options", "") if config else ""
        
    def _find_steam(self) -> Optional[str]:
        """Find Steam installation"""
        import sys
        platform_paths = self.STEAM_PATHS.get(sys.platform, [])
        
        # Check configured path first
        if self.config.get("steam_path"):
            if os.path.exists(self.config["steam_path"]):
                return self.config["steam_path"]
        
        # Search default paths
        for path in platform_paths:
            if os.path.exists(path):
                logger.info(f"Found Steam at: {path}")
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(["which", "steam"], capture_output=True, text=True)
            if result.returncode == 0:
                steam_path = result.stdout.strip()
                if os.path.exists(steam_path):
                    return steam_path
        except:
            pass
        
        logger.warning("Steam not found")
        return None
    
    async def launch(self) -> bool:
        """Launch game through Steam"""
        if not self.steam_path:
            logger.error("Steam not found, cannot launch game")
            return False
        
        try:
            # Build Steam URL
            steam_url = f"steam://run/{self.app_id}"
            
            if self.launch_options:
                steam_url += f"//{self.launch_options}"
            
            logger.info(f"Launching {self.game_name} via Steam: {steam_url}")
            
            # Launch using Steam protocol
            if os.name == 'nt':  # Windows
                subprocess.Popen(['cmd', '/c', 'start', '', steam_url], shell=False)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.Popen(['xdg-open', steam_url])
            
            # Wait for game to start
            await asyncio.sleep(5)  # Give Steam time to launch
            
            # Find game process
            for i in range(30):  # Try for 30 seconds
                if self.is_game_running():
                    self.is_running = True
                    logger.info(f"Game {self.game_name} launched successfully")
                    return True
                await asyncio.sleep(1)
            
            logger.error(f"Game {self.game_name} failed to launch")
            return False
            
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")
            return False
    
    async def terminate(self) -> bool:
        """Terminate the game"""
        try:
            if self.pid:
                process = psutil.Process(self.pid)
                process.terminate()
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    # Force kill if needed
                    process.kill()
                
                self.pid = None
                self.is_running = False
                logger.info(f"Game {self.game_name} terminated")
                return True
            
            # Try to find and kill by name
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if self.game_name.lower() in proc.info['name'].lower():
                        proc.terminate()
                        self.is_running = False
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to terminate game: {e}")
            return False
    
    def is_game_running(self) -> bool:
        """Check if game is running"""
        # Check by PID if we have it
        if self.pid:
            try:
                process = psutil.Process(self.pid)
                return process.is_running()
            except psutil.NoSuchProcess:
                self.pid = None
        
        # Search by process name
        game_executables = self.config.get("executables", [self.game_name])
        if isinstance(game_executables, str):
            game_executables = [game_executables]
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_name = proc.info['name'].lower()
                
                # Check process name
                for exe in game_executables:
                    if exe.lower() in proc_name:
                        self.pid = proc.info['pid']
                        return True
                
                # Check command line for app ID
                if proc.info.get('cmdline'):
                    cmdline = ' '.join(proc.info['cmdline']).lower()
                    if self.app_id in cmdline:
                        self.pid = proc.info['pid']
                        return True
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
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
                                "height": rect[3] - rect[1]
                            })
                
                windows = []
                win32gui.EnumWindows(callback, windows)
                
                if windows:
                    # Return main window (largest)
                    main_window = max(windows, key=lambda w: w["width"] * w["height"])
                    return main_window
            
            # Fallback for non-Windows
            return {
                "title": self.game_name,
                "pid": self.pid
            }
            
        except Exception as e:
            logger.error(f"Failed to get window info: {e}")
            return None
    
    def get_steam_info(self) -> Dict[str, Any]:
        """Get Steam game information"""
        info = {
            "app_id": self.app_id,
            "steam_path": self.steam_path,
            "launch_options": self.launch_options
        }
        
        # Try to get game install path
        if os.name == 'nt':
            # Check common Steam library folders
            steam_apps = [
                "C:\\Program Files (x86)\\Steam\\steamapps\\common",
                "D:\\SteamLibrary\\steamapps\\common",
                "E:\\SteamLibrary\\steamapps\\common"
            ]
            
            for library in steam_apps:
                if os.path.exists(library):
                    # Look for game folder
                    for folder in os.listdir(library):
                        if self.game_name.lower() in folder.lower():
                            info["install_path"] = os.path.join(library, folder)
                            break
        
        return info
    
    def get_steam_library_folders(self) -> list:
        """Get all Steam library folders"""
        libraries = []
        
        if not self.steam_path:
            return libraries
        
        # Steam config path
        steam_dir = Path(self.steam_path).parent
        config_vdf = steam_dir / "config" / "libraryfolders.vdf"
        
        if config_vdf.exists():
            try:
                with open(config_vdf, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse VDF to find library paths
                import re
                paths = re.findall(r'"path"\s+"([^"]+)"', content)
                for path in paths:
                    steamapps_path = Path(path) / "steamapps" / "common"
                    if steamapps_path.exists():
                        libraries.append(str(steamapps_path))
            except Exception as e:
                logger.error(f"Failed to parse library folders: {e}")
        
        # Fallback to default paths
        if not libraries:
            default_paths = [
                steam_dir / "steamapps" / "common",
                Path("D:") / "SteamLibrary" / "steamapps" / "common",
                Path("E:") / "SteamLibrary" / "steamapps" / "common"
            ]
            
            for path in default_paths:
                if path.exists():
                    libraries.append(str(path))
        
        return libraries
    
    def find_game_executable(self) -> Optional[str]:
        """Find the game executable file"""
        libraries = self.get_steam_library_folders()
        
        for library in libraries:
            game_dirs = []
            try:
                # Look for directories matching the game name
                for item in os.listdir(library):
                    item_path = Path(library) / item
                    if item_path.is_dir() and self.game_name.lower() in item.lower():
                        game_dirs.append(item_path)
            except OSError:
                continue
            
            # Check each potential game directory
            for game_dir in game_dirs:
                # Look for common executable patterns
                exe_patterns = [
                    f"{self.game_name}*.exe",
                    "*.exe",
                    f"{self.game_name}",
                    f"{self.game_name}.x86_64"
                ]
                
                for pattern in exe_patterns:
                    try:
                        import glob
                        matches = glob.glob(str(game_dir / pattern))
                        if matches:
                            return matches[0]
                    except Exception:
                        continue
        
        return None
    
    def get_installed_games(self) -> Dict[str, Dict[str, Any]]:
        """Get list of installed Steam games"""
        games = {}
        libraries = self.get_steam_library_folders()
        
        for library in libraries:
            try:
                # Get app cache info if available
                steam_dir = Path(self.steam_path).parent if self.steam_path else None
                if steam_dir:
                    appcache_dir = steam_dir / "appcache"
                    if appcache_dir.exists():
                        # Try to read Steam's app info
                        for item in os.listdir(library):
                            item_path = Path(library) / item
                            if item_path.is_dir():
                                # Try to find app manifest
                                parent_dir = item_path.parent.parent
                                for manifest_file in parent_dir.glob(f"appmanifest_*.acf"):
                                    try:
                                        with open(manifest_file, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                            
                                        # Parse ACF format
                                        import re
                                        app_id_match = re.search(r'"appid"\s+"(\d+)"', content)
                                        name_match = re.search(r'"name"\s+"([^"]+)"', content)
                                        installdir_match = re.search(r'"installdir"\s+"([^"]+)"', content)
                                        
                                        if app_id_match and name_match and installdir_match:
                                            if installdir_match.group(1) == item:
                                                games[app_id_match.group(1)] = {
                                                    "name": name_match.group(1),
                                                    "path": str(item_path),
                                                    "app_id": app_id_match.group(1)
                                                }
                                    except Exception:
                                        continue
            except OSError:
                continue
        
        return games
    
    async def launch_with_arguments(self, arguments: list = None) -> bool:
        """Launch game with additional command line arguments"""
        if not self.steam_path:
            logger.error("Steam not found, cannot launch game")
            return False
        
        try:
            cmd = [self.steam_path, "-applaunch", str(self.app_id)]
            
            if arguments:
                cmd.extend(arguments)
            
            if self.launch_options:
                cmd.extend(self.launch_options.split())
            
            logger.info(f"Launching {self.game_name} with command: {' '.join(cmd)}")
            
            # Launch process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for game to start
            await asyncio.sleep(5)
            
            # Check if game is running
            for i in range(30):
                if self.is_game_running():
                    self.is_running = True
                    logger.info(f"Game {self.game_name} launched successfully")
                    return True
                await asyncio.sleep(1)
            
            logger.error(f"Game {self.game_name} failed to launch")
            return False
            
        except Exception as e:
            logger.error(f"Failed to launch game with arguments: {e}")
            return False
    
    async def validate_installation(self) -> Dict[str, Any]:
        """Validate the Steam installation and game"""
        validation = {
            "steam_found": self.steam_path is not None,
            "steam_path": self.steam_path,
            "game_installed": False,
            "game_path": None,
            "executable_found": False,
            "executable_path": None,
            "issues": []
        }
        
        if not validation["steam_found"]:
            validation["issues"].append("Steam installation not found")
            return validation
        
        # Check if game is installed
        installed_games = self.get_installed_games()
        if self.app_id in installed_games:
            validation["game_installed"] = True
            validation["game_path"] = installed_games[self.app_id]["path"]
        else:
            validation["issues"].append(f"Game with App ID {self.app_id} not found in Steam library")
        
        # Check for executable
        executable_path = self.find_game_executable()
        if executable_path:
            validation["executable_found"] = True
            validation["executable_path"] = executable_path
        else:
            validation["issues"].append("Game executable not found")
        
        return validation