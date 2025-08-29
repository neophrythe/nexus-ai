"""Enhanced Game Registry System with Auto-Discovery and Plugin Management

Combines SerpentAI's game discovery with modern architecture and additional features.
"""

import os
import sys
import json
import yaml
import toml
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import pickle
import structlog
from datetime import datetime
import inspect
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = structlog.get_logger()


class GamePlatform(Enum):
    """Supported game platforms"""
    STEAM = "steam"
    EPIC = "epic"
    ORIGIN = "origin"
    UPLAY = "uplay"
    GOG = "gog"
    BATTLENET = "battlenet"
    EXECUTABLE = "executable"
    WEB_BROWSER = "web_browser"
    ANDROID = "android"
    IOS = "ios"
    CUSTOM = "custom"


class GameGenre(Enum):
    """Game genres for categorization"""
    ACTION = "action"
    ADVENTURE = "adventure"
    RPG = "rpg"
    STRATEGY = "strategy"
    SIMULATION = "simulation"
    SPORTS = "sports"
    RACING = "racing"
    PUZZLE = "puzzle"
    SHOOTER = "shooter"
    PLATFORMER = "platformer"
    MOBA = "moba"
    BATTLE_ROYALE = "battle_royale"
    CARD = "card"
    OTHER = "other"


@dataclass
class GameMetadata:
    """Complete game metadata"""
    name: str
    display_name: str
    platform: GamePlatform
    genre: GameGenre = GameGenre.OTHER
    version: str = "1.0.0"
    author: str = "Unknown"
    description: str = ""
    window_name: Optional[str] = None
    process_name: Optional[str] = None
    executable_path: Optional[str] = None
    app_id: Optional[str] = None  # Steam/Epic app ID
    url: Optional[str] = None  # Web game URL
    fps: int = 60
    resolution: Tuple[int, int] = (1920, 1080)
    input_mode: str = "keyboard_mouse"  # keyboard_mouse, controller, touch
    requires_admin: bool = False
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    sprites_path: Optional[str] = None
    api_hooks: Dict[str, str] = field(default_factory=dict)
    frame_transforms: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameMetadata':
        """Create from dictionary"""
        # Convert platform and genre strings to enums
        if isinstance(data.get('platform'), str):
            data['platform'] = GamePlatform(data['platform'])
        if isinstance(data.get('genre'), str):
            data['genre'] = GameGenre(data['genre'])
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['platform'] = self.platform.value
        data['genre'] = self.genre.value
        return data


@dataclass
class GamePlugin:
    """Game plugin container"""
    metadata: GameMetadata
    game_class: Type['Game']
    plugin_path: Path
    manifest_path: Path
    loaded_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def get_instance(self, **kwargs) -> 'Game':
        """Create game instance"""
        return self.game_class(metadata=self.metadata, **kwargs)


class Game:
    """Base game class - SerpentAI compatible with enhancements"""
    
    def __init__(self, metadata: GameMetadata, **kwargs):
        self.metadata = metadata
        self.name = metadata.name
        self.platform = metadata.platform
        self.window_name = metadata.window_name
        
        # Frame capture
        self.frame_grabber = None
        self.frame_transformation_pipeline = None
        
        # Sprites
        self.sprites = {}
        self.sprite_identifier = None
        
        # API hooks
        self.api_hooks = metadata.api_hooks
        
        # State
        self.is_launched = False
        self.window_controller = None
        self.game_launcher = None
        
        # Custom initialization
        self.initialize(**kwargs)
        
    def initialize(self, **kwargs):
        """Custom initialization - override in subclasses"""
        # Base initialization - can be extended by subclasses
        logger.info(f"Initializing game plugin: {self.name}")
        
        # Load additional config from kwargs
        for key, value in kwargs.items():
            if hasattr(self.metadata, 'config') and hasattr(self.metadata.config, key):
                setattr(self.metadata.config, key, value)
        
        # Initialize sprite identifier if needed
        if self.sprites and not self.sprite_identifier:
            from nexus.sprites.sprite_identifier import SpriteIdentifier
            self.sprite_identifier = SpriteIdentifier()
            
        logger.debug(f"Game plugin {self.name} initialized successfully")
        
    def launch(self, dry_run: bool = False) -> bool:
        """Launch the game"""
        from nexus.launchers.game_launcher import GameLauncherFactory, LaunchConfig
        
        config = LaunchConfig(
            launcher_type=self.platform,
            game_path=self.metadata.executable_path,
            app_id=self.metadata.app_id,
            url=self.metadata.url,
            window_name=self.window_name,
            arguments=self.metadata.config.get('launch_args', [])
        )
        
        if not dry_run:
            self.game_launcher = GameLauncherFactory.launch_game(config)
            self.is_launched = self.game_launcher.wait_for_window()
            
            if self.is_launched:
                self._setup_window_controller()
                self._setup_frame_grabber()
                
        return self.is_launched
        
    def _setup_window_controller(self):
        """Setup window controller"""
        from nexus.window.window_controller import WindowController
        
        self.window_controller = WindowController()
        window = self.window_controller.locate_window(self.window_name)
        
        if window:
            self.window_controller.focus_window(window)
            
    def _setup_frame_grabber(self):
        """Setup frame grabber"""
        from nexus.capture.frame_grabber import FrameGrabber
        
        if self.window_controller and self.window_controller.current_window:
            window = self.window_controller.current_window
            
            self.frame_grabber = FrameGrabber(
                width=window.width,
                height=window.height,
                x_offset=window.x,
                y_offset=window.y,
                fps=self.metadata.fps,
                pipeline_string="|".join(self.metadata.frame_transforms)
            )
            
            # Start capture in background
            import threading
            capture_thread = threading.Thread(target=self.frame_grabber.start, daemon=True)
            capture_thread.start()
            
    def grab_frame(self):
        """Grab current frame"""
        if self.frame_grabber:
            return self.frame_grabber.grab_frame()
        return None
        
    def get_sprite(self, sprite_name: str):
        """Get sprite by name"""
        return self.sprites.get(sprite_name)
        
    def locate_sprite(self, sprite_name: str, frame=None):
        """Locate sprite in frame"""
        if not self.sprite_identifier:
            from nexus.sprites.sprite_identifier import SpriteIdentifier
            self.sprite_identifier = SpriteIdentifier()
            
        sprite = self.get_sprite(sprite_name)
        if sprite and frame is not None:
            return self.sprite_identifier.locate(sprite, frame)
        return None
        
    def api_action(self, action: str, *args, **kwargs):
        """Execute API action"""
        hook = self.api_hooks.get(action)
        if hook and hasattr(self, hook):
            method = getattr(self, hook)
            return method(*args, **kwargs)
        return None
        
    def is_running(self) -> bool:
        """Check if game is running"""
        if self.window_controller:
            return self.window_controller.window_exists(self.window_name)
        return False
        
    def close(self):
        """Close the game"""
        if self.game_launcher:
            self.game_launcher.close()
        if self.frame_grabber:
            self.frame_grabber.stop()
        self.is_launched = False


class GameRegistry:
    """Enhanced game registry with auto-discovery and hot-reload"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.games: Dict[str, GamePlugin] = {}
            self.plugin_dirs: List[Path] = [
                Path.home() / ".nexus" / "plugins" / "games",
                Path.cwd() / "plugins" / "games",
                Path(__file__).parent / "games"
            ]
            
            # Add SerpentAI plugin paths for compatibility
            serpent_path = Path.home() / "SerpentAI" / "plugins"
            if serpent_path.exists():
                self.plugin_dirs.append(serpent_path)
                
            self.file_observer = None
            self.auto_reload = False
            self.discovery_cache = {}
            self.initialized = True
            
    def add_plugin_directory(self, path: Path):
        """Add a plugin directory"""
        path = Path(path)
        if path not in self.plugin_dirs:
            self.plugin_dirs.append(path)
            logger.info(f"Added plugin directory: {path}")
            
    def discover_games(self, force_reload: bool = False) -> List[str]:
        """Discover all available games"""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
                
            # Check for game plugins
            for game_dir in plugin_dir.iterdir():
                if not game_dir.is_dir():
                    continue
                    
                # Look for manifest files
                manifest_path = None
                for ext in ['.yaml', '.yml', '.json', '.toml']:
                    potential = game_dir / f"manifest{ext}"
                    if potential.exists():
                        manifest_path = potential
                        break
                        
                # Fallback to plugin.json (SerpentAI compatibility)
                if not manifest_path:
                    potential = game_dir / "plugin.json"
                    if potential.exists():
                        manifest_path = potential
                        
                if manifest_path:
                    try:
                        game_name = self._load_game_plugin(game_dir, manifest_path, force_reload)
                        if game_name:
                            discovered.append(game_name)
                    except Exception as e:
                        logger.error(f"Failed to load game plugin from {game_dir}: {e}")
                        
        logger.info(f"Discovered {len(discovered)} games: {discovered}")
        return discovered
        
    def _load_game_plugin(self, plugin_dir: Path, manifest_path: Path, 
                         force_reload: bool = False) -> Optional[str]:
        """Load a game plugin"""
        # Check cache
        cache_key = str(manifest_path)
        if not force_reload and cache_key in self.discovery_cache:
            mtime = manifest_path.stat().st_mtime
            if self.discovery_cache[cache_key]['mtime'] == mtime:
                return self.discovery_cache[cache_key]['name']
                
        # Load manifest
        manifest_data = self._load_manifest(manifest_path)
        if not manifest_data:
            return None
            
        # Create metadata
        metadata = self._create_metadata(manifest_data)
        
        # Find and load game class
        game_class = self._load_game_class(plugin_dir, manifest_data)
        if not game_class:
            return None
            
        # Create plugin
        plugin = GamePlugin(
            metadata=metadata,
            game_class=game_class,
            plugin_path=plugin_dir,
            manifest_path=manifest_path
        )
        
        # Register
        self.games[metadata.name] = plugin
        
        # Update cache
        self.discovery_cache[cache_key] = {
            'name': metadata.name,
            'mtime': manifest_path.stat().st_mtime
        }
        
        logger.info(f"Loaded game plugin: {metadata.name} v{metadata.version}")
        return metadata.name
        
    def _load_manifest(self, manifest_path: Path) -> Optional[Dict]:
        """Load manifest file"""
        try:
            content = manifest_path.read_text()
            
            if manifest_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            elif manifest_path.suffix == '.json':
                return json.loads(content)
            elif manifest_path.suffix == '.toml':
                return toml.loads(content)
                
        except Exception as e:
            logger.error(f"Failed to load manifest {manifest_path}: {e}")
            
        return None
        
    def _create_metadata(self, manifest_data: Dict) -> GameMetadata:
        """Create game metadata from manifest"""
        # Handle different manifest formats
        if 'game' in manifest_data:
            # Nexus format
            game_data = manifest_data['game']
        elif 'config' in manifest_data:
            # SerpentAI format
            game_data = manifest_data['config']
            game_data['name'] = manifest_data.get('name', 'unknown')
            game_data['version'] = manifest_data.get('version', '1.0.0')
        else:
            game_data = manifest_data
            
        # Map old field names for compatibility
        if 'window_name' not in game_data and 'window_title' in game_data:
            game_data['window_name'] = game_data['window_title']
            
        # Set defaults
        game_data.setdefault('display_name', game_data.get('name', 'Unknown Game'))
        game_data.setdefault('platform', 'executable')
        game_data.setdefault('genre', 'other')
        
        return GameMetadata.from_dict(game_data)
        
    def _load_game_class(self, plugin_dir: Path, manifest_data: Dict) -> Optional[Type[Game]]:
        """Load game class from plugin"""
        # Find entry point
        entry_point = manifest_data.get('entry_point')
        if not entry_point:
            # Look for standard names
            for name in ['game.py', f"{plugin_dir.name}.py", 'serpent_game.py']:
                potential = plugin_dir / name
                if potential.exists():
                    entry_point = name
                    break
                    
        if not entry_point:
            return None
            
        entry_file = plugin_dir / entry_point
        if not entry_file.exists():
            return None
            
        # Load module
        try:
            spec = importlib.util.spec_from_file_location(
                f"game_plugin_{plugin_dir.name}",
                entry_file
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            # Find game class
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Game) and obj != Game:
                    return obj
                    
            # Fallback: look for class with Game in name
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and 'Game' in name:
                    # Wrap in our Game class if needed
                    if not issubclass(obj, Game):
                        class WrappedGame(Game):
                            _wrapped_class = obj
                            
                            def initialize(self, **kwargs):
                                self._wrapped = self._wrapped_class(**kwargs)
                                
                        return WrappedGame
                    return obj
                    
        except Exception as e:
            logger.error(f"Failed to load game class from {entry_file}: {e}")
            
        return None
        
    def get_game(self, name: str) -> Optional[GamePlugin]:
        """Get game plugin by name"""
        # Try exact match first
        if name in self.games:
            return self.games[name]
            
        # Try case-insensitive match
        for game_name, plugin in self.games.items():
            if game_name.lower() == name.lower():
                return plugin
                
        # Try partial match
        for game_name, plugin in self.games.items():
            if name.lower() in game_name.lower():
                return plugin
                
        return None
        
    def create_game(self, name: str, **kwargs) -> Optional[Game]:
        """Create game instance"""
        plugin = self.get_game(name)
        if plugin:
            return plugin.get_instance(**kwargs)
        return None
        
    def list_games(self) -> List[Dict]:
        """List all registered games"""
        games = []
        for name, plugin in self.games.items():
            games.append({
                'name': name,
                'display_name': plugin.metadata.display_name,
                'platform': plugin.metadata.platform.value,
                'genre': plugin.metadata.genre.value,
                'version': plugin.metadata.version,
                'author': plugin.metadata.author,
                'description': plugin.metadata.description,
                'is_active': plugin.is_active
            })
        return games
        
    def enable_auto_reload(self):
        """Enable hot-reload for game plugins"""
        if self.auto_reload:
            return
            
        self.auto_reload = True
        
        class PluginReloadHandler(FileSystemEventHandler):
            def __init__(self, registry):
                self.registry = registry
                
            def on_modified(self, event):
                if event.src_path.endswith('.py'):
                    logger.info(f"Plugin file modified: {event.src_path}")
                    self.registry.discover_games(force_reload=True)
                    
        handler = PluginReloadHandler(self)
        self.file_observer = Observer()
        
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                self.file_observer.schedule(handler, str(plugin_dir), recursive=True)
                
        self.file_observer.start()
        logger.info("Auto-reload enabled for game plugins")
        
    def disable_auto_reload(self):
        """Disable hot-reload"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
        self.auto_reload = False
        logger.info("Auto-reload disabled")
        
    def export_registry(self, path: Path):
        """Export registry to file"""
        export_data = {
            'games': self.list_games(),
            'plugin_dirs': [str(p) for p in self.plugin_dirs],
            'exported_at': datetime.now().isoformat()
        }
        
        path = Path(path)
        if path.suffix == '.json':
            path.write_text(json.dumps(export_data, indent=2))
        elif path.suffix in ['.yaml', '.yml']:
            path.write_text(yaml.dump(export_data))
            
        logger.info(f"Registry exported to {path}")
        
    def import_registry(self, path: Path):
        """Import registry from file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")
            
        if path.suffix == '.json':
            data = json.loads(path.read_text())
        elif path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(path.read_text())
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        # Add plugin directories
        for dir_path in data.get('plugin_dirs', []):
            self.add_plugin_directory(Path(dir_path))
            
        # Discover games
        self.discover_games()
        
        logger.info(f"Registry imported from {path}")


# Global registry instance
game_registry = GameRegistry()


# Convenience functions for SerpentAI compatibility
def initialize_game(game_name: str, **kwargs) -> Optional[Game]:
    """Initialize game by name - SerpentAI compatible"""
    game_registry.discover_games()
    return game_registry.create_game(game_name, **kwargs)


def discover_games() -> List[str]:
    """Discover available games - SerpentAI compatible"""
    return game_registry.discover_games()


def get_game_list() -> List[Dict]:
    """Get list of all games"""
    return game_registry.list_games()