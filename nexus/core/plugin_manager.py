import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
from datetime import datetime
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from nexus.core.base import (
    BasePlugin, PluginManifest, PluginStatus, PluginType,
    GamePlugin, AgentPlugin, CapturePlugin, VisionPlugin, InputPlugin
)


logger = structlog.get_logger()


class PluginFileHandler(FileSystemEventHandler):
    
    def __init__(self, plugin_manager: "PluginManager"):
        self.plugin_manager = plugin_manager
        self.reload_cooldown = {}
        
    def on_modified(self, event: FileModifiedEvent):
        if event.is_directory:
            return
            
        path = Path(event.src_path)
        
        if path.suffix not in ['.py', '.yaml', '.yml', '.toml']:
            return
            
        current_time = datetime.now()
        last_reload = self.reload_cooldown.get(str(path))
        
        if last_reload and (current_time - last_reload).total_seconds() < 1:
            return
            
        self.reload_cooldown[str(path)] = current_time
        
        for plugin_name, plugin_path in self.plugin_manager.plugin_paths.items():
            if plugin_path.parent in path.parents or plugin_path.parent == path.parent:
                asyncio.create_task(self.plugin_manager.reload_plugin(plugin_name))
                break


class PluginManager:
    
    PLUGIN_CLASSES = {
        PluginType.GAME: GamePlugin,
        PluginType.AGENT: AgentPlugin,
        PluginType.CAPTURE: CapturePlugin,
        PluginType.VISION: VisionPlugin,
        PluginType.INPUT: InputPlugin,
    }
    
    def __init__(self, plugin_dirs: List[Path], enable_hot_reload: bool = True):
        self.plugin_dirs = plugin_dirs
        self.enable_hot_reload = enable_hot_reload
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_paths: Dict[str, Path] = {}
        self.plugin_manifests: Dict[str, PluginManifest] = {}
        self.observer: Optional[Observer] = None
        self._lock = asyncio.Lock()
        
        if self.enable_hot_reload:
            self._setup_file_watcher()
    
    def _setup_file_watcher(self):
        self.observer = Observer()
        handler = PluginFileHandler(self)
        
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                self.observer.schedule(handler, str(plugin_dir), recursive=True)
        
        self.observer.start()
        logger.info("Hot-reload enabled for plugins")
    
    async def discover_plugins(self) -> List[str]:
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
                
            for manifest_path in plugin_dir.glob("*/manifest.*"):
                if manifest_path.suffix in ['.yaml', '.yml']:
                    manifest = PluginManifest.from_yaml(manifest_path)
                elif manifest_path.suffix == '.toml':
                    manifest = PluginManifest.from_toml(manifest_path)
                else:
                    continue
                
                plugin_name = manifest.name
                self.plugin_manifests[plugin_name] = manifest
                self.plugin_paths[plugin_name] = manifest_path.parent
                discovered.append(plugin_name)
                
                logger.info(f"Discovered plugin: {plugin_name} v{manifest.version}")
        
        return discovered
    
    async def load_plugin(self, name: str, config: Optional[Dict[str, Any]] = None) -> BasePlugin:
        async with self._lock:
            if name in self.plugins:
                logger.warning(f"Plugin {name} already loaded")
                return self.plugins[name]
            
            if name not in self.plugin_manifests:
                raise ValueError(f"Plugin {name} not found")
            
            manifest = self.plugin_manifests[name]
            plugin_path = self.plugin_paths[name]
            
            await self._check_dependencies(manifest)
            
            config = config or {}
            
            try:
                module = self._load_module(name, plugin_path / manifest.entry_point)
                
                plugin_class = None
                base_class = self.PLUGIN_CLASSES.get(manifest.plugin_type)
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, base_class) and attr != base_class:
                        plugin_class = attr
                        break
                
                if not plugin_class:
                    raise ValueError(f"No valid plugin class found in {name}")
                
                plugin = plugin_class(manifest, config)
                await plugin.initialize()
                
                if not await plugin.validate():
                    raise ValueError(f"Plugin {name} validation failed")
                
                plugin.status = PluginStatus.LOADED
                plugin.loaded_at = datetime.now()
                
                self.plugins[name] = plugin
                logger.info(f"Loaded plugin: {name}")
                
                return plugin
                
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}")
                raise
    
    async def unload_plugin(self, name: str) -> None:
        async with self._lock:
            if name not in self.plugins:
                logger.warning(f"Plugin {name} not loaded")
                return
            
            plugin = self.plugins[name]
            
            try:
                await plugin.shutdown()
                plugin.status = PluginStatus.UNLOADED
                del self.plugins[name]
                
                module_name = f"nexus_plugin_{name}"
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                logger.info(f"Unloaded plugin: {name}")
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {name}: {e}")
                plugin.status = PluginStatus.ERROR
                raise
    
    async def reload_plugin(self, name: str) -> BasePlugin:
        logger.info(f"Reloading plugin: {name}")
        
        config = None
        if name in self.plugins:
            config = self.plugins[name].config
            await self.unload_plugin(name)
        
        manifest_path = self.plugin_paths[name] / "manifest"
        if (manifest_path.with_suffix('.yaml')).exists():
            self.plugin_manifests[name] = PluginManifest.from_yaml(manifest_path.with_suffix('.yaml'))
        elif (manifest_path.with_suffix('.yml')).exists():
            self.plugin_manifests[name] = PluginManifest.from_yaml(manifest_path.with_suffix('.yml'))
        elif (manifest_path.with_suffix('.toml')).exists():
            self.plugin_manifests[name] = PluginManifest.from_toml(manifest_path.with_suffix('.toml'))
        
        return await self.load_plugin(name, config)
    
    async def _check_dependencies(self, manifest: PluginManifest) -> None:
        for dep in manifest.dependencies:
            if dep not in self.plugins:
                logger.info(f"Loading dependency: {dep}")
                await self.load_plugin(dep)
    
    def _load_module(self, name: str, path: Path) -> Any:
        module_name = f"nexus_plugin_{name}"
        
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load module from {path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        return self.plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        return [
            plugin for plugin in self.plugins.values()
            if plugin.manifest.plugin_type == plugin_type
        ]
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: plugin.get_info()
            for name, plugin in self.plugins.items()
        }
    
    async def shutdown(self):
        for name in list(self.plugins.keys()):
            await self.unload_plugin(name)
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            
        logger.info("Plugin manager shutdown complete")