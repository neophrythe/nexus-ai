"""Enhanced Plugin System for Nexus Framework - Full SerpentAI Parity"""

import os
import sys
import shutil
import importlib
import importlib.util
import subprocess
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
import git
from jinja2 import Template, Environment, FileSystemLoader
import requests
from packaging import version
import hashlib
import zipfile
import tarfile
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger()


class PluginSource(Enum):
    """Plugin installation sources"""
    LOCAL = "local"
    GIT = "git"
    URL = "url"
    REGISTRY = "registry"
    FILE = "file"


class PluginLifecycleHook(Enum):
    """Plugin lifecycle hooks"""
    PRE_INSTALL = "pre_install"
    POST_INSTALL = "post_install"
    PRE_UNINSTALL = "pre_uninstall"
    POST_UNINSTALL = "post_uninstall"
    PRE_ACTIVATE = "pre_activate"
    POST_ACTIVATE = "post_activate"
    PRE_DEACTIVATE = "pre_deactivate"
    POST_DEACTIVATE = "post_deactivate"
    PRE_UPDATE = "pre_update"
    POST_UPDATE = "post_update"


@dataclass
class PluginMetadata:
    """Complete plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str
    entry_point: str
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    python_dependencies: List[str] = field(default_factory=list)
    system_dependencies: List[str] = field(default_factory=list)
    
    # Compatibility
    min_nexus_version: str = "0.1.0"
    max_nexus_version: Optional[str] = None
    compatible_platforms: List[str] = field(default_factory=lambda: ["windows", "linux", "darwin"])
    
    # Files and resources
    files: List[Dict[str, str]] = field(default_factory=list)
    templates: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    
    # Configuration
    config_schema: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Hooks
    hooks: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "PluginMetadata":
        """Load from manifest file"""
        if manifest_path.suffix in ['.yaml', '.yml']:
            with open(manifest_path, 'r') as f:
                data = yaml.safe_load(f)
        elif manifest_path.suffix == '.json':
            with open(manifest_path, 'r') as f:
                data = json.load(f)
        elif manifest_path.suffix == '.toml':
            with open(manifest_path, 'r') as f:
                data = toml.load(f)
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type,
            "entry_point": self.entry_point,
            "dependencies": self.dependencies,
            "python_dependencies": self.python_dependencies,
            "system_dependencies": self.system_dependencies,
            "min_nexus_version": self.min_nexus_version,
            "max_nexus_version": self.max_nexus_version,
            "compatible_platforms": self.compatible_platforms,
            "files": self.files,
            "templates": self.templates,
            "assets": self.assets,
            "config_schema": self.config_schema,
            "default_config": self.default_config,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
            "tags": self.tags,
            "hooks": self.hooks
        }


@dataclass
class InstalledPlugin:
    """Installed plugin information"""
    metadata: PluginMetadata
    install_path: Path
    install_date: datetime
    source: PluginSource
    source_url: Optional[str]
    is_active: bool
    config: Dict[str, Any]
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metadata": self.metadata.to_dict(),
            "install_path": str(self.install_path),
            "install_date": self.install_date.isoformat(),
            "source": self.source.value,
            "source_url": self.source_url,
            "is_active": self.is_active,
            "config": self.config,
            "checksum": self.checksum
        }


class PluginRegistry:
    """Plugin registry for managing available plugins"""
    
    def __init__(self, registry_url: Optional[str] = None):
        self.registry_url = registry_url or "https://nexus-plugins.io/registry.json"
        self.cache_dir = Path.home() / ".nexus" / "plugin_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_cache: Dict[str, Any] = {}
        
    async def fetch_registry(self) -> Dict[str, Any]:
        """Fetch plugin registry"""
        try:
            response = requests.get(self.registry_url, timeout=10)
            response.raise_for_status()
            self.registry_cache = response.json()
            
            # Cache locally
            cache_file = self.cache_dir / "registry.json"
            with open(cache_file, 'w') as f:
                json.dump(self.registry_cache, f, indent=2)
            
            return self.registry_cache
            
        except Exception as e:
            logger.warning(f"Failed to fetch registry: {e}")
            
            # Try local cache
            cache_file = self.cache_dir / "registry.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.registry_cache = json.load(f)
                return self.registry_cache
            
            return {}
    
    async def search_plugins(self, query: str, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for plugins in registry"""
        if not self.registry_cache:
            await self.fetch_registry()
        
        results = []
        
        for plugin_name, plugin_info in self.registry_cache.get("plugins", {}).items():
            # Check name and description
            if query.lower() in plugin_name.lower() or \
               query.lower() in plugin_info.get("description", "").lower():
                
                # Filter by type if specified
                if plugin_type and plugin_info.get("type") != plugin_type:
                    continue
                
                results.append({
                    "name": plugin_name,
                    **plugin_info
                })
        
        return results
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin information"""
        if not self.registry_cache:
            await self.fetch_registry()
        
        return self.registry_cache.get("plugins", {}).get(plugin_name)


class PluginInstaller:
    """Plugin installation and management"""
    
    def __init__(self, plugins_dir: Path, temp_dir: Optional[Path] = None):
        self.plugins_dir = plugins_dir
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = temp_dir or Path.home() / ".nexus" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.installed_plugins: Dict[str, InstalledPlugin] = {}
        self.load_installed_plugins()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def load_installed_plugins(self):
        """Load list of installed plugins"""
        plugins_file = self.plugins_dir / "installed.json"
        
        if plugins_file.exists():
            with open(plugins_file, 'r') as f:
                data = json.load(f)
                
            for plugin_name, plugin_data in data.items():
                metadata = PluginMetadata(**plugin_data["metadata"])
                self.installed_plugins[plugin_name] = InstalledPlugin(
                    metadata=metadata,
                    install_path=Path(plugin_data["install_path"]),
                    install_date=datetime.fromisoformat(plugin_data["install_date"]),
                    source=PluginSource(plugin_data["source"]),
                    source_url=plugin_data.get("source_url"),
                    is_active=plugin_data["is_active"],
                    config=plugin_data.get("config", {}),
                    checksum=plugin_data["checksum"]
                )
    
    def save_installed_plugins(self):
        """Save list of installed plugins"""
        plugins_file = self.plugins_dir / "installed.json"
        
        data = {
            name: plugin.to_dict()
            for name, plugin in self.installed_plugins.items()
        }
        
        with open(plugins_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def install_plugin(self, source: str, source_type: PluginSource = PluginSource.LOCAL,
                           config: Optional[Dict[str, Any]] = None) -> InstalledPlugin:
        """
        Install a plugin from various sources
        
        Args:
            source: Plugin source (path, URL, git repo, etc.)
            source_type: Type of source
            config: Plugin configuration
        
        Returns:
            Installed plugin information
        """
        logger.info(f"Installing plugin from {source_type.value}: {source}")
        
        # Download/copy plugin to temp directory
        temp_plugin_dir = await self._fetch_plugin(source, source_type)
        
        # Load manifest
        manifest_path = self._find_manifest(temp_plugin_dir)
        if not manifest_path:
            raise ValueError(f"No manifest found in {temp_plugin_dir}")
        
        metadata = PluginMetadata.from_manifest(manifest_path)
        
        # Check compatibility
        self._check_compatibility(metadata)
        
        # Check dependencies
        await self._install_dependencies(metadata)
        
        # Run pre-install hook
        await self._run_hook(metadata, PluginLifecycleHook.PRE_INSTALL, temp_plugin_dir)
        
        # Install plugin files
        install_path = self.plugins_dir / metadata.name
        if install_path.exists():
            if metadata.name in self.installed_plugins:
                # Update existing plugin
                await self.uninstall_plugin(metadata.name, keep_config=True)
            else:
                shutil.rmtree(install_path)
        
        shutil.copytree(temp_plugin_dir, install_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(install_path)
        
        # Create installed plugin record
        plugin = InstalledPlugin(
            metadata=metadata,
            install_path=install_path,
            install_date=datetime.now(),
            source=source_type,
            source_url=source if source_type != PluginSource.LOCAL else None,
            is_active=False,
            config=config or metadata.default_config,
            checksum=checksum
        )
        
        self.installed_plugins[metadata.name] = plugin
        self.save_installed_plugins()
        
        # Run post-install hook
        await self._run_hook(metadata, PluginLifecycleHook.POST_INSTALL, install_path)
        
        # Clean up temp directory
        shutil.rmtree(temp_plugin_dir)
        
        logger.info(f"Successfully installed plugin: {metadata.name} v{metadata.version}")
        return plugin
    
    async def uninstall_plugin(self, plugin_name: str, keep_config: bool = False) -> bool:
        """
        Uninstall a plugin
        
        Args:
            plugin_name: Name of plugin to uninstall
            keep_config: Whether to keep configuration
        
        Returns:
            True if successful
        """
        if plugin_name not in self.installed_plugins:
            logger.warning(f"Plugin {plugin_name} not installed")
            return False
        
        plugin = self.installed_plugins[plugin_name]
        
        # Deactivate if active
        if plugin.is_active:
            await self.deactivate_plugin(plugin_name)
        
        # Run pre-uninstall hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.PRE_UNINSTALL, plugin.install_path)
        
        # Save config if requested
        config_backup = None
        if keep_config:
            config_backup = plugin.config.copy()
        
        # Remove plugin files
        if plugin.install_path.exists():
            shutil.rmtree(plugin.install_path)
        
        # Remove from installed list
        del self.installed_plugins[plugin_name]
        self.save_installed_plugins()
        
        # Restore config if keeping
        if keep_config and config_backup:
            config_file = self.plugins_dir / f"{plugin_name}.config.json"
            with open(config_file, 'w') as f:
                json.dump(config_backup, f, indent=2)
        
        # Run post-uninstall hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.POST_UNINSTALL, None)
        
        logger.info(f"Successfully uninstalled plugin: {plugin_name}")
        return True
    
    async def update_plugin(self, plugin_name: str) -> Optional[InstalledPlugin]:
        """Update a plugin to latest version"""
        if plugin_name not in self.installed_plugins:
            logger.warning(f"Plugin {plugin_name} not installed")
            return None
        
        plugin = self.installed_plugins[plugin_name]
        
        # Only update if source is tracked
        if not plugin.source_url:
            logger.warning(f"Cannot update plugin {plugin_name}: no source URL")
            return None
        
        # Run pre-update hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.PRE_UPDATE, plugin.install_path)
        
        # Save current config
        config_backup = plugin.config.copy()
        
        # Reinstall from source
        updated_plugin = await self.install_plugin(
            plugin.source_url,
            plugin.source,
            config_backup
        )
        
        # Run post-update hook
        await self._run_hook(updated_plugin.metadata, PluginLifecycleHook.POST_UPDATE, 
                           updated_plugin.install_path)
        
        logger.info(f"Successfully updated plugin: {plugin_name}")
        return updated_plugin
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """Activate an installed plugin"""
        if plugin_name not in self.installed_plugins:
            logger.warning(f"Plugin {plugin_name} not installed")
            return False
        
        plugin = self.installed_plugins[plugin_name]
        
        if plugin.is_active:
            logger.info(f"Plugin {plugin_name} already active")
            return True
        
        # Run pre-activate hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.PRE_ACTIVATE, plugin.install_path)
        
        # Add to Python path
        if str(plugin.install_path) not in sys.path:
            sys.path.insert(0, str(plugin.install_path))
        
        plugin.is_active = True
        self.save_installed_plugins()
        
        # Run post-activate hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.POST_ACTIVATE, plugin.install_path)
        
        logger.info(f"Activated plugin: {plugin_name}")
        return True
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin"""
        if plugin_name not in self.installed_plugins:
            logger.warning(f"Plugin {plugin_name} not installed")
            return False
        
        plugin = self.installed_plugins[plugin_name]
        
        if not plugin.is_active:
            logger.info(f"Plugin {plugin_name} already inactive")
            return True
        
        # Run pre-deactivate hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.PRE_DEACTIVATE, plugin.install_path)
        
        # Remove from Python path
        if str(plugin.install_path) in sys.path:
            sys.path.remove(str(plugin.install_path))
        
        # Unload modules
        modules_to_remove = []
        for module_name in sys.modules:
            if hasattr(sys.modules[module_name], '__file__'):
                module_file = sys.modules[module_name].__file__
                if module_file and str(plugin.install_path) in module_file:
                    modules_to_remove.append(module_name)
        
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        plugin.is_active = False
        self.save_installed_plugins()
        
        # Run post-deactivate hook
        await self._run_hook(plugin.metadata, PluginLifecycleHook.POST_DEACTIVATE, plugin.install_path)
        
        logger.info(f"Deactivated plugin: {plugin_name}")
        return True
    
    async def _fetch_plugin(self, source: str, source_type: PluginSource) -> Path:
        """Fetch plugin from source to temp directory"""
        temp_dir = self.temp_dir / f"plugin_{int(datetime.now().timestamp())}"
        temp_dir.mkdir(parents=True)
        
        if source_type == PluginSource.LOCAL:
            # Copy local directory
            source_path = Path(source)
            if not source_path.exists():
                raise ValueError(f"Local path does not exist: {source}")
            
            if source_path.is_dir():
                shutil.copytree(source_path, temp_dir / "plugin")
            else:
                # Extract archive
                self._extract_archive(source_path, temp_dir / "plugin")
            
            return temp_dir / "plugin"
        
        elif source_type == PluginSource.GIT:
            # Clone git repository
            git.Repo.clone_from(source, temp_dir / "plugin")
            return temp_dir / "plugin"
        
        elif source_type == PluginSource.URL:
            # Download from URL
            response = requests.get(source, stream=True)
            response.raise_for_status()
            
            # Determine filename
            filename = source.split("/")[-1]
            if not filename:
                filename = "plugin.zip"
            
            download_path = temp_dir / filename
            
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract if archive
            if download_path.suffix in ['.zip', '.tar', '.gz', '.bz2']:
                extract_dir = temp_dir / "plugin"
                self._extract_archive(download_path, extract_dir)
                return extract_dir
            else:
                return download_path.parent
        
        elif source_type == PluginSource.REGISTRY:
            # Fetch from registry
            registry = PluginRegistry()
            plugin_info = await registry.get_plugin_info(source)
            
            if not plugin_info:
                raise ValueError(f"Plugin {source} not found in registry")
            
            # Download from registry URL
            return await self._fetch_plugin(plugin_info["url"], PluginSource.URL)
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _extract_archive(self, archive_path: Path, extract_dir: Path):
        """Extract archive file"""
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(extract_dir)
        
        elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
            mode = 'r'
            if archive_path.suffix == '.gz':
                mode = 'r:gz'
            elif archive_path.suffix == '.bz2':
                mode = 'r:bz2'
            
            with tarfile.open(archive_path, mode) as t:
                t.extractall(extract_dir)
        
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    def _find_manifest(self, plugin_dir: Path) -> Optional[Path]:
        """Find plugin manifest file"""
        for pattern in ["manifest.yaml", "manifest.yml", "manifest.json", "manifest.toml",
                       "plugin.yaml", "plugin.yml", "plugin.json", "plugin.toml"]:
            manifest_path = plugin_dir / pattern
            if manifest_path.exists():
                return manifest_path
        
        # Check subdirectories
        for subdir in plugin_dir.iterdir():
            if subdir.is_dir():
                manifest_path = self._find_manifest(subdir)
                if manifest_path:
                    return manifest_path
        
        return None
    
    def _check_compatibility(self, metadata: PluginMetadata):
        """Check plugin compatibility"""
        # Check Nexus version
        from nexus import __version__ as nexus_version
        
        if metadata.min_nexus_version:
            if version.parse(nexus_version) < version.parse(metadata.min_nexus_version):
                raise ValueError(f"Plugin requires Nexus >= {metadata.min_nexus_version}")
        
        if metadata.max_nexus_version:
            if version.parse(nexus_version) > version.parse(metadata.max_nexus_version):
                raise ValueError(f"Plugin requires Nexus <= {metadata.max_nexus_version}")
        
        # Check platform
        import platform
        current_platform = platform.system().lower()
        
        if metadata.compatible_platforms:
            if current_platform not in metadata.compatible_platforms:
                raise ValueError(f"Plugin not compatible with {current_platform}")
    
    async def _install_dependencies(self, metadata: PluginMetadata):
        """Install plugin dependencies"""
        # Install Python dependencies
        if metadata.python_dependencies:
            logger.info(f"Installing Python dependencies: {metadata.python_dependencies}")
            
            for dep in metadata.python_dependencies:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {dep}: {e}")
                    raise
        
        # Install other plugins
        if metadata.dependencies:
            for dep in metadata.dependencies:
                if dep not in self.installed_plugins:
                    logger.info(f"Installing dependency: {dep}")
                    
                    # Try to install from registry
                    try:
                        await self.install_plugin(dep, PluginSource.REGISTRY)
                    except Exception as e:
                        logger.error(f"Failed to install dependency {dep}: {e}")
                        raise
    
    async def _run_hook(self, metadata: PluginMetadata, hook: PluginLifecycleHook,
                       plugin_path: Optional[Path]):
        """Run plugin lifecycle hook"""
        hook_script = metadata.hooks.get(hook.value)
        
        if not hook_script:
            return
        
        logger.info(f"Running {hook.value} hook for {metadata.name}")
        
        if plugin_path and (plugin_path / hook_script).exists():
            # Run hook script
            script_path = plugin_path / hook_script
            
            if script_path.suffix == '.py':
                # Run Python script
                spec = importlib.util.spec_from_file_location("hook", script_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Call hook function if exists
                    if hasattr(module, hook.value):
                        hook_func = getattr(module, hook.value)
                        if asyncio.iscoroutinefunction(hook_func):
                            await hook_func()
                        else:
                            hook_func()
            
            elif script_path.suffix in ['.sh', '.bat', '.cmd']:
                # Run shell script
                subprocess.run([str(script_path)], check=True)
    
    def _calculate_checksum(self, plugin_path: Path) -> str:
        """Calculate plugin checksum"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(plugin_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()


class PluginGenerator:
    """Generate plugin templates"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
    
    def generate_plugin(self, plugin_type: str, name: str, output_dir: Path,
                       config: Optional[Dict[str, Any]] = None):
        """
        Generate a new plugin from template
        
        Args:
            plugin_type: Type of plugin to generate
            name: Plugin name
            output_dir: Output directory
            config: Template configuration
        """
        config = config or {}
        
        # Set default values
        config.setdefault("name", name)
        config.setdefault("version", "0.1.0")
        config.setdefault("author", "Unknown")
        config.setdefault("description", f"A {plugin_type} plugin for Nexus")
        config.setdefault("plugin_type", plugin_type)
        
        # Create output directory
        plugin_dir = output_dir / name
        plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Get template for plugin type
        template_name = f"{plugin_type}_plugin"
        
        if not (self.templates_dir / template_name).exists():
            # Use generic template
            template_name = "generic_plugin"
        
        template_dir = self.templates_dir / template_name
        
        # Generate files from templates
        for template_file in template_dir.rglob("*"):
            if template_file.is_file():
                # Calculate relative path
                rel_path = template_file.relative_to(template_dir)
                
                # Render template
                if template_file.suffix in ['.j2', '.jinja', '.jinja2']:
                    # Jinja template
                    template = self.env.get_template(f"{template_name}/{rel_path}")
                    content = template.render(**config)
                    
                    # Remove template extension
                    output_file = plugin_dir / str(rel_path).replace('.j2', '').replace('.jinja2', '').replace('.jinja', '')
                else:
                    # Copy as-is
                    output_file = plugin_dir / rel_path
                    content = template_file.read_text()
                
                # Create output file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(content)
        
        # Generate manifest
        manifest = {
            "name": config["name"],
            "version": config["version"],
            "author": config["author"],
            "description": config["description"],
            "plugin_type": config["plugin_type"],
            "entry_point": f"{name}.py",
            "dependencies": [],
            "python_dependencies": [],
            "min_nexus_version": "0.1.0",
            "files": [],
            "config_schema": {},
            "default_config": {},
            "hooks": {
                "post_install": "scripts/install.py",
                "pre_uninstall": "scripts/uninstall.py"
            }
        }
        
        manifest_path = plugin_dir / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        logger.info(f"Generated {plugin_type} plugin: {name} at {plugin_dir}")
        return plugin_dir


class EnhancedPluginManager:
    """Enhanced plugin manager with full SerpentAI parity"""
    
    def __init__(self, plugins_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.plugins_dir = plugins_dir
        self.config = config or {}
        
        # Initialize components
        self.installer = PluginInstaller(plugins_dir)
        self.registry = PluginRegistry(self.config.get("registry_url"))
        self.generator = PluginGenerator()
        
        # Loaded plugin instances
        self.loaded_plugins: Dict[str, Any] = {}
        
        # Plugin hooks
        self.hooks: Dict[str, List[Callable]] = {
            hook.value: [] for hook in PluginLifecycleHook
        }
        
        logger.info(f"Enhanced Plugin Manager initialized with {len(self.installer.installed_plugins)} plugins")
    
    async def install(self, source: str, source_type: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None) -> InstalledPlugin:
        """Install a plugin"""
        if source_type:
            source_type = PluginSource(source_type)
        else:
            # Auto-detect source type
            if source.startswith("http://") or source.startswith("https://"):
                source_type = PluginSource.URL
            elif source.endswith(".git") or "github.com" in source:
                source_type = PluginSource.GIT
            elif Path(source).exists():
                source_type = PluginSource.LOCAL
            else:
                source_type = PluginSource.REGISTRY
        
        return await self.installer.install_plugin(source, source_type, config)
    
    async def uninstall(self, plugin_name: str, keep_config: bool = False) -> bool:
        """Uninstall a plugin"""
        # Unload if loaded
        if plugin_name in self.loaded_plugins:
            await self.unload(plugin_name)
        
        return await self.installer.uninstall_plugin(plugin_name, keep_config)
    
    async def update(self, plugin_name: str) -> Optional[InstalledPlugin]:
        """Update a plugin"""
        # Unload if loaded
        if plugin_name in self.loaded_plugins:
            await self.unload(plugin_name)
        
        updated = await self.installer.update_plugin(plugin_name)
        
        # Reload if was loaded
        if updated and plugin_name in self.loaded_plugins:
            await self.load(plugin_name)
        
        return updated
    
    async def load(self, plugin_name: str) -> Any:
        """Load and instantiate a plugin"""
        if plugin_name in self.loaded_plugins:
            logger.info(f"Plugin {plugin_name} already loaded")
            return self.loaded_plugins[plugin_name]
        
        if plugin_name not in self.installer.installed_plugins:
            raise ValueError(f"Plugin {plugin_name} not installed")
        
        plugin_info = self.installer.installed_plugins[plugin_name]
        
        # Activate plugin
        await self.installer.activate_plugin(plugin_name)
        
        # Load plugin module
        entry_point = plugin_info.install_path / plugin_info.metadata.entry_point
        
        spec = importlib.util.spec_from_file_location(plugin_name, entry_point)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load plugin from {entry_point}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_name] = module
        spec.loader.exec_module(module)
        
        # Find and instantiate plugin class
        plugin_class = None
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith("Plugin"):
                plugin_class = attr
                break
        
        if not plugin_class:
            raise ValueError(f"No plugin class found in {plugin_name}")
        
        # Instantiate plugin
        plugin_instance = plugin_class(plugin_info.config)
        
        # Initialize if has method
        if hasattr(plugin_instance, 'initialize'):
            if asyncio.iscoroutinefunction(plugin_instance.initialize):
                await plugin_instance.initialize()
            else:
                plugin_instance.initialize()
        
        self.loaded_plugins[plugin_name] = plugin_instance
        
        logger.info(f"Loaded plugin: {plugin_name}")
        return plugin_instance
    
    async def unload(self, plugin_name: str):
        """Unload a plugin"""
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return
        
        plugin_instance = self.loaded_plugins[plugin_name]
        
        # Shutdown if has method
        if hasattr(plugin_instance, 'shutdown'):
            if asyncio.iscoroutinefunction(plugin_instance.shutdown):
                await plugin_instance.shutdown()
            else:
                plugin_instance.shutdown()
        
        del self.loaded_plugins[plugin_name]
        
        # Deactivate plugin
        await self.installer.deactivate_plugin(plugin_name)
        
        logger.info(f"Unloaded plugin: {plugin_name}")
    
    async def reload(self, plugin_name: str) -> Any:
        """Reload a plugin"""
        config = None
        
        if plugin_name in self.installer.installed_plugins:
            config = self.installer.installed_plugins[plugin_name].config
        
        await self.unload(plugin_name)
        return await self.load(plugin_name)
    
    def list_installed(self) -> Dict[str, InstalledPlugin]:
        """List installed plugins"""
        return self.installer.installed_plugins
    
    def list_loaded(self) -> Dict[str, Any]:
        """List loaded plugins"""
        return self.loaded_plugins
    
    async def search(self, query: str, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for plugins in registry"""
        return await self.registry.search_plugins(query, plugin_type)
    
    def generate(self, plugin_type: str, name: str, output_dir: Optional[Path] = None,
                config: Optional[Dict[str, Any]] = None) -> Path:
        """Generate a new plugin from template"""
        output_dir = output_dir or Path.cwd()
        return self.generator.generate_plugin(plugin_type, name, output_dir, config)
    
    def register_hook(self, hook: PluginLifecycleHook, callback: Callable):
        """Register a lifecycle hook"""
        self.hooks[hook.value].append(callback)
    
    async def execute_hook(self, hook: PluginLifecycleHook, *args, **kwargs):
        """Execute lifecycle hooks"""
        for callback in self.hooks[hook.value]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook {hook.value} failed: {e}")
    
    def get_plugin(self, plugin_name: str) -> Any:
        """Get a loaded plugin instance"""
        return self.loaded_plugins.get(plugin_name)
    
    def validate_manifest(self, manifest_path: Path) -> bool:
        """Validate a plugin manifest"""
        try:
            metadata = PluginMetadata.from_manifest(manifest_path)
            
            # Check required fields
            required = ["name", "version", "author", "plugin_type", "entry_point"]
            for field in required:
                if not getattr(metadata, field):
                    logger.error(f"Missing required field: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Invalid manifest: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown plugin manager"""
        # Unload all plugins
        for plugin_name in list(self.loaded_plugins.keys()):
            await self.unload(plugin_name)
        
        logger.info("Plugin manager shutdown complete")