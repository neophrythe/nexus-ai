"""Enhanced Plugin System Module for Nexus Framework"""

from nexus.plugins.plugin_system import (
    # Core components
    EnhancedPluginManager,
    PluginInstaller,
    PluginRegistry,
    PluginGenerator,
    
    # Data classes
    PluginMetadata,
    InstalledPlugin,
    
    # Enums
    PluginSource,
    PluginLifecycleHook
)

# Import existing plugin manager for compatibility
from nexus.core.plugin_manager import PluginManager
from nexus.core.base import (
    BasePlugin,
    GamePlugin,
    AgentPlugin,
    CapturePlugin,
    VisionPlugin,
    InputPlugin,
    PluginType,
    PluginStatus,
    PluginManifest
)

__all__ = [
    # Enhanced system
    "EnhancedPluginManager",
    "PluginInstaller",
    "PluginRegistry",
    "PluginGenerator",
    "PluginMetadata",
    "InstalledPlugin",
    "PluginSource",
    "PluginLifecycleHook",
    
    # Existing system
    "PluginManager",
    "BasePlugin",
    "GamePlugin",
    "AgentPlugin",
    "CapturePlugin",
    "VisionPlugin",
    "InputPlugin",
    "PluginType",
    "PluginStatus",
    "PluginManifest"
]