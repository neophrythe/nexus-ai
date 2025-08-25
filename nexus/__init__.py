"""
Nexus Game Automation Framework
A modern, modular game automation framework with AI integration
"""

from nexus.core import (
    PluginManager,
    BasePlugin,
    PluginType,
    ConfigManager,
    get_logger,
)

from nexus.capture import (
    CaptureManager,
    DXCamBackend,
    Frame,
)

from nexus.environments import (
    GameEnvironment,
    GameState,
)

from nexus.agents import (
    BaseAgent,
    AgentType,
)

__version__ = "0.1.0"
__author__ = "Nexus Team"

__all__ = [
    "PluginManager",
    "BasePlugin",
    "PluginType",
    "ConfigManager",
    "get_logger",
    "CaptureManager",
    "DXCamBackend",
    "Frame",
    "GameEnvironment",
    "GameState",
    "BaseAgent",
    "AgentType",
]