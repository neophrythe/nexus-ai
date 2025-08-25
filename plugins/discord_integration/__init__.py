"""
Discord Integration Plugin

Production-ready Discord integration for game notifications and events.
"""

from plugins.discord_integration.plugin import DiscordIntegrationPlugin, create_plugin
from plugins.discord_integration.config import (
    DiscordConfig, WebhookConfig, BotConfig,
    EventFilters, EmbedStyle, NotificationLevel
)
from plugins.discord_integration.discord_client import (
    DiscordClient, WebhookClient, BotClient
)

__all__ = [
    'DiscordIntegrationPlugin',
    'DiscordConfig',
    'WebhookConfig',
    'BotConfig',
    'EventFilters',
    'EmbedStyle',
    'NotificationLevel',
    'DiscordClient',
    'WebhookClient',
    'BotClient',
    'create_plugin'
]

__version__ = '1.0.0'