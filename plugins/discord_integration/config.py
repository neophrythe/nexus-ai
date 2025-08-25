"""
Configuration schema for Discord Integration Plugin.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import re
from pathlib import Path


class NotificationLevel(Enum):
    """Notification importance levels."""
    ALL = "all"
    IMPORTANT = "important"
    CRITICAL = "critical"
    NONE = "none"


class ComparisonMode(Enum):
    """Comparison modes for speedruns."""
    PERSONAL_BEST = "pb"
    WORLD_RECORD = "wr"
    AVERAGE = "average"
    CUSTOM = "custom"


@dataclass
class WebhookConfig:
    """Discord webhook configuration."""
    url: str
    username: str = "Nexus AI"
    avatar_url: Optional[str] = None
    rate_limit_per_minute: int = 30
    retry_on_rate_limit: bool = True
    max_retries: int = 3
    
    def validate(self) -> List[str]:
        """Validate webhook configuration."""
        errors = []
        
        # Validate webhook URL format
        webhook_pattern = r'^https://discord(app)?\.com/api/webhooks/\d+/[\w-]+$'
        if not re.match(webhook_pattern, self.url):
            errors.append("Invalid Discord webhook URL format")
        
        # Validate rate limit
        if not 1 <= self.rate_limit_per_minute <= 60:
            errors.append("rate_limit_per_minute must be between 1 and 60")
        
        # Validate retries
        if not 0 <= self.max_retries <= 10:
            errors.append("max_retries must be between 0 and 10")
        
        return errors


@dataclass
class BotConfig:
    """Discord bot configuration."""
    token: str
    application_id: Optional[str] = None
    guild_id: Optional[str] = None
    channel_id: Optional[str] = None
    command_prefix: str = "!"
    enable_rich_presence: bool = True
    status_update_interval: float = 60.0
    
    def validate(self) -> List[str]:
        """Validate bot configuration."""
        errors = []
        
        # Validate token format (basic check)
        if not self.token or len(self.token) < 50:
            errors.append("Invalid bot token")
        
        # Validate IDs if provided
        if self.guild_id and not self.guild_id.isdigit():
            errors.append("guild_id must be numeric")
        
        if self.channel_id and not self.channel_id.isdigit():
            errors.append("channel_id must be numeric")
        
        # Validate command prefix
        if not self.command_prefix or len(self.command_prefix) > 5:
            errors.append("command_prefix must be 1-5 characters")
        
        return errors


@dataclass
class EventFilters:
    """Event filtering configuration."""
    # Event types to send
    achievements: bool = True
    level_completion: bool = True
    boss_defeats: bool = True
    high_scores: bool = True
    deaths: bool = False
    milestones: bool = True
    performance_updates: bool = False
    screenshots: bool = True
    speedrun_splits: bool = True
    
    # Filtering rules
    min_achievement_rarity: str = "common"  # common, uncommon, rare, epic, legendary
    min_score_improvement: int = 100  # Minimum score improvement to notify
    death_notification_threshold: int = 5  # Deaths before notification
    performance_interval: float = 300.0  # Seconds between performance updates
    
    # Screenshot settings
    screenshot_quality: int = 80  # JPEG quality (1-100)
    screenshot_max_size: int = 8388608  # 8MB Discord limit
    screenshot_format: str = "jpeg"  # jpeg, png, webp
    
    def validate(self) -> List[str]:
        """Validate event filters."""
        errors = []
        
        # Validate rarity
        valid_rarities = ['common', 'uncommon', 'rare', 'epic', 'legendary']
        if self.min_achievement_rarity not in valid_rarities:
            errors.append(f"min_achievement_rarity must be one of {valid_rarities}")
        
        # Validate screenshot settings
        if not 1 <= self.screenshot_quality <= 100:
            errors.append("screenshot_quality must be between 1 and 100")
        
        if self.screenshot_max_size > 8388608:
            errors.append("screenshot_max_size cannot exceed 8MB (Discord limit)")
        
        valid_formats = ['jpeg', 'png', 'webp']
        if self.screenshot_format not in valid_formats:
            errors.append(f"screenshot_format must be one of {valid_formats}")
        
        return errors


@dataclass
class EmbedStyle:
    """Discord embed styling configuration."""
    # Color scheme (hex values)
    colors: Dict[str, int] = field(default_factory=lambda: {
        'achievement': 0xFFD700,  # Gold
        'level_complete': 0x00FF00,  # Green
        'boss_defeat': 0xFF0000,  # Red
        'high_score': 0x9B59B6,  # Purple
        'death': 0x000000,  # Black
        'milestone': 0x3498DB,  # Blue
        'performance': 0x2ECC71,  # Emerald
        'info': 0x95A5A6,  # Gray
        'warning': 0xF39C12,  # Orange
        'error': 0xE74C3C  # Red
    })
    
    # Embed settings
    show_timestamp: bool = True
    show_footer: bool = True
    footer_text: str = "Nexus AI Gaming Framework"
    show_thumbnail: bool = True
    show_author: bool = True
    author_name: str = "Nexus AI"
    author_icon_url: Optional[str] = None
    
    # Field settings
    inline_fields: bool = True
    max_fields: int = 25  # Discord limit
    field_value_max_length: int = 1024  # Discord limit
    
    def validate(self) -> List[str]:
        """Validate embed style."""
        errors = []
        
        # Validate colors
        for name, color in self.colors.items():
            if not 0 <= color <= 0xFFFFFF:
                errors.append(f"Color '{name}' must be a valid hex value (0-0xFFFFFF)")
        
        # Validate field limits
        if not 1 <= self.max_fields <= 25:
            errors.append("max_fields must be between 1 and 25 (Discord limit)")
        
        if self.field_value_max_length > 1024:
            errors.append("field_value_max_length cannot exceed 1024 (Discord limit)")
        
        return errors


@dataclass
class DiscordConfig:
    """Complete Discord Integration configuration."""
    
    # Connection settings
    enabled: bool = True
    connection_type: str = "webhook"  # webhook, bot, both
    
    # Webhook configuration
    webhook: Optional[WebhookConfig] = None
    
    # Bot configuration
    bot: Optional[BotConfig] = None
    
    # Event configuration
    event_filters: EventFilters = field(default_factory=EventFilters)
    
    # Styling
    embed_style: EmbedStyle = field(default_factory=EmbedStyle)
    
    # Notification settings
    notification_level: NotificationLevel = NotificationLevel.IMPORTANT
    batch_messages: bool = True
    batch_interval: float = 5.0  # Seconds to batch messages
    max_batch_size: int = 10
    
    # Leaderboard settings
    enable_leaderboards: bool = True
    leaderboard_size: int = 10
    leaderboard_update_interval: float = 300.0  # 5 minutes
    
    # Session settings
    send_session_start: bool = True
    send_session_end: bool = True
    session_summary_stats: List[str] = field(default_factory=lambda: [
        'duration', 'achievements', 'high_scores', 'deaths', 'levels_completed'
    ])
    
    # Debug settings
    debug_mode: bool = False
    log_discord_errors: bool = True
    dry_run: bool = False  # Don't actually send messages
    
    @classmethod
    def from_file(cls, filepath: str) -> 'DiscordConfig':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Parse webhook config
        webhook = None
        if 'webhook' in data and data['webhook']:
            webhook = WebhookConfig(**data['webhook'])
        
        # Parse bot config
        bot = None
        if 'bot' in data and data['bot']:
            bot = BotConfig(**data['bot'])
        
        # Parse event filters
        event_filters = EventFilters()
        if 'event_filters' in data:
            for key, value in data['event_filters'].items():
                if hasattr(event_filters, key):
                    setattr(event_filters, key, value)
        
        # Parse embed style
        embed_style = EmbedStyle()
        if 'embed_style' in data:
            for key, value in data['embed_style'].items():
                if hasattr(embed_style, key):
                    setattr(embed_style, key, value)
        
        # Create config
        config = cls(
            enabled=data.get('enabled', True),
            connection_type=data.get('connection_type', 'webhook'),
            webhook=webhook,
            bot=bot,
            event_filters=event_filters,
            embed_style=embed_style
        )
        
        # Set remaining fields
        for key in ['notification_level', 'batch_messages', 'batch_interval',
                    'max_batch_size', 'enable_leaderboards', 'leaderboard_size',
                    'send_session_start', 'send_session_end', 'debug_mode',
                    'log_discord_errors', 'dry_run']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_file(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'enabled': self.enabled,
            'connection_type': self.connection_type,
            'notification_level': self.notification_level.value,
            'batch_messages': self.batch_messages,
            'batch_interval': self.batch_interval,
            'max_batch_size': self.max_batch_size,
            'enable_leaderboards': self.enable_leaderboards,
            'leaderboard_size': self.leaderboard_size,
            'send_session_start': self.send_session_start,
            'send_session_end': self.send_session_end,
            'session_summary_stats': self.session_summary_stats,
            'debug_mode': self.debug_mode,
            'log_discord_errors': self.log_discord_errors,
            'dry_run': self.dry_run
        }
        
        if self.webhook:
            data['webhook'] = {
                'url': self.webhook.url,
                'username': self.webhook.username,
                'avatar_url': self.webhook.avatar_url,
                'rate_limit_per_minute': self.webhook.rate_limit_per_minute,
                'retry_on_rate_limit': self.webhook.retry_on_rate_limit,
                'max_retries': self.webhook.max_retries
            }
        
        if self.bot:
            data['bot'] = {
                'token': self.bot.token,
                'application_id': self.bot.application_id,
                'guild_id': self.bot.guild_id,
                'channel_id': self.bot.channel_id,
                'command_prefix': self.bot.command_prefix,
                'enable_rich_presence': self.bot.enable_rich_presence
            }
        
        # Save event filters and embed style
        data['event_filters'] = self.event_filters.__dict__
        data['embed_style'] = {
            'colors': self.embed_style.colors,
            'show_timestamp': self.embed_style.show_timestamp,
            'show_footer': self.embed_style.show_footer,
            'footer_text': self.embed_style.footer_text,
            'inline_fields': self.embed_style.inline_fields
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        # Check connection type
        valid_types = ['webhook', 'bot', 'both']
        if self.connection_type not in valid_types:
            errors.append(f"connection_type must be one of {valid_types}")
        
        # Validate webhook if needed
        if self.connection_type in ['webhook', 'both']:
            if not self.webhook:
                errors.append("Webhook configuration required for webhook connection")
            else:
                errors.extend(self.webhook.validate())
        
        # Validate bot if needed
        if self.connection_type in ['bot', 'both']:
            if not self.bot:
                errors.append("Bot configuration required for bot connection")
            else:
                errors.extend(self.bot.validate())
        
        # Validate event filters
        errors.extend(self.event_filters.validate())
        
        # Validate embed style
        errors.extend(self.embed_style.validate())
        
        # Validate batch settings
        if self.batch_messages:
            if not 0.1 <= self.batch_interval <= 60:
                errors.append("batch_interval must be between 0.1 and 60 seconds")
            
            if not 1 <= self.max_batch_size <= 10:
                errors.append("max_batch_size must be between 1 and 10")
        
        # Validate leaderboard settings
        if self.enable_leaderboards:
            if not 1 <= self.leaderboard_size <= 25:
                errors.append("leaderboard_size must be between 1 and 25")
        
        return errors
    
    def apply_preset(self, preset: str):
        """Apply configuration preset."""
        presets = {
            'minimal': {
                'notification_level': NotificationLevel.CRITICAL,
                'event_filters': {
                    'achievements': True,
                    'high_scores': True,
                    'deaths': False,
                    'performance_updates': False,
                    'screenshots': False
                },
                'batch_messages': True,
                'send_session_start': False,
                'send_session_end': True
            },
            'standard': {
                'notification_level': NotificationLevel.IMPORTANT,
                'event_filters': {
                    'achievements': True,
                    'level_completion': True,
                    'boss_defeats': True,
                    'high_scores': True,
                    'deaths': False,
                    'milestones': True,
                    'screenshots': True
                },
                'batch_messages': True,
                'enable_leaderboards': True
            },
            'verbose': {
                'notification_level': NotificationLevel.ALL,
                'event_filters': {
                    'achievements': True,
                    'level_completion': True,
                    'boss_defeats': True,
                    'high_scores': True,
                    'deaths': True,
                    'milestones': True,
                    'performance_updates': True,
                    'screenshots': True,
                    'speedrun_splits': True
                },
                'batch_messages': False,
                'enable_leaderboards': True,
                'debug_mode': True
            },
            'speedrun': {
                'notification_level': NotificationLevel.IMPORTANT,
                'event_filters': {
                    'achievements': False,
                    'level_completion': False,
                    'boss_defeats': False,
                    'high_scores': False,
                    'deaths': True,
                    'milestones': True,
                    'speedrun_splits': True
                },
                'batch_messages': False,
                'enable_leaderboards': True,
                'leaderboard_update_interval': 60.0
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        preset_data = presets[preset]
        
        # Apply notification level
        if 'notification_level' in preset_data:
            self.notification_level = preset_data['notification_level']
        
        # Apply event filters
        if 'event_filters' in preset_data:
            for key, value in preset_data['event_filters'].items():
                if hasattr(self.event_filters, key):
                    setattr(self.event_filters, key, value)
        
        # Apply other settings
        for key in ['batch_messages', 'enable_leaderboards', 'debug_mode',
                    'send_session_start', 'send_session_end',
                    'leaderboard_update_interval']:
            if key in preset_data:
                setattr(self, key, preset_data[key])