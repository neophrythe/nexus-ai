"""
Discord Integration Plugin for Nexus Game AI Framework

Sends game events, achievements, and metrics to Discord.
"""

import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import numpy as np
from PIL import Image
import structlog
from pathlib import Path

from nexus.core.plugin_base import PluginBase
from plugins.discord_integration.config import (
    DiscordConfig, WebhookConfig, BotConfig,
    EventFilters, EmbedStyle, NotificationLevel
)
from plugins.discord_integration.discord_client import DiscordClient

logger = structlog.get_logger()


class DiscordIntegrationPlugin(PluginBase):
    """
    Production-ready Discord integration plugin with:
    - Webhook and bot support
    - Rate limiting and batching
    - Screenshot compression and upload
    - Rich embeds and notifications
    - Session tracking and statistics
    - Performance monitoring
    - Leaderboard updates
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Discord Integration"
        self.version = "1.0.0"
        self.description = "Send game events and notifications to Discord"
        
        # Configuration
        self.discord_config: Optional[DiscordConfig] = None
        self.discord_client: Optional[DiscordClient] = None
        
        # Session tracking
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.achievements = 0
        self.levels_completed = 0
        self.bosses_defeated = 0
        self.high_scores = 0
        self.deaths = 0
        self.milestones = 0
        
        # Screenshot settings
        self.last_screenshot_time = 0
        self.screenshot_interval = 60  # seconds
        self.screenshot_on_achievement = True
        
        # Performance tracking
        self.last_performance_update = 0
        self.performance_update_interval = 300  # 5 minutes
        
        # Leaderboard
        self.last_leaderboard_update = 0
        self.leaderboard_data: List[Dict] = []
        
    def on_load(self):
        """Initialize and start Discord integration."""
        logger.info(f"Loading {self.name} v{self.version}")
        
        try:
            # Build configuration from plugin config
            self.discord_config = self._build_config()
            
            if not self.discord_config.enabled:
                logger.info("Discord integration disabled in config")
                return
            
            # Validate configuration
            errors = self.discord_config.validate()
            if errors:
                logger.error(f"Discord config validation errors: {errors}")
                return
            
            # Create Discord client
            self.discord_client = DiscordClient(self.discord_config)
            
            # Set game callbacks for bot commands
            self.discord_client.set_game_callbacks({
                'get_stats': self.get_session_stats,
                'take_screenshot': self.capture_screenshot,
                'pause_game': self.pause_game,
                'resume_game': self.resume_game,
                'get_leaderboard': self.get_leaderboard,
                'get_activity': self.get_activity
            })
            
            # Start client
            self.discord_client.start()
            
            # Send session start notification
            game_name = self.config.get('game_name', 'Unknown Game')
            self.discord_client.send_session_start(game_name, self.session_id)
            
            logger.info("Discord integration started successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Discord integration: {e}")
        
    def on_unload(self):
        """Clean up and send session summary."""
        if self.discord_client:
            # Send session end summary
            duration = time.time() - self.session_start_time
            stats = self.get_session_stats()
            self.discord_client.send_session_end(duration, stats)
            
            # Close client
            self.discord_client.close()
        
        logger.info(f"Unloaded {self.name}")
    
    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and handle screenshot capture."""
        # Check if we should take a screenshot
        current_time = time.time()
        
        if self.discord_client and self.discord_config.event_filters.screenshots:
            if current_time - self.last_screenshot_time >= self.screenshot_interval:
                self._capture_and_send_screenshot(frame)
                self.last_screenshot_time = current_time
        
        return frame
    
    def on_achievement(self, achievement: Dict[str, Any]):
        """Handle achievement unlocked."""
        if not self.discord_client:
            return
        
        self.achievements += 1
        
        # Check rarity filter
        rarity_levels = ['common', 'uncommon', 'rare', 'epic', 'legendary']
        min_rarity = self.discord_config.event_filters.min_achievement_rarity
        achievement_rarity = achievement.get('rarity', 'common').lower()
        
        if achievement_rarity in rarity_levels:
            if rarity_levels.index(achievement_rarity) < rarity_levels.index(min_rarity):
                return
        
        # Send achievement notification
        self.discord_client.send_event('achievement', {
            'title': '🏆 Achievement Unlocked!',
            'description': achievement.get('name', 'Unknown Achievement'),
            'fields': [
                {'name': 'Description', 'value': achievement.get('description', 'N/A'), 'inline': False},
                {'name': 'Points', 'value': str(achievement.get('points', 0)), 'inline': True},
                {'name': 'Rarity', 'value': achievement_rarity.title(), 'inline': True}
            ]
        })
        
        # Take screenshot if enabled
        if self.screenshot_on_achievement:
            self._request_screenshot()
    
    def on_level_complete(self, level_data: Dict[str, Any]):
        """Handle level completion."""
        if not self.discord_client:
            return
        
        self.levels_completed += 1
        
        self.discord_client.send_event('level_complete', {
            'title': '✅ Level Complete!',
            'description': f"Level {level_data.get('name', 'Unknown')} completed",
            'fields': [
                {'name': 'Time', 'value': level_data.get('time', 'N/A'), 'inline': True},
                {'name': 'Score', 'value': str(level_data.get('score', 0)), 'inline': True},
                {'name': 'Collectibles', 'value': f"{level_data.get('collected', 0)}/{level_data.get('total', 0)}", 'inline': True}
            ]
        })
    
    def on_boss_defeat(self, boss_data: Dict[str, Any]):
        """Handle boss defeat."""
        if not self.discord_client:
            return
        
        self.bosses_defeated += 1
        
        self.discord_client.send_event('boss_defeat', {
            'title': '⚔️ Boss Defeated!',
            'description': boss_data.get('name', 'Unknown Boss'),
            'fields': [
                {'name': 'Battle Time', 'value': boss_data.get('time', 'N/A'), 'inline': True},
                {'name': 'Attempts', 'value': str(boss_data.get('attempts', 1)), 'inline': True},
                {'name': 'Damage Dealt', 'value': str(boss_data.get('damage', 0)), 'inline': True}
            ],
            'immediate': True  # Don't batch boss defeats
        })
    
    def on_high_score(self, score_data: Dict[str, Any]):
        """Handle new high score."""
        if not self.discord_client:
            return
        
        self.high_scores += 1
        
        # Check minimum improvement
        improvement = score_data.get('score', 0) - score_data.get('previous', 0)
        if improvement < self.discord_config.event_filters.min_score_improvement:
            return
        
        self.discord_client.send_event('high_score', {
            'title': '🌟 New High Score!',
            'description': f"Score: {score_data.get('score', 0):,}",
            'fields': [
                {'name': 'Previous Best', 'value': f"{score_data.get('previous', 0):,}", 'inline': True},
                {'name': 'Improvement', 'value': f"+{improvement:,}", 'inline': True},
                {'name': 'Rank', 'value': score_data.get('rank', 'N/A'), 'inline': True}
            ]
        })
    
    def on_death(self, death_data: Dict[str, Any]):
        """Handle player death."""
        if not self.discord_client:
            return
        
        self.deaths += 1
        
        # Check death threshold
        if self.deaths % self.discord_config.event_filters.death_notification_threshold == 0:
            self.discord_client.send_event('death', {
                'title': '💀 Multiple Deaths',
                'description': f"Died {self.deaths} times",
                'fields': [
                    {'name': 'Last Cause', 'value': death_data.get('cause', 'Unknown'), 'inline': True},
                    {'name': 'Location', 'value': death_data.get('location', 'Unknown'), 'inline': True}
                ]
            })
    
    def on_milestone(self, milestone_data: Dict[str, Any]):
        """Handle milestone reached."""
        if not self.discord_client:
            return
        
        self.milestones += 1
        
        self.discord_client.send_event('milestone', {
            'title': '🎉 Milestone Reached!',
            'description': milestone_data.get('name', 'Milestone'),
            'fields': [
                {'name': 'Type', 'value': milestone_data.get('type', 'General'), 'inline': True},
                {'name': 'Progress', 'value': milestone_data.get('progress', 'N/A'), 'inline': True}
            ]
        })
    
    def on_performance_update(self, metrics: Dict[str, Any]):
        """Handle performance metrics update."""
        if not self.discord_client or not self.discord_config.event_filters.performance_updates:
            return
        
        current_time = time.time()
        if current_time - self.last_performance_update < self.performance_update_interval:
            return
        
        self.last_performance_update = current_time
        
        self.discord_client.send_event('performance', {
            'title': '📊 Performance Update',
            'description': 'Current system metrics',
            'fields': [
                {'name': 'FPS', 'value': f"{metrics.get('fps', 0):.1f}", 'inline': True},
                {'name': 'CPU', 'value': f"{metrics.get('cpu', 0):.1f}%", 'inline': True},
                {'name': 'Memory', 'value': f"{metrics.get('memory', 0):.1f}%", 'inline': True},
                {'name': 'GPU', 'value': f"{metrics.get('gpu', 0):.1f}%", 'inline': True}
            ]
        })
    
    def on_speedrun_split(self, split_data: Dict[str, Any]):
        """Handle speedrun split."""
        if not self.discord_client or not self.discord_config.event_filters.speedrun_splits:
            return
        
        # Format time difference
        diff = split_data.get('time', 0) - split_data.get('best', 0)
        diff_str = f"+{diff:.2f}s" if diff > 0 else f"{diff:.2f}s"
        color = 0xFF0000 if diff > 0 else 0x00FF00  # Red if behind, green if ahead
        
        self.discord_client.send_event('speedrun_split', {
            'title': '⏱️ Split',
            'description': split_data.get('name', 'Split'),
            'fields': [
                {'name': 'Time', 'value': f"{split_data.get('time', 0):.2f}s", 'inline': True},
                {'name': 'Best', 'value': f"{split_data.get('best', 0):.2f}s", 'inline': True},
                {'name': 'Difference', 'value': diff_str, 'inline': True}
            ],
            'color': color
        })
    
    
    def _build_config(self) -> DiscordConfig:
        """Build Discord configuration from plugin config."""
        config = DiscordConfig()
        
        # Check if Discord is enabled
        config.enabled = self.config.get('enabled', True)
        
        # Set connection type
        webhook_url = self.config.get('webhook_url')
        bot_token = self.config.get('bot_token')
        
        if webhook_url and bot_token:
            config.connection_type = 'both'
        elif webhook_url:
            config.connection_type = 'webhook'
        elif bot_token:
            config.connection_type = 'bot'
        else:
            config.enabled = False
            return config
        
        # Configure webhook
        if webhook_url:
            config.webhook = WebhookConfig(
                url=webhook_url,
                username=self.config.get('username', 'Nexus AI'),
                avatar_url=self.config.get('avatar_url'),
                rate_limit_per_minute=self.config.get('rate_limit', 30)
            )
        
        # Configure bot
        if bot_token:
            config.bot = BotConfig(
                token=bot_token,
                channel_id=self.config.get('channel_id'),
                guild_id=self.config.get('guild_id'),
                command_prefix=self.config.get('command_prefix', '!'),
                enable_rich_presence=self.config.get('rich_presence', True)
            )
        
        # Configure event filters
        config.event_filters.achievements = self.config.get('send_achievements', True)
        config.event_filters.level_completion = self.config.get('send_level_complete', True)
        config.event_filters.boss_defeats = self.config.get('send_boss_defeats', True)
        config.event_filters.high_scores = self.config.get('send_high_scores', True)
        config.event_filters.deaths = self.config.get('send_deaths', False)
        config.event_filters.milestones = self.config.get('send_milestones', True)
        config.event_filters.performance_updates = self.config.get('send_performance', False)
        config.event_filters.screenshots = self.config.get('send_screenshots', True)
        config.event_filters.speedrun_splits = self.config.get('send_speedrun_splits', True)
        
        # Configure batching
        config.batch_messages = self.config.get('batch_messages', True)
        config.batch_interval = self.config.get('batch_interval', 5.0)
        config.max_batch_size = self.config.get('max_batch_size', 10)
        
        # Configure session tracking
        config.send_session_start = self.config.get('send_session_start', True)
        config.send_session_end = self.config.get('send_session_summary', True)
        
        # Configure leaderboards
        config.enable_leaderboards = self.config.get('enable_leaderboards', True)
        config.leaderboard_size = self.config.get('leaderboard_size', 10)
        
        # Apply preset if specified
        preset = self.config.get('preset')
        if preset:
            try:
                config.apply_preset(preset)
            except ValueError:
                logger.warning(f"Unknown Discord preset: {preset}")
        
        return config
    
    def _capture_and_send_screenshot(self, frame: np.ndarray):
        """Capture and send screenshot to Discord."""
        try:
            # Convert numpy array to PIL Image
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB image
                image = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
            else:
                # Grayscale or other format
                image = Image.fromarray(frame)
            
            # Send screenshot
            caption = f"Screenshot from {self.config.get('game_name', 'Game')}"
            self.discord_client.send_screenshot(image, caption)
            
        except Exception as e:
            logger.error(f"Failed to send screenshot: {e}")
    
    def _request_screenshot(self):
        """Request screenshot on next frame."""
        self.last_screenshot_time = 0  # Force screenshot on next frame
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        duration = time.time() - self.session_start_time
        return {
            'duration': duration,
            'achievements': self.achievements,
            'levels_completed': self.levels_completed,
            'bosses_defeated': self.bosses_defeated,
            'high_scores': self.high_scores,
            'deaths': self.deaths,
            'milestones': self.milestones
        }
    
    def capture_screenshot(self) -> Optional[Image.Image]:
        """Capture current game screenshot."""
        # This would capture the current frame
        # For now, return None (would be implemented by game)
        return None
    
    def pause_game(self):
        """Pause the game."""
        # Would be implemented by game
        logger.info("Game pause requested via Discord")
    
    def resume_game(self):
        """Resume the game."""
        # Would be implemented by game
        logger.info("Game resume requested via Discord")
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current leaderboard data."""
        return self.leaderboard_data
    
    def update_leaderboard(self, scores: List[Dict[str, Any]]):
        """Update and send leaderboard."""
        self.leaderboard_data = sorted(
            scores,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:self.discord_config.leaderboard_size]
        
        current_time = time.time()
        if current_time - self.last_leaderboard_update >= self.discord_config.leaderboard_update_interval:
            self.discord_client.send_leaderboard(self.leaderboard_data)
            self.last_leaderboard_update = current_time
    
    def get_activity(self) -> Dict[str, str]:
        """Get current activity for rich presence."""
        return {
            'game': self.config.get('game_name', 'Unknown Game'),
            'details': f"Session: {self.session_id}",
            'state': f"Achievements: {self.achievements} | Deaths: {self.deaths}"
        }


# Plugin registration
def create_plugin() -> DiscordIntegrationPlugin:
    """Create plugin instance."""
    return DiscordIntegrationPlugin()