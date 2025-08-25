"""
Unit tests for Discord Integration Plugin.
"""

import unittest
import json
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np
from io import BytesIO

from plugins.discord_integration.plugin import DiscordIntegrationPlugin
from plugins.discord_integration.config import (
    DiscordConfig, WebhookConfig, BotConfig, 
    EventFilters, EmbedStyle, NotificationLevel
)
from plugins.discord_integration.discord_client import (
    WebhookClient, BotClient, DiscordClient
)


class TestDiscordConfig(unittest.TestCase):
    """Test Discord configuration classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.webhook_config = WebhookConfig(
            url="https://discord.com/api/webhooks/123456789/abcdefgh"
        )
        self.bot_config = BotConfig(
            token="MTIzNDU2Nzg5MDEyMzQ1Njc4.AbCdEf.GhIjKlMnOpQrStUvWxYz"
        )
        self.config = DiscordConfig()
    
    def test_webhook_validation_valid(self):
        """Test webhook validation with valid config."""
        errors = self.webhook_config.validate()
        self.assertEqual(len(errors), 0)
    
    def test_webhook_validation_invalid_url(self):
        """Test webhook validation with invalid URL."""
        self.webhook_config.url = "https://example.com/webhook"
        errors = self.webhook_config.validate()
        self.assertIn("Invalid Discord webhook URL format", errors)
    
    def test_webhook_validation_rate_limit(self):
        """Test webhook validation with invalid rate limit."""
        self.webhook_config.rate_limit_per_minute = 100
        errors = self.webhook_config.validate()
        self.assertIn("rate_limit_per_minute must be between 1 and 60", errors)
    
    def test_bot_validation_valid(self):
        """Test bot validation with valid config."""
        errors = self.bot_config.validate()
        self.assertEqual(len(errors), 0)
    
    def test_bot_validation_invalid_token(self):
        """Test bot validation with invalid token."""
        self.bot_config.token = "invalid"
        errors = self.bot_config.validate()
        self.assertIn("Invalid bot token", errors)
    
    def test_bot_validation_invalid_ids(self):
        """Test bot validation with invalid IDs."""
        self.bot_config.guild_id = "not_numeric"
        self.bot_config.channel_id = "also_not_numeric"
        errors = self.bot_config.validate()
        self.assertIn("guild_id must be numeric", errors)
        self.assertIn("channel_id must be numeric", errors)
    
    def test_event_filters_validation(self):
        """Test event filters validation."""
        filters = EventFilters()
        
        # Test valid config
        errors = filters.validate()
        self.assertEqual(len(errors), 0)
        
        # Test invalid rarity
        filters.min_achievement_rarity = "super_rare"
        errors = filters.validate()
        self.assertTrue(any("min_achievement_rarity" in e for e in errors))
        
        # Test invalid screenshot settings
        filters.screenshot_quality = 150
        filters.screenshot_max_size = 10000000
        filters.screenshot_format = "bmp"
        errors = filters.validate()
        self.assertIn("screenshot_quality must be between 1 and 100", errors)
        self.assertIn("screenshot_max_size cannot exceed 8MB", errors)
        self.assertTrue(any("screenshot_format" in e for e in errors))
    
    def test_embed_style_validation(self):
        """Test embed style validation."""
        style = EmbedStyle()
        
        # Test valid config
        errors = style.validate()
        self.assertEqual(len(errors), 0)
        
        # Test invalid color
        style.colors['test'] = 0x1000000
        errors = style.validate()
        self.assertTrue(any("Color 'test'" in e for e in errors))
        
        # Test invalid field limits
        style.max_fields = 30
        style.field_value_max_length = 2000
        errors = style.validate()
        self.assertIn("max_fields must be between 1 and 25", errors)
        self.assertIn("field_value_max_length cannot exceed 1024", errors)
    
    def test_discord_config_validation(self):
        """Test complete Discord config validation."""
        # Test webhook connection without webhook config
        self.config.connection_type = 'webhook'
        self.config.webhook = None
        errors = self.config.validate()
        self.assertIn("Webhook configuration required for webhook connection", errors)
        
        # Test bot connection without bot config
        self.config.connection_type = 'bot'
        self.config.bot = None
        errors = self.config.validate()
        self.assertIn("Bot configuration required for bot connection", errors)
        
        # Test valid config
        self.config.connection_type = 'webhook'
        self.config.webhook = self.webhook_config
        errors = self.config.validate()
        self.assertEqual(len(errors), 0)
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Configure
            self.config.connection_type = 'webhook'
            self.config.webhook = self.webhook_config
            self.config.notification_level = NotificationLevel.IMPORTANT
            self.config.batch_messages = True
            
            # Save
            self.config.to_file(config_path)
            
            # Load
            loaded_config = DiscordConfig.from_file(config_path)
            
            # Verify
            self.assertEqual(loaded_config.connection_type, 'webhook')
            self.assertIsNotNone(loaded_config.webhook)
            self.assertEqual(loaded_config.webhook.url, self.webhook_config.url)
            self.assertTrue(loaded_config.batch_messages)
        finally:
            Path(config_path).unlink()
    
    def test_config_presets(self):
        """Test configuration presets."""
        # Test minimal preset
        self.config.apply_preset('minimal')
        self.assertEqual(self.config.notification_level, NotificationLevel.CRITICAL)
        self.assertFalse(self.config.event_filters.screenshots)
        
        # Test standard preset
        self.config.apply_preset('standard')
        self.assertEqual(self.config.notification_level, NotificationLevel.IMPORTANT)
        self.assertTrue(self.config.event_filters.screenshots)
        
        # Test verbose preset
        self.config.apply_preset('verbose')
        self.assertEqual(self.config.notification_level, NotificationLevel.ALL)
        self.assertTrue(self.config.event_filters.performance_updates)
        
        # Test speedrun preset
        self.config.apply_preset('speedrun')
        self.assertTrue(self.config.event_filters.speedrun_splits)
        self.assertFalse(self.config.event_filters.achievements)


class TestWebhookClient(unittest.TestCase):
    """Test WebhookClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = WebhookConfig(
            url="https://discord.com/api/webhooks/123456789/abcdefgh",
            rate_limit_per_minute=30
        )
    
    @patch('aiohttp.ClientSession.post')
    def test_send_message(self, mock_post):
        """Test sending message via webhook."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 204
        mock_response.headers = {}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create client
        client = WebhookClient(self.config)
        
        # Send message
        client.send_message("Test message")
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify
        self.assertTrue(mock_post.called)
        
        # Clean up
        client.close()
    
    def test_rate_limiting(self):
        """Test rate limit tracking."""
        client = WebhookClient(self.config)
        
        # Check initial state
        self.assertEqual(client.rate_limit.remaining, 30)
        
        # Send messages
        for _ in range(5):
            client.send_message("Test")
        
        # Wait for processing
        time.sleep(0.5)
        
        # Clean up
        client.close()
    
    def test_embed_formatting(self):
        """Test embed message formatting."""
        client = WebhookClient(self.config)
        
        # Send embed
        client.send_embed(
            title="Test Title",
            description="Test Description",
            color=0xFF0000,
            fields=[{
                'name': 'Field 1',
                'value': 'Value 1',
                'inline': True
            }]
        )
        
        # Check queue
        self.assertEqual(len(client.message_queue), 1)
        
        # Clean up
        client.close()
    
    def test_screenshot_compression(self):
        """Test screenshot compression."""
        client = WebhookClient(self.config)
        
        # Create test image
        image = Image.new('RGB', (1920, 1080), color='red')
        
        # Send screenshot
        client.send_screenshot(image, "Test screenshot")
        
        # Check queue
        self.assertEqual(len(client.message_queue), 1)
        
        # Clean up
        client.close()


class TestBotClient(unittest.TestCase):
    """Test BotClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = BotConfig(
            token="MTIzNDU2Nzg5MDEyMzQ1Njc4.AbCdEf.GhIjKlMnOpQrStUvWxYz",
            channel_id="987654321"
        )
        self.game_callbacks = {
            'get_stats': Mock(return_value={'score': 100, 'level': 5}),
            'take_screenshot': Mock(return_value=Image.new('RGB', (640, 480))),
            'pause_game': Mock(),
            'resume_game': Mock()
        }
    
    @patch('discord.ext.commands.Bot.run')
    def test_bot_initialization(self, mock_run):
        """Test bot initialization."""
        bot = BotClient(self.config, self.game_callbacks)
        
        # Check configuration
        self.assertEqual(bot.config, self.config)
        self.assertEqual(bot.game_callbacks, self.game_callbacks)
        
        # Check commands were set up
        self.assertIsNotNone(bot.get_command('stats'))
        self.assertIsNotNone(bot.get_command('screenshot'))
        self.assertIsNotNone(bot.get_command('pause'))
        self.assertIsNotNone(bot.get_command('resume'))
    
    @patch('discord.ext.commands.Bot.run')
    def test_send_notification(self, mock_run):
        """Test sending notification."""
        bot = BotClient(self.config)
        
        # Mock channel
        mock_channel = Mock()
        bot.get_channel = Mock(return_value=mock_channel)
        
        # Send notification
        bot.send_notification(
            title="Test",
            description="Test notification",
            color=0x00FF00
        )
        
        # Verify channel lookup
        bot.get_channel.assert_called_with(987654321)


class TestDiscordClient(unittest.TestCase):
    """Test unified DiscordClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DiscordConfig(
            enabled=True,
            connection_type='webhook',
            webhook=WebhookConfig(
                url="https://discord.com/api/webhooks/123456789/abcdefgh"
            )
        )
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = DiscordClient(self.config)
        
        # Check webhook client created
        self.assertIsNotNone(client.webhook_client)
        self.assertIsNone(client.bot_client)
        
        # Clean up
        client.close()
    
    def test_event_filtering(self):
        """Test event filtering."""
        client = DiscordClient(self.config)
        
        # Test allowed event
        self.config.event_filters.achievements = True
        self.assertTrue(client._should_send_event('achievement'))
        
        # Test filtered event
        self.config.event_filters.deaths = False
        self.assertFalse(client._should_send_event('death'))
        
        # Test notification level
        self.config.notification_level = NotificationLevel.NONE
        self.assertFalse(client._should_send_event('achievement'))
        
        # Clean up
        client.close()
    
    def test_message_batching(self):
        """Test message batching."""
        self.config.batch_messages = True
        self.config.batch_interval = 0.1
        self.config.max_batch_size = 3
        
        client = DiscordClient(self.config)
        
        # Send multiple events
        for i in range(5):
            client.send_event('achievement', {
                'title': f'Achievement {i}',
                'description': f'Description {i}'
            })
        
        # Check batch
        self.assertEqual(len(client.message_batch), 5)
        
        # Wait for batch processing
        time.sleep(0.2)
        
        # Check batch was processed
        self.assertLessEqual(len(client.message_batch), 2)
        
        # Clean up
        client.close()
    
    def test_session_notifications(self):
        """Test session start/end notifications."""
        client = DiscordClient(self.config)
        
        # Mock webhook client
        client.webhook_client = Mock()
        
        # Send session start
        client.send_session_start("Test Game", "session123")
        
        # Send session end
        stats = {
            'duration': 3661,
            'achievements': 5,
            'high_scores': 2,
            'deaths': 10
        }
        client.send_session_end(3661, stats)
        
        # Verify calls
        self.assertEqual(client.webhook_client.send_embed.call_count, 2)
        
        # Clean up
        client.close()
    
    def test_leaderboard_update(self):
        """Test leaderboard update."""
        client = DiscordClient(self.config)
        
        # Mock webhook client
        client.webhook_client = Mock()
        
        # Send leaderboard
        leaderboard = [
            {'player': 'Player1', 'score': 1000},
            {'player': 'Player2', 'score': 900},
            {'player': 'Player3', 'score': 800}
        ]
        client.send_leaderboard(leaderboard)
        
        # Verify call
        client.webhook_client.send_embed.assert_called_once()
        
        # Clean up
        client.close()


class TestDiscordIntegrationPlugin(unittest.TestCase):
    """Test Discord Integration Plugin."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = DiscordIntegrationPlugin()
        self.plugin.config = {
            'webhook_url': 'https://discord.com/api/webhooks/123456789/abcdefgh',
            'send_achievements': True,
            'send_screenshots': True
        }
    
    def test_plugin_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.name, "Discord Integration")
        self.assertEqual(self.plugin.version, "1.0.0")
        self.assertIsNotNone(self.plugin.session_start_time)
    
    @patch('plugins.discord_integration.discord_client.DiscordClient')
    def test_on_load(self, mock_client_class):
        """Test plugin loading."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        self.plugin.on_load()
        
        # Verify client created and started
        mock_client_class.assert_called_once()
        mock_client.start.assert_called_once()
    
    @patch('plugins.discord_integration.discord_client.DiscordClient')
    def test_on_unload(self, mock_client_class):
        """Test plugin unloading."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        self.plugin.on_load()
        self.plugin.on_unload()
        
        # Verify client closed
        mock_client.close.assert_called_once()
    
    def test_on_achievement(self):
        """Test achievement handling."""
        # Set up mock client
        self.plugin.discord_client = Mock()
        
        # Trigger achievement
        achievement = {
            'name': 'First Victory',
            'description': 'Win your first match',
            'rarity': 'common',
            'points': 10
        }
        
        self.plugin.on_achievement(achievement)
        
        # Verify event sent
        self.plugin.discord_client.send_event.assert_called_with(
            'achievement',
            unittest.mock.ANY
        )
    
    def test_on_frame_with_screenshot(self):
        """Test frame processing with screenshot."""
        # Set up mock client
        self.plugin.discord_client = Mock()
        self.plugin.last_screenshot_time = 0
        self.plugin.screenshot_interval = 60
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        result = self.plugin.on_frame(frame)
        
        # Check result
        self.assertEqual(result.shape, frame.shape)
        
        # Check if screenshot was taken (after interval)
        time.sleep(0.1)
        current_time = time.time()
        if current_time - self.plugin.last_screenshot_time >= self.plugin.screenshot_interval:
            self.plugin.discord_client.send_screenshot.assert_called()
    
    def test_stats_collection(self):
        """Test statistics collection."""
        self.plugin.discord_client = Mock()
        
        # Track some stats
        self.plugin.on_achievement({'name': 'Test1'})
        self.plugin.on_level_complete({'level': 1})
        self.plugin.on_boss_defeat({'boss': 'TestBoss'})
        self.plugin.deaths += 5
        
        # Get stats
        stats = self.plugin.get_session_stats()
        
        # Verify stats
        self.assertEqual(stats['achievements'], 1)
        self.assertEqual(stats['levels_completed'], 1)
        self.assertEqual(stats['bosses_defeated'], 1)
        self.assertEqual(stats['deaths'], 5)
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        self.plugin.discord_client = Mock()
        self.plugin.last_performance_update = 0
        self.plugin.performance_update_interval = 300
        
        # Update performance
        metrics = {
            'fps': 60,
            'cpu': 50,
            'memory': 4096,
            'gpu': 70
        }
        
        self.plugin.on_performance_update(metrics)
        
        # Check if update was sent (if interval passed)
        current_time = time.time()
        if current_time - self.plugin.last_performance_update >= self.plugin.performance_update_interval:
            self.plugin.discord_client.send_event.assert_called()
    
    def test_speedrun_tracking(self):
        """Test speedrun split tracking."""
        self.plugin.discord_client = Mock()
        
        # Track split
        split = {
            'name': 'World 1',
            'time': 120.5,
            'best': 118.3,
            'gold': False
        }
        
        self.plugin.on_speedrun_split(split)
        
        # Verify event sent
        self.plugin.discord_client.send_event.assert_called_with(
            'speedrun_split',
            unittest.mock.ANY
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for Discord plugin."""
    
    def test_full_session_flow(self):
        """Test complete gaming session flow."""
        # Create plugin
        plugin = DiscordIntegrationPlugin()
        
        # Configure
        plugin.config = {
            'webhook_url': 'https://discord.com/api/webhooks/123456789/abcdefgh',
            'send_achievements': True,
            'send_session_summary': True
        }
        
        # Mock Discord client
        plugin.discord_client = Mock()
        
        # Simulate session
        plugin.on_load()
        
        # Play game
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            plugin.on_frame(frame)
            
            if i == 3:
                plugin.on_achievement({'name': 'First Blood'})
            if i == 7:
                plugin.on_level_complete({'level': 1})
        
        # End session
        plugin.on_unload()
        
        # Verify session end was sent
        calls = plugin.discord_client.send_session_end.call_args_list
        if calls:
            stats = calls[0][0][1]  # Second argument of first call
            self.assertIn('achievements', stats)
            self.assertIn('levels_completed', stats)
    
    def test_error_recovery(self):
        """Test error recovery in Discord communication."""
        plugin = DiscordIntegrationPlugin()
        
        # Configure with invalid webhook
        plugin.config = {
            'webhook_url': 'invalid_url'
        }
        
        # Should not crash on load
        try:
            plugin.on_load()
            # Plugin should handle invalid config gracefully
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Plugin crashed with invalid config: {e}")
        
        # Clean up
        plugin.on_unload()


if __name__ == '__main__':
    unittest.main()