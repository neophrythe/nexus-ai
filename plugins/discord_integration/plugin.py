"""
Discord Integration Plugin for Nexus Game AI Framework

Sends game events, achievements, and metrics to Discord.
"""

import json
import time
import asyncio
import threading
import queue
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import structlog

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import discord
    from discord import Webhook
    HAS_DISCORD_PY = True
except ImportError:
    HAS_DISCORD_PY = False

from nexus.core.plugin_base import PluginBase

logger = structlog.get_logger()


@dataclass
class DiscordMessage:
    """Discord message structure."""
    content: str = None
    embeds: List[Dict] = None
    username: str = "Nexus AI"
    avatar_url: str = None
    tts: bool = False
    
    def to_dict(self) -> Dict:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return data


class DiscordIntegrationPlugin(PluginBase):
    """
    Plugin for Discord integration including:
    - Game event notifications
    - Achievement tracking
    - Performance metrics
    - Screenshot sharing
    - Leaderboard updates
    - Custom alerts
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Discord Integration"
        self.version = "1.0.0"
        self.description = "Send game events and notifications to Discord"
        
        # Discord settings
        self.webhook_url = None
        self.bot_token = None
        self.channel_id = None
        self.guild_id = None
        
        # Message queue
        self.message_queue = queue.Queue()
        self.sender_thread = None
        self.is_running = False
        
        # Rate limiting
        self.rate_limit = 5  # messages per second
        self.last_message_time = 0
        
        # Event tracking
        self.event_types = {
            'achievement': True,
            'level_complete': True,
            'boss_defeated': True,
            'high_score': True,
            'death': False,
            'milestone': True,
            'performance': False,
            'screenshot': True
        }
        
        # Stats tracking
        self.session_stats = {
            'start_time': None,
            'achievements': [],
            'high_scores': [],
            'deaths': 0,
            'bosses_defeated': 0,
            'levels_completed': 0
        }
        
        # Rich presence (if bot token provided)
        self.discord_client = None
        self.presence_enabled = False
        
    def on_load(self):
        """Called when plugin is loaded."""
        logger.info(f"Loading {self.name} v{self.version}")
        
        # Load configuration
        self.webhook_url = self.config.get('webhook_url')
        self.bot_token = self.config.get('bot_token')
        self.channel_id = self.config.get('channel_id')
        self.guild_id = self.config.get('guild_id')
        
        if not self.webhook_url and not self.bot_token:
            logger.warning("Discord plugin loaded but no webhook URL or bot token configured")
            return
        
        # Start message sender thread
        self.is_running = True
        self.sender_thread = threading.Thread(target=self._message_sender_loop, daemon=True)
        self.sender_thread.start()
        
        # Start session
        self.session_stats['start_time'] = datetime.now()
        
        # Send startup message
        self.send_embed(
            title="🎮 Nexus AI Started",
            description="Game AI session started",
            color=0x00FF00,
            fields=[
                {"name": "Version", "value": self.version, "inline": True},
                {"name": "Time", "value": datetime.now().strftime("%H:%M:%S"), "inline": True}
            ]
        )
        
    def on_unload(self):
        """Called when plugin is unloaded."""
        # Send session summary
        self.send_session_summary()
        
        # Stop sender thread
        self.is_running = False
        if self.sender_thread:
            self.sender_thread.join(timeout=2.0)
        
        logger.info(f"Unloaded {self.name}")
    
    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process game frame."""
        # No frame processing needed
        return frame
    
    def on_game_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle game events."""
        if not self.event_types.get(event_type, False):
            return
        
        # Update stats
        if event_type == 'achievement':
            self.session_stats['achievements'].append(event_data)
            self.send_achievement(event_data)
        
        elif event_type == 'level_complete':
            self.session_stats['levels_completed'] += 1
            self.send_level_complete(event_data)
        
        elif event_type == 'boss_defeated':
            self.session_stats['bosses_defeated'] += 1
            self.send_boss_defeated(event_data)
        
        elif event_type == 'high_score':
            self.session_stats['high_scores'].append(event_data)
            self.send_high_score(event_data)
        
        elif event_type == 'death':
            self.session_stats['deaths'] += 1
            if self.event_types['death']:
                self.send_death_notification(event_data)
        
        elif event_type == 'milestone':
            self.send_milestone(event_data)
        
        elif event_type == 'screenshot':
            self.send_screenshot(event_data)
    
    def send_achievement(self, achievement: Dict[str, Any]):
        """Send achievement notification."""
        self.send_embed(
            title="🏆 Achievement Unlocked!",
            description=achievement.get('name', 'Unknown Achievement'),
            color=0xFFD700,  # Gold
            fields=[
                {"name": "Description", "value": achievement.get('description', ''), "inline": False},
                {"name": "Points", "value": str(achievement.get('points', 0)), "inline": True},
                {"name": "Rarity", "value": achievement.get('rarity', 'Common'), "inline": True}
            ],
            thumbnail_url=achievement.get('icon_url'),
            timestamp=True
        )
    
    def send_level_complete(self, level_data: Dict[str, Any]):
        """Send level completion notification."""
        self.send_embed(
            title="✅ Level Complete!",
            description=f"Level {level_data.get('level_name', 'Unknown')} completed",
            color=0x00FF00,  # Green
            fields=[
                {"name": "Time", "value": level_data.get('completion_time', 'N/A'), "inline": True},
                {"name": "Score", "value": str(level_data.get('score', 0)), "inline": True},
                {"name": "Deaths", "value": str(level_data.get('deaths', 0)), "inline": True},
                {"name": "Collectibles", "value": f"{level_data.get('collectibles_found', 0)}/{level_data.get('total_collectibles', 0)}", "inline": True}
            ]
        )
    
    def send_boss_defeated(self, boss_data: Dict[str, Any]):
        """Send boss defeat notification."""
        self.send_embed(
            title="⚔️ Boss Defeated!",
            description=f"{boss_data.get('boss_name', 'Unknown Boss')} has been defeated!",
            color=0xFF0000,  # Red
            fields=[
                {"name": "Battle Duration", "value": boss_data.get('battle_time', 'N/A'), "inline": True},
                {"name": "Attempts", "value": str(boss_data.get('attempts', 1)), "inline": True},
                {"name": "Damage Dealt", "value": str(boss_data.get('damage_dealt', 0)), "inline": True},
                {"name": "Damage Taken", "value": str(boss_data.get('damage_taken', 0)), "inline": True}
            ],
            image_url=boss_data.get('screenshot_url')
        )
    
    def send_high_score(self, score_data: Dict[str, Any]):
        """Send high score notification."""
        self.send_embed(
            title="🌟 New High Score!",
            description="A new high score has been achieved!",
            color=0x9B59B6,  # Purple
            fields=[
                {"name": "Score", "value": f"{score_data.get('score', 0):,}", "inline": True},
                {"name": "Previous Best", "value": f"{score_data.get('previous_best', 0):,}", "inline": True},
                {"name": "Rank", "value": score_data.get('rank', 'N/A'), "inline": True},
                {"name": "Game Mode", "value": score_data.get('mode', 'Classic'), "inline": True}
            ]
        )
    
    def send_death_notification(self, death_data: Dict[str, Any]):
        """Send death notification."""
        self.send_embed(
            title="💀 Death",
            description=death_data.get('cause', 'Unknown cause'),
            color=0x000000,  # Black
            fields=[
                {"name": "Location", "value": death_data.get('location', 'Unknown'), "inline": True},
                {"name": "Lives Remaining", "value": str(death_data.get('lives_remaining', 0)), "inline": True}
            ]
        )
    
    def send_milestone(self, milestone_data: Dict[str, Any]):
        """Send milestone notification."""
        self.send_embed(
            title="🎉 Milestone Reached!",
            description=milestone_data.get('description', 'Milestone achieved'),
            color=0x3498DB,  # Blue
            fields=[
                {"name": "Type", "value": milestone_data.get('type', 'General'), "inline": True},
                {"name": "Progress", "value": milestone_data.get('progress', 'N/A'), "inline": True}
            ]
        )
    
    def send_screenshot(self, screenshot_data: Dict[str, Any]):
        """Send screenshot to Discord."""
        # This would need to upload the image first
        # For now, just send a message
        self.send_embed(
            title="📷 Screenshot",
            description=screenshot_data.get('caption', 'Game screenshot'),
            color=0x00CED1,  # Dark turquoise
            image_url=screenshot_data.get('url'),
            fields=[
                {"name": "Location", "value": screenshot_data.get('location', 'Unknown'), "inline": True},
                {"name": "Time", "value": datetime.now().strftime("%H:%M:%S"), "inline": True}
            ]
        )
    
    def send_performance_update(self, metrics: Dict[str, Any]):
        """Send performance metrics update."""
        self.send_embed(
            title="📊 Performance Update",
            description="Current performance metrics",
            color=0x2ECC71,  # Green
            fields=[
                {"name": "FPS", "value": f"{metrics.get('fps', 0):.1f}", "inline": True},
                {"name": "CPU Usage", "value": f"{metrics.get('cpu_percent', 0):.1f}%", "inline": True},
                {"name": "Memory Usage", "value": f"{metrics.get('memory_percent', 0):.1f}%", "inline": True},
                {"name": "GPU Usage", "value": f"{metrics.get('gpu_percent', 0):.1f}%", "inline": True},
                {"name": "Frame Time", "value": f"{metrics.get('frame_time', 0):.1f}ms", "inline": True},
                {"name": "Network Latency", "value": f"{metrics.get('latency', 0):.1f}ms", "inline": True}
            ]
        )
    
    def send_session_summary(self):
        """Send session summary."""
        if not self.session_stats['start_time']:
            return
        
        duration = datetime.now() - self.session_stats['start_time']
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.send_embed(
            title="🎮 Session Summary",
            description="Game AI session ended",
            color=0xE74C3C,  # Red
            fields=[
                {"name": "Duration", "value": f"{hours}h {minutes}m {seconds}s", "inline": True},
                {"name": "Achievements", "value": str(len(self.session_stats['achievements'])), "inline": True},
                {"name": "High Scores", "value": str(len(self.session_stats['high_scores'])), "inline": True},
                {"name": "Levels Completed", "value": str(self.session_stats['levels_completed']), "inline": True},
                {"name": "Bosses Defeated", "value": str(self.session_stats['bosses_defeated']), "inline": True},
                {"name": "Deaths", "value": str(self.session_stats['deaths']), "inline": True}
            ],
            timestamp=True
        )
    
    def send_message(self, content: str, **kwargs):
        """Send text message to Discord."""
        message = DiscordMessage(content=content, **kwargs)
        self.message_queue.put(message)
    
    def send_embed(self, title: str, description: str = None, color: int = 0x000000,
                   fields: List[Dict] = None, thumbnail_url: str = None,
                   image_url: str = None, timestamp: bool = False, **kwargs):
        """Send embed message to Discord."""
        embed = {
            "title": title,
            "color": color
        }
        
        if description:
            embed["description"] = description
        
        if fields:
            embed["fields"] = fields
        
        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}
        
        if image_url:
            embed["image"] = {"url": image_url}
        
        if timestamp:
            embed["timestamp"] = datetime.now().isoformat()
        
        message = DiscordMessage(embeds=[embed], **kwargs)
        self.message_queue.put(message)
    
    def _message_sender_loop(self):
        """Background thread for sending messages."""
        while self.is_running:
            try:
                # Get message from queue (with timeout)
                message = self.message_queue.get(timeout=1.0)
                
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_message_time
                if time_since_last < (1.0 / self.rate_limit):
                    time.sleep((1.0 / self.rate_limit) - time_since_last)
                
                # Send message
                self._send_to_discord(message)
                self.last_message_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending Discord message: {e}")
    
    def _send_to_discord(self, message: DiscordMessage):
        """Send message to Discord via webhook."""
        if not self.webhook_url:
            return
        
        try:
            if HAS_HTTPX:
                # Use httpx for async sending
                with httpx.Client() as client:
                    response = client.post(
                        self.webhook_url,
                        json=message.to_dict(),
                        timeout=10.0
                    )
                    
                    if response.status_code == 429:  # Rate limited
                        retry_after = response.json().get('retry_after', 5)
                        logger.warning(f"Discord rate limited, retry after {retry_after}s")
                        time.sleep(retry_after)
                        # Retry
                        response = client.post(
                            self.webhook_url,
                            json=message.to_dict(),
                            timeout=10.0
                        )
            else:
                # Fallback to requests
                import requests
                response = requests.post(
                    self.webhook_url,
                    json=message.to_dict(),
                    timeout=10.0
                )
                
                if response.status_code == 429:
                    retry_after = response.json().get('retry_after', 5)
                    logger.warning(f"Discord rate limited, retry after {retry_after}s")
                    time.sleep(retry_after)
                    response = requests.post(
                        self.webhook_url,
                        json=message.to_dict(),
                        timeout=10.0
                    )
            
            if response.status_code not in [200, 204]:
                logger.error(f"Discord webhook error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
    
    def set_presence(self, activity: str, status: str = "online"):
        """Set Discord rich presence (requires bot token)."""
        if not self.discord_client:
            return
        
        # This would set rich presence via Discord bot
        logger.info(f"Setting presence: {activity}")
    
    def create_leaderboard(self, scores: List[Dict[str, Any]], title: str = "Leaderboard"):
        """Create and send leaderboard."""
        # Sort scores
        sorted_scores = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
        
        # Format leaderboard
        leaderboard_text = ""
        for i, entry in enumerate(sorted_scores[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            leaderboard_text += f"{medal} **{entry.get('player', 'Unknown')}** - {entry.get('score', 0):,}\n"
        
        self.send_embed(
            title=f"🏆 {title}",
            description=leaderboard_text,
            color=0xFFD700,
            timestamp=True
        )
    
    def enable_event_type(self, event_type: str, enabled: bool = True):
        """Enable or disable specific event types."""
        if event_type in self.event_types:
            self.event_types[event_type] = enabled
            logger.info(f"Discord event '{event_type}' {'enabled' if enabled else 'disabled'}")
    
    def set_webhook_url(self, url: str):
        """Set Discord webhook URL."""
        self.webhook_url = url
        logger.info("Discord webhook URL updated")


# Plugin registration
def create_plugin() -> DiscordIntegrationPlugin:
    """Create plugin instance."""
    return DiscordIntegrationPlugin()