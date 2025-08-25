"""
Discord client implementation with webhook and bot support.
"""

import asyncio
import aiohttp
import discord
from discord.ext import commands
import time
import json
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, List, Callable
from collections import deque
from threading import Thread, Lock
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from plugins.discord_integration.config import (
    DiscordConfig, WebhookConfig, BotConfig, NotificationLevel
)


@dataclass
class RateLimitBucket:
    """Rate limit tracking for Discord API."""
    limit: int
    remaining: int
    reset: float
    retry_after: Optional[float] = None


class WebhookClient:
    """Discord webhook client with rate limiting."""
    
    def __init__(self, config: WebhookConfig):
        """
        Initialize webhook client.
        
        Args:
            config: Webhook configuration
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = RateLimitBucket(
            limit=config.rate_limit_per_minute,
            remaining=config.rate_limit_per_minute,
            reset=time.time() + 60
        )
        self.message_queue = deque()
        self.queue_lock = Lock()
        self.logger = logging.getLogger(__name__)
        
        # Start async event loop in thread
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
    
    def _run_event_loop(self):
        """Run async event loop in thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._process_queue())
    
    async def _process_queue(self):
        """Process message queue."""
        self.session = aiohttp.ClientSession()
        
        try:
            while True:
                # Check queue
                with self.queue_lock:
                    if self.message_queue:
                        message = self.message_queue.popleft()
                    else:
                        message = None
                
                if message:
                    await self._send_webhook(message)
                
                await asyncio.sleep(0.1)
        finally:
            await self.session.close()
    
    async def _send_webhook(self, data: Dict[str, Any]):
        """Send webhook with rate limiting."""
        # Check rate limit
        if not await self._check_rate_limit():
            if self.config.retry_on_rate_limit:
                # Re-queue message
                with self.queue_lock:
                    self.message_queue.append(data)
                await asyncio.sleep(self.rate_limit.retry_after or 1)
            return
        
        # Prepare webhook data
        webhook_data = {
            'username': self.config.username,
            'avatar_url': self.config.avatar_url
        }
        webhook_data.update(data)
        
        # Send request
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    self.config.url,
                    json=webhook_data
                ) as response:
                    # Update rate limit from headers
                    self._update_rate_limit(response.headers)
                    
                    if response.status == 204:
                        # Success
                        return
                    elif response.status == 429:
                        # Rate limited
                        retry_data = await response.json()
                        self.rate_limit.retry_after = retry_data.get('retry_after', 1)
                        
                        if self.config.retry_on_rate_limit and attempt < self.config.max_retries:
                            await asyncio.sleep(self.rate_limit.retry_after)
                            continue
                    else:
                        self.logger.error(f"Webhook error: {response.status}")
                        break
            except Exception as e:
                self.logger.error(f"Webhook send error: {e}")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                break
    
    async def _check_rate_limit(self) -> bool:
        """Check if rate limit allows sending."""
        now = time.time()
        
        # Reset if needed
        if now >= self.rate_limit.reset:
            self.rate_limit.remaining = self.rate_limit.limit
            self.rate_limit.reset = now + 60
        
        # Check if we have remaining requests
        if self.rate_limit.remaining > 0:
            self.rate_limit.remaining -= 1
            return True
        
        return False
    
    def _update_rate_limit(self, headers: Dict[str, str]):
        """Update rate limit from response headers."""
        if 'X-RateLimit-Limit' in headers:
            self.rate_limit.limit = int(headers['X-RateLimit-Limit'])
        if 'X-RateLimit-Remaining' in headers:
            self.rate_limit.remaining = int(headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in headers:
            self.rate_limit.reset = float(headers['X-RateLimit-Reset'])
    
    def send_message(self, content: str, embed: Optional[Dict] = None):
        """Queue message to send."""
        data = {}
        
        if content:
            data['content'] = content[:2000]  # Discord limit
        
        if embed:
            data['embeds'] = [embed]
        
        with self.queue_lock:
            self.message_queue.append(data)
    
    def send_embed(self, title: str, description: str, 
                   color: int = 0x3498DB, fields: List[Dict] = None,
                   thumbnail: Optional[str] = None, image: Optional[str] = None,
                   footer: Optional[str] = None):
        """Send rich embed message."""
        embed = {
            'title': title[:256],  # Discord limit
            'description': description[:4096],  # Discord limit
            'color': color,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if fields:
            embed['fields'] = fields[:25]  # Discord limit
        
        if thumbnail:
            embed['thumbnail'] = {'url': thumbnail}
        
        if image:
            embed['image'] = {'url': image}
        
        if footer:
            embed['footer'] = {'text': footer[:2048]}  # Discord limit
        
        self.send_message(None, embed)
    
    def send_screenshot(self, image: Image.Image, caption: str = "",
                       quality: int = 80, max_size: int = 8388608):
        """Send screenshot with compression."""
        # Compress image
        output = BytesIO()
        
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = rgb_image
        
        # Try different quality levels to fit size limit
        for q in range(quality, 10, -10):
            output.seek(0)
            output.truncate()
            image.save(output, format='JPEG', quality=q, optimize=True)
            
            if output.tell() <= max_size:
                break
        
        # Encode to base64
        output.seek(0)
        encoded = base64.b64encode(output.read()).decode('utf-8')
        
        # Send as embedded image
        embed = {
            'title': 'Screenshot',
            'description': caption[:4096] if caption else 'Game screenshot',
            'image': {
                'url': f"data:image/jpeg;base64,{encoded}"
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.send_message(None, embed)
    
    def close(self):
        """Close webhook client."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join(timeout=5)


class BotClient(commands.Bot):
    """Discord bot client with game integration."""
    
    def __init__(self, config: BotConfig, game_callbacks: Dict[str, Callable] = None):
        """
        Initialize bot client.
        
        Args:
            config: Bot configuration
            game_callbacks: Callbacks for game commands
        """
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(
            command_prefix=config.command_prefix,
            intents=intents
        )
        
        self.config = config
        self.game_callbacks = game_callbacks or {}
        self.activity_update_task = None
        self.logger = logging.getLogger(__name__)
        
        # Add commands
        self._setup_commands()
    
    def _setup_commands(self):
        """Set up bot commands."""
        
        @self.command(name='stats')
        async def stats(ctx):
            """Show game statistics."""
            if 'get_stats' in self.game_callbacks:
                stats_data = self.game_callbacks['get_stats']()
                
                embed = discord.Embed(
                    title="Game Statistics",
                    color=0x3498DB
                )
                
                for key, value in stats_data.items():
                    embed.add_field(name=key, value=str(value), inline=True)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("Stats not available")
        
        @self.command(name='screenshot')
        async def screenshot(ctx):
            """Take and send screenshot."""
            if 'take_screenshot' in self.game_callbacks:
                image = self.game_callbacks['take_screenshot']()
                
                if image:
                    # Convert PIL image to Discord file
                    output = BytesIO()
                    image.save(output, format='PNG')
                    output.seek(0)
                    
                    file = discord.File(output, filename='screenshot.png')
                    await ctx.send("Current game view:", file=file)
                else:
                    await ctx.send("Failed to capture screenshot")
            else:
                await ctx.send("Screenshot not available")
        
        @self.command(name='pause')
        async def pause(ctx):
            """Pause the game."""
            if 'pause_game' in self.game_callbacks:
                self.game_callbacks['pause_game']()
                await ctx.send("Game paused")
            else:
                await ctx.send("Pause not available")
        
        @self.command(name='resume')
        async def resume(ctx):
            """Resume the game."""
            if 'resume_game' in self.game_callbacks:
                self.game_callbacks['resume_game']()
                await ctx.send("Game resumed")
            else:
                await ctx.send("Resume not available")
        
        @self.command(name='leaderboard')
        async def leaderboard(ctx):
            """Show leaderboard."""
            if 'get_leaderboard' in self.game_callbacks:
                leaderboard_data = self.game_callbacks['get_leaderboard']()
                
                embed = discord.Embed(
                    title="Leaderboard",
                    color=0xFFD700
                )
                
                for i, entry in enumerate(leaderboard_data[:10], 1):
                    embed.add_field(
                        name=f"{i}. {entry['player']}",
                        value=f"Score: {entry['score']}",
                        inline=False
                    )
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("Leaderboard not available")
    
    async def on_ready(self):
        """Called when bot is ready."""
        self.logger.info(f"Bot logged in as {self.user}")
        
        # Start activity updates if enabled
        if self.config.enable_rich_presence:
            self.activity_update_task = self.loop.create_task(self._update_activity())
        
        # Send to specific channel if configured
        if self.config.channel_id:
            channel = self.get_channel(int(self.config.channel_id))
            if channel:
                await channel.send("🎮 Nexus AI Bot Online!")
    
    async def _update_activity(self):
        """Update bot activity/presence."""
        while not self.is_closed():
            try:
                if 'get_activity' in self.game_callbacks:
                    activity_data = self.game_callbacks['get_activity']()
                    
                    activity = discord.Game(
                        name=activity_data.get('game', 'Unknown Game'),
                        details=activity_data.get('details'),
                        state=activity_data.get('state')
                    )
                    
                    await self.change_presence(activity=activity)
            except Exception as e:
                self.logger.error(f"Activity update error: {e}")
            
            await asyncio.sleep(self.config.status_update_interval)
    
    def send_notification(self, title: str, description: str,
                         color: int = 0x3498DB, channel_id: Optional[str] = None):
        """Send notification to channel."""
        target_channel_id = channel_id or self.config.channel_id
        
        if not target_channel_id:
            return
        
        channel = self.get_channel(int(target_channel_id))
        if channel:
            embed = discord.Embed(
                title=title,
                description=description,
                color=color,
                timestamp=datetime.utcnow()
            )
            
            asyncio.run_coroutine_threadsafe(
                channel.send(embed=embed),
                self.loop
            )
    
    def run_bot(self):
        """Run bot in background thread."""
        thread = Thread(target=lambda: self.run(self.config.token), daemon=True)
        thread.start()
        return thread


class DiscordClient:
    """Unified Discord client supporting webhooks and bots."""
    
    def __init__(self, config: DiscordConfig):
        """
        Initialize Discord client.
        
        Args:
            config: Discord configuration
        """
        self.config = config
        self.webhook_client: Optional[WebhookClient] = None
        self.bot_client: Optional[BotClient] = None
        self.message_batch = []
        self.batch_lock = Lock()
        self.last_batch_time = time.time()
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients based on config
        if config.enabled:
            if config.connection_type in ['webhook', 'both'] and config.webhook:
                self.webhook_client = WebhookClient(config.webhook)
            
            if config.connection_type in ['bot', 'both'] and config.bot:
                self.bot_client = BotClient(config.bot)
        
        # Start batch processor if enabled
        if config.batch_messages:
            self._start_batch_processor()
    
    def _start_batch_processor(self):
        """Start message batch processor."""
        def process_batches():
            while True:
                time.sleep(self.config.batch_interval)
                self._send_batch()
        
        thread = Thread(target=process_batches, daemon=True)
        thread.start()
    
    def _send_batch(self):
        """Send batched messages."""
        with self.batch_lock:
            if not self.message_batch:
                return
            
            # Combine messages
            combined_fields = []
            for msg in self.message_batch[:self.config.max_batch_size]:
                combined_fields.append({
                    'name': msg.get('title', 'Event'),
                    'value': msg.get('description', '')[:1024],
                    'inline': True
                })
            
            # Clear batch
            self.message_batch = self.message_batch[self.config.max_batch_size:]
        
        # Send combined message
        if self.webhook_client:
            self.webhook_client.send_embed(
                title="Batch Update",
                description=f"Multiple events ({len(combined_fields)} total)",
                fields=combined_fields,
                color=0x95A5A6
            )
    
    def send_event(self, event_type: str, data: Dict[str, Any]):
        """Send game event to Discord."""
        # Check if event should be filtered
        if not self._should_send_event(event_type):
            return
        
        # Format message based on event type
        message = self._format_event_message(event_type, data)
        
        # Handle batching
        if self.config.batch_messages and not data.get('immediate', False):
            with self.batch_lock:
                self.message_batch.append(message)
        else:
            # Send immediately
            if self.webhook_client:
                self.webhook_client.send_embed(**message)
            
            if self.bot_client and self.bot_client.config.channel_id:
                self.bot_client.send_notification(
                    message['title'],
                    message['description'],
                    message.get('color', 0x3498DB)
                )
    
    def _should_send_event(self, event_type: str) -> bool:
        """Check if event should be sent based on filters."""
        # Check notification level
        if self.config.notification_level == NotificationLevel.NONE:
            return False
        
        # Check event filters
        filters = self.config.event_filters
        
        event_map = {
            'achievement': filters.achievements,
            'level_complete': filters.level_completion,
            'boss_defeat': filters.boss_defeats,
            'high_score': filters.high_scores,
            'death': filters.deaths,
            'milestone': filters.milestones,
            'performance': filters.performance_updates,
            'screenshot': filters.screenshots,
            'speedrun_split': filters.speedrun_splits
        }
        
        return event_map.get(event_type, True)
    
    def _format_event_message(self, event_type: str, data: Dict[str, Any]) -> Dict:
        """Format event into Discord message."""
        # Get color from style config
        color = self.config.embed_style.colors.get(
            event_type,
            self.config.embed_style.colors.get('info')
        )
        
        # Create base message
        message = {
            'title': data.get('title', event_type.replace('_', ' ').title()),
            'description': data.get('description', ''),
            'color': color
        }
        
        # Add fields if present
        if 'fields' in data:
            message['fields'] = data['fields']
        
        # Add footer if enabled
        if self.config.embed_style.show_footer:
            message['footer'] = self.config.embed_style.footer_text
        
        return message
    
    def send_screenshot(self, image: Image.Image, caption: str = ""):
        """Send screenshot to Discord."""
        if not self.config.event_filters.screenshots:
            return
        
        if self.webhook_client:
            self.webhook_client.send_screenshot(
                image,
                caption,
                self.config.event_filters.screenshot_quality,
                self.config.event_filters.screenshot_max_size
            )
    
    def send_leaderboard(self, leaderboard_data: List[Dict[str, Any]]):
        """Send leaderboard update."""
        if not self.config.enable_leaderboards:
            return
        
        fields = []
        for i, entry in enumerate(leaderboard_data[:self.config.leaderboard_size], 1):
            fields.append({
                'name': f"{i}. {entry.get('player', 'Unknown')}",
                'value': f"Score: {entry.get('score', 0):,}",
                'inline': False
            })
        
        if self.webhook_client:
            self.webhook_client.send_embed(
                title="🏆 Leaderboard Update",
                description="Current top players",
                color=0xFFD700,
                fields=fields
            )
    
    def send_session_start(self, game_name: str, session_id: str):
        """Send session start notification."""
        if not self.config.send_session_start:
            return
        
        self.send_event('session_start', {
            'title': '🎮 Gaming Session Started',
            'description': f"Playing: {game_name}\nSession: {session_id}",
            'immediate': True
        })
    
    def send_session_end(self, duration: float, stats: Dict[str, Any]):
        """Send session end summary."""
        if not self.config.send_session_end:
            return
        
        # Format duration
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"{hours}h {minutes}m {seconds}s"
        
        # Build fields from stats
        fields = []
        for stat in self.config.session_summary_stats:
            if stat in stats:
                fields.append({
                    'name': stat.replace('_', ' ').title(),
                    'value': str(stats[stat]),
                    'inline': True
                })
        
        self.send_event('session_end', {
            'title': '🏁 Gaming Session Ended',
            'description': f"Duration: {duration_str}",
            'fields': fields,
            'immediate': True
        })
    
    def set_game_callbacks(self, callbacks: Dict[str, Callable]):
        """Set game callbacks for bot commands."""
        if self.bot_client:
            self.bot_client.game_callbacks = callbacks
    
    def start(self):
        """Start Discord clients."""
        if self.bot_client:
            self.bot_client.run_bot()
    
    def close(self):
        """Close all Discord clients."""
        if self.webhook_client:
            self.webhook_client.close()
        
        if self.bot_client:
            asyncio.run_coroutine_threadsafe(
                self.bot_client.close(),
                self.bot_client.loop
            )