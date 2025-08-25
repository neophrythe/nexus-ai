"""Enhanced Crossbar WAMP Integration - SerpentAI Compatible with Modern Features

Provides real-time communication for game AI framework using WAMP protocol.
"""

import asyncio
import json
import time
import queue
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import structlog
import pickle

# WAMP client imports
try:
    from autobahn.asyncio.wamp import ApplicationSession, ApplicationRunner
    from autobahn.wamp.types import RegisterOptions, PublishOptions
    from autobahn.wamp import auth
    HAS_AUTOBAHN = True
except ImportError:
    HAS_AUTOBAHN = False

# Redis support (optional)
try:
    from redis import StrictRedis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = structlog.get_logger()


class WAMPRole(Enum):
    """WAMP roles"""
    BACKEND = "backend"  # Full permissions
    CLIENT = "client"    # Read-only
    ADMIN = "admin"      # Administrative


class EventType(Enum):
    """Event types for real-time communication"""
    FRAME = "frame"
    INPUT = "input"
    ANALYTICS = "analytics"
    METRICS = "metrics"
    STATE = "state"
    COMMAND = "command"
    LOG = "log"
    ERROR = "error"


@dataclass
class WAMPConfig:
    """WAMP configuration"""
    url: str = "ws://localhost:9999/ws"
    realm: str = "nexus"
    auth_method: str = "wampcra"
    auth_id: str = "nexus"
    auth_secret: str = "nexus_secret"
    role: WAMPRole = WAMPRole.BACKEND
    
    # Redis bridge
    use_redis_bridge: bool = False
    redis_config: Optional[Dict] = None
    
    # Event filtering
    event_whitelist: Optional[List[EventType]] = None
    event_blacklist: Optional[List[EventType]] = None
    
    # Performance
    batch_size: int = 10
    batch_timeout: float = 0.1
    max_queue_size: int = 1000
    
    # Persistence
    persist_events: bool = True
    event_log_path: str = "./logs/wamp_events.jsonl"


class WAMPComponent(ApplicationSession):
    """Base WAMP component - SerpentAI compatible with enhancements"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.wamp_config = WAMPConfig()
        self.event_queue = queue.Queue(maxsize=self.wamp_config.max_queue_size)
        self.redis_client = None
        self.running = False
        self.event_log = None
        
        # Statistics
        self.stats = {
            'events_sent': 0,
            'events_received': 0,
            'events_dropped': 0,
            'connection_time': None,
            'last_event_time': None
        }
        
    def onConnect(self):
        """Called when transport is connected"""
        logger.info(f"WAMP transport connected to {self.wamp_config.url}")
        
        # Configure authentication
        if self.wamp_config.auth_method == "wampcra":
            self.join(
                self.wamp_config.realm,
                authmethods=["wampcra"],
                authid=self.wamp_config.auth_id
            )
        else:
            self.join(self.wamp_config.realm)
            
    def onChallenge(self, challenge):
        """Handle authentication challenge"""
        if challenge.method == "wampcra":
            signature = auth.compute_wcs(
                self.wamp_config.auth_secret.encode('utf8'),
                challenge.extra['challenge'].encode('utf8')
            )
            return signature.decode('ascii')
        else:
            raise Exception(f"Unknown authentication method: {challenge.method}")
            
    async def onJoin(self, details):
        """Called when session joins realm"""
        logger.info(f"WAMP session joined: {details}")
        self.stats['connection_time'] = time.time()
        self.running = True
        
        # Initialize components
        await self._initialize_components()
        
        # Register procedures
        await self._register_procedures()
        
        # Start event processing
        asyncio.create_task(self._process_events())
        
        # Start Redis bridge if configured
        if self.wamp_config.use_redis_bridge:
            await self._start_redis_bridge()
            
        # Open event log
        if self.wamp_config.persist_events:
            self._open_event_log()
            
    async def _initialize_components(self):
        """Initialize WAMP components - override in subclasses"""
        pass
        
    async def _register_procedures(self):
        """Register WAMP procedures - override in subclasses"""
        pass
        
    async def _process_events(self):
        """Process event queue"""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Get event from queue
                try:
                    event = self.event_queue.get(timeout=0.01)
                    batch.append(event)
                except queue.Empty:
                    pass
                    
                # Check if we should flush batch
                should_flush = (
                    len(batch) >= self.wamp_config.batch_size or
                    time.time() - last_flush >= self.wamp_config.batch_timeout
                )
                
                if should_flush and batch:
                    await self._flush_events(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                
            await asyncio.sleep(0.001)
            
    async def _flush_events(self, events: List[Dict]):
        """Flush event batch"""
        for event in events:
            try:
                # Apply filtering
                if not self._should_process_event(event):
                    continue
                    
                # Publish event
                topic = f"nexus.{event['type']}.{event.get('subtype', 'general')}"
                await self.publish(topic, event, options=PublishOptions(acknowledge=True))
                
                self.stats['events_sent'] += 1
                self.stats['last_event_time'] = time.time()
                
                # Log event if configured
                if self.event_log:
                    self._log_event(event)
                    
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")
                self.stats['events_dropped'] += 1
                
    def _should_process_event(self, event: Dict) -> bool:
        """Check if event should be processed"""
        event_type = EventType(event.get('type', 'general'))
        
        # Check whitelist
        if self.wamp_config.event_whitelist:
            if event_type not in self.wamp_config.event_whitelist:
                return False
                
        # Check blacklist
        if self.wamp_config.event_blacklist:
            if event_type in self.wamp_config.event_blacklist:
                return False
                
        return True
        
    async def _start_redis_bridge(self):
        """Start Redis to WAMP bridge"""
        if not HAS_REDIS or not self.wamp_config.redis_config:
            return
            
        try:
            self.redis_client = StrictRedis(**self.wamp_config.redis_config)
            asyncio.create_task(self._redis_consumer())
            logger.info("Redis bridge started")
        except Exception as e:
            logger.error(f"Failed to start Redis bridge: {e}")
            
    async def _redis_consumer(self):
        """Consume events from Redis"""
        redis_key = "nexus:events"
        
        while self.running:
            try:
                # Get event from Redis
                event_data = self.redis_client.brpop(redis_key, timeout=1)
                
                if event_data:
                    _, raw_event = event_data
                    
                    # Deserialize event
                    try:
                        event = json.loads(raw_event)
                    except:
                        event = pickle.loads(raw_event)
                        
                    # Add to queue
                    if not self.event_queue.full():
                        self.event_queue.put(event)
                    else:
                        self.stats['events_dropped'] += 1
                        
            except Exception as e:
                logger.error(f"Redis consumer error: {e}")
                
            await asyncio.sleep(0.001)
            
    def _open_event_log(self):
        """Open event log file"""
        log_path = Path(self.wamp_config.event_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log = open(log_path, 'a')
        
    def _log_event(self, event: Dict):
        """Log event to file"""
        if self.event_log:
            event['logged_at'] = datetime.now().isoformat()
            self.event_log.write(json.dumps(event) + '\n')
            self.event_log.flush()
            
    def onLeave(self, details):
        """Called when session leaves realm"""
        logger.info(f"WAMP session left: {details}")
        self.running = False
        
        if self.event_log:
            self.event_log.close()
            
    def onDisconnect(self):
        """Called when transport is disconnected"""
        logger.info("WAMP transport disconnected")


class AnalyticsComponent(WAMPComponent):
    """Analytics WAMP component - SerpentAI compatible"""
    
    async def _register_procedures(self):
        """Register analytics procedures"""
        # Register RPC methods
        await self.register(self.get_analytics, 'nexus.analytics.get')
        await self.register(self.track_event, 'nexus.analytics.track')
        await self.register(self.get_metrics, 'nexus.analytics.metrics')
        await self.register(self.get_statistics, 'nexus.analytics.stats')
        
        logger.info("Analytics procedures registered")
        
    async def track_event(self, event_key: str, data: Dict, 
                          timestamp: Optional[float] = None):
        """Track analytics event - SerpentAI compatible"""
        event = {
            'type': EventType.ANALYTICS.value,
            'key': event_key,
            'data': data,
            'timestamp': timestamp or time.time()
        }
        
        # Add to queue
        if not self.event_queue.full():
            self.event_queue.put(event)
            self.stats['events_received'] += 1
        else:
            self.stats['events_dropped'] += 1
            
        # Store in Redis if configured
        if self.redis_client:
            redis_key = f"nexus:analytics:{event_key}"
            self.redis_client.lpush(redis_key, json.dumps(event))
            self.redis_client.ltrim(redis_key, 0, 1000)  # Keep last 1000
            
        return True
        
    async def get_analytics(self, event_key: Optional[str] = None, 
                           limit: int = 100) -> List[Dict]:
        """Get analytics events"""
        if not self.redis_client:
            return []
            
        if event_key:
            redis_key = f"nexus:analytics:{event_key}"
        else:
            redis_key = "nexus:analytics:*"
            
        events = []
        
        if '*' in redis_key:
            # Get all analytics keys
            for key in self.redis_client.scan_iter(redis_key):
                for item in self.redis_client.lrange(key, 0, limit):
                    events.append(json.loads(item))
        else:
            # Get specific key
            for item in self.redis_client.lrange(redis_key, 0, limit):
                events.append(json.loads(item))
                
        return events
        
    async def get_metrics(self) -> Dict:
        """Get current metrics"""
        metrics = {}
        
        if self.redis_client:
            # Aggregate metrics from Redis
            for key in self.redis_client.scan_iter("nexus:metrics:*"):
                metric_name = key.decode().split(':')[-1]
                value = self.redis_client.get(key)
                
                try:
                    metrics[metric_name] = float(value)
                except:
                    metrics[metric_name] = value.decode() if value else None
                    
        return metrics
        
    async def get_statistics(self) -> Dict:
        """Get component statistics"""
        return self.stats


class InputControllerComponent(WAMPComponent):
    """Input controller WAMP component - SerpentAI compatible"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.input_controller = None
        
    async def _initialize_components(self):
        """Initialize input controller"""
        from nexus.input.controller import InputController
        self.input_controller = InputController()
        
    async def _register_procedures(self):
        """Register input procedures"""
        await self.register(self.send_input, 'nexus.input.send')
        await self.register(self.press_key, 'nexus.input.key.press')
        await self.register(self.release_key, 'nexus.input.key.release')
        await self.register(self.move_mouse, 'nexus.input.mouse.move')
        await self.register(self.click_mouse, 'nexus.input.mouse.click')
        
        logger.info("Input controller procedures registered")
        
    async def send_input(self, input_type: str, data: Dict) -> bool:
        """Send input command"""
        if not self.input_controller:
            return False
            
        try:
            if input_type == "keyboard":
                key = data.get('key')
                action = data.get('action', 'press')
                
                if action == 'press':
                    await self.input_controller.press_key(key)
                elif action == 'release':
                    await self.input_controller.release_key(key)
                    
            elif input_type == "mouse":
                action = data.get('action')
                
                if action == 'move':
                    x, y = data.get('x'), data.get('y')
                    await self.input_controller.move_mouse(x, y)
                elif action == 'click':
                    button = data.get('button', 'left')
                    await self.input_controller.click(button=button)
                    
            # Track event
            event = {
                'type': EventType.INPUT.value,
                'subtype': input_type,
                'data': data,
                'timestamp': time.time()
            }
            
            if not self.event_queue.full():
                self.event_queue.put(event)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to send input: {e}")
            return False
            
    async def press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press keyboard key"""
        return await self.send_input("keyboard", {
            'key': key,
            'action': 'press',
            'duration': duration
        })
        
    async def release_key(self, key: str) -> bool:
        """Release keyboard key"""
        return await self.send_input("keyboard", {
            'key': key,
            'action': 'release'
        })
        
    async def move_mouse(self, x: int, y: int) -> bool:
        """Move mouse to position"""
        return await self.send_input("mouse", {
            'action': 'move',
            'x': x,
            'y': y
        })
        
    async def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None,
                         button: str = "left") -> bool:
        """Click mouse button"""
        data = {'action': 'click', 'button': button}
        
        if x is not None and y is not None:
            data['x'] = x
            data['y'] = y
            
        return await self.send_input("mouse", data)


class DashboardAPIComponent(WAMPComponent):
    """Dashboard API WAMP component"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.dashboards = {}
        self.widgets = {}
        
    async def _register_procedures(self):
        """Register dashboard procedures"""
        await self.register(self.list_dashboards, 'nexus.dashboard.list')
        await self.register(self.create_dashboard, 'nexus.dashboard.create')
        await self.register(self.update_dashboard, 'nexus.dashboard.update')
        await self.register(self.delete_dashboard, 'nexus.dashboard.delete')
        await self.register(self.add_widget, 'nexus.dashboard.widget.add')
        await self.register(self.update_widget, 'nexus.dashboard.widget.update')
        await self.register(self.get_widget_data, 'nexus.dashboard.widget.data')
        
        logger.info("Dashboard API procedures registered")
        
    async def list_dashboards(self) -> List[Dict]:
        """List all dashboards"""
        return [
            {
                'id': dashboard_id,
                'name': dashboard['name'],
                'created_at': dashboard['created_at'],
                'widgets': len(dashboard['widgets'])
            }
            for dashboard_id, dashboard in self.dashboards.items()
        ]
        
    async def create_dashboard(self, name: str, layout: Optional[Dict] = None) -> str:
        """Create new dashboard"""
        dashboard_id = f"dashboard_{int(time.time() * 1000)}"
        
        self.dashboards[dashboard_id] = {
            'id': dashboard_id,
            'name': name,
            'layout': layout or {},
            'widgets': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Publish creation event
        await self.publish('nexus.dashboard.created', {
            'dashboard_id': dashboard_id,
            'name': name
        })
        
        return dashboard_id
        
    async def update_dashboard(self, dashboard_id: str, updates: Dict) -> bool:
        """Update dashboard"""
        if dashboard_id not in self.dashboards:
            return False
            
        dashboard = self.dashboards[dashboard_id]
        dashboard.update(updates)
        dashboard['updated_at'] = datetime.now().isoformat()
        
        # Publish update event
        await self.publish(f'nexus.dashboard.{dashboard_id}.updated', updates)
        
        return True
        
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete dashboard"""
        if dashboard_id not in self.dashboards:
            return False
            
        del self.dashboards[dashboard_id]
        
        # Publish deletion event
        await self.publish('nexus.dashboard.deleted', {'dashboard_id': dashboard_id})
        
        return True
        
    async def add_widget(self, dashboard_id: str, widget_type: str, 
                        config: Dict) -> str:
        """Add widget to dashboard"""
        if dashboard_id not in self.dashboards:
            raise Exception(f"Dashboard {dashboard_id} not found")
            
        widget_id = f"widget_{int(time.time() * 1000)}"
        
        widget = {
            'id': widget_id,
            'type': widget_type,
            'config': config,
            'data': {},
            'created_at': datetime.now().isoformat()
        }
        
        self.dashboards[dashboard_id]['widgets'].append(widget_id)
        self.widgets[widget_id] = widget
        
        # Publish widget creation event
        await self.publish(f'nexus.dashboard.{dashboard_id}.widget.added', widget)
        
        return widget_id
        
    async def update_widget(self, widget_id: str, data: Dict) -> bool:
        """Update widget data"""
        if widget_id not in self.widgets:
            return False
            
        self.widgets[widget_id]['data'] = data
        self.widgets[widget_id]['updated_at'] = datetime.now().isoformat()
        
        # Publish widget update event
        await self.publish(f'nexus.widget.{widget_id}.updated', data)
        
        return True
        
    async def get_widget_data(self, widget_id: str) -> Dict:
        """Get widget data"""
        if widget_id not in self.widgets:
            return {}
            
        return self.widgets[widget_id].get('data', {})


class WAMPRunner:
    """WAMP application runner"""
    
    def __init__(self, config: WAMPConfig):
        self.config = config
        self.runner = None
        self.components = []
        
    def add_component(self, component_class: type):
        """Add WAMP component"""
        self.components.append(component_class)
        
    def run(self):
        """Run WAMP application"""
        if not HAS_AUTOBAHN:
            logger.error("Autobahn not installed. Install with: pip install autobahn")
            return
            
        # Create runner
        self.runner = ApplicationRunner(
            url=self.config.url,
            realm=self.config.realm
        )
        
        # Run each component
        for component_class in self.components:
            try:
                self.runner.run(component_class)
            except Exception as e:
                logger.error(f"Failed to run component {component_class.__name__}: {e}")


# SerpentAI compatibility functions
def start_analytics_component(project_key: str, **kwargs):
    """Start analytics component - SerpentAI compatible"""
    config = WAMPConfig(**kwargs)
    
    runner = ApplicationRunner(
        url=config.url,
        realm=config.realm
    )
    
    class ProjectAnalyticsComponent(AnalyticsComponent):
        def __init__(self, config=None):
            super().__init__(config)
            self.project_key = project_key
            
    runner.run(ProjectAnalyticsComponent)


def start_input_controller_component(**kwargs):
    """Start input controller component - SerpentAI compatible"""
    config = WAMPConfig(**kwargs)
    
    runner = ApplicationRunner(
        url=config.url,
        realm=config.realm
    )
    
    runner.run(InputControllerComponent)


def start_dashboard_api_component(**kwargs):
    """Start dashboard API component - SerpentAI compatible"""
    config = WAMPConfig(**kwargs)
    
    runner = ApplicationRunner(
        url=config.url,
        realm=config.realm
    )
    
    runner.run(DashboardAPIComponent)