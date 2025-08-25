from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import yaml
import toml
from datetime import datetime


class PluginType(Enum):
    GAME = "game"
    AGENT = "agent" 
    CAPTURE = "capture"
    VISION = "vision"
    INPUT = "input"
    PROCESSOR = "processor"
    EXTENSION = "extension"


class PluginStatus(Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginManifest:
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    min_nexus_version: str = "0.1.0"
    max_nexus_version: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> "PluginManifest":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        data['plugin_type'] = PluginType(data['plugin_type'])
        return cls(**data)
    
    @classmethod
    def from_toml(cls, path: Path) -> "PluginManifest":
        with open(path, 'r') as f:
            data = toml.load(f)
        data['plugin_type'] = PluginType(data['plugin_type'])
        return cls(**data)


class BasePlugin(ABC):
    
    def __init__(self, manifest: PluginManifest, config: Dict[str, Any]):
        self.manifest = manifest
        self.config = config
        self.status = PluginStatus.UNLOADED
        self.loaded_at: Optional[datetime] = None
        self._resources: Dict[str, Any] = {}
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize plugin resources"""
        self._resources = {}
        self.status = PluginStatus.LOADING
        logger.info(f"Initializing plugin: {self.manifest.name}")
        self.loaded_at = datetime.now()
        self.status = PluginStatus.LOADED
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown plugin and cleanup resources"""
        logger.info(f"Shutting down plugin: {self.manifest.name}")
        for resource_name, resource in self._resources.items():
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, 'shutdown'):
                await resource.shutdown()
        self._resources.clear()
        self.status = PluginStatus.UNLOADED
        self.loaded_at = None
    
    @abstractmethod
    async def validate(self) -> bool:
        """Validate plugin configuration and requirements"""
        # Check required fields
        if not self.manifest.name or not self.manifest.version:
            logger.error("Plugin missing required manifest fields")
            return False
        
        # Check dependencies
        for dep in self.manifest.dependencies:
            try:
                __import__(dep)
            except ImportError:
                logger.error(f"Plugin dependency not found: {dep}")
                return False
        
        # Validate configuration
        if self.config:
            required_configs = self.manifest.metadata.get('required_config', [])
            for req in required_configs:
                if req not in self.config:
                    logger.error(f"Plugin missing required config: {req}")
                    return False
        
        return True
    
    async def reload(self) -> None:
        await self.shutdown()
        await self.initialize()
        
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "type": self.manifest.plugin_type.value,
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None
        }


class GamePlugin(BasePlugin):
    
    @abstractmethod
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state"""
        return {
            'frame_number': getattr(self, 'frame_count', 0),
            'score': getattr(self, 'score', 0),
            'health': getattr(self, 'health', 100),
            'is_running': getattr(self, 'is_running', False),
            'timestamp': datetime.now().isoformat()
        }
    
    @abstractmethod 
    def get_observation_space(self) -> Any:
        """Get observation space definition"""
        return {
            'type': 'box',
            'shape': (84, 84, 3),
            'dtype': 'uint8',
            'low': 0,
            'high': 255
        }
    
    @abstractmethod
    def get_action_space(self) -> Any:
        """Get action space definition"""
        return {
            'type': 'discrete',
            'n': 10,
            'actions': ['UP', 'DOWN', 'LEFT', 'RIGHT', 'ACTION',
                       'JUMP', 'SHOOT', 'RELOAD', 'INTERACT', 'NOTHING']
        }


class AgentPlugin(BasePlugin):
    
    @abstractmethod
    async def act(self, observation: Any) -> Any:
        """Select action based on observation"""
        import numpy as np
        # Default implementation: random action
        if hasattr(self, 'action_space'):
            if self.action_space.get('type') == 'discrete':
                return np.random.randint(0, self.action_space.get('n', 10))
        return 0
    
    @abstractmethod
    async def learn(self, experience: Any) -> None:
        """Learn from experience"""
        if not hasattr(self, 'memory'):
            self.memory = []
        self.memory.append(experience)
        
        # Trigger training if buffer is full
        if len(self.memory) >= getattr(self, 'batch_size', 32):
            await self._train_batch(self.memory[-self.batch_size:])
    
    async def _train_batch(self, batch):
        """Train on a batch of experiences"""
        # Default training logic
        if hasattr(self, 'model') and hasattr(self.model, 'train'):
            losses = []
            for experience in batch:
                loss = self.model.train(experience)
                losses.append(loss)
            return sum(losses) / len(losses) if losses else 0.0
        return 0.0


class CapturePlugin(BasePlugin):
    
    @abstractmethod
    async def capture_frame(self, region: Optional[tuple] = None) -> Any:
        """Capture a frame from the screen"""
        import mss
        import numpy as np
        
        with mss.mss() as sct:
            if region:
                monitor = {"left": region[0], "top": region[1], 
                          "width": region[2], "height": region[3]}
            else:
                monitor = sct.monitors[1]
            
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)[:, :, :3]  # Remove alpha channel
            return frame
    
    @abstractmethod
    def get_capture_info(self) -> Dict[str, Any]:
        """Get capture backend information"""
        return {
            'backend': getattr(self, 'backend_name', 'mss'),
            'fps': getattr(self, 'current_fps', 0),
            'frames_captured': getattr(self, 'frame_count', 0),
            'region': getattr(self, 'capture_region', None),
            'last_capture_time': getattr(self, 'last_capture_time', None)
        }


class VisionPlugin(BasePlugin):
    
    @abstractmethod
    async def detect(self, frame: Any) -> Any:
        """Detect objects/features in frame"""
        detections = []
        
        # Basic template matching as default
        if hasattr(self, 'templates'):
            import cv2
            for name, template in self.templates.items():
                result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.8:  # Threshold
                    h, w = template.shape[:2]
                    detections.append({
                        'type': name,
                        'bbox': [max_loc[0], max_loc[1], w, h],
                        'confidence': max_val
                    })
        
        return detections
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get vision model information"""
        return {
            'model_type': getattr(self, 'model_type', 'unknown'),
            'model_name': getattr(self, 'model_name', 'default'),
            'input_shape': getattr(self, 'input_shape', (640, 640)),
            'num_classes': getattr(self, 'num_classes', 0),
            'confidence_threshold': getattr(self, 'confidence_threshold', 0.5)
        }


class InputPlugin(BasePlugin):
    
    @abstractmethod
    async def send_input(self, action: Any) -> None:
        """Send input action to the system"""
        import pyautogui
        
        if isinstance(action, str):
            # Keyboard action
            pyautogui.press(action)
        elif isinstance(action, dict):
            action_type = action.get('type')
            
            if action_type == 'key':
                pyautogui.press(action.get('key'))
            elif action_type == 'mouse_move':
                pyautogui.moveTo(action.get('x'), action.get('y'))
            elif action_type == 'mouse_click':
                pyautogui.click(button=action.get('button', 'left'))
            elif action_type == 'sequence':
                for sub_action in action.get('actions', []):
                    await self.send_input(sub_action)
        
        # Track input statistics
        if not hasattr(self, 'input_count'):
            self.input_count = 0
        self.input_count += 1
    
    @abstractmethod
    def get_input_info(self) -> Dict[str, Any]:
        """Get input controller information"""
        return {
            'backend': getattr(self, 'backend_name', 'pyautogui'),
            'inputs_sent': getattr(self, 'input_count', 0),
            'last_action': getattr(self, 'last_action', None),
            'human_like': getattr(self, 'human_like_enabled', False),
            'delay_range': getattr(self, 'delay_range', [0.05, 0.15])
        }