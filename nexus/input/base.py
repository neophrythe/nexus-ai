from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import asyncio
import time
import structlog

logger = structlog.get_logger()


class InputType(Enum):
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GAMEPAD = "gamepad"
    COMPOSITE = "composite"


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class InputAction:
    action_type: str  # "key_press", "key_release", "mouse_move", "mouse_click", etc.
    data: Dict[str, Any]
    timestamp: float = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class InputController(ABC):
    
    def __init__(self, human_like: bool = True, delay_range: Tuple[float, float] = (0.05, 0.15)):
        self.human_like = human_like
        self.delay_range = delay_range
        self.action_history: List[InputAction] = []
        self.max_history = 1000
        self._last_action_time = 0
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the input controller"""
        self.action_history = []
        self._last_action_time = time.time()
        logger.info(f"Input controller initialized with human_like={self.human_like}")
    
    @abstractmethod
    async def key_press(self, key: str) -> None:
        """Press a key"""
        import pyautogui
        pyautogui.keyDown(key)
        action = InputAction("key_press", {"key": key})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def key_release(self, key: str) -> None:
        """Release a key"""
        import pyautogui
        pyautogui.keyUp(key)
        action = InputAction("key_release", {"key": key})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def key_tap(self, key: str, duration: Optional[float] = None) -> None:
        """Press and release a key"""
        import pyautogui
        if duration:
            pyautogui.keyDown(key)
            await asyncio.sleep(duration)
            pyautogui.keyUp(key)
        else:
            pyautogui.press(key)
        action = InputAction("key_tap", {"key": key, "duration": duration or 0.05})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def type_text(self, text: str, interval: Optional[float] = None) -> None:
        """Type text with optional interval between characters"""
        import pyautogui
        import random
        
        for char in text:
            pyautogui.write(char)
            if interval:
                await asyncio.sleep(interval)
            elif self.human_like:
                await asyncio.sleep(random.uniform(0.05, 0.15))
        
        action = InputAction("type_text", {"text": text, "interval": interval})
        self._add_to_history(action)
    
    @abstractmethod
    async def mouse_move(self, x: int, y: int, duration: Optional[float] = None) -> None:
        """Move mouse to position"""
        import pyautogui
        
        if duration and duration > 0:
            pyautogui.moveTo(x, y, duration=duration)
        else:
            pyautogui.moveTo(x, y)
        
        action = InputAction("mouse_move", {"x": x, "y": y, "duration": duration})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def mouse_click(self, button: MouseButton = MouseButton.LEFT, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Click mouse button"""
        import pyautogui
        
        if x is not None and y is not None:
            pyautogui.click(x=x, y=y, button=button.value)
        else:
            pyautogui.click(button=button.value)
        
        action = InputAction("mouse_click", {"button": button.value, "x": x, "y": y})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def mouse_down(self, button: MouseButton = MouseButton.LEFT) -> None:
        """Press mouse button"""
        import pyautogui
        pyautogui.mouseDown(button=button.value)
        action = InputAction("mouse_down", {"button": button.value})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def mouse_up(self, button: MouseButton = MouseButton.LEFT) -> None:
        """Release mouse button"""
        import pyautogui
        pyautogui.mouseUp(button=button.value)
        action = InputAction("mouse_up", {"button": button.value})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def mouse_scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll mouse wheel"""
        import pyautogui
        
        if x is not None and y is not None:
            pyautogui.moveTo(x, y)
        pyautogui.scroll(clicks)
        
        action = InputAction("mouse_scroll", {"clicks": clicks, "x": x, "y": y})
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    async def mouse_drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                         button: MouseButton = MouseButton.LEFT, duration: float = 1.0) -> None:
        """Drag mouse from start to end position"""
        import pyautogui
        
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=duration, button=button.value)
        
        action = InputAction("mouse_drag", {
            "start_x": start_x, "start_y": start_y,
            "end_x": end_x, "end_y": end_y,
            "button": button.value, "duration": duration
        })
        self._add_to_history(action)
        if self.human_like:
            await self._add_human_delay()
    
    @abstractmethod
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        import pyautogui
        pos = pyautogui.position()
        return (pos.x, pos.y)
    
    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        import pyautogui
        size = pyautogui.size()
        return (size.width, size.height)
    
    async def combo(self, keys: List[str], hold_time: float = 0.1) -> None:
        """Execute key combination"""
        # Press all keys
        for key in keys:
            await self.key_press(key)
            await asyncio.sleep(0.01)
        
        # Hold
        await asyncio.sleep(hold_time)
        
        # Release in reverse order
        for key in reversed(keys):
            await self.key_release(key)
            await asyncio.sleep(0.01)
    
    async def multi_click(self, x: int, y: int, clicks: int = 2, 
                         interval: float = 0.1, button: MouseButton = MouseButton.LEFT) -> None:
        """Perform multiple clicks"""
        await self.mouse_move(x, y)
        for _ in range(clicks):
            await self.mouse_click(button)
            if interval > 0:
                await asyncio.sleep(interval)
    
    def _add_to_history(self, action: InputAction) -> None:
        """Add action to history"""
        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
    
    async def _add_human_delay(self) -> None:
        """Add human-like delay between actions"""
        if self.human_like:
            import random
            delay = random.uniform(*self.delay_range)
            await asyncio.sleep(delay)
    
    def get_action_history(self, n: Optional[int] = None) -> List[InputAction]:
        """Get recent action history"""
        if n is None:
            return self.action_history.copy()
        return self.action_history[-n:] if n <= len(self.action_history) else self.action_history.copy()
    
    def clear_history(self) -> None:
        """Clear action history"""
        self.action_history.clear()
    
    async def replay_actions(self, actions: List[InputAction], speed_multiplier: float = 1.0) -> None:
        """Replay a sequence of actions"""
        for i, action in enumerate(actions):
            if i > 0:
                # Calculate time delta
                time_delta = (action.timestamp - actions[i-1].timestamp) / speed_multiplier
                await asyncio.sleep(max(0, time_delta))
            
            # Execute action based on type
            if action.action_type == "key_press":
                await self.key_press(action.data["key"])
            elif action.action_type == "key_release":
                await self.key_release(action.data["key"])
            elif action.action_type == "key_tap":
                await self.key_tap(action.data["key"])
            elif action.action_type == "mouse_move":
                await self.mouse_move(action.data["x"], action.data["y"])
            elif action.action_type == "mouse_click":
                button = MouseButton(action.data.get("button", "left"))
                await self.mouse_click(button, action.data.get("x"), action.data.get("y"))
            elif action.action_type == "type_text":
                await self.type_text(action.data["text"])
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        self.action_history.clear()
        self._last_action_time = 0
        logger.info("Input controller cleaned up")


class InputRecorder:
    """Record and save input sequences"""
    
    def __init__(self, controller: InputController):
        self.controller = controller
        self.recording = False
        self.recorded_actions: List[InputAction] = []
        self.start_time = None
    
    def start_recording(self) -> None:
        """Start recording input actions"""
        self.recording = True
        self.recorded_actions = []
        self.start_time = time.time()
        logger.info("Input recording started")
    
    def stop_recording(self) -> List[InputAction]:
        """Stop recording and return actions"""
        self.recording = False
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Input recording stopped - {len(self.recorded_actions)} actions in {duration:.2f}s")
        return self.recorded_actions.copy()
    
    def add_action(self, action: InputAction) -> None:
        """Add action to recording"""
        if self.recording:
            self.recorded_actions.append(action)
    
    def save_recording(self, path: str) -> None:
        """Save recording to file"""
        import json
        
        data = {
            "actions": [
                {
                    "type": action.action_type,
                    "data": action.data,
                    "timestamp": action.timestamp,
                    "duration": action.duration
                }
                for action in self.recorded_actions
            ],
            "duration": self.recorded_actions[-1].timestamp - self.recorded_actions[0].timestamp if self.recorded_actions else 0
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Recording saved to {path}")
    
    def load_recording(self, path: str) -> List[InputAction]:
        """Load recording from file"""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        actions = [
            InputAction(
                action_type=item["type"],
                data=item["data"],
                timestamp=item["timestamp"],
                duration=item.get("duration", 0)
            )
            for item in data["actions"]
        ]
        
        logger.info(f"Loaded {len(actions)} actions from {path}")
        return actions