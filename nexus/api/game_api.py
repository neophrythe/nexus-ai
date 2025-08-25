"""Unified Game API System for Nexus Framework"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from abc import ABC, abstractmethod
import itertools

from nexus.input.input_controller import InputController
from nexus.capture.screen_capture import ScreenCapture
from nexus.window.window_controller import WindowController
from nexus.launchers.game_launcher import GameLauncherFactory, LaunchConfig
from nexus.ocr.ocr_engine import OCRService

logger = structlog.get_logger()


class GameState(Enum):
    """Game state enumeration"""
    NOT_LAUNCHED = "not_launched"
    LAUNCHING = "launching"
    MENU = "menu"
    LOADING = "loading"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


@dataclass
class GameInput:
    """Game input definition"""
    name: str
    key: Optional[str] = None
    button: Optional[str] = None
    axis: Optional[str] = None
    value: Optional[Any] = None
    hold_time: float = 0.0
    delay_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameAction:
    """High-level game action"""
    name: str
    inputs: List[GameInput]
    description: str = ""
    cooldown: float = 0.0
    last_executed: float = 0.0
    
    def is_ready(self) -> bool:
        """Check if action is ready (cooldown expired)"""
        return time.time() - self.last_executed >= self.cooldown
    
    def execute(self, input_controller: InputController):
        """Execute the action"""
        for game_input in self.inputs:
            if game_input.key:
                if game_input.hold_time > 0:
                    input_controller.press_and_hold_key(game_input.key, game_input.hold_time)
                else:
                    input_controller.press_key(game_input.key)
            
            elif game_input.button:
                input_controller.click(game_input.button)
            
            if game_input.delay_after > 0:
                time.sleep(game_input.delay_after)
        
        self.last_executed = time.time()


class BaseGameAPI(ABC):
    """Base class for game-specific APIs"""
    
    def __init__(self, game_name: str, launch_config: Optional[LaunchConfig] = None):
        """
        Initialize game API
        
        Args:
            game_name: Name of the game
            launch_config: Launch configuration
        """
        self.game_name = game_name
        self.launch_config = launch_config
        
        # Core components
        self.window_controller = WindowController()
        self.screen_capture = ScreenCapture()
        self.input_controller = None
        self.ocr_service = OCRService()
        
        # Game state
        self.game_launcher = None
        self.window_info = None
        self.current_state = GameState.NOT_LAUNCHED
        self.current_frame = None
        
        # Input definitions
        self.game_inputs: Dict[str, GameInput] = {}
        self.game_actions: Dict[str, GameAction] = {}
        
        # Initialize game-specific inputs
        self._define_inputs()
        self._define_actions()
        
        logger.info(f"Initialized Game API for {game_name}")
    
    @abstractmethod
    def _define_inputs(self):
        """Define game-specific inputs"""
        pass
    
    @abstractmethod
    def _define_actions(self):
        """Define game-specific actions"""
        pass
    
    @abstractmethod
    def detect_game_state(self, frame: Optional[np.ndarray] = None) -> GameState:
        """
        Detect current game state
        
        Args:
            frame: Game frame to analyze
        
        Returns:
            Current game state
        """
        pass
    
    def launch(self) -> bool:
        """
        Launch the game
        
        Returns:
            True if successful
        """
        if not self.launch_config:
            logger.error("No launch configuration provided")
            return False
        
        try:
            self.current_state = GameState.LAUNCHING
            self.game_launcher = GameLauncherFactory.launch_game(self.launch_config)
            
            # Wait for window
            if self.launch_config.window_name:
                self.window_info = self.window_controller.locate_window(
                    self.launch_config.window_name
                )
                
                if self.window_info:
                    # Initialize input controller for the window
                    self.input_controller = InputController(
                        window_id=self.window_info.window_id
                    )
                    
                    self.current_state = GameState.MENU
                    logger.info(f"Game launched successfully: {self.game_name}")
                    return True
            
            logger.warning("Game window not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to launch game: {e}")
            self.current_state = GameState.NOT_LAUNCHED
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture current game frame
        
        Returns:
            Game frame as numpy array
        """
        if self.window_info:
            frame = self.window_controller.capture_window(self.window_info.window_id)
            if frame is not None:
                self.current_frame = frame
                return frame
        
        # Fallback to screen capture
        frame = self.screen_capture.capture()
        self.current_frame = frame
        return frame
    
    def send_input(self, input_name: str):
        """
        Send input to game
        
        Args:
            input_name: Name of input to send
        """
        if not self.input_controller:
            logger.error("Input controller not initialized")
            return
        
        if input_name in self.game_inputs:
            game_input = self.game_inputs[input_name]
            
            if game_input.key:
                self.input_controller.press_key(game_input.key)
            elif game_input.button:
                self.input_controller.click(game_input.button)
    
    def perform_action(self, action_name: str) -> bool:
        """
        Perform high-level game action
        
        Args:
            action_name: Name of action to perform
        
        Returns:
            True if action was performed
        """
        if action_name not in self.game_actions:
            logger.error(f"Unknown action: {action_name}")
            return False
        
        action = self.game_actions[action_name]
        
        if not action.is_ready():
            logger.debug(f"Action {action_name} on cooldown")
            return False
        
        if not self.input_controller:
            logger.error("Input controller not initialized")
            return False
        
        action.execute(self.input_controller)
        logger.debug(f"Performed action: {action_name}")
        return True
    
    def combine_inputs(self, combination: List[Union[str, List[str]]]) -> List[Dict[str, Any]]:
        """
        Combine game inputs for complex actions
        
        Args:
            combination: List of input names or groups
        
        Returns:
            Flattened list of input combinations
        """
        # Validate inputs
        for entry in combination:
            if isinstance(entry, list):
                for item in entry:
                    if item not in self.game_inputs:
                        raise ValueError(f"Unknown input: {item}")
            else:
                if entry not in self.game_inputs:
                    raise ValueError(f"Unknown input: {entry}")
        
        # Build input groups
        input_groups = []
        
        for entry in combination:
            if isinstance(entry, str):
                input_groups.append({entry: self.game_inputs[entry]})
            elif isinstance(entry, list):
                group = {name: self.game_inputs[name] for name in entry}
                input_groups.append(group)
        
        # Generate combinations
        if len(input_groups) == 1:
            return list(input_groups[0].values())
        
        # Create cartesian product of input groups
        combinations = []
        for combo in itertools.product(*[g.values() for g in input_groups]):
            combinations.append(list(combo))
        
        return combinations
    
    def wait_for_state(self, target_state: GameState, timeout: float = 30) -> bool:
        """
        Wait for specific game state
        
        Args:
            target_state: Target game state
            timeout: Maximum wait time
        
        Returns:
            True if state reached
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame = self.capture_frame()
            current_state = self.detect_game_state(frame)
            
            if current_state == target_state:
                self.current_state = current_state
                return True
            
            time.sleep(0.5)
        
        logger.warning(f"Timeout waiting for state: {target_state}")
        return False
    
    def read_text(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Read text from game screen
        
        Args:
            region: Region to read from (x, y, width, height)
        
        Returns:
            Extracted text
        """
        frame = self.current_frame or self.capture_frame()
        
        if frame is None:
            return ""
        
        return self.ocr_service.get_text(frame, region=region)
    
    def find_text(self, target_text: str) -> List[Tuple[int, int, int, int]]:
        """
        Find text on game screen
        
        Args:
            target_text: Text to find
        
        Returns:
            List of bounding boxes
        """
        frame = self.current_frame or self.capture_frame()
        
        if frame is None:
            return []
        
        results = self.ocr_service.find_text(frame, target_text)
        return [r.bbox for r in results]
    
    def click_text(self, target_text: str) -> bool:
        """
        Click on text in game
        
        Args:
            target_text: Text to click
        
        Returns:
            True if text was found and clicked
        """
        bboxes = self.find_text(target_text)
        
        if not bboxes:
            logger.warning(f"Text not found: {target_text}")
            return False
        
        # Click center of first match
        x, y, w, h = bboxes[0]
        center_x = x + w // 2
        center_y = y + h // 2
        
        if self.input_controller:
            self.input_controller.move(center_x, center_y)
            self.input_controller.click()
            return True
        
        return False
    
    def get_game_info(self) -> Dict[str, Any]:
        """
        Get game information
        
        Returns:
            Dictionary of game information
        """
        return {
            "name": self.game_name,
            "state": self.current_state.value,
            "window": self.window_info.__dict__ if self.window_info else None,
            "is_running": self.game_launcher.is_running() if self.game_launcher else False,
            "inputs": list(self.game_inputs.keys()),
            "actions": list(self.game_actions.keys())
        }
    
    def shutdown(self):
        """Shutdown game and clean up resources"""
        if self.game_launcher:
            self.game_launcher.terminate()
        
        self.current_state = GameState.NOT_LAUNCHED
        logger.info(f"Game API shutdown: {self.game_name}")


class GenericGameAPI(BaseGameAPI):
    """Generic game API for common games"""
    
    def _define_inputs(self):
        """Define common game inputs"""
        # Movement
        self.game_inputs["up"] = GameInput("up", key="w")
        self.game_inputs["down"] = GameInput("down", key="s")
        self.game_inputs["left"] = GameInput("left", key="a")
        self.game_inputs["right"] = GameInput("right", key="d")
        
        # Actions
        self.game_inputs["jump"] = GameInput("jump", key="space")
        self.game_inputs["action"] = GameInput("action", key="e")
        self.game_inputs["attack"] = GameInput("attack", button="left")
        self.game_inputs["defend"] = GameInput("defend", button="right")
        
        # Menu
        self.game_inputs["pause"] = GameInput("pause", key="escape")
        self.game_inputs["confirm"] = GameInput("confirm", key="enter")
        self.game_inputs["cancel"] = GameInput("cancel", key="escape")
    
    def _define_actions(self):
        """Define common game actions"""
        # Movement combos
        self.game_actions["move_forward"] = GameAction(
            "move_forward",
            [self.game_inputs["up"]],
            "Move forward"
        )
        
        self.game_actions["jump_forward"] = GameAction(
            "jump_forward",
            [
                GameInput("jump", key="space"),
                GameInput("forward", key="w", hold_time=0.5)
            ],
            "Jump forward",
            cooldown=1.0
        )
        
        # Combat
        self.game_actions["combo_attack"] = GameAction(
            "combo_attack",
            [
                GameInput("attack1", button="left", delay_after=0.2),
                GameInput("attack2", button="left", delay_after=0.2),
                GameInput("attack3", button="left", delay_after=0.5)
            ],
            "Combo attack",
            cooldown=2.0
        )
    
    def detect_game_state(self, frame: Optional[np.ndarray] = None) -> GameState:
        """Basic game state detection"""
        if not self.game_launcher or not self.game_launcher.is_running():
            return GameState.NOT_LAUNCHED
        
        if frame is None:
            frame = self.capture_frame()
        
        if frame is None:
            return GameState.UNKNOWN
        
        # Simple detection based on text
        text = self.read_text()
        text_lower = text.lower()
        
        if "menu" in text_lower or "start" in text_lower:
            return GameState.MENU
        elif "loading" in text_lower:
            return GameState.LOADING
        elif "game over" in text_lower or "defeated" in text_lower:
            return GameState.GAME_OVER
        elif "paused" in text_lower:
            return GameState.PAUSED
        
        # Default to playing if game is running
        return GameState.PLAYING


# Factory for creating game APIs
class GameAPIFactory:
    """Factory for creating game-specific APIs"""
    
    registered_apis: Dict[str, type] = {}
    
    @classmethod
    def register(cls, game_name: str, api_class: type):
        """Register a game API class"""
        cls.registered_apis[game_name.lower()] = api_class
    
    @classmethod
    def create(cls, game_name: str, launch_config: Optional[LaunchConfig] = None) -> BaseGameAPI:
        """
        Create game API instance
        
        Args:
            game_name: Name of the game
            launch_config: Launch configuration
        
        Returns:
            Game API instance
        """
        game_key = game_name.lower()
        
        if game_key in cls.registered_apis:
            api_class = cls.registered_apis[game_key]
            return api_class(game_name, launch_config)
        
        # Return generic API as fallback
        logger.info(f"Using generic API for {game_name}")
        return GenericGameAPI(game_name, launch_config)