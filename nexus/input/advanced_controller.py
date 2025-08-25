"""Advanced input controller with comprehensive key mappings and game-specific actions"""

import asyncio
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import numpy as np
import structlog

from nexus.input.base import InputController

logger = structlog.get_logger()


class MouseButton(Enum):
    """Mouse button definitions"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


# Comprehensive US keyboard layout character mappings
CHARACTER_KEY_MAPPING = {
    # Numbers and symbols
    "`": ["grave"],
    "~": ["shift", "grave"],
    "1": ["1"],
    "!": ["shift", "1"],
    "2": ["2"],
    "@": ["shift", "2"],
    "3": ["3"],
    "#": ["shift", "3"],
    "4": ["4"],
    "$": ["shift", "4"],
    "5": ["5"],
    "%": ["shift", "5"],
    "6": ["6"],
    "^": ["shift", "6"],
    "7": ["7"],
    "&": ["shift", "7"],
    "8": ["8"],
    "*": ["shift", "8"],
    "9": ["9"],
    "(": ["shift", "9"],
    "0": ["0"],
    ")": ["shift", "0"],
    "-": ["minus"],
    "_": ["shift", "minus"],
    "=": ["equals"],
    "+": ["shift", "equals"],
    
    # Letters (lowercase and uppercase)
    "q": ["q"], "Q": ["shift", "q"],
    "w": ["w"], "W": ["shift", "w"],
    "e": ["e"], "E": ["shift", "e"],
    "r": ["r"], "R": ["shift", "r"],
    "t": ["t"], "T": ["shift", "t"],
    "y": ["y"], "Y": ["shift", "y"],
    "u": ["u"], "U": ["shift", "u"],
    "i": ["i"], "I": ["shift", "i"],
    "o": ["o"], "O": ["shift", "o"],
    "p": ["p"], "P": ["shift", "p"],
    "a": ["a"], "A": ["shift", "a"],
    "s": ["s"], "S": ["shift", "s"],
    "d": ["d"], "D": ["shift", "d"],
    "f": ["f"], "F": ["shift", "f"],
    "g": ["g"], "G": ["shift", "g"],
    "h": ["h"], "H": ["shift", "h"],
    "j": ["j"], "J": ["shift", "j"],
    "k": ["k"], "K": ["shift", "k"],
    "l": ["l"], "L": ["shift", "l"],
    "z": ["z"], "Z": ["shift", "z"],
    "x": ["x"], "X": ["shift", "x"],
    "c": ["c"], "C": ["shift", "c"],
    "v": ["v"], "V": ["shift", "v"],
    "b": ["b"], "B": ["shift", "b"],
    "n": ["n"], "N": ["shift", "n"],
    "m": ["m"], "M": ["shift", "m"],
    
    # Brackets and punctuation
    "[": ["left_bracket"],
    "{": ["shift", "left_bracket"],
    "]": ["right_bracket"],
    "}": ["shift", "right_bracket"],
    ";": ["semicolon"],
    ":": ["shift", "semicolon"],
    "'": ["apostrophe"],
    '"': ["shift", "apostrophe"],
    ",": ["comma"],
    "<": ["shift", "comma"],
    ".": ["period"],
    ">": ["shift", "period"],
    "/": ["slash"],
    "?": ["shift", "slash"],
    "\\": ["backslash"],
    "|": ["shift", "backslash"],
    
    # Special characters
    " ": ["space"],
    "\n": ["return"],
    "\t": ["tab"],
    "\b": ["backspace"]
}


class AdvancedInputController(InputController):
    """Advanced input controller with game-specific features"""
    
    def __init__(self, window_controller=None):
        super().__init__()
        self.window_controller = window_controller
        self.key_states: Dict[str, bool] = {}
        self.mouse_position: Tuple[int, int] = (0, 0)
        self.game_window_info: Optional[Dict[str, Any]] = None
        
    async def initialize(self):
        """Initialize the input controller"""
        await super().initialize()
        
        # Get initial game window info if available
        if self.window_controller:
            self.game_window_info = self.window_controller.get_foreground_window()
    
    async def type_string(self, text: str, delay: float = 0.05,
                         use_human_like: bool = False) -> None:
        """
        Type a string with proper character mapping.
        
        Args:
            text: Text to type
            delay: Delay between keystrokes
            use_human_like: Use human-like typing patterns
        """
        prev_char = None
        
        for char in text:
            # Get key combination for character
            keys = CHARACTER_KEY_MAPPING.get(char, [char.lower()])
            
            # Calculate delay
            if use_human_like and hasattr(self, 'human_like'):
                char_delay = self.human_like.get_typing_delay(char, prev_char)
            else:
                char_delay = delay
            
            # Type the character
            if len(keys) == 1:
                # Single key
                await self.tap_key(keys[0], duration=char_delay)
            else:
                # Key combination (e.g., shift + key)
                modifier = keys[0]
                key = keys[1]
                
                await self.press_key(modifier)
                await asyncio.sleep(0.01)
                await self.tap_key(key, duration=char_delay)
                await self.release_key(modifier)
            
            prev_char = char
    
    async def tap_keys(self, keys: List[str], duration: float = 0.05) -> None:
        """
        Tap multiple keys in sequence.
        
        Args:
            keys: List of keys to tap
            duration: Duration of each key press
        """
        for key in keys:
            await self.tap_key(key, duration)
            await asyncio.sleep(0.01)  # Small delay between keys
    
    async def combo_keys(self, keys: List[str], hold_duration: float = 0.1) -> None:
        """
        Press multiple keys simultaneously (combo/chord).
        
        Args:
            keys: List of keys to press together
            hold_duration: How long to hold the combo
        """
        # Press all keys
        for key in keys:
            await self.press_key(key)
            await asyncio.sleep(0.01)
        
        # Hold
        await asyncio.sleep(hold_duration)
        
        # Release all keys in reverse order
        for key in reversed(keys):
            await self.release_key(key)
            await asyncio.sleep(0.01)
    
    async def click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT,
                       duration: float = 0.05) -> None:
        """
        Click at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click
            duration: Click duration
        """
        await self.move_mouse(x, y)
        await asyncio.sleep(0.05)
        await self.click(button, duration)
    
    async def click_region(self, region: Tuple[int, int, int, int],
                          button: MouseButton = MouseButton.LEFT) -> None:
        """
        Click in the center of a screen region.
        
        Args:
            region: Region as (x1, y1, x2, y2)
            button: Mouse button to click
        """
        x1, y1, x2, y2 = region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        await self.click_at(center_x, center_y, button)
    
    async def click_sprite(self, sprite_location: Tuple[int, int],
                          sprite_size: Tuple[int, int],
                          button: MouseButton = MouseButton.LEFT,
                          offset: Tuple[int, int] = (0, 0)) -> None:
        """
        Click on a detected sprite.
        
        Args:
            sprite_location: Top-left corner of sprite
            sprite_size: Size of sprite (width, height)
            button: Mouse button to click
            offset: Offset from sprite center
        """
        x, y = sprite_location
        w, h = sprite_size
        
        # Calculate center with offset
        click_x = x + w // 2 + offset[0]
        click_y = y + h // 2 + offset[1]
        
        await self.click_at(click_x, click_y, button)
    
    async def click_text(self, text_location: Tuple[int, int],
                        button: MouseButton = MouseButton.LEFT) -> None:
        """
        Click on detected text location.
        
        Args:
            text_location: Location of text
            button: Mouse button to click
        """
        await self.click_at(text_location[0], text_location[1], button)
    
    async def drag(self, start: Tuple[int, int], end: Tuple[int, int],
                  button: MouseButton = MouseButton.LEFT,
                  duration: float = 1.0) -> None:
        """
        Drag from start to end position.
        
        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            button: Mouse button to use
            duration: Duration of drag
        """
        # Move to start
        await self.move_mouse(start[0], start[1])
        await asyncio.sleep(0.1)
        
        # Press button
        await self.mouse_down(button.value)
        await asyncio.sleep(0.05)
        
        # Move to end
        await self.move_mouse(end[0], end[1], duration=duration)
        
        # Release button
        await self.mouse_up(button.value)
    
    async def scroll(self, direction: str = "down", amount: int = 3) -> None:
        """
        Scroll the mouse wheel.
        
        Args:
            direction: "up" or "down"
            amount: Number of scroll units
        """
        scroll_direction = -1 if direction.lower() == "up" else 1
        
        for _ in range(amount):
            await self.mouse_scroll(0, scroll_direction)
            await asyncio.sleep(0.05)
    
    async def game_action(self, action: str, **kwargs) -> None:
        """
        Perform a game-specific action.
        
        Args:
            action: Action name (e.g., "jump", "attack", "menu")
            **kwargs: Additional parameters for the action
        """
        # Common game actions
        game_actions = {
            "jump": lambda: self.tap_key("space"),
            "crouch": lambda: self.press_key("ctrl"),
            "sprint": lambda: self.press_key("shift"),
            "attack": lambda: self.click(MouseButton.LEFT),
            "secondary_attack": lambda: self.click(MouseButton.RIGHT),
            "interact": lambda: self.tap_key("e"),
            "inventory": lambda: self.tap_key("i"),
            "menu": lambda: self.tap_key("escape"),
            "quick_save": lambda: self.tap_key("f5"),
            "quick_load": lambda: self.tap_key("f9"),
            
            # WASD movement
            "move_forward": lambda: self.press_key("w"),
            "move_backward": lambda: self.press_key("s"),
            "move_left": lambda: self.press_key("a"),
            "move_right": lambda: self.press_key("d"),
            "stop_move_forward": lambda: self.release_key("w"),
            "stop_move_backward": lambda: self.release_key("s"),
            "stop_move_left": lambda: self.release_key("a"),
            "stop_move_right": lambda: self.release_key("d"),
            
            # Arrow key movement
            "arrow_up": lambda: self.tap_key("up"),
            "arrow_down": lambda: self.tap_key("down"),
            "arrow_left": lambda: self.tap_key("left"),
            "arrow_right": lambda: self.tap_key("right"),
        }
        
        if action in game_actions:
            await game_actions[action]()
        else:
            logger.warning(f"Unknown game action: {action}")
    
    def ratios_to_coordinates(self, ratios: Tuple[float, float],
                            region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int]:
        """
        Convert relative ratios to absolute coordinates.
        
        Args:
            ratios: Relative position (0-1, 0-1)
            region: Optional region to use as reference
        
        Returns:
            Absolute coordinates (x, y)
        """
        if region:
            x1, y1, x2, y2 = region
            width = x2 - x1
            height = y2 - y1
            
            x = x1 + int(ratios[0] * width)
            y = y1 + int(ratios[1] * height)
        else:
            # Use game window or screen size
            if self.game_window_info:
                width = self.game_window_info.get("width", 1920)
                height = self.game_window_info.get("height", 1080)
                x_offset = self.game_window_info.get("x", 0)
                y_offset = self.game_window_info.get("y", 0)
            else:
                # Default to common resolution
                width = 1920
                height = 1080
                x_offset = 0
                y_offset = 0
            
            x = x_offset + int(ratios[0] * width)
            y = y_offset + int(ratios[1] * height)
        
        return (x, y)
    
    def coordinates_to_ratios(self, coordinates: Tuple[int, int],
                            region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, float]:
        """
        Convert absolute coordinates to relative ratios.
        
        Args:
            coordinates: Absolute position (x, y)
            region: Optional region to use as reference
        
        Returns:
            Relative ratios (0-1, 0-1)
        """
        x, y = coordinates
        
        if region:
            x1, y1, x2, y2 = region
            width = x2 - x1
            height = y2 - y1
            
            ratio_x = (x - x1) / width if width > 0 else 0
            ratio_y = (y - y1) / height if height > 0 else 0
        else:
            # Use game window or screen size
            if self.game_window_info:
                width = self.game_window_info.get("width", 1920)
                height = self.game_window_info.get("height", 1080)
                x_offset = self.game_window_info.get("x", 0)
                y_offset = self.game_window_info.get("y", 0)
            else:
                width = 1920
                height = 1080
                x_offset = 0
                y_offset = 0
            
            ratio_x = (x - x_offset) / width if width > 0 else 0
            ratio_y = (y - y_offset) / height if height > 0 else 0
        
        # Clamp to 0-1 range
        ratio_x = max(0, min(1, ratio_x))
        ratio_y = max(0, min(1, ratio_y))
        
        return (ratio_x, ratio_y)
    
    async def record_macro(self, duration: float = 10.0) -> List[Dict[str, Any]]:
        """
        Record user input for creating macros.
        
        Args:
            duration: How long to record
        
        Returns:
            List of recorded actions
        """
        logger.info(f"Recording macro for {duration} seconds...")
        actions = []
        
        # This would need actual input hooks implementation
        # Placeholder for now
        start_time = time.time()
        while time.time() - start_time < duration:
            await asyncio.sleep(0.01)
        
        logger.info(f"Recorded {len(actions)} actions")
        return actions
    
    async def play_macro(self, actions: List[Dict[str, Any]]) -> None:
        """
        Play back recorded macro actions.
        
        Args:
            actions: List of actions to play
        """
        for action in actions:
            action_type = action.get("type")
            
            if action_type == "key":
                await self.tap_key(action["key"], action.get("duration", 0.05))
            elif action_type == "mouse_move":
                await self.move_mouse(action["x"], action["y"])
            elif action_type == "mouse_click":
                await self.click(MouseButton(action.get("button", "left")))
            elif action_type == "delay":
                await asyncio.sleep(action["duration"])
            
            # Small delay between actions
            await asyncio.sleep(0.01)