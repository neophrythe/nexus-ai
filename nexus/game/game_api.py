"""Game API system for complex input management and game control - adapted from SerpentAI"""

import itertools
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import structlog

from nexus.input.advanced_controller import AdvancedInputController, MouseButton

logger = structlog.get_logger()


class InputType(Enum):
    """Types of game inputs"""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GAMEPAD = "gamepad"
    COMPOSITE = "composite"


@dataclass
class GameInput:
    """Single game input definition"""
    name: str
    input_type: InputType
    keys: List[str] = field(default_factory=list)
    mouse_button: Optional[MouseButton] = None
    mouse_position: Optional[Tuple[int, int]] = None
    duration: float = 0.05
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameInputAxis:
    """Collection of related game inputs forming an axis"""
    name: str
    inputs: Dict[str, GameInput]
    exclusive: bool = True  # Only one input active at a time
    metadata: Dict[str, Any] = field(default_factory=dict)


class GameAPI:
    """Advanced game API for managing complex input combinations and game control"""
    
    def __init__(self, game_name: str, input_controller: Optional[AdvancedInputController] = None):
        """
        Initialize Game API.
        
        Args:
            game_name: Name of the game
            input_controller: Input controller instance
        """
        self.game_name = game_name
        self.input_controller = input_controller or AdvancedInputController()
        
        # Game input definitions
        self.game_inputs: Dict[str, GameInputAxis] = {}
        self.input_combinations: Dict[str, List[GameInput]] = {}
        
        # State tracking
        self.current_state: Dict[str, Any] = {}
        self.input_history: List[Tuple[float, str, Any]] = []
        self.max_history_size = 1000
        
        # Callbacks
        self.state_callbacks: Dict[str, List[Callable]] = {}
        
        # Initialize common game inputs
        self._initialize_common_inputs()
    
    def _initialize_common_inputs(self):
        """Initialize common game input patterns"""
        
        # Movement axis (WASD)
        self.register_input_axis(
            "movement",
            {
                "forward": GameInput("forward", InputType.KEYBOARD, ["w"]),
                "backward": GameInput("backward", InputType.KEYBOARD, ["s"]),
                "left": GameInput("left", InputType.KEYBOARD, ["a"]),
                "right": GameInput("right", InputType.KEYBOARD, ["d"]),
                "none": GameInput("none", InputType.KEYBOARD, [])
            },
            exclusive=True
        )
        
        # Arrow keys movement
        self.register_input_axis(
            "arrows",
            {
                "up": GameInput("up", InputType.KEYBOARD, ["up"]),
                "down": GameInput("down", InputType.KEYBOARD, ["down"]),
                "left": GameInput("left", InputType.KEYBOARD, ["left"]),
                "right": GameInput("right", InputType.KEYBOARD, ["right"]),
                "none": GameInput("none", InputType.KEYBOARD, [])
            },
            exclusive=True
        )
        
        # Actions
        self.register_input_axis(
            "actions",
            {
                "jump": GameInput("jump", InputType.KEYBOARD, ["space"]),
                "crouch": GameInput("crouch", InputType.KEYBOARD, ["ctrl"]),
                "sprint": GameInput("sprint", InputType.KEYBOARD, ["shift"]),
                "interact": GameInput("interact", InputType.KEYBOARD, ["e"]),
                "none": GameInput("none", InputType.KEYBOARD, [])
            },
            exclusive=False
        )
        
        # Mouse actions
        self.register_input_axis(
            "mouse",
            {
                "left_click": GameInput("left_click", InputType.MOUSE, mouse_button=MouseButton.LEFT),
                "right_click": GameInput("right_click", InputType.MOUSE, mouse_button=MouseButton.RIGHT),
                "middle_click": GameInput("middle_click", InputType.MOUSE, mouse_button=MouseButton.MIDDLE),
                "none": GameInput("none", InputType.MOUSE)
            },
            exclusive=True
        )
    
    def register_input_axis(self, name: str, inputs: Dict[str, GameInput], exclusive: bool = True):
        """
        Register a new input axis.
        
        Args:
            name: Axis name
            inputs: Dictionary of inputs for this axis
            exclusive: Whether inputs are mutually exclusive
        """
        self.game_inputs[name] = GameInputAxis(name, inputs, exclusive)
        logger.debug(f"Registered input axis '{name}' with {len(inputs)} inputs")
    
    def combine_game_inputs(self, combination: List[Union[str, List[str]]]) -> Dict[str, List[GameInput]]:
        """
        Combine game input axes into a flattened collection.
        
        Args:
            combination: List of axis names or grouped axis names
        
        Returns:
            Dictionary mapping compound labels to input lists
        """
        # Validation
        for entry in combination:
            if isinstance(entry, list):
                for item in entry:
                    if item not in self.game_inputs:
                        raise ValueError(f"Unknown game input axis: {item}")
            else:
                if entry not in self.game_inputs:
                    raise ValueError(f"Unknown game input axis: {entry}")
        
        # Prepare axes for combination
        game_input_axes = []
        
        for entry in combination:
            if isinstance(entry, str):
                # Single axis
                axis = self.game_inputs[entry]
                game_input_axes.append(axis.inputs)
            elif isinstance(entry, list):
                # Grouped axes - concatenate their inputs
                concatenated_inputs = {}
                for axis_name in entry:
                    axis = self.game_inputs[axis_name]
                    concatenated_inputs.update(axis.inputs)
                game_input_axes.append(concatenated_inputs)
        
        # Generate all combinations
        combined_inputs = {}
        
        if not game_input_axes:
            return combined_inputs
        
        # Get all possible combinations
        for input_combo in itertools.product(*[axis.keys() for axis in game_input_axes]):
            # Build compound label and input list
            compound_label = " - ".join(input_combo)
            input_list = []
            
            for axis_idx, input_key in enumerate(input_combo):
                game_input = game_input_axes[axis_idx][input_key]
                if game_input.name != "none":  # Skip "none" inputs
                    input_list.append(game_input)
            
            combined_inputs[compound_label] = input_list
        
        # Store for later use
        self.input_combinations = combined_inputs
        
        logger.info(f"Generated {len(combined_inputs)} input combinations from {len(combination)} axes")
        return combined_inputs
    
    async def execute_input(self, input_label: str, duration: Optional[float] = None):
        """
        Execute a combined input by label.
        
        Args:
            input_label: Label of the combined input
            duration: Override duration for the input
        """
        if input_label not in self.input_combinations:
            logger.warning(f"Unknown input combination: {input_label}")
            return
        
        inputs = self.input_combinations[input_label]
        
        # Execute all inputs in the combination
        for game_input in inputs:
            await self._execute_single_input(game_input, duration)
        
        # Record in history
        import time
        self.input_history.append((time.time(), input_label, inputs))
        
        # Trim history if needed
        if len(self.input_history) > self.max_history_size:
            self.input_history.pop(0)
    
    async def _execute_single_input(self, game_input: GameInput, duration: Optional[float] = None):
        """Execute a single game input"""
        duration = duration or game_input.duration
        
        if game_input.input_type == InputType.KEYBOARD:
            if len(game_input.keys) == 1:
                # Single key
                await self.input_controller.tap_key(game_input.keys[0], duration)
            elif len(game_input.keys) > 1:
                # Key combination
                await self.input_controller.combo_keys(game_input.keys, duration)
        
        elif game_input.input_type == InputType.MOUSE:
            if game_input.mouse_button:
                if game_input.mouse_position:
                    # Click at position
                    await self.input_controller.click_at(
                        game_input.mouse_position[0],
                        game_input.mouse_position[1],
                        game_input.mouse_button,
                        duration
                    )
                else:
                    # Click at current position
                    await self.input_controller.click(game_input.mouse_button, duration)
        
        elif game_input.input_type == InputType.COMPOSITE:
            # Handle composite inputs (keyboard + mouse)
            tasks = []
            if game_input.keys:
                tasks.append(self.input_controller.combo_keys(game_input.keys, duration))
            if game_input.mouse_button:
                tasks.append(self.input_controller.click(game_input.mouse_button, duration))
            
            if tasks:
                await asyncio.gather(*tasks)
    
    def create_action_space(self, axes: List[str]) -> List[str]:
        """
        Create action space for reinforcement learning.
        
        Args:
            axes: List of axis names to include
        
        Returns:
            List of action labels
        """
        combined = self.combine_game_inputs(axes)
        return list(combined.keys())
    
    def get_action_mapping(self, axes: List[str]) -> Dict[int, str]:
        """
        Get action index to label mapping for RL agents.
        
        Args:
            axes: List of axis names to include
        
        Returns:
            Dictionary mapping action indices to labels
        """
        action_space = self.create_action_space(axes)
        return {i: label for i, label in enumerate(action_space)}
    
    async def execute_action_sequence(self, sequence: List[Tuple[str, float]]):
        """
        Execute a sequence of actions with timing.
        
        Args:
            sequence: List of (action_label, delay) tuples
        """
        for action_label, delay in sequence:
            await self.execute_input(action_label)
            if delay > 0:
                await asyncio.sleep(delay)
    
    def register_state_callback(self, state_name: str, callback: Callable):
        """
        Register callback for state changes.
        
        Args:
            state_name: Name of the state to monitor
            callback: Function to call on state change
        """
        if state_name not in self.state_callbacks:
            self.state_callbacks[state_name] = []
        self.state_callbacks[state_name].append(callback)
    
    def update_state(self, state_name: str, value: Any):
        """
        Update game state and trigger callbacks.
        
        Args:
            state_name: Name of the state
            value: New state value
        """
        old_value = self.current_state.get(state_name)
        self.current_state[state_name] = value
        
        # Trigger callbacks if value changed
        if old_value != value and state_name in self.state_callbacks:
            for callback in self.state_callbacks[state_name]:
                try:
                    callback(old_value, value)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    def get_input_statistics(self) -> Dict[str, Any]:
        """Get statistics about input usage"""
        if not self.input_history:
            return {}
        
        # Count input usage
        input_counts = {}
        for _, label, _ in self.input_history:
            input_counts[label] = input_counts.get(label, 0) + 1
        
        # Calculate timing statistics
        timestamps = [t for t, _, _ in self.input_history]
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            actions_per_second = 1.0 / avg_interval if avg_interval > 0 else 0
        else:
            actions_per_second = 0
        
        return {
            'total_inputs': len(self.input_history),
            'unique_inputs': len(set(label for _, label, _ in self.input_history)),
            'input_counts': input_counts,
            'most_common': max(input_counts.items(), key=lambda x: x[1])[0] if input_counts else None,
            'actions_per_second': actions_per_second
        }
    
    def clear_history(self):
        """Clear input history"""
        self.input_history.clear()
    
    def save_input_config(self, filepath: str):
        """Save input configuration to file"""
        import json
        
        config = {
            'game_name': self.game_name,
            'axes': {}
        }
        
        for axis_name, axis in self.game_inputs.items():
            config['axes'][axis_name] = {
                'exclusive': axis.exclusive,
                'inputs': {
                    key: {
                        'name': inp.name,
                        'type': inp.input_type.value,
                        'keys': inp.keys,
                        'mouse_button': inp.mouse_button.value if inp.mouse_button else None,
                        'mouse_position': inp.mouse_position,
                        'duration': inp.duration
                    }
                    for key, inp in axis.inputs.items()
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved input configuration to {filepath}")
    
    def load_input_config(self, filepath: str):
        """Load input configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.game_name = config['game_name']
        self.game_inputs.clear()
        
        for axis_name, axis_config in config['axes'].items():
            inputs = {}
            for key, inp_config in axis_config['inputs'].items():
                game_input = GameInput(
                    name=inp_config['name'],
                    input_type=InputType(inp_config['type']),
                    keys=inp_config.get('keys', []),
                    mouse_button=MouseButton(inp_config['mouse_button']) if inp_config.get('mouse_button') else None,
                    mouse_position=tuple(inp_config['mouse_position']) if inp_config.get('mouse_position') else None,
                    duration=inp_config.get('duration', 0.05)
                )
                inputs[key] = game_input
            
            self.register_input_axis(axis_name, inputs, axis_config.get('exclusive', True))
        
        logger.info(f"Loaded input configuration from {filepath}")