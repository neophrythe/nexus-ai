"""
Controller Mapper for Button Remapping and Profiles

Provides button remapping, profile management, and input transformation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import structlog

from nexus.input.gamepad.gamepad_base import (
    GamepadBase, Button, Axis, ControllerState, ControllerEvent
)

logger = structlog.get_logger()


@dataclass
class MappingProfile:
    """Controller mapping profile."""
    name: str
    description: str = ""
    button_map: Dict[Button, Button] = field(default_factory=dict)
    axis_map: Dict[Axis, Axis] = field(default_factory=dict)
    button_to_axis: Dict[Button, Tuple[Axis, float]] = field(default_factory=dict)
    axis_to_button: Dict[Axis, List[Tuple[Button, float, float]]] = field(default_factory=dict)
    axis_invert: Dict[Axis, bool] = field(default_factory=dict)
    axis_sensitivity: Dict[Axis, float] = field(default_factory=dict)
    axis_deadzone: Dict[Axis, float] = field(default_factory=dict)
    enabled: bool = True


class ControllerMapper:
    """
    Maps controller inputs with profiles and transformations.
    
    Features:
    - Button remapping
    - Axis remapping and inversion
    - Button-to-axis conversion
    - Axis-to-button conversion
    - Sensitivity and deadzone adjustment
    - Profile management
    - Input modifiers and macros
    """
    
    # Predefined profiles
    PRESET_PROFILES = {
        'southpaw': MappingProfile(
            name='southpaw',
            description='Swap left and right sticks',
            axis_map={
                Axis.LEFT_X: Axis.RIGHT_X,
                Axis.LEFT_Y: Axis.RIGHT_Y,
                Axis.RIGHT_X: Axis.LEFT_X,
                Axis.RIGHT_Y: Axis.LEFT_Y
            }
        ),
        'legacy': MappingProfile(
            name='legacy',
            description='Legacy FPS controls',
            axis_map={
                Axis.LEFT_Y: Axis.RIGHT_Y,
                Axis.RIGHT_Y: Axis.LEFT_Y
            }
        ),
        'inverted': MappingProfile(
            name='inverted',
            description='Inverted Y axes',
            axis_invert={
                Axis.LEFT_Y: True,
                Axis.RIGHT_Y: True
            }
        ),
        'racing': MappingProfile(
            name='racing',
            description='Racing game layout',
            button_to_axis={
                Button.RB: (Axis.RIGHT_TRIGGER, 1.0),  # Accelerate
                Button.LB: (Axis.LEFT_TRIGGER, 1.0)    # Brake
            }
        ),
        'fighting': MappingProfile(
            name='fighting',
            description='Fighting game layout',
            button_map={
                Button.A: Button.X,  # Light punch
                Button.B: Button.Y,  # Heavy punch
                Button.X: Button.A,  # Light kick
                Button.Y: Button.B   # Heavy kick
            }
        )
    }
    
    def __init__(self, controller: GamepadBase):
        """
        Initialize controller mapper.
        
        Args:
            controller: Controller to map
        """
        self.controller = controller
        
        # Profiles
        self.profiles: Dict[str, MappingProfile] = self.PRESET_PROFILES.copy()
        self.active_profile: Optional[MappingProfile] = None
        
        # Custom mappings
        self.custom_mappings: List[Callable[[ControllerState], ControllerState]] = []
        
        # Modifier keys
        self.modifier_buttons: List[Button] = []
        self.modifier_active = False
        
        # Mapping state
        self.original_state: Optional[ControllerState] = None
        self.mapped_state: Optional[ControllerState] = None
        
        # Profile directory
        self.profile_dir = Path.home() / '.nexus' / 'controller_profiles'
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Load saved profiles
        self._load_profiles()
        
        logger.info(f"Controller mapper initialized with {len(self.profiles)} profiles")
    
    def apply_profile(self, profile_name: str) -> bool:
        """
        Apply a mapping profile.
        
        Args:
            profile_name: Name of profile to apply
        
        Returns:
            True if profile applied successfully
        """
        if profile_name not in self.profiles:
            logger.warning(f"Profile '{profile_name}' not found")
            return False
        
        self.active_profile = self.profiles[profile_name]
        logger.info(f"Applied profile: {profile_name}")
        return True
    
    def clear_profile(self):
        """Clear active profile (use direct input)."""
        self.active_profile = None
        logger.info("Cleared active profile")
    
    def map_state(self, state: ControllerState) -> ControllerState:
        """
        Map controller state through active profile.
        
        Args:
            state: Original controller state
        
        Returns:
            Mapped controller state
        """
        self.original_state = state
        
        if not self.active_profile or not self.active_profile.enabled:
            self.mapped_state = state
            return state
        
        # Create new mapped state
        mapped = ControllerState(
            controller_id=state.controller_id,
            controller_type=state.controller_type,
            is_connected=state.is_connected,
            battery_level=state.battery_level
        )
        
        # Apply button mappings
        for button, pressed in state.buttons.items():
            if button in self.active_profile.button_map:
                mapped_button = self.active_profile.button_map[button]
                mapped.buttons[mapped_button] = pressed
            else:
                mapped.buttons[button] = pressed
        
        # Apply axis mappings
        for axis, value in state.axes.items():
            mapped_axis = axis
            mapped_value = value
            
            # Remap axis
            if axis in self.active_profile.axis_map:
                mapped_axis = self.active_profile.axis_map[axis]
            
            # Invert axis
            if axis in self.active_profile.axis_invert:
                if self.active_profile.axis_invert[axis]:
                    mapped_value = -mapped_value
            
            # Apply sensitivity
            if axis in self.active_profile.axis_sensitivity:
                sensitivity = self.active_profile.axis_sensitivity[axis]
                mapped_value = max(-1.0, min(1.0, mapped_value * sensitivity))
            
            # Apply deadzone
            if axis in self.active_profile.axis_deadzone:
                deadzone = self.active_profile.axis_deadzone[axis]
                if abs(mapped_value) < deadzone:
                    mapped_value = 0.0
                else:
                    # Rescale to maintain range
                    sign = 1 if mapped_value > 0 else -1
                    mapped_value = sign * (abs(mapped_value) - deadzone) / (1.0 - deadzone)
            
            mapped.axes[mapped_axis] = mapped_value
        
        # Apply button-to-axis conversions
        for button, (axis, value) in self.active_profile.button_to_axis.items():
            if button in state.buttons and state.buttons[button]:
                mapped.axes[axis] = value
        
        # Apply axis-to-button conversions
        for axis, conversions in self.active_profile.axis_to_button.items():
            if axis in state.axes:
                axis_value = state.axes[axis]
                for button, min_val, max_val in conversions:
                    mapped.buttons[button] = min_val <= axis_value <= max_val
        
        # Apply custom mappings
        for custom_map in self.custom_mappings:
            mapped = custom_map(mapped)
        
        # Apply modifier logic
        if self.modifier_buttons:
            self.modifier_active = any(
                state.buttons.get(btn, False) for btn in self.modifier_buttons
            )
            if self.modifier_active:
                mapped = self._apply_modifiers(mapped)
        
        self.mapped_state = mapped
        return mapped
    
    def create_profile(self, name: str, description: str = "") -> MappingProfile:
        """
        Create a new mapping profile.
        
        Args:
            name: Profile name
            description: Profile description
        
        Returns:
            New profile
        """
        profile = MappingProfile(name=name, description=description)
        self.profiles[name] = profile
        logger.info(f"Created profile: {name}")
        return profile
    
    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile.
        
        Args:
            name: Profile name
        
        Returns:
            True if deleted successfully
        """
        if name in self.PRESET_PROFILES:
            logger.warning(f"Cannot delete preset profile: {name}")
            return False
        
        if name in self.profiles:
            del self.profiles[name]
            if self.active_profile and self.active_profile.name == name:
                self.active_profile = None
            logger.info(f"Deleted profile: {name}")
            return True
        
        return False
    
    def save_profiles(self):
        """Save custom profiles to disk."""
        custom_profiles = {
            name: profile for name, profile in self.profiles.items()
            if name not in self.PRESET_PROFILES
        }
        
        for name, profile in custom_profiles.items():
            filepath = self.profile_dir / f"{name}.json"
            self._save_profile(profile, filepath)
        
        logger.info(f"Saved {len(custom_profiles)} custom profiles")
    
    def add_button_mapping(self, from_button: Button, to_button: Button):
        """
        Add button mapping to active profile.
        
        Args:
            from_button: Source button
            to_button: Target button
        """
        if not self.active_profile:
            logger.warning("No active profile")
            return
        
        self.active_profile.button_map[from_button] = to_button
        logger.info(f"Mapped {from_button.name} -> {to_button.name}")
    
    def add_axis_mapping(self, from_axis: Axis, to_axis: Axis, invert: bool = False):
        """
        Add axis mapping to active profile.
        
        Args:
            from_axis: Source axis
            to_axis: Target axis
            invert: Whether to invert axis
        """
        if not self.active_profile:
            logger.warning("No active profile")
            return
        
        self.active_profile.axis_map[from_axis] = to_axis
        if invert:
            self.active_profile.axis_invert[from_axis] = True
        
        logger.info(f"Mapped {from_axis.name} -> {to_axis.name} (inverted={invert})")
    
    def set_axis_sensitivity(self, axis: Axis, sensitivity: float):
        """
        Set axis sensitivity.
        
        Args:
            axis: Target axis
            sensitivity: Sensitivity multiplier
        """
        if not self.active_profile:
            logger.warning("No active profile")
            return
        
        self.active_profile.axis_sensitivity[axis] = sensitivity
        logger.info(f"Set {axis.name} sensitivity to {sensitivity}")
    
    def set_axis_deadzone(self, axis: Axis, deadzone: float):
        """
        Set axis deadzone.
        
        Args:
            axis: Target axis
            deadzone: Deadzone value (0.0-1.0)
        """
        if not self.active_profile:
            logger.warning("No active profile")
            return
        
        self.active_profile.axis_deadzone[axis] = deadzone
        logger.info(f"Set {axis.name} deadzone to {deadzone}")
    
    def add_custom_mapping(self, mapping_func: Callable[[ControllerState], ControllerState]):
        """
        Add custom mapping function.
        
        Args:
            mapping_func: Function that transforms controller state
        """
        self.custom_mappings.append(mapping_func)
        logger.info("Added custom mapping function")
    
    def set_modifier_buttons(self, buttons: List[Button]):
        """
        Set modifier buttons.
        
        Args:
            buttons: List of modifier buttons
        """
        self.modifier_buttons = buttons
        logger.info(f"Set {len(buttons)} modifier buttons")
    
    def calibrate_axis(self, axis: Axis) -> Dict[str, float]:
        """
        Calibrate an axis by detecting range.
        
        Args:
            axis: Axis to calibrate
        
        Returns:
            Calibration data (min, max, center)
        """
        logger.info(f"Calibrating {axis.name}...")
        logger.info("Move axis to extremes, then center it")
        
        min_val = float('inf')
        max_val = float('-inf')
        samples = []
        
        # Collect samples for 5 seconds
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            state = self.controller.get_state()
            if axis in state.axes:
                value = state.axes[axis]
                min_val = min(min_val, value)
                max_val = max(max_val, value)
                samples.append(value)
            time.sleep(0.01)
        
        # Calculate center (average of last 100 samples)
        center = sum(samples[-100:]) / min(100, len(samples))
        
        calibration = {
            'min': min_val,
            'max': max_val,
            'center': center,
            'range': max_val - min_val
        }
        
        logger.info(f"Calibration complete: {calibration}")
        return calibration
    
    def get_profile_list(self) -> List[Dict[str, Any]]:
        """Get list of available profiles."""
        return [
            {
                'name': name,
                'description': profile.description,
                'is_preset': name in self.PRESET_PROFILES,
                'is_active': self.active_profile and self.active_profile.name == name
            }
            for name, profile in self.profiles.items()
        ]
    
    # Private methods
    
    def _apply_modifiers(self, state: ControllerState) -> ControllerState:
        """Apply modifier logic to state."""
        # Example: When modifier is held, face buttons become d-pad
        if self.modifier_active:
            modified = state.copy()
            
            # Swap face buttons with d-pad
            modified.buttons[Button.DPAD_UP] = state.buttons.get(Button.Y, False)
            modified.buttons[Button.DPAD_DOWN] = state.buttons.get(Button.A, False)
            modified.buttons[Button.DPAD_LEFT] = state.buttons.get(Button.X, False)
            modified.buttons[Button.DPAD_RIGHT] = state.buttons.get(Button.B, False)
            
            # Clear face buttons
            modified.buttons[Button.A] = False
            modified.buttons[Button.B] = False
            modified.buttons[Button.X] = False
            modified.buttons[Button.Y] = False
            
            return modified
        
        return state
    
    def _save_profile(self, profile: MappingProfile, filepath: Path):
        """Save profile to file."""
        data = {
            'name': profile.name,
            'description': profile.description,
            'button_map': {btn.name: mapped.name for btn, mapped in profile.button_map.items()},
            'axis_map': {axis.name: mapped.name for axis, mapped in profile.axis_map.items()},
            'button_to_axis': {
                btn.name: [axis.name, value]
                for btn, (axis, value) in profile.button_to_axis.items()
            },
            'axis_to_button': {
                axis.name: [[btn.name, min_val, max_val] for btn, min_val, max_val in conversions]
                for axis, conversions in profile.axis_to_button.items()
            },
            'axis_invert': {axis.name: invert for axis, invert in profile.axis_invert.items()},
            'axis_sensitivity': {axis.name: sens for axis, sens in profile.axis_sensitivity.items()},
            'axis_deadzone': {axis.name: dz for axis, dz in profile.axis_deadzone.items()},
            'enabled': profile.enabled
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_profiles(self):
        """Load saved profiles from disk."""
        if not self.profile_dir.exists():
            return
        
        for filepath in self.profile_dir.glob('*.json'):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                profile = MappingProfile(
                    name=data['name'],
                    description=data.get('description', ''),
                    enabled=data.get('enabled', True)
                )
                
                # Load button mappings
                for from_btn, to_btn in data.get('button_map', {}).items():
                    profile.button_map[Button[from_btn]] = Button[to_btn]
                
                # Load axis mappings
                for from_axis, to_axis in data.get('axis_map', {}).items():
                    profile.axis_map[Axis[from_axis]] = Axis[to_axis]
                
                # Load other mappings...
                
                self.profiles[profile.name] = profile
                logger.info(f"Loaded profile: {profile.name}")
                
            except Exception as e:
                logger.error(f"Failed to load profile from {filepath}: {e}")


# Import Tuple for type hints
from typing import Tuple