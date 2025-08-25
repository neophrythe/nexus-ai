"""
Nexus Controller Input Module - Gamepad and Controller Support

Provides comprehensive controller support for game AI including:
- Xbox controller (XInput)
- PlayStation controller (DS4/DS5)
- Generic gamepad support
- Controller recording and playback
- Virtual controller for testing
"""

from nexus.input.controller.gamepad_base import (
    GamepadBase,
    Button,
    Axis,
    ControllerState,
    ControllerType
)
from nexus.input.controller.xbox_controller import XboxController
from nexus.input.controller.playstation_controller import PlayStationController
from nexus.input.controller.generic_controller import GenericController
from nexus.input.controller.virtual_gamepad import VirtualGamepad
from nexus.input.controller.controller_recorder import ControllerRecorder
from nexus.input.controller.controller_mapper import ControllerMapper
from nexus.input.controller.haptic_feedback import HapticFeedback

__all__ = [
    'GamepadBase',
    'Button',
    'Axis',
    'ControllerState',
    'ControllerType',
    'XboxController',
    'PlayStationController',
    'GenericController',
    'VirtualGamepad',
    'ControllerRecorder',
    'ControllerMapper',
    'HapticFeedback'
]

# Version info
__version__ = '1.0.0'