"""
Nexus Controller Input Module - Gamepad and Controller Support

Provides comprehensive controller support for game AI including:
- Xbox controller (XInput)
- PlayStation controller (DS4/DS5)
- Generic gamepad support
- Controller recording and playback
- Virtual controller for testing
"""

from nexus.input.gamepad.gamepad_base import (
    GamepadBase,
    Button,
    Axis,
    ControllerState,
    ControllerType
)
from nexus.input.gamepad.xbox_controller import XboxController
from nexus.input.gamepad.playstation_controller import PlayStationController
from nexus.input.gamepad.generic_controller import GenericController
from nexus.input.gamepad.virtual_gamepad import VirtualGamepad
from nexus.input.gamepad.controller_recorder import ControllerRecorder
from nexus.input.gamepad.controller_mapper import ControllerMapper
from nexus.input.gamepad.haptic_feedback import HapticFeedback

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