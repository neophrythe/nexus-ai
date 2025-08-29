from nexus.input.base import InputController as BaseInputController, InputAction, InputType

# Lazy import controllers to avoid X11 issues at import time
def _get_pyautogui_controller():
    from nexus.input.pyautogui_controller import PyAutoGUIController
    return PyAutoGUIController

def _get_native_controller():
    from nexus.input.native_controller import NativeInputController
    return NativeInputController

# For backward compatibility
PyAutoGUIController = _get_pyautogui_controller
NativeInputController = _get_native_controller
from nexus.input.human_like import HumanLikeInput

# New Input Recording/Playback System
from nexus.input.recorder import (
    InputRecorder,
    InputEvent,
    EventType,
    KeyboardAction,
    MouseAction,
    RecordingState
)

from nexus.input.playback import (
    InputPlayback,
    PlaybackState,
    PlaybackSpeed,
    PlaybackConfig
)

from nexus.input.controller import (
    InputController,
    InputBackend
)

__all__ = [
    # Legacy Input System
    "BaseInputController",
    "InputAction", 
    "InputType",
    "PyAutoGUIController",
    "NativeInputController",
    "HumanLikeInput",
    
    # New Input Recording/Playback System
    "InputRecorder",
    "InputEvent",
    "EventType",
    "KeyboardAction",
    "MouseAction",
    "RecordingState",
    "InputPlayback",
    "PlaybackState",
    "PlaybackSpeed",
    "PlaybackConfig",
    "InputController",
    "InputBackend",
]