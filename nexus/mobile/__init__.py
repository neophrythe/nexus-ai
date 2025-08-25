"""
Nexus Mobile Game Support Module

Provides support for mobile games through Android emulators and devices.
"""

from nexus.mobile.adb_client import ADBClient, ADBDevice
from nexus.mobile.emulator_detector import EmulatorDetector, EmulatorType
from nexus.mobile.touch_controller import TouchController, TouchGesture
from nexus.mobile.mobile_launcher import MobileLauncher
from nexus.mobile.screen_capture import MobileScreenCapture
from nexus.mobile.input_injector import InputInjector
from nexus.mobile.app_manager import AppManager

__all__ = [
    'ADBClient',
    'ADBDevice',
    'EmulatorDetector',
    'EmulatorType',
    'TouchController',
    'TouchGesture',
    'MobileLauncher',
    'MobileScreenCapture',
    'InputInjector',
    'AppManager'
]

__version__ = '1.0.0'