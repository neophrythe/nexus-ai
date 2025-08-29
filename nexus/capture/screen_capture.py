"""
Screen capture compatibility module.
Maps old screen_capture imports to new capture_manager.
"""

from nexus.capture.capture_manager import CaptureManager

# Alias for backward compatibility
ScreenCapture = CaptureManager

# Re-export all methods
__all__ = ['ScreenCapture', 'CaptureManager']