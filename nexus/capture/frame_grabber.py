"""Advanced Frame Grabber System without Redis dependency

This module provides high-performance frame capture with multiple backends,
in-memory queuing, and WebSocket streaming for visual debugging.
"""

import numpy as np
import time
import threading
import queue
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import cv2
import sys

# Capture backends
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import d3dshot
    HAS_D3D = True
except ImportError:
    HAS_D3D = False

try:
    import pyautogui
    HAS_PYAUTOGUI = True  
except ImportError:
    HAS_PYAUTOGUI = False

from nexus.vision.game_frame import GameFrame, FrameMetadata
from nexus.vision.frame_buffer import FrameBuffer
from nexus.transformations.frame_transformation_pipeline import FrameTransformationPipeline

import structlog
logger = structlog.get_logger()


class FrameGrabber:
    """Advanced frame grabber without Redis dependency - SerpentAI compatible"""
    
    def __init__(self, width=640, height=480, x_offset=0, y_offset=0, fps=30, 
                 pipeline_string=None, buffer_seconds=5):
        """Initialize frame grabber with SerpentAI-compatible interface"""
        # Store parameters
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.frame_time = 1.0 / fps
        self.frame_buffer_size = fps * buffer_seconds
        
        # Initialize capture backend
        self.screen_grabber = self._init_capture_backend()
        
        # Frame storage (in-memory instead of Redis)
        self.frame_storage = []
        self.pipeline_storage = []
        self.storage_lock = threading.Lock()
        self.max_frames = int(self.frame_buffer_size)
        
        # Frame transformation pipeline
        self.frame_transformation_pipeline = None
        if pipeline_string is not None and isinstance(pipeline_string, str):
            self.frame_transformation_pipeline = FrameTransformationPipeline(pipeline_string=pipeline_string)
        
        # Running state
        self.running = False
        self.capture_thread = None
        
    def _init_capture_backend(self):
        """Initialize capture backend"""
        if HAS_MSS:
            return mss.mss()
        elif HAS_D3D and sys.platform == 'win32':
            d = d3dshot.create(capture_output="numpy")
            return d
        elif HAS_PYAUTOGUI:
            return pyautogui
        else:
            # Fallback to OpenCV
            return cv2.VideoCapture(0)
            
    def start(self):
        """Start frame capture loop - SerpentAI compatible"""
        self.running = True
        
        while self.running:
            cycle_start = time.time()
            
            # Capture frame
            frame = self.grab_frame()
            
            # Apply pipeline transformation
            if self.frame_transformation_pipeline is not None:
                frame_pipeline = self.frame_transformation_pipeline.transform(frame)
            else:
                frame_pipeline = frame
            
            # Store frame data (SerpentAI format)
            frame_shape = str(frame.shape).replace("(", "").replace(")", "")
            frame_dtype = str(frame.dtype)
            
            frame_bytes = f"{cycle_start}~{frame_shape}~{frame_dtype}~".encode("utf-8") + frame.tobytes()
            
            # Store in memory instead of Redis
            with self.storage_lock:
                self.frame_storage.append(frame_bytes)
                if len(self.frame_storage) > self.max_frames:
                    self.frame_storage.pop(0)
            
            # Store pipeline frame
            if self._has_png_transformation_pipeline():
                frame_pipeline_shape = "PNG"
                frame_pipeline_dtype = "PNG"
                frame_pipeline_bytes = f"{cycle_start}~{frame_pipeline_shape}~{frame_pipeline_dtype}~".encode("utf-8") + frame_pipeline
            else:
                frame_pipeline_shape = str(frame_pipeline.shape).replace("(", "").replace(")", "")
                frame_pipeline_dtype = str(frame_pipeline.dtype)
                frame_pipeline_bytes = f"{cycle_start}~{frame_pipeline_shape}~{frame_pipeline_dtype}~".encode("utf-8") + frame_pipeline.tobytes()
            
            with self.storage_lock:
                self.pipeline_storage.append(frame_pipeline_bytes)
                if len(self.pipeline_storage) > self.max_frames:
                    self.pipeline_storage.pop(0)
            
            # Frame timing
            cycle_end = time.time()
            cycle_duration = (cycle_end - cycle_start)
            cycle_duration -= int(cycle_duration)
            
            frame_time_left = self.frame_time - cycle_duration
            
            if frame_time_left > 0:
                time.sleep(frame_time_left)
                
    def grab_frame(self):
        """Grab a single frame - SerpentAI compatible"""
        if HAS_MSS and isinstance(self.screen_grabber, mss.mss):
            frame = np.array(
                self.screen_grabber.grab({
                    "top": self.y_offset,
                    "left": self.x_offset,
                    "width": self.width,
                    "height": self.height
                }),
                dtype="uint8"
            )
            # Convert BGRA to RGB
            frame = frame[..., [2, 1, 0, 3]]
            return frame[..., :3]
            
        elif HAS_D3D and hasattr(self.screen_grabber, 'screenshot'):
            frame = self.screen_grabber.screenshot(region=(
                self.x_offset, self.y_offset,
                self.x_offset + self.width,
                self.y_offset + self.height
            ))
            return np.array(frame) if frame else np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        elif HAS_PYAUTOGUI and self.screen_grabber == pyautogui:
            screenshot = pyautogui.screenshot(
                region=(self.x_offset, self.y_offset, self.width, self.height)
            )
            return np.array(screenshot)
            
        else:
            # OpenCV fallback
            ret, frame = self.screen_grabber.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
    def _has_png_transformation_pipeline(self):
        """Check if pipeline produces PNG output"""
        return (self.frame_transformation_pipeline and 
                self.frame_transformation_pipeline.pipeline_string and 
                self.frame_transformation_pipeline.pipeline_string.endswith("|PNG"))
                
    @classmethod
    def get_frames(cls, frame_buffer_indices, frame_type="FULL", **kwargs):
        """Get frames - SerpentAI compatible class method"""
        # Wait for frames to be available
        import time
        max_wait = 5.0
        start = time.time()
        
        # Get global instance or create one
        if not hasattr(cls, '_global_instance'):
            cls._global_instance = cls(**kwargs)
            # Start capture in background thread
            import threading
            t = threading.Thread(target=cls._global_instance.start, daemon=True)
            t.start()
            
        instance = cls._global_instance
        
        # Wait for enough frames
        while len(instance.frame_storage) < 150:
            if time.time() - start > max_wait:
                break
            time.sleep(0.1)
        
        # Create game frame buffer
        from nexus.vision.frame_buffer import FrameBuffer
        game_frame_buffer = FrameBuffer(size=len(frame_buffer_indices))
        
        with instance.storage_lock:
            storage = instance.pipeline_storage if frame_type == "PIPELINE" else instance.frame_storage
            
            for i in frame_buffer_indices:
                if i < len(storage):
                    frame_data = storage[-(i+1)]  # Get from end (most recent)
                    
                    timestamp, shape, dtype, frame_bytes = frame_data.split(b"~", maxsplit=3)
                    
                    if dtype == b"PNG":
                        frame_array = frame_bytes
                    else:
                        frame_shape = [int(x) for x in shape.decode("utf-8").split(", ")]
                        frame_array = np.frombuffer(frame_bytes, dtype=dtype.decode("utf-8")).reshape(frame_shape)
                    
                    game_frame = GameFrame(frame_array, timestamp=float(timestamp))
                    game_frame_buffer.add_frame(game_frame)
        
        return game_frame_buffer
        
    @classmethod
    def get_frames_with_pipeline(cls, frame_buffer_indices, **kwargs):
        """Get both original and pipeline frames - SerpentAI compatible"""
        # Get both frame types
        original_buffer = cls.get_frames(frame_buffer_indices, "FULL", **kwargs)
        pipeline_buffer = cls.get_frames(frame_buffer_indices, "PIPELINE", **kwargs)
        
        return [original_buffer, pipeline_buffer]
