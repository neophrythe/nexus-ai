"""
Frame processor for vision pipeline.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple


class FrameProcessor:
    """Processes game frames for vision pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize frame processor."""
        self.config = config or {}
        self.target_size = self.config.get('target_size', (84, 84))
        self.grayscale = self.config.get('grayscale', True)
        self.normalize = self.config.get('normalize', True)
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        if frame is None or frame.size == 0:
            return np.zeros((*self.target_size, 1 if self.grayscale else 3))
        
        # Resize frame
        processed = cv2.resize(frame, self.target_size)
        
        # Convert to grayscale if needed
        if self.grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = np.expand_dims(processed, axis=-1)
        
        # Normalize if needed
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def preprocess_batch(self, frames: list) -> np.ndarray:
        """Process a batch of frames."""
        return np.array([self.process(frame) for frame in frames])
    
    def extract_region(self, frame: np.ndarray, 
                      x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extract a region from frame."""
        return frame[y:y+h, x:x+w]
    
    def apply_filters(self, frame: np.ndarray, 
                     filters: Optional[list] = None) -> np.ndarray:
        """Apply filters to frame."""
        if filters is None:
            return frame
        
        result = frame.copy()
        for filter_name in filters:
            if filter_name == 'blur':
                result = cv2.GaussianBlur(result, (5, 5), 0)
            elif filter_name == 'edge':
                result = cv2.Canny(result, 100, 200)
            elif filter_name == 'sharpen':
                kernel = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
                result = cv2.filter2D(result, -1, kernel)
        
        return result