"""Advanced frame processing utilities adapted from SerpentAI"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
import io
import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import structlog

logger = structlog.get_logger()


@dataclass
class GameFrame:
    """Enhanced game frame with multi-resolution variants and processing capabilities"""
    
    frame_data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    offset_x: int = 0
    offset_y: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _variants: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    
    @property
    def frame(self) -> np.ndarray:
        """Get the original frame data"""
        return self.frame_data
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get frame shape"""
        return self.frame_data.shape
    
    @property
    def half_resolution(self) -> np.ndarray:
        """Get half resolution variant (1/4 size)"""
        if "half" not in self._variants:
            self._variants["half"] = self._resize_frame(0.5)
        return self._variants["half"]
    
    @property
    def quarter_resolution(self) -> np.ndarray:
        """Get quarter resolution variant (1/16 size)"""
        if "quarter" not in self._variants:
            self._variants["quarter"] = self._resize_frame(0.25)
        return self._variants["quarter"]
    
    @property
    def eighth_resolution(self) -> np.ndarray:
        """Get eighth resolution variant (1/64 size)"""
        if "eighth" not in self._variants:
            self._variants["eighth"] = self._resize_frame(0.125)
        return self._variants["eighth"]
    
    @property
    def grayscale(self) -> np.ndarray:
        """Get grayscale version of frame"""
        if "grayscale" not in self._variants:
            if len(self.frame_data.shape) == 3:
                self._variants["grayscale"] = cv2.cvtColor(self.frame_data, cv2.COLOR_BGR2GRAY)
            else:
                self._variants["grayscale"] = self.frame_data
        return self._variants["grayscale"]
    
    @property
    def grayscale_eighth(self) -> np.ndarray:
        """Get grayscale eighth resolution variant"""
        if "grayscale_eighth" not in self._variants:
            eighth = self.eighth_resolution
            if len(eighth.shape) == 3:
                self._variants["grayscale_eighth"] = cv2.cvtColor(eighth, cv2.COLOR_BGR2GRAY)
            else:
                self._variants["grayscale_eighth"] = eighth
        return self._variants["grayscale_eighth"]
    
    @property
    def ssim_frame(self) -> np.ndarray:
        """Get normalized frame for SSIM comparison (100x100 grayscale)"""
        if "ssim" not in self._variants:
            # Resize to standard size for SSIM
            resized = cv2.resize(self.grayscale, (100, 100))
            # Normalize to 0-1 range
            self._variants["ssim"] = resized.astype(np.float32) / 255.0
        return self._variants["ssim"]
    
    @property
    def dominant_color(self) -> Tuple[int, int, int]:
        """Get the most dominant color in the frame"""
        # Use eighth resolution for speed
        eighth = self.eighth_resolution
        if len(eighth.shape) == 3:
            pixels = eighth.reshape(-1, eighth.shape[-1])
        else:
            # Grayscale
            return (int(np.mean(eighth)),) * 3
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Return most common color
        dominant_idx = np.argmax(counts)
        return tuple(map(int, unique_colors[dominant_idx]))
    
    def _resize_frame(self, scale: float) -> np.ndarray:
        """Resize frame by scale factor"""
        new_size = (int(self.frame_data.shape[1] * scale), 
                   int(self.frame_data.shape[0] * scale))
        return cv2.resize(self.frame_data, new_size, interpolation=cv2.INTER_AREA)
    
    def compare_ssim(self, other: 'GameFrame') -> float:
        """
        Compare structural similarity with another frame.
        
        Args:
            other: Frame to compare with
        
        Returns:
            SSIM score between 0 and 1
        """
        return ssim(self.ssim_frame, other.ssim_frame)
    
    def difference(self, other: 'GameFrame', blur_sigma: float = 2.0) -> np.ndarray:
        """
        Calculate difference between frames.
        
        Args:
            other: Frame to compare with
            blur_sigma: Gaussian blur sigma for noise reduction
        
        Returns:
            Difference image
        """
        # Apply Gaussian blur to reduce noise
        current = cv2.GaussianBlur(self.grayscale, (0, 0), blur_sigma)
        previous = cv2.GaussianBlur(other.grayscale, (0, 0), blur_sigma)
        
        # Calculate absolute difference
        diff = cv2.absdiff(current, previous)
        
        return diff
    
    def motion_mask(self, other: 'GameFrame', threshold: int = 25) -> np.ndarray:
        """
        Create binary mask of motion areas.
        
        Args:
            other: Previous frame
            threshold: Motion threshold
        
        Returns:
            Binary motion mask
        """
        diff = self.difference(other)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def extract_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract a region from the frame.
        
        Args:
            region: Region as (x1, y1, x2, y2)
        
        Returns:
            Extracted region
        """
        x1, y1, x2, y2 = region
        return self.frame_data[y1:y2, x1:x2]
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image"""
        if len(self.frame_data.shape) == 3:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(self.frame_data, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        else:
            return Image.fromarray(self.frame_data)
    
    def to_bytes(self, format: str = "PNG", quality: int = 95) -> bytes:
        """
        Convert frame to bytes.
        
        Args:
            format: Image format (PNG, JPEG, etc.)
            quality: JPEG quality (1-100)
        
        Returns:
            Image bytes
        """
        pil_image = self.to_pil()
        buffer = io.BytesIO()
        
        if format.upper() == "JPEG":
            pil_image.save(buffer, format=format, quality=quality, optimize=True)
        else:
            pil_image.save(buffer, format=format)
        
        buffer.seek(0)
        return buffer.read()
    
    def apply_transform(self, transform: str, **kwargs) -> np.ndarray:
        """
        Apply various transformations to the frame.
        
        Args:
            transform: Transform type
            **kwargs: Transform-specific parameters
        
        Returns:
            Transformed frame
        """
        transforms = {
            "blur": lambda: cv2.GaussianBlur(self.frame_data, (5, 5), kwargs.get("sigma", 1.0)),
            "sharpen": lambda: self._sharpen(kwargs.get("amount", 1.0)),
            "edge": lambda: cv2.Canny(self.grayscale, kwargs.get("low", 50), kwargs.get("high", 150)),
            "threshold": lambda: cv2.threshold(self.grayscale, kwargs.get("thresh", 127), 255, cv2.THRESH_BINARY)[1],
            "adaptive_threshold": lambda: cv2.adaptiveThreshold(
                self.grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, kwargs.get("block_size", 11), kwargs.get("c", 2)
            ),
            "denoise": lambda: cv2.fastNlMeansDenoisingColored(self.frame_data) if len(self.frame_data.shape) == 3 else cv2.fastNlMeansDenoising(self.frame_data),
            "histogram_equalize": lambda: self._histogram_equalize(),
            "gamma_correction": lambda: self._gamma_correction(kwargs.get("gamma", 1.0))
        }
        
        if transform in transforms:
            return transforms[transform]()
        else:
            logger.warning(f"Unknown transform: {transform}")
            return self.frame_data
    
    def _sharpen(self, amount: float = 1.0) -> np.ndarray:
        """Apply sharpening filter"""
        kernel = np.array([[-1, -1, -1],
                          [-1, 8 + amount, -1],
                          [-1, -1, -1]]) / (amount + 1)
        return cv2.filter2D(self.frame_data, -1, kernel)
    
    def _histogram_equalize(self) -> np.ndarray:
        """Apply histogram equalization"""
        if len(self.frame_data.shape) == 3:
            # Convert to YCrCb and equalize Y channel
            ycrcb = cv2.cvtColor(self.frame_data, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(self.frame_data)
    
    def _gamma_correction(self, gamma: float = 1.0) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(self.frame_data, table)


class FrameProcessor:
    """Advanced frame processing pipeline"""
    
    def __init__(self):
        self.processing_pipeline: List[Tuple[str, Dict[str, Any]]] = []
        self.frame_buffer: List[GameFrame] = []
        self.max_buffer_size: int = 30
    
    def add_processing_step(self, transform: str, **kwargs):
        """Add a processing step to the pipeline"""
        self.processing_pipeline.append((transform, kwargs))
    
    def process_frame(self, frame: GameFrame) -> GameFrame:
        """Process frame through pipeline"""
        processed_data = frame.frame_data.copy()
        
        for transform, kwargs in self.processing_pipeline:
            temp_frame = GameFrame(processed_data)
            processed_data = temp_frame.apply_transform(transform, **kwargs)
        
        return GameFrame(
            processed_data,
            timestamp=frame.timestamp,
            offset_x=frame.offset_x,
            offset_y=frame.offset_y,
            metadata={**frame.metadata, "processed": True}
        )
    
    def add_to_buffer(self, frame: GameFrame):
        """Add frame to buffer"""
        self.frame_buffer.append(frame)
        
        # Maintain max buffer size
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def get_motion_areas(self, current: GameFrame, 
                        previous: Optional[GameFrame] = None,
                        min_area: int = 100) -> List[Tuple[int, int, int, int]]:
        """
        Detect areas of motion between frames.
        
        Args:
            current: Current frame
            previous: Previous frame (uses buffer if None)
            min_area: Minimum area for motion detection
        
        Returns:
            List of motion bounding boxes
        """
        if previous is None and self.frame_buffer:
            previous = self.frame_buffer[-1]
        
        if previous is None:
            return []
        
        # Get motion mask
        motion_mask = current.motion_mask(previous)
        
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append((x, y, x + w, y + h))
        
        return motion_areas
    
    def calculate_frame_similarity(self, frame1: GameFrame, frame2: GameFrame,
                                  method: str = "ssim") -> float:
        """
        Calculate similarity between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            method: Similarity method ("ssim", "mse", "correlation")
        
        Returns:
            Similarity score
        """
        if method == "ssim":
            return frame1.compare_ssim(frame2)
        elif method == "mse":
            # Mean Squared Error (lower is more similar)
            mse = np.mean((frame1.grayscale.astype(float) - frame2.grayscale.astype(float)) ** 2)
            # Convert to similarity (0-1, higher is more similar)
            return 1.0 / (1.0 + mse / 1000.0)
        elif method == "correlation":
            # Normalized cross-correlation
            corr = cv2.matchTemplate(frame1.grayscale, frame2.grayscale, cv2.TM_CCORR_NORMED)
            return float(corr[0, 0])
        else:
            logger.warning(f"Unknown similarity method: {method}")
            return 0.0
    
    def detect_scene_change(self, threshold: float = 0.3) -> bool:
        """
        Detect if a scene change occurred.
        
        Args:
            threshold: SSIM threshold for scene change
        
        Returns:
            True if scene changed
        """
        if len(self.frame_buffer) < 2:
            return False
        
        current = self.frame_buffer[-1]
        previous = self.frame_buffer[-2]
        
        similarity = current.compare_ssim(previous)
        return similarity < threshold
    
    def create_frame_composite(self, frames: List[GameFrame], 
                              layout: str = "grid") -> np.ndarray:
        """
        Create composite image from multiple frames.
        
        Args:
            frames: List of frames
            layout: Layout type ("grid", "horizontal", "vertical")
        
        Returns:
            Composite image
        """
        if not frames:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        if layout == "horizontal":
            return np.hstack([f.frame_data for f in frames])
        elif layout == "vertical":
            return np.vstack([f.frame_data for f in frames])
        elif layout == "grid":
            # Calculate grid dimensions
            n = len(frames)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            
            # Resize frames to same size
            h, w = frames[0].shape[:2]
            grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            
            for idx, frame in enumerate(frames):
                row = idx // cols
                col = idx % cols
                resized = cv2.resize(frame.frame_data, (w, h))
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = resized
            
            return grid
        
        return frames[0].frame_data