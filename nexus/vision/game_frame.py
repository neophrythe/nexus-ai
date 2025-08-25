"""Enhanced Game Frame with Resolution Variants for Nexus Framework"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
import hashlib
from PIL import Image
import io
import skimage.color
import skimage.measure
import skimage.transform
import skimage.filters
import skimage.morphology
from scipy.stats import mode
import structlog

logger = structlog.get_logger()


class GameFrameError(Exception):
    """Game frame exception"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        logger.error(f"GameFrameError: {message}")


@dataclass
class FrameMetadata:
    """Frame metadata"""
    timestamp: float
    frame_number: int
    offset_x: int = 0
    offset_y: int = 0
    source: str = "unknown"
    game_name: Optional[str] = None
    window_id: Optional[Any] = None
    
    
class GameFrame:
    """Enhanced game frame with multiple resolution variants and analysis"""
    
    def __init__(self, frame_data: Union[np.ndarray, bytes], 
                 frame_variants: Optional[Dict[str, np.ndarray]] = None,
                 timestamp: Optional[float] = None,
                 metadata: Optional[FrameMetadata] = None,
                 **kwargs):
        """
        Initialize game frame
        
        Args:
            frame_data: Frame data as numpy array or bytes
            frame_variants: Pre-computed frame variants
            timestamp: Frame timestamp
            metadata: Frame metadata
            **kwargs: Additional metadata fields
        """
        # Store frame data
        if isinstance(frame_data, bytes):
            self.frame_bytes = frame_data
            self.frame_array = None
        elif isinstance(frame_data, np.ndarray):
            self.frame_bytes = None
            self.frame_array = frame_data
        else:
            raise GameFrameError(f"Invalid frame data type: {type(frame_data)}")
        
        # Frame variants cache
        self.frame_variants = frame_variants or {}
        
        # Metadata
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = FrameMetadata(
                timestamp=timestamp or time.time(),
                frame_number=kwargs.get('frame_number', 0),
                offset_x=kwargs.get('offset_x', 0),
                offset_y=kwargs.get('offset_y', 0),
                source=kwargs.get('source', 'unknown')
            )
        
        # Analysis cache
        self._analysis_cache = {}
        
    @property
    def frame(self) -> np.ndarray:
        """Get frame as numpy array"""
        if self.frame_array is not None:
            return self.frame_array
        elif self.frame_bytes is not None:
            # Convert bytes to array
            pil_image = Image.open(io.BytesIO(self.frame_bytes))
            self.frame_array = np.array(pil_image)
            return self.frame_array
        else:
            raise GameFrameError("No frame data available")
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get frame shape (height, width, channels)"""
        return self.frame.shape
    
    @property
    def width(self) -> int:
        """Get frame width"""
        return self.frame.shape[1]
    
    @property
    def height(self) -> int:
        """Get frame height"""
        return self.frame.shape[0]
    
    @property
    def channels(self) -> int:
        """Get number of channels"""
        return self.frame.shape[2] if len(self.frame.shape) > 2 else 1
    
    @property
    def half_resolution_frame(self) -> np.ndarray:
        """Get half resolution frame (1/4 size)"""
        if "half" not in self.frame_variants:
            self.frame_variants["half"] = self._to_half_resolution()
        return self.frame_variants["half"]
    
    @property
    def quarter_resolution_frame(self) -> np.ndarray:
        """Get quarter resolution frame (1/16 size)"""
        if "quarter" not in self.frame_variants:
            self.frame_variants["quarter"] = self._to_quarter_resolution()
        return self.frame_variants["quarter"]
    
    @property
    def eighth_resolution_frame(self) -> np.ndarray:
        """Get eighth resolution frame (1/64 size)"""
        if "eighth" not in self.frame_variants:
            self.frame_variants["eighth"] = self._to_eighth_resolution()
        return self.frame_variants["eighth"]
    
    @property
    def grayscale_frame(self) -> np.ndarray:
        """Get grayscale version of frame"""
        if "grayscale" not in self.frame_variants:
            self.frame_variants["grayscale"] = self._to_grayscale()
        return self.frame_variants["grayscale"]
    
    @property
    def eighth_resolution_grayscale_frame(self) -> np.ndarray:
        """Get eighth resolution grayscale frame"""
        if "eighth_grayscale" not in self.frame_variants:
            self.frame_variants["eighth_grayscale"] = self._to_eighth_grayscale_resolution()
        return self.frame_variants["eighth_grayscale"]
    
    @property
    def ssim_frame(self) -> np.ndarray:
        """Get 100x100 grayscale frame for SSIM comparison"""
        if "ssim" not in self.frame_variants:
            self.frame_variants["ssim"] = self._to_ssim()
        return self.frame_variants["ssim"]
    
    @property
    def edges_frame(self) -> np.ndarray:
        """Get edge detection frame"""
        if "edges" not in self.frame_variants:
            self.frame_variants["edges"] = self._detect_edges()
        return self.frame_variants["edges"]
    
    @property
    def top_color(self) -> Tuple[int, int, int]:
        """Get most common color in frame"""
        if "top_color" not in self._analysis_cache:
            self._analysis_cache["top_color"] = self._compute_top_color()
        return self._analysis_cache["top_color"]
    
    @property
    def dominant_colors(self) -> list:
        """Get top 5 dominant colors"""
        if "dominant_colors" not in self._analysis_cache:
            self._analysis_cache["dominant_colors"] = self._compute_dominant_colors()
        return self._analysis_cache["dominant_colors"]
    
    @property
    def frame_hash(self) -> str:
        """Get perceptual hash of frame"""
        if "frame_hash" not in self._analysis_cache:
            self._analysis_cache["frame_hash"] = self._compute_hash()
        return self._analysis_cache["frame_hash"]
    
    def _to_half_resolution(self) -> np.ndarray:
        """Convert to half resolution"""
        return cv2.resize(self.frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    def _to_quarter_resolution(self) -> np.ndarray:
        """Convert to quarter resolution"""
        return cv2.resize(self.frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    
    def _to_eighth_resolution(self) -> np.ndarray:
        """Convert to eighth resolution"""
        return cv2.resize(self.frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    
    def _to_grayscale(self) -> np.ndarray:
        """Convert to grayscale"""
        if len(self.frame.shape) == 2:
            return self.frame
        return cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
    
    def _to_eighth_grayscale_resolution(self) -> np.ndarray:
        """Convert to eighth resolution grayscale"""
        eighth = self.eighth_resolution_frame
        if len(eighth.shape) == 2:
            return eighth
        return cv2.cvtColor(eighth, cv2.COLOR_RGB2GRAY)
    
    def _to_ssim(self) -> np.ndarray:
        """Convert to SSIM comparison format"""
        gray = self.grayscale_frame
        return cv2.resize(gray, (100, 100), interpolation=cv2.INTER_AREA)
    
    def _detect_edges(self) -> np.ndarray:
        """Detect edges using Canny"""
        gray = self.grayscale_frame
        return cv2.Canny(gray, 50, 150)
    
    def _compute_top_color(self) -> Tuple[int, int, int]:
        """Compute most common color"""
        eighth = self.eighth_resolution_frame
        
        if len(eighth.shape) == 2:
            # Grayscale
            values, counts = np.unique(eighth.flatten(), return_counts=True)
            top_value = values[np.argmax(counts)]
            return (int(top_value), int(top_value), int(top_value))
        else:
            # Color
            height, width, channels = eighth.shape
            reshaped = eighth.reshape(width * height, channels)
            
            # Use mode to find most common color
            mode_result = mode(reshaped, axis=0, keepdims=False)
            return tuple(int(v) for v in mode_result.mode)
    
    def _compute_dominant_colors(self, n_colors: int = 5) -> list:
        """Compute dominant colors using k-means"""
        from sklearn.cluster import KMeans
        
        eighth = self.eighth_resolution_frame
        
        if len(eighth.shape) == 2:
            # Grayscale
            values, counts = np.unique(eighth.flatten(), return_counts=True)
            sorted_indices = np.argsort(counts)[::-1][:n_colors]
            return [(int(values[i]), int(values[i]), int(values[i])) for i in sorted_indices]
        else:
            # Color
            height, width, channels = eighth.shape
            reshaped = eighth.reshape(width * height, channels)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(reshaped)
            
            colors = kmeans.cluster_centers_.astype(int)
            return [tuple(color) for color in colors]
    
    def _compute_hash(self) -> str:
        """Compute perceptual hash"""
        # Use small grayscale version
        small = cv2.resize(self.grayscale_frame, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Compute average
        avg = small.mean()
        
        # Create binary hash
        binary_hash = (small > avg).flatten()
        
        # Convert to hex string
        hash_int = 0
        for bit in binary_hash:
            hash_int = (hash_int << 1) | int(bit)
        
        return hex(hash_int)[2:].zfill(16)
    
    def compare_ssim(self, other: 'GameFrame') -> float:
        """
        Compare structural similarity with another frame
        
        Args:
            other: Frame to compare with
        
        Returns:
            SSIM score (0-1)
        """
        from skimage.metrics import structural_similarity
        
        return structural_similarity(self.ssim_frame, other.ssim_frame)
    
    def compare_hash(self, other: 'GameFrame') -> int:
        """
        Compare perceptual hash distance
        
        Args:
            other: Frame to compare with
        
        Returns:
            Hamming distance between hashes
        """
        hash1 = int(self.frame_hash, 16)
        hash2 = int(other.frame_hash, 16)
        
        # XOR and count bits
        xor = hash1 ^ hash2
        distance = bin(xor).count('1')
        
        return distance
    
    def extract_region(self, x: int, y: int, width: int, height: int) -> 'GameFrame':
        """
        Extract region as new GameFrame
        
        Args:
            x: X coordinate
            y: Y coordinate  
            width: Region width
            height: Region height
        
        Returns:
            New GameFrame with extracted region
        """
        region = self.frame[y:y+height, x:x+width]
        
        return GameFrame(
            region,
            metadata=FrameMetadata(
                timestamp=self.metadata.timestamp,
                frame_number=self.metadata.frame_number,
                offset_x=self.metadata.offset_x + x,
                offset_y=self.metadata.offset_y + y,
                source=f"region_of_{self.metadata.source}"
            )
        )
    
    def apply_mask(self, mask: np.ndarray) -> 'GameFrame':
        """
        Apply mask to frame
        
        Args:
            mask: Binary mask
        
        Returns:
            New GameFrame with mask applied
        """
        if mask.shape[:2] != self.frame.shape[:2]:
            mask = cv2.resize(mask, (self.width, self.height))
        
        if len(mask.shape) == 2:
            mask = np.stack([mask] * self.channels, axis=-1)
        
        masked = cv2.bitwise_and(self.frame, self.frame, mask=mask.astype(np.uint8))
        
        return GameFrame(
            masked,
            metadata=FrameMetadata(
                timestamp=self.metadata.timestamp,
                frame_number=self.metadata.frame_number,
                offset_x=self.metadata.offset_x,
                offset_y=self.metadata.offset_y,
                source=f"masked_{self.metadata.source}"
            )
        )
    
    def annotate(self, annotations: list) -> 'GameFrame':
        """
        Create annotated copy of frame
        
        Args:
            annotations: List of annotation dicts with 'type', 'data', 'color', etc.
        
        Returns:
            New annotated GameFrame
        """
        annotated = self.frame.copy()
        
        for ann in annotations:
            ann_type = ann.get('type')
            
            if ann_type == 'box':
                x, y, w, h = ann['data']
                color = ann.get('color', (0, 255, 0))
                thickness = ann.get('thickness', 2)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, thickness)
            
            elif ann_type == 'text':
                text = ann['data']
                position = ann.get('position', (10, 30))
                color = ann.get('color', (0, 255, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = ann.get('scale', 1)
                thickness = ann.get('thickness', 2)
                cv2.putText(annotated, text, position, font, scale, color, thickness)
            
            elif ann_type == 'circle':
                center, radius = ann['data']
                color = ann.get('color', (0, 255, 0))
                thickness = ann.get('thickness', 2)
                cv2.circle(annotated, center, radius, color, thickness)
            
            elif ann_type == 'line':
                pt1, pt2 = ann['data']
                color = ann.get('color', (0, 255, 0))
                thickness = ann.get('thickness', 2)
                cv2.line(annotated, pt1, pt2, color, thickness)
        
        return GameFrame(
            annotated,
            metadata=FrameMetadata(
                timestamp=self.metadata.timestamp,
                frame_number=self.metadata.frame_number,
                offset_x=self.metadata.offset_x,
                offset_y=self.metadata.offset_y,
                source=f"annotated_{self.metadata.source}"
            )
        )
    
    def save(self, path: str, quality: int = 95):
        """
        Save frame to file
        
        Args:
            path: Output path
            quality: JPEG quality (0-100)
        """
        if path.endswith('.png'):
            cv2.imwrite(path, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(path, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    def to_bytes(self, format: str = 'PNG') -> bytes:
        """
        Convert frame to bytes
        
        Args:
            format: Image format (PNG, JPEG)
        
        Returns:
            Frame as bytes
        """
        pil_image = Image.fromarray(self.frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()
    
    def __repr__(self) -> str:
        return (f"GameFrame(shape={self.shape}, "
                f"timestamp={self.metadata.timestamp:.3f}, "
                f"source={self.metadata.source})")