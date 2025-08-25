"""Enhanced Sprite Management System - SerpentAI Compatible with Modern Features

Provides comprehensive sprite identification and location with multiple algorithms.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
import json
import hashlib
import pickle
from collections import defaultdict
import structlog
from enum import Enum
import time

# Feature detection imports
try:
    import skimage.measure
    import skimage.feature
    from skimage.metrics import structural_similarity
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

logger = structlog.get_logger()


class MatchingMode(Enum):
    """Sprite matching modes - SerpentAI compatible"""
    SIGNATURE_COLORS = "SIGNATURE_COLORS"
    CONSTELLATION_OF_PIXELS = "CONSTELLATION_OF_PIXELS"
    SSIM = "SSIM"
    # Additional modes
    TEMPLATE_MATCHING = "TEMPLATE_MATCHING"
    FEATURE_MATCHING = "FEATURE_MATCHING"
    HISTOGRAM = "HISTOGRAM"
    NEURAL_EMBEDDING = "NEURAL_EMBEDDING"
    MULTI_METHOD = "MULTI_METHOD"


@dataclass
class SpriteMetadata:
    """Sprite metadata"""
    name: str
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    click_offset: Tuple[int, int] = (0, 0)  # Offset for clicking
    animation_frames: int = 1
    frame_delay: float = 0.1
    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotation_range: Tuple[float, float] = (-10, 10)
    confidence_threshold: float = 0.75
    matching_modes: List[MatchingMode] = field(default_factory=lambda: [MatchingMode.SIGNATURE_COLORS])


class Sprite:
    """Enhanced Sprite class - SerpentAI compatible with improvements"""
    
    def __init__(self, name: str, image_data: Union[np.ndarray, List[np.ndarray], str]):
        """
        Initialize sprite
        
        Args:
            name: Sprite name
            image_data: Image array, list of arrays, or path to image file
        """
        self.name = name
        
        # Load image data
        if isinstance(image_data, str):
            # Load from file
            image_data = cv2.imread(image_data, cv2.IMREAD_UNCHANGED)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            
        # Convert to 4D array for SerpentAI compatibility
        if isinstance(image_data, list):
            # Multiple frames
            self.image_data = np.array(image_data)
            if len(self.image_data.shape) == 3:
                # Add animation dimension
                self.image_data = np.expand_dims(self.image_data, axis=3)
        else:
            # Single frame
            if len(image_data.shape) == 2:
                # Grayscale
                image_data = np.stack([image_data] * 3, axis=2)
            if len(image_data.shape) == 3:
                # Add animation dimension
                self.image_data = np.expand_dims(image_data, axis=3)
            else:
                self.image_data = image_data
                
        # Ensure 4D shape: (height, width, channels, animations)
        if len(self.image_data.shape) != 4:
            raise ValueError(f"Invalid image shape: {self.image_data.shape}")
            
        # Extract alpha channel if present
        if self.image_data.shape[2] == 4:
            self.alpha_channel = self.image_data[:, :, 3, :]
            self.image_data = self.image_data[:, :, :3, :]
        else:
            self.alpha_channel = None
            
        # Sprite properties
        self.height = self.image_data.shape[0]
        self.width = self.image_data.shape[1]
        self.channels = self.image_data.shape[2]
        self.animation_states = self.image_data.shape[3]
        
        # Cached computations
        self._signature_colors = None
        self._constellation_of_pixels = None
        self._keypoints = None
        self._descriptors = None
        self._histogram = None
        self._hash = None
        
        # Metadata
        self.metadata = SpriteMetadata(name=name)
        
    @property
    def signature_colors(self) -> np.ndarray:
        """Get signature colors - SerpentAI compatible"""
        if self._signature_colors is None:
            self._signature_colors = self._generate_signature_colors()
        return self._signature_colors
        
    @property
    def constellation_of_pixels(self) -> List[Dict]:
        """Get constellation of pixels - SerpentAI compatible"""
        if self._constellation_of_pixels is None:
            self._constellation_of_pixels = self._generate_constellation_of_pixels()
        return self._constellation_of_pixels
        
    def _generate_signature_colors(self, quantity: int = 8) -> np.ndarray:
        """Generate signature colors - SerpentAI compatible"""
        # Use first animation frame
        image = self.image_data[:, :, :, 0]
        
        # Reshape to list of pixels
        pixels = image.reshape(-1, self.channels)
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        # Get top colors
        signature_colors = unique_colors[sorted_indices[:quantity]]
        
        # Pad if needed
        if len(signature_colors) < quantity:
            padding = np.zeros((quantity - len(signature_colors), self.channels))
            signature_colors = np.vstack([signature_colors, padding])
            
        return signature_colors
        
    def _generate_constellation_of_pixels(self, quantity: int = 8) -> List[Dict]:
        """Generate constellation of pixels - SerpentAI compatible with improvements"""
        constellation = []
        image = self.image_data[:, :, :, 0]
        
        # Method 1: Use signature colors (SerpentAI compatible)
        for color in self.signature_colors[:quantity]:
            # Find pixels matching this color
            mask = np.all(image == color, axis=2)
            locations = np.argwhere(mask)
            
            if len(locations) > 0:
                # Pick random location
                idx = np.random.randint(0, len(locations))
                y, x = locations[idx]
                
                constellation.append({
                    'coordinates': (int(y), int(x)),
                    'color': color.tolist()
                })
                
        # Method 2: Use corner detection for more robust points
        if HAS_SKIMAGE and len(constellation) < quantity:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=quantity - len(constellation),
                qualityLevel=0.01, minDistance=10
            )
            
            if corners is not None:
                for corner in corners:
                    x, y = corner[0]
                    x, y = int(x), int(y)
                    
                    if y < image.shape[0] and x < image.shape[1]:
                        constellation.append({
                            'coordinates': (y, x),
                            'color': image[y, x].tolist()
                        })
                        
        return constellation[:quantity]
        
    def get_frame(self, index: int = 0) -> np.ndarray:
        """Get specific animation frame"""
        if index >= self.animation_states:
            index = index % self.animation_states
        return self.image_data[:, :, :, index]
        
    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        """Get alpha mask for frame"""
        if self.alpha_channel is None:
            return None
        if index >= self.animation_states:
            index = index % self.animation_states
        return self.alpha_channel[:, :, index]
        
    @classmethod
    def locate_color(cls, color: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Locate all pixels of a specific color - SerpentAI compatible"""
        mask = np.all(image == color, axis=2)
        return np.argwhere(mask)
        
    def to_dict(self) -> Dict:
        """Convert sprite to dictionary"""
        return {
            'name': self.name,
            'width': self.width,
            'height': self.height,
            'channels': self.channels,
            'animation_states': self.animation_states,
            'signature_colors': self.signature_colors.tolist(),
            'constellation': self.constellation_of_pixels,
            'metadata': {
                'tags': self.metadata.tags,
                'category': self.metadata.category,
                'click_offset': self.metadata.click_offset
            }
        }
        
    def save(self, path: str):
        """Save sprite to file"""
        save_path = Path(path)
        
        # Save image data
        for i in range(self.animation_states):
            frame = self.get_frame(i)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if self.animation_states == 1:
                image_path = save_path / f"{self.name}.png"
            else:
                image_path = save_path / f"{self.name}_{i}.png"
                
            cv2.imwrite(str(image_path), frame_bgr)
            
        # Save metadata
        metadata_path = save_path / f"{self.name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str, name: Optional[str] = None) -> 'Sprite':
        """Load sprite from file"""
        load_path = Path(path)
        
        if load_path.is_file():
            # Single file
            sprite_name = name or load_path.stem
            return cls(sprite_name, str(load_path))
        else:
            # Directory with multiple frames
            sprite_name = name or load_path.name
            frames = []
            
            # Load all frames
            for image_path in sorted(load_path.glob(f"{sprite_name}*.png")):
                frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            return cls(sprite_name, frames)


class SpriteIdentifier:
    """Enhanced Sprite Identifier - SerpentAI compatible with improvements"""
    
    def __init__(self, sprites: Optional[Dict[str, Sprite]] = None):
        """
        Initialize sprite identifier
        
        Args:
            sprites: Dictionary of sprites to identify
        """
        self.sprites = sprites or {}
        
        # Cache for performance
        self.cache = {}
        self.cache_size = 100
        
    def identify(self, sprite: Union[Sprite, np.ndarray], 
                mode: Union[str, MatchingMode] = "SIGNATURE_COLORS",
                score_threshold: float = 75, debug: bool = False) -> str:
        """
        Identify sprite - SerpentAI compatible
        
        Args:
            sprite: Sprite or image array to identify
            mode: Matching mode
            score_threshold: Minimum score for match
            debug: Enable debug output
            
        Returns:
            Sprite name or "UNKNOWN"
        """
        if isinstance(mode, str):
            mode = MatchingMode[mode]
            
        # Convert image to sprite if needed
        if isinstance(sprite, np.ndarray):
            sprite = Sprite("query", sprite)
            
        best_match = None
        best_score = 0
        scores = {}
        
        # Try to match against all sprites
        for name, candidate in self.sprites.items():
            if mode == MatchingMode.SIGNATURE_COLORS:
                score = self._identify_by_signature_colors(sprite, candidate)
            elif mode == MatchingMode.CONSTELLATION_OF_PIXELS:
                score = self._identify_by_constellation(sprite, candidate)
            elif mode == MatchingMode.SSIM:
                score = self._identify_by_ssim(sprite, candidate)
            elif mode == MatchingMode.TEMPLATE_MATCHING:
                score = self._identify_by_template(sprite, candidate)
            elif mode == MatchingMode.HISTOGRAM:
                score = self._identify_by_histogram(sprite, candidate)
            elif mode == MatchingMode.MULTI_METHOD:
                score = self._identify_multi_method(sprite, candidate)
            else:
                score = 0
                
            scores[name] = score
            
            if score > best_score:
                best_score = score
                best_match = name
                
        if debug:
            logger.debug(f"Identification scores: {scores}")
            
        if best_score >= score_threshold:
            return best_match
        else:
            return "UNKNOWN"
            
    def _identify_by_signature_colors(self, sprite: Sprite, candidate: Sprite) -> float:
        """Identify by signature colors - SerpentAI compatible"""
        sprite_colors = set(map(tuple, sprite.signature_colors))
        candidate_colors = set(map(tuple, candidate.signature_colors))
        
        if not candidate_colors:
            return 0
            
        intersection = len(sprite_colors & candidate_colors)
        union = len(sprite_colors | candidate_colors)
        
        if union == 0:
            return 0
            
        return (intersection / union) * 100
        
    def _identify_by_constellation(self, sprite: Sprite, candidate: Sprite) -> float:
        """Identify by constellation of pixels - SerpentAI compatible"""
        if sprite.width != candidate.width or sprite.height != candidate.height:
            return 0
            
        matches = 0
        sprite_frame = sprite.get_frame(0)
        candidate_constellation = candidate.constellation_of_pixels
        
        for point in candidate_constellation:
            y, x = point['coordinates']
            expected_color = np.array(point['color'])
            
            if y < sprite_frame.shape[0] and x < sprite_frame.shape[1]:
                actual_color = sprite_frame[y, x]
                if np.allclose(actual_color, expected_color, atol=10):
                    matches += 1
                    
        if len(candidate_constellation) == 0:
            return 0
            
        return (matches / len(candidate_constellation)) * 100
        
    def _identify_by_ssim(self, sprite: Sprite, candidate: Sprite) -> float:
        """Identify by SSIM - SerpentAI compatible"""
        if not HAS_SKIMAGE:
            return 0
            
        # Resize to same size
        sprite_frame = sprite.get_frame(0)
        candidate_frame = candidate.get_frame(0)
        
        if sprite_frame.shape != candidate_frame.shape:
            # Resize sprite to candidate size
            sprite_frame = cv2.resize(sprite_frame, (candidate.width, candidate.height))
            
        # Convert to grayscale
        sprite_gray = cv2.cvtColor(sprite_frame, cv2.COLOR_RGB2GRAY)
        candidate_gray = cv2.cvtColor(candidate_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        score = structural_similarity(sprite_gray, candidate_gray)
        
        return score * 100
        
    def _identify_by_template(self, sprite: Sprite, candidate: Sprite) -> float:
        """Identify by template matching"""
        sprite_frame = sprite.get_frame(0)
        candidate_frame = candidate.get_frame(0)
        
        # Must be same size for template matching
        if sprite_frame.shape != candidate_frame.shape:
            return 0
            
        # Use normalized cross-correlation
        sprite_gray = cv2.cvtColor(sprite_frame, cv2.COLOR_RGB2GRAY)
        candidate_gray = cv2.cvtColor(candidate_frame, cv2.COLOR_RGB2GRAY)
        
        result = cv2.matchTemplate(sprite_gray, candidate_gray, cv2.TM_CCOEFF_NORMED)
        score = result[0, 0] if result.size > 0 else 0
        
        return max(0, score * 100)
        
    def _identify_by_histogram(self, sprite: Sprite, candidate: Sprite) -> float:
        """Identify by histogram comparison"""
        sprite_frame = sprite.get_frame(0)
        candidate_frame = candidate.get_frame(0)
        
        # Calculate histograms
        hist_sprite = cv2.calcHist([sprite_frame], [0, 1, 2], None, 
                                  [32, 32, 32], [0, 256, 0, 256, 0, 256])
        hist_candidate = cv2.calcHist([candidate_frame], [0, 1, 2], None,
                                     [32, 32, 32], [0, 256, 0, 256, 0, 256])
        
        # Normalize
        hist_sprite = cv2.normalize(hist_sprite, hist_sprite).flatten()
        hist_candidate = cv2.normalize(hist_candidate, hist_candidate).flatten()
        
        # Compare using correlation
        score = cv2.compareHist(hist_sprite, hist_candidate, cv2.HISTCMP_CORREL)
        
        return max(0, score * 100)
        
    def _identify_multi_method(self, sprite: Sprite, candidate: Sprite) -> float:
        """Identify using multiple methods"""
        scores = []
        weights = []
        
        # Signature colors (fast, weight=1)
        scores.append(self._identify_by_signature_colors(sprite, candidate))
        weights.append(1.0)
        
        # Histogram (fast, weight=1)
        scores.append(self._identify_by_histogram(sprite, candidate))
        weights.append(1.0)
        
        # SSIM if same size (accurate, weight=2)
        if sprite.width == candidate.width and sprite.height == candidate.height:
            scores.append(self._identify_by_ssim(sprite, candidate))
            weights.append(2.0)
            
        # Weighted average
        if sum(weights) == 0:
            return 0
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return weighted_score
        
    def add_sprite(self, sprite: Sprite):
        """Add sprite to identifier"""
        self.sprites[sprite.name] = sprite
        
    def remove_sprite(self, name: str):
        """Remove sprite from identifier"""
        if name in self.sprites:
            del self.sprites[name]


class SpriteLocator:
    """Enhanced Sprite Locator - SerpentAI compatible with improvements"""
    
    def __init__(self):
        """Initialize sprite locator"""
        self.cache = {}
        self.last_locations = {}
        
    def locate(self, sprite: Optional[Sprite] = None, 
              game_frame: Optional[np.ndarray] = None,
              screen_region: Optional[Tuple[int, int, int, int]] = None,
              use_global_location: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate sprite in frame - SerpentAI compatible
        
        Args:
            sprite: Sprite to locate
            game_frame: Frame to search in (can be GameFrame object)
            screen_region: Region to search (y, x, height, width)
            use_global_location: Return global coordinates
            
        Returns:
            Bounding box (y1, x1, y2, x2) or None
        """
        if sprite is None or game_frame is None:
            return None
            
        # Extract frame from GameFrame if needed
        if hasattr(game_frame, 'frame'):
            frame = game_frame.frame
        else:
            frame = game_frame
            
        # Extract search region
        if screen_region:
            y, x, h, w = screen_region
            search_frame = frame[y:y+h, x:x+w]
            offset_y, offset_x = y, x
        else:
            search_frame = frame
            offset_y, offset_x = 0, 0
            
        # Try constellation matching first (SerpentAI method)
        location = self._locate_by_constellation(sprite, search_frame)
        
        # Fallback to template matching
        if location is None:
            location = self._locate_by_template(sprite, search_frame)
            
        # Fallback to feature matching
        if location is None and HAS_SKIMAGE:
            location = self._locate_by_features(sprite, search_frame)
            
        if location is not None and use_global_location:
            # Adjust to global coordinates
            y1, x1, y2, x2 = location
            location = (y1 + offset_y, x1 + offset_x, 
                       y2 + offset_y, x2 + offset_x)
            
        # Cache result
        if location is not None:
            self.last_locations[sprite.name] = location
            
        return location
        
    def _locate_by_constellation(self, sprite: Sprite, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Locate by constellation - SerpentAI method"""
        constellation = sprite.constellation_of_pixels
        if not constellation:
            return None
            
        # Use first constellation point as seed
        seed_point = constellation[0]
        seed_color = np.array(seed_point['color'])
        
        # Find all pixels matching seed color
        color_locations = Sprite.locate_color(seed_color, frame)
        
        if len(color_locations) == 0:
            return None
            
        # Check each potential location
        for y, x in color_locations:
            # Check if all constellation points match
            matches = 0
            
            for point in constellation:
                py, px = point['coordinates']
                expected_color = np.array(point['color'])
                
                # Calculate absolute position
                check_y = y + (py - constellation[0]['coordinates'][0])
                check_x = x + (px - constellation[0]['coordinates'][1])
                
                # Check bounds
                if (0 <= check_y < frame.shape[0] and 
                    0 <= check_x < frame.shape[1]):
                    actual_color = frame[check_y, check_x]
                    
                    if np.allclose(actual_color, expected_color, atol=10):
                        matches += 1
                        
            # If enough matches, we found it
            if matches >= len(constellation) * 0.75:  # 75% threshold
                # Calculate bounding box
                y1 = y - constellation[0]['coordinates'][0]
                x1 = x - constellation[0]['coordinates'][1]
                y2 = y1 + sprite.height
                x2 = x1 + sprite.width
                
                # Validate bounds
                if (0 <= y1 < frame.shape[0] and 0 <= x1 < frame.shape[1] and
                    0 <= y2 <= frame.shape[0] and 0 <= x2 <= frame.shape[1]):
                    return (y1, x1, y2, x2)
                    
        return None
        
    def _locate_by_template(self, sprite: Sprite, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Locate by template matching"""
        template = sprite.get_frame(0)
        
        # Check if template fits in frame
        if (template.shape[0] > frame.shape[0] or 
            template.shape[1] > frame.shape[1]):
            return None
            
        # Convert to grayscale for matching
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check threshold
        if max_val >= sprite.metadata.confidence_threshold:
            x1, y1 = max_loc
            x2 = x1 + sprite.width
            y2 = y1 + sprite.height
            return (y1, x1, y2, x2)
            
        return None
        
    def _locate_by_features(self, sprite: Sprite, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Locate by feature matching"""
        template = sprite.get_frame(0)
        
        # Create ORB detector
        orb = cv2.ORB_create()
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(frame, None)
        
        if des1 is None or des2 is None:
            return None
            
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Need minimum matches
        if len(matches) < 10:
            return None
            
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return None
            
        # Transform corners
        h, w = template.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        # Get bounding box
        x_coords = dst[:, 0, 0]
        y_coords = dst[:, 0, 1]
        
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        # Validate bounds
        if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
            0 <= x2 <= frame.shape[1] and 0 <= y2 <= frame.shape[0]):
            return (y1, x1, y2, x2)
            
        return None
        
    def locate_all(self, sprite: Sprite, frame: np.ndarray, 
                  threshold: float = 0.8) -> List[Tuple[int, int, int, int]]:
        """Locate all instances of sprite in frame"""
        locations = []
        template = sprite.get_frame(0)
        
        # Template matching with multiple detections
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find all matches above threshold
        loc = np.where(result >= threshold)
        
        for pt in zip(*loc[::-1]):
            x1, y1 = pt
            x2 = x1 + sprite.width
            y2 = y1 + sprite.height
            locations.append((y1, x1, y2, x2))
            
        # Non-maximum suppression to remove duplicates
        if locations:
            locations = self._non_max_suppression(locations)
            
        return locations
        
    def _non_max_suppression(self, boxes: List[Tuple[int, int, int, int]], 
                            overlap_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Apply non-maximum suppression to remove duplicate detections"""
        if len(boxes) == 0:
            return []
            
        boxes = np.array(boxes)
        
        # Calculate areas
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        pick = []
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find overlap with other boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / areas[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlap_thresh)[0])))
                
        return [tuple(boxes[i]) for i in pick]


# SerpentAI compatibility functions
def identify_sprite(sprite_identifier: SpriteIdentifier, sprite: Union[Sprite, np.ndarray],
                   mode: str = "SIGNATURE_COLORS") -> str:
    """Identify sprite - SerpentAI compatible"""
    return sprite_identifier.identify(sprite, mode)


def locate_sprite(sprite_locator: SpriteLocator, sprite: Sprite, 
                 game_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Locate sprite - SerpentAI compatible"""
    return sprite_locator.locate(sprite, game_frame)