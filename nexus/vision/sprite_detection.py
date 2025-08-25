"""Advanced sprite detection algorithms adapted from SerpentAI"""

import numpy as np
import cv2
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger()


@dataclass
class SpriteSignature:
    """Sprite signature for fast detection"""
    name: str
    signature_colors: List[Set[Tuple[int, int, int]]]
    constellation_of_pixels: List[Dict[Tuple[int, int], Tuple[int, int, int]]]
    image_shape: Tuple[int, int]
    hash: str = field(default="")
    
    def __post_init__(self):
        if not self.hash:
            # Generate unique hash for sprite
            data = f"{self.name}_{self.image_shape}_{self.signature_colors}"
            self.hash = hashlib.md5(data.encode()).hexdigest()


class AdvancedSpriteDetector:
    """Advanced sprite detection using color signatures and pixel constellations"""
    
    def __init__(self):
        self.sprite_signatures: Dict[str, SpriteSignature] = {}
        self._cache: Dict[str, Any] = {}
    
    def register_sprite(self, name: str, image_data: np.ndarray,
                       signature_colors: Optional[List[Set[Tuple[int, int, int]]]] = None,
                       constellation_of_pixels: Optional[List[Dict]] = None) -> SpriteSignature:
        """
        Register a sprite for detection.
        
        Args:
            name: Sprite name
            image_data: Sprite image(s) as 3D or 4D numpy array
            signature_colors: Pre-computed signature colors
            constellation_of_pixels: Pre-computed pixel constellation
        
        Returns:
            SpriteSignature object
        """
        # Ensure 4D array (height, width, channels, frames)
        if len(image_data.shape) == 3:
            image_data = image_data[:, :, :, np.newaxis]
        elif len(image_data.shape) != 4:
            raise ValueError(f"Image data must be 3D or 4D, got {len(image_data.shape)}D")
        
        # Generate signature colors if not provided
        if signature_colors is None:
            signature_colors = self._generate_signature_colors(image_data)
        
        # Generate constellation of pixels if not provided
        if constellation_of_pixels is None:
            constellation_of_pixels = self._generate_constellation_of_pixels(
                image_data, signature_colors
            )
        
        # Create sprite signature
        signature = SpriteSignature(
            name=name,
            signature_colors=signature_colors,
            constellation_of_pixels=constellation_of_pixels,
            image_shape=image_data.shape[:2]
        )
        
        self.sprite_signatures[name] = signature
        logger.info(f"Registered sprite '{name}' with shape {signature.image_shape}")
        
        return signature
    
    def detect_sprite(self, image: np.ndarray, sprite_name: str,
                     confidence_threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Detect sprite in image using signature matching.
        
        Args:
            image: Image to search in
            sprite_name: Name of sprite to detect
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of (x, y, confidence) tuples
        """
        if sprite_name not in self.sprite_signatures:
            logger.warning(f"Sprite '{sprite_name}' not registered")
            return []
        
        signature = self.sprite_signatures[sprite_name]
        detections = []
        
        # Use constellation of pixels for fast detection
        for frame_idx, constellation in enumerate(signature.constellation_of_pixels):
            matches = self._find_constellation_matches(
                image, constellation, signature.image_shape
            )
            
            for x, y, confidence in matches:
                if confidence >= confidence_threshold:
                    detections.append((x, y, confidence))
        
        # Non-maximum suppression
        detections = self._non_max_suppression(
            detections, signature.image_shape, overlap_threshold=0.5
        )
        
        return detections
    
    def detect_all_sprites(self, image: np.ndarray,
                          confidence_threshold: float = 0.8) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Detect all registered sprites in image.
        
        Args:
            image: Image to search in
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Dictionary mapping sprite names to detection lists
        """
        all_detections = {}
        
        for sprite_name in self.sprite_signatures:
            detections = self.detect_sprite(image, sprite_name, confidence_threshold)
            if detections:
                all_detections[sprite_name] = detections
        
        return all_detections
    
    def _generate_signature_colors(self, image_data: np.ndarray,
                                  quantity: int = 8) -> List[Set[Tuple[int, int, int]]]:
        """
        Generate signature colors for sprite frames.
        
        Args:
            image_data: 4D array of sprite frames
            quantity: Number of signature colors to extract
        
        Returns:
            List of signature color sets for each frame
        """
        signature_colors = []
        height, width, channels, frames = image_data.shape
        
        for frame_idx in range(frames):
            frame = image_data[:, :, :, frame_idx]
            
            # Reshape for unique color analysis
            pixels = frame.reshape(-1, channels)
            
            # Handle alpha channel if present
            if channels == 4:
                # Filter out transparent pixels
                opaque_mask = pixels[:, 3] > 0
                pixels = pixels[opaque_mask][:, :3]
            else:
                pixels = pixels[:, :3]
            
            if len(pixels) == 0:
                signature_colors.append(set())
                continue
            
            # Find unique colors and their counts
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # Get top colors by frequency
            if len(unique_colors) > quantity:
                top_indices = np.argsort(counts)[-quantity:][::-1]
                top_colors = unique_colors[top_indices]
            else:
                top_colors = unique_colors
            
            # Convert to tuples for hashability
            color_set = set(tuple(map(int, color)) for color in top_colors)
            signature_colors.append(color_set)
        
        return signature_colors
    
    def _generate_constellation_of_pixels(self, image_data: np.ndarray,
                                         signature_colors: List[Set[Tuple[int, int, int]]],
                                         quantity: int = 8) -> List[Dict[Tuple[int, int], Tuple[int, int, int]]]:
        """
        Generate constellation of pixels for sprite frames.
        
        Args:
            image_data: 4D array of sprite frames
            signature_colors: Signature colors for each frame
            quantity: Number of constellation points
        
        Returns:
            List of constellation dictionaries for each frame
        """
        constellation_of_pixels = []
        height, width, channels, frames = image_data.shape
        
        for frame_idx in range(frames):
            frame = image_data[:, :, :3, frame_idx]
            colors = signature_colors[frame_idx]
            
            if not colors:
                constellation_of_pixels.append({})
                continue
            
            constellation = {}
            used_positions = set()
            
            # Sample pixels from signature colors
            for _ in range(min(quantity, len(colors))):
                # Pick a random signature color
                color = random.choice(list(colors))
                
                # Find all locations of this color
                color_locations = self._locate_color(color, frame)
                
                if not color_locations:
                    continue
                
                # Pick a random location not already used
                available_locations = [
                    loc for loc in color_locations 
                    if loc not in used_positions
                ]
                
                if available_locations:
                    y, x = random.choice(available_locations)
                    constellation[(y, x)] = color
                    used_positions.add((y, x))
            
            constellation_of_pixels.append(constellation)
        
        return constellation_of_pixels
    
    def _locate_color(self, color: Tuple[int, int, int],
                     image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find all locations of a specific color in image.
        
        Args:
            color: RGB color tuple
            image: Image to search in
        
        Returns:
            List of (y, x) coordinates
        """
        # Vectorized color matching
        mask = np.all(image == color, axis=-1)
        locations = np.where(mask)
        
        return list(zip(locations[0], locations[1]))
    
    def _find_constellation_matches(self, image: np.ndarray,
                                   constellation: Dict[Tuple[int, int], Tuple[int, int, int]],
                                   sprite_shape: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """
        Find constellation matches in image.
        
        Args:
            image: Image to search in
            constellation: Constellation of pixels to match
            sprite_shape: Shape of the sprite
        
        Returns:
            List of (x, y, confidence) tuples
        """
        if not constellation:
            return []
        
        matches = []
        height, width = image.shape[:2]
        sprite_h, sprite_w = sprite_shape
        
        # Sliding window search with optimization
        step_size = max(1, min(sprite_w // 4, sprite_h // 4))
        
        for y in range(0, height - sprite_h + 1, step_size):
            for x in range(0, width - sprite_w + 1, step_size):
                # Check constellation match
                match_count = 0
                total_points = len(constellation)
                
                for (rel_y, rel_x), expected_color in constellation.items():
                    check_y = y + rel_y
                    check_x = x + rel_x
                    
                    if check_y < height and check_x < width:
                        actual_color = tuple(image[check_y, check_x, :3])
                        
                        # Allow small color variance
                        if self._colors_match(actual_color, expected_color, tolerance=10):
                            match_count += 1
                
                confidence = match_count / total_points if total_points > 0 else 0
                
                if confidence > 0.6:  # Minimum threshold for consideration
                    matches.append((x, y, confidence))
        
        return matches
    
    def _colors_match(self, color1: Tuple[int, int, int],
                     color2: Tuple[int, int, int],
                     tolerance: int = 10) -> bool:
        """
        Check if two colors match within tolerance.
        
        Args:
            color1: First RGB color
            color2: Second RGB color
            tolerance: Maximum difference per channel
        
        Returns:
            True if colors match within tolerance
        """
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))
    
    def _non_max_suppression(self, detections: List[Tuple[int, int, float]],
                           sprite_shape: Tuple[int, int],
                           overlap_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Apply non-maximum suppression to remove duplicate detections.
        
        Args:
            detections: List of (x, y, confidence) tuples
            sprite_shape: Shape of the sprite
            overlap_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        keep = []
        sprite_h, sprite_w = sprite_shape
        
        while detections:
            # Take highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Filter remaining detections
            filtered = []
            x1, y1, _ = current
            box1 = (x1, y1, x1 + sprite_w, y1 + sprite_h)
            
            for detection in detections:
                x2, y2, conf = detection
                box2 = (x2, y2, x2 + sprite_w, y2 + sprite_h)
                
                if self._calculate_iou(box1, box2) < overlap_threshold:
                    filtered.append(detection)
            
            detections = filtered
        
        return keep
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First box (x1, y1, x2, y2)
            box2: Second box (x1, y1, x2, y2)
        
        Returns:
            IoU score
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(self, image: np.ndarray,
                            detections: Dict[str, List[Tuple[int, int, float]]]) -> np.ndarray:
        """
        Visualize sprite detections on image.
        
        Args:
            image: Original image
            detections: Dictionary of sprite detections
        
        Returns:
            Image with visualized detections
        """
        result = image.copy()
        
        # Color palette for different sprites
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for idx, (sprite_name, sprite_detections) in enumerate(detections.items()):
            color = colors[idx % len(colors)]
            signature = self.sprite_signatures.get(sprite_name)
            
            if not signature:
                continue
            
            sprite_h, sprite_w = signature.image_shape
            
            for x, y, confidence in sprite_detections:
                # Draw bounding box
                cv2.rectangle(result, (x, y), (x + sprite_w, y + sprite_h), color, 2)
                
                # Draw label
                label = f"{sprite_name}: {confidence:.2f}"
                cv2.putText(result, label, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result