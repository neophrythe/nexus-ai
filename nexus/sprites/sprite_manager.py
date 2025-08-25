"""Sprite identification and management system"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class Sprite:
    """Sprite definition"""
    name: str
    image: np.ndarray
    mask: Optional[np.ndarray]
    metadata: Dict[str, Any]
    hash: str
    
    @property
    def size(self) -> Tuple[int, int]:
        return (self.image.shape[1], self.image.shape[0])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "hash": self.hash,
            "metadata": self.metadata
        }


@dataclass
class SpriteMatch:
    """Result of sprite matching"""
    sprite: Sprite
    location: Tuple[int, int]
    confidence: float
    bbox: Tuple[int, int, int, int]
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class SpriteManager:
    """Manage game sprites for identification"""
    
    def __init__(self, sprites_dir: Optional[Path] = None):
        self.sprites_dir = Path(sprites_dir) if sprites_dir else Path("sprites")
        self.sprites: Dict[str, Sprite] = {}
        self.sprite_groups: Dict[str, List[str]] = {}
        self._cache: Dict[str, Any] = {}
        
    def load_sprites(self, game_name: Optional[str] = None) -> None:
        """Load sprites from directory"""
        
        if game_name:
            sprite_path = self.sprites_dir / game_name
        else:
            sprite_path = self.sprites_dir
        
        if not sprite_path.exists():
            logger.warning(f"Sprite directory not found: {sprite_path}")
            return
        
        # Load sprite manifest if exists
        manifest_file = sprite_path / "manifest.json"
        manifest = {}
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        
        # Load sprite images
        for image_file in sprite_path.glob("*.png"):
            sprite_name = image_file.stem
            
            # Load image
            image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            
            # Extract alpha channel as mask if present
            mask = None
            if image.shape[2] == 4:
                mask = image[:, :, 3]
                image = image[:, :, :3]
            
            # Get metadata from manifest
            metadata = manifest.get(sprite_name, {})
            
            # Calculate hash
            sprite_hash = hashlib.md5(image.tobytes()).hexdigest()
            
            # Create sprite
            sprite = Sprite(
                name=sprite_name,
                image=image,
                mask=mask,
                metadata=metadata,
                hash=sprite_hash
            )
            
            self.sprites[sprite_name] = sprite
            
            # Add to groups
            groups = metadata.get("groups", [])
            for group in groups:
                if group not in self.sprite_groups:
                    self.sprite_groups[group] = []
                self.sprite_groups[group].append(sprite_name)
        
        logger.info(f"Loaded {len(self.sprites)} sprites")
    
    def add_sprite(self, name: str, image: np.ndarray, 
                  mask: Optional[np.ndarray] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Sprite:
        """Add a sprite manually"""
        
        # Calculate hash
        sprite_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        sprite = Sprite(
            name=name,
            image=image,
            mask=mask,
            metadata=metadata or {},
            hash=sprite_hash
        )
        
        self.sprites[name] = sprite
        return sprite
    
    def find_sprite(self, frame: np.ndarray, sprite_name: str,
                   confidence_threshold: float = 0.8,
                   use_mask: bool = True) -> Optional[SpriteMatch]:
        """Find a specific sprite in frame"""
        
        if sprite_name not in self.sprites:
            logger.warning(f"Sprite {sprite_name} not found")
            return None
        
        sprite = self.sprites[sprite_name]
        
        # Use template matching
        if use_mask and sprite.mask is not None:
            # Match with mask
            result = cv2.matchTemplate(
                frame, sprite.image, cv2.TM_CCORR_NORMED, mask=sprite.mask
            )
        else:
            # Match without mask
            result = cv2.matchTemplate(
                frame, sprite.image, cv2.TM_CCOEFF_NORMED
            )
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= confidence_threshold:
            x, y = max_loc
            w, h = sprite.size
            
            return SpriteMatch(
                sprite=sprite,
                location=(x, y),
                confidence=float(max_val),
                bbox=(x, y, x + w, y + h)
            )
        
        return None
    
    def find_all_sprites(self, frame: np.ndarray, sprite_name: str,
                        confidence_threshold: float = 0.8,
                        use_mask: bool = True,
                        max_matches: int = 100) -> List[SpriteMatch]:
        """Find all occurrences of a sprite"""
        
        if sprite_name not in self.sprites:
            logger.warning(f"Sprite {sprite_name} not found")
            return []
        
        sprite = self.sprites[sprite_name]
        
        # Use template matching
        if use_mask and sprite.mask is not None:
            result = cv2.matchTemplate(
                frame, sprite.image, cv2.TM_CCORR_NORMED, mask=sprite.mask
            )
        else:
            result = cv2.matchTemplate(
                frame, sprite.image, cv2.TM_CCOEFF_NORMED
            )
        
        # Find all matches above threshold
        locations = np.where(result >= confidence_threshold)
        
        matches = []
        w, h = sprite.size
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            confidence = float(result[y, x])
            
            match = SpriteMatch(
                sprite=sprite,
                location=(x, y),
                confidence=confidence,
                bbox=(x, y, x + w, y + h)
            )
            matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Non-maximum suppression
        filtered_matches = []
        for match in matches[:max_matches]:
            # Check if overlaps with existing matches
            overlap = False
            for existing in filtered_matches:
                if self._iou(match.bbox, existing.bbox) > 0.5:
                    overlap = True
                    break
            
            if not overlap:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def find_sprites_in_group(self, frame: np.ndarray, group_name: str,
                             confidence_threshold: float = 0.8) -> List[SpriteMatch]:
        """Find all sprites in a group"""
        
        if group_name not in self.sprite_groups:
            logger.warning(f"Sprite group {group_name} not found")
            return []
        
        all_matches = []
        
        for sprite_name in self.sprite_groups[group_name]:
            matches = self.find_all_sprites(
                frame, sprite_name, confidence_threshold
            )
            all_matches.extend(matches)
        
        return all_matches
    
    def identify_sprite_at_location(self, frame: np.ndarray,
                                   location: Tuple[int, int],
                                   search_radius: int = 50) -> Optional[SpriteMatch]:
        """Identify sprite at specific location"""
        
        x, y = location
        
        # Extract region around location
        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(frame.shape[1], x + search_radius)
        y2 = min(frame.shape[0], y + search_radius)
        
        region = frame[y1:y2, x1:x2]
        
        best_match = None
        best_confidence = 0
        
        # Try all sprites
        for sprite_name, sprite in self.sprites.items():
            # Skip if sprite is larger than region
            if sprite.size[0] > region.shape[1] or sprite.size[1] > region.shape[0]:
                continue
            
            match = self.find_sprite(region, sprite_name)
            
            if match and match.confidence > best_confidence:
                # Adjust coordinates to frame space
                match.location = (match.location[0] + x1, match.location[1] + y1)
                match.bbox = (
                    match.bbox[0] + x1,
                    match.bbox[1] + y1,
                    match.bbox[2] + x1,
                    match.bbox[3] + y1
                )
                best_match = match
                best_confidence = match.confidence
        
        return best_match
    
    def extract_sprite_from_frame(self, frame: np.ndarray,
                                 bbox: Tuple[int, int, int, int],
                                 name: str) -> Sprite:
        """Extract and save a sprite from frame"""
        
        x1, y1, x2, y2 = bbox
        sprite_image = frame[y1:y2, x1:x2].copy()
        
        # Save sprite
        sprite = self.add_sprite(name, sprite_image)
        
        # Save to file
        if self.sprites_dir.exists():
            sprite_file = self.sprites_dir / f"{name}.png"
            cv2.imwrite(str(sprite_file), sprite_image)
            logger.info(f"Saved sprite {name} to {sprite_file}")
        
        return sprite
    
    def create_sprite_mask(self, sprite_image: np.ndarray,
                          lower_bound: tuple, upper_bound: tuple) -> np.ndarray:
        """Create mask for sprite based on color range"""
        
        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(sprite_image, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def match_with_rotation(self, frame: np.ndarray, sprite_name: str,
                          angle_range: Tuple[float, float] = (-30, 30),
                          angle_step: float = 5,
                          confidence_threshold: float = 0.8) -> Optional[SpriteMatch]:
        """Match sprite with rotation"""
        
        if sprite_name not in self.sprites:
            return None
        
        sprite = self.sprites[sprite_name]
        best_match = None
        best_confidence = 0
        
        for angle in np.arange(angle_range[0], angle_range[1], angle_step):
            # Rotate sprite
            center = (sprite.size[0] // 2, sprite.size[1] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(sprite.image, matrix, sprite.size)
            
            # Try matching
            result = cv2.matchTemplate(frame, rotated, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence and max_val >= confidence_threshold:
                x, y = max_loc
                w, h = sprite.size
                
                best_match = SpriteMatch(
                    sprite=sprite,
                    location=(x, y),
                    confidence=float(max_val),
                    bbox=(x, y, x + w, y + h)
                )
                best_match.metadata = {"rotation": angle}
                best_confidence = max_val
        
        return best_match
    
    def match_with_scale(self, frame: np.ndarray, sprite_name: str,
                        scale_range: Tuple[float, float] = (0.8, 1.2),
                        scale_step: float = 0.1,
                        confidence_threshold: float = 0.8) -> Optional[SpriteMatch]:
        """Match sprite with scaling"""
        
        if sprite_name not in self.sprites:
            return None
        
        sprite = self.sprites[sprite_name]
        best_match = None
        best_confidence = 0
        
        for scale in np.arange(scale_range[0], scale_range[1], scale_step):
            # Scale sprite
            new_size = (int(sprite.size[0] * scale), int(sprite.size[1] * scale))
            scaled = cv2.resize(sprite.image, new_size)
            
            # Skip if scaled sprite is larger than frame
            if new_size[0] > frame.shape[1] or new_size[1] > frame.shape[0]:
                continue
            
            # Try matching
            result = cv2.matchTemplate(frame, scaled, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence and max_val >= confidence_threshold:
                x, y = max_loc
                w, h = new_size
                
                best_match = SpriteMatch(
                    sprite=sprite,
                    location=(x, y),
                    confidence=float(max_val),
                    bbox=(x, y, x + w, y + h)
                )
                best_match.metadata = {"scale": scale}
                best_confidence = max_val
        
        return best_match
    
    def _iou(self, bbox1: Tuple[int, int, int, int],
            bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
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
    
    def draw_matches(self, frame: np.ndarray, matches: List[SpriteMatch]) -> np.ndarray:
        """Draw sprite matches on frame"""
        
        result = frame.copy()
        
        for match in matches:
            x1, y1, x2, y2 = match.bbox
            
            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{match.sprite.name}: {match.confidence:.2f}"
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw center point
            cv2.circle(result, match.center, 3, (0, 0, 255), -1)
        
        return result