"""Advanced Sprite Locator with multiple detection algorithms"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import structlog
from collections import Counter
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
import hashlib

from nexus.sprites.sprite_manager import Sprite, SpriteMatch

logger = structlog.get_logger()


@dataclass
class ConstellationPoint:
    """A key point in the sprite constellation"""
    x: int
    y: int
    color: Tuple[int, int, int]
    
    def matches(self, other_color: Tuple[int, int, int], tolerance: int = 10) -> bool:
        """Check if colors match within tolerance"""
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(self.color, other_color))


@dataclass
class SpriteSignature:
    """Advanced sprite signature for matching"""
    name: str
    signature_colors: List[Tuple[int, int, int]]
    constellation: List[ConstellationPoint]
    histogram: np.ndarray
    edge_features: np.ndarray
    shape_descriptor: np.ndarray
    hash: str
    
    @property
    def color_set(self) -> Set[Tuple[int, int, int]]:
        """Get unique colors as set"""
        return set(self.signature_colors)


class AdvancedSpriteLocator:
    """Advanced sprite detection using multiple algorithms"""
    
    def __init__(self, constellation_size: int = 16, signature_colors: int = 8):
        """
        Initialize advanced sprite locator
        
        Args:
            constellation_size: Number of key points in constellation
            signature_colors: Number of dominant colors to extract
        """
        self.constellation_size = constellation_size
        self.signature_colors = signature_colors
        self.signatures: Dict[str, SpriteSignature] = {}
        self._cache: Dict[str, Any] = {}
        
        # Feature detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
    
    def create_sprite_signature(self, sprite: Sprite) -> SpriteSignature:
        """Create advanced signature for sprite"""
        
        image = sprite.image
        
        # 1. Extract signature colors
        colors = self._extract_signature_colors(image, sprite.mask)
        
        # 2. Create constellation of pixels
        constellation = self._create_constellation(image, sprite.mask)
        
        # 3. Create color histogram
        histogram = self._create_histogram(image, sprite.mask)
        
        # 4. Extract edge features
        edge_features = self._extract_edge_features(image)
        
        # 5. Extract shape descriptor
        shape_descriptor = self._extract_shape_descriptor(image, sprite.mask)
        
        # 6. Calculate hash
        sig_hash = hashlib.md5(
            str(colors + [(p.x, p.y, p.color) for p in constellation]).encode()
        ).hexdigest()
        
        signature = SpriteSignature(
            name=sprite.name,
            signature_colors=colors,
            constellation=constellation,
            histogram=histogram,
            edge_features=edge_features,
            shape_descriptor=shape_descriptor,
            hash=sig_hash
        )
        
        self.signatures[sprite.name] = signature
        return signature
    
    def _extract_signature_colors(self, image: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from sprite"""
        
        # Apply mask if provided
        if mask is not None:
            pixels = image[mask > 0].reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        
        n_colors = min(self.signature_colors, len(pixels))
        if n_colors == 0:
            return []
        
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Get cluster centers as dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to tuples
        return [tuple(color) for color in colors]
    
    def _create_constellation(self, image: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> List[ConstellationPoint]:
        """Create constellation of key pixels"""
        
        h, w = image.shape[:2]
        constellation = []
        
        # Use feature detection to find key points
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect corners using Harris corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Get coordinates of corners
        threshold = 0.01 * corners.max()
        corner_coords = np.where(corners > threshold)
        
        # Limit to constellation_size points
        n_points = min(len(corner_coords[0]), self.constellation_size)
        
        if n_points > 0:
            # Sample points evenly
            indices = np.linspace(0, len(corner_coords[0]) - 1, n_points, dtype=int)
            
            for idx in indices:
                y, x = corner_coords[0][idx], corner_coords[1][idx]
                
                # Check mask if provided
                if mask is not None and mask[y, x] == 0:
                    continue
                
                color = tuple(image[y, x].tolist())
                constellation.append(ConstellationPoint(x, y, color))
        
        # If not enough corners, add grid points
        if len(constellation) < self.constellation_size:
            grid_size = int(np.sqrt(self.constellation_size))
            step_y = h // (grid_size + 1)
            step_x = w // (grid_size + 1)
            
            for i in range(1, grid_size + 1):
                for j in range(1, grid_size + 1):
                    y = i * step_y
                    x = j * step_x
                    
                    if mask is not None and mask[y, x] == 0:
                        continue
                    
                    if len(constellation) < self.constellation_size:
                        color = tuple(image[y, x].tolist())
                        constellation.append(ConstellationPoint(x, y, color))
        
        return constellation
    
    def _create_histogram(self, image: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create color histogram for sprite"""
        
        # Calculate histograms for each channel
        hist_b = cv2.calcHist([image], [0], mask, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], mask, [32], [0, 256])
        hist_r = cv2.calcHist([image], [2], mask, [32], [0, 256])
        
        # Concatenate and normalize
        histogram = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        histogram = histogram / histogram.sum()
        
        return histogram
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge histogram
        edge_hist = cv2.calcHist([edges], [0], None, [2], [0, 256])
        edge_hist = edge_hist.flatten() / edge_hist.sum()
        
        # Calculate Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and angle
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx)
        
        # Create histogram of oriented gradients (simplified HOG)
        angle_hist, _ = np.histogram(angle.flatten(), bins=8, range=(-np.pi, np.pi))
        angle_hist = angle_hist / angle_hist.sum()
        
        # Combine features
        features = np.concatenate([edge_hist, angle_hist])
        
        return features
    
    def _extract_shape_descriptor(self, image: np.ndarray,
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract shape descriptors using Hu moments"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calculate moments
        moments = cv2.moments(gray)
        
        # Calculate Hu moments (rotation invariant)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Log scale for better numerical stability
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def locate_by_constellation(self, frame: np.ndarray, sprite_name: str,
                              tolerance: int = 10, min_matches: int = 8) -> List[SpriteMatch]:
        """Locate sprite using constellation matching"""
        
        if sprite_name not in self.signatures:
            logger.warning(f"No signature for sprite {sprite_name}")
            return []
        
        signature = self.signatures[sprite_name]
        constellation = signature.constellation
        
        if len(constellation) < min_matches:
            return []
        
        h, w = frame.shape[:2]
        sprite_h, sprite_w = constellation[-1].y + 1, constellation[-1].x + 1
        
        matches = []
        
        # Sliding window search
        step_size = max(1, min(sprite_w // 4, sprite_h // 4))
        
        for y in range(0, h - sprite_h + 1, step_size):
            for x in range(0, w - sprite_w + 1, step_size):
                # Check constellation points
                matched_points = 0
                total_points = len(constellation)
                
                for point in constellation:
                    px, py = x + point.x, y + point.y
                    
                    if px < w and py < h:
                        frame_color = tuple(frame[py, px].tolist())
                        if point.matches(frame_color, tolerance):
                            matched_points += 1
                
                # Calculate confidence
                confidence = matched_points / total_points
                
                if matched_points >= min_matches:
                    match = SpriteMatch(
                        sprite=Sprite(sprite_name, frame[y:y+sprite_h, x:x+sprite_w], None, {}, ""),
                        location=(x, y),
                        confidence=confidence,
                        bbox=(x, y, x + sprite_w, y + sprite_h)
                    )
                    matches.append(match)
        
        # Non-maximum suppression
        return self._nms(matches, iou_threshold=0.5)
    
    def locate_by_signature_colors(self, frame: np.ndarray, sprite_name: str,
                                  color_tolerance: int = 20,
                                  min_color_matches: int = 4) -> List[SpriteMatch]:
        """Locate sprite using signature color matching"""
        
        if sprite_name not in self.signatures:
            return []
        
        signature = self.signatures[sprite_name]
        target_colors = signature.signature_colors
        
        if len(target_colors) < min_color_matches:
            return []
        
        # Create color masks for each signature color
        masks = []
        for color in target_colors:
            lower = np.array([max(0, c - color_tolerance) for c in color])
            upper = np.array([min(255, c + color_tolerance) for c in color])
            mask = cv2.inRange(frame, lower, upper)
            masks.append(mask)
        
        # Combine masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find contours in combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        matches = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip small areas
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            region = frame[y:y+h, x:x+w]
            
            # Check color distribution in region
            region_colors = self._extract_signature_colors(region)
            
            # Calculate color match score
            matched_colors = 0
            for target_color in target_colors:
                for region_color in region_colors:
                    if all(abs(t - r) <= color_tolerance for t, r in zip(target_color, region_color)):
                        matched_colors += 1
                        break
            
            confidence = matched_colors / len(target_colors)
            
            if matched_colors >= min_color_matches:
                match = SpriteMatch(
                    sprite=Sprite(sprite_name, region, None, {}, ""),
                    location=(x, y),
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h)
                )
                matches.append(match)
        
        return self._nms(matches, iou_threshold=0.5)
    
    def locate_by_histogram(self, frame: np.ndarray, sprite_name: str,
                          correlation_threshold: float = 0.7) -> List[SpriteMatch]:
        """Locate sprite using histogram correlation"""
        
        if sprite_name not in self.signatures:
            return []
        
        signature = self.signatures[sprite_name]
        target_hist = signature.histogram
        
        h, w = frame.shape[:2]
        sprite_size = int(np.sqrt(self.constellation_size) * 10)  # Estimate sprite size
        
        matches = []
        step_size = max(1, sprite_size // 4)
        
        for y in range(0, h - sprite_size + 1, step_size):
            for x in range(0, w - sprite_size + 1, step_size):
                region = frame[y:y+sprite_size, x:x+sprite_size]
                region_hist = self._create_histogram(region)
                
                # Calculate histogram correlation
                correlation = cv2.compareHist(
                    target_hist.reshape(-1, 1),
                    region_hist.reshape(-1, 1),
                    cv2.HISTCMP_CORREL
                )
                
                if correlation >= correlation_threshold:
                    match = SpriteMatch(
                        sprite=Sprite(sprite_name, region, None, {}, ""),
                        location=(x, y),
                        confidence=float(correlation),
                        bbox=(x, y, x + sprite_size, y + sprite_size)
                    )
                    matches.append(match)
        
        return self._nms(matches, iou_threshold=0.5)
    
    def locate_by_features(self, frame: np.ndarray, sprite: Sprite,
                          ratio_threshold: float = 0.7) -> List[SpriteMatch]:
        """Locate sprite using SIFT/ORB feature matching"""
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_sprite = cv2.cvtColor(sprite.image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(gray_sprite, None)
        kp2, des2 = self.sift.detectAndCompute(gray_frame, None)
        
        if des1 is None or des2 is None:
            return []
        
        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_list = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches_list:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:  # Need at least 4 points for homography
            return []
        
        # Find homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            return []
        
        # Transform sprite corners
        h, w = sprite.image.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(dst)
        
        # Calculate confidence based on number of matches
        confidence = min(1.0, len(good_matches) / 20.0)
        
        match = SpriteMatch(
            sprite=sprite,
            location=(x, y),
            confidence=confidence,
            bbox=(x, y, x + w, y + h)
        )
        
        return [match]
    
    def locate_by_ssim(self, frame: np.ndarray, sprite: Sprite,
                      ssim_threshold: float = 0.7) -> List[SpriteMatch]:
        """Locate sprite using structural similarity"""
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_sprite = cv2.cvtColor(sprite.image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray_sprite.shape
        fh, fw = gray_frame.shape
        
        matches = []
        step_size = max(1, min(w // 4, h // 4))
        
        for y in range(0, fh - h + 1, step_size):
            for x in range(0, fw - w + 1, step_size):
                region = gray_frame[y:y+h, x:x+w]
                
                # Calculate SSIM
                score, _ = ssim(gray_sprite, region, full=True)
                
                if score >= ssim_threshold:
                    match = SpriteMatch(
                        sprite=sprite,
                        location=(x, y),
                        confidence=float(score),
                        bbox=(x, y, x + w, y + h)
                    )
                    matches.append(match)
        
        return self._nms(matches, iou_threshold=0.5)
    
    def locate_multi_method(self, frame: np.ndarray, sprite: Sprite,
                          methods: List[str] = None) -> List[SpriteMatch]:
        """Locate sprite using multiple methods and combine results"""
        
        if methods is None:
            methods = ["constellation", "signature_colors", "histogram"]
        
        all_matches = []
        
        # Ensure signature exists
        if sprite.name not in self.signatures:
            self.create_sprite_signature(sprite)
        
        # Apply each method
        if "constellation" in methods:
            matches = self.locate_by_constellation(frame, sprite.name)
            all_matches.extend(matches)
        
        if "signature_colors" in methods:
            matches = self.locate_by_signature_colors(frame, sprite.name)
            all_matches.extend(matches)
        
        if "histogram" in methods:
            matches = self.locate_by_histogram(frame, sprite.name)
            all_matches.extend(matches)
        
        if "features" in methods:
            matches = self.locate_by_features(frame, sprite)
            all_matches.extend(matches)
        
        if "ssim" in methods:
            matches = self.locate_by_ssim(frame, sprite)
            all_matches.extend(matches)
        
        # Combine and filter matches
        return self._combine_matches(all_matches)
    
    def _nms(self, matches: List[SpriteMatch], iou_threshold: float = 0.5) -> List[SpriteMatch]:
        """Non-maximum suppression to remove duplicate detections"""
        
        if not matches:
            return []
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        keep = []
        
        for match in matches:
            # Check overlap with already kept matches
            overlapping = False
            
            for kept_match in keep:
                iou = self._calculate_iou(match.bbox, kept_match.bbox)
                if iou > iou_threshold:
                    overlapping = True
                    break
            
            if not overlapping:
                keep.append(match)
        
        return keep
    
    def _combine_matches(self, matches: List[SpriteMatch]) -> List[SpriteMatch]:
        """Combine matches from multiple methods"""
        
        if not matches:
            return []
        
        # Group matches by location proximity
        groups = []
        
        for match in matches:
            added = False
            
            for group in groups:
                # Check if match belongs to this group
                for existing in group:
                    iou = self._calculate_iou(match.bbox, existing.bbox)
                    if iou > 0.3:  # Lower threshold for grouping
                        group.append(match)
                        added = True
                        break
                
                if added:
                    break
            
            if not added:
                groups.append([match])
        
        # Create combined matches from groups
        combined = []
        
        for group in groups:
            # Average location and confidence
            avg_x = sum(m.location[0] for m in group) / len(group)
            avg_y = sum(m.location[1] for m in group) / len(group)
            avg_confidence = sum(m.confidence for m in group) / len(group)
            
            # Use bbox from highest confidence match
            best_match = max(group, key=lambda m: m.confidence)
            
            combined_match = SpriteMatch(
                sprite=best_match.sprite,
                location=(int(avg_x), int(avg_y)),
                confidence=avg_confidence,
                bbox=best_match.bbox
            )
            
            # Add metadata about methods used
            combined_match.metadata = {"methods_count": len(group)}
            
            combined.append(combined_match)
        
        return combined
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
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
    
    def adaptive_threshold_matching(self, frame: np.ndarray, sprite: Sprite,
                                  initial_threshold: float = 0.8,
                                  min_threshold: float = 0.5,
                                  threshold_step: float = 0.05) -> List[SpriteMatch]:
        """Adaptive threshold matching that lowers threshold until matches found"""
        
        threshold = initial_threshold
        matches = []
        
        while threshold >= min_threshold and not matches:
            # Try template matching with current threshold
            result = cv2.matchTemplate(frame, sprite.image, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                x, y = pt
                w, h = sprite.image.shape[1], sprite.image.shape[0]
                confidence = float(result[y, x])
                
                match = SpriteMatch(
                    sprite=sprite,
                    location=(x, y),
                    confidence=confidence,
                    bbox=(x, y, x + w, y + h)
                )
                matches.append(match)
            
            if not matches:
                threshold -= threshold_step
        
        return self._nms(matches, iou_threshold=0.5)
    
    def get_sprite_at_cursor(self, frame: np.ndarray, cursor_pos: Tuple[int, int],
                           search_radius: int = 50) -> Optional[SpriteMatch]:
        """Identify sprite under cursor position"""
        
        x, y = cursor_pos
        
        # Extract region around cursor
        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(frame.shape[1], x + search_radius)
        y2 = min(frame.shape[0], y + search_radius)
        
        region = frame[y1:y2, x1:x2]
        
        # Try to match all known sprites in region
        best_match = None
        best_confidence = 0
        
        for sprite_name, signature in self.signatures.items():
            # Quick color check first
            region_colors = self._extract_signature_colors(region)
            color_match = any(
                any(all(abs(s - r) <= 20 for s, r in zip(sig_color, reg_color))
                    for reg_color in region_colors)
                for sig_color in signature.signature_colors
            )
            
            if color_match:
                # Try multiple methods
                matches = self.locate_multi_method(
                    region, 
                    Sprite(sprite_name, region, None, {}, ""),
                    methods=["histogram", "ssim"]
                )
                
                for match in matches:
                    if match.confidence > best_confidence:
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