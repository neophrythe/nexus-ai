"""
Advanced Asset Extraction Utilities - Automated sprite and asset extraction from games.
SerpentAI-inspired but with modern computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from collections import defaultdict
from sklearn.cluster import DBSCAN
import logging


@dataclass
class ExtractedAsset:
    """Represents an extracted game asset."""
    image: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    asset_type: str  # sprite, text, ui_element, background
    metadata: Dict[str, Any]
    hash: str
    
    def save(self, filepath: str):
        """Save asset to file."""
        cv2.imwrite(filepath, self.image)
        
        # Save metadata
        meta_path = Path(filepath).with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'bbox': self.bbox,
                'confidence': self.confidence,
                'asset_type': self.asset_type,
                'metadata': self.metadata,
                'hash': self.hash
            }, f, indent=2)


class AdvancedAssetExtractor:
    """Advanced asset extraction from game screenshots."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.extracted_assets: Dict[str, ExtractedAsset] = {}
        
    def extract_all_assets(self, screenshot: np.ndarray,
                          min_size: int = 16,
                          max_size: int = 512) -> List[ExtractedAsset]:
        """Extract all possible assets from a screenshot."""
        assets = []
        
        # Extract sprites
        sprites = self.extract_sprites(screenshot, min_size, max_size)
        assets.extend(sprites)
        
        # Extract UI elements
        ui_elements = self.extract_ui_elements(screenshot)
        assets.extend(ui_elements)
        
        # Extract text regions
        text_regions = self.extract_text_regions(screenshot)
        assets.extend(text_regions)
        
        # Extract backgrounds/tiles
        backgrounds = self.extract_background_patterns(screenshot)
        assets.extend(backgrounds)
        
        # Deduplicate assets
        assets = self._deduplicate_assets(assets)
        
        return assets
    
    def extract_sprites(self, screenshot: np.ndarray,
                       min_size: int = 16,
                       max_size: int = 256) -> List[ExtractedAsset]:
        """Extract sprite-like objects from screenshot."""
        assets = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Edge-based detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if min_size <= w <= max_size and min_size <= h <= max_size:
                # Check aspect ratio (sprites usually square-ish)
                aspect_ratio = w / h
                if 0.5 <= aspect_ratio <= 2.0:
                    sprite = screenshot[y:y+h, x:x+w]
                    
                    # Check if it's a valid sprite (not just noise)
                    if self._is_valid_sprite(sprite):
                        asset = ExtractedAsset(
                            image=sprite,
                            bbox=(x, y, w, h),
                            confidence=0.8,
                            asset_type='sprite',
                            metadata={'method': 'edge_detection'},
                            hash=self._compute_hash(sprite)
                        )
                        assets.append(asset)
        
        # Method 2: Color clustering
        color_sprites = self._extract_by_color_clustering(screenshot, min_size, max_size)
        assets.extend(color_sprites)
        
        return assets
    
    def _extract_by_color_clustering(self, screenshot: np.ndarray,
                                    min_size: int, max_size: int) -> List[ExtractedAsset]:
        """Extract sprites using color clustering."""
        assets = []
        
        # Reduce colors using k-means
        h, w = screenshot.shape[:2]
        pixels = screenshot.reshape((-1, 3))
        
        # Use a limited color palette
        from sklearn.cluster import KMeans
        n_colors = 16
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Create quantized image
        labels = kmeans.labels_.reshape((h, w))
        
        # Find connected components for each color
        for color_idx in range(n_colors):
            mask = (labels == color_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if min_size <= w <= max_size and min_size <= h <= max_size:
                    sprite = screenshot[y:y+h, x:x+w]
                    
                    if self._is_valid_sprite(sprite):
                        asset = ExtractedAsset(
                            image=sprite,
                            bbox=(x, y, w, h),
                            confidence=0.7,
                            asset_type='sprite',
                            metadata={
                                'method': 'color_clustering',
                                'dominant_color': kmeans.cluster_centers_[color_idx].tolist()
                            },
                            hash=self._compute_hash(sprite)
                        )
                        assets.append(asset)
        
        return assets
    
    def extract_ui_elements(self, screenshot: np.ndarray) -> List[ExtractedAsset]:
        """Extract UI elements like buttons, panels, health bars."""
        assets = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangles (common UI shape)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangles (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # UI elements are often wider than tall
                if w > h * 1.5 and w > 50:
                    element = screenshot[y:y+h, x:x+w]
                    
                    asset = ExtractedAsset(
                        image=element,
                        bbox=(x, y, w, h),
                        confidence=0.75,
                        asset_type='ui_element',
                        metadata={
                            'shape': 'rectangle',
                            'vertices': approx.tolist()
                        },
                        hash=self._compute_hash(element)
                    )
                    assets.append(asset)
        
        # Detect circular elements (e.g., minimap, radial menus)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=50, param2=30, minRadius=20, maxRadius=200)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                x, y, r = circle
                x1, y1 = max(0, x-r), max(0, y-r)
                x2, y2 = min(screenshot.shape[1], x+r), min(screenshot.shape[0], y+r)
                
                element = screenshot[y1:y2, x1:x2]
                
                asset = ExtractedAsset(
                    image=element,
                    bbox=(x1, y1, x2-x1, y2-y1),
                    confidence=0.7,
                    asset_type='ui_element',
                    metadata={
                        'shape': 'circle',
                        'center': (int(x), int(y)),
                        'radius': int(r)
                    },
                    hash=self._compute_hash(element)
                )
                assets.append(asset)
        
        return assets
    
    def extract_text_regions(self, screenshot: np.ndarray) -> List[ExtractedAsset]:
        """Extract regions containing text."""
        assets = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text regions are usually wider than tall
            if w > h * 2 and w > 30:
                text_region = screenshot[y:y+h, x:x+w]
                
                # Verify it looks like text (high contrast, horizontal features)
                if self._is_text_region(text_region):
                    asset = ExtractedAsset(
                        image=text_region,
                        bbox=(x, y, w, h),
                        confidence=0.6,
                        asset_type='text',
                        metadata={'detected_method': 'morphological'},
                        hash=self._compute_hash(text_region)
                    )
                    assets.append(asset)
        
        return assets
    
    def extract_background_patterns(self, screenshot: np.ndarray,
                                   tile_size: int = 32) -> List[ExtractedAsset]:
        """Extract repeating background patterns and tiles."""
        assets = []
        h, w = screenshot.shape[:2]
        
        # Sample tiles from the image
        tiles = []
        positions = []
        
        for y in range(0, h - tile_size, tile_size):
            for x in range(0, w - tile_size, tile_size):
                tile = screenshot[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((x, y))
        
        # Find repeating patterns
        unique_tiles = {}
        tile_counts = defaultdict(int)
        
        for i, tile in enumerate(tiles):
            tile_hash = self._compute_hash(tile)
            tile_counts[tile_hash] += 1
            
            if tile_hash not in unique_tiles:
                unique_tiles[tile_hash] = (tile, positions[i])
        
        # Extract tiles that repeat (likely background)
        for tile_hash, count in tile_counts.items():
            if count > 5:  # Appears at least 5 times
                tile, (x, y) = unique_tiles[tile_hash]
                
                asset = ExtractedAsset(
                    image=tile,
                    bbox=(x, y, tile_size, tile_size),
                    confidence=0.9,
                    asset_type='background',
                    metadata={
                        'tile_size': tile_size,
                        'repetitions': count
                    },
                    hash=tile_hash
                )
                assets.append(asset)
        
        return assets
    
    def extract_animated_sprites(self, frames: List[np.ndarray],
                                min_changes: int = 3) -> List[ExtractedAsset]:
        """Extract animated sprites from multiple frames."""
        if len(frames) < 2:
            return []
        
        assets = []
        
        # Compute frame differences
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            diffs.append(thresh)
        
        # Find regions that change consistently (animated sprites)
        accumulated_diff = np.zeros_like(diffs[0])
        for diff in diffs:
            accumulated_diff = cv2.add(accumulated_diff, diff)
        
        # Normalize
        accumulated_diff = (accumulated_diff / len(diffs)).astype(np.uint8)
        
        # Find contours of changing regions
        contours, _ = cv2.findContours(accumulated_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if region changes in enough frames
            change_count = 0
            for diff in diffs:
                region_diff = diff[y:y+h, x:x+w]
                if np.mean(region_diff) > 10:
                    change_count += 1
            
            if change_count >= min_changes:
                # Extract sprite frames
                sprite_frames = [frame[y:y+h, x:x+w] for frame in frames]
                
                # Use first frame as representative
                asset = ExtractedAsset(
                    image=sprite_frames[0],
                    bbox=(x, y, w, h),
                    confidence=0.85,
                    asset_type='animated_sprite',
                    metadata={
                        'frame_count': len(sprite_frames),
                        'change_count': change_count
                    },
                    hash=self._compute_hash(sprite_frames[0])
                )
                assets.append(asset)
        
        return assets
    
    def _is_valid_sprite(self, image: np.ndarray) -> bool:
        """Check if an image region is likely a valid sprite."""
        if image.size == 0:
            return False
        
        # Check for transparency or consistent background
        # Sprites often have transparent or solid color backgrounds
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # Valid sprites have some edges but not too many (not noise)
        return 0.05 < edge_ratio < 0.5
    
    def _is_text_region(self, image: np.ndarray) -> bool:
        """Check if region likely contains text."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Text has high contrast
        contrast = np.std(gray)
        if contrast < 30:
            return False
        
        # Text has horizontal features
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        horizontal_strength = np.mean(np.abs(sobel_x))
        vertical_strength = np.mean(np.abs(sobel_y))
        
        # Text usually has more horizontal than vertical edges
        return horizontal_strength > vertical_strength * 1.2
    
    def _compute_hash(self, image: np.ndarray) -> str:
        """Compute a hash for an image."""
        # Resize to standard size for consistent hashing
        standard = cv2.resize(image, (32, 32))
        return hashlib.md5(standard.tobytes()).hexdigest()
    
    def _deduplicate_assets(self, assets: List[ExtractedAsset]) -> List[ExtractedAsset]:
        """Remove duplicate assets based on hash."""
        unique = {}
        for asset in assets:
            if asset.hash not in unique:
                unique[asset.hash] = asset
            else:
                # Keep the one with higher confidence
                if asset.confidence > unique[asset.hash].confidence:
                    unique[asset.hash] = asset
        
        return list(unique.values())
    
    def save_assets(self, assets: List[ExtractedAsset], output_dir: str):
        """Save extracted assets to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Organize by type
        for asset in assets:
            type_dir = output_path / asset.asset_type
            type_dir.mkdir(exist_ok=True)
            
            # Generate filename
            filename = f"{asset.asset_type}_{asset.hash[:8]}.png"
            filepath = type_dir / filename
            
            asset.save(str(filepath))
            self.logger.info(f"Saved asset: {filepath}")
    
    def create_sprite_sheet(self, assets: List[ExtractedAsset],
                           output_path: str, 
                           columns: int = 10):
        """Create a sprite sheet from extracted assets."""
        if not assets:
            return
        
        # Find max dimensions
        max_w = max(asset.image.shape[1] for asset in assets)
        max_h = max(asset.image.shape[0] for asset in assets)
        
        # Calculate sheet dimensions
        rows = (len(assets) + columns - 1) // columns
        sheet_w = columns * max_w
        sheet_h = rows * max_h
        
        # Create sprite sheet
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
        
        # Place sprites
        for i, asset in enumerate(assets):
            row = i // columns
            col = i % columns
            
            y = row * max_h
            x = col * max_w
            
            h, w = asset.image.shape[:2]
            sheet[y:y+h, x:x+w] = asset.image
        
        cv2.imwrite(output_path, sheet)
        self.logger.info(f"Created sprite sheet: {output_path}")


class AssetDatabase:
    """Database for managing extracted assets."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_path / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load asset index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {'assets': {}, 'categories': defaultdict(list)}
    
    def add_asset(self, asset: ExtractedAsset, name: str):
        """Add asset to database."""
        # Save asset image
        asset_path = self.db_path / f"{asset.hash}.png"
        cv2.imwrite(str(asset_path), asset.image)
        
        # Update index
        self.index['assets'][asset.hash] = {
            'name': name,
            'type': asset.asset_type,
            'bbox': asset.bbox,
            'confidence': asset.confidence,
            'metadata': asset.metadata,
            'path': str(asset_path.relative_to(self.db_path))
        }
        
        self.index['categories'][asset.asset_type].append(asset.hash)
        self._save_index()
    
    def _save_index(self):
        """Save asset index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def find_similar(self, image: np.ndarray, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar assets in database."""
        similar = []
        
        for asset_hash, asset_info in self.index['assets'].items():
            asset_path = self.db_path / asset_info['path']
            if asset_path.exists():
                stored_image = cv2.imread(str(asset_path))
                
                # Compare using template matching
                result = cv2.matchTemplate(image, stored_image, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                
                if score > threshold:
                    similar.append({
                        'hash': asset_hash,
                        'similarity': score,
                        **asset_info
                    })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)