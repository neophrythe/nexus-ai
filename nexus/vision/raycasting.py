"""Raycasting utilities for game vision and collision detection"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
import structlog

logger = structlog.get_logger()


class RayMode(Enum):
    """Ray generation modes"""
    UNIFORM = "uniform"          # Evenly spaced rays
    ADAPTIVE = "adaptive"        # More rays in areas of interest
    CONE = "cone"               # Cone-shaped ray spread
    RANDOM = "random"           # Random ray distribution
    GRID = "grid"               # Grid-based rays


@dataclass
class Ray:
    """Single ray definition"""
    origin: Tuple[int, int]
    angle: float
    max_distance: float
    label: str
    
    @property
    def direction(self) -> Tuple[float, float]:
        """Get normalized direction vector"""
        rad = math.radians(self.angle)
        return (math.cos(rad), math.sin(rad))
    
    @property
    def end_point(self) -> Tuple[int, int]:
        """Get ray end point at max distance"""
        dx, dy = self.direction
        return (
            int(self.origin[0] + dx * self.max_distance),
            int(self.origin[1] + dy * self.max_distance)
        )


@dataclass
class RaycastHit:
    """Result of a raycast"""
    ray: Ray
    hit: bool
    distance: float
    point: Optional[Tuple[int, int]]
    normal: Optional[Tuple[float, float]]
    color: Optional[Tuple[int, int, int]]
    object_id: Optional[int]


class Raycaster:
    """Advanced raycasting system for games"""
    
    def __init__(self):
        """Initialize raycaster"""
        self.angle_cache: Dict[float, np.ndarray] = {}
        self.distance_cache: Dict[Tuple[int, int], np.ndarray] = {}
    
    def generate_rays(self, 
                     origin: Tuple[int, int],
                     base_angle: float = 0,
                     mode: RayMode = RayMode.UNIFORM,
                     quantity: int = 12,
                     spread_angle: float = 360,
                     max_distance: float = 1000) -> List[Ray]:
        """
        Generate rays from origin point
        
        Args:
            origin: Starting point (x, y)
            base_angle: Base angle in degrees
            mode: Ray generation mode
            quantity: Number of rays to generate
            spread_angle: Total angle spread for rays
            max_distance: Maximum ray distance
        
        Returns:
            List of rays
        """
        rays = []
        
        if mode == RayMode.UNIFORM:
            rays = self._generate_uniform_rays(
                origin, base_angle, quantity, spread_angle, max_distance
            )
        elif mode == RayMode.ADAPTIVE:
            rays = self._generate_adaptive_rays(
                origin, base_angle, quantity, spread_angle, max_distance
            )
        elif mode == RayMode.CONE:
            rays = self._generate_cone_rays(
                origin, base_angle, quantity, spread_angle, max_distance
            )
        elif mode == RayMode.RANDOM:
            rays = self._generate_random_rays(
                origin, base_angle, quantity, spread_angle, max_distance
            )
        elif mode == RayMode.GRID:
            rays = self._generate_grid_rays(
                origin, base_angle, quantity, max_distance
            )
        
        return rays
    
    def _generate_uniform_rays(self, origin: Tuple[int, int], base_angle: float,
                              quantity: int, spread_angle: float, 
                              max_distance: float) -> List[Ray]:
        """Generate uniformly distributed rays"""
        rays = []
        
        # Calculate angle step
        angle_step = spread_angle / max(1, quantity - 1) if quantity > 1 else 0
        start_angle = base_angle - spread_angle / 2
        
        for i in range(quantity):
            angle = start_angle + i * angle_step
            angle = angle % 360  # Normalize to 0-360
            
            ray = Ray(
                origin=origin,
                angle=angle,
                max_distance=max_distance,
                label=f"Ray_{i:03d}_angle_{angle:.1f}"
            )
            rays.append(ray)
        
        return rays
    
    def _generate_adaptive_rays(self, origin: Tuple[int, int], base_angle: float,
                               quantity: int, spread_angle: float,
                               max_distance: float) -> List[Ray]:
        """Generate adaptive rays with more density near center"""
        rays = []
        
        # More rays near the center angle
        for i in range(quantity):
            # Use exponential distribution for adaptive spacing
            t = i / max(1, quantity - 1)
            
            # Bias towards center
            if i < quantity // 2:
                t = t ** 2  # Quadratic distribution
            else:
                t = 1 - (1 - t) ** 2
            
            angle = base_angle - spread_angle/2 + t * spread_angle
            angle = angle % 360
            
            ray = Ray(
                origin=origin,
                angle=angle,
                max_distance=max_distance,
                label=f"AdaptiveRay_{i:03d}"
            )
            rays.append(ray)
        
        return rays
    
    def _generate_cone_rays(self, origin: Tuple[int, int], base_angle: float,
                           quantity: int, spread_angle: float,
                           max_distance: float) -> List[Ray]:
        """Generate cone-shaped ray distribution"""
        rays = []
        
        # Center ray
        rays.append(Ray(
            origin=origin,
            angle=base_angle,
            max_distance=max_distance,
            label="ConeCenter"
        ))
        
        # Side rays
        remaining = quantity - 1
        for i in range(remaining):
            side = 1 if i % 2 == 0 else -1
            offset = ((i // 2) + 1) * (spread_angle / (remaining // 2 + 1))
            angle = (base_angle + side * offset) % 360
            
            ray = Ray(
                origin=origin,
                angle=angle,
                max_distance=max_distance,
                label=f"ConeSide_{i:03d}"
            )
            rays.append(ray)
        
        return rays
    
    def _generate_random_rays(self, origin: Tuple[int, int], base_angle: float,
                             quantity: int, spread_angle: float,
                             max_distance: float) -> List[Ray]:
        """Generate randomly distributed rays"""
        rays = []
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(quantity):
            # Random angle within spread
            random_offset = np.random.uniform(-spread_angle/2, spread_angle/2)
            angle = (base_angle + random_offset) % 360
            
            ray = Ray(
                origin=origin,
                angle=angle,
                max_distance=max_distance,
                label=f"RandomRay_{i:03d}"
            )
            rays.append(ray)
        
        return rays
    
    def _generate_grid_rays(self, origin: Tuple[int, int], base_angle: float,
                          quantity: int, max_distance: float) -> List[Ray]:
        """Generate grid-aligned rays (8 directions)"""
        rays = []
        
        # 8 cardinal and diagonal directions
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for i, angle in enumerate(directions[:min(quantity, 8)]):
            adjusted_angle = (base_angle + angle) % 360
            
            ray = Ray(
                origin=origin,
                angle=adjusted_angle,
                max_distance=max_distance,
                label=f"GridRay_{angle}"
            )
            rays.append(ray)
        
        return rays
    
    def cast_ray(self, ray: Ray, collision_mask: np.ndarray,
                frame: Optional[np.ndarray] = None) -> RaycastHit:
        """
        Cast a single ray and check for collision
        
        Args:
            ray: Ray to cast
            collision_mask: Binary mask where 1 = collision
            frame: Optional color frame for additional info
        
        Returns:
            RaycastHit with collision information
        """
        # Use Bresenham's line algorithm for ray marching
        points = self._bresenham_line(ray.origin, ray.end_point)
        
        for i, point in enumerate(points):
            x, y = point
            
            # Check bounds
            if (x < 0 or x >= collision_mask.shape[1] or 
                y < 0 or y >= collision_mask.shape[0]):
                break
            
            # Check collision
            if collision_mask[y, x] > 0:
                distance = np.sqrt((x - ray.origin[0])**2 + (y - ray.origin[1])**2)
                
                # Calculate normal (simplified)
                normal = self._calculate_normal(collision_mask, x, y)
                
                # Get color if frame provided
                color = None
                if frame is not None:
                    color = tuple(frame[y, x].tolist())
                
                return RaycastHit(
                    ray=ray,
                    hit=True,
                    distance=distance,
                    point=(x, y),
                    normal=normal,
                    color=color,
                    object_id=int(collision_mask[y, x])
                )
        
        # No collision
        return RaycastHit(
            ray=ray,
            hit=False,
            distance=ray.max_distance,
            point=None,
            normal=None,
            color=None,
            object_id=None
        )
    
    def cast_rays(self, rays: List[Ray], collision_mask: np.ndarray,
                 frame: Optional[np.ndarray] = None,
                 parallel: bool = True) -> List[RaycastHit]:
        """
        Cast multiple rays
        
        Args:
            rays: List of rays to cast
            collision_mask: Binary collision mask
            frame: Optional color frame
            parallel: Use parallel processing
        
        Returns:
            List of raycast hits
        """
        hits = []
        
        if parallel and len(rays) > 10:
            # Use vectorized operations for efficiency
            hits = self._cast_rays_vectorized(rays, collision_mask, frame)
        else:
            # Sequential casting
            for ray in rays:
                hit = self.cast_ray(ray, collision_mask, frame)
                hits.append(hit)
        
        return hits
    
    def _cast_rays_vectorized(self, rays: List[Ray], collision_mask: np.ndarray,
                             frame: Optional[np.ndarray] = None) -> List[RaycastHit]:
        """Vectorized ray casting for performance"""
        hits = []
        
        # Pre-compute angle and distance arrays if not cached
        h, w = collision_mask.shape
        
        for ray in rays:
            # Fast ray marching using pre-computed arrays
            hit = self._fast_ray_march(ray, collision_mask, frame)
            hits.append(hit)
        
        return hits
    
    def _fast_ray_march(self, ray: Ray, collision_mask: np.ndarray,
                       frame: Optional[np.ndarray] = None) -> RaycastHit:
        """Fast ray marching using DDA algorithm"""
        x0, y0 = ray.origin
        x1, y1 = ray.end_point
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x = x0
        y = y0
        
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        
        if dx > dy:
            error = dx / 2
            while x != x1:
                if 0 <= x < collision_mask.shape[1] and 0 <= y < collision_mask.shape[0]:
                    if collision_mask[y, x] > 0:
                        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
                        color = tuple(frame[y, x].tolist()) if frame is not None else None
                        
                        return RaycastHit(
                            ray=ray,
                            hit=True,
                            distance=distance,
                            point=(x, y),
                            normal=self._calculate_normal(collision_mask, x, y),
                            color=color,
                            object_id=int(collision_mask[y, x])
                        )
                
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2
            while y != y1:
                if 0 <= x < collision_mask.shape[1] and 0 <= y < collision_mask.shape[0]:
                    if collision_mask[y, x] > 0:
                        distance = np.sqrt((x - x0)**2 + (y - y0)**2)
                        color = tuple(frame[y, x].tolist()) if frame is not None else None
                        
                        return RaycastHit(
                            ray=ray,
                            hit=True,
                            distance=distance,
                            point=(x, y),
                            normal=self._calculate_normal(collision_mask, x, y),
                            color=color,
                            object_id=int(collision_mask[y, x])
                        )
                
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
        
        return RaycastHit(
            ray=ray,
            hit=False,
            distance=ray.max_distance,
            point=None,
            normal=None,
            color=None,
            object_id=None
        )
    
    def _bresenham_line(self, start: Tuple[int, int], 
                       end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for pixel-perfect lines"""
        points = []
        
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x0 += sx
            
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def _calculate_normal(self, mask: np.ndarray, x: int, y: int) -> Tuple[float, float]:
        """Calculate surface normal at collision point"""
        # Simple gradient-based normal calculation
        dx = 0
        dy = 0
        
        if x > 0 and x < mask.shape[1] - 1:
            dx = float(mask[y, x + 1]) - float(mask[y, x - 1])
        
        if y > 0 and y < mask.shape[0] - 1:
            dy = float(mask[y + 1, x]) - float(mask[y - 1, x])
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            return (-dx / length, -dy / length)
        
        return (0, 0)
    
    def create_collision_mask(self, frame: np.ndarray,
                            color_ranges: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Create collision mask from color ranges
        
        Args:
            frame: Input frame
            color_ranges: List of (lower, upper) color bounds
        
        Returns:
            Binary collision mask
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for i, (lower, upper) in enumerate(color_ranges, 1):
            color_mask = cv2.inRange(frame, lower, upper)
            mask[color_mask > 0] = i  # Use index as object ID
        
        return mask
    
    def create_distance_field(self, origin: Tuple[int, int],
                            shape: Tuple[int, int]) -> np.ndarray:
        """Create distance field from origin point"""
        cache_key = (origin, shape)
        
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        h, w = shape
        y_coords, x_coords = np.ogrid[:h, :w]
        
        distance_field = np.sqrt(
            (x_coords - origin[0])**2 + (y_coords - origin[1])**2
        )
        
        self.distance_cache[cache_key] = distance_field
        return distance_field
    
    def create_angle_field(self, origin: Tuple[int, int],
                         shape: Tuple[int, int]) -> np.ndarray:
        """Create angle field from origin point"""
        h, w = shape
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Calculate angles
        dx = x_coords - origin[0]
        dy = y_coords - origin[1]
        
        angle_field = np.degrees(np.arctan2(dy, dx))
        angle_field = (angle_field + 360) % 360  # Normalize to 0-360
        
        return angle_field
    
    def find_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int],
                          collision_mask: np.ndarray) -> bool:
        """
        Check if there's line of sight between two points
        
        Args:
            start: Starting point
            end: Ending point
            collision_mask: Collision mask
        
        Returns:
            True if line of sight exists
        """
        points = self._bresenham_line(start, end)
        
        for x, y in points:
            if (x < 0 or x >= collision_mask.shape[1] or 
                y < 0 or y >= collision_mask.shape[0]):
                return False
            
            if collision_mask[y, x] > 0:
                return False
        
        return True
    
    def calculate_visibility_polygon(self, origin: Tuple[int, int],
                                   collision_mask: np.ndarray,
                                   max_distance: float = 500,
                                   num_rays: int = 360) -> np.ndarray:
        """
        Calculate visibility polygon from a point
        
        Args:
            origin: View point
            collision_mask: Collision mask
            max_distance: Maximum view distance
            num_rays: Number of rays for accuracy
        
        Returns:
            Visibility mask
        """
        visibility_mask = np.zeros_like(collision_mask)
        
        # Cast rays in all directions
        rays = self.generate_rays(
            origin, 0, RayMode.UNIFORM, num_rays, 360, max_distance
        )
        
        # Find visibility boundaries
        boundary_points = []
        
        for ray in rays:
            hit = self.cast_ray(ray, collision_mask)
            
            if hit.hit and hit.point:
                boundary_points.append(hit.point)
            else:
                boundary_points.append(ray.end_point)
        
        # Fill visibility polygon
        if boundary_points:
            points = np.array(boundary_points, dtype=np.int32)
            cv2.fillPoly(visibility_mask, [points], 255)
        
        return visibility_mask
    
    def shadow_casting(self, light_source: Tuple[int, int],
                      obstacles_mask: np.ndarray,
                      frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Calculate shadows from a light source
        
        Args:
            light_source: Light position
            obstacles_mask: Mask of obstacles
            frame_shape: Shape of output frame
        
        Returns:
            Shadow mask (255 = lit, 0 = shadow)
        """
        shadow_mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
        
        # Find obstacle edges
        edges = cv2.Canny(obstacles_mask.astype(np.uint8), 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        
        for point in edge_points:
            y, x = point
            
            # Cast ray from light to edge and beyond
            dx = x - light_source[0]
            dy = y - light_source[1]
            
            # Extend ray
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx = dx / length * 1000  # Extend to edge of frame
                dy = dy / length * 1000
                
                end_point = (
                    int(x + dx),
                    int(y + dy)
                )
                
                # Draw shadow line
                shadow_points = self._bresenham_line((x, y), end_point)
                
                for sx, sy in shadow_points:
                    if 0 <= sx < frame_shape[1] and 0 <= sy < frame_shape[0]:
                        shadow_mask[sy, sx] = 0
        
        return shadow_mask
    
    def visualize_rays(self, frame: np.ndarray, rays: List[Ray],
                      hits: Optional[List[RaycastHit]] = None,
                      ray_color: Tuple[int, int, int] = (0, 255, 0),
                      hit_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Visualize rays on frame
        
        Args:
            frame: Input frame
            rays: List of rays
            hits: Optional raycast hits
            ray_color: Color for rays
            hit_color: Color for hit points
        
        Returns:
            Frame with visualized rays
        """
        result = frame.copy()
        
        for i, ray in enumerate(rays):
            # Draw ray
            end_point = ray.end_point
            
            if hits and i < len(hits):
                hit = hits[i]
                if hit.hit and hit.point:
                    end_point = hit.point
                    # Draw hit point
                    cv2.circle(result, hit.point, 5, hit_color, -1)
            
            # Draw ray line
            cv2.line(result, ray.origin, end_point, ray_color, 1)
        
        # Draw origin
        if rays:
            cv2.circle(result, rays[0].origin, 8, (0, 0, 255), -1)
        
        return result
    
    def get_collision_distances(self, rays: List[Ray],
                               collision_mask: np.ndarray) -> Dict[str, float]:
        """
        Get collision distances for all rays
        
        Args:
            rays: List of rays
            collision_mask: Collision mask
        
        Returns:
            Dictionary of ray labels to distances
        """
        distances = {}
        
        for ray in rays:
            hit = self.cast_ray(ray, collision_mask)
            distances[ray.label] = hit.distance
        
        return distances