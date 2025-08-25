"""
Target tracking and prediction module for Auto-Aim Assistant.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time
import math
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


@dataclass
class TrackedTarget:
    """Represents a tracked target with history and predictions."""
    id: int
    current_bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    target_type: str = "enemy"
    
    # Tracking history
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    size_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Predictions
    predicted_position: Optional[Tuple[float, float]] = None
    predicted_velocity: Optional[Tuple[float, float]] = None
    
    # Tracking metrics
    age: int = 0  # Frames since first detection
    time_since_update: int = 0  # Frames since last update
    hits: int = 0  # Total number of updates
    hit_streak: int = 0  # Consecutive frames with update
    
    # Kalman filter for smooth tracking
    kalman_filter: Optional[Any] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of target."""
        x, y, w, h = self.current_bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def head_position(self) -> Tuple[int, int]:
        """Estimate head position (top 20% of bbox)."""
        x, y, w, h = self.current_bbox
        return (x + w // 2, y + int(h * 0.2))
    
    @property
    def is_confirmed(self) -> bool:
        """Check if target is confirmed (stable tracking)."""
        return self.hits >= 3 and self.hit_streak >= 2
    
    @property
    def is_lost(self) -> bool:
        """Check if target is lost."""
        return self.time_since_update > 5


class TargetTracker:
    """Tracks targets across frames using Kalman filtering and Hungarian algorithm."""
    
    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize target tracker.
        
        Args:
            max_age: Maximum frames to keep target without update
            min_hits: Minimum hits to confirm target
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[TrackedTarget]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y, width, height, confidence)
        
        Returns:
            List of confirmed tracked targets
        """
        self.frame_count += 1
        
        # Predict new positions for existing targets
        for target in self.tracked_targets.values():
            self._predict_target(target)
        
        # Match detections to existing targets
        matched, unmatched_dets, unmatched_trks = self._match_detections(
            detections, list(self.tracked_targets.values())
        )
        
        # Update matched targets
        for det_idx, trk_id in matched:
            detection = detections[det_idx]
            target = self.tracked_targets[trk_id]
            self._update_target(target, detection)
        
        # Create new targets for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            self._create_target(detection)
        
        # Handle unmatched targets
        for trk_id in unmatched_trks:
            target = self.tracked_targets[trk_id]
            target.time_since_update += 1
            target.hit_streak = 0
        
        # Remove lost targets
        self._remove_lost_targets()
        
        # Return confirmed targets
        return [t for t in self.tracked_targets.values() if t.is_confirmed]
    
    def _predict_target(self, target: TrackedTarget):
        """Predict target's next position using Kalman filter."""
        if target.kalman_filter is None:
            self._init_kalman_filter(target)
        
        # Predict next state
        target.kalman_filter.predict()
        predicted_state = target.kalman_filter.x
        
        # Extract predicted position
        target.predicted_position = (
            predicted_state[0, 0],  # x
            predicted_state[2, 0]   # y
        )
        
        # Extract predicted velocity
        target.predicted_velocity = (
            predicted_state[1, 0],  # vx
            predicted_state[3, 0]   # vy
        )
    
    def _update_target(self, target: TrackedTarget, detection: Tuple):
        """Update target with new detection."""
        x, y, w, h, conf = detection
        
        # Update Kalman filter
        if target.kalman_filter:
            measurement = np.array([[x + w/2], [y + h/2]])
            target.kalman_filter.update(measurement)
        
        # Update target properties
        target.current_bbox = (x, y, w, h)
        target.confidence = conf
        
        # Update history
        target.position_history.append((x + w/2, y + h/2))
        target.size_history.append((w, h))
        
        # Calculate velocity
        if len(target.position_history) >= 2:
            prev_pos = target.position_history[-2]
            curr_pos = target.position_history[-1]
            velocity = (
                curr_pos[0] - prev_pos[0],
                curr_pos[1] - prev_pos[1]
            )
            target.velocity_history.append(velocity)
        
        # Update tracking metrics
        target.time_since_update = 0
        target.hits += 1
        target.hit_streak += 1
        target.age += 1
    
    def _create_target(self, detection: Tuple) -> TrackedTarget:
        """Create new tracked target."""
        x, y, w, h, conf = detection
        
        target = TrackedTarget(
            id=self.next_id,
            current_bbox=(x, y, w, h),
            confidence=conf
        )
        
        # Initialize history
        target.position_history.append((x + w/2, y + h/2))
        target.size_history.append((w, h))
        
        # Initialize Kalman filter
        self._init_kalman_filter(target)
        
        # Add to tracked targets
        self.tracked_targets[self.next_id] = target
        self.next_id += 1
        
        return target
    
    def _init_kalman_filter(self, target: TrackedTarget):
        """Initialize Kalman filter for target."""
        kf = KalmanFilter(dim_x=6, dim_z=2)
        
        # State: [x, vx, y, vy, w, h]
        # Measurement: [x, y]
        
        # State transition matrix
        dt = 1.0  # Assume 1 frame time unit
        kf.F = np.array([
            [1, dt, 0, 0,  0, 0],
            [0, 1,  0, 0,  0, 0],
            [0, 0,  1, dt, 0, 0],
            [0, 0,  0, 1,  0, 0],
            [0, 0,  0, 0,  1, 0],
            [0, 0,  0, 0,  0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise
        kf.Q *= 0.01
        
        # Measurement noise
        kf.R *= 10
        
        # Initial state
        x, y, w, h = target.current_bbox
        kf.x = np.array([[x + w/2], [0], [y + h/2], [0], [w], [h]])
        
        # Initial covariance
        kf.P *= 100
        
        target.kalman_filter = kf
    
    def _match_detections(self, detections: List, targets: List) -> Tuple:
        """Match detections to existing targets using Hungarian algorithm."""
        if not detections or not targets:
            return [], list(range(len(detections))), [t.id for t in targets]
        
        # Calculate cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(detections), len(targets)))
        
        for i, det in enumerate(detections):
            for j, trk in enumerate(targets):
                cost_matrix[i, j] = 1 - self._calculate_iou(
                    det[:4], trk.current_bbox
                )
        
        # Solve assignment problem
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matched = []
        for det_idx, trk_idx in zip(det_indices, trk_indices):
            if cost_matrix[det_idx, trk_idx] < (1 - self.iou_threshold):
                matched.append((det_idx, targets[trk_idx].id))
        
        # Find unmatched
        matched_det_indices = {m[0] for m in matched}
        matched_trk_ids = {m[1] for m in matched}
        
        unmatched_dets = [i for i in range(len(detections)) 
                         if i not in matched_det_indices]
        unmatched_trks = [t.id for t in targets 
                         if t.id not in matched_trk_ids]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_lost_targets(self):
        """Remove targets that have been lost."""
        ids_to_remove = []
        
        for target_id, target in self.tracked_targets.items():
            if target.time_since_update > self.max_age:
                ids_to_remove.append(target_id)
        
        for target_id in ids_to_remove:
            del self.tracked_targets[target_id]
    
    def get_target_by_id(self, target_id: int) -> Optional[TrackedTarget]:
        """Get target by ID."""
        return self.tracked_targets.get(target_id)
    
    def get_confirmed_targets(self) -> List[TrackedTarget]:
        """Get all confirmed targets."""
        return [t for t in self.tracked_targets.values() if t.is_confirmed]
    
    def predict_position(self, target: TrackedTarget, time_ahead: float) -> Tuple[float, float]:
        """Predict target position at future time."""
        if not target.velocity_history:
            return target.center
        
        # Average velocity
        avg_velocity = np.mean(target.velocity_history, axis=0)
        
        # Current position
        current_pos = target.center
        
        # Predicted position
        predicted_x = current_pos[0] + avg_velocity[0] * time_ahead * 60  # Assume 60 FPS
        predicted_y = current_pos[1] + avg_velocity[1] * time_ahead * 60
        
        return (predicted_x, predicted_y)
    
    def calculate_lead_angle(self, target: TrackedTarget, 
                            projectile_speed: float,
                            origin: Tuple[int, int]) -> float:
        """Calculate lead angle for moving target."""
        if not target.velocity_history:
            return 0.0
        
        # Target position and velocity
        target_pos = np.array(target.center)
        target_vel = np.mean(target.velocity_history, axis=0)
        
        # Calculate distance
        distance = np.linalg.norm(target_pos - origin)
        
        # Time for projectile to reach target
        travel_time = distance / projectile_speed
        
        # Predicted position
        predicted_pos = target_pos + target_vel * travel_time * 60
        
        # Calculate angle
        dx = predicted_pos[0] - origin[0]
        dy = predicted_pos[1] - origin[1]
        
        return math.atan2(dy, dx)
    
    def reset(self):
        """Reset tracker."""
        self.tracked_targets.clear()
        self.next_id = 0
        self.frame_count = 0