"""Specialized Game Frame Buffer System for Nexus Framework"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
from dataclasses import dataclass, field
import time
import threading
import structlog
from enum import Enum
import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor

from nexus.vision.frame_processing import GameFrame

logger = structlog.get_logger()


class BufferMode(Enum):
    """Frame buffer operation modes"""
    RING = "ring"              # Circular buffer, overwrites old frames
    QUEUE = "queue"            # FIFO queue with max size
    PRIORITY = "priority"      # Priority-based buffer
    TEMPORAL = "temporal"      # Time-based buffer with expiration
    ADAPTIVE = "adaptive"      # Adaptive size based on memory


@dataclass
class BufferedFrame:
    """Frame with buffer metadata"""
    frame: GameFrame
    buffer_timestamp: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    
    @property
    def age(self) -> float:
        """Age of frame in buffer (seconds)"""
        return time.time() - self.buffer_timestamp
    
    @property
    def size_bytes(self) -> int:
        """Estimated size in bytes"""
        return self.frame.frame_data.nbytes


class GameFrameBuffer:
    """Specialized frame buffer for game automation"""
    
    def __init__(self, 
                 max_frames: int = 30,
                 mode: BufferMode = BufferMode.RING,
                 max_memory_mb: float = 512):
        """
        Initialize game frame buffer
        
        Args:
            max_frames: Maximum number of frames to buffer
            mode: Buffer operation mode
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_frames = max_frames
        self.mode = mode
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Buffer storage
        self.frames: Deque[BufferedFrame] = deque(maxlen=max_frames if mode == BufferMode.RING else None)
        self.frame_index: Dict[str, BufferedFrame] = {}  # Fast lookup by ID
        
        # Threading
        self.lock = threading.RLock()
        self.frame_available = threading.Event()
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "total_frames_buffered": 0,
            "frames_dropped": 0,
            "frames_retrieved": 0,
            "current_memory_bytes": 0,
            "peak_memory_bytes": 0,
            "average_frame_age": 0.0,
            "buffer_utilization": 0.0
        }
        
        # Processing pipeline
        self.preprocessors: List[callable] = []
        self.postprocessors: List[callable] = []
        
        # Async support
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Initialized GameFrameBuffer: mode={mode.value}, max_frames={max_frames}")
    
    def add_frame(self, frame: GameFrame, priority: int = 0, 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add frame to buffer
        
        Args:
            frame: Game frame to add
            priority: Frame priority (higher = more important)
            metadata: Additional metadata
        
        Returns:
            True if frame was added
        """
        with self.lock:
            # Check memory limit
            if self._check_memory_limit(frame):
                # Apply preprocessors
                processed_frame = self._preprocess_frame(frame)
                
                # Create buffered frame
                buffered = BufferedFrame(
                    frame=processed_frame,
                    buffer_timestamp=time.time(),
                    priority=priority,
                    metadata=metadata or {}
                )
                
                # Add based on mode
                if self.mode == BufferMode.RING:
                    self._add_ring_mode(buffered)
                elif self.mode == BufferMode.QUEUE:
                    self._add_queue_mode(buffered)
                elif self.mode == BufferMode.PRIORITY:
                    self._add_priority_mode(buffered)
                elif self.mode == BufferMode.TEMPORAL:
                    self._add_temporal_mode(buffered)
                elif self.mode == BufferMode.ADAPTIVE:
                    self._add_adaptive_mode(buffered)
                
                # Update statistics
                self.stats["total_frames_buffered"] += 1
                self._update_memory_stats()
                
                # Signal frame available
                self.frame_available.set()
                
                return True
            else:
                self.stats["frames_dropped"] += 1
                logger.warning("Frame dropped due to memory limit")
                return False
    
    def get_frame(self, timeout: Optional[float] = None) -> Optional[GameFrame]:
        """
        Get next frame from buffer
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            Game frame or None if buffer is empty
        """
        if self.frame_available.wait(timeout):
            with self.lock:
                if self.frames:
                    buffered = self.frames.popleft()
                    
                    # Remove from index
                    frame_id = str(id(buffered.frame))
                    if frame_id in self.frame_index:
                        del self.frame_index[frame_id]
                    
                    # Apply postprocessors
                    processed_frame = self._postprocess_frame(buffered.frame)
                    
                    # Update statistics
                    self.stats["frames_retrieved"] += 1
                    self._update_memory_stats()
                    
                    # Clear event if buffer is empty
                    if not self.frames:
                        self.frame_available.clear()
                    
                    return processed_frame
        
        return None
    
    def get_latest_frame(self) -> Optional[GameFrame]:
        """Get most recent frame without removing it"""
        with self.lock:
            if self.frames:
                return self.frames[-1].frame
        return None
    
    def get_frames(self, n: int) -> List[GameFrame]:
        """
        Get up to n frames from buffer
        
        Args:
            n: Number of frames to get
        
        Returns:
            List of frames
        """
        frames = []
        
        for _ in range(n):
            frame = self.get_frame(timeout=0)
            if frame:
                frames.append(frame)
            else:
                break
        
        return frames
    
    def get_frame_sequence(self, n: int) -> List[GameFrame]:
        """
        Get sequence of n most recent frames without removing
        
        Args:
            n: Number of frames
        
        Returns:
            List of frames in temporal order
        """
        with self.lock:
            n_frames = min(n, len(self.frames))
            return [bf.frame for bf in list(self.frames)[-n_frames:]]
    
    def peek_frames(self, n: Optional[int] = None) -> List[GameFrame]:
        """
        Peek at frames without removing them
        
        Args:
            n: Number of frames to peek (None for all)
        
        Returns:
            List of frames
        """
        with self.lock:
            if n is None:
                return [bf.frame for bf in self.frames]
            else:
                return [bf.frame for bf in list(self.frames)[:n]]
    
    async def add_frame_async(self, frame: GameFrame, priority: int = 0,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of add_frame"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.add_frame,
            frame,
            priority,
            metadata
        )
    
    async def get_frame_async(self, timeout: Optional[float] = None) -> Optional[GameFrame]:
        """Async version of get_frame"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.get_frame,
            timeout
        )
    
    def _add_ring_mode(self, buffered: BufferedFrame):
        """Add frame in ring buffer mode"""
        self.frames.append(buffered)
        frame_id = str(id(buffered.frame))
        self.frame_index[frame_id] = buffered
    
    def _add_queue_mode(self, buffered: BufferedFrame):
        """Add frame in queue mode"""
        if len(self.frames) >= self.max_frames:
            # Remove oldest frame
            old = self.frames.popleft()
            old_id = str(id(old.frame))
            if old_id in self.frame_index:
                del self.frame_index[old_id]
            self.stats["frames_dropped"] += 1
        
        self.frames.append(buffered)
        frame_id = str(id(buffered.frame))
        self.frame_index[frame_id] = buffered
    
    def _add_priority_mode(self, buffered: BufferedFrame):
        """Add frame in priority mode"""
        # Insert in priority order
        inserted = False
        
        for i, existing in enumerate(self.frames):
            if buffered.priority > existing.priority:
                self.frames.insert(i, buffered)
                inserted = True
                break
        
        if not inserted:
            self.frames.append(buffered)
        
        # Maintain max size
        while len(self.frames) > self.max_frames:
            # Remove lowest priority frame
            removed = self.frames.pop()
            removed_id = str(id(removed.frame))
            if removed_id in self.frame_index:
                del self.frame_index[removed_id]
            self.stats["frames_dropped"] += 1
        
        frame_id = str(id(buffered.frame))
        self.frame_index[frame_id] = buffered
    
    def _add_temporal_mode(self, buffered: BufferedFrame):
        """Add frame in temporal mode with expiration"""
        # Remove expired frames (older than 30 seconds)
        max_age = 30.0
        
        while self.frames and self.frames[0].age > max_age:
            expired = self.frames.popleft()
            expired_id = str(id(expired.frame))
            if expired_id in self.frame_index:
                del self.frame_index[expired_id]
            self.stats["frames_dropped"] += 1
        
        # Add new frame
        self.frames.append(buffered)
        frame_id = str(id(buffered.frame))
        self.frame_index[frame_id] = buffered
        
        # Maintain max size
        while len(self.frames) > self.max_frames:
            old = self.frames.popleft()
            old_id = str(id(old.frame))
            if old_id in self.frame_index:
                del self.frame_index[old_id]
            self.stats["frames_dropped"] += 1
    
    def _add_adaptive_mode(self, buffered: BufferedFrame):
        """Add frame in adaptive mode based on memory"""
        # Dynamically adjust buffer size based on memory usage
        current_memory = self._calculate_memory_usage()
        
        if current_memory > self.max_memory_bytes * 0.9:  # 90% threshold
            # Reduce buffer size
            while self.frames and current_memory > self.max_memory_bytes * 0.8:
                old = self.frames.popleft()
                old_id = str(id(old.frame))
                if old_id in self.frame_index:
                    del self.frame_index[old_id]
                current_memory -= old.size_bytes
                self.stats["frames_dropped"] += 1
        
        self.frames.append(buffered)
        frame_id = str(id(buffered.frame))
        self.frame_index[frame_id] = buffered
    
    def _check_memory_limit(self, frame: GameFrame) -> bool:
        """Check if adding frame would exceed memory limit"""
        frame_size = frame.frame_data.nbytes
        current_memory = self._calculate_memory_usage()
        
        return (current_memory + frame_size) <= self.max_memory_bytes
    
    def _calculate_memory_usage(self) -> int:
        """Calculate current memory usage in bytes"""
        total = sum(bf.size_bytes for bf in self.frames)
        return total
    
    def _update_memory_stats(self):
        """Update memory statistics"""
        current = self._calculate_memory_usage()
        self.stats["current_memory_bytes"] = current
        
        if current > self.stats["peak_memory_bytes"]:
            self.stats["peak_memory_bytes"] = current
        
        self.stats["buffer_utilization"] = len(self.frames) / max(1, self.max_frames)
        
        if self.frames:
            avg_age = sum(bf.age for bf in self.frames) / len(self.frames)
            self.stats["average_frame_age"] = avg_age
    
    def _preprocess_frame(self, frame: GameFrame) -> GameFrame:
        """Apply preprocessing to frame"""
        processed = frame
        
        for preprocessor in self.preprocessors:
            try:
                processed = preprocessor(processed)
            except Exception as e:
                logger.error(f"Preprocessor error: {e}")
        
        return processed
    
    def _postprocess_frame(self, frame: GameFrame) -> GameFrame:
        """Apply postprocessing to frame"""
        processed = frame
        
        for postprocessor in self.postprocessors:
            try:
                processed = postprocessor(processed)
            except Exception as e:
                logger.error(f"Postprocessor error: {e}")
        
        return processed
    
    def add_preprocessor(self, func: callable):
        """Add frame preprocessor function"""
        self.preprocessors.append(func)
    
    def add_postprocessor(self, func: callable):
        """Add frame postprocessor function"""
        self.postprocessors.append(func)
    
    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            self.frames.clear()
            self.frame_index.clear()
            self.frame_available.clear()
            self._update_memory_stats()
            logger.info("Frame buffer cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            stats = self.stats.copy()
            stats["current_frames"] = len(self.frames)
            stats["memory_usage_mb"] = stats["current_memory_bytes"] / (1024 * 1024)
            stats["peak_memory_mb"] = stats["peak_memory_bytes"] / (1024 * 1024)
            return stats
    
    def save_buffer_sequence(self, output_path: str, format: str = "video"):
        """
        Save buffered frames as video or image sequence
        
        Args:
            output_path: Output file path
            format: "video" or "images"
        """
        with self.lock:
            frames = [bf.frame for bf in self.frames]
        
        if not frames:
            logger.warning("No frames to save")
            return
        
        if format == "video":
            self._save_as_video(frames, output_path)
        elif format == "images":
            self._save_as_images(frames, output_path)
    
    def _save_as_video(self, frames: List[GameFrame], output_path: str):
        """Save frames as video"""
        if not frames:
            return
        
        # Get frame dimensions
        h, w = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
        
        for frame in frames:
            out.write(frame.frame_data)
        
        out.release()
        logger.info(f"Saved {len(frames)} frames to video: {output_path}")
    
    def _save_as_images(self, frames: List[GameFrame], output_dir: str):
        """Save frames as image sequence"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            image_path = output_path / f"frame_{i:06d}.png"
            cv2.imwrite(str(image_path), frame.frame_data)
        
        logger.info(f"Saved {len(frames)} frames to: {output_dir}")
    
    def create_motion_summary(self) -> Optional[np.ndarray]:
        """
        Create motion summary image from buffered frames
        
        Returns:
            Motion summary image
        """
        with self.lock:
            if len(self.frames) < 2:
                return None
            
            frames = [bf.frame for bf in self.frames]
        
        # Calculate motion between consecutive frames
        motion_frames = []
        
        for i in range(1, len(frames)):
            prev = frames[i-1].grayscale
            curr = frames[i].grayscale
            
            # Calculate difference
            diff = cv2.absdiff(prev, curr)
            
            # Threshold to get motion mask
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_frames.append(motion)
        
        if not motion_frames:
            return None
        
        # Combine motion frames
        motion_summary = np.zeros_like(motion_frames[0])
        
        for motion in motion_frames:
            motion_summary = cv2.bitwise_or(motion_summary, motion)
        
        # Apply color map for visualization
        motion_colored = cv2.applyColorMap(motion_summary, cv2.COLORMAP_JET)
        
        return motion_colored
    
    def find_stable_frame(self, stability_threshold: float = 0.95) -> Optional[GameFrame]:
        """
        Find most stable (least motion) frame in buffer
        
        Args:
            stability_threshold: Minimum stability score
        
        Returns:
            Most stable frame or None
        """
        with self.lock:
            if len(self.frames) < 2:
                return self.frames[0].frame if self.frames else None
            
            best_frame = None
            best_stability = 0.0
            
            for i in range(1, len(self.frames)):
                prev = self.frames[i-1].frame
                curr = self.frames[i].frame
                
                # Calculate stability using SSIM
                stability = prev.compare_ssim(curr)
                
                if stability > best_stability:
                    best_stability = stability
                    best_frame = curr
                
                if best_stability >= stability_threshold:
                    break
            
            return best_frame
    
    def shutdown(self):
        """Shutdown buffer and clean up resources"""
        self.stop_event.set()
        self.frame_available.set()  # Wake up any waiting threads
        self.clear()
        self.executor.shutdown(wait=True)
        logger.info("GameFrameBuffer shutdown complete")