"""
Video Logger for TensorBoard

Logs gameplay videos and training visualizations to TensorBoard.
"""

import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class VideoFrame:
    """Single video frame with metadata."""
    image: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = None


class VideoLogger:
    """
    Video logging for TensorBoard with frame buffering and encoding.
    
    Features:
    - Real-time video capture from gameplay
    - Frame buffering for smooth recording
    - Video compilation and encoding
    - Overlay support (scores, actions, etc.)
    - Multiple video streams
    """
    
    def __init__(self,
                 fps: int = 30,
                 max_buffer_size: int = 300,
                 video_codec: str = 'mp4v',
                 quality: int = 95):
        """
        Initialize video logger.
        
        Args:
            fps: Frames per second for output video
            max_buffer_size: Maximum frames to buffer
            video_codec: Video codec for encoding
            quality: Video quality (0-100)
        """
        self.fps = fps
        self.max_buffer_size = max_buffer_size
        self.video_codec = video_codec
        self.quality = quality
        
        # Frame buffers for different streams
        self.buffers: Dict[str, queue.Queue] = {}
        self.recording_threads: Dict[str, threading.Thread] = {}
        self.recording_flags: Dict[str, bool] = {}
        
        # Video writers
        self.writers: Dict[str, cv2.VideoWriter] = {}
        
        # Frame processors
        self.processors: Dict[str, callable] = {}
        
        # Statistics
        self.frame_counts: Dict[str, int] = {}
        self.dropped_frames: Dict[str, int] = {}
        
        logger.info(f"Video logger initialized: {fps} FPS, codec={video_codec}")
    
    def start_recording(self, stream_name: str = "gameplay",
                       output_path: Optional[str] = None,
                       frame_processor: Optional[callable] = None):
        """
        Start recording a video stream.
        
        Args:
            stream_name: Name of the video stream
            output_path: Path to save video file
            frame_processor: Function to process frames before saving
        """
        if stream_name in self.recording_flags and self.recording_flags[stream_name]:
            logger.warning(f"Stream {stream_name} already recording")
            return
        
        # Initialize buffer
        self.buffers[stream_name] = queue.Queue(maxsize=self.max_buffer_size)
        self.recording_flags[stream_name] = True
        self.frame_counts[stream_name] = 0
        self.dropped_frames[stream_name] = 0
        
        # Set processor
        if frame_processor:
            self.processors[stream_name] = frame_processor
        
        # Initialize video writer if output path provided
        if output_path:
            self._init_writer(stream_name, output_path)
        
        logger.info(f"Started recording stream: {stream_name}")
    
    def stop_recording(self, stream_name: str = "gameplay") -> Optional[np.ndarray]:
        """
        Stop recording and return video array.
        
        Args:
            stream_name: Name of the video stream
            
        Returns:
            Video as numpy array (THWC format) or None
        """
        if stream_name not in self.recording_flags:
            logger.warning(f"Stream {stream_name} not recording")
            return None
        
        # Stop recording
        self.recording_flags[stream_name] = False
        
        # Get all frames from buffer
        frames = []
        buffer = self.buffers.get(stream_name)
        
        if buffer:
            while not buffer.empty():
                try:
                    frame = buffer.get_nowait()
                    frames.append(frame.image)
                except queue.Empty:
                    break
        
        # Release writer if exists
        if stream_name in self.writers:
            self.writers[stream_name].release()
            del self.writers[stream_name]
        
        # Clean up
        if stream_name in self.buffers:
            del self.buffers[stream_name]
        if stream_name in self.processors:
            del self.processors[stream_name]
        
        logger.info(f"Stopped recording stream: {stream_name}. "
                   f"Captured {self.frame_counts[stream_name]} frames, "
                   f"dropped {self.dropped_frames[stream_name]}")
        
        # Return video array if frames captured
        if frames:
            return np.array(frames)
        return None
    
    def add_frame(self, frame: np.ndarray, stream_name: str = "gameplay",
                 metadata: Dict[str, Any] = None):
        """
        Add a frame to the video buffer.
        
        Args:
            frame: Frame image (HWC format, RGB)
            stream_name: Name of the video stream
            metadata: Frame metadata
        """
        if stream_name not in self.recording_flags or not self.recording_flags[stream_name]:
            return
        
        # Process frame if processor exists
        if stream_name in self.processors:
            frame = self.processors[stream_name](frame, metadata)
        
        # Create video frame
        video_frame = VideoFrame(
            image=frame,
            timestamp=time.time(),
            metadata=metadata
        )
        
        # Add to buffer
        buffer = self.buffers[stream_name]
        try:
            buffer.put_nowait(video_frame)
            self.frame_counts[stream_name] += 1
            
            # Write to file if writer exists
            if stream_name in self.writers:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.writers[stream_name].write(bgr_frame)
                
        except queue.Full:
            # Buffer full, drop oldest frame
            try:
                buffer.get_nowait()
                buffer.put_nowait(video_frame)
                self.dropped_frames[stream_name] += 1
            except queue.Empty:
                pass
    
    def add_overlay(self, frame: np.ndarray, overlays: Dict[str, Any]) -> np.ndarray:
        """
        Add overlays to a frame.
        
        Args:
            frame: Base frame
            overlays: Dictionary of overlays to add
            
        Returns:
            Frame with overlays
        """
        frame = frame.copy()
        
        # Add text overlays
        if 'text' in overlays:
            for text_info in overlays['text']:
                text = text_info['content']
                pos = text_info.get('position', (10, 30))
                color = text_info.get('color', (255, 255, 255))
                scale = text_info.get('scale', 0.7)
                thickness = text_info.get('thickness', 2)
                
                cv2.putText(frame, text, pos,
                          cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        
        # Add bounding boxes
        if 'boxes' in overlays:
            for box in overlays['boxes']:
                x1, y1, x2, y2 = box['coords']
                color = box.get('color', (0, 255, 0))
                thickness = box.get('thickness', 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add label if provided
                if 'label' in box:
                    label = box['label']
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add circles/keypoints
        if 'circles' in overlays:
            for circle in overlays['circles']:
                center = circle['center']
                radius = circle.get('radius', 5)
                color = circle.get('color', (255, 0, 0))
                thickness = circle.get('thickness', -1)
                
                cv2.circle(frame, center, radius, color, thickness)
        
        # Add lines
        if 'lines' in overlays:
            for line in overlays['lines']:
                pt1 = line['pt1']
                pt2 = line['pt2']
                color = line.get('color', (0, 0, 255))
                thickness = line.get('thickness', 2)
                
                cv2.line(frame, pt1, pt2, color, thickness)
        
        # Add progress bar
        if 'progress' in overlays:
            progress = overlays['progress']
            value = progress['value']
            max_value = progress.get('max', 100)
            pos = progress.get('position', (10, frame.shape[0] - 30))
            width = progress.get('width', 200)
            height = progress.get('height', 20)
            
            # Draw background
            cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height),
                        (100, 100, 100), -1)
            
            # Draw progress
            progress_width = int(width * (value / max_value))
            cv2.rectangle(frame, pos, (pos[0] + progress_width, pos[1] + height),
                        (0, 255, 0), -1)
            
            # Draw text
            text = f"{value}/{max_value}"
            cv2.putText(frame, text, (pos[0] + 5, pos[1] + 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def create_comparison_video(self, videos: List[np.ndarray], 
                              labels: List[str] = None) -> np.ndarray:
        """
        Create side-by-side comparison video.
        
        Args:
            videos: List of video arrays (THWC format)
            labels: Labels for each video
            
        Returns:
            Combined video array
        """
        if not videos:
            return None
        
        # Ensure all videos have same number of frames
        min_frames = min(len(v) for v in videos)
        videos = [v[:min_frames] for v in videos]
        
        # Create grid layout
        n_videos = len(videos)
        grid_size = int(np.ceil(np.sqrt(n_videos)))
        
        combined_frames = []
        
        for frame_idx in range(min_frames):
            # Get frames from each video
            frames = [v[frame_idx] for v in videos]
            
            # Add labels if provided
            if labels:
                for i, (frame, label) in enumerate(zip(frames, labels)):
                    cv2.putText(frame, label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize frames to same size
            target_size = frames[0].shape[:2]
            frames = [cv2.resize(f, (target_size[1], target_size[0])) 
                     for f in frames]
            
            # Create grid
            rows = []
            for i in range(0, len(frames), grid_size):
                row = frames[i:i+grid_size]
                # Pad row if needed
                while len(row) < grid_size:
                    row.append(np.zeros_like(frames[0]))
                rows.append(np.hstack(row))
            
            combined_frame = np.vstack(rows)
            combined_frames.append(combined_frame)
        
        return np.array(combined_frames)
    
    def create_highlight_reel(self, frames: List[np.ndarray],
                            scores: List[float],
                            threshold: float = 0.8,
                            max_duration: int = 300) -> np.ndarray:
        """
        Create highlight reel from frames based on scores.
        
        Args:
            frames: List of frames
            scores: Score for each frame (e.g., reward, interest)
            threshold: Score threshold for highlights
            max_duration: Maximum frames in highlight reel
            
        Returns:
            Highlight video array
        """
        # Find highlight segments
        highlights = []
        in_highlight = False
        segment_start = 0
        
        for i, score in enumerate(scores):
            if score >= threshold and not in_highlight:
                # Start new highlight segment
                in_highlight = True
                segment_start = i
            elif score < threshold and in_highlight:
                # End highlight segment
                in_highlight = False
                highlights.append((segment_start, i))
        
        # Close last segment if needed
        if in_highlight:
            highlights.append((segment_start, len(frames)))
        
        # Collect highlight frames
        highlight_frames = []
        for start, end in highlights:
            # Add transition effect at start
            if highlight_frames and len(highlight_frames) < max_duration:
                # Add fade transition
                for alpha in np.linspace(0, 1, 5):
                    if len(highlight_frames) >= max_duration:
                        break
                    transition_frame = cv2.addWeighted(
                        highlight_frames[-1], 1 - alpha,
                        frames[start], alpha, 0
                    )
                    highlight_frames.append(transition_frame)
            
            # Add segment frames
            for i in range(start, min(end, start + 30)):  # Limit segment length
                if len(highlight_frames) >= max_duration:
                    break
                highlight_frames.append(frames[i])
        
        if not highlight_frames:
            logger.warning("No highlights found above threshold")
            return None
        
        return np.array(highlight_frames)
    
    def save_video(self, video: np.ndarray, output_path: str,
                  add_timestamp: bool = True):
        """
        Save video array to file.
        
        Args:
            video: Video array (THWC format)
            output_path: Output file path
            add_timestamp: Add timestamp to filename
        """
        if add_timestamp:
            path = Path(output_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(path.parent / f"{path.stem}_{timestamp}{path.suffix}")
        
        # Get video dimensions
        height, width = video[0].shape[:2]
        
        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in video:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        
        writer.release()
        logger.info(f"Video saved to {output_path}")
    
    def _init_writer(self, stream_name: str, output_path: str):
        """Initialize video writer for a stream."""
        # This will be initialized when first frame is received
        # (need to know frame dimensions)
        pass
    
    def log_to_tensorboard(self, tb_logger, tag: str, video: np.ndarray,
                          step: int = 0):
        """
        Log video to TensorBoard.
        
        Args:
            tb_logger: TensorBoard logger instance
            tag: Video tag
            video: Video array (THWC format)
            step: Global step
        """
        if tb_logger:
            tb_logger.log_video(tag, video, self.fps, step)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recording statistics."""
        stats = {}
        
        for stream_name in self.frame_counts:
            stats[stream_name] = {
                'frames_captured': self.frame_counts[stream_name],
                'frames_dropped': self.dropped_frames[stream_name],
                'drop_rate': (self.dropped_frames[stream_name] / 
                            max(1, self.frame_counts[stream_name]))
            }
        
        return stats