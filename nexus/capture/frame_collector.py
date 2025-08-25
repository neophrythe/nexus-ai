"""Frame Collection Utilities - SerpentAI Compatible with Enhancements

Provides various frame collection modes for dataset creation and analysis.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import queue
from datetime import datetime
import structlog
import hashlib
from collections import defaultdict

from nexus.vision.game_frame import GameFrame, FrameMetadata
from nexus.capture.frame_grabber import FrameGrabber
from nexus.game_registry import initialize_game

logger = structlog.get_logger()


class CollectionMode(Enum):
    """Frame collection modes - SerpentAI compatible"""
    COLLECT_FRAMES = "COLLECT_FRAMES"  # Raw frame collection
    COLLECT_FRAMES_FOR_CONTEXT = "COLLECT_FRAMES_FOR_CONTEXT"  # Context classification
    COLLECT_FRAME_REGIONS = "COLLECT_FRAME_REGIONS"  # Specific regions
    COLLECT_FRAME_SEQUENCES = "COLLECT_FRAME_SEQUENCES"  # Temporal sequences
    COLLECT_ANNOTATED_FRAMES = "COLLECT_ANNOTATED_FRAMES"  # With annotations
    COLLECT_AUGMENTED_FRAMES = "COLLECT_AUGMENTED_FRAMES"  # With augmentations


@dataclass
class RegionOfInterest:
    """Region of interest for collection"""
    name: str
    x: int
    y: int
    width: int
    height: int
    tags: List[str] = field(default_factory=list)
    
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract region from frame"""
        return frame[self.y:self.y+self.height, self.x:self.x+self.width]


@dataclass
class CollectionConfig:
    """Frame collection configuration"""
    mode: CollectionMode = CollectionMode.COLLECT_FRAMES
    game_name: Optional[str] = None
    output_dir: str = "./collected_frames"
    fps: int = 10
    max_frames: int = 1000
    frame_delay: float = 0.1
    
    # Context collection
    context_labels: List[str] = field(default_factory=list)
    current_context: Optional[str] = None
    auto_context_detection: bool = False
    
    # Region collection
    regions: List[RegionOfInterest] = field(default_factory=list)
    
    # Sequence collection
    sequence_length: int = 10
    sequence_overlap: int = 5
    
    # Augmentation
    augmentations: List[str] = field(default_factory=list)
    augmentation_probability: float = 0.5
    
    # Deduplication
    deduplicate: bool = True
    similarity_threshold: float = 0.95
    
    # Annotations
    annotation_file: Optional[str] = None
    auto_annotate: bool = False
    
    # Storage
    save_format: str = "png"  # png, jpg, npy
    compress: bool = False
    organize_by_context: bool = True


class FrameAugmenter:
    """Frame augmentation utilities"""
    
    @staticmethod
    def flip_horizontal(frame: np.ndarray) -> np.ndarray:
        """Flip frame horizontally"""
        return cv2.flip(frame, 1)
        
    @staticmethod
    def flip_vertical(frame: np.ndarray) -> np.ndarray:
        """Flip frame vertically"""
        return cv2.flip(frame, 0)
        
    @staticmethod
    def rotate(frame: np.ndarray, angle: float) -> np.ndarray:
        """Rotate frame by angle"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))
        
    @staticmethod
    def adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] *= factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    @staticmethod
    def add_noise(frame: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add random noise"""
        noise = np.random.randn(*frame.shape) * intensity * 255
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    @staticmethod
    def blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
    @staticmethod
    def crop_random(frame: np.ndarray, crop_percent: float = 0.1) -> np.ndarray:
        """Random crop"""
        h, w = frame.shape[:2]
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        
        y = np.random.randint(0, crop_h)
        x = np.random.randint(0, crop_w)
        
        return frame[y:h-crop_h+y, x:w-crop_w+x]


class FrameCollector:
    """Enhanced frame collector with multiple modes"""
    
    def __init__(self, config: CollectionConfig):
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame grabber
        self.frame_grabber = None
        self.game = None
        
        # Collection state
        self.collecting = False
        self.paused = False
        self.frames_collected = 0
        self.frame_buffer = []
        self.context_frames = defaultdict(list)
        
        # Deduplication
        self.frame_hashes = set()
        
        # Augmenter
        self.augmenter = FrameAugmenter()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'unique_frames': 0,
            'duplicates_skipped': 0,
            'contexts': defaultdict(int),
            'regions_extracted': 0,
            'sequences_created': 0,
            'augmentations_applied': 0
        }
        
        # Annotations
        self.annotations = {}
        if self.config.annotation_file:
            self._load_annotations()
            
    def _load_annotations(self):
        """Load annotation file"""
        anno_path = Path(self.config.annotation_file)
        if anno_path.exists():
            with open(anno_path, 'r') as f:
                self.annotations = json.load(f)
                
    def start(self):
        """Start frame collection"""
        logger.info(f"Starting frame collection in {self.config.mode.value} mode")
        
        # Initialize game if specified
        if self.config.game_name:
            self.game = initialize_game(self.config.game_name)
            if self.game:
                self.game.launch()
                time.sleep(2)  # Wait for game to load
                
        # Setup frame grabber
        self._setup_frame_grabber()
        
        # Start collection based on mode
        self.collecting = True
        
        if self.config.mode == CollectionMode.COLLECT_FRAMES:
            self._collect_frames()
        elif self.config.mode == CollectionMode.COLLECT_FRAMES_FOR_CONTEXT:
            self._collect_frames_for_context()
        elif self.config.mode == CollectionMode.COLLECT_FRAME_REGIONS:
            self._collect_frame_regions()
        elif self.config.mode == CollectionMode.COLLECT_FRAME_SEQUENCES:
            self._collect_frame_sequences()
        elif self.config.mode == CollectionMode.COLLECT_ANNOTATED_FRAMES:
            self._collect_annotated_frames()
        elif self.config.mode == CollectionMode.COLLECT_AUGMENTED_FRAMES:
            self._collect_augmented_frames()
            
    def _setup_frame_grabber(self):
        """Setup frame grabber"""
        if self.game and self.game.frame_grabber:
            self.frame_grabber = self.game.frame_grabber
        else:
            # Create default frame grabber
            self.frame_grabber = FrameGrabber(
                fps=self.config.fps,
                buffer_seconds=5
            )
            
            # Start capture in background
            capture_thread = threading.Thread(target=self.frame_grabber.start, daemon=True)
            capture_thread.start()
            time.sleep(1)  # Wait for capture to start
            
    def _collect_frames(self):
        """Collect raw frames - COLLECT_FRAMES mode"""
        logger.info(f"Collecting {self.config.max_frames} raw frames")
        
        while self.collecting and self.frames_collected < self.config.max_frames:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Grab frame
            frame = self.frame_grabber.grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Check for duplicates
            if self.config.deduplicate:
                frame_hash = self._hash_frame(frame)
                if frame_hash in self.frame_hashes:
                    self.stats['duplicates_skipped'] += 1
                    continue
                self.frame_hashes.add(frame_hash)
                
            # Save frame
            self._save_frame(frame, f"frame_{self.frames_collected:06d}")
            
            self.frames_collected += 1
            self.stats['total_frames'] += 1
            self.stats['unique_frames'] += 1
            
            # Progress update
            if self.frames_collected % 100 == 0:
                logger.info(f"Collected {self.frames_collected}/{self.config.max_frames} frames")
                
            time.sleep(self.config.frame_delay)
            
        logger.info(f"Collection complete. Collected {self.frames_collected} frames")
        
    def _collect_frames_for_context(self):
        """Collect frames for context classification - COLLECT_FRAMES_FOR_CONTEXT mode"""
        logger.info(f"Collecting frames for contexts: {self.config.context_labels}")
        
        if not self.config.context_labels:
            logger.error("No context labels provided")
            return
            
        # Create context directories
        for context in self.config.context_labels:
            context_dir = self.output_dir / context
            context_dir.mkdir(exist_ok=True)
            
        # Collection loop
        while self.collecting and self.frames_collected < self.config.max_frames:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Get current context
            if self.config.auto_context_detection:
                context = self._detect_context()
            else:
                context = self.config.current_context
                
            if not context:
                logger.warning("No context set, waiting...")
                time.sleep(1)
                continue
                
            # Grab frame
            frame = self.frame_grabber.grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Check for duplicates within context
            if self.config.deduplicate:
                frame_hash = self._hash_frame(frame)
                if frame_hash in self.frame_hashes:
                    self.stats['duplicates_skipped'] += 1
                    continue
                self.frame_hashes.add(frame_hash)
                
            # Save to context directory
            context_dir = self.output_dir / context
            frame_name = f"{context}_{self.stats['contexts'][context]:06d}"
            self._save_frame(frame, frame_name, context_dir)
            
            self.context_frames[context].append(frame)
            self.stats['contexts'][context] += 1
            self.frames_collected += 1
            self.stats['total_frames'] += 1
            
            # Progress update
            if self.frames_collected % 50 == 0:
                context_stats = ", ".join([f"{c}: {n}" for c, n in self.stats['contexts'].items()])
                logger.info(f"Progress: {self.frames_collected}/{self.config.max_frames} ({context_stats})")
                
            time.sleep(self.config.frame_delay)
            
        logger.info(f"Context collection complete. Distribution: {dict(self.stats['contexts'])}")
        
    def _collect_frame_regions(self):
        """Collect specific frame regions - COLLECT_FRAME_REGIONS mode"""
        if not self.config.regions:
            logger.error("No regions defined")
            return
            
        logger.info(f"Collecting {len(self.config.regions)} regions from frames")
        
        # Create region directories
        for region in self.config.regions:
            region_dir = self.output_dir / region.name
            region_dir.mkdir(exist_ok=True)
            
        while self.collecting and self.frames_collected < self.config.max_frames:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Grab frame
            frame = self.frame_grabber.grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Extract regions
            for region in self.config.regions:
                try:
                    region_frame = region.extract(frame)
                    
                    # Save region
                    region_dir = self.output_dir / region.name
                    region_name = f"{region.name}_{self.stats['regions_extracted']:06d}"
                    self._save_frame(region_frame, region_name, region_dir)
                    
                    self.stats['regions_extracted'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to extract region {region.name}: {e}")
                    
            self.frames_collected += 1
            self.stats['total_frames'] += 1
            
            # Progress update
            if self.frames_collected % 50 == 0:
                logger.info(f"Collected {self.frames_collected}/{self.config.max_frames} frames, "
                          f"{self.stats['regions_extracted']} regions")
                
            time.sleep(self.config.frame_delay)
            
        logger.info(f"Region collection complete. Extracted {self.stats['regions_extracted']} regions")
        
    def _collect_frame_sequences(self):
        """Collect temporal frame sequences - COLLECT_FRAME_SEQUENCES mode"""
        logger.info(f"Collecting frame sequences (length: {self.config.sequence_length})")
        
        sequence_buffer = []
        sequence_count = 0
        
        while self.collecting and sequence_count < self.config.max_frames // self.config.sequence_length:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Grab frame
            frame = self.frame_grabber.grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            sequence_buffer.append(frame)
            
            # Check if we have a complete sequence
            if len(sequence_buffer) >= self.config.sequence_length:
                # Save sequence
                sequence_dir = self.output_dir / f"sequence_{sequence_count:04d}"
                sequence_dir.mkdir(exist_ok=True)
                
                for i, seq_frame in enumerate(sequence_buffer):
                    frame_name = f"frame_{i:03d}"
                    self._save_frame(seq_frame, frame_name, sequence_dir)
                    
                # Save sequence metadata
                metadata = {
                    'sequence_id': sequence_count,
                    'length': len(sequence_buffer),
                    'timestamp': datetime.now().isoformat(),
                    'fps': self.config.fps
                }
                
                with open(sequence_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                sequence_count += 1
                self.stats['sequences_created'] += 1
                
                # Overlap for next sequence
                if self.config.sequence_overlap > 0:
                    sequence_buffer = sequence_buffer[-self.config.sequence_overlap:]
                else:
                    sequence_buffer = []
                    
                logger.info(f"Saved sequence {sequence_count}")
                
            self.frames_collected += 1
            self.stats['total_frames'] += 1
            
            time.sleep(self.config.frame_delay)
            
        logger.info(f"Sequence collection complete. Created {self.stats['sequences_created']} sequences")
        
    def _collect_annotated_frames(self):
        """Collect frames with annotations - COLLECT_ANNOTATED_FRAMES mode"""
        logger.info("Collecting annotated frames")
        
        annotations_data = []
        
        while self.collecting and self.frames_collected < self.config.max_frames:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Grab frame
            frame = self.frame_grabber.grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Get annotations
            frame_annotations = {}
            
            if self.config.auto_annotate:
                # Auto-detect annotations
                frame_annotations = self._auto_annotate(frame)
            elif self.annotations:
                # Use provided annotations
                frame_id = f"frame_{self.frames_collected:06d}"
                frame_annotations = self.annotations.get(frame_id, {})
                
            # Save frame
            frame_name = f"frame_{self.frames_collected:06d}"
            self._save_frame(frame, frame_name)
            
            # Save annotations
            if frame_annotations:
                annotations_data.append({
                    'frame_id': frame_name,
                    'annotations': frame_annotations
                })
                
            self.frames_collected += 1
            self.stats['total_frames'] += 1
            
            # Progress update
            if self.frames_collected % 50 == 0:
                logger.info(f"Collected {self.frames_collected}/{self.config.max_frames} annotated frames")
                
            time.sleep(self.config.frame_delay)
            
        # Save all annotations
        if annotations_data:
            anno_file = self.output_dir / "annotations.json"
            with open(anno_file, 'w') as f:
                json.dump(annotations_data, f, indent=2)
                
        logger.info(f"Annotation collection complete. Saved {len(annotations_data)} annotations")
        
    def _collect_augmented_frames(self):
        """Collect frames with augmentations - COLLECT_AUGMENTED_FRAMES mode"""
        logger.info(f"Collecting augmented frames (augmentations: {self.config.augmentations})")
        
        augmentation_funcs = {
            'flip_h': self.augmenter.flip_horizontal,
            'flip_v': self.augmenter.flip_vertical,
            'rotate': lambda f: self.augmenter.rotate(f, np.random.uniform(-30, 30)),
            'brightness': lambda f: self.augmenter.adjust_brightness(f, np.random.uniform(0.5, 1.5)),
            'noise': self.augmenter.add_noise,
            'blur': self.augmenter.blur,
            'crop': self.augmenter.crop_random
        }
        
        while self.collecting and self.frames_collected < self.config.max_frames:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Grab frame
            frame = self.frame_grabber.grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Save original
            frame_name = f"frame_{self.frames_collected:06d}_original"
            self._save_frame(frame, frame_name)
            
            # Apply augmentations
            for aug_name in self.config.augmentations:
                if aug_name in augmentation_funcs:
                    if np.random.random() < self.config.augmentation_probability:
                        try:
                            augmented = augmentation_funcs[aug_name](frame)
                            aug_frame_name = f"frame_{self.frames_collected:06d}_{aug_name}"
                            self._save_frame(augmented, aug_frame_name)
                            self.stats['augmentations_applied'] += 1
                        except Exception as e:
                            logger.error(f"Augmentation {aug_name} failed: {e}")
                            
            self.frames_collected += 1
            self.stats['total_frames'] += 1
            
            # Progress update
            if self.frames_collected % 50 == 0:
                logger.info(f"Collected {self.frames_collected}/{self.config.max_frames} frames, "
                          f"{self.stats['augmentations_applied']} augmentations")
                
            time.sleep(self.config.frame_delay)
            
        logger.info(f"Augmented collection complete. Applied {self.stats['augmentations_applied']} augmentations")
        
    def _save_frame(self, frame: np.ndarray, name: str, directory: Optional[Path] = None):
        """Save frame to disk"""
        save_dir = directory or self.output_dir
        
        if self.config.save_format == "npy":
            # Save as numpy array
            file_path = save_dir / f"{name}.npy"
            np.save(file_path, frame)
        else:
            # Save as image
            file_path = save_dir / f"{name}.{self.config.save_format}"
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
                
            # Compression settings
            if self.config.save_format == "jpg":
                quality = 85 if self.config.compress else 95
                cv2.imwrite(str(file_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(file_path), frame_bgr)
                
    def _hash_frame(self, frame: np.ndarray) -> str:
        """Generate hash for frame deduplication"""
        # Resize for faster hashing
        small = cv2.resize(frame, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()
        
    def _detect_context(self) -> Optional[str]:
        """Auto-detect current context"""
        # This would use a context classifier
        # For now, return None
        return None
        
    def _auto_annotate(self, frame: np.ndarray) -> Dict:
        """Auto-generate annotations for frame"""
        annotations = {}
        
        # Example: detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        annotations['edge_count'] = np.sum(edges > 0)
        
        # Example: dominant color
        avg_color = frame.mean(axis=(0, 1))
        annotations['dominant_color'] = avg_color.tolist()
        
        return annotations
        
    def set_context(self, context: str):
        """Set current collection context"""
        if context in self.config.context_labels:
            self.config.current_context = context
            logger.info(f"Context set to: {context}")
        else:
            logger.warning(f"Unknown context: {context}")
            
    def pause(self):
        """Pause collection"""
        self.paused = True
        logger.info("Collection paused")
        
    def resume(self):
        """Resume collection"""
        self.paused = False
        logger.info("Collection resumed")
        
    def stop(self):
        """Stop collection"""
        self.collecting = False
        
        # Stop frame grabber
        if self.frame_grabber:
            self.frame_grabber.stop()
            
        # Close game if launched
        if self.game:
            self.game.close()
            
        # Save final statistics
        stats_file = self.output_dir / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
            
        logger.info(f"Collection stopped. Stats saved to {stats_file}")
        
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        return {
            'mode': self.config.mode.value,
            'total_frames': self.stats['total_frames'],
            'unique_frames': self.stats['unique_frames'],
            'duplicates_skipped': self.stats['duplicates_skipped'],
            'contexts': dict(self.stats['contexts']),
            'regions_extracted': self.stats['regions_extracted'],
            'sequences_created': self.stats['sequences_created'],
            'augmentations_applied': self.stats['augmentations_applied'],
            'output_directory': str(self.output_dir)
        }


# SerpentAI compatibility functions
def collect_frames(game_name: str, count: int = 1000, mode: str = "COLLECT_FRAMES"):
    """Collect frames - SerpentAI compatible"""
    config = CollectionConfig(
        mode=CollectionMode[mode],
        game_name=game_name,
        max_frames=count
    )
    
    collector = FrameCollector(config)
    collector.start()
    
    # Wait for collection to complete
    while collector.collecting:
        time.sleep(1)
        
    return collector


def collect_frames_for_context(game_name: str, contexts: List[str], count_per_context: int = 100):
    """Collect frames for context classification - SerpentAI compatible"""
    config = CollectionConfig(
        mode=CollectionMode.COLLECT_FRAMES_FOR_CONTEXT,
        game_name=game_name,
        max_frames=count_per_context * len(contexts),
        context_labels=contexts
    )
    
    collector = FrameCollector(config)
    
    # Interactive context collection
    logger.info(f"Collecting frames for contexts: {contexts}")
    logger.info("Press Enter to cycle through contexts...")
    
    collector.start()
    
    for context in contexts:
        input(f"\nPress Enter to start collecting for context: {context}")
        collector.set_context(context)
        
        # Wait for enough frames
        while collector.stats['contexts'][context] < count_per_context:
            time.sleep(0.5)
            
    collector.stop()
    return collector