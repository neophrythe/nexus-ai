"""RecorderAgent - Synchronized recording agent for training data generation

SerpentAI compatible agent that records synchronized gameplay sessions with
frames, input events, and rewards for creating high-quality training datasets.
"""

import time
import json
import pickle
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import threading
import queue
import structlog

from nexus.agents.game_agent import GameAgent, FrameHandlerMode, AgentConfig
from nexus.vision.game_frame import GameFrame
from nexus.input.controller import InputController

logger = structlog.get_logger()


@dataclass
class RecordingMetadata:
    """Metadata for a recording session"""
    session_id: str
    game_name: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_frames: int = 0
    total_events: int = 0
    total_rewards: float = 0.0
    fps: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    config: Dict[str, Any] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class FrameRecord:
    """Single frame record with metadata"""
    frame_number: int
    timestamp: float
    frame_data: np.ndarray
    reward: float = 0.0
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputRecord:
    """Single input event record"""
    frame_number: int
    timestamp: float
    event_type: str  # 'key_press', 'key_release', 'mouse_move', 'mouse_click'
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecorderAgent(GameAgent):
    """
    RecorderAgent for synchronized gameplay recording - SerpentAI compatible
    
    Records gameplay sessions with synchronized frames, inputs, and rewards
    for training data generation.
    """
    
    def __init__(self, game, config: Optional[AgentConfig] = None,
                 output_dir: str = "recordings",
                 recording_format: str = "hdf5",
                 buffer_size: int = 1000,
                 compress: bool = True,
                 record_inputs: bool = True,
                 record_rewards: bool = True,
                 record_metadata: bool = True):
        """
        Initialize RecorderAgent
        
        Args:
            game: Game instance
            config: Agent configuration
            output_dir: Directory to save recordings
            recording_format: Format for saving ('hdf5', 'pickle', 'json', 'numpy')
            buffer_size: Buffer size for batched writes
            compress: Whether to compress data
            record_inputs: Record input events
            record_rewards: Record rewards
            record_metadata: Record metadata
        """
        super().__init__(game, config)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.recording_format = recording_format
        self.buffer_size = buffer_size
        self.compress = compress
        self.record_inputs = record_inputs
        self.record_rewards = record_rewards
        self.record_metadata = record_metadata
        
        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.session_id = None
        self.recording_metadata = None
        
        # Data buffers
        self.frame_buffer = deque(maxlen=buffer_size)
        self.input_buffer = deque(maxlen=buffer_size)
        self.reward_buffer = deque(maxlen=buffer_size)
        
        # Synchronization
        self.frame_counter = 0
        self.start_time = None
        self.last_frame_time = None
        
        # Storage handlers
        self.storage_handler = None
        self.writer_thread = None
        self.write_queue = queue.Queue()
        
        # Input monitoring
        self.input_monitor = None
        if self.record_inputs:
            self._setup_input_monitoring()
        
        # Set recorder mode
        self.set_frame_handler(FrameHandlerMode.RECORD)
        
    def _setup_input_monitoring(self):
        """Setup input event monitoring"""
        try:
            from pynput import keyboard, mouse
            
            self.keyboard_events = []
            self.mouse_events = []
            
            # Keyboard listener
            def on_key_press(key):
                if self.is_recording and not self.is_paused:
                    event = InputRecord(
                        frame_number=self.frame_counter,
                        timestamp=time.time() - self.start_time,
                        event_type='key_press',
                        data={'key': str(key)}
                    )
                    self.input_buffer.append(event)
            
            def on_key_release(key):
                if self.is_recording and not self.is_paused:
                    event = InputRecord(
                        frame_number=self.frame_counter,
                        timestamp=time.time() - self.start_time,
                        event_type='key_release',
                        data={'key': str(key)}
                    )
                    self.input_buffer.append(event)
            
            # Mouse listener
            def on_mouse_move(x, y):
                if self.is_recording and not self.is_paused:
                    # Throttle mouse move events
                    if len(self.mouse_events) == 0 or \
                       time.time() - self.mouse_events[-1].timestamp > 0.05:
                        event = InputRecord(
                            frame_number=self.frame_counter,
                            timestamp=time.time() - self.start_time,
                            event_type='mouse_move',
                            data={'x': x, 'y': y}
                        )
                        self.input_buffer.append(event)
                        self.mouse_events.append(event)
            
            def on_mouse_click(x, y, button, pressed):
                if self.is_recording and not self.is_paused:
                    event = InputRecord(
                        frame_number=self.frame_counter,
                        timestamp=time.time() - self.start_time,
                        event_type='mouse_click' if pressed else 'mouse_release',
                        data={'x': x, 'y': y, 'button': str(button)}
                    )
                    self.input_buffer.append(event)
            
            self.keyboard_listener = keyboard.Listener(
                on_press=on_key_press,
                on_release=on_key_release
            )
            self.mouse_listener = mouse.Listener(
                on_move=on_mouse_move,
                on_click=on_mouse_click
            )
            
        except ImportError:
            logger.warning("pynput not available, input recording disabled")
            self.record_inputs = False
    
    def start_recording(self, session_name: Optional[str] = None,
                       labels: List[str] = None,
                       tags: List[str] = None) -> str:
        """
        Start recording session
        
        Args:
            session_name: Optional session name
            labels: Labels for the recording
            tags: Tags for the recording
            
        Returns:
            Session ID
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return self.session_id
        
        # Generate session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{self.game.name}_{session_name or 'session'}_{timestamp}"
        
        # Create metadata
        self.recording_metadata = RecordingMetadata(
            session_id=self.session_id,
            game_name=self.game.name,
            agent_name=self.__class__.__name__,
            start_time=datetime.now(),
            labels=labels or [],
            tags=tags or [],
            config=self.config.__dict__ if self.config else {}
        )
        
        # Setup storage
        self._setup_storage()
        
        # Start monitoring
        if self.record_inputs and hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.start()
            self.mouse_listener.start()
        
        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_worker)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        
        # Set recording state
        self.is_recording = True
        self.is_paused = False
        self.start_time = time.time()
        self.frame_counter = 0
        
        logger.info(f"Recording started: {self.session_id}")
        return self.session_id
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording session
        
        Returns:
            Recording summary
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return {}
        
        # Update metadata
        self.recording_metadata.end_time = datetime.now()
        self.recording_metadata.total_frames = self.frame_counter
        
        # Calculate FPS
        if self.start_time:
            duration = time.time() - self.start_time
            self.recording_metadata.fps = self.frame_counter / duration if duration > 0 else 0
        
        # Flush buffers
        self._flush_buffers()
        
        # Stop monitoring
        if self.record_inputs and hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
            self.mouse_listener.stop()
        
        # Stop writer thread
        if self.writer_thread:
            self.write_queue.put(None)  # Signal to stop
            self.writer_thread.join(timeout=5.0)
        
        # Finalize storage
        self._finalize_storage()
        
        # Reset state
        self.is_recording = False
        self.is_paused = False
        self.session_id = None
        
        summary = {
            'session_id': self.recording_metadata.session_id,
            'duration': (self.recording_metadata.end_time - 
                        self.recording_metadata.start_time).total_seconds(),
            'total_frames': self.recording_metadata.total_frames,
            'total_events': self.recording_metadata.total_events,
            'total_rewards': self.recording_metadata.total_rewards,
            'fps': self.recording_metadata.fps,
            'output_file': self._get_output_path()
        }
        
        logger.info(f"Recording stopped: {summary}")
        return summary
    
    def pause_recording(self):
        """Pause recording"""
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            logger.info("Recording paused")
    
    def resume_recording(self):
        """Resume recording"""
        if self.is_recording and self.is_paused:
            self.is_paused = False
            logger.info("Recording resumed")
    
    def _handle_frame_record(self, game_frame: GameFrame):
        """Handle frame in record mode"""
        if not self.is_recording or self.is_paused:
            return
        
        # Create frame record
        frame_record = FrameRecord(
            frame_number=self.frame_counter,
            timestamp=time.time() - self.start_time,
            frame_data=game_frame.frame.copy(),
            metadata={
                'resolution': game_frame.frame.shape[:2],
                'channels': game_frame.frame.shape[2] if len(game_frame.frame.shape) > 2 else 1
            }
        )
        
        # Add to buffer
        self.frame_buffer.append(frame_record)
        
        # Get reward if available
        if self.record_rewards:
            reward = self.get_reward(game_frame)
            frame_record.reward = reward
            self.recording_metadata.total_rewards += reward
        
        # Update counter
        self.frame_counter += 1
        self.recording_metadata.total_frames = self.frame_counter
        
        # Update resolution
        if self.recording_metadata.resolution == (0, 0):
            self.recording_metadata.resolution = game_frame.frame.shape[:2]
        
        # Flush if buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            self._flush_buffers()
    
    def _flush_buffers(self):
        """Flush data buffers to storage"""
        if not self.frame_buffer and not self.input_buffer:
            return
        
        # Prepare data batch
        batch = {
            'frames': list(self.frame_buffer),
            'inputs': list(self.input_buffer) if self.record_inputs else [],
            'rewards': [f.reward for f in self.frame_buffer] if self.record_rewards else [],
            'metadata': {
                'batch_number': self.frame_counter // self.buffer_size,
                'timestamp': time.time()
            }
        }
        
        # Add to write queue
        self.write_queue.put(batch)
        
        # Clear buffers
        self.frame_buffer.clear()
        self.input_buffer.clear()
        
        logger.debug(f"Flushed buffers: {len(batch['frames'])} frames")
    
    def _writer_worker(self):
        """Background thread for writing data"""
        while True:
            try:
                batch = self.write_queue.get(timeout=1.0)
                if batch is None:  # Stop signal
                    break
                
                self._write_batch(batch)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer thread error: {e}")
    
    def _setup_storage(self):
        """Setup storage handler based on format"""
        output_path = self._get_output_path()
        
        if self.recording_format == 'hdf5':
            self._setup_hdf5_storage(output_path)
        elif self.recording_format == 'pickle':
            self.storage_handler = {'frames': [], 'inputs': [], 'metadata': {}}
        elif self.recording_format == 'numpy':
            self.storage_handler = {'frames': [], 'inputs': [], 'metadata': {}}
        else:
            raise ValueError(f"Unsupported format: {self.recording_format}")
    
    def _setup_hdf5_storage(self, filepath):
        """Setup HDF5 storage"""
        self.storage_handler = h5py.File(filepath, 'w')
        
        # Create groups
        self.storage_handler.create_group('frames')
        self.storage_handler.create_group('inputs')
        self.storage_handler.create_group('rewards')
        self.storage_handler.create_group('metadata')
        
        # Store initial metadata
        for key, value in asdict(self.recording_metadata).items():
            if value is not None:
                try:
                    if isinstance(value, (list, dict)):
                        self.storage_handler['metadata'].attrs[key] = json.dumps(value)
                    else:
                        self.storage_handler['metadata'].attrs[key] = value
                except:
                    pass
    
    def _write_batch(self, batch: Dict[str, Any]):
        """Write batch to storage"""
        if self.recording_format == 'hdf5':
            self._write_batch_hdf5(batch)
        elif self.recording_format == 'pickle':
            self._write_batch_pickle(batch)
        elif self.recording_format == 'numpy':
            self._write_batch_numpy(batch)
    
    def _write_batch_hdf5(self, batch: Dict[str, Any]):
        """Write batch to HDF5"""
        try:
            # Write frames
            for frame_record in batch['frames']:
                frame_name = f"frame_{frame_record.frame_number:08d}"
                
                # Create dataset with compression
                if self.compress:
                    self.storage_handler['frames'].create_dataset(
                        frame_name,
                        data=frame_record.frame_data,
                        compression='gzip',
                        compression_opts=4
                    )
                else:
                    self.storage_handler['frames'][frame_name] = frame_record.frame_data
                
                # Add attributes
                self.storage_handler['frames'][frame_name].attrs['timestamp'] = frame_record.timestamp
                self.storage_handler['frames'][frame_name].attrs['reward'] = frame_record.reward
                self.storage_handler['frames'][frame_name].attrs['terminal'] = frame_record.terminal
            
            # Write inputs
            if batch['inputs']:
                for input_record in batch['inputs']:
                    input_name = f"event_{input_record.frame_number:08d}_{input_record.timestamp:.4f}"
                    
                    # Store as JSON
                    self.storage_handler['inputs'].create_dataset(
                        input_name,
                        data=json.dumps({
                            'frame_number': input_record.frame_number,
                            'timestamp': input_record.timestamp,
                            'event_type': input_record.event_type,
                            'data': input_record.data
                        })
                    )
            
            # Flush to disk
            self.storage_handler.flush()
            
        except Exception as e:
            logger.error(f"Failed to write HDF5 batch: {e}")
    
    def _write_batch_pickle(self, batch: Dict[str, Any]):
        """Write batch to pickle storage"""
        self.storage_handler['frames'].extend(batch['frames'])
        self.storage_handler['inputs'].extend(batch['inputs'])
    
    def _write_batch_numpy(self, batch: Dict[str, Any]):
        """Write batch to numpy storage"""
        # Convert to numpy arrays
        frames = np.array([f.frame_data for f in batch['frames']])
        timestamps = np.array([f.timestamp for f in batch['frames']])
        rewards = np.array([f.reward for f in batch['frames']])
        
        self.storage_handler['frames'].append(frames)
        self.storage_handler['inputs'].extend(batch['inputs'])
    
    def _finalize_storage(self):
        """Finalize and close storage"""
        output_path = self._get_output_path()
        
        if self.recording_format == 'hdf5':
            # Update final metadata
            self.storage_handler['metadata'].attrs['end_time'] = self.recording_metadata.end_time.isoformat()
            self.storage_handler['metadata'].attrs['total_frames'] = self.recording_metadata.total_frames
            self.storage_handler['metadata'].attrs['fps'] = self.recording_metadata.fps
            
            # Close file
            self.storage_handler.close()
            
        elif self.recording_format == 'pickle':
            # Save to pickle file
            with open(output_path, 'wb') as f:
                data = {
                    'frames': self.storage_handler['frames'],
                    'inputs': self.storage_handler['inputs'],
                    'metadata': asdict(self.recording_metadata)
                }
                pickle.dump(data, f)
                
        elif self.recording_format == 'numpy':
            # Save numpy arrays
            np.savez_compressed(
                output_path,
                frames=np.concatenate(self.storage_handler['frames']),
                metadata=asdict(self.recording_metadata)
            )
        
        logger.info(f"Recording saved to: {output_path}")
    
    def _get_output_path(self) -> Path:
        """Get output file path"""
        extensions = {
            'hdf5': '.h5',
            'pickle': '.pkl',
            'numpy': '.npz',
            'json': '.json'
        }
        
        ext = extensions.get(self.recording_format, '.dat')
        return self.output_dir / f"{self.session_id}{ext}"
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """Get current recording statistics"""
        if not self.is_recording:
            return {}
        
        duration = time.time() - self.start_time if self.start_time else 0
        current_fps = self.frame_counter / duration if duration > 0 else 0
        
        return {
            'session_id': self.session_id,
            'is_recording': self.is_recording,
            'is_paused': self.is_paused,
            'duration': duration,
            'frames_recorded': self.frame_counter,
            'current_fps': current_fps,
            'buffer_frames': len(self.frame_buffer),
            'buffer_inputs': len(self.input_buffer),
            'total_rewards': self.recording_metadata.total_rewards if self.recording_metadata else 0
        }
    
    def export_to_dataset(self, output_path: str, 
                         dataset_format: str = 'pytorch',
                         split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Export recording to ML dataset format
        
        Args:
            output_path: Output path for dataset
            dataset_format: Format ('pytorch', 'tensorflow', 'numpy')
            split_ratio: Train/val/test split ratio
        """
        # Load recording
        recording_path = self._get_output_path()
        
        if self.recording_format == 'hdf5':
            with h5py.File(recording_path, 'r') as f:
                frames = []
                rewards = []
                
                for frame_name in sorted(f['frames'].keys()):
                    frames.append(f['frames'][frame_name][:])
                    rewards.append(f['frames'][frame_name].attrs['reward'])
                
                frames = np.array(frames)
                rewards = np.array(rewards)
        
        # Create dataset splits
        n_samples = len(frames)
        n_train = int(n_samples * split_ratio[0])
        n_val = int(n_samples * split_ratio[1])
        
        train_frames = frames[:n_train]
        train_rewards = rewards[:n_train]
        
        val_frames = frames[n_train:n_train+n_val]
        val_rewards = rewards[n_train:n_train+n_val]
        
        test_frames = frames[n_train+n_val:]
        test_rewards = rewards[n_train+n_val:]
        
        # Export based on format
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if dataset_format == 'pytorch':
            import torch
            torch.save({
                'train': {'frames': train_frames, 'rewards': train_rewards},
                'val': {'frames': val_frames, 'rewards': val_rewards},
                'test': {'frames': test_frames, 'rewards': test_rewards},
                'metadata': asdict(self.recording_metadata)
            }, output_path / 'dataset.pt')
            
        elif dataset_format == 'tensorflow':
            import tensorflow as tf
            
            # Create TFRecord files
            for split_name, split_frames, split_rewards in [
                ('train', train_frames, train_rewards),
                ('val', val_frames, val_rewards),
                ('test', test_frames, test_rewards)
            ]:
                writer = tf.io.TFRecordWriter(str(output_path / f'{split_name}.tfrecord'))
                
                for frame, reward in zip(split_frames, split_rewards):
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'frame': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frame.tobytes()])),
                        'reward': tf.train.Feature(float_list=tf.train.FloatList(value=[reward]))
                    }))
                    writer.write(example.SerializeToString())
                
                writer.close()
                
        elif dataset_format == 'numpy':
            np.savez_compressed(
                output_path / 'dataset.npz',
                train_frames=train_frames,
                train_rewards=train_rewards,
                val_frames=val_frames,
                val_rewards=val_rewards,
                test_frames=test_frames,
                test_rewards=test_rewards,
                metadata=asdict(self.recording_metadata)
            )
        
        logger.info(f"Dataset exported to: {output_path}")


# SerpentAI compatibility
RecorderGameAgent = RecorderAgent