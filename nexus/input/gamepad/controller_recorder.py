"""
Controller Recorder for Input Recording and Playback

Records and replays controller inputs for testing and automation.
"""

import json
import time
import threading
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO
from dataclasses import dataclass, asdict
import structlog

from nexus.input.gamepad.gamepad_base import (
    GamepadBase, Button, Axis, ControllerState, ControllerEvent
)

logger = structlog.get_logger()


@dataclass
class RecordedFrame:
    """Single recorded frame of controller input."""
    timestamp: float
    state: ControllerState
    events: List[ControllerEvent] = None
    metadata: Dict[str, Any] = None


class ControllerRecorder:
    """
    Records and replays controller inputs with compression and metadata.
    
    Features:
    - High-precision timing recording
    - Compressed storage format
    - Event-based and state-based recording
    - Metadata and annotation support
    - Multiple recording formats
    - Playback with speed control
    """
    
    def __init__(self, controller: GamepadBase):
        """
        Initialize controller recorder.
        
        Args:
            controller: Controller to record from
        """
        self.controller = controller
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = 0
        self.recorded_frames: List[RecordedFrame] = []
        self.recording_metadata: Dict[str, Any] = {}
        
        # Playback state
        self.is_playing = False
        self.playback_thread = None
        self.playback_speed = 1.0
        self.playback_index = 0
        
        # Recording options
        self.record_events = True
        self.record_states = True
        self.record_interval = 1.0 / 60  # 60 FPS default
        self.compression = True
        
        # Event listener for event-based recording
        self.event_buffer: List[ControllerEvent] = []
        
        logger.info(f"Controller recorder initialized for controller {controller.controller_id}")
    
    def start_recording(self, metadata: Dict[str, Any] = None):
        """
        Start recording controller input.
        
        Args:
            metadata: Optional metadata to store with recording
        """
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        # Clear previous recording
        self.recorded_frames = []
        self.event_buffer = []
        self.recording_metadata = metadata or {}
        
        # Add recording info to metadata
        self.recording_metadata.update({
            'start_time': time.time(),
            'controller_type': self.controller.controller_type.value,
            'controller_id': self.controller.controller_id,
            'record_events': self.record_events,
            'record_states': self.record_states,
            'record_interval': self.record_interval
        })
        
        # Start recording
        self.recording_start_time = time.time()
        self.is_recording = True
        
        # Add event listener if recording events
        if self.record_events:
            self.controller.add_event_listener(self._on_controller_event)
        
        # Start state recording thread if recording states
        if self.record_states:
            self._start_state_recording()
        
        logger.info("Started recording controller input")
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and return recording data.
        
        Returns:
            Recording data with frames and metadata
        """
        if not self.is_recording:
            logger.warning("Not recording")
            return {}
        
        self.is_recording = False
        
        # Remove event listener
        if self.record_events:
            self.controller.remove_event_listener(self._on_controller_event)
        
        # Update metadata
        self.recording_metadata['end_time'] = time.time()
        self.recording_metadata['duration'] = (
            self.recording_metadata['end_time'] - self.recording_metadata['start_time']
        )
        self.recording_metadata['frame_count'] = len(self.recorded_frames)
        
        logger.info(f"Stopped recording. Captured {len(self.recorded_frames)} frames")
        
        return {
            'metadata': self.recording_metadata,
            'frames': self.recorded_frames
        }
    
    def save_recording(self, filepath: str, format: str = 'json'):
        """
        Save recording to file.
        
        Args:
            filepath: Path to save file
            format: File format ('json', 'binary', 'compressed')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._save_json(path)
        elif format == 'compressed':
            self._save_compressed(path)
        elif format == 'binary':
            self._save_binary(path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved recording to {filepath} ({format} format)")
    
    def load_recording(self, filepath: str, format: str = None) -> bool:
        """
        Load recording from file.
        
        Args:
            filepath: Path to recording file
            format: File format (auto-detect if None)
        
        Returns:
            True if loaded successfully
        """
        path = Path(filepath)
        
        if not path.exists():
            logger.error(f"Recording file not found: {filepath}")
            return False
        
        # Auto-detect format
        if format is None:
            if path.suffix == '.json':
                format = 'json'
            elif path.suffix == '.gz':
                format = 'compressed'
            elif path.suffix == '.bin':
                format = 'binary'
            else:
                format = 'json'
        
        try:
            if format == 'json':
                self._load_json(path)
            elif format == 'compressed':
                self._load_compressed(path)
            elif format == 'binary':
                self._load_binary(path)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            logger.info(f"Loaded recording from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return False
    
    def start_playback(self, loop: bool = False, speed: float = 1.0):
        """
        Start playing back recorded input.
        
        Args:
            loop: Whether to loop playback
            speed: Playback speed multiplier
        """
        if not self.recorded_frames:
            logger.warning("No recording to play back")
            return
        
        if self.is_playing:
            logger.warning("Already playing")
            return
        
        self.is_playing = True
        self.playback_speed = speed
        self.playback_index = 0
        
        self.playback_thread = threading.Thread(
            target=self._playback_loop,
            args=(loop,),
            daemon=True
        )
        self.playback_thread.start()
        
        logger.info(f"Started playback (speed={speed}x, loop={loop})")
    
    def stop_playback(self):
        """Stop playback."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
            self.playback_thread = None
        
        logger.info("Stopped playback")
    
    def pause_playback(self):
        """Pause playback."""
        self.is_playing = False
        logger.info("Paused playback")
    
    def resume_playback(self):
        """Resume playback."""
        if self.playback_thread and not self.is_playing:
            self.is_playing = True
            logger.info("Resumed playback")
    
    def seek(self, timestamp: float):
        """
        Seek to specific timestamp in recording.
        
        Args:
            timestamp: Target timestamp in seconds
        """
        # Find closest frame
        for i, frame in enumerate(self.recorded_frames):
            if frame.timestamp >= timestamp:
                self.playback_index = i
                logger.info(f"Seeked to {timestamp}s (frame {i})")
                return
        
        # Seek to end if timestamp is beyond recording
        self.playback_index = len(self.recorded_frames) - 1
    
    def get_recording_info(self) -> Dict[str, Any]:
        """Get information about current recording."""
        if not self.recorded_frames:
            return {'status': 'empty'}
        
        return {
            'status': 'loaded',
            'metadata': self.recording_metadata,
            'frame_count': len(self.recorded_frames),
            'duration': self.recording_metadata.get('duration', 0),
            'current_index': self.playback_index,
            'is_playing': self.is_playing,
            'playback_speed': self.playback_speed
        }
    
    # Private methods
    
    def _on_controller_event(self, event: ControllerEvent):
        """Handle controller event during recording."""
        if self.is_recording:
            self.event_buffer.append(event)
    
    def _start_state_recording(self):
        """Start thread for state-based recording."""
        def record_states():
            last_time = time.time()
            
            while self.is_recording:
                current_time = time.time()
                
                if current_time - last_time >= self.record_interval:
                    # Record current state
                    state = self.controller.get_state()
                    
                    # Get buffered events
                    events = self.event_buffer.copy()
                    self.event_buffer.clear()
                    
                    frame = RecordedFrame(
                        timestamp=current_time - self.recording_start_time,
                        state=state,
                        events=events if self.record_events else None
                    )
                    
                    self.recorded_frames.append(frame)
                    last_time = current_time
                
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
        
        threading.Thread(target=record_states, daemon=True).start()
    
    def _playback_loop(self, loop: bool):
        """Main playback loop."""
        start_time = time.time()
        
        while self.is_playing:
            if self.playback_index >= len(self.recorded_frames):
                if loop:
                    self.playback_index = 0
                    start_time = time.time()
                else:
                    self.is_playing = False
                    break
            
            frame = self.recorded_frames[self.playback_index]
            
            # Calculate target time
            target_time = start_time + (frame.timestamp / self.playback_speed)
            
            # Wait until target time
            while time.time() < target_time and self.is_playing:
                time.sleep(0.001)
            
            if not self.is_playing:
                break
            
            # Apply frame state to controller
            if isinstance(self.controller, VirtualGamepad):
                # For virtual gamepad, directly set state
                self.controller.state = frame.state.copy()
                
                # Fire events if recorded
                if frame.events:
                    for event in frame.events:
                        self.controller._fire_event(event)
            else:
                # For real controllers, this would need different handling
                logger.debug(f"Playback frame {self.playback_index}")
            
            self.playback_index += 1
        
        logger.info("Playback loop ended")
    
    def _save_json(self, path: Path):
        """Save recording as JSON."""
        data = {
            'metadata': self.recording_metadata,
            'frames': []
        }
        
        for frame in self.recorded_frames:
            frame_data = {
                'timestamp': frame.timestamp,
                'state': {
                    'buttons': {btn.name: val for btn, val in frame.state.buttons.items()},
                    'axes': {axis.name: val for axis, val in frame.state.axes.items()}
                }
            }
            
            if frame.events:
                frame_data['events'] = [
                    {
                        'type': event.event_type,
                        'control': event.control.name if hasattr(event.control, 'name') else str(event.control),
                        'value': event.value,
                        'timestamp': event.timestamp
                    }
                    for event in frame.events
                ]
            
            data['frames'].append(frame_data)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_compressed(self, path: Path):
        """Save recording as compressed JSON."""
        # Create JSON data
        json_path = path.with_suffix('.json')
        self._save_json(json_path)
        
        # Compress
        with open(json_path, 'rb') as f_in:
            with gzip.open(path.with_suffix('.gz'), 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove temporary JSON
        json_path.unlink()
    
    def _save_binary(self, path: Path):
        """Save recording in binary format."""
        import struct
        import pickle
        
        with open(path, 'wb') as f:
            # Write header
            f.write(b'NXRC')  # Magic number
            f.write(struct.pack('<H', 1))  # Version
            
            # Write metadata
            metadata_bytes = pickle.dumps(self.recording_metadata)
            f.write(struct.pack('<I', len(metadata_bytes)))
            f.write(metadata_bytes)
            
            # Write frames
            f.write(struct.pack('<I', len(self.recorded_frames)))
            
            for frame in self.recorded_frames:
                # Serialize frame
                frame_bytes = pickle.dumps(frame)
                f.write(struct.pack('<I', len(frame_bytes)))
                f.write(frame_bytes)
    
    def _load_json(self, path: Path):
        """Load recording from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.recording_metadata = data['metadata']
        self.recorded_frames = []
        
        for frame_data in data['frames']:
            # Reconstruct state
            state = ControllerState()
            
            for btn_name, val in frame_data['state']['buttons'].items():
                state.buttons[Button[btn_name]] = val
            
            for axis_name, val in frame_data['state']['axes'].items():
                state.axes[Axis[axis_name]] = val
            
            # Reconstruct events
            events = None
            if 'events' in frame_data:
                events = []
                for event_data in frame_data['events']:
                    # Parse control
                    control_name = event_data['control']
                    if control_name in [b.name for b in Button]:
                        control = Button[control_name]
                    elif control_name in [a.name for a in Axis]:
                        control = Axis[control_name]
                    else:
                        control = control_name
                    
                    events.append(ControllerEvent(
                        event_type=event_data['type'],
                        control=control,
                        value=event_data['value'],
                        timestamp=event_data['timestamp']
                    ))
            
            frame = RecordedFrame(
                timestamp=frame_data['timestamp'],
                state=state,
                events=events
            )
            
            self.recorded_frames.append(frame)
    
    def _load_compressed(self, path: Path):
        """Load recording from compressed JSON."""
        with gzip.open(path, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
        
        # Use JSON loading logic
        temp_path = path.with_suffix('.json')
        with open(temp_path, 'w') as f:
            json.dump(data, f)
        
        self._load_json(temp_path)
        temp_path.unlink()
    
    def _load_binary(self, path: Path):
        """Load recording from binary format."""
        import struct
        import pickle
        
        with open(path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'NXRC':
                raise ValueError("Invalid binary recording file")
            
            version = struct.unpack('<H', f.read(2))[0]
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")
            
            # Read metadata
            metadata_len = struct.unpack('<I', f.read(4))[0]
            self.recording_metadata = pickle.loads(f.read(metadata_len))
            
            # Read frames
            frame_count = struct.unpack('<I', f.read(4))[0]
            self.recorded_frames = []
            
            for _ in range(frame_count):
                frame_len = struct.unpack('<I', f.read(4))[0]
                frame = pickle.loads(f.read(frame_len))
                self.recorded_frames.append(frame)


# Import VirtualGamepad for playback support
from nexus.input.gamepad.virtual_gamepad import VirtualGamepad