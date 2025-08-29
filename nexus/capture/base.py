from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Any, Dict
import numpy as np


class CaptureBackend(Enum):
    DXCAM = "dxcam"
    MSS = "mss"
    WINDOWS_GRAPHICS = "windows_graphics"
    OBS_VIRTUAL = "obs_virtual"
    WEBRTC = "webrtc"


class CaptureError(Exception):
    """Exception raised for capture-related errors"""
    def __init__(self, message="Capture error occurred", *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.message = message


@dataclass
class Frame:
    data: np.ndarray
    timestamp: datetime
    frame_id: int
    capture_time_ms: float
    region: Optional[Tuple[int, int, int, int]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    def to_rgb(self) -> np.ndarray:
        if len(self.data.shape) == 2:
            return np.stack([self.data] * 3, axis=-1)
        elif self.data.shape[2] == 4:
            return self.data[:, :, :3]
        return self.data
    
    def to_bgr(self) -> np.ndarray:
        rgb = self.to_rgb()
        return rgb[:, :, ::-1]


class CaptureBackendBase(ABC):
    
    def __init__(self, device_idx: int = 0, output_idx: Optional[int] = None):
        self.device_idx = device_idx
        self.output_idx = output_idx
        self.is_capturing = False
        self.frame_count = 0
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._fps_limit: Optional[int] = None
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize capture backend"""
        self.is_initialized = True
        self.start_time = time.time()
        logger.info(f"Capture backend {self.name} initialized")
    
    @abstractmethod
    async def capture(self) -> Optional[Frame]:
        """Capture a single frame"""
        import mss
        import numpy as np
        from datetime import datetime
        
        try:
            with mss.mss() as sct:
                if self._region:
                    monitor = {
                        "left": self._region[0],
                        "top": self._region[1],
                        "width": self._region[2],
                        "height": self._region[3]
                    }
                else:
                    monitor = sct.monitors[1]
                
                screenshot = sct.grab(monitor)
                frame_data = np.array(screenshot)[:, :, :3]  # Remove alpha
                
                return Frame(
                    data=frame_data,
                    timestamp=datetime.now(),
                    frame_number=self.frame_count,
                    metadata={}
                )
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
    
    @abstractmethod
    async def start_capture(self) -> None:
        """Start continuous capture"""
        self.is_capturing = True
        self.capture_start_time = time.time()
        logger.info(f"Started capture on {self.name}")
    
    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop continuous capture"""
        self.is_capturing = False
        if hasattr(self, 'capture_start_time'):
            duration = time.time() - self.capture_start_time
            logger.info(f"Stopped capture on {self.name} after {duration:.2f}s")
    
    @abstractmethod
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            return (monitor['width'], monitor['height'])
    
    @abstractmethod
    def get_available_outputs(self) -> list:
        """Get list of available capture outputs/monitors"""
        import mss
        outputs = []
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                outputs.append({
                    'id': i,
                    'name': f"Monitor {i}",
                    'x': monitor.get('left', 0),
                    'y': monitor.get('top', 0),
                    'width': monitor.get('width', 0),
                    'height': monitor.get('height', 0),
                    'is_primary': i == 1
                })
        return outputs
    
    def set_region(self, x: int, y: int, width: int, height: int) -> None:
        self._region = (x, y, width, height)
    
    def clear_region(self) -> None:
        self._region = None
    
    def set_fps_limit(self, fps: int) -> None:
        self._fps_limit = fps
    
    async def cleanup(self) -> None:
        if self.is_capturing:
            await self.stop_capture()


class FrameBuffer:
    
    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self.frames: list[Frame] = []
        self._write_idx = 0
        
    def add(self, frame: Frame) -> None:
        if len(self.frames) < self.max_size:
            self.frames.append(frame)
        else:
            self.frames[self._write_idx] = frame
            self._write_idx = (self._write_idx + 1) % self.max_size
    
    def get_latest(self, n: int = 1) -> list[Frame]:
        if n >= len(self.frames):
            return self.frames.copy()
        return self.frames[-n:]
    
    def clear(self) -> None:
        self.frames.clear()
        self._write_idx = 0
    
    @property
    def size(self) -> int:
        return len(self.frames)