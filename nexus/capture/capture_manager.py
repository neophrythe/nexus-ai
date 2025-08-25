import asyncio
from typing import Optional, Dict, Any, List, Tuple
import structlog
from datetime import datetime
import psutil

from nexus.capture.base import (
    CaptureBackend, CaptureBackendType, Frame, FrameBuffer, CaptureError
)
from nexus.capture.dxcam_backend import DXCamBackend

logger = structlog.get_logger()


class CaptureManager:
    
    BACKEND_CLASSES = {
        CaptureBackendType.DXCAM: DXCamBackend,
    }
    
    def __init__(self, 
                 backend_type: CaptureBackendType = CaptureBackendType.DXCAM,
                 device_idx: int = 0,
                 output_idx: Optional[int] = None,
                 buffer_size: int = 64):
        
        self.backend_type = backend_type
        self.device_idx = device_idx
        self.output_idx = output_idx
        self.buffer_size = buffer_size
        
        self.backend: Optional[CaptureBackend] = None
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        
        self._stats = {
            "total_frames": 0,
            "dropped_frames": 0,
            "avg_capture_time": 0,
            "min_capture_time": float('inf'),
            "max_capture_time": 0,
            "current_fps": 0,
            "start_time": None
        }
        
        self._capture_times = []
        self._last_fps_update = datetime.now()
        self._fps_frame_count = 0
    
    async def initialize(self) -> None:
        try:
            backend_class = self.BACKEND_CLASSES.get(self.backend_type)
            if not backend_class:
                raise CaptureError(f"Unsupported backend: {self.backend_type}")
            
            self.backend = backend_class(self.device_idx, self.output_idx)
            await self.backend.initialize()
            
            self._stats["start_time"] = datetime.now()
            
            logger.info(f"Capture manager initialized with {self.backend_type.value} backend")
            
        except Exception as e:
            raise CaptureError(f"Failed to initialize capture manager: {e}")
    
    async def capture_frame(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Frame]:
        if not self.backend:
            await self.initialize()
        
        if region:
            self.backend.set_region(*region)
        
        frame = await self.backend.capture()
        
        if frame:
            self.frame_buffer.add(frame)
            self._update_stats(frame)
        else:
            self._stats["dropped_frames"] += 1
        
        return frame
    
    async def start_continuous_capture(self, fps: int = 60, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        if not self.backend:
            await self.initialize()
        
        if region:
            self.backend.set_region(*region)
        
        self.backend.set_fps_limit(fps)
        await self.backend.start_capture()
        
        logger.info(f"Started continuous capture at {fps} FPS")
    
    async def stop_continuous_capture(self) -> None:
        if self.backend:
            await self.backend.stop_capture()
    
    def get_latest_frame(self) -> Optional[Frame]:
        frames = self.frame_buffer.get_latest(1)
        return frames[0] if frames else None
    
    def get_frame_buffer(self, n: int = 10) -> List[Frame]:
        return self.frame_buffer.get_latest(n)
    
    def set_region_of_interest(self, x: int, y: int, width: int, height: int) -> None:
        if self.backend:
            self.backend.set_region(x, y, width, height)
    
    def clear_region_of_interest(self) -> None:
        if self.backend:
            self.backend.clear_region()
    
    def _update_stats(self, frame: Frame) -> None:
        self._stats["total_frames"] += 1
        
        self._capture_times.append(frame.capture_time_ms)
        if len(self._capture_times) > 100:
            self._capture_times.pop(0)
        
        self._stats["avg_capture_time"] = sum(self._capture_times) / len(self._capture_times)
        self._stats["min_capture_time"] = min(self._stats["min_capture_time"], frame.capture_time_ms)
        self._stats["max_capture_time"] = max(self._stats["max_capture_time"], frame.capture_time_ms)
        
        self._fps_frame_count += 1
        now = datetime.now()
        time_diff = (now - self._last_fps_update).total_seconds()
        
        if time_diff >= 1.0:
            self._stats["current_fps"] = self._fps_frame_count / time_diff
            self._fps_frame_count = 0
            self._last_fps_update = now
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        
        if stats["start_time"]:
            uptime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["uptime_seconds"] = uptime
            stats["avg_fps"] = stats["total_frames"] / uptime if uptime > 0 else 0
        
        process = psutil.Process()
        stats["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
        stats["cpu_percent"] = process.cpu_percent()
        
        return stats
    
    def get_screen_info(self) -> Dict[str, Any]:
        if not self.backend:
            return {}
        
        width, height = self.backend.get_screen_size()
        outputs = self.backend.get_available_outputs()
        
        return {
            "primary_resolution": (width, height),
            "outputs": outputs,
            "backend": self.backend_type.value
        }
    
    async def benchmark(self, duration: int = 10, region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        logger.info(f"Starting {duration}s benchmark...")
        
        if not self.backend:
            await self.initialize()
        
        if region:
            self.backend.set_region(*region)
        
        start_time = asyncio.get_event_loop().time()
        frame_count = 0
        capture_times = []
        
        while asyncio.get_event_loop().time() - start_time < duration:
            frame = await self.backend.capture()
            if frame:
                frame_count += 1
                capture_times.append(frame.capture_time_ms)
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        results = {
            "duration": elapsed,
            "total_frames": frame_count,
            "avg_fps": frame_count / elapsed,
            "avg_capture_time_ms": sum(capture_times) / len(capture_times) if capture_times else 0,
            "min_capture_time_ms": min(capture_times) if capture_times else 0,
            "max_capture_time_ms": max(capture_times) if capture_times else 0,
            "region": region,
            "backend": self.backend_type.value
        }
        
        logger.info(f"Benchmark complete: {results['avg_fps']:.2f} FPS, {results['avg_capture_time_ms']:.2f}ms avg capture time")
        
        return results
    
    async def cleanup(self) -> None:
        if self.backend:
            await self.backend.cleanup()
            self.backend = None
        
        self.frame_buffer.clear()
        logger.info("Capture manager cleaned up")