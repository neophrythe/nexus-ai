import asyncio
import time
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np
import mss
import structlog

from nexus.capture.base import CaptureBackend, Frame, CaptureError

logger = structlog.get_logger()


class MSSBackend(CaptureBackend):
    """MSS (Multi-Screen Shot) backend for cross-platform capture"""
    
    def __init__(self, device_idx: int = 0, output_idx: Optional[int] = None):
        super().__init__(device_idx, output_idx)
        self.sct = None
        self.monitors = []
        self._last_frame_time = 0
        
    async def initialize(self) -> None:
        """Initialize MSS backend"""
        try:
            self.sct = mss.mss()
            self.monitors = self.sct.monitors
            
            # Select monitor (0 = all monitors, 1+ = specific monitor)
            self.monitor_idx = self.output_idx if self.output_idx else 1
            
            if self.monitor_idx >= len(self.monitors):
                self.monitor_idx = 1
            
            logger.info(f"MSS initialized - Monitor: {self.monitor_idx}, Total monitors: {len(self.monitors) - 1}")
            
        except Exception as e:
            raise CaptureError(f"Failed to initialize MSS: {e}")
    
    async def capture(self) -> Optional[Frame]:
        """Capture a frame"""
        if not self.sct:
            await self.initialize()
        
        try:
            start_time = time.perf_counter()
            
            # Apply FPS limit
            if self._fps_limit:
                min_frame_time = 1.0 / self._fps_limit
                time_since_last = time.perf_counter() - self._last_frame_time
                if time_since_last < min_frame_time:
                    await asyncio.sleep(min_frame_time - time_since_last)
            
            # Determine capture region
            if self._region:
                x, y, w, h = self._region
                monitor = {"left": x, "top": y, "width": w, "height": h}
            else:
                monitor = self.monitors[self.monitor_idx]
            
            # Capture screenshot
            screenshot = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.sct.grab(monitor)
            )
            
            # Convert to numpy array (BGRA to RGB)
            frame_data = np.array(screenshot)
            frame_data = frame_data[:, :, [2, 1, 0]]  # BGRA to RGB
            
            capture_time = (time.perf_counter() - start_time) * 1000
            
            self.frame_count += 1
            self._last_frame_time = time.perf_counter()
            
            frame = Frame(
                data=frame_data,
                timestamp=datetime.now(),
                frame_id=self.frame_count,
                capture_time_ms=capture_time,
                region=self._region,
                metadata={
                    "backend": "mss",
                    "monitor": self.monitor_idx
                }
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"MSS capture failed: {e}")
            return None
    
    async def start_capture(self) -> None:
        """Start continuous capture"""
        if self.is_capturing:
            logger.warning("Capture already started")
            return
        
        if not self.sct:
            await self.initialize()
        
        self.is_capturing = True
        logger.info("MSS continuous capture started")
    
    async def stop_capture(self) -> None:
        """Stop continuous capture"""
        if not self.is_capturing:
            logger.warning("Capture not started")
            return
        
        self.is_capturing = False
        logger.info("MSS capture stopped")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        if not self.sct:
            self.sct = mss.mss()
            self.monitors = self.sct.monitors
        
        monitor = self.monitors[1]  # Primary monitor
        return (monitor["width"], monitor["height"])
    
    def get_available_outputs(self) -> List[dict]:
        """Get available monitors"""
        if not self.sct:
            self.sct = mss.mss()
            self.monitors = self.sct.monitors
        
        outputs = []
        for idx, monitor in enumerate(self.monitors[1:], 1):  # Skip "all monitors"
            outputs.append({
                'device_idx': 0,
                'output_idx': idx,
                'resolution': (monitor["width"], monitor["height"]),
                'position': (monitor["left"], monitor["top"]),
                'primary': idx == 1
            })
        
        return outputs
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        await super().cleanup()
        if self.sct:
            self.sct.close()
            self.sct = None