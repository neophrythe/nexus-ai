import asyncio
import time
from datetime import datetime
from typing import Optional, Tuple, List, Any
import numpy as np
import structlog

from nexus.capture.base import CaptureBackend, Frame, CaptureError

logger = structlog.get_logger()


class DXCamBackend(CaptureBackend):
    
    def __init__(self, device_idx: int = 0, output_idx: Optional[int] = None):
        super().__init__(device_idx, output_idx)
        self.camera = None
        self._capture_task = None
        self._frame_buffer = []
        self._last_frame_time = 0
        
        try:
            import sys
            sys.path.append('/mnt/c/Users/neoph/Desktop/GAMEAI/DXcam')
            import dxcam
            self.dxcam = dxcam
        except ImportError:
            raise CaptureError("DXCam not available. Please ensure DXcam is installed.")
    
    async def initialize(self) -> None:
        try:
            self.camera = self.dxcam.create(
                device_idx=self.device_idx,
                output_idx=self.output_idx,
                output_color="RGB"
            )
            
            if not self.camera:
                raise CaptureError("Failed to create DXCam camera instance")
            
            logger.info(f"DXCam initialized - Device: {self.device_idx}, Output: {self.output_idx}")
            
        except Exception as e:
            raise CaptureError(f"Failed to initialize DXCam: {e}")
    
    async def capture(self) -> Optional[Frame]:
        if not self.camera:
            await self.initialize()
        
        try:
            start_time = time.perf_counter()
            
            if self._fps_limit:
                min_frame_time = 1.0 / self._fps_limit
                time_since_last = time.perf_counter() - self._last_frame_time
                if time_since_last < min_frame_time:
                    await asyncio.sleep(min_frame_time - time_since_last)
            
            if self._region:
                x, y, w, h = self._region
                region = (x, y, x + w, y + h)
            else:
                region = None
            
            frame_data = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.camera.grab(region=region)
            )
            
            if frame_data is None:
                return None
            
            capture_time = (time.perf_counter() - start_time) * 1000
            
            self.frame_count += 1
            self._last_frame_time = time.perf_counter()
            
            frame = Frame(
                data=np.array(frame_data),
                timestamp=datetime.now(),
                frame_id=self.frame_count,
                capture_time_ms=capture_time,
                region=self._region,
                metadata={
                    "backend": "dxcam",
                    "device_idx": self.device_idx,
                    "output_idx": self.output_idx
                }
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
    
    async def start_capture(self) -> None:
        if self.is_capturing:
            logger.warning("Capture already started")
            return
        
        if not self.camera:
            await self.initialize()
        
        try:
            if self._region:
                x, y, w, h = self._region
                region = (x, y, x + w, y + h)
            else:
                region = None
            
            target_fps = self._fps_limit or 60
            
            self.camera.start(region=region, target_fps=target_fps)
            self.is_capturing = True
            
            self._capture_task = asyncio.create_task(self._capture_loop())
            
            logger.info(f"Started continuous capture at {target_fps} FPS")
            
        except Exception as e:
            raise CaptureError(f"Failed to start capture: {e}")
    
    async def stop_capture(self) -> None:
        if not self.is_capturing:
            logger.warning("Capture not started")
            return
        
        try:
            if self._capture_task:
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    pass
            
            if self.camera:
                self.camera.stop()
            
            self.is_capturing = False
            logger.info("Stopped capture")
            
        except Exception as e:
            logger.error(f"Error stopping capture: {e}")
    
    async def _capture_loop(self) -> None:
        while self.is_capturing:
            try:
                start_time = time.perf_counter()
                
                frame_data = self.camera.get_latest_frame()
                
                if frame_data is not None:
                    capture_time = (time.perf_counter() - start_time) * 1000
                    self.frame_count += 1
                    
                    frame = Frame(
                        data=np.array(frame_data),
                        timestamp=datetime.now(),
                        frame_id=self.frame_count,
                        capture_time_ms=capture_time,
                        region=self._region,
                        metadata={
                            "backend": "dxcam",
                            "continuous": True
                        }
                    )
                    
                    self._frame_buffer.append(frame)
                    if len(self._frame_buffer) > 64:
                        self._frame_buffer.pop(0)
                
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                await asyncio.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[Frame]:
        if self._frame_buffer:
            return self._frame_buffer[-1]
        return None
    
    def get_screen_size(self) -> Tuple[int, int]:
        try:
            info = self.dxcam.output_info()
            
            for line in info.split('\n'):
                if 'Res:' in line and 'Primary:True' in line:
                    res_part = line.split('Res:')[1].split()[0]
                    res = res_part.strip('()').split(',')
                    return (int(res[0]), int(res[1]))
            
            return (1920, 1080)
            
        except Exception as e:
            logger.error(f"Failed to get screen size: {e}")
            return (1920, 1080)
    
    def get_available_outputs(self) -> List[dict]:
        try:
            info = self.dxcam.output_info()
            outputs = []
            
            for line in info.split('\n'):
                if 'Device[' in line and 'Output[' in line:
                    parts = line.split()
                    device_idx = int(line.split('Device[')[1].split(']')[0])
                    output_idx = int(line.split('Output[')[1].split(']')[0].replace(':', ''))
                    
                    res_part = line.split('Res:')[1].split()[0]
                    res = res_part.strip('()').split(',')
                    
                    is_primary = 'Primary:True' in line
                    
                    outputs.append({
                        'device_idx': device_idx,
                        'output_idx': output_idx,
                        'resolution': (int(res[0]), int(res[1])),
                        'primary': is_primary
                    })
            
            return outputs
            
        except Exception as e:
            logger.error(f"Failed to get available outputs: {e}")
            return []
    
    async def cleanup(self) -> None:
        await super().cleanup()
        if self.camera:
            del self.camera
            self.camera = None