"""Visual Debugger for Nexus Framework - Integrated with API server"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np
import cv2
import base64
import weakref
import structlog

from nexus.core import get_logger
from nexus.core.exceptions import ResourceError

logger = get_logger("nexus.visual_debugger")


class DebugFrame:
    """Debug frame data structure"""
    
    def __init__(self, frame_id: int, timestamp: float, image: np.ndarray, 
                 detections: List[Dict] = None, ocr_results: List[Dict] = None,
                 metadata: Dict[str, Any] = None):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.image = image
        self.detections = detections or []
        self.ocr_results = ocr_results or []
        self.metadata = metadata or {}
        self.processing_time_ms = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        # Encode image as base64
        _, buffer = cv2.imencode('.jpg', self.image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "image": img_base64,
            "shape": self.image.shape,
            "detections": self.detections,
            "ocr_results": self.ocr_results,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms
        }


class DebugSession:
    """Debug session for tracking debug state"""
    
    def __init__(self, session_id: str, max_frames: int = 50):
        self.session_id = session_id
        self.max_frames = max_frames
        self.frames: deque = deque(maxlen=max_frames)
        self.active = True
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # Debug settings
        self.settings = {
            "show_detections": True,
            "show_ocr": True,
            "show_sprites": True,
            "capture_fps": 30,
            "detection_confidence_threshold": 0.5,
            "annotation_color": [0, 255, 0],  # Green
            "font_scale": 0.7
        }
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "total_ocr_results": 0,
            "avg_processing_time_ms": 0,
            "session_duration_s": 0
        }
    
    def add_frame(self, frame: DebugFrame):
        """Add frame to session"""
        self.frames.append(frame)
        self.last_activity = time.time()
        
        # Update statistics
        self.stats["frames_processed"] += 1
        self.stats["total_detections"] += len(frame.detections)
        self.stats["total_ocr_results"] += len(frame.ocr_results)
        
        # Update average processing time
        total_time = (self.stats["avg_processing_time_ms"] * (self.stats["frames_processed"] - 1) + 
                     frame.processing_time_ms)
        self.stats["avg_processing_time_ms"] = total_time / self.stats["frames_processed"]
        
        self.stats["session_duration_s"] = time.time() - self.created_at
    
    def get_latest_frame(self) -> Optional[DebugFrame]:
        """Get most recent frame"""
        return self.frames[-1] if self.frames else None
    
    def get_frame_history(self, count: int = 10) -> List[DebugFrame]:
        """Get recent frames"""
        return list(self.frames)[-count:]
    
    def is_expired(self, timeout_seconds: float = 3600) -> bool:
        """Check if session is expired"""
        return time.time() - self.last_activity > timeout_seconds


class VisualDebugger:
    """Integrated visual debugger for Nexus framework"""
    
    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, DebugSession] = {}
        self.global_frame_id = 0
        
        # WebSocket connections
        self.websocket_connections: List[weakref.ref] = []
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Performance monitoring
        self.performance_stats = {
            "total_frames_processed": 0,
            "total_processing_time_ms": 0,
            "peak_memory_mb": 0,
            "active_sessions": 0
        }
    
    async def start(self):
        """Start the visual debugger"""
        logger.info("Starting visual debugger")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
        
    async def stop(self):
        """Stop the visual debugger"""
        logger.info("Stopping visual debugger")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        for session in self.sessions.values():
            session.active = False
        
        self.sessions.clear()
    
    def create_session(self, session_id: str = None, max_frames: int = 50) -> str:
        """Create a new debug session"""
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())[:8]
        
        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            oldest_session_id = min(self.sessions.keys(), 
                                  key=lambda sid: self.sessions[sid].created_at)
            self.close_session(oldest_session_id)
        
        session = DebugSession(session_id, max_frames)
        self.sessions[session_id] = session
        
        logger.info(f"Created debug session: {session_id}")
        return session_id
    
    def close_session(self, session_id: str) -> bool:
        """Close a debug session"""
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            del self.sessions[session_id]
            logger.info(f"Closed debug session: {session_id}")
            return True
        return False
    
    def get_session(self, session_id: str) -> Optional[DebugSession]:
        """Get debug session by ID"""
        return self.sessions.get(session_id)
    
    async def process_frame(self, session_id: str, image: np.ndarray, 
                          detections: List[Dict] = None, ocr_results: List[Dict] = None,
                          metadata: Dict[str, Any] = None) -> DebugFrame:
        """Process a frame for debugging"""
        session = self.get_session(session_id)
        if not session:
            raise ResourceError("debug_session", f"Session {session_id} not found")
        
        start_time = time.time()
        
        # Create debug frame
        self.global_frame_id += 1
        debug_frame = DebugFrame(
            frame_id=self.global_frame_id,
            timestamp=time.time(),
            image=image.copy(),
            detections=detections or [],
            ocr_results=ocr_results or [],
            metadata=metadata or {}
        )
        
        # Add processing annotations if enabled
        if session.settings["show_detections"] or session.settings["show_ocr"]:
            debug_frame.image = self._create_annotated_frame(debug_frame, session.settings)
        
        debug_frame.processing_time_ms = (time.time() - start_time) * 1000
        
        # Add to session
        session.add_frame(debug_frame)
        
        # Update global stats
        self.performance_stats["total_frames_processed"] += 1
        self.performance_stats["total_processing_time_ms"] += debug_frame.processing_time_ms
        self.performance_stats["active_sessions"] = len([s for s in self.sessions.values() if s.active])
        
        # Broadcast to WebSocket connections
        await self._broadcast_frame_update(session_id, debug_frame)
        
        return debug_frame
    
    def _create_annotated_frame(self, debug_frame: DebugFrame, settings: Dict[str, Any]) -> np.ndarray:
        """Create annotated frame with overlays"""
        annotated = debug_frame.image.copy()
        color = tuple(settings["annotation_color"])
        font_scale = settings["font_scale"]
        
        # Draw detection boxes
        if settings["show_detections"]:
            for detection in debug_frame.detections:
                if detection.get("confidence", 0) >= settings["detection_confidence_threshold"]:
                    bbox = detection.get("bbox", [])
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{detection.get('class', 'object')}: {detection.get('confidence', 0):.2f}"
                        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, color, 2)
        
        # Draw OCR results
        if settings["show_ocr"]:
            for ocr_result in debug_frame.ocr_results:
                bbox = ocr_result.get("bbox", [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for text
                    
                    # Add text
                    text = ocr_result.get("text", "")
                    if text:
                        cv2.putText(annotated, text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, (0, 255, 255), 2)
        
        return annotated
    
    async def _broadcast_frame_update(self, session_id: str, debug_frame: DebugFrame):
        """Broadcast frame update to WebSocket connections"""
        message = {
            "type": "debug_frame",
            "session_id": session_id,
            "frame_data": debug_frame.to_dict()
        }
        
        # Clean up dead connections
        self.websocket_connections = [ref for ref in self.websocket_connections if ref() is not None]
        
        # Send to all active connections
        for conn_ref in self.websocket_connections:
            conn = conn_ref()
            if conn:
                try:
                    await conn.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Failed to send WebSocket message: {e}")
    
    def register_websocket(self, websocket) -> None:
        """Register a WebSocket connection for debug updates"""
        self.websocket_connections.append(weakref.ref(websocket))
        logger.debug(f"Registered WebSocket connection. Total: {len(self.websocket_connections)}")
    
    def update_session_settings(self, session_id: str, settings: Dict[str, Any]) -> bool:
        """Update session settings"""
        session = self.get_session(session_id)
        if session:
            session.settings.update(settings)
            return True
        return False
    
    def get_session_info(self, session_id: str = None) -> Dict[str, Any]:
        """Get session information"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return {
                    "session_id": session.session_id,
                    "created_at": session.created_at,
                    "last_activity": session.last_activity,
                    "active": session.active,
                    "settings": session.settings,
                    "stats": session.stats,
                    "frame_count": len(session.frames)
                }
            return {}
        else:
            # Return all sessions
            return {
                "sessions": {
                    sid: {
                        "session_id": session.session_id,
                        "created_at": session.created_at,
                        "last_activity": session.last_activity,
                        "active": session.active,
                        "frame_count": len(session.frames),
                        "stats": session.stats
                    }
                    for sid, session in self.sessions.items()
                },
                "global_stats": self.performance_stats
            }
    
    async def get_frame_data(self, session_id: str, frame_id: int = None) -> Optional[Dict[str, Any]]:
        """Get frame data for a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        if frame_id is None:
            # Return latest frame
            latest = session.get_latest_frame()
            return latest.to_dict() if latest else None
        else:
            # Find specific frame
            for frame in reversed(session.frames):
                if frame.frame_id == frame_id:
                    return frame.to_dict()
            return None
    
    async def get_frame_history(self, session_id: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get frame history for a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        frames = session.get_frame_history(count)
        return [frame.to_dict() for frame in frames]
    
    async def _cleanup_sessions(self):
        """Periodically cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                expired_sessions = [
                    sid for sid, session in self.sessions.items()
                    if session.is_expired()
                ]
                
                for session_id in expired_sessions:
                    self.close_session(session_id)
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired debug sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.performance_stats["peak_memory_mb"]:
            self.performance_stats["peak_memory_mb"] = memory_mb
        
        return {
            **self.performance_stats,
            "current_memory_mb": memory_mb,
            "active_websocket_connections": len([ref for ref in self.websocket_connections if ref() is not None]),
            "total_sessions": len(self.sessions)
        }


# Global debugger instance
_global_debugger: Optional[VisualDebugger] = None


def get_debugger() -> VisualDebugger:
    """Get global visual debugger instance"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = VisualDebugger()
    return _global_debugger


async def debug_frame(image: np.ndarray, session_id: str = "default", 
                     detections: List[Dict] = None, ocr_results: List[Dict] = None,
                     metadata: Dict[str, Any] = None) -> DebugFrame:
    """Convenience function to debug a frame"""
    debugger = get_debugger()
    
    # Create session if it doesn't exist
    if not debugger.get_session(session_id):
        debugger.create_session(session_id)
    
    return await debugger.process_frame(session_id, image, detections, ocr_results, metadata)


def create_debug_session(session_id: str = None, max_frames: int = 50) -> str:
    """Create a new debug session"""
    debugger = get_debugger()
    return debugger.create_session(session_id, max_frames)