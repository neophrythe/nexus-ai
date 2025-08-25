from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import json
import base64
import io
from datetime import datetime
import structlog
import cv2
import numpy as np

from nexus.core import PluginManager, ConfigManager, get_logger
from nexus.capture import CaptureManager, CaptureBackendType
from nexus.agents.base import BaseAgent
from nexus.environments.base import GameEnvironment

logger = get_logger("nexus.api")


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


class NexusAPI:
    """Core API server"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.app = FastAPI(title="Nexus Game Automation API", version="0.1.0")
        self.connection_manager = ConnectionManager()
        
        # Core components
        self.plugin_manager: Optional[PluginManager] = None
        self.capture_manager: Optional[CaptureManager] = None
        self.current_agent: Optional[BaseAgent] = None
        self.current_environment: Optional[GameEnvironment] = None
        
        # Stream control
        self.stream_active = False
        self.stream_task: Optional[asyncio.Task] = None
        
        self._setup_cors()
        self._setup_routes()
        self._setup_websocket()
    
    def _setup_cors(self):
        """Setup CORS middleware"""
        if self.config.get("api.cors_enabled", True):
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "Nexus Game Automation API", "version": "0.1.0"}
        
        @self.app.get("/status")
        async def get_status():
            return {
                "status": "online",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "plugin_manager": self.plugin_manager is not None,
                    "capture_manager": self.capture_manager is not None,
                    "agent": self.current_agent is not None,
                    "environment": self.current_environment is not None,
                    "stream": self.stream_active
                }
            }
        
        @self.app.get("/config")
        async def get_config():
            return self.config.to_dict()
        
        @self.app.post("/config")
        async def update_config(updates: Dict[str, Any]):
            for key, value in updates.items():
                self.config.set(key, value)
            self.config.save()
            return {"status": "updated"}
        
        # Plugin routes
        @self.app.get("/plugins")
        async def list_plugins():
            if not self.plugin_manager:
                await self._initialize_plugin_manager()
            
            await self.plugin_manager.discover_plugins()
            return {
                "available": list(self.plugin_manager.plugin_manifests.keys()),
                "loaded": self.plugin_manager.list_plugins()
            }
        
        @self.app.post("/plugins/{plugin_name}/load")
        async def load_plugin(plugin_name: str, config: Optional[Dict[str, Any]] = None):
            if not self.plugin_manager:
                await self._initialize_plugin_manager()
            
            try:
                plugin = await self.plugin_manager.load_plugin(plugin_name, config)
                return {"status": "loaded", "info": plugin.get_info()}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/plugins/{plugin_name}/unload")
        async def unload_plugin(plugin_name: str):
            if not self.plugin_manager:
                return {"error": "Plugin manager not initialized"}
            
            await self.plugin_manager.unload_plugin(plugin_name)
            return {"status": "unloaded"}
        
        @self.app.post("/plugins/{plugin_name}/reload")
        async def reload_plugin(plugin_name: str):
            if not self.plugin_manager:
                await self._initialize_plugin_manager()
            
            try:
                plugin = await self.plugin_manager.reload_plugin(plugin_name)
                return {"status": "reloaded", "info": plugin.get_info()}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Capture routes
        @self.app.get("/capture/info")
        async def get_capture_info():
            if not self.capture_manager:
                await self._initialize_capture_manager()
            
            return self.capture_manager.get_screen_info()
        
        @self.app.get("/capture/stats")
        async def get_capture_stats():
            if not self.capture_manager:
                return {"error": "Capture manager not initialized"}
            
            return self.capture_manager.get_stats()
        
        @self.app.post("/capture/frame")
        async def capture_frame(region: Optional[List[int]] = None):
            if not self.capture_manager:
                await self._initialize_capture_manager()
            
            region_tuple = tuple(region) if region else None
            frame = await self.capture_manager.capture_frame(region_tuple)
            
            if frame:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame.to_bgr())
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp.isoformat(),
                    "capture_time_ms": frame.capture_time_ms,
                    "shape": frame.shape,
                    "image": img_base64
                }
            
            return {"error": "Failed to capture frame"}
        
        @self.app.post("/capture/stream/start")
        async def start_stream(fps: int = 30):
            if self.stream_active:
                return {"status": "already_streaming"}
            
            if not self.capture_manager:
                await self._initialize_capture_manager()
            
            self.stream_active = True
            self.stream_task = asyncio.create_task(self._stream_frames(fps))
            return {"status": "stream_started", "fps": fps}
        
        @self.app.post("/capture/stream/stop")
        async def stop_stream():
            if not self.stream_active:
                return {"status": "not_streaming"}
            
            self.stream_active = False
            if self.stream_task:
                self.stream_task.cancel()
            
            return {"status": "stream_stopped"}
        
        # Agent routes
        @self.app.get("/agents")
        async def list_agents():
            if not self.plugin_manager:
                await self._initialize_plugin_manager()
            
            from nexus.core.base import PluginType
            agents = self.plugin_manager.get_plugins_by_type(PluginType.AGENT)
            
            return {
                "agents": [agent.get_info() for agent in agents],
                "current": self.current_agent.name if self.current_agent else None
            }
        
        @self.app.post("/agents/{agent_name}/activate")
        async def activate_agent(agent_name: str):
            if not self.plugin_manager:
                await self._initialize_plugin_manager()
            
            agent = self.plugin_manager.get_plugin(agent_name)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            self.current_agent = agent
            return {"status": "activated", "agent": agent_name}
        
        # Training routes
        @self.app.post("/training/start")
        async def start_training(config: Dict[str, Any], background_tasks: BackgroundTasks):
            if not self.current_agent:
                raise HTTPException(status_code=400, detail="No agent selected")
            
            if not self.current_environment:
                raise HTTPException(status_code=400, detail="No environment loaded")
            
            from nexus.training import Trainer, TrainingConfig
            
            training_config = TrainingConfig(**config)
            trainer = Trainer(self.current_agent, self.current_environment, training_config)
            
            background_tasks.add_task(trainer.train)
            
            return {"status": "training_started", "config": config}
        
        # Visual Debugger routes
        @self.app.post("/debug/session")
        async def create_debug_session(session_id: str = None, max_frames: int = 50):
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            session_id = debugger.create_session(session_id, max_frames)
            
            return {"session_id": session_id}
        
        @self.app.delete("/debug/session/{session_id}")
        async def close_debug_session(session_id: str):
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            success = debugger.close_session(session_id)
            
            return {"success": success}
        
        @self.app.get("/debug/sessions")
        async def get_debug_sessions():
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            return debugger.get_session_info()
        
        @self.app.get("/debug/session/{session_id}")
        async def get_debug_session_info(session_id: str):
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            return debugger.get_session_info(session_id)
        
        @self.app.post("/debug/session/{session_id}/settings")
        async def update_debug_settings(session_id: str, settings: Dict[str, Any]):
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            success = debugger.update_session_settings(session_id, settings)
            
            return {"success": success}
        
        @self.app.get("/debug/session/{session_id}/frame")
        async def get_debug_frame(session_id: str, frame_id: int = None):
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            frame_data = await debugger.get_frame_data(session_id, frame_id)
            
            if frame_data:
                return frame_data
            else:
                raise HTTPException(status_code=404, detail="Frame not found")
        
        @self.app.get("/debug/session/{session_id}/history")
        async def get_debug_history(session_id: str, count: int = 10):
            from nexus.visual_debugger import get_debugger
            
            debugger = get_debugger()
            frames = await debugger.get_frame_history(session_id, count)
            
            return {"frames": frames}
        
        @self.app.post("/debug/frame")
        async def debug_frame_endpoint(
            session_id: str = "default",
            region: Optional[List[int]] = None
        ):
            """Capture and debug a frame"""
            if not self.capture_manager:
                await self._initialize_capture_manager()
            
            # Capture frame
            region_tuple = tuple(region) if region else None
            frame = await self.capture_manager.capture_frame(region_tuple)
            
            if frame:
                from nexus.visual_debugger import debug_frame
                
                # Debug the frame (this will process it through vision pipeline if available)
                debug_result = await debug_frame(
                    frame.to_rgb(),
                    session_id=session_id,
                    metadata={"capture_time_ms": frame.capture_time_ms}
                )
                
                return debug_result.to_dict()
            
            raise HTTPException(status_code=500, detail="Failed to capture frame")
        
        # Input Recording/Playback routes
        @self.app.post("/input/recorder/start")
        async def start_input_recording(enable_frame_sync: bool = False):
            """Start input recording"""
            from nexus.input import InputRecorder
            
            if not hasattr(self, 'input_recorder') or self.input_recorder is None:
                self.input_recorder = InputRecorder(
                    storage_backend="hybrid",
                    redis_config={"host": "127.0.0.1", "port": 6379, "db": 0}
                )
            
            success = self.input_recorder.start_recording(enable_frame_sync)
            return {"success": success, "message": "Recording started" if success else "Failed to start recording"}
        
        @self.app.post("/input/recorder/stop")
        async def stop_input_recording():
            """Stop input recording"""
            if hasattr(self, 'input_recorder') and self.input_recorder:
                success = self.input_recorder.stop_recording()
                return {"success": success, "message": "Recording stopped" if success else "Failed to stop recording"}
            
            return {"success": False, "message": "No active recorder"}
        
        @self.app.post("/input/recorder/pause")
        async def pause_input_recording():
            """Pause input recording"""
            if hasattr(self, 'input_recorder') and self.input_recorder:
                success = self.input_recorder.pause_recording()
                return {"success": success, "message": "Recording paused" if success else "Failed to pause recording"}
            
            return {"success": False, "message": "No active recorder"}
        
        @self.app.post("/input/recorder/resume")
        async def resume_input_recording():
            """Resume input recording"""
            if hasattr(self, 'input_recorder') and self.input_recorder:
                success = self.input_recorder.resume_recording()
                return {"success": success, "message": "Recording resumed" if success else "Failed to resume recording"}
            
            return {"success": False, "message": "No active recorder"}
        
        @self.app.get("/input/recorder/stats")
        async def get_recorder_stats():
            """Get recorder statistics"""
            if hasattr(self, 'input_recorder') and self.input_recorder:
                return self.input_recorder.get_stats()
            
            return {"error": "No active recorder"}
        
        @self.app.post("/input/recorder/save")
        async def save_recording(filepath: str, format: str = "json"):
            """Save current recording"""
            if hasattr(self, 'input_recorder') and self.input_recorder:
                success = self.input_recorder.save_recording(filepath, format)
                return {"success": success, "message": f"Recording saved to {filepath}" if success else "Failed to save recording"}
            
            return {"success": False, "message": "No active recorder"}
        
        @self.app.post("/input/recorder/load")
        async def load_recording(filepath: str):
            """Load recording for playback"""
            from nexus.input import InputRecorder
            
            if not hasattr(self, 'input_recorder') or self.input_recorder is None:
                self.input_recorder = InputRecorder()
            
            success = self.input_recorder.load_recording(filepath)
            return {"success": success, "message": f"Recording loaded from {filepath}" if success else "Failed to load recording"}
        
        @self.app.get("/input/recorder/events")
        async def get_recorded_events(
            start_time: Optional[float] = None,
            end_time: Optional[float] = None,
            event_type: Optional[str] = None
        ):
            """Get recorded events with filtering"""
            if hasattr(self, 'input_recorder') and self.input_recorder:
                from nexus.input import EventType
                
                filter_type = None
                if event_type:
                    try:
                        filter_type = EventType(event_type.lower())
                    except ValueError:
                        pass
                
                events = self.input_recorder.get_events(start_time, end_time, filter_type)
                return {"events": [event.to_dict() for event in events]}
            
            return {"error": "No active recorder"}
        
        @self.app.post("/input/playback/start")
        async def start_input_playback(config: Optional[Dict[str, Any]] = None):
            """Start input playback"""
            from nexus.input import InputPlayback, PlaybackConfig
            
            if not hasattr(self, 'input_playback') or self.input_playback is None:
                self.input_playback = InputPlayback()
            
            # Load events from recorder if available
            if hasattr(self, 'input_recorder') and self.input_recorder:
                events = self.input_recorder.get_events()
                if not self.input_playback.load_events(events):
                    return {"success": False, "message": "Failed to load events for playback"}
            
            # Apply config if provided
            if config:
                playback_config = PlaybackConfig(**config)
                self.input_playback.config = playback_config
            
            success = self.input_playback.start_playback()
            return {"success": success, "message": "Playback started" if success else "Failed to start playback"}
        
        @self.app.post("/input/playback/stop")
        async def stop_input_playback():
            """Stop input playback"""
            if hasattr(self, 'input_playback') and self.input_playback:
                success = self.input_playback.stop_playback()
                return {"success": success, "message": "Playback stopped" if success else "Failed to stop playback"}
            
            return {"success": False, "message": "No active playback"}
        
        @self.app.post("/input/playback/pause")
        async def pause_input_playback():
            """Pause input playback"""
            if hasattr(self, 'input_playback') and self.input_playback:
                success = self.input_playback.pause_playback()
                return {"success": success, "message": "Playback paused" if success else "Failed to pause playback"}
            
            return {"success": False, "message": "No active playback"}
        
        @self.app.post("/input/playback/resume")
        async def resume_input_playback():
            """Resume input playback"""
            if hasattr(self, 'input_playback') and self.input_playback:
                success = self.input_playback.resume_playback()
                return {"success": success, "message": "Playback resumed" if success else "Failed to resume playback"}
            
            return {"success": False, "message": "No active playback"}
        
        @self.app.post("/input/playback/seek")
        async def seek_playback(position: float):
            """Seek to position in playback (0.0-1.0)"""
            if hasattr(self, 'input_playback') and self.input_playback:
                success = self.input_playback.seek_to_position(position)
                return {"success": success, "message": f"Seeked to {position:.1%}" if success else "Failed to seek"}
            
            return {"success": False, "message": "No active playback"}
        
        @self.app.post("/input/playback/speed")
        async def set_playback_speed(speed: float):
            """Set playback speed"""
            if hasattr(self, 'input_playback') and self.input_playback:
                success = self.input_playback.set_speed(speed)
                return {"success": success, "message": f"Speed set to {speed}x" if success else "Failed to set speed"}
            
            return {"success": False, "message": "No active playback"}
        
        @self.app.get("/input/playback/progress")
        async def get_playback_progress():
            """Get playback progress"""
            if hasattr(self, 'input_playback') and self.input_playback:
                return self.input_playback.get_progress()
            
            return {"error": "No active playback"}
        
        @self.app.get("/input/playback/stats")
        async def get_playback_stats():
            """Get playback statistics"""
            if hasattr(self, 'input_playback') and self.input_playback:
                return self.input_playback.get_stats()
            
            return {"error": "No active playback"}
        
        # Performance routes
        @self.app.get("/performance")
        async def get_performance():
            import psutil
            
            process = psutil.Process()
            
            performance = {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
                "capture_stats": self.capture_manager.get_stats() if self.capture_manager else None
            }
            
            # Add debug performance stats
            try:
                from nexus.visual_debugger import get_debugger
                debugger = get_debugger()
                performance["debug_stats"] = debugger.get_performance_stats()
            except Exception:
                pass
            
            return performance
    
    def _setup_websocket(self):
        """Setup WebSocket endpoints"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connection_manager.connect(websocket)
            
            # Register with visual debugger for debug updates
            try:
                from nexus.visual_debugger import get_debugger
                debugger = get_debugger()
                debugger.register_websocket(websocket)
            except Exception as e:
                logger.error(f"Failed to register WebSocket with debugger: {e}")
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message["type"] == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                    elif message["type"] == "capture":
                        if self.capture_manager:
                            frame = await self.capture_manager.capture_frame()
                            if frame:
                                _, buffer = cv2.imencode('.jpg', frame.to_bgr())
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                
                                await websocket.send_json({
                                    "type": "frame",
                                    "data": img_base64,
                                    "timestamp": frame.timestamp.isoformat()
                                })
                    
                    elif message["type"] == "debug_capture":
                        # Capture frame and send to debugger
                        session_id = message.get("session_id", "default")
                        
                        if self.capture_manager:
                            frame = await self.capture_manager.capture_frame()
                            if frame:
                                from nexus.visual_debugger import debug_frame
                                
                                debug_result = await debug_frame(
                                    frame.to_rgb(),
                                    session_id=session_id,
                                    metadata={"capture_time_ms": frame.capture_time_ms}
                                )
                                
                                # Frame will be broadcast automatically to all WebSocket connections
                    
                    elif message["type"] == "debug_settings":
                        # Update debug session settings
                        session_id = message.get("session_id", "default")
                        settings = message.get("settings", {})
                        
                        try:
                            from nexus.visual_debugger import get_debugger
                            debugger = get_debugger()
                            success = debugger.update_session_settings(session_id, settings)
                            
                            await websocket.send_json({
                                "type": "debug_settings_updated",
                                "session_id": session_id,
                                "success": success
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to update debug settings: {e}"
                            })
                    
                    elif message["type"] == "action":
                        if self.current_agent:
                            action = await self.current_agent.act(message.get("observation"))
                            await websocket.send_json({
                                "type": "action",
                                "action": action
                            })
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)
    
    async def _initialize_plugin_manager(self):
        """Initialize plugin manager"""
        plugin_dirs = [Path(d) for d in self.config.get("nexus.plugin_dirs", ["plugins"])]
        self.plugin_manager = PluginManager(
            plugin_dirs,
            enable_hot_reload=self.config.get("plugins.hot_reload", True)
        )
        await self.plugin_manager.discover_plugins()
        logger.info("Plugin manager initialized")
    
    async def _initialize_capture_manager(self):
        """Initialize capture manager"""
        backend = CaptureBackendType(self.config.get("capture.backend", "dxcam"))
        self.capture_manager = CaptureManager(
            backend_type=backend,
            device_idx=self.config.get("capture.device_idx", 0),
            output_idx=self.config.get("capture.output_idx"),
            buffer_size=self.config.get("capture.buffer_size", 64)
        )
        await self.capture_manager.initialize()
        logger.info("Capture manager initialized")
    
    async def _stream_frames(self, fps: int):
        """Stream frames via WebSocket"""
        frame_interval = 1.0 / fps
        
        while self.stream_active:
            try:
                frame = await self.capture_manager.capture_frame()
                
                if frame:
                    # Convert to JPEG
                    _, buffer = cv2.imencode('.jpg', frame.to_bgr(), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Broadcast to all connected clients
                    await self.connection_manager.broadcast(json.dumps({
                        "type": "stream_frame",
                        "data": img_base64,
                        "frame_id": frame.frame_id,
                        "timestamp": frame.timestamp.isoformat()
                    }))
                
                await asyncio.sleep(frame_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(1)


def create_app(config: ConfigManager) -> FastAPI:
    """Create FastAPI application"""
    api = NexusAPI(config)
    return api.app


def run_server(app: FastAPI, host: str = "127.0.0.1", port: int = 8000):
    """Run the API server"""
    import uvicorn
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )