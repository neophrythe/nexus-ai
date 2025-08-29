"""
Visual Debugger for Nexus Game AI Framework.
Real-time visualization of AI decision making and game state.
"""

import threading
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = structlog.get_logger()


@dataclass
class DebugState:
    """Current debug state."""
    frame_number: int = 0
    current_frame: Optional[np.ndarray] = None
    agent_state: Dict[str, Any] = field(default_factory=dict)
    game_state: Dict[str, Any] = field(default_factory=dict)
    action_history: deque = field(default_factory=lambda: deque(maxlen=100))
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    metrics: Dict[str, float] = field(default_factory=dict)
    breakpoints: List[str] = field(default_factory=list)
    is_paused: bool = False


class VisualDebugger:
    """Visual debugging interface for Nexus."""
    
    def __init__(self, host: str = 'localhost', port: int = 8080):
        self.host = host
        self.port = port
        self.state = DebugState()
        self.server = None
        self.server_thread = None
        self.running = False
        self.clients = []
        
        # Data buffers
        self.frame_buffer = deque(maxlen=60)  # 1 second at 60 FPS
        self.event_log = deque(maxlen=1000)
        
        # Visualization settings
        self.show_overlays = True
        self.show_heatmap = False
        self.show_attention = False
        self.show_predictions = True
        
    def start(self):
        """Start the visual debugger server."""
        if self.running:
            return
        
        self.running = True
        
        # Start web server in background thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        logger.info(f"Visual debugger started on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the visual debugger."""
        self.running = False
        
        if self.server_thread:
            self.server_thread.join(timeout=2)
        
        logger.info("Visual debugger stopped")
    
    def _run_server(self):
        """Run the web server for the debugger interface."""
        try:
            # Simple HTTP server implementation
            import http.server
            import socketserver
            
            class DebugHandler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(self._get_html_interface().encode())
                    
                    elif self.path == '/api/state':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        state_json = self._serialize_state()
                        self.wfile.write(json.dumps(state_json).encode())
                    
                    elif self.path.startswith('/api/'):
                        self._handle_api_request()
                    
                    else:
                        self.send_error(404)
                
                def do_POST(self):
                    if self.path == '/api/pause':
                        self.server.debugger.pause()
                        self.send_response(200)
                        self.end_headers()
                    
                    elif self.path == '/api/resume':
                        self.server.debugger.resume()
                        self.send_response(200)
                        self.end_headers()
                    
                    elif self.path == '/api/step':
                        self.server.debugger.step()
                        self.send_response(200)
                        self.end_headers()
                    
                    elif self.path.startswith('/api/breakpoint/'):
                        self._handle_breakpoint()
                    
                    else:
                        self.send_error(404)
                
                def _get_html_interface(self):
                    """Generate HTML interface."""
                    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Nexus Visual Debugger</title>
    <style>
        body {
            font-family: 'Monaco', 'Courier New', monospace;
            background: #1a1a1a;
            color: #00ff00;
            margin: 0;
            padding: 20px;
        }
        
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            height: calc(100vh - 40px);
        }
        
        .main-view {
            background: #000;
            border: 2px solid #00ff00;
            padding: 10px;
            position: relative;
        }
        
        #game-frame {
            width: 100%;
            height: 70%;
            background: #222;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .controls {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        
        button {
            background: #003300;
            color: #00ff00;
            border: 1px solid #00ff00;
            padding: 10px 20px;
            cursor: pointer;
            font-family: inherit;
        }
        
        button:hover {
            background: #004400;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: #000;
            border: 1px solid #00ff00;
            padding: 10px;
            flex: 1;
            overflow-y: auto;
        }
        
        .panel h3 {
            margin: 0 0 10px 0;
            color: #00ff00;
            border-bottom: 1px solid #00ff00;
            padding-bottom: 5px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        
        .event-log {
            font-size: 12px;
            line-height: 1.4;
        }
        
        .event {
            margin: 2px 0;
            padding: 2px 5px;
            background: #002200;
        }
        
        .event.error {
            background: #330000;
            color: #ff3333;
        }
        
        .event.warning {
            background: #332200;
            color: #ffaa00;
        }
        
        canvas {
            max-width: 100%;
            height: auto;
        }
        
        .status-bar {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            padding: 5px 10px;
            border: 1px solid #00ff00;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-view">
            <div class="status-bar">
                <span id="status">CONNECTED</span> | 
                Frame: <span id="frame-number">0</span> | 
                FPS: <span id="fps">0</span>
            </div>
            
            <div id="game-frame">
                <canvas id="display-canvas"></canvas>
            </div>
            
            <div class="controls">
                <button onclick="togglePause()">⏸ PAUSE</button>
                <button onclick="step()">⏭ STEP</button>
                <button onclick="toggleOverlay()">👁 OVERLAY</button>
                <button onclick="toggleHeatmap()">🔥 HEATMAP</button>
                <button onclick="clearLog()">🗑 CLEAR</button>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="panel">
                <h3>AGENT STATE</h3>
                <div id="agent-state">
                    <div class="metric">
                        <span>Action:</span>
                        <span id="current-action">None</span>
                    </div>
                    <div class="metric">
                        <span>Reward:</span>
                        <span id="current-reward">0.00</span>
                    </div>
                    <div class="metric">
                        <span>Q-Value:</span>
                        <span id="q-value">0.00</span>
                    </div>
                    <div class="metric">
                        <span>Epsilon:</span>
                        <span id="epsilon">0.00</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>METRICS</h3>
                <div id="metrics">
                    <div class="metric">
                        <span>Avg Reward:</span>
                        <span id="avg-reward">0.00</span>
                    </div>
                    <div class="metric">
                        <span>Actions/Min:</span>
                        <span id="apm">0</span>
                    </div>
                    <div class="metric">
                        <span>Success Rate:</span>
                        <span id="success-rate">0%</span>
                    </div>
                </div>
            </div>
            
            <div class="panel event-log">
                <h3>EVENT LOG</h3>
                <div id="event-log"></div>
            </div>
        </div>
    </div>
    
    <script>
        let isPaused = false;
        let showOverlay = true;
        let showHeatmap = false;
        
        function updateState() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('frame-number').textContent = data.frame_number;
                    document.getElementById('current-action').textContent = data.current_action || 'None';
                    document.getElementById('current-reward').textContent = data.current_reward?.toFixed(2) || '0.00';
                    
                    // Update metrics
                    if (data.metrics) {
                        document.getElementById('avg-reward').textContent = data.metrics.avg_reward?.toFixed(2) || '0.00';
                        document.getElementById('apm').textContent = data.metrics.apm || '0';
                        document.getElementById('fps').textContent = data.metrics.fps || '0';
                    }
                    
                    // Update event log
                    if (data.events) {
                        const logEl = document.getElementById('event-log');
                        data.events.forEach(event => {
                            const eventEl = document.createElement('div');
                            eventEl.className = 'event ' + (event.level || '');
                            eventEl.textContent = `[${event.timestamp}] ${event.message}`;
                            logEl.appendChild(eventEl);
                        });
                        logEl.scrollTop = logEl.scrollHeight;
                    }
                });
        }
        
        function togglePause() {
            isPaused = !isPaused;
            fetch('/api/' + (isPaused ? 'pause' : 'resume'), {method: 'POST'});
        }
        
        function step() {
            fetch('/api/step', {method: 'POST'});
        }
        
        function toggleOverlay() {
            showOverlay = !showOverlay;
            // Update display
        }
        
        function toggleHeatmap() {
            showHeatmap = !showHeatmap;
            // Update display
        }
        
        function clearLog() {
            document.getElementById('event-log').innerHTML = '';
        }
        
        // Update state every 100ms
        setInterval(updateState, 100);
    </script>
</body>
</html>
                    '''
                
                def _serialize_state(self):
                    """Serialize current state to JSON."""
                    state = self.server.debugger.state
                    
                    return {
                        'frame_number': state.frame_number,
                        'is_paused': state.is_paused,
                        'current_action': state.agent_state.get('action'),
                        'current_reward': state.agent_state.get('reward', 0),
                        'metrics': state.metrics,
                        'events': []  # Would be populated with recent events
                    }
                
                def _handle_api_request(self):
                    """Handle API requests."""
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'ok'}).encode())
                
                def _handle_breakpoint(self):
                    """Handle breakpoint management."""
                    self.send_response(200)
                    self.end_headers()
                
                def log_message(self, format, *args):
                    # Suppress default logging
                    pass
            
            # Create custom server
            Handler = DebugHandler
            with socketserver.TCPServer((self.host, self.port), Handler) as httpd:
                httpd.debugger = self
                
                while self.running:
                    httpd.handle_request()
        
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    def update_frame(self, frame: np.ndarray):
        """Update current frame being debugged."""
        self.state.current_frame = frame
        self.state.frame_number += 1
        self.frame_buffer.append(frame)
    
    def update_agent_state(self, state: Dict[str, Any]):
        """Update agent state information."""
        self.state.agent_state.update(state)
        
        if 'action' in state:
            self.state.action_history.append(state['action'])
        
        if 'reward' in state:
            self.state.reward_history.append(state['reward'])
    
    def update_game_state(self, state: Dict[str, Any]):
        """Update game state information."""
        self.state.game_state.update(state)
    
    def log_event(self, event_type: str, message: str, level: str = 'info'):
        """Log an event to the debugger."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'message': message,
            'level': level
        }
        self.event_log.append(event)
    
    def set_breakpoint(self, condition: str):
        """Set a breakpoint condition."""
        self.state.breakpoints.append(condition)
        logger.info(f"Breakpoint set: {condition}")
    
    def pause(self):
        """Pause execution."""
        self.state.is_paused = True
        logger.info("Debugger paused")
    
    def resume(self):
        """Resume execution."""
        self.state.is_paused = False
        logger.info("Debugger resumed")
    
    def step(self):
        """Execute single step."""
        # This would trigger single step execution
        logger.info("Single step")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {
            'frame_number': self.state.frame_number,
            'fps': self._calculate_fps(),
            'avg_reward': np.mean(list(self.state.reward_history)) if self.state.reward_history else 0,
            'action_count': len(self.state.action_history)
        }
        
        return metrics
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.frame_buffer) < 2:
            return 0
        
        # Estimate based on buffer fill rate
        return len(self.frame_buffer)  # Simplified
    
    def visualize_attention(self, attention_weights: np.ndarray):
        """Visualize attention weights as overlay."""
        # This would create attention visualization
        pass
    
    def visualize_q_values(self, q_values: np.ndarray):
        """Visualize Q-values for actions."""
        # This would create Q-value visualization
        pass
    
    def export_session(self, output_path: str):
        """Export debug session data."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export state history
        state_data = {
            'frame_count': self.state.frame_number,
            'action_history': list(self.state.action_history),
            'reward_history': list(self.state.reward_history),
            'events': list(self.event_log),
            'metrics': self.state.metrics
        }
        
        with open(output_path / 'debug_session.json', 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Debug session exported to {output_path}")