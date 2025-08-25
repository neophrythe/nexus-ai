"""Game Agent Base Class with Frame Handler Modes - SerpentAI Compatible

Provides the core game agent architecture with frame handling modes.
"""

import time
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import structlog
from datetime import datetime
from collections import deque
import threading
import queue

from nexus.vision.game_frame import GameFrame
from nexus.input.controller import InputController
from nexus.game_registry import Game

logger = structlog.get_logger()


class FrameHandlerMode(Enum):
    """Frame handler modes - SerpentAI compatible"""
    NOOP = "NOOP"
    COLLECT_FRAMES = "COLLECT_FRAMES"
    COLLECT_FRAME_REGIONS = "COLLECT_FRAME_REGIONS"
    COLLECT_FRAMES_FOR_CONTEXT = "COLLECT_FRAMES_FOR_CONTEXT"
    PLAY = "PLAY"
    RECORD = "RECORD"
    TRAIN = "TRAIN"
    BENCHMARK = "BENCHMARK"


@dataclass
class AgentConfig:
    """Game agent configuration"""
    frame_handler: FrameHandlerMode = FrameHandlerMode.PLAY
    
    # Frame collection
    collect_frames_count: int = 1000
    collect_frames_interval: float = 0.1
    frame_buffer_size: int = 10
    
    # Context collection
    context_labels: List[str] = field(default_factory=list)
    current_context: Optional[str] = None
    auto_context_detection: bool = False
    
    # Region collection
    regions: List[Dict] = field(default_factory=list)
    
    # Recording
    record_inputs: bool = True
    record_rewards: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    
    # Performance
    fps_target: int = 30
    action_delay: float = 0.0
    
    # Debugging
    debug: bool = False
    save_frames: bool = False
    visualize: bool = False


class GameAgent:
    """Base game agent class - SerpentAI compatible with enhancements"""
    
    def __init__(self, game: Game, config: Optional[AgentConfig] = None):
        """
        Initialize game agent
        
        Args:
            game: Game instance
            config: Agent configuration
        """
        self.game = game
        self.config = config or AgentConfig()
        
        # Frame handler
        self.frame_handler = self.config.frame_handler
        self.frame_handlers = self._setup_frame_handlers()
        
        # Input controller
        self.input_controller = InputController()
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=self.config.frame_buffer_size)
        self.frame_counter = 0
        self.last_frame_time = 0
        
        # Collection storage
        self.collected_frames = []
        self.collected_regions = []
        self.context_frames = {label: [] for label in self.config.context_labels}
        
        # Recording storage
        self.recording = {
            'frames': [],
            'inputs': [],
            'rewards': [],
            'metadata': []
        }
        
        # Performance tracking
        self.fps_tracker = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_collected': 0,
            'actions_taken': 0,
            'total_reward': 0,
            'episode_count': 0,
            'start_time': None
        }
        
        # Agent state
        self.running = False
        self.paused = False
        self.episode_started = False
        
        # Initialize agent
        self.setup()
        
    def setup(self):
        """Setup agent - override in subclasses"""
        # Initialize components
        self.stats['start_time'] = time.time()
        
        # Load any saved models
        if hasattr(self.config, 'model_path') and self.config.model_path:
            self._load_model(self.config.model_path)
        
        # Setup visualization if enabled
        if self.config.visualize:
            self._setup_visualization()
        
        logger.info(f"Agent setup complete for {self.game.name}")
    
    def _load_model(self, path):
        """Load saved model"""
        if Path(path).exists():
            logger.info(f"Loading model from {path}")
            # Model loading logic here
    
    def _setup_visualization(self):
        """Setup visualization components"""
        logger.info("Visualization enabled")
        
    def _setup_frame_handlers(self) -> Dict[FrameHandlerMode, Callable]:
        """Setup frame handler functions"""
        return {
            FrameHandlerMode.NOOP: self._handle_frame_noop,
            FrameHandlerMode.COLLECT_FRAMES: self._handle_frame_collect,
            FrameHandlerMode.COLLECT_FRAME_REGIONS: self._handle_frame_collect_regions,
            FrameHandlerMode.COLLECT_FRAMES_FOR_CONTEXT: self._handle_frame_collect_context,
            FrameHandlerMode.PLAY: self._handle_frame_play,
            FrameHandlerMode.RECORD: self._handle_frame_record,
            FrameHandlerMode.TRAIN: self._handle_frame_train,
            FrameHandlerMode.BENCHMARK: self._handle_frame_benchmark
        }
        
    def on_game_frame(self, game_frame: GameFrame):
        """Handle game frame - SerpentAI compatible"""
        # Update frame buffer
        self.frame_buffer.append(game_frame)
        self.frame_counter += 1
        
        # Calculate FPS
        current_time = time.time()
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_tracker.append(fps)
        self.last_frame_time = current_time
        
        # Update statistics
        self.stats['frames_processed'] += 1
        
        # Call appropriate frame handler
        handler = self.frame_handlers.get(self.frame_handler)
        if handler:
            handler(game_frame)
            
        # Debug visualization
        if self.config.visualize:
            self._visualize_frame(game_frame)
            
    def _handle_frame_noop(self, game_frame: GameFrame):
        """NOOP frame handler - does nothing"""
        # Just count frames, no processing
        self.stats['frames_processed'] += 1
        
    def _handle_frame_collect(self, game_frame: GameFrame):
        """COLLECT_FRAMES handler - collect raw frames"""
        if len(self.collected_frames) < self.config.collect_frames_count:
            # Store frame
            self.collected_frames.append({
                'frame': game_frame.frame.copy(),
                'timestamp': game_frame.metadata.timestamp,
                'frame_number': self.frame_counter
            })
            
            self.stats['frames_collected'] += 1
            
            # Save if configured
            if self.config.save_frames:
                self._save_frame(game_frame, f"collected_{self.frame_counter:06d}")
                
            # Log progress
            if self.stats['frames_collected'] % 100 == 0:
                logger.info(f"Collected {self.stats['frames_collected']}/{self.config.collect_frames_count} frames")
                
        elif self.stats['frames_collected'] >= self.config.collect_frames_count:
            logger.info(f"Frame collection complete: {self.stats['frames_collected']} frames")
            self.stop()
            
    def _handle_frame_collect_regions(self, game_frame: GameFrame):
        """COLLECT_FRAME_REGIONS handler - collect specific regions"""
        for region in self.config.regions:
            # Extract region
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            region_frame = game_frame.frame[y:y+h, x:x+w]
            
            # Store region
            self.collected_regions.append({
                'region': region['name'],
                'frame': region_frame.copy(),
                'timestamp': game_frame.metadata.timestamp,
                'frame_number': self.frame_counter
            })
            
        self.stats['frames_collected'] += 1
        
        if self.stats['frames_collected'] >= self.config.collect_frames_count:
            logger.info(f"Region collection complete: {len(self.collected_regions)} regions")
            self.stop()
            
    def _handle_frame_collect_context(self, game_frame: GameFrame):
        """COLLECT_FRAMES_FOR_CONTEXT handler - collect frames by context"""
        # Determine context
        if self.config.auto_context_detection:
            context = self._detect_context(game_frame)
        else:
            context = self.config.current_context
            
        if context and context in self.config.context_labels:
            # Store frame in context
            self.context_frames[context].append({
                'frame': game_frame.frame.copy(),
                'timestamp': game_frame.metadata.timestamp,
                'frame_number': self.frame_counter
            })
            
            self.stats['frames_collected'] += 1
            
            # Log progress
            if self.stats['frames_collected'] % 50 == 0:
                context_counts = {c: len(frames) for c, frames in self.context_frames.items()}
                logger.info(f"Context collection progress: {context_counts}")
                
    def _handle_frame_play(self, game_frame: GameFrame):
        """PLAY handler - normal gameplay"""
        # Get actions from agent
        actions = self.handle_play(game_frame)
        
        if actions:
            # Execute actions
            for action in actions:
                self._execute_action(action)
                self.stats['actions_taken'] += 1
                
            # Track actions
            self.action_history.append({
                'frame': self.frame_counter,
                'actions': actions,
                'timestamp': time.time()
            })
            
    def _handle_frame_record(self, game_frame: GameFrame):
        """RECORD handler - record gameplay"""
        # Store frame
        if self.config.record_inputs:
            self.recording['frames'].append({
                'frame': game_frame.frame.copy(),
                'timestamp': game_frame.metadata.timestamp,
                'frame_number': self.frame_counter
            })
            
        # Get and execute actions
        actions = self.handle_play(game_frame)
        
        if actions:
            # Execute actions
            for action in actions:
                self._execute_action(action)
                
            # Record inputs
            if self.config.record_inputs:
                self.recording['inputs'].append({
                    'frame': self.frame_counter,
                    'actions': actions,
                    'timestamp': time.time()
                })
                
        # Record reward if available
        if self.config.record_rewards:
            reward = self.get_reward(game_frame)
            self.recording['rewards'].append({
                'frame': self.frame_counter,
                'reward': reward,
                'timestamp': time.time()
            })
            self.stats['total_reward'] += reward
            
    def _handle_frame_train(self, game_frame: GameFrame):
        """TRAIN handler - training mode"""
        # Get current state
        state = self.get_state(game_frame)
        
        # Get action from policy
        action = self.get_action(state)
        
        # Execute action
        if action:
            self._execute_action(action)
            
        # Get reward
        reward = self.get_reward(game_frame)
        
        # Get next state
        next_state = self.get_state(game_frame)
        
        # Check if episode is done
        done = self.is_episode_done(game_frame)
        
        # Train the agent
        self.train_step(state, action, reward, next_state, done)
        
        # Update statistics
        self.stats['total_reward'] += reward
        
        if done:
            self.stats['episode_count'] += 1
            self.on_episode_end()
            
    def _handle_frame_benchmark(self, game_frame: GameFrame):
        """BENCHMARK handler - performance testing"""
        # Measure action generation time
        start_time = time.time()
        actions = self.handle_play(game_frame)
        action_time = time.time() - start_time
        
        # Execute actions
        if actions:
            for action in actions:
                self._execute_action(action)
                
        # Track performance
        self.action_history.append({
            'frame': self.frame_counter,
            'action_time': action_time,
            'fps': self.get_current_fps()
        })
        
    def handle_play(self, game_frame: GameFrame) -> List[Dict]:
        """Handle play mode - override in subclasses"""
        return []
        
    def get_state(self, game_frame: GameFrame) -> Any:
        """Get current state - override in subclasses"""
        return game_frame.frame
        
    def get_action(self, state: Any) -> Optional[Dict]:
        """Get action from state - override in subclasses"""
        return None
        
    def get_reward(self, game_frame: GameFrame) -> float:
        """Get reward - override in subclasses"""
        return 0.0
        
    def is_episode_done(self, game_frame: GameFrame) -> bool:
        """Check if episode is done - override in subclasses"""
        return False
        
    def train_step(self, state: Any, action: Any, reward: float, 
                  next_state: Any, done: bool):
        """Training step - override in subclasses"""
        # Store experience
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        # Add to replay buffer if available
        if hasattr(self, 'replay_buffer'):
            self.replay_buffer.append(experience)
        
        # Trigger learning if enough samples
        if hasattr(self, 'replay_buffer') and len(self.replay_buffer) >= self.config.batch_size:
            self._perform_training_step()
    
    def _perform_training_step(self):
        """Perform actual training step"""
        # Implemented by specific agent types
        pass
        
    def on_episode_end(self):
        """Called when episode ends - override in subclasses"""
        self.episode_started = False
        
    def _detect_context(self, game_frame: GameFrame) -> Optional[str]:
        """Auto-detect context - override in subclasses"""
        return None
        
    def _execute_action(self, action: Dict):
        """Execute an action"""
        action_type = action.get('type')
        
        if action_type == 'key':
            key = action.get('key')
            duration = action.get('duration', 0.05)
            self.input_controller.tap_key(key, duration)
            
        elif action_type == 'keys':
            keys = action.get('keys', [])
            for key in keys:
                self.input_controller.tap_key(key)
                
        elif action_type == 'mouse_move':
            x, y = action.get('x'), action.get('y')
            self.input_controller.move_mouse(x, y)
            
        elif action_type == 'mouse_click':
            button = action.get('button', 'left')
            x = action.get('x')
            y = action.get('y')
            
            if x and y:
                self.input_controller.move_mouse(x, y)
            self.input_controller.click(button=button)
            
        elif action_type == 'mouse_drag':
            start_x, start_y = action.get('start_x'), action.get('start_y')
            end_x, end_y = action.get('end_x'), action.get('end_y')
            button = action.get('button', 'left')
            
            self.input_controller.drag(start_x, start_y, end_x, end_y, button)
            
        # Add configured delay
        if self.config.action_delay > 0:
            time.sleep(self.config.action_delay)
            
    def _save_frame(self, game_frame: GameFrame, filename: str):
        """Save frame to disk"""
        output_dir = Path("collected_frames") / self.game.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / f"{filename}.png"
        game_frame.save(str(filepath))
        
    def _visualize_frame(self, game_frame: GameFrame):
        """Visualize frame for debugging"""
        import cv2
        
        frame = game_frame.frame.copy()
        
        # Add debug overlays
        # FPS
        fps = self.get_current_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_counter}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mode
        cv2.putText(frame, f"Mode: {self.frame_handler.value}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Stats
        if hasattr(self, 'stats'):
            cv2.putText(frame, f"Reward: {self.stats.get('total_reward', 0):.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow(f"Nexus Debug - {self.game.name}", frame)
        cv2.waitKey(1)
        
    def set_frame_handler(self, mode: FrameHandlerMode):
        """Set frame handler mode"""
        self.frame_handler = mode
        logger.info(f"Frame handler set to: {mode.value}")
        
    def set_context(self, context: str):
        """Set current context for collection"""
        if context in self.config.context_labels:
            self.config.current_context = context
            logger.info(f"Context set to: {context}")
        else:
            logger.warning(f"Unknown context: {context}")
            
    def get_current_fps(self) -> float:
        """Get current FPS"""
        if self.fps_tracker:
            return sum(self.fps_tracker) / len(self.fps_tracker)
        return 0.0
        
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            **self.stats,
            'runtime': runtime,
            'current_fps': self.get_current_fps(),
            'buffer_size': len(self.frame_buffer),
            'mode': self.frame_handler.value
        }
        
    def save_recording(self, filepath: str):
        """Save recording to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.recording, f)
            
        logger.info(f"Recording saved to {filepath}")
        
    def load_recording(self, filepath: str):
        """Load recording from file"""
        with open(filepath, 'rb') as f:
            self.recording = pickle.load(f)
            
        logger.info(f"Recording loaded from {filepath}")
        
    def start(self):
        """Start the agent"""
        self.running = True
        self.stats['start_time'] = time.time()
        logger.info(f"Agent started in {self.frame_handler.value} mode")
        
    def pause(self):
        """Pause the agent"""
        self.paused = True
        logger.info("Agent paused")
        
    def resume(self):
        """Resume the agent"""
        self.paused = False
        logger.info("Agent resumed")
        
    def stop(self):
        """Stop the agent"""
        self.running = False
        
        # Save any collected data
        if self.collected_frames:
            self._save_collected_frames()
        if self.recording['frames']:
            self.save_recording(f"recording_{int(time.time())}.pkl")
            
        logger.info("Agent stopped")
        
    def _save_collected_frames(self):
        """Save collected frames"""
        output_dir = Path("collected_frames") / self.game.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frames
        data = {
            'frames': self.collected_frames,
            'regions': self.collected_regions,
            'context_frames': self.context_frames,
            'metadata': {
                'game': self.game.name,
                'total_frames': self.stats['frames_collected'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        filepath = output_dir / f"collection_{int(time.time())}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Collected frames saved to {filepath}")


# SerpentAI compatibility
class GameAgentError(Exception):
    """Game agent exception"""
    def __init__(self, message="Game agent error occurred", *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.message = message
        logger.error(f"GameAgentError: {message}")