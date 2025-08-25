from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import structlog

logger = structlog.get_logger()


class GamePhase(Enum):
    MENU = "menu"
    LOADING = "loading"
    PLAYING = "playing"
    PAUSED = "paused"
    ENDED = "ended"
    UNKNOWN = "unknown"


@dataclass
class GameState:
    phase: GamePhase
    observation: Any
    info: Dict[str, Any]
    timestamp: datetime
    frame_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
            "info": self.info,
            "metadata": self.metadata
        }


class GameEnvironment(gym.Env, ABC):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, 
                 game_name: str,
                 capture_manager: Any = None,
                 input_controller: Any = None,
                 render_mode: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        
        super().__init__()
        
        self.game_name = game_name
        self.capture_manager = capture_manager
        self.input_controller = input_controller
        self.render_mode = render_mode
        self.config = config or {}
        
        self.current_phase = GamePhase.UNKNOWN
        self.frame_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        self._state_history: List[GameState] = []
        self._max_history_size = self.config.get("max_history_size", 1000)
        
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()
        
        self._last_observation = None
        self._last_action = None
        self._episode_start_time = None
        
        logger.info(f"Game environment initialized: {game_name}")
    
    @abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        """Build observation space for the environment"""
        # Default: RGB image observation space
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.frame_height, self.config.frame_width, 3),
            dtype=np.uint8
        )
    
    @abstractmethod
    def _build_action_space(self) -> spaces.Space:
        """Build action space for the environment"""
        # Default: Discrete action space
        num_actions = len(self.config.actions) if hasattr(self.config, 'actions') else 10
        return spaces.Discrete(num_actions)
    
    @abstractmethod
    def _get_observation(self) -> Any:
        """Get current observation from the game"""
        # Capture frame
        frame = self.capture_manager.capture_frame()
        
        if frame is None:
            # Return blank frame if capture fails
            frame = np.zeros((self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)
        
        # Apply preprocessing if configured
        if self.config.preprocess:
            frame = self._preprocess_frame(frame)
        
        return frame
    
    def _preprocess_frame(self, frame):
        """Preprocess captured frame"""
        import cv2
        # Resize if needed
        if frame.shape[:2] != (self.config.frame_height, self.config.frame_width):
            frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
        return frame
    
    @abstractmethod
    def _calculate_reward(self, observation: Any, action: Any) -> float:
        """Calculate reward for the current state and action"""
        reward = 0.0
        
        # Basic survival reward
        reward += 0.1
        
        # Score-based reward if available
        if hasattr(self, 'last_score'):
            current_score = self._extract_score(observation)
            if current_score is not None:
                reward += (current_score - self.last_score) * 0.01
                self.last_score = current_score
        
        # Penalty for repetitive actions
        if hasattr(self, 'last_action') and action == self.last_action:
            reward -= 0.05
        
        self.last_action = action
        return reward
    
    def _extract_score(self, observation):
        """Extract score from observation (override in subclass)"""
        return None
    
    @abstractmethod
    def _is_terminated(self, observation: Any) -> bool:
        """Check if episode is terminated (game over)"""
        # Check for game over conditions
        if self.current_phase == GamePhase.GAME_OVER:
            return True
        
        # Check for victory condition
        if self.current_phase == GamePhase.VICTORY:
            return True
        
        # Additional termination conditions can be added in subclasses
        return False
    
    @abstractmethod
    def _is_truncated(self, observation: Any) -> bool:
        """Check if episode is truncated (time limit reached)"""
        # Check frame limit
        if self.config.max_frames and self.frame_count >= self.config.max_frames:
            return True
        
        # Check time limit
        if self.config.max_time:
            elapsed = time.time() - self.episode_start_time
            if elapsed >= self.config.max_time:
                return True
        
        return False
    
    @abstractmethod
    def _execute_action(self, action: Any) -> None:
        """Execute action in the game"""
        # Convert action index to actual input
        if isinstance(action, (int, np.integer)):
            if hasattr(self.config, 'actions'):
                action_name = self.config.actions[action]
            else:
                action_name = str(action)
        else:
            action_name = action
        
        # Send input to game
        self.input_controller.send_action(action_name)
        
        # Apply action delay if configured
        if self.config.action_delay:
            time.sleep(self.config.action_delay)
    
    @abstractmethod
    def _detect_game_phase(self, observation: Any) -> GamePhase:
        """Detect current game phase from observation"""
        # Default implementation using template matching or OCR
        
        # Check for menu screen
        if self._is_menu_visible(observation):
            return GamePhase.MENU
        
        # Check for loading screen
        if self._is_loading_screen(observation):
            return GamePhase.LOADING
        
        # Check for game over
        if self._is_game_over_screen(observation):
            return GamePhase.GAME_OVER
        
        # Check for victory
        if self._is_victory_screen(observation):
            return GamePhase.VICTORY
        
        # Default to playing
        return GamePhase.PLAYING
    
    def _is_menu_visible(self, observation):
        """Check if menu is visible (override in subclass)"""
        return False
    
    def _is_loading_screen(self, observation):
        """Check if loading screen is visible (override in subclass)"""
        return False
    
    def _is_game_over_screen(self, observation):
        """Check if game over screen is visible (override in subclass)"""
        return False
    
    def _is_victory_screen(self, observation):
        """Check if victory screen is visible (override in subclass)"""
        return False
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.frame_count = 0
        self.total_reward = 0.0
        self._episode_start_time = datetime.now()
        self.episode_count += 1
        
        observation = self._get_observation()
        self._last_observation = observation
        
        self.current_phase = self._detect_game_phase(observation)
        
        info = self._get_info()
        
        self._add_state_to_history(observation, info)
        
        logger.info(f"Environment reset - Episode {self.episode_count}")
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._last_action = action
        
        self._execute_action(action)
        
        observation = self._get_observation()
        self._last_observation = observation
        
        self.current_phase = self._detect_game_phase(observation)
        
        reward = self._calculate_reward(observation, action)
        self.total_reward += reward
        
        terminated = self._is_terminated(observation)
        truncated = self._is_truncated(observation)
        
        info = self._get_info()
        info["reward"] = reward
        info["total_reward"] = self.total_reward
        
        self.frame_count += 1
        
        self._add_state_to_history(observation, info)
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, None]]:
        if self.render_mode == "rgb_array":
            if self.capture_manager:
                frame = self.capture_manager.get_latest_frame()
                if frame:
                    return frame.to_rgb()
            return self._last_observation if isinstance(self._last_observation, np.ndarray) else None
        elif self.render_mode == "human":
            # Display frame in window for human viewing
            import cv2
            if hasattr(self, '_last_observation') and isinstance(self._last_observation, np.ndarray):
                frame = self._last_observation.copy()
                # Add overlay information
                cv2.putText(frame, f"Episode: {self.episode_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Reward: {self.total_reward:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Nexus - {self.game_name}", frame)
                cv2.waitKey(1)
        return None
    
    def close(self) -> None:
        logger.info(f"Closing environment: {self.game_name}")
        if self.capture_manager:
            asyncio.create_task(self.capture_manager.cleanup())
        if self.input_controller:
            asyncio.create_task(self.input_controller.cleanup())
    
    def _get_info(self) -> Dict[str, Any]:
        info = {
            "frame": self.frame_count,
            "episode": self.episode_count,
            "phase": self.current_phase.value,
            "timestamp": datetime.now().isoformat()
        }
        
        if self._episode_start_time:
            elapsed = (datetime.now() - self._episode_start_time).total_seconds()
            info["episode_time"] = elapsed
            info["fps"] = self.frame_count / elapsed if elapsed > 0 else 0
        
        return info
    
    def _add_state_to_history(self, observation: Any, info: Dict[str, Any]) -> None:
        state = GameState(
            phase=self.current_phase,
            observation=observation,
            info=info,
            timestamp=datetime.now(),
            frame_id=self.frame_count
        )
        
        self._state_history.append(state)
        
        if len(self._state_history) > self._max_history_size:
            self._state_history.pop(0)
    
    def get_state_history(self, n: Optional[int] = None) -> List[GameState]:
        if n is None:
            return self._state_history.copy()
        return self._state_history[-n:] if n <= len(self._state_history) else self._state_history.copy()
    
    def save_state(self, path: str) -> None:
        import pickle
        state_data = {
            "game_name": self.game_name,
            "frame_count": self.frame_count,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "current_phase": self.current_phase,
            "state_history": self._state_history[-100:],
            "config": self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state_data, f)
        
        logger.info(f"State saved to {path}")
    
    def load_state(self, path: str) -> None:
        import pickle
        
        with open(path, 'rb') as f:
            state_data = pickle.load(f)
        
        self.frame_count = state_data["frame_count"]
        self.episode_count = state_data["episode_count"]
        self.total_reward = state_data["total_reward"]
        self.current_phase = state_data["current_phase"]
        self._state_history = state_data["state_history"]
        
        logger.info(f"State loaded from {path}")


class MultiAgentGameEnvironment(GameEnvironment):
    
    def __init__(self, *args, num_agents: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_agents = num_agents
        self.agent_observations = {}
        self.agent_rewards = {}
        self.agent_infos = {}
    
    @abstractmethod
    def _get_agent_observation(self, agent_id: int) -> Any:
        """Get observation for specific agent in multi-agent environment"""
        # Get base observation
        base_obs = self._get_observation()
        
        # Add agent-specific information
        agent_obs = {
            'frame': base_obs,
            'agent_id': agent_id,
            'agent_position': self.agent_positions.get(agent_id, [0, 0]) if hasattr(self, 'agent_positions') else [0, 0],
            'other_agents': [pos for aid, pos in self.agent_positions.items() if aid != agent_id] if hasattr(self, 'agent_positions') else []
        }
        
        return agent_obs
    
    @abstractmethod
    def _calculate_agent_reward(self, agent_id: int, observation: Any, action: Any) -> float:
        """Calculate reward for specific agent in multi-agent environment"""
        # Base reward calculation
        base_reward = self._calculate_reward(observation, action)
        
        # Agent-specific adjustments
        agent_reward = base_reward
        
        # Cooperation bonus
        if hasattr(self, 'encourage_cooperation') and self.encourage_cooperation:
            # Check if agents are working together
            if self._check_cooperation(agent_id):
                agent_reward += 1.0
        
        # Competition penalty
        if hasattr(self, 'competitive') and self.competitive:
            # Penalty for being behind other agents
            agent_score = self.agent_scores.get(agent_id, 0) if hasattr(self, 'agent_scores') else 0
            avg_score = sum(self.agent_scores.values()) / len(self.agent_scores) if hasattr(self, 'agent_scores') and self.agent_scores else 0
            if agent_score < avg_score:
                agent_reward -= 0.5
        
        return agent_reward
    
    def _check_cooperation(self, agent_id):
        """Check if agent is cooperating with others"""
        # Simple proximity-based cooperation check
        if hasattr(self, 'agent_positions') and len(self.agent_positions) > 1:
            agent_pos = self.agent_positions.get(agent_id)
            if agent_pos:
                for other_id, other_pos in self.agent_positions.items():
                    if other_id != agent_id:
                        dist = np.linalg.norm(np.array(agent_pos) - np.array(other_pos))
                        if dist < 50:  # Within cooperation range
                            return True
        return False
    
    def reset(self, **kwargs) -> Tuple[Dict[int, Any], Dict[int, Dict[str, Any]]]:
        super().reset(**kwargs)
        
        observations = {}
        infos = {}
        
        for agent_id in range(self.num_agents):
            observations[agent_id] = self._get_agent_observation(agent_id)
            infos[agent_id] = self._get_info()
        
        self.agent_observations = observations
        self.agent_infos = infos
        
        return observations, infos
    
    def step(self, actions: Dict[int, Any]) -> Tuple[
        Dict[int, Any], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Dict[str, Any]]
    ]:
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        for agent_id, action in actions.items():
            self._execute_action(action)
            
            obs = self._get_agent_observation(agent_id)
            observations[agent_id] = obs
            
            reward = self._calculate_agent_reward(agent_id, obs, action)
            rewards[agent_id] = reward
            
            terminateds[agent_id] = self._is_terminated(obs)
            truncateds[agent_id] = self._is_truncated(obs)
            
            info = self._get_info()
            info["agent_id"] = agent_id
            info["reward"] = reward
            infos[agent_id] = info
        
        self.agent_observations = observations
        self.agent_rewards = rewards
        self.agent_infos = infos
        
        self.frame_count += 1
        
        return observations, rewards, terminateds, truncateds, infos