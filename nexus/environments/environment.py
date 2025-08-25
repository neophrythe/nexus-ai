"""Complete Environment System with Episode Management for Nexus Framework"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import structlog
from collections import deque
import json
from pathlib import Path
import pickle

from nexus.vision.frame_processing import GameFrame

logger = structlog.get_logger()


class EpisodeState(Enum):
    """Episode states"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RewardType(Enum):
    """Types of rewards"""
    SPARSE = "sparse"          # Only at episode end
    DENSE = "dense"            # Every step
    SHAPED = "shaped"          # Custom reward shaping
    CURRICULUM = "curriculum"   # Curriculum learning rewards


@dataclass
class StepResult:
    """Result from environment step"""
    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]
    
    @property
    def terminal(self) -> bool:
        """Check if episode is terminal"""
        return self.done or self.truncated


@dataclass
class Episode:
    """Single episode data"""
    episode_id: str
    start_time: float
    end_time: Optional[float]
    total_reward: float
    steps: int
    state: EpisodeState
    metadata: Dict[str, Any]
    
    # Episode data
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Episode duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def average_reward(self) -> float:
        """Average reward per step"""
        return self.total_reward / max(1, self.steps)
    
    def add_step(self, observation: np.ndarray, action: Any, 
                 reward: float, info: Dict[str, Any]):
        """Add step data to episode"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
        self.total_reward += reward
        self.steps += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary"""
        return {
            "episode_id": self.episode_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "state": self.state.value,
            "metadata": self.metadata,
            "duration": self.duration,
            "average_reward": self.average_reward
        }


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    name: str
    observation_shape: Tuple[int, ...]
    action_space: Union[int, Tuple[int, ...]]
    max_episode_steps: int = 1000
    reward_type: RewardType = RewardType.DENSE
    frame_skip: int = 1
    render_mode: Optional[str] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GameEnvironment:
    """Complete game environment with episode management"""
    
    def __init__(self, config: EnvironmentConfig):
        """
        Initialize game environment
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.current_episode: Optional[Episode] = None
        self.episode_history: deque = deque(maxlen=100)
        self.total_episodes = 0
        
        # State management
        self.current_observation: Optional[np.ndarray] = None
        self.previous_observation: Optional[np.ndarray] = None
        self.episode_step_count = 0
        self.total_step_count = 0
        
        # Reward management
        self.reward_function: Optional[Callable] = None
        self.reward_shaping_enabled = False
        
        # Callbacks
        self.episode_start_callbacks: List[Callable] = []
        self.episode_end_callbacks: List[Callable] = []
        self.step_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "total_steps": 0,
            "total_reward": 0.0,
            "best_episode_reward": float('-inf'),
            "average_episode_length": 0.0
        }
        
        # Initialize random seed if provided
        if config.seed is not None:
            self.seed(config.seed)
        
        logger.info(f"Initialized environment: {config.name}")
    
    def reset(self, seed: Optional[int] = None, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode
        
        Args:
            seed: Random seed
            options: Reset options
        
        Returns:
            Initial observation and info
        """
        # End current episode if exists
        if self.current_episode and self.current_episode.state == EpisodeState.IN_PROGRESS:
            self._end_episode(success=False, reason="reset")
        
        # Set seed if provided
        if seed is not None:
            self.seed(seed)
        
        # Create new episode
        episode_id = f"{self.config.name}_{self.total_episodes:06d}_{int(time.time())}"
        self.current_episode = Episode(
            episode_id=episode_id,
            start_time=time.time(),
            end_time=None,
            total_reward=0.0,
            steps=0,
            state=EpisodeState.IN_PROGRESS,
            metadata=options or {}
        )
        
        # Reset step counts
        self.episode_step_count = 0
        
        # Get initial observation
        self.current_observation = self._get_initial_observation()
        self.previous_observation = None
        
        # Trigger callbacks
        for callback in self.episode_start_callbacks:
            callback(self.current_episode)
        
        info = {
            "episode_id": episode_id,
            "episode_number": self.total_episodes,
            "reset_options": options
        }
        
        logger.info(f"Started episode {episode_id}")
        
        return self.current_observation, info
    
    def step(self, action: Any) -> StepResult:
        """
        Execute action in environment
        
        Args:
            action: Action to execute
        
        Returns:
            Step result with observation, reward, done, truncated, info
        """
        if self.current_episode is None or self.current_episode.state != EpisodeState.IN_PROGRESS:
            raise RuntimeError("Environment must be reset before calling step")
        
        # Store previous observation
        self.previous_observation = self.current_observation.copy()
        
        # Execute action (with frame skip)
        for _ in range(self.config.frame_skip):
            self._execute_action(action)
        
        # Get new observation
        self.current_observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        done = self._is_episode_done()
        truncated = self.episode_step_count >= self.config.max_episode_steps
        
        # Prepare info
        info = self._get_step_info(action)
        
        # Update episode
        self.current_episode.add_step(
            self.current_observation.copy(),
            action,
            reward,
            info
        )
        
        # Update counters
        self.episode_step_count += 1
        self.total_step_count += 1
        
        # Trigger callbacks
        for callback in self.step_callbacks:
            callback(self.current_observation, action, reward, done, truncated, info)
        
        # End episode if terminal
        if done or truncated:
            self._end_episode(success=done and not truncated, reason="terminal")
        
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info
        )
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render environment
        
        Returns:
            Rendered frame if applicable
        """
        if self.config.render_mode == "rgb_array":
            return self._render_frame()
        elif self.config.render_mode == "human":
            self._render_human()
            return None
        return None
    
    def close(self):
        """Clean up environment resources"""
        if self.current_episode and self.current_episode.state == EpisodeState.IN_PROGRESS:
            self._end_episode(success=False, reason="closed")
        
        logger.info(f"Environment {self.config.name} closed")
    
    def seed(self, seed: int):
        """Set random seed"""
        np.random.seed(seed)
        logger.info(f"Set environment seed: {seed}")
    
    def _get_initial_observation(self) -> np.ndarray:
        """Get initial observation (override in subclass)"""
        # Default: random observation with correct shape
        return np.random.randn(*self.config.observation_shape).astype(np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (override in subclass)"""
        # Default: slightly modified previous observation
        if self.previous_observation is not None:
            noise = np.random.randn(*self.config.observation_shape) * 0.01
            return np.clip(self.previous_observation + noise, -1, 1).astype(np.float32)
        return self._get_initial_observation()
    
    def _execute_action(self, action: Any):
        """Execute action in environment (override in subclass)"""
        pass  # Implement in subclass
    
    def _calculate_reward(self, action: Any) -> float:
        """
        Calculate reward for action
        
        Args:
            action: Executed action
        
        Returns:
            Reward value
        """
        if self.reward_function:
            return self.reward_function(
                self.previous_observation,
                action,
                self.current_observation,
                self.current_episode
            )
        
        # Default reward based on type
        if self.config.reward_type == RewardType.SPARSE:
            return 0.0  # Only reward at episode end
        elif self.config.reward_type == RewardType.DENSE:
            return self._default_dense_reward(action)
        elif self.config.reward_type == RewardType.SHAPED:
            return self._shaped_reward(action)
        elif self.config.reward_type == RewardType.CURRICULUM:
            return self._curriculum_reward(action)
        
        return 0.0
    
    def _default_dense_reward(self, action: Any) -> float:
        """Default dense reward (override in subclass)"""
        return -0.01  # Small negative reward to encourage efficiency
    
    def _shaped_reward(self, action: Any) -> float:
        """Shaped reward with multiple components (override in subclass)"""
        return 0.0
    
    def _curriculum_reward(self, action: Any) -> float:
        """Curriculum learning reward (override in subclass)"""
        # Adjust reward based on learning progress
        progress = min(1.0, self.total_episodes / 1000.0)
        base_reward = self._default_dense_reward(action)
        return base_reward * (1.0 + progress)
    
    def _is_episode_done(self) -> bool:
        """Check if episode is done (override in subclass)"""
        return False
    
    def _get_step_info(self, action: Any) -> Dict[str, Any]:
        """Get step information"""
        return {
            "episode_step": self.episode_step_count,
            "total_step": self.total_step_count,
            "action": action,
            "episode_id": self.current_episode.episode_id if self.current_episode else None
        }
    
    def _render_frame(self) -> np.ndarray:
        """Render frame as RGB array (override in subclass)"""
        # Default: return current observation as image
        if self.current_observation is not None:
            # Normalize to 0-255 range
            normalized = ((self.current_observation + 1) * 127.5).astype(np.uint8)
            
            # Ensure 3 channels
            if len(normalized.shape) == 2:
                normalized = np.stack([normalized] * 3, axis=-1)
            elif normalized.shape[-1] == 1:
                normalized = np.repeat(normalized, 3, axis=-1)
            
            return normalized
        
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def _render_human(self):
        """Render for human viewing (override in subclass)"""
        pass
    
    def _end_episode(self, success: bool, reason: str):
        """End current episode"""
        if not self.current_episode:
            return
        
        self.current_episode.end_time = time.time()
        self.current_episode.state = EpisodeState.COMPLETED if success else EpisodeState.FAILED
        self.current_episode.metadata["end_reason"] = reason
        
        # Update statistics
        self.total_episodes += 1
        self.stats["total_episodes"] = self.total_episodes
        self.stats["total_steps"] += self.current_episode.steps
        self.stats["total_reward"] += self.current_episode.total_reward
        
        if success:
            self.stats["successful_episodes"] += 1
        else:
            self.stats["failed_episodes"] += 1
        
        if self.current_episode.total_reward > self.stats["best_episode_reward"]:
            self.stats["best_episode_reward"] = self.current_episode.total_reward
        
        self.stats["average_episode_length"] = self.stats["total_steps"] / max(1, self.total_episodes)
        
        # Add to history
        self.episode_history.append(self.current_episode)
        
        # Trigger callbacks
        for callback in self.episode_end_callbacks:
            callback(self.current_episode)
        
        logger.info(f"Episode {self.current_episode.episode_id} ended: "
                   f"success={success}, reward={self.current_episode.total_reward:.2f}, "
                   f"steps={self.current_episode.steps}")
    
    def set_reward_function(self, reward_function: Callable):
        """Set custom reward function"""
        self.reward_function = reward_function
        logger.info("Custom reward function set")
    
    def add_episode_start_callback(self, callback: Callable):
        """Add callback for episode start"""
        self.episode_start_callbacks.append(callback)
    
    def add_episode_end_callback(self, callback: Callable):
        """Add callback for episode end"""
        self.episode_end_callbacks.append(callback)
    
    def add_step_callback(self, callback: Callable):
        """Add callback for each step"""
        self.step_callbacks.append(callback)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics"""
        return self.stats.copy()
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes"""
        return list(self.episode_history)[-n:]
    
    def save_episode(self, episode: Episode, path: str):
        """Save episode data to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(episode, f)
        
        logger.info(f"Episode saved to {path}")
    
    def load_episode(self, path: str) -> Episode:
        """Load episode data from file"""
        with open(path, 'rb') as f:
            episode = pickle.load(f)
        
        logger.info(f"Episode loaded from {path}")
        return episode
    
    def save_state(self, path: str):
        """Save environment state"""
        state = {
            "config": self.config,
            "stats": self.stats,
            "total_episodes": self.total_episodes,
            "total_step_count": self.total_step_count,
            "episode_history": list(self.episode_history)
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Environment state saved to {path}")
    
    def load_state(self, path: str):
        """Load environment state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.stats = state["stats"]
        self.total_episodes = state["total_episodes"]
        self.total_step_count = state["total_step_count"]
        self.episode_history = deque(state["episode_history"], maxlen=100)
        
        logger.info(f"Environment state loaded from {path}")


class VectorizedEnvironment:
    """Vectorized environment for parallel execution"""
    
    def __init__(self, env_fn: Callable, num_envs: int = 4):
        """
        Initialize vectorized environment
        
        Args:
            env_fn: Function that creates environment
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]
        
        # Get observation and action shapes from first env
        self.observation_shape = self.envs[0].config.observation_shape
        self.action_space = self.envs[0].config.action_space
        
        logger.info(f"Initialized vectorized environment with {num_envs} instances")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Reset all environments
        
        Args:
            seed: Base random seed
        
        Returns:
            Stacked observations and list of infos
        """
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        
        return np.stack(observations), infos
    
    def step(self, actions: Union[np.ndarray, List]) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray,
                                                              List[Dict[str, Any]]]:
        """
        Step all environments
        
        Args:
            actions: Actions for each environment
        
        Returns:
            observations, rewards, dones, truncateds, infos
        """
        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            result = env.step(action)
            observations.append(result.observation)
            rewards.append(result.reward)
            dones.append(result.done)
            truncateds.append(result.truncated)
            infos.append(result.info)
        
        return (np.stack(observations), 
                np.array(rewards),
                np.array(dones),
                np.array(truncateds),
                infos)
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()