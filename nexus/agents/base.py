from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import structlog

logger = structlog.get_logger()


class AgentType(Enum):
    SCRIPTED = "scripted"
    RL = "reinforcement_learning"
    IMITATION = "imitation_learning"
    HYBRID = "hybrid"
    LLM = "llm_based"
    HUMAN = "human"


class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    LEARNING = "learning"
    ERROR = "error"


@dataclass
class Experience:
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    done: bool
    info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentMetrics:
    total_steps: int = 0
    total_episodes: int = 0
    total_reward: float = 0.0
    avg_reward_per_episode: float = 0.0
    avg_steps_per_episode: float = 0.0
    win_rate: float = 0.0
    best_episode_reward: float = float('-inf')
    worst_episode_reward: float = float('inf')
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_episode(self, reward: float, steps: int, won: bool = False):
        self.total_episodes += 1
        self.total_steps += steps
        self.total_reward += reward
        
        self.avg_reward_per_episode = self.total_reward / self.total_episodes
        self.avg_steps_per_episode = self.total_steps / self.total_episodes
        
        if won:
            wins = self.win_rate * (self.total_episodes - 1)
            self.win_rate = (wins + 1) / self.total_episodes
        else:
            wins = self.win_rate * (self.total_episodes - 1)
            self.win_rate = wins / self.total_episodes
        
        self.best_episode_reward = max(self.best_episode_reward, reward)
        self.worst_episode_reward = min(self.worst_episode_reward, reward)
        self.last_update = datetime.now()


class BaseAgent(ABC):
    
    def __init__(self, 
                 name: str,
                 agent_type: AgentType,
                 observation_space: Any = None,
                 action_space: Any = None,
                 config: Optional[Dict[str, Any]] = None):
        
        self.name = name
        self.agent_type = agent_type
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}
        
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        
        self.experience_buffer: List[Experience] = []
        self.max_buffer_size = self.config.get("max_buffer_size", 10000)
        
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self._initialized = False
        
        logger.info(f"Agent {name} ({agent_type.value}) created")
    
    @abstractmethod
    async def initialize(self) -> None:
        self._initialized = True
    
    @abstractmethod
    async def act(self, observation: Any) -> Any:
        """Select action based on observation - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        if hasattr(self, 'action_space'):
            if hasattr(self.action_space, 'sample'):
                return self.action_space.sample()
        # Fallback: return random integer action
        import random
        return random.randint(0, 9)
    
    @abstractmethod
    async def learn(self, experience: Experience) -> Dict[str, Any]:
        """Learn from experience - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        logger.debug(f"BaseAgent.learn() called - should be overridden by {self.__class__.__name__}")
        return {"loss": 0.0, "status": "no_learning"}
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        import json
        from pathlib import Path
        
        state = {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "config": self.config,
            "metrics": self.get_metrics()
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Agent {self.name} saved to {path}")
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent state - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        import json
        from pathlib import Path
        
        if not Path(path).exists():
            logger.warning(f"Agent file not found: {path}")
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.name = state.get("name", self.name)
        self.config.update(state.get("config", {}))
        
        logger.info(f"Agent {self.name} loaded from {path}")
    
    async def step(self, observation: Any) -> Tuple[Any, Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        
        self.status = AgentStatus.THINKING
        
        action = await self.act(observation)
        
        self.status = AgentStatus.ACTING
        self._current_episode_steps += 1
        
        info = {
            "agent": self.name,
            "type": self.agent_type.value,
            "step": self._current_episode_steps,
            "status": self.status.value
        }
        
        return action, info
    
    def add_experience(self, experience: Experience) -> None:
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        self._current_episode_reward += experience.reward
        
        if experience.done:
            won = experience.info.get("won", False)
            self.metrics.update_episode(
                self._current_episode_reward,
                self._current_episode_steps,
                won
            )
            self._current_episode_reward = 0.0
            self._current_episode_steps = 0
    
    async def train(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        if not self.experience_buffer:
            return {"error": "No experiences to train on"}
        
        self.status = AgentStatus.LEARNING
        
        batch_size = batch_size or self.config.get("batch_size", 32)
        batch_size = min(batch_size, len(self.experience_buffer))
        
        import random
        batch = random.sample(self.experience_buffer, batch_size)
        
        total_loss = 0.0
        for exp in batch:
            result = await self.learn(exp)
            if "loss" in result:
                total_loss += result["loss"]
        
        self.status = AgentStatus.IDLE
        
        return {
            "batch_size": batch_size,
            "avg_loss": total_loss / batch_size if batch_size > 0 else 0,
            "buffer_size": len(self.experience_buffer)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.status.value,
            "total_steps": self.metrics.total_steps,
            "total_episodes": self.metrics.total_episodes,
            "total_reward": self.metrics.total_reward,
            "avg_reward": self.metrics.avg_reward_per_episode,
            "avg_steps": self.metrics.avg_steps_per_episode,
            "win_rate": self.metrics.win_rate,
            "best_reward": self.metrics.best_episode_reward,
            "worst_reward": self.metrics.worst_episode_reward,
            "buffer_size": len(self.experience_buffer),
            "last_update": self.metrics.last_update.isoformat()
        }
    
    def reset(self) -> None:
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self.status = AgentStatus.IDLE
    
    def clear_buffer(self) -> None:
        self.experience_buffer.clear()
        logger.info(f"Experience buffer cleared for agent {self.name}")


class PolicyBasedAgent(BaseAgent):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = None
    
    @abstractmethod
    def _build_policy(self) -> Any:
        """Build policy - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        logger.warning(f"PolicyBasedAgent._build_policy() called - should be overridden by {self.__class__.__name__}")
        return None
    
    @abstractmethod
    def _update_policy(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Update policy - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        logger.warning(f"PolicyBasedAgent._update_policy() called - should be overridden by {self.__class__.__name__}")
        return {"loss": 0.0, "status": "no_update"}
    
    async def initialize(self) -> None:
        await super().initialize()
        self.policy = self._build_policy()
        logger.info(f"Policy initialized for agent {self.name}")
    
    async def act(self, observation: Any) -> Any:
        if self.policy is None:
            raise RuntimeError("Policy not initialized")
        
        return self._select_action(observation)
    
    @abstractmethod
    def _select_action(self, observation: Any) -> Any:
        """Select action using policy - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        logger.warning(f"PolicyBasedAgent._select_action() called - should be overridden by {self.__class__.__name__}")
        # Fallback: random action
        if hasattr(self, 'action_space') and hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        import random
        return random.randint(0, 9)


class ValueBasedAgent(BaseAgent):
    
    def __init__(self, *args, epsilon: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.q_values = {}
    
    @abstractmethod
    def _compute_q_values(self, observation: Any) -> Dict[Any, float]:
        """Compute Q-values for observation - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        logger.warning(f"ValueBasedAgent._compute_q_values() called - should be overridden by {self.__class__.__name__}")
        # Return uniform Q-values
        if hasattr(self, 'action_space'):
            if hasattr(self.action_space, 'n'):
                return {i: 0.0 for i in range(self.action_space.n)}
        return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}  # Default actions
    
    @abstractmethod
    def _update_q_values(self, experience: Experience) -> Dict[str, Any]:
        """Update Q-values based on experience - must be implemented by subclasses"""
        # Base implementation for abstract method - should be overridden
        logger.warning(f"ValueBasedAgent._update_q_values() called - should be overridden by {self.__class__.__name__}")
        return {"loss": 0.0, "status": "no_update"}
    
    async def act(self, observation: Any) -> Any:
        import random
        
        if random.random() < self.epsilon:
            if hasattr(self.action_space, 'sample'):
                return self.action_space.sample()
            else:
                return random.choice(list(self.action_space))
        
        q_values = self._compute_q_values(observation)
        
        if not q_values:
            if hasattr(self.action_space, 'sample'):
                return self.action_space.sample()
            else:
                return random.choice(list(self.action_space))
        
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    async def learn(self, experience: Experience) -> Dict[str, Any]:
        return self._update_q_values(experience)
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)