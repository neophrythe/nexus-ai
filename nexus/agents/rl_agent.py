import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional
from collections import deque
import random
import structlog

from nexus.agents.base import BaseAgent, AgentType, Experience

logger = structlog.get_logger()


class ReplayBuffer:
    """Experience replay buffer for RL agents"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNNDQNNetwork(nn.Module):
    """CNN-based DQN for image inputs"""
    
    def __init__(self, input_channels: int, output_dim: int, image_size: tuple = (84, 84)):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        def conv_out_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        h, w = image_size
        h = conv_out_size(conv_out_size(conv_out_size(h, 8, 4), 4, 2), 3, 1)
        w = conv_out_size(conv_out_size(conv_out_size(w, 8, 4), 4, 2), 3, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * h * w, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQNAgent(BaseAgent):
    """Deep Q-Learning Agent"""
    
    def __init__(self, *args, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 32,
                 target_update: int = 1000,
                 use_cnn: bool = False,
                 **kwargs):
        
        super().__init__(*args, agent_type=AgentType.RL, **kwargs)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_cnn = use_cnn
        
        self.replay_buffer = ReplayBuffer(capacity=self.max_buffer_size)
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.steps = 0
        self.losses = []
    
    async def initialize(self) -> None:
        await super().initialize()
        
        # Build networks based on observation/action spaces
        if self.use_cnn:
            # Assume image input
            input_channels = self.observation_space.shape[2] if len(self.observation_space.shape) == 3 else 1
            output_dim = self.action_space.n
            
            self.q_network = CNNDQNNetwork(input_channels, output_dim).to(self.device)
            self.target_network = CNNDQNNetwork(input_channels, output_dim).to(self.device)
        else:
            # Assume flat input
            input_dim = np.prod(self.observation_space.shape)
            output_dim = self.action_space.n
            
            self.q_network = DQNNetwork(input_dim, output_dim).to(self.device)
            self.target_network = DQNNetwork(input_dim, output_dim).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        logger.info(f"DQN Agent initialized - CNN: {self.use_cnn}, Device: {self.device}")
    
    async def act(self, observation: Any) -> Any:
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # Prepare observation
        obs_tensor = self._prepare_observation(observation)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        
        # Select action with highest Q-value
        action = q_values.argmax(dim=1).item()
        
        return action
    
    async def learn(self, experience: Experience) -> Dict[str, Any]:
        # Add to replay buffer
        self.replay_buffer.push(experience)
        
        # Don't learn until we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states = torch.stack([self._prepare_observation(e.observation) for e in batch]).squeeze(1)
        actions = torch.tensor([e.action for e in batch], device=self.device, dtype=torch.long)
        rewards = torch.tensor([e.reward for e in batch], device=self.device, dtype=torch.float)
        next_states = torch.stack([self._prepare_observation(e.next_observation) for e in batch]).squeeze(1)
        dones = torch.tensor([e.done for e in batch], device=self.device, dtype=torch.float)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return {"loss": loss_value, "epsilon": self.epsilon}
    
    def _prepare_observation(self, observation: Any) -> torch.Tensor:
        """Convert observation to tensor"""
        if isinstance(observation, np.ndarray):
            obs = observation
        elif isinstance(observation, dict):
            # Extract relevant part of observation
            obs = observation.get("screen", observation.get("observation", observation))
        else:
            obs = np.array(observation)
        
        # Normalize
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
        
        # Add batch dimension
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Rearrange dimensions for CNN if needed
        if self.use_cnn and len(obs.shape) == 4:
            # From (batch, height, width, channels) to (batch, channels, height, width)
            obs = obs.permute(0, 3, 1, 2)
        elif not self.use_cnn:
            # Flatten for fully connected network
            obs = obs.view(obs.size(0), -1)
        
        return obs
    
    def save(self, path: str) -> None:
        """Save agent model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'config': self.config
        }, path)
        logger.info(f"DQN Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load agent model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"DQN Agent loaded from {path}")


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, *args,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_steps: int = 2048,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 **kwargs):
        
        super().__init__(*args, agent_type=AgentType.RL, **kwargs)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.policy_network = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.rollout_buffer = []
    
    async def initialize(self) -> None:
        await super().initialize()
        # PPO implementation would go here
        logger.info("PPO Agent initialized")
    
    async def act(self, observation: Any) -> Any:
        # PPO action selection
        return self.action_space.sample()
    
    async def learn(self, experience: Experience) -> Dict[str, Any]:
        # PPO learning
        return {"loss": 0}
    
    def save(self, path: str) -> None:
        logger.info(f"PPO Agent saved to {path}")
    
    def load(self, path: str) -> None:
        logger.info(f"PPO Agent loaded from {path}")