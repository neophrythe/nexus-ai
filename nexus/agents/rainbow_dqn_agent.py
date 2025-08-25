"""Rainbow DQN agent implementation adapted from SerpentAI

Rainbow DQN combines multiple DQN improvements:
- Double Q-learning
- Prioritized experience replay
- Dueling networks
- Multi-step learning
- Distributional RL
- Noisy networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

from nexus.agents.base import BaseAgent, AgentType

logger = structlog.get_logger()


# ============= Neural Network Components =============

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """Dueling network architecture for Rainbow DQN"""
    
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int,
                 atoms: int = 51, hidden_size: int = 512, noisy_std: float = 0.5):
        super().__init__()
        self.num_actions = num_actions
        self.atoms = atoms
        
        # Determine if input is image or vector
        self.is_image = len(input_shape) == 3
        
        if self.is_image:
            # CNN feature extractor
            channels = input_shape[0]
            self.features = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate feature dimension
            with torch.no_grad():
                sample = torch.zeros(1, *input_shape)
                feature_dim = self.features(sample).shape[1]
        else:
            # MLP feature extractor
            input_dim = np.prod(input_shape)
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden_size),
                nn.ReLU()
            )
            feature_dim = hidden_size
        
        # Dueling streams with noisy layers
        self.advantage_hidden = NoisyLinear(feature_dim, hidden_size, noisy_std)
        self.advantage_out = NoisyLinear(hidden_size, num_actions * atoms, noisy_std)
        
        self.value_hidden = NoisyLinear(feature_dim, hidden_size, noisy_std)
        self.value_out = NoisyLinear(hidden_size, atoms, noisy_std)
    
    def forward(self, x: torch.Tensor, log: bool = False) -> torch.Tensor:
        """Forward pass returning distribution over atoms"""
        features = self.features(x)
        
        # Value stream
        value = F.relu(self.value_hidden(features))
        value = self.value_out(value).view(-1, 1, self.atoms)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(-1, self.num_actions, self.atoms)
        
        # Combine streams
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        if log:
            return F.log_softmax(q_dist, dim=2)
        else:
            return F.softmax(q_dist, dim=2)
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()


# ============= Prioritized Replay Memory =============

class SumTree:
    """Sum tree for efficient prioritized sampling"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority"""
        return self.tree[0]
    
    def add(self, priority: float, data: Any):
        """Add new experience"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get experience by priority sum"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayMemory:
    """Prioritized experience replay memory"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """Add experience with max priority"""
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with importance weights"""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_increment])
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)
        
        # Calculate importance weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        # Unpack batch
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return idxs, states, actions, rewards, next_states, dones, is_weight
    
    def update_priorities(self, idxs: List[int], priorities: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, priority in zip(idxs, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Get current size of memory"""
        return self.tree.n_entries


# ============= Rainbow DQN Agent =============

class RainbowDQNAgent(BaseAgent):
    """Rainbow DQN Agent combining all improvements"""
    
    def __init__(self, *args,
                 # Rainbow specific parameters
                 atoms: int = 51,
                 v_min: float = -10,
                 v_max: float = 10,
                 replay_capacity: int = 100000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 multi_step: int = 3,
                 priority_alpha: float = 0.6,
                 priority_beta: float = 0.4,
                 priority_beta_increment: float = 0.001,
                 hidden_size: int = 512,
                 noisy_std: float = 0.5,
                 learning_rate: float = 6.25e-5,
                 adam_epsilon: float = 1.5e-4,
                 max_grad_norm: float = 10,
                 target_update_freq: int = 10000,
                 learning_starts: int = 50000,
                 **kwargs):
        
        super().__init__(*args, agent_type=AgentType.RL, **kwargs)
        
        # Rainbow hyperparameters
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.multi_step = multi_step
        self.hidden_size = hidden_size
        self.noisy_std = noisy_std
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Distributional RL
        self.support = torch.linspace(v_min, v_max, atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (atoms - 1)
        
        # Training state
        self.training_steps = 0
        self.epsilon = 0.0  # Not used due to noisy networks
    
    async def initialize(self) -> None:
        """Initialize Rainbow DQN agent"""
        await super().initialize()
        
        # Get observation and action dimensions
        obs_shape = self._get_obs_shape()
        num_actions = self._get_num_actions()
        
        # Initialize networks
        self.online_net = DuelingNetwork(
            input_shape=obs_shape,
            num_actions=num_actions,
            atoms=self.atoms,
            hidden_size=self.hidden_size,
            noisy_std=self.noisy_std
        ).to(self.device)
        
        self.target_net = DuelingNetwork(
            input_shape=obs_shape,
            num_actions=num_actions,
            atoms=self.atoms,
            hidden_size=self.hidden_size,
            noisy_std=self.noisy_std
        ).to(self.device)
        
        # Initialize target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )
        
        # Initialize prioritized replay memory
        self.memory = PrioritizedReplayMemory(
            capacity=self.config.get('replay_capacity', 100000),
            alpha=self.config.get('priority_alpha', 0.6),
            beta=self.config.get('priority_beta', 0.4),
            beta_increment=self.config.get('priority_beta_increment', 0.001)
        )
        
        # Multi-step buffer
        self.multi_step_buffer = []
        
        logger.info(f"Rainbow DQN Agent initialized on {self.device}")
        logger.info(f"Observation shape: {obs_shape}, Actions: {num_actions}")
    
    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Get observation shape from space"""
        if hasattr(self.observation_space, 'shape'):
            return self.observation_space.shape
        else:
            return (self.observation_space.n,)
    
    def _get_num_actions(self) -> int:
        """Get number of actions from space"""
        if hasattr(self.action_space, 'n'):
            return self.action_space.n
        else:
            return self.action_space.shape[0]
    
    async def act(self, observation: Any) -> Any:
        """Select action using noisy networks"""
        # Reset noise for exploration
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Get Q-distribution
            q_dist = self.online_net(obs_tensor)
            
            # Calculate Q-values as expected values
            q_values = (q_dist * self.support).sum(2)
            
            # Select action with highest Q-value
            action = q_values.argmax(1).item()
        
        return action
    
    async def observe(self, reward: float, next_observation: Any, done: bool, info: Dict[str, Any] = None) -> None:
        """Observe environment feedback and store in replay memory"""
        await super().observe(reward, next_observation, done, info)
        
        if self.last_observation is not None:
            # Add to multi-step buffer
            self.multi_step_buffer.append((
                self.last_observation,
                self.last_action,
                reward,
                next_observation,
                done
            ))
            
            # Process multi-step returns
            if len(self.multi_step_buffer) >= self.multi_step or done:
                # Calculate n-step return
                n_step_state = self.multi_step_buffer[0][0]
                n_step_action = self.multi_step_buffer[0][1]
                n_step_reward = sum([self.gamma ** i * t[2] for i, t in enumerate(self.multi_step_buffer)])
                n_step_next_state = self.multi_step_buffer[-1][3]
                n_step_done = self.multi_step_buffer[-1][4]
                
                # Add to replay memory
                self.memory.push(
                    n_step_state,
                    n_step_action,
                    n_step_reward,
                    n_step_next_state,
                    n_step_done
                )
                
                # Remove oldest transition
                if not done:
                    self.multi_step_buffer.pop(0)
                else:
                    self.multi_step_buffer.clear()
            
            # Train if enough experiences
            if len(self.memory) >= self.batch_size and self.training_steps >= self.learning_starts:
                await self._train()
            
            self.training_steps += 1
            
            # Update target network
            if self.training_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                logger.debug(f"Target network updated at step {self.training_steps}")
    
    async def _train(self) -> Dict[str, float]:
        """Train the Rainbow DQN agent"""
        # Sample batch from replay memory
        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current Q-distribution
        log_q_dist = self.online_net(states, log=True)
        log_q_dist_a = log_q_dist[range(self.batch_size), actions]
        
        with torch.no_grad():
            # Double Q-learning: use online network to select actions
            next_q_dist = self.online_net(next_states)
            next_q_values = (next_q_dist * self.support).sum(2)
            next_actions = next_q_values.argmax(1)
            
            # Use target network to evaluate actions
            next_q_dist_target = self.target_net(next_states)
            next_q_dist_a = next_q_dist_target[range(self.batch_size), next_actions]
            
            # Compute projected distribution
            Tz = rewards.unsqueeze(1) + (1 - dones).unsqueeze(1) * (self.gamma ** self.multi_step) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            
            # Compute projection indices
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Fix edge cases
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            
            # Distribute probability
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.atoms).to(self.device)
            
            m.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist_a * (b - l.float())).view(-1))
        
        # Cross-entropy loss
        loss = -(m * log_q_dist_a).sum(1)
        
        # Apply importance weights
        weighted_loss = (weights * loss).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update priorities
        priorities = loss.detach().cpu().numpy()
        self.memory.update_priorities(idxs, priorities)
        
        return {
            'loss': weighted_loss.item(),
            'mean_q': next_q_values.mean().item(),
            'training_steps': self.training_steps
        }
    
    async def save(self, path: str) -> None:
        """Save agent state"""
        state = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'hyperparameters': {
                'atoms': self.atoms,
                'v_min': self.v_min,
                'v_max': self.v_max,
                'gamma': self.gamma,
                'multi_step': self.multi_step,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(state, path)
        logger.info(f"Rainbow DQN agent saved to {path}")
    
    async def load(self, path: str) -> None:
        """Load agent state"""
        state = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state['online_net_state_dict'])
        self.target_net.load_state_dict(state['target_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.training_steps = state.get('training_steps', 0)
        logger.info(f"Rainbow DQN agent loaded from {path}")