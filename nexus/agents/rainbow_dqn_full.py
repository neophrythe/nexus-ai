"""Complete Rainbow DQN Implementation for Nexus Framework"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import structlog
from collections import deque
import random
import math

logger = structlog.get_logger()


@dataclass
class RainbowConfig:
    """Rainbow DQN configuration"""
    # Network architecture
    hidden_size: int = 512
    num_layers: int = 2
    
    # Rainbow components
    use_double_dqn: bool = True
    use_dueling: bool = True
    use_noisy: bool = True
    use_categorical: bool = True
    use_n_step: bool = True
    use_prioritized: bool = True
    
    # Categorical DQN
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    
    # N-step returns
    n_step: int = 3
    
    # Prioritized replay
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.001
    priority_epsilon: float = 1e-6
    
    # Training parameters
    learning_rate: float = 6.25e-5
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 100000
    learning_starts: int = 10000
    target_update_interval: int = 8000
    
    # Epsilon greedy (if not using noisy nets)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 100000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10000


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Register buffers for noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """Rainbow DQN network with all components"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: RainbowConfig):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        
        # Feature extraction layers
        self.features = nn.Sequential()
        input_dim = obs_dim
        
        for i in range(config.num_layers):
            if config.use_noisy:
                self.features.add_module(f"noisy_{i}", NoisyLinear(input_dim, config.hidden_size))
            else:
                self.features.add_module(f"linear_{i}", nn.Linear(input_dim, config.hidden_size))
            self.features.add_module(f"relu_{i}", nn.ReLU())
            input_dim = config.hidden_size
        
        # Dueling architecture
        if config.use_dueling:
            # Value stream
            if config.use_noisy:
                self.value_stream = nn.Sequential(
                    NoisyLinear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    NoisyLinear(config.hidden_size, config.num_atoms if config.use_categorical else 1)
                )
            else:
                self.value_stream = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, config.num_atoms if config.use_categorical else 1)
                )
            
            # Advantage stream
            if config.use_noisy:
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    NoisyLinear(config.hidden_size, action_dim * (config.num_atoms if config.use_categorical else 1))
                )
            else:
                self.advantage_stream = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size, action_dim * (config.num_atoms if config.use_categorical else 1))
                )
        else:
            # Standard Q-network
            if config.use_noisy:
                self.q_values = NoisyLinear(config.hidden_size, action_dim * (config.num_atoms if config.use_categorical else 1))
            else:
                self.q_values = nn.Linear(config.hidden_size, action_dim * (config.num_atoms if config.use_categorical else 1))
        
        # Support for categorical DQN
        if config.use_categorical:
            self.register_buffer("support", torch.linspace(config.v_min, config.v_max, config.num_atoms))
            self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.features(x)
        
        if self.config.use_dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            if self.config.use_categorical:
                value = value.view(-1, 1, self.config.num_atoms)
                advantage = advantage.view(-1, self.action_dim, self.config.num_atoms)
                
                # Combine streams
                q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
                
                # Apply softmax
                q_dist = self.softmax(q_dist)
                
                return q_dist
            else:
                # Standard dueling
                q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
                return q_values
        else:
            if self.config.use_categorical:
                q_dist = self.q_values(features).view(-1, self.action_dim, self.config.num_atoms)
                q_dist = self.softmax(q_dist)
                return q_dist
            else:
                return self.q_values(features)
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Get Q-values from network output"""
        output = self.forward(x)
        
        if self.config.use_categorical:
            # Calculate Q-values as expectation over distribution
            q_values = (output * self.support).sum(dim=-1)
            return q_values
        else:
            return output
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.config.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch with priorities"""
        if self.size == 0:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get samples
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def increase_beta(self, increment: float):
        """Increase beta for importance sampling"""
        self.beta = min(1.0, self.beta + increment)


class NStepBuffer:
    """N-step return buffer"""
    
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, experience: Tuple):
        """Add experience to n-step buffer"""
        self.buffer.append(experience)
    
    def get_n_step_return(self) -> Tuple:
        """Calculate n-step return"""
        if len(self.buffer) < self.n_step:
            return None
        
        # Calculate n-step return
        n_step_return = 0
        for i, (_, _, reward, _, done) in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * reward
            if done:
                break
        
        # Get first state and last next_state
        first_state = self.buffer[0][0]
        action = self.buffer[0][1]
        last_next_state = self.buffer[-1][3]
        done = self.buffer[-1][4]
        
        return first_state, action, n_step_return, last_next_state, done
    
    def is_ready(self) -> bool:
        """Check if buffer is ready for n-step return"""
        return len(self.buffer) == self.n_step


class RainbowDQNAgent:
    """Complete Rainbow DQN agent"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Optional[RainbowConfig] = None):
        """
        Initialize Rainbow DQN agent
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: Rainbow configuration
        """
        self.config = config or RainbowConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Networks
        self.q_network = RainbowNetwork(obs_dim, action_dim, self.config).to(self.config.device)
        self.target_network = RainbowNetwork(obs_dim, action_dim, self.config).to(self.config.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Replay buffer
        if self.config.use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.config.buffer_size,
                self.config.priority_alpha,
                self.config.priority_beta
            )
        else:
            self.replay_buffer = deque(maxlen=self.config.buffer_size)
        
        # N-step buffer
        if self.config.use_n_step:
            self.n_step_buffer = NStepBuffer(self.config.n_step, self.config.gamma)
        
        # Statistics
        self.training_stats = {
            "total_steps": 0,
            "total_episodes": 0,
            "epsilon": self.config.epsilon_start,
            "loss": 0,
            "q_value": 0,
            "learning_rate": self.config.learning_rate
        }
        
        logger.info(f"Initialized Rainbow DQN agent with obs_dim={obs_dim}, action_dim={action_dim}")
    
    def select_action(self, observation: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy or noisy networks
        
        Args:
            observation: Current observation
            epsilon: Override epsilon value
        
        Returns:
            Selected action
        """
        if epsilon is None:
            if self.config.use_noisy:
                epsilon = 0  # No epsilon-greedy with noisy networks
            else:
                # Calculate epsilon based on decay
                steps = self.training_stats["total_steps"]
                epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                         math.exp(-steps / self.config.epsilon_decay)
                self.training_stats["epsilon"] = epsilon
        
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.config.device)
            q_values = self.q_network.get_q_values(obs_tensor)
            return q_values.argmax(dim=1).item()
    
    def add_experience(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool):
        """Add experience to replay buffer"""
        if self.config.use_n_step:
            self.n_step_buffer.add((state, action, reward, next_state, done))
            
            if self.n_step_buffer.is_ready() or done:
                n_step_experience = self.n_step_buffer.get_n_step_return()
                if n_step_experience:
                    if self.config.use_prioritized:
                        self.replay_buffer.add(*n_step_experience)
                    else:
                        self.replay_buffer.append(n_step_experience)
        else:
            if self.config.use_prioritized:
                self.replay_buffer.add(state, action, reward, next_state, done)
            else:
                self.replay_buffer.append((state, action, reward, next_state, done))
        
        self.training_stats["total_steps"] += 1
        
        if done:
            self.training_stats["total_episodes"] += 1
            if self.config.use_n_step:
                self.n_step_buffer.buffer.clear()
    
    def update(self) -> Dict[str, float]:
        """Update Q-network"""
        if self.config.use_prioritized:
            if self.replay_buffer.size < self.config.learning_starts:
                return self.training_stats
            
            batch = self.replay_buffer.sample(self.config.batch_size)
            if batch is None:
                return self.training_stats
            
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights = torch.FloatTensor(weights).to(self.config.device)
        else:
            if len(self.replay_buffer) < self.config.learning_starts:
                return self.training_stats
            
            batch = random.sample(self.replay_buffer, self.config.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.ones(self.config.batch_size).to(self.config.device)
            indices = None
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).to(self.config.device)
        
        # Reset noise if using noisy networks
        if self.config.use_noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Calculate loss
        if self.config.use_categorical:
            loss, td_errors = self._categorical_dqn_loss(states, actions, rewards, next_states, dones)
        else:
            loss, td_errors = self._standard_dqn_loss(states, actions, rewards, next_states, dones)
        
        # Apply importance sampling weights
        loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.config.use_prioritized and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
            self.replay_buffer.increase_beta(self.config.priority_beta_increment)
        
        # Update statistics
        self.training_stats["loss"] = loss.item()
        
        with torch.no_grad():
            q_values = self.q_network.get_q_values(states)
            self.training_stats["q_value"] = q_values.mean().item()
        
        return self.training_stats
    
    def _standard_dqn_loss(self, states, actions, rewards, next_states, dones):
        """Calculate standard DQN loss"""
        current_q_values = self.q_network.get_q_values(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select actions
                next_actions = self.q_network.get_q_values(next_states).argmax(dim=1)
                next_q_values = self.target_network.get_q_values(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN
                next_q_values = self.target_network.get_q_values(next_states).max(dim=1, keepdim=True)[0]
            
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        td_errors = target_q_values - current_q_values
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none').squeeze()
        
        return loss, td_errors.squeeze()
    
    def _categorical_dqn_loss(self, states, actions, rewards, next_states, dones):
        """Calculate categorical DQN loss"""
        # Get current distribution
        current_dist = self.q_network.forward(states)
        current_dist = current_dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.num_atoms)).squeeze(1)
        
        with torch.no_grad():
            # Get next distribution
            if self.config.use_double_dqn:
                next_actions = self.q_network.get_q_values(next_states).argmax(dim=1)
                next_dist = self.target_network.forward(next_states)
                next_dist = next_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.num_atoms)).squeeze(1)
            else:
                next_dist = self.target_network.forward(next_states)
                next_q_values = (next_dist * self.q_network.support).sum(dim=-1)
                next_actions = next_q_values.argmax(dim=1)
                next_dist = next_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.num_atoms)).squeeze(1)
            
            # Project distribution
            target_dist = self._project_distribution(rewards, dones, next_dist)
        
        # Calculate cross-entropy loss
        loss = -(target_dist * current_dist.log()).sum(dim=1)
        
        # Calculate TD errors for prioritized replay
        with torch.no_grad():
            current_q = (current_dist * self.q_network.support).sum(dim=1)
            target_q = (target_dist * self.q_network.support).sum(dim=1)
            td_errors = target_q - current_q
        
        return loss, td_errors
    
    def _project_distribution(self, rewards, dones, next_dist):
        """Project distribution for categorical DQN"""
        batch_size = rewards.size(0)
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)
        support = self.q_network.support
        
        # Compute projected support
        next_support = rewards.unsqueeze(1) + self.config.gamma * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
        next_support = next_support.clamp(self.config.v_min, self.config.v_max)
        
        # Compute projection indices
        b = (next_support - self.config.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix for when b is exactly on grid
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.config.num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability
        projected_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * self.config.num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, self.config.num_atoms).to(self.config.device)
        
        projected_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        projected_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
        return projected_dist
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated")
    
    def train_step(self, env, num_steps: int) -> Dict[str, float]:
        """
        Perform training step
        
        Args:
            env: Environment
            num_steps: Number of steps to train
        
        Returns:
            Training statistics
        """
        obs = env.reset()[0]
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Select action
            action = self.select_action(obs)
            
            # Step environment
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Add experience
            self.add_experience(obs, action, reward, next_obs, done or truncated)
            
            # Update
            if step % 4 == 0:  # Update every 4 steps
                self.update()
            
            # Update target network
            if self.training_stats["total_steps"] % self.config.target_update_interval == 0:
                self.update_target_network()
            
            # Track episode statistics
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                logger.info(f"Episode {self.training_stats['total_episodes']}: "
                          f"reward={episode_reward:.2f}, length={episode_length}")
                obs = env.reset()[0]
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
        
        return self.training_stats
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": self.training_stats
        }, path)
        logger.info(f"Rainbow DQN agent saved to {path}")
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", {})
        logger.info(f"Rainbow DQN agent loaded from {path}")