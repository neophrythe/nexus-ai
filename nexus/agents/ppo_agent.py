"""PPO (Proximal Policy Optimization) agent implementation adapted from SerpentAI"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import structlog

from nexus.agents.base import BaseAgent, AgentType

logger = structlog.get_logger()


@dataclass
class RolloutBufferSample:
    """Single sample from rollout buffer"""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Rollout buffer for PPO training"""
    
    def __init__(self, buffer_size: int, observation_space: Any, action_space: Any,
                 device: torch.device, gae_lambda: float = 0.95, gamma: float = 0.99):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        self.reset()
    
    def reset(self):
        """Reset the buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = None
        self.returns = None
        self.pos = 0
    
    @property
    def full(self) -> bool:
        """Check if buffer is full"""
        return self.pos >= self.buffer_size
    
    def add(self, obs: np.ndarray, action: Union[int, np.ndarray], reward: float,
            done: bool, value: torch.Tensor, log_prob: torch.Tensor):
        """Add experience to buffer"""
        if self.full:
            return
        
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value.cpu().numpy())
        self.log_probs.append(log_prob.cpu().numpy())
        self.pos += 1
    
    def compute_returns_and_advantage(self, last_value: torch.Tensor):
        """Compute returns and GAE advantages"""
        last_value = last_value.cpu().numpy()
        
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values).squeeze()
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam
        
        # Compute returns
        returns = advantages + values
        
        # Convert to tensors
        self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def get(self, batch_size: int):
        """Get batches for training"""
        indices = np.random.permutation(self.pos)
        
        # Convert to tensors
        obs_tensor = torch.tensor(np.array(self.observations), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(np.array(self.actions), device=self.device)
        values_tensor = torch.tensor(np.array(self.values), dtype=torch.float32, device=self.device)
        log_probs_tensor = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        
        # Yield batches
        start_idx = 0
        while start_idx < self.pos:
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield RolloutBufferSample(
                observations=obs_tensor[batch_indices],
                actions=actions_tensor[batch_indices],
                old_values=values_tensor[batch_indices],
                old_log_prob=log_probs_tensor[batch_indices],
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices]
            )
            
            start_idx += batch_size


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, obs_shape: Tuple[int, ...], action_dim: int,
                 hidden_sizes: List[int] = [256, 256], activation: str = 'relu',
                 continuous_actions: bool = False):
        super().__init__()
        
        self.continuous_actions = continuous_actions
        
        # Select activation function
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU
        }
        activation_fn = activations.get(activation, nn.ReLU)
        
        # Feature extractor
        if len(obs_shape) == 3:  # Image input
            # CNN feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
                activation_fn(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                activation_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                activation_fn(),
                nn.Flatten()
            )
            
            # Calculate feature dimension
            with torch.no_grad():
                sample = torch.zeros(1, *obs_shape)
                features = self.feature_extractor(sample)
                feature_dim = features.shape[1]
        else:
            # MLP feature extractor
            input_dim = np.prod(obs_shape)
            self.feature_extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, hidden_sizes[0]),
                activation_fn()
            )
            feature_dim = hidden_sizes[0]
        
        # Shared hidden layers
        hidden_layers = []
        prev_size = feature_dim
        for hidden_size in hidden_sizes[1:]:
            hidden_layers.append(nn.Linear(prev_size, hidden_size))
            hidden_layers.append(activation_fn())
            prev_size = hidden_size
        
        self.shared_net = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()
        
        # Actor head
        if continuous_actions:
            self.actor_mean = nn.Linear(prev_size, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Linear(prev_size, action_dim)
        
        # Critic head
        self.critic = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits/params and value"""
        features = self.feature_extractor(obs)
        hidden = self.shared_net(features)
        
        if self.continuous_actions:
            action_mean = self.actor_mean(hidden)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            value = self.critic(hidden)
            return action_mean, action_logstd, value
        else:
            action_logits = self.actor(hidden)
            value = self.critic(hidden)
            return action_logits, value
    
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using current policy"""
        if self.continuous_actions:
            action_mean, action_logstd, value = self.forward(obs)
            action_std = torch.exp(action_logstd)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            action_logits, value = self.forward(obs)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).unsqueeze(-1)
        
        return action, action_log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training"""
        if self.continuous_actions:
            action_mean, action_logstd, value = self.forward(obs)
            action_std = torch.exp(action_logstd)
            
            dist = Normal(action_mean, action_std)
            action_log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1)
        else:
            action_logits, value = self.forward(obs)
            dist = Categorical(logits=action_logits)
            action_log_prob = dist.log_prob(actions).unsqueeze(-1)
            entropy = dist.entropy()
        
        return value, action_log_prob, entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate"""
        features = self.feature_extractor(obs)
        hidden = self.shared_net(features)
        return self.critic(hidden)


class PPOAgent(BaseAgent):
    """PPO Agent with actor-critic architecture"""
    
    def __init__(self, *args,
                 # PPO specific parameters
                 clip_param: float = 0.2,
                 ppo_epochs: int = 4,
                 num_mini_batch: int = 32,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 learning_rate: float = 3e-4,
                 max_grad_norm: float = 0.5,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 use_gae: bool = True,
                 rollout_buffer_size: int = 2048,
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu',
                 continuous_actions: bool = False,
                 **kwargs):
        
        super().__init__(*args, agent_type=AgentType.RL, **kwargs)
        
        # PPO hyperparameters
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        self.rollout_buffer_size = rollout_buffer_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.continuous_actions = continuous_actions
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Storage for last observation/action
        self.last_observation = None
        self.last_action = None
        self.last_action_log_prob = None
        self.last_value = None
        
        # Training metrics
        self.training_step = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
    
    async def initialize(self) -> None:
        """Initialize PPO agent"""
        await super().initialize()
        
        # Get observation and action dimensions
        obs_shape = self._get_obs_shape()
        action_dim = self._get_action_dim()
        
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(
            obs_shape=obs_shape,
            action_dim=action_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            continuous_actions=self.continuous_actions
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # Initialize rollout buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.rollout_buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma
        )
        
        logger.info(f"PPO Agent initialized on {self.device}")
        logger.info(f"Observation shape: {obs_shape}, Action dim: {action_dim}")
    
    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Get observation shape from space"""
        if hasattr(self.observation_space, 'shape'):
            return self.observation_space.shape
        else:
            return (self.observation_space.n,)
    
    def _get_action_dim(self) -> int:
        """Get action dimension from space"""
        if self.continuous_actions:
            return self.action_space.shape[0]
        elif hasattr(self.action_space, 'n'):
            return self.action_space.n
        else:
            return 1
    
    async def act(self, observation: Any) -> Any:
        """Select action using current policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Get action from actor network
            action, action_log_prob, value = self.actor_critic.act(obs_tensor)
            
            # Store for later use in learning
            self.last_observation = observation
            self.last_action = action.cpu().numpy()[0]
            self.last_action_log_prob = action_log_prob
            self.last_value = value
            
            # Return action for environment
            if self.continuous_actions:
                return self.last_action
            else:
                return int(self.last_action)
    
    async def observe(self, reward: float, next_observation: Any, done: bool, info: Dict[str, Any] = None) -> None:
        """Observe environment feedback"""
        await super().observe(reward, next_observation, done, info)
        
        # Add to rollout buffer if we have a previous action
        if self.last_observation is not None:
            self.rollout_buffer.add(
                obs=self.last_observation,
                action=self.last_action,
                reward=reward,
                done=done,
                value=self.last_value,
                log_prob=self.last_action_log_prob
            )
            
            # Train if buffer is full
            if self.rollout_buffer.full:
                await self._update()
    
    async def _update(self) -> Dict[str, float]:
        """Update policy using PPO algorithm"""
        # Compute returns and advantages
        with torch.no_grad():
            last_obs = torch.FloatTensor(self.last_observation).unsqueeze(0).to(self.device)
            last_value = self.actor_critic.get_value(last_obs)
        
        self.rollout_buffer.compute_returns_and_advantage(last_value)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # PPO training loop
        for epoch in range(self.ppo_epochs):
            for rollout_data in self.rollout_buffer.get(self.num_mini_batch):
                observations = rollout_data.observations
                actions = rollout_data.actions
                old_values = rollout_data.old_values
                old_log_prob = rollout_data.old_log_prob
                advantages = rollout_data.advantages
                returns = rollout_data.returns
                
                # Evaluate actions
                values, log_prob, entropy = self.actor_critic.evaluate_actions(observations, actions)
                values = values.flatten()
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_prob.flatten() - old_log_prob.flatten())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                values_pred = values
                value_loss = F.mse_loss(returns, values_pred)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())
                
                # Clip fraction for monitoring
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_param).float()).item()
                clip_fractions.append(clip_fraction)
        
        # Reset rollout buffer
        self.rollout_buffer.reset()
        
        # Update training step
        self.training_step += 1
        
        # Store metrics
        self.policy_losses.append(np.mean(policy_losses))
        self.value_losses.append(np.mean(value_losses))
        self.entropies.append(np.mean(entropy_losses))
        
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.learning_rate,
            'training_step': self.training_step
        }
        
        logger.debug(f"PPO Update - Step {self.training_step}: {metrics}")
        
        return metrics
    
    async def save(self, path: str) -> None:
        """Save agent state"""
        state = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'hyperparameters': {
                'clip_param': self.clip_param,
                'ppo_epochs': self.ppo_epochs,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }
        torch.save(state, path)
        logger.info(f"PPO agent saved to {path}")
    
    async def load(self, path: str) -> None:
        """Load agent state"""
        state = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(state['actor_critic_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.training_step = state.get('training_step', 0)
        logger.info(f"PPO agent loaded from {path}")