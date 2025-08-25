"""Complete PPO (Proximal Policy Optimization) Implementation for Nexus Framework"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import structlog
from collections import deque
import random

logger = structlog.get_logger()


@dataclass
class PPOConfig:
    """PPO algorithm configuration"""
    # Network architecture
    hidden_size: int = 256
    num_layers: int = 2
    activation: str = "relu"
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training settings
    num_epochs: int = 4
    num_mini_batches: int = 4
    batch_size: int = 64
    rollout_length: int = 2048
    
    # Environment settings
    num_envs: int = 8
    max_episode_length: int = 1000
    
    # Action space
    action_type: str = "discrete"  # "discrete" or "continuous"
    action_dim: int = 4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        super().__init__()
        self.config = config
        self.action_type = config.action_type
        
        # Shared layers
        layers = []
        input_dim = obs_dim
        
        for _ in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_size))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            input_dim = config.hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head
        if self.action_type == "discrete":
            self.actor = nn.Linear(config.hidden_size, action_dim)
        else:
            # Continuous action space
            self.actor_mean = nn.Linear(config.hidden_size, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head
        self.critic = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for output layers
        if self.action_type == "discrete":
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.shared(obs)
        
        if self.action_type == "discrete":
            action_logits = self.actor(features)
            value = self.critic(features)
            return action_logits, value
        else:
            action_mean = self.actor_mean(features)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            value = self.critic(features)
            return (action_mean, action_std), value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from observation"""
        if self.action_type == "discrete":
            action_logits, value = self.forward(obs)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
                action_probs = F.softmax(action_logits, dim=-1)
                log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1))).squeeze(-1)
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob, value.squeeze(-1)
        else:
            (action_mean, action_std), value = self.forward(obs)
            
            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
            
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given observations"""
        if self.action_type == "discrete":
            action_logits, values = self.forward(obs)
            
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            return log_probs, values.squeeze(-1), entropy
        else:
            (action_mean, action_std), values = self.forward(obs)
            
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], 
                 action_shape: Tuple[int, ...], device: str = "cpu"):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        
        # Buffers
        self.observations = torch.zeros((buffer_size,) + obs_shape).to(device)
        self.actions = torch.zeros((buffer_size,) + action_shape).to(device)
        self.rewards = torch.zeros(buffer_size).to(device)
        self.returns = torch.zeros(buffer_size).to(device)
        self.advantages = torch.zeros(buffer_size).to(device)
        self.values = torch.zeros(buffer_size).to(device)
        self.log_probs = torch.zeros(buffer_size).to(device)
        self.dones = torch.zeros(buffer_size).to(device)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: float,
            value: float, log_prob: float, done: bool):
        """Add experience to buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute returns and advantages using GAE"""
        last_gae_lambda = 0
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]
    
    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Get batches for training"""
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        
        batches = []
        for start_idx in range(0, self.size, batch_size):
            end_idx = min(start_idx + batch_size, self.size)
            batch_indices = indices[start_idx:end_idx]
            
            batch = {
                "observations": self.observations[batch_indices],
                "actions": self.actions[batch_indices],
                "old_log_probs": self.log_probs[batch_indices],
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices]
            }
            batches.append(batch)
        
        return batches
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """Complete PPO agent implementation"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Optional[PPOConfig] = None):
        """
        Initialize PPO agent
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: PPO configuration
        """
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Network
        self.network = ActorCriticNetwork(obs_dim, action_dim, self.config).to(self.config.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Rollout buffer
        action_shape = () if self.config.action_type == "discrete" else (action_dim,)
        self.rollout_buffer = RolloutBuffer(
            self.config.rollout_length,
            (obs_dim,),
            action_shape,
            self.config.device
        )
        
        # Statistics
        self.training_stats = {
            "total_steps": 0,
            "total_episodes": 0,
            "mean_reward": 0,
            "mean_episode_length": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "learning_rate": self.config.learning_rate
        }
        
        logger.info(f"Initialized PPO agent with obs_dim={obs_dim}, action_dim={action_dim}")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action given observation
        
        Args:
            observation: Current observation
            deterministic: Use deterministic policy
        
        Returns:
            Action and additional info
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.config.device)
            action, log_prob, value = self.network.get_action(obs_tensor, deterministic)
            
            action_np = action.cpu().numpy()[0]
            
            info = {
                "log_prob": log_prob.cpu().item(),
                "value": value.cpu().item()
            }
        
        return action_np, info
    
    def update(self) -> Dict[str, float]:
        """Update policy using collected rollout"""
        # Compute returns and advantages
        with torch.no_grad():
            last_obs = self.rollout_buffer.observations[self.rollout_buffer.size - 1]
            _, last_value = self.network(last_obs.unsqueeze(0))
            last_value = last_value.squeeze().item()
        
        self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )
        
        # Normalize advantages
        advantages = self.rollout_buffer.advantages[:self.rollout_buffer.size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.rollout_buffer.advantages[:self.rollout_buffer.size] = advantages
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for epoch in range(self.config.num_epochs):
            batches = self.rollout_buffer.get_batches(self.config.batch_size)
            
            for batch in batches:
                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch["observations"], batch["actions"]
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch["returns"])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Clear rollout buffer
        self.rollout_buffer.clear()
        
        # Update statistics
        self.training_stats["policy_loss"] = total_policy_loss / num_updates
        self.training_stats["value_loss"] = total_value_loss / num_updates
        self.training_stats["entropy"] = total_entropy / num_updates
        
        return self.training_stats
    
    def add_experience(self, obs: np.ndarray, action: np.ndarray, reward: float,
                      done: bool, info: Dict[str, Any]):
        """Add experience to rollout buffer"""
        obs_tensor = torch.FloatTensor(obs).to(self.config.device)
        action_tensor = torch.FloatTensor(action).to(self.config.device) if self.config.action_type == "continuous" else torch.LongTensor([action]).to(self.config.device)
        
        self.rollout_buffer.add(
            obs_tensor,
            action_tensor,
            reward,
            info["value"],
            info["log_prob"],
            done
        )
        
        self.training_stats["total_steps"] += 1
        
        if done:
            self.training_stats["total_episodes"] += 1
    
    def train_step(self, env, num_steps: int) -> Dict[str, float]:
        """
        Perform training step
        
        Args:
            env: Environment
            num_steps: Number of steps to collect
        
        Returns:
            Training statistics
        """
        obs = env.reset()[0]
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(num_steps):
            # Select action
            action, info = self.select_action(obs)
            
            # Step environment
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Add experience
            self.add_experience(obs, action, reward, done or truncated, info)
            
            # Update statistics
            current_episode_reward += reward
            current_episode_length += 1
            
            if done or truncated:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                obs = env.reset()[0]
            else:
                obs = next_obs
            
            # Update policy if buffer is full
            if self.rollout_buffer.size >= self.config.rollout_length:
                update_stats = self.update()
                
                if episode_rewards:
                    update_stats["mean_reward"] = np.mean(episode_rewards)
                    update_stats["mean_episode_length"] = np.mean(episode_lengths)
                
                return update_stats
        
        return self.training_stats
    
    def save(self, path: str):
        """Save agent"""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_stats": self.training_stats
        }, path)
        logger.info(f"PPO agent saved to {path}")
    
    def load(self, path: str):
        """Load agent"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint.get("training_stats", {})
        logger.info(f"PPO agent loaded from {path}")
    
    def set_learning_rate(self, lr: float):
        """Update learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.config.learning_rate = lr
        self.training_stats["learning_rate"] = lr