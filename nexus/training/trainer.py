import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
import time
import json
from datetime import datetime
import structlog
import numpy as np

from nexus.agents.base import BaseAgent, Experience
from nexus.environments.base import GameEnvironment

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Configuration for training"""
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    save_frequency: int = 100
    evaluation_frequency: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    render: bool = False
    verbose: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 50
    target_reward: Optional[float] = None
    use_tensorboard: bool = True
    parallel_envs: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "max_steps": self.max_steps_per_episode,
            "save_freq": self.save_frequency,
            "eval_freq": self.evaluation_frequency,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "render": self.render,
            "early_stopping": self.early_stopping,
            "target_reward": self.target_reward
        }


class Trainer:
    """Training orchestrator for agents"""
    
    def __init__(self,
                 agent: BaseAgent,
                 environment: GameEnvironment,
                 config: TrainingConfig = None):
        
        self.agent = agent
        self.environment = environment
        self.config = config or TrainingConfig()
        
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_start_time = None
        self.best_reward = float('-inf')
        self.patience_counter = 0
        
        # Setup directories
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # Tensorboard writer
        self.writer = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir / "tensorboard")
            except ImportError:
                logger.warning("Tensorboard not available")
    
    async def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.episodes} episodes")
        self.training_start_time = time.time()
        
        try:
            for episode in range(self.config.episodes):
                # Run episode
                episode_reward, episode_length = await self._run_episode(episode)
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Log progress
                if self.config.verbose and episode % 10 == 0:
                    self._log_progress(episode)
                
                # Save checkpoint
                if episode % self.config.save_frequency == 0 and episode > 0:
                    await self._save_checkpoint(episode)
                
                # Evaluate
                if episode % self.config.evaluation_frequency == 0 and episode > 0:
                    await self._evaluate(episode)
                
                # Early stopping
                if self._should_stop_early(episode_reward):
                    logger.info(f"Early stopping at episode {episode}")
                    break
                
                # Target reached
                if self.config.target_reward and episode_reward >= self.config.target_reward:
                    logger.info(f"Target reward {self.config.target_reward} reached!")
                    break
                
                # Run callbacks
                for callback in self.callbacks:
                    callback(self, episode)
            
            # Final save
            await self._save_checkpoint(self.config.episodes)
            
            # Training summary
            training_time = time.time() - self.training_start_time
            return self._get_training_summary(training_time)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self._get_training_summary(time.time() - self.training_start_time)
        
        finally:
            if self.writer:
                self.writer.close()
    
    async def _run_episode(self, episode_num: int) -> Tuple[float, int]:
        """Run a single training episode"""
        obs, info = self.environment.reset()
        self.agent.reset()
        
        total_reward = 0
        steps = 0
        
        for step in range(self.config.max_steps_per_episode):
            # Agent selects action
            action, action_info = await self.agent.step(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, env_info = self.environment.step(action)
            
            # Create experience
            experience = Experience(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=terminated or truncated,
                info={**action_info, **env_info}
            )
            
            # Add to agent's buffer
            self.agent.add_experience(experience)
            
            # Train agent
            if len(self.agent.experience_buffer) >= self.agent.config.get("batch_size", 32):
                train_info = await self.agent.train()
                
                # Log training metrics
                if self.writer and "loss" in train_info:
                    global_step = episode_num * self.config.max_steps_per_episode + step
                    self.writer.add_scalar("Loss/train", train_info["loss"], global_step)
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            # Render if requested
            if self.config.render:
                self.environment.render()
            
            if terminated or truncated:
                break
        
        # Log episode metrics
        if self.writer:
            self.writer.add_scalar("Reward/episode", total_reward, episode_num)
            self.writer.add_scalar("Length/episode", steps, episode_num)
        
        return total_reward, steps
    
    async def _evaluate(self, episode: int) -> Dict[str, float]:
        """Evaluate agent performance"""
        logger.info(f"Evaluating at episode {episode}")
        
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(5):  # Run 5 evaluation episodes
            obs, _ = self.environment.reset()
            total_reward = 0
            steps = 0
            
            for _ in range(self.config.max_steps_per_episode):
                action, _ = await self.agent.step(obs)
                obs, reward, terminated, truncated, _ = self.environment.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        
        logger.info(f"Evaluation - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        if self.writer:
            self.writer.add_scalar("Evaluation/reward", avg_reward, episode)
            self.writer.add_scalar("Evaluation/length", avg_length, episode)
        
        return {"avg_reward": avg_reward, "avg_length": avg_length}
    
    async def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pth"
        self.agent.save(str(checkpoint_path))
        
        # Save training state
        state = {
            "episode": episode,
            "episode_rewards": self.episode_rewards[-100:],  # Last 100
            "episode_lengths": self.episode_lengths[-100:],
            "best_reward": self.best_reward,
            "config": self.config.to_dict()
        }
        
        state_path = self.checkpoint_dir / f"training_state_ep{episode}.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved at episode {episode}")
    
    def _log_progress(self, episode: int) -> None:
        """Log training progress"""
        if len(self.episode_rewards) == 0:
            return
        
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        logger.info(
            f"Episode {episode} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Length: {avg_length:.2f} | "
            f"Best: {self.best_reward:.2f}"
        )
    
    def _should_stop_early(self, episode_reward: float) -> bool:
        """Check if training should stop early"""
        if not self.config.early_stopping:
            return False
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _get_training_summary(self, training_time: float) -> Dict[str, Any]:
        """Get training summary statistics"""
        return {
            "total_episodes": len(self.episode_rewards),
            "total_steps": sum(self.episode_lengths),
            "training_time": training_time,
            "best_reward": self.best_reward,
            "final_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback function"""
        self.callbacks.append(callback)
    
    def plot_training_curve(self) -> None:
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Rewards
            ax1.plot(self.episode_rewards)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Training Rewards")
            ax1.grid(True)
            
            # Episode lengths
            ax2.plot(self.episode_lengths)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Steps")
            ax2.set_title("Episode Lengths")
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / "training_curves.png")
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")