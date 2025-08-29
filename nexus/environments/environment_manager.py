import asyncio
from typing import Dict, List, Optional, Any, Type, Tuple
from pathlib import Path
import structlog

from nexus.environments.base import GameEnvironment, GamePhase, GameState

logger = structlog.get_logger()


class EnvironmentManager:
    """Manages game environments"""
    
    def __init__(self):
        self.environments: Dict[str, GameEnvironment] = {}
        self.active_environment: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def create_environment(self,
                                name: str,
                                environment_class: Type[GameEnvironment],
                                capture_manager: Any = None,
                                input_controller: Any = None,
                                config: Optional[Dict[str, Any]] = None) -> GameEnvironment:
        """Create and register a new environment"""
        
        async with self._lock:
            if name in self.environments:
                logger.warning(f"Environment {name} already exists")
                return self.environments[name]
            
            # Create environment
            env = environment_class(
                game_name=name,
                capture_manager=capture_manager,
                input_controller=input_controller,
                config=config
            )
            
            self.environments[name] = env
            logger.info(f"Created environment: {name}")
            
            return env
    
    def get_environment(self, name: str) -> Optional[GameEnvironment]:
        """Get an environment by name"""
        return self.environments.get(name)
    
    def set_active_environment(self, name: str) -> None:
        """Set the active environment"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        self.active_environment = name
        logger.info(f"Active environment set to: {name}")
    
    def get_active_environment(self) -> Optional[GameEnvironment]:
        """Get the active environment"""
        if self.active_environment:
            return self.environments.get(self.active_environment)
        return None
    
    def list_environments(self) -> List[Dict[str, Any]]:
        """List all environments"""
        return [
            {
                "name": name,
                "game": env.game_name,
                "phase": env.current_phase.value,
                "frame_count": env.frame_count,
                "episode_count": env.episode_count
            }
            for name, env in self.environments.items()
        ]
    
    async def remove_environment(self, name: str) -> None:
        """Remove an environment"""
        async with self._lock:
            if name in self.environments:
                env = self.environments[name]
                env.close()
                del self.environments[name]
                
                if self.active_environment == name:
                    self.active_environment = None
                
                logger.info(f"Removed environment: {name}")
    
    def reset_environment(self, name: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset an environment"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        env = self.environments[name]
        return env.reset(**kwargs)
    
    def step_environment(self, name: str, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step an environment"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        env = self.environments[name]
        return env.step(action)
    
    def save_environment_state(self, name: str, path: str) -> None:
        """Save environment state"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        env = self.environments[name]
        env.save_state(path)
        
        logger.info(f"Saved environment {name} state to {path}")
    
    def load_environment_state(self, name: str, path: str) -> None:
        """Load environment state"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        env = self.environments[name]
        env.load_state(path)
        
        logger.info(f"Loaded environment {name} state from {path}")
    
    def get_environment_info(self, name: str) -> Dict[str, Any]:
        """Get detailed environment information"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        env = self.environments[name]
        
        return {
            "name": name,
            "game": env.game_name,
            "phase": env.current_phase.value,
            "frame_count": env.frame_count,
            "episode_count": env.episode_count,
            "total_reward": env.total_reward,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "render_modes": env.metadata.get("render_modes", [])
        }
    
    def get_environment_history(self, name: str, n: Optional[int] = None) -> List[GameState]:
        """Get environment state history"""
        if name not in self.environments:
            raise ValueError(f"Environment {name} not found")
        
        env = self.environments[name]
        return env.get_state_history(n)
    
    async def run_episode(self,
                         environment_name: str,
                         agent: Any,
                         max_steps: int = 1000,
                         render: bool = False) -> Dict[str, Any]:
        """Run a single episode"""
        
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        env = self.environments[environment_name]
        obs, info = env.reset()
        agent.reset()
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Agent selects action
            action, _ = await agent.step(obs)
            
            # Environment step
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            if terminated or truncated:
                break
        
        return {
            "reward": total_reward,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    
    async def run_parallel_episodes(self,
                                  environment_names: List[str],
                                  agents: List[Any],
                                  max_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run multiple episodes in parallel"""
        
        tasks = []
        for env_name, agent in zip(environment_names, agents):
            task = self.run_episode(env_name, agent, max_steps)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def cleanup(self) -> None:
        """Cleanup all environments"""
        for name in list(self.environments.keys()):
            await self.remove_environment(name)
        
        logger.info("Environment manager cleaned up")