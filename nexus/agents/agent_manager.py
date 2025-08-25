import asyncio
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import structlog

from nexus.agents.base import BaseAgent, AgentType
from nexus.agents.scripted_agent import ScriptedAgent
from nexus.agents.rl_agent import DQNAgent, PPOAgent

logger = structlog.get_logger()


class AgentManager:
    """Manages agent lifecycle and coordination"""
    
    AGENT_CLASSES = {
        AgentType.SCRIPTED: ScriptedAgent,
        AgentType.RL: {
            "dqn": DQNAgent,
            "ppo": PPOAgent
        }
    }
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.active_agent: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def create_agent(self,
                          name: str,
                          agent_type: AgentType,
                          observation_space: Any = None,
                          action_space: Any = None,
                          config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Create and register a new agent"""
        
        async with self._lock:
            if name in self.agents:
                logger.warning(f"Agent {name} already exists")
                return self.agents[name]
            
            # Get agent class
            if agent_type == AgentType.SCRIPTED:
                agent_class = ScriptedAgent
            elif agent_type == AgentType.RL:
                # Default to DQN for RL
                rl_type = config.get("rl_algorithm", "dqn") if config else "dqn"
                agent_class = self.AGENT_CLASSES[AgentType.RL].get(rl_type, DQNAgent)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Create agent
            agent = agent_class(
                name=name,
                agent_type=agent_type,
                observation_space=observation_space,
                action_space=action_space,
                config=config
            )
            
            await agent.initialize()
            
            self.agents[name] = agent
            logger.info(f"Created agent: {name} ({agent_type.value})")
            
            return agent
    
    async def load_agent(self, name: str, path: str) -> BaseAgent:
        """Load an agent from file"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        
        agent = self.agents[name]
        agent.load(path)
        
        logger.info(f"Loaded agent {name} from {path}")
        return agent
    
    async def save_agent(self, name: str, path: str) -> None:
        """Save an agent to file"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        
        agent = self.agents[name]
        agent.save(path)
        
        logger.info(f"Saved agent {name} to {path}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def set_active_agent(self, name: str) -> None:
        """Set the active agent"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        
        self.active_agent = name
        logger.info(f"Active agent set to: {name}")
    
    def get_active_agent(self) -> Optional[BaseAgent]:
        """Get the active agent"""
        if self.active_agent:
            return self.agents.get(self.active_agent)
        return None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents"""
        return [
            {
                "name": name,
                "type": agent.agent_type.value,
                "status": agent.status.value,
                "metrics": agent.get_metrics()
            }
            for name, agent in self.agents.items()
        ]
    
    async def remove_agent(self, name: str) -> None:
        """Remove an agent"""
        async with self._lock:
            if name in self.agents:
                del self.agents[name]
                if self.active_agent == name:
                    self.active_agent = None
                logger.info(f"Removed agent: {name}")
    
    async def train_agent(self,
                         name: str,
                         environment: Any,
                         episodes: int = 1000,
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train an agent"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        
        agent = self.agents[name]
        
        from nexus.training import Trainer, TrainingConfig
        
        config = TrainingConfig(episodes=episodes)
        trainer = Trainer(agent, environment, config)
        
        results = await trainer.train()
        
        if save_path:
            agent.save(save_path)
        
        return results
    
    async def evaluate_agent(self,
                            name: str,
                            environment: Any,
                            episodes: int = 10) -> Dict[str, Any]:
        """Evaluate an agent"""
        if name not in self.agents:
            raise ValueError(f"Agent {name} not found")
        
        agent = self.agents[name]
        
        total_reward = 0
        episode_rewards = []
        
        for episode in range(episodes):
            obs, _ = environment.reset()
            agent.reset()
            
            episode_reward = 0
            
            while True:
                action, _ = await agent.step(obs)
                obs, reward, terminated, truncated, _ = environment.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
        
        return {
            "episodes": episodes,
            "total_reward": total_reward,
            "avg_reward": total_reward / episodes,
            "episode_rewards": episode_rewards
        }
    
    async def cleanup(self) -> None:
        """Cleanup all agents"""
        for name in list(self.agents.keys()):
            await self.remove_agent(name)
        
        logger.info("Agent manager cleaned up")