"""
Agent implementations for the Nexus framework.
"""

from nexus.agents.base import BaseAgent, AgentStatus, AgentType
from nexus.agents.scripted_agent import ScriptedAgent
from nexus.agents.rl_agent import DQNAgent, PPOAgent
# RainbowAgent is defined but not exported separately
RainbowAgent = DQNAgent  # Use DQN as fallback for Rainbow
from nexus.agents.agent_manager import AgentManager
from typing import Dict, Any, Optional
import torch

__all__ = [
    'BaseAgent',
    'AgentStatus',
    'AgentType',
    'ScriptedAgent',
    'DQNAgent',
    'PPOAgent',
    'RainbowAgent',
    'AgentManager',
    'create_agent',
    'load_agent'
]


def create_agent(agent_type: str, 
                observation_space: Any,
                action_space: Any,
                config: Optional[Dict[str, Any]] = None) -> BaseAgent:
    """Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent ('dqn', 'ppo', 'rainbow', 'random', 'scripted')
        observation_space: Observation space for the agent
        action_space: Action space for the agent
        config: Agent configuration
        
    Returns:
        Created agent instance
    """
    config = config or {}
    
    # Map agent type strings to classes
    agent_map = {
        'dqn': DQNAgent,
        'ppo': PPOAgent,
        'rainbow': RainbowAgent,
        'scripted': ScriptedAgent,
        'random': ScriptedAgent  # Random uses scripted with random policy
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = agent_map[agent_type]
    
    # Set agent type
    if agent_type in ['dqn', 'ppo', 'rainbow']:
        agent_type_enum = AgentType.RL
    else:
        agent_type_enum = AgentType.SCRIPTED
    
    # Create name if not provided
    name = config.pop('name', f"{agent_type}_agent")
    
    # For random agent, set random policy
    if agent_type == 'random':
        config['policy'] = 'random'
    
    # Create agent
    agent = agent_class(
        name=name,
        agent_type=agent_type_enum,
        observation_space=observation_space,
        action_space=action_space,
        config=config
    )
    
    return agent


def load_agent(path: str) -> BaseAgent:
    """Load an agent from a saved file.
    
    Args:
        path: Path to the saved agent file
        
    Returns:
        Loaded agent instance
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    
    # Determine agent type from checkpoint
    agent_type = checkpoint.get('agent_type', 'dqn')
    observation_space = checkpoint.get('observation_space')
    action_space = checkpoint.get('action_space')
    config = checkpoint.get('config', {})
    
    # Create agent
    agent = create_agent(agent_type, observation_space, action_space, config)
    
    # Load state
    agent.load(path)
    
    return agent