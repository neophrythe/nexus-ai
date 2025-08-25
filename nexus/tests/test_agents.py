"""Tests for agent implementations"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import torch
import gymnasium as gym

from nexus.agents.base import BaseAgent, AgentType, Experience
from nexus.agents.scripted_agent import ScriptedAgent, Rule, GameScriptedAgent
from nexus.agents.ppo_agent import PPOAgent
from nexus.agents.rainbow_dqn_agent import RainbowDQNAgent


class TestBaseAgent:
    """Test base agent functionality"""
    
    def test_agent_creation(self, config_manager):
        """Test creating a base agent"""
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            async def act(self, observation):
                return 0
            
            async def learn(self, experience):
                return {"loss": 0.0}
        
        agent = TestAgent("test_agent", AgentType.SCRIPTED, config=config_manager)
        
        assert agent.name == "test_agent"
        assert agent.agent_type == AgentType.SCRIPTED
        assert agent.config == config_manager
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, config_manager):
        """Test agent initialization"""
        class TestAgent(BaseAgent):
            async def act(self, observation):
                return 0
            
            async def learn(self, experience):
                return {"loss": 0.0}
            
            async def initialize(self):
                await super().initialize()
                self.custom_init = True
        
        agent = TestAgent("test_agent", AgentType.SCRIPTED, config=config_manager)
        await agent.initialize()
        
        assert agent.initialized
        assert hasattr(agent, 'custom_init')
    
    def test_experience_creation(self):
        """Test creating experience objects"""
        observation = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_observation = np.array([2, 3, 4])
        done = False
        
        experience = Experience(observation, action, reward, next_observation, done)
        
        assert np.array_equal(experience.observation, observation)
        assert experience.action == action
        assert experience.reward == reward
        assert np.array_equal(experience.next_observation, next_observation)
        assert experience.done == done


class TestScriptedAgent:
    """Test scripted agent implementation"""
    
    @pytest.mark.asyncio
    async def test_scripted_agent_creation(self, config_manager):
        """Test creating a scripted agent"""
        agent = ScriptedAgent("scripted_test", config=config_manager)
        await agent.initialize()
        
        assert agent.name == "scripted_test"
        assert agent.agent_type == AgentType.SCRIPTED
        assert len(agent.rules) == 0  # No default rules in base class
    
    @pytest.mark.asyncio
    async def test_rule_management(self, config_manager):
        """Test adding and managing rules"""
        agent = ScriptedAgent("rule_test", config=config_manager)
        await agent.initialize()
        
        # Create test rules
        rule1 = Rule("high_priority", lambda obs: True, lambda obs: "action1", priority=10)
        rule2 = Rule("low_priority", lambda obs: True, lambda obs: "action2", priority=1)
        
        agent.add_rule(rule1)
        agent.add_rule(rule2)
        
        # Rules should be sorted by priority
        assert agent.rules[0].name == "high_priority"
        assert agent.rules[1].name == "low_priority"
        
        # Test rule removal
        agent.remove_rule("high_priority")
        assert len(agent.rules) == 1
        assert agent.rules[0].name == "low_priority"
    
    @pytest.mark.asyncio
    async def test_rule_evaluation(self, config_manager):
        """Test rule evaluation and action selection"""
        agent = ScriptedAgent("eval_test", config=config_manager)
        await agent.initialize()
        
        # Add rules with different conditions
        agent.add_rule(Rule("never_match", lambda obs: False, lambda obs: "never", priority=10))
        agent.add_rule(Rule("always_match", lambda obs: True, lambda obs: "always", priority=5))
        
        # Should return action from first matching rule
        observation = {"test": "data"}
        action = await agent.act(observation)
        assert action == "always"
    
    @pytest.mark.asyncio
    async def test_default_action(self, config_manager):
        """Test default action when no rules match"""
        agent = ScriptedAgent("default_test", config=config_manager)
        await agent.initialize()
        
        # Set default action
        agent.set_default_action("default_action")
        
        # Add rule that won't match
        agent.add_rule(Rule("no_match", lambda obs: False, lambda obs: "no_match"))
        
        observation = {"test": "data"}
        action = await agent.act(observation)
        assert action == "default_action"
    
    @pytest.mark.asyncio
    async def test_state_machine(self, config_manager):
        """Test state machine functionality"""
        agent = ScriptedAgent("state_test", config=config_manager)
        await agent.initialize()
        
        # Create rules for different states
        idle_rules = [Rule("idle_rule", lambda obs: True, lambda obs: "idle_action")]
        combat_rules = [Rule("combat_rule", lambda obs: True, lambda obs: "combat_action")]
        
        agent.add_state("idle", idle_rules)
        agent.add_state("combat", combat_rules)
        
        # Test state transitions
        agent.transition_to_state("combat")
        assert agent.current_state == "combat"
        assert len(agent.rules) == 1
        assert agent.rules[0].name == "combat_rule"
        
        observation = {"test": "data"}
        action = await agent.act(observation)
        assert action == "combat_action"


class TestGameScriptedAgent:
    """Test game-specific scripted agent"""
    
    @pytest.mark.asyncio
    async def test_game_agent_rules(self, config_manager):
        """Test game-specific rule setup"""
        agent = GameScriptedAgent("game_test", config=config_manager)
        await agent.initialize()
        
        # Should have default game rules
        assert len(agent.rules) > 0
        
        # Check for expected rule names
        rule_names = [rule.name for rule in agent.rules]
        assert "low_health" in rule_names
        assert "attack_enemy" in rule_names
        assert "collect_items" in rule_names
        assert "explore" in rule_names
    
    @pytest.mark.asyncio
    async def test_health_detection(self, config_manager):
        """Test health-based rule triggering"""
        agent = GameScriptedAgent("health_test", config=config_manager)
        await agent.initialize()
        
        # Test low health condition
        low_health_obs = {"health": 20, "enemies": [], "items": []}
        action = await agent.act(low_health_obs)
        # Should trigger low health rule (highest priority)
        # Exact action depends on implementation, but shouldn't be explore
        
        # Test normal health
        normal_health_obs = {"health": 80, "enemies": [], "items": []}
        action = await agent.act(normal_health_obs)
        # Should fall through to explore (lowest priority)


@pytest.mark.asyncio
class TestPPOAgent:
    """Test PPO agent implementation"""
    
    async def test_ppo_agent_creation(self, config_manager):
        """Test creating a PPO agent"""
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        
        agent = PPOAgent(
            "ppo_test",
            observation_space=observation_space,
            action_space=action_space,
            config=config_manager
        )
        await agent.initialize()
        
        assert agent.name == "ppo_test"
        assert agent.agent_type == AgentType.RL
        assert agent.observation_space == observation_space
        assert agent.action_space == action_space
    
    async def test_ppo_action_selection(self, config_manager):
        """Test PPO action selection"""
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        
        agent = PPOAgent(
            "ppo_action_test",
            observation_space=observation_space,
            action_space=action_space,
            config=config_manager
        )
        await agent.initialize()
        
        # Test action selection
        observation = np.array([0.1, 0.2, 0.3, 0.4])
        action = await agent.act(observation)
        
        # Should return valid action
        assert action in range(action_space.n)
    
    async def test_ppo_learning(self, config_manager):
        """Test PPO learning from experience"""
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        
        agent = PPOAgent(
            "ppo_learn_test",
            observation_space=observation_space,
            action_space=action_space,
            config=config_manager
        )
        await agent.initialize()
        
        # Create sample experiences
        experiences = []
        for _ in range(32):  # Batch size
            obs = np.random.random(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = np.random.random()
            next_obs = np.random.random(4).astype(np.float32)
            done = np.random.choice([True, False])
            
            experiences.append(Experience(obs, action, reward, next_obs, done))
        
        # Test learning
        result = await agent.learn(experiences)
        
        assert "policy_loss" in result
        assert "value_loss" in result
        assert isinstance(result["policy_loss"], float)


@pytest.mark.asyncio
class TestRainbowDQNAgent:
    """Test Rainbow DQN agent implementation"""
    
    async def test_rainbow_agent_creation(self, config_manager):
        """Test creating a Rainbow DQN agent"""
        state_size = 4
        action_size = 2
        
        agent = RainbowDQNAgent(
            "rainbow_test",
            state_size=state_size,
            action_size=action_size,
            config=config_manager
        )
        await agent.initialize()
        
        assert agent.name == "rainbow_test"
        assert agent.agent_type == AgentType.RL
        assert agent.state_size == state_size
        assert agent.action_size == action_size
    
    async def test_rainbow_action_selection(self, config_manager):
        """Test Rainbow DQN action selection"""
        state_size = 4
        action_size = 2
        
        agent = RainbowDQNAgent(
            "rainbow_action_test",
            state_size=state_size,
            action_size=action_size,
            config=config_manager
        )
        await agent.initialize()
        
        # Test action selection
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = await agent.act(state)
        
        # Should return valid action
        assert action in range(action_size)
    
    async def test_rainbow_memory_replay(self, config_manager):
        """Test Rainbow DQN experience replay"""
        state_size = 4
        action_size = 2
        
        agent = RainbowDQNAgent(
            "rainbow_memory_test",
            state_size=state_size,
            action_size=action_size,
            config=config_manager
        )
        await agent.initialize()
        
        # Add experiences to memory
        for _ in range(100):
            state = np.random.random(state_size).astype(np.float32)
            action = np.random.randint(0, action_size)
            reward = np.random.random()
            next_state = np.random.random(state_size).astype(np.float32)
            done = np.random.choice([True, False])
            
            experience = Experience(state, action, reward, next_state, done)
            agent.memory.push(experience)
        
        # Test learning with replay
        result = await agent.learn(None)  # Rainbow uses internal memory
        
        assert "loss" in result
        assert isinstance(result["loss"], float)


class TestAgentIntegration:
    """Test agent integration with other components"""
    
    @pytest.mark.asyncio
    async def test_agent_with_environment(self, config_manager, mock_frame):
        """Test agent interaction with game environment"""
        # Mock environment
        class MockEnvironment:
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3))
                self.action_space = gym.spaces.Discrete(4)
            
            async def reset(self):
                return np.random.randint(0, 255, (480, 640, 3))
            
            async def step(self, action):
                observation = np.random.randint(0, 255, (480, 640, 3))
                reward = 1.0
                done = False
                info = {}
                return observation, reward, done, info
        
        environment = MockEnvironment()
        
        # Create agent
        agent = ScriptedAgent("env_test", config=config_manager)
        await agent.initialize()
        
        # Add simple rule
        agent.add_rule(Rule("test_rule", lambda obs: True, lambda obs: 1))
        
        # Test interaction
        observation = await environment.reset()
        action = await agent.act(observation)
        next_observation, reward, done, info = await environment.step(action)
        
        assert action in range(environment.action_space.n)
    
    @pytest.mark.asyncio
    async def test_agent_save_load(self, config_manager, temp_dir):
        """Test agent save and load functionality"""
        agent = ScriptedAgent("save_test", config=config_manager)
        await agent.initialize()
        
        # Add some rules
        agent.add_rule(Rule("test_rule", lambda obs: True, lambda obs: "test_action"))
        agent.current_state = "test_state"
        
        # Save agent
        save_path = temp_dir / "agent_save.json"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = ScriptedAgent("load_test", config=config_manager)
        await new_agent.initialize()
        new_agent.add_rule(Rule("test_rule", lambda obs: True, lambda obs: "test_action"))  # Need to recreate rules
        new_agent.load(str(save_path))
        
        # Verify loaded state
        assert new_agent.current_state == "test_state"


@pytest.mark.asyncio 
class TestAgentPerformance:
    """Test agent performance and optimization"""
    
    async def test_agent_action_timing(self, config_manager):
        """Test agent action selection timing"""
        import time
        
        agent = ScriptedAgent("timing_test", config=config_manager)
        await agent.initialize()
        
        # Add multiple rules to test performance
        for i in range(100):
            agent.add_rule(Rule(
                f"rule_{i}", 
                lambda obs, i=i: obs.get("value", 0) == i,
                lambda obs, i=i: f"action_{i}",
                priority=i
            ))
        
        observation = {"value": 50}  # Will match rule_50
        
        start_time = time.time()
        action = await agent.act(observation)
        end_time = time.time()
        
        # Action selection should be fast
        assert (end_time - start_time) < 0.01  # Less than 10ms
        assert action == "action_50"
    
    async def test_memory_usage(self, config_manager):
        """Test agent memory usage doesn't grow unbounded"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        agent = ScriptedAgent("memory_test", config=config_manager)
        await agent.initialize()
        
        # Add many rules and run many actions
        for i in range(1000):
            agent.add_rule(Rule(f"rule_{i}", lambda obs: i % 100 == 0, lambda obs: i))
        
        # Execute many actions
        for _ in range(1000):
            await agent.act({"test": "observation"})
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024