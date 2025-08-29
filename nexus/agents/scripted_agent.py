import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import structlog

from nexus.agents.base import BaseAgent, AgentType, Experience

logger = structlog.get_logger()


@dataclass
class Rule:
    """Rule for scripted behavior"""
    name: str
    condition: Callable[[Any], bool]
    action: Callable[[Any], Any]
    priority: int = 0
    enabled: bool = True
    
    def evaluate(self, observation: Any) -> Optional[Any]:
        """Evaluate rule and return action if condition is met"""
        if self.enabled and self.condition(observation):
            return self.action(observation)
        return None


class ScriptedAgent(BaseAgent):
    """Rule-based scripted agent"""
    
    def __init__(self, *args, **kwargs):
        # Remove agent_type from kwargs if it exists to avoid duplicate
        kwargs.pop('agent_type', None)
        super().__init__(*args, agent_type=AgentType.SCRIPTED, **kwargs)
        self.rules: List[Rule] = []
        self.default_action = None
        self.last_observation = None
        self.state_machine = {}
        self.current_state = "default"
    
    async def initialize(self) -> None:
        await super().initialize()
        self._setup_default_rules()
        logger.info(f"Scripted Agent initialized with {len(self.rules)} rules")
    
    def _setup_default_rules(self):
        """Setup default rules - override in subclasses"""
        # Add common game rules
        
        # Rule: Avoid danger
        self.add_rule(Rule(
            name="avoid_danger",
            condition=lambda obs: self._is_danger_nearby(obs),
            action=lambda obs: self._escape_danger(obs),
            priority=10
        ))
        
        # Rule: Collect items
        self.add_rule(Rule(
            name="collect_items",
            condition=lambda obs: self._are_items_nearby(obs),
            action=lambda obs: self._move_to_nearest_item(obs),
            priority=5
        ))
        
        # Rule: Attack enemies
        self.add_rule(Rule(
            name="attack_enemies",
            condition=lambda obs: self._are_enemies_nearby(obs),
            action=lambda obs: self._attack_nearest_enemy(obs),
            priority=7
        ))
        
        # Rule: Explore
        self.add_rule(Rule(
            name="explore",
            condition=lambda obs: True,  # Always true as fallback
            action=lambda obs: self._explore(obs),
            priority=1
        ))
    
    def _is_danger_nearby(self, obs):
        """Check if danger is nearby"""
        if isinstance(obs, dict) and 'dangers' in obs:
            return len(obs['dangers']) > 0
        return False
    
    def _escape_danger(self, obs):
        """Escape from danger"""
        # Move away from danger
        return "MOVE_AWAY"
    
    def _are_items_nearby(self, obs):
        """Check if items are nearby"""
        if isinstance(obs, dict) and 'items' in obs:
            return len(obs['items']) > 0
        return False
    
    def _move_to_nearest_item(self, obs):
        """Move to nearest item"""
        return "MOVE_TO_ITEM"
    
    def _are_enemies_nearby(self, obs):
        """Check if enemies are nearby"""
        if isinstance(obs, dict) and 'enemies' in obs:
            return len(obs['enemies']) > 0
        return False
    
    def _attack_nearest_enemy(self, obs):
        """Attack nearest enemy"""
        return "ATTACK"
    
    def _explore(self, obs):
        """Explore the environment"""
        import random
        return random.choice(["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"])
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the agent"""
        self.rules.append(rule)
        # Sort by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, name: str) -> None:
        """Remove a rule by name"""
        self.rules = [r for r in self.rules if r.name != name]
    
    def enable_rule(self, name: str) -> None:
        """Enable a rule by name"""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                break
    
    def disable_rule(self, name: str) -> None:
        """Disable a rule by name"""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                break
    
    def predict(self, observation: Any) -> Any:
        """Synchronous predict method for compatibility."""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, can't use run_until_complete
            import random
            return random.randint(0, self.action_space - 1) if hasattr(self, 'action_space') else 0
        else:
            return loop.run_until_complete(self.act(observation))
    
    async def act(self, observation: Any) -> Any:
        """Select action based on rules"""
        self.last_observation = observation
        
        # Evaluate rules in priority order
        for rule in self.rules:
            action = rule.evaluate(observation)
            if action is not None:
                logger.debug(f"Rule '{rule.name}' triggered, action: {action}")
                return action
        
        # No rule matched, use default action
        if self.default_action is not None:
            if callable(self.default_action):
                return self.default_action(observation)
            return self.default_action
        
        # No default action, return random
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        
        return 0
    
    async def learn(self, experience: Experience) -> Dict[str, Any]:
        """Scripted agents don't learn, but can adapt rules"""
        # Could implement rule adaptation based on success/failure
        return {"status": "no_learning"}
    
    def set_default_action(self, action: Any) -> None:
        """Set default action when no rules match"""
        self.default_action = action
    
    def add_state(self, state_name: str, rules: List[Rule]) -> None:
        """Add a state with associated rules for state machine"""
        self.state_machine[state_name] = rules
    
    def transition_to_state(self, state_name: str) -> None:
        """Transition to a different state"""
        if state_name in self.state_machine:
            self.current_state = state_name
            self.rules = self.state_machine[state_name].copy()
            self.rules.sort(key=lambda r: r.priority, reverse=True)
            logger.info(f"Transitioned to state: {state_name}")
    
    def save(self, path: str) -> None:
        """Save agent configuration"""
        import json
        
        config = {
            "current_state": self.current_state,
            "rules": [
                {
                    "name": r.name,
                    "priority": r.priority,
                    "enabled": r.enabled
                }
                for r in self.rules
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Scripted Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load agent configuration"""
        import json
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.current_state = config.get("current_state", "default")
        
        # Update rule states
        for rule_config in config.get("rules", []):
            for rule in self.rules:
                if rule.name == rule_config["name"]:
                    rule.priority = rule_config["priority"]
                    rule.enabled = rule_config["enabled"]
        
        logger.info(f"Scripted Agent loaded from {path}")


class GameScriptedAgent(ScriptedAgent):
    """Example scripted agent for games"""
    
    def _setup_default_rules(self):
        """Setup game-specific rules"""
        
        # Rule: Low health - find health pack
        self.add_rule(Rule(
            name="low_health",
            condition=lambda obs: self._get_health(obs) < 30,
            action=lambda obs: self._find_health_action(obs),
            priority=100
        ))
        
        # Rule: Enemy nearby - attack
        self.add_rule(Rule(
            name="attack_enemy",
            condition=lambda obs: self._enemy_nearby(obs),
            action=lambda obs: self._attack_action(obs),
            priority=50
        ))
        
        # Rule: Collect items
        self.add_rule(Rule(
            name="collect_items",
            condition=lambda obs: self._item_nearby(obs),
            action=lambda obs: self._collect_action(obs),
            priority=30
        ))
        
        # Rule: Explore
        self.add_rule(Rule(
            name="explore",
            condition=lambda obs: True,  # Always true - lowest priority
            action=lambda obs: self._explore_action(obs),
            priority=0
        ))
    
    def _get_health(self, observation: Any) -> float:
        """Extract health from observation"""
        if isinstance(observation, dict):
            return observation.get("health", 100)
        return 100
    
    def _enemy_nearby(self, observation: Any) -> bool:
        """Check if enemy is nearby"""
        if isinstance(observation, dict):
            enemies = observation.get("enemies", [])
            return len(enemies) > 0
        return False
    
    def _item_nearby(self, observation: Any) -> bool:
        """Check if item is nearby"""
        if isinstance(observation, dict):
            items = observation.get("items", [])
            return len(items) > 0
        return False
    
    def _find_health_action(self, observation: Any) -> Any:
        """Action to find health pack"""
        # Implement pathfinding to health pack
        return 0  # Placeholder
    
    def _attack_action(self, observation: Any) -> Any:
        """Action to attack enemy"""
        # Implement attack logic
        return 1  # Placeholder
    
    def _collect_action(self, observation: Any) -> Any:
        """Action to collect item"""
        # Implement collection logic
        return 2  # Placeholder
    
    def _explore_action(self, observation: Any) -> Any:
        """Action to explore map"""
        # Implement exploration logic
        import random
        return random.randint(0, 3)  # Random movement