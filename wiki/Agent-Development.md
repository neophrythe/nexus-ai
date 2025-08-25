# Agent Development

Learn how to create intelligent game-playing agents using various AI techniques, from simple scripted behaviors to advanced deep reinforcement learning.

## Table of Contents
- [Agent Types](#agent-types)
- [Creating Your First Agent](#creating-your-first-agent)
- [Reinforcement Learning Agents](#reinforcement-learning-agents)
- [State Representation](#state-representation)
- [Reward Engineering](#reward-engineering)
- [Training Strategies](#training-strategies)
- [Advanced Techniques](#advanced-techniques)
- [Examples](#examples)

## Agent Types

### Scripted Agents
Simple rule-based agents that follow predefined logic:
```python
from nexus.agents import ScriptedAgent

class SimpleBot(ScriptedAgent):
    def act(self, state):
        if state["enemy_nearby"]:
            return "ATTACK"
        elif state["health"] < 50:
            return "HEAL"
        else:
            return "EXPLORE"
```

### Reinforcement Learning Agents
Agents that learn through trial and error:
- **DQN (Deep Q-Network)**: Discrete action spaces
- **PPO (Proximal Policy Optimization)**: Continuous/discrete actions
- **Rainbow DQN**: State-of-the-art DQN variant
- **A2C/A3C**: Actor-Critic methods

### Imitation Learning Agents
Agents that learn from human demonstrations:
```python
from nexus.agents import ImitationAgent

agent = ImitationAgent()
agent.learn_from_demonstrations("gameplay_recordings/")
```

### Hybrid Agents
Combine multiple approaches:
```python
class HybridAgent(Agent):
    def __init__(self):
        self.rl_agent = DQNAgent()
        self.scripted_agent = ScriptedAgent()
    
    def act(self, state):
        if self.is_critical_situation(state):
            return self.scripted_agent.act(state)
        return self.rl_agent.act(state)
```

## Creating Your First Agent

### Basic Structure

```python
from nexus.agents import BaseAgent
import numpy as np

class MyFirstAgent(BaseAgent):
    def __init__(self, game):
        super().__init__(game)
        self.action_space = game.get_action_space()
        self.observation_space = game.get_observation_space()
        
    def setup(self):
        """Initialize agent components"""
        self.model = self.build_model()
        self.memory = []
        
    def observe(self, state):
        """Process game state"""
        # Convert raw frame to features
        features = self.extract_features(state)
        return features
    
    def act(self, observation):
        """Choose action based on observation"""
        # Simple random action
        return np.random.choice(self.action_space)
    
    def learn(self, experience):
        """Learn from experience"""
        self.memory.append(experience)
        if len(self.memory) > 1000:
            self.train()
    
    def extract_features(self, state):
        """Extract relevant features from state"""
        features = []
        features.append(state.get("player_health", 0) / 100)
        features.append(state.get("enemy_distance", 1.0))
        features.append(state.get("ammo", 0) / 50)
        return np.array(features)
```

## Reinforcement Learning Agents

### DQN Agent

```python
from nexus.agents import DQNAgent
import torch
import torch.nn as nn

class GameDQN(DQNAgent):
    def __init__(self, game):
        super().__init__(
            game=game,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            memory_size=10000,
            batch_size=32
        )
    
    def build_model(self):
        """Build neural network"""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space.n)
        )
    
    def preprocess_frame(self, frame):
        """Preprocess game frame"""
        # Resize to 84x84
        frame = cv2.resize(frame, (84, 84))
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Normalize
        frame = frame / 255.0
        return frame
```

### PPO Agent

```python
from nexus.agents import PPOAgent

class GamePPO(PPOAgent):
    def __init__(self, game):
        super().__init__(
            game=game,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_steps=2048,
            n_epochs=10
        )
    
    def get_value(self, state):
        """Estimate state value"""
        return self.critic(state)
    
    def get_action_distribution(self, state):
        """Get action probability distribution"""
        return self.actor(state)
```

### Rainbow DQN

```python
from nexus.agents import RainbowDQN

agent = RainbowDQN(
    game=game,
    # Double DQN
    double_dqn=True,
    # Dueling networks
    dueling=True,
    # Prioritized replay
    prioritized_replay=True,
    alpha=0.6,
    beta=0.4,
    # Noisy networks
    noisy=True,
    # Categorical DQN
    categorical=True,
    n_atoms=51,
    # N-step learning
    n_step=3
)
```

## State Representation

### Frame Stacking
```python
class FrameStackAgent(Agent):
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frame_buffer = deque(maxlen=n_frames)
    
    def get_state(self, frame):
        self.frame_buffer.append(frame)
        
        # Stack frames
        if len(self.frame_buffer) == self.n_frames:
            return np.stack(self.frame_buffer, axis=0)
        else:
            # Pad with zeros
            padding = self.n_frames - len(self.frame_buffer)
            frames = list(self.frame_buffer) + [np.zeros_like(frame)] * padding
            return np.stack(frames, axis=0)
```

### Feature Engineering
```python
def extract_features(self, game_state):
    """Extract hand-crafted features"""
    features = {
        # Spatial features
        "player_position": self.normalize_position(game_state["player_pos"]),
        "nearest_enemy": self.find_nearest_enemy(game_state["enemies"]),
        
        # Temporal features
        "velocity": self.calculate_velocity(),
        "acceleration": self.calculate_acceleration(),
        
        # Game-specific features
        "health_ratio": game_state["health"] / game_state["max_health"],
        "ammo_ratio": game_state["ammo"] / game_state["max_ammo"],
        "score_delta": game_state["score"] - self.last_score,
        
        # Relative features
        "enemy_angle": self.calculate_angle_to_enemy(),
        "distance_to_objective": self.distance_to_objective()
    }
    
    return self.features_to_vector(features)
```

### Attention Mechanisms
```python
class AttentionAgent(Agent):
    def __init__(self):
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
    
    def process_entities(self, entities):
        """Process multiple entities with attention"""
        # Convert entities to embeddings
        embeddings = [self.embed_entity(e) for e in entities]
        embeddings = torch.stack(embeddings)
        
        # Apply attention
        attended, weights = self.attention(embeddings, embeddings, embeddings)
        
        return attended, weights
```

## Reward Engineering

### Basic Rewards
```python
def calculate_reward(self, state, action, next_state):
    reward = 0
    
    # Survival reward
    reward += 0.1
    
    # Score-based reward
    score_delta = next_state["score"] - state["score"]
    reward += score_delta * 0.01
    
    # Health penalty
    health_delta = next_state["health"] - state["health"]
    if health_delta < 0:
        reward += health_delta * 0.1
    
    # Death penalty
    if next_state["is_dead"]:
        reward -= 10
    
    # Victory bonus
    if next_state["is_victory"]:
        reward += 100
    
    return reward
```

### Shaped Rewards
```python
class ShapedRewardCalculator:
    def __init__(self):
        self.potential_prev = 0
    
    def calculate_potential(self, state):
        """Calculate potential function"""
        potential = 0
        
        # Distance to goal
        potential -= state["distance_to_goal"] * 0.01
        
        # Resources collected
        potential += state["resources"] * 0.1
        
        # Enemies defeated
        potential += state["enemies_defeated"] * 1.0
        
        return potential
    
    def get_shaped_reward(self, state, base_reward):
        """Apply reward shaping"""
        potential = self.calculate_potential(state)
        shaped_reward = base_reward + (potential - self.potential_prev)
        self.potential_prev = potential
        return shaped_reward
```

### Curiosity-Driven Rewards
```python
class CuriosityReward:
    def __init__(self):
        self.state_predictor = self.build_predictor()
        self.visited_states = set()
    
    def get_intrinsic_reward(self, state, action, next_state):
        """Calculate curiosity bonus"""
        # Prediction error as curiosity signal
        predicted_next = self.state_predictor(state, action)
        prediction_error = np.mean((predicted_next - next_state) ** 2)
        
        # Novelty bonus
        state_hash = self.hash_state(state)
        if state_hash not in self.visited_states:
            novelty_bonus = 1.0
            self.visited_states.add(state_hash)
        else:
            novelty_bonus = 0.0
        
        return prediction_error * 0.1 + novelty_bonus * 0.5
```

## Training Strategies

### Curriculum Learning
```python
class CurriculumTrainer:
    def __init__(self, agent, game):
        self.agent = agent
        self.game = game
        self.difficulty_levels = [
            {"enemies": 1, "speed": 0.5},
            {"enemies": 3, "speed": 0.75},
            {"enemies": 5, "speed": 1.0},
            {"enemies": 10, "speed": 1.5}
        ]
        self.current_level = 0
    
    def train(self):
        for episode in range(1000):
            # Set difficulty
            self.game.set_difficulty(self.difficulty_levels[self.current_level])
            
            # Train episode
            reward = self.train_episode()
            
            # Progress to next level
            if self.agent.success_rate() > 0.8:
                self.current_level = min(
                    self.current_level + 1,
                    len(self.difficulty_levels) - 1
                )
```

### Self-Play
```python
class SelfPlayTrainer:
    def __init__(self, agent_class):
        self.main_agent = agent_class()
        self.opponent_pool = [agent_class() for _ in range(10)]
    
    def train(self):
        for episode in range(1000):
            # Select opponent
            opponent = random.choice(self.opponent_pool)
            
            # Play match
            winner = self.play_match(self.main_agent, opponent)
            
            # Update agents
            if winner == self.main_agent:
                self.main_agent.update(positive_reward=True)
            else:
                opponent.update(positive_reward=True)
            
            # Periodically update pool
            if episode % 100 == 0:
                self.opponent_pool.append(self.main_agent.clone())
```

### Experience Replay Strategies
```python
class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
    
    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            idx = np.argmin(self.priorities)
            self.buffer[idx] = experience
            self.priorities[idx] = priority
    
    def sample(self, batch_size, beta=0.4):
        # Calculate sampling probabilities
        probs = np.array(self.priorities) ** beta
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probs
        )
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights, indices
```

## Advanced Techniques

### Multi-Agent Coordination
```python
class MultiAgentCoordinator:
    def __init__(self, n_agents):
        self.agents = [DQNAgent() for _ in range(n_agents)]
        self.communication_network = self.build_comm_network()
    
    def coordinate_actions(self, states):
        # Get individual observations
        observations = [agent.observe(state) 
                       for agent, state in zip(self.agents, states)]
        
        # Share information
        messages = self.communication_network(observations)
        
        # Coordinated action selection
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(observations[i], messages[i])
            actions.append(action)
        
        return actions
```

### Meta-Learning
```python
class MAMLAgent(Agent):
    """Model-Agnostic Meta-Learning"""
    
    def __init__(self):
        self.meta_model = self.build_model()
        self.meta_optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=0.001
        )
    
    def meta_train(self, task_batch):
        meta_loss = 0
        
        for task in task_batch:
            # Clone model for task
            task_model = self.clone_model(self.meta_model)
            
            # Inner loop: adapt to task
            for _ in range(5):
                loss = self.train_on_task(task_model, task)
                self.inner_update(task_model, loss)
            
            # Outer loop: meta-update
            meta_loss += self.evaluate_task(task_model, task)
        
        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
```

### Hierarchical Agents
```python
class HierarchicalAgent(Agent):
    def __init__(self):
        self.high_level_policy = self.build_high_level()
        self.low_level_policies = {
            "combat": CombatPolicy(),
            "exploration": ExplorationPolicy(),
            "resource": ResourcePolicy()
        }
    
    def act(self, state):
        # High-level decision
        goal = self.high_level_policy(state)
        
        # Low-level execution
        policy = self.low_level_policies[goal]
        action = policy.act(state, goal)
        
        return action
```

## Examples

### FPS Agent
```python
class FPSAgent(DQNAgent):
    def __init__(self, game):
        super().__init__(game)
        self.aim_controller = AimController()
        self.movement_controller = MovementController()
    
    def act(self, state):
        # Separate aim and movement
        aim_action = self.aim_controller.compute_aim(
            state["crosshair"],
            state["enemies"]
        )
        
        movement_action = self.movement_controller.compute_movement(
            state["player_pos"],
            state["obstacles"],
            state["objectives"]
        )
        
        # Combine actions
        return {
            "aim": aim_action,
            "movement": movement_action,
            "shoot": self.should_shoot(state)
        }
    
    def should_shoot(self, state):
        # Check if enemy in crosshair
        if state["enemy_in_crosshair"]:
            # Check ammo
            if state["ammo"] > 0:
                return True
        return False
```

### Strategy Game Agent
```python
class RTSAgent(Agent):
    def __init__(self, game):
        super().__init__(game)
        self.strategy_network = StrategyNetwork()
        self.tactics_network = TacticsNetwork()
        self.micro_controller = MicroController()
    
    def act(self, state):
        # Strategic decisions (build order, tech path)
        strategy = self.strategy_network(state["global_state"])
        
        # Tactical decisions (unit composition, positioning)
        tactics = self.tactics_network(state["local_state"], strategy)
        
        # Micro management (individual unit control)
        unit_actions = self.micro_controller(state["units"], tactics)
        
        return self.execute_actions(unit_actions)
```

## Performance Optimization

### Frame Skipping
```python
class FrameSkipAgent(Agent):
    def __init__(self, skip_frames=4):
        self.skip_frames = skip_frames
        self.action_repeat = None
    
    def act(self, state):
        if self.action_repeat is not None and self.action_repeat > 0:
            self.action_repeat -= 1
            return self.last_action
        
        # Compute new action
        action = super().act(state)
        self.last_action = action
        self.action_repeat = self.skip_frames - 1
        
        return action
```

### Parallel Training
```python
import multiprocessing as mp

class ParallelTrainer:
    def __init__(self, agent_class, n_workers=4):
        self.n_workers = n_workers
        self.agent_class = agent_class
    
    def train(self):
        with mp.Pool(self.n_workers) as pool:
            # Collect experiences in parallel
            experiences = pool.map(
                self.collect_episode,
                range(self.n_workers)
            )
            
            # Aggregate and train
            all_experiences = [e for exp_list in experiences 
                              for e in exp_list]
            self.agent.train_on_batch(all_experiences)
```

## Debugging and Visualization

### Action Visualization
```python
def visualize_actions(self, frame, action, q_values=None):
    vis_frame = frame.copy()
    
    # Draw action
    cv2.putText(vis_frame, f"Action: {action}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw Q-values
    if q_values is not None:
        y_offset = 60
        for i, q in enumerate(q_values):
            text = f"Q[{i}]: {q:.3f}"
            color = (0, 255, 0) if i == action else (255, 255, 255)
            cv2.putText(vis_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    return vis_frame
```

### Training Metrics
```python
class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot(self):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward plot
        axes[0, 0].plot(self.metrics["reward"])
        axes[0, 0].set_title("Episode Reward")
        
        # Loss plot
        axes[0, 1].plot(self.metrics["loss"])
        axes[0, 1].set_title("Training Loss")
        
        # Success rate
        axes[1, 0].plot(self.metrics["success_rate"])
        axes[1, 0].set_title("Success Rate")
        
        # Epsilon
        axes[1, 1].plot(self.metrics["epsilon"])
        axes[1, 1].set_title("Exploration Rate")
        
        plt.tight_layout()
        plt.show()
```

## Next Steps

- Explore [[Reinforcement Learning]] in depth
- Learn about [[Computer Vision Models]]
- See [[Training Best Practices]]
- Check out [[Example Agents]]

---

<p align="center">
  <a href="https://github.com/neophrythe/nexus-ai/wiki/CLI-Commands">Next: CLI Commands →</a>
</p>