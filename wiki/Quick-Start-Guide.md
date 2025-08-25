# Quick Start Guide

Get your first AI agent running in 5 minutes! This guide will walk you through creating a simple game-playing agent.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Setup Nexus](#step-1-setup-nexus)
- [Step 2: Create a Game Plugin](#step-2-create-a-game-plugin)
- [Step 3: Create an Agent](#step-3-create-an-agent)
- [Step 4: Train Your Agent](#step-4-train-your-agent)
- [Step 5: Watch It Play](#step-5-watch-it-play)
- [Next Steps](#next-steps)

## Prerequisites

Make sure you have:
- ✅ Nexus installed (see [[Installation]])
- ✅ A game running on your computer
- ✅ 5 minutes of time

## Step 1: Setup Nexus

Initialize Nexus configuration:

```bash
# Run initial setup
nexus setup

# This will:
# - Create configuration files
# - Set up directories
# - Detect system capabilities
# - Initialize plugin system
```

Verify everything is working:

```bash
# Run system check
nexus doctor

# Expected output:
# ✓ Python version: 3.10.0
# ✓ Nexus version: 1.0.0
# ✓ Frame capture: Available
# ✓ Input control: Available
# ✓ GPU support: CUDA available
```

## Step 2: Create a Game Plugin

Let's create a plugin for your game. We'll use a simple example with a browser game:

```bash
# Generate game plugin
nexus generate game FlappyBird

# This creates:
# plugins/FlappyBird/
# ├── __init__.py
# ├── flappy_bird.py
# └── manifest.yaml
```

Edit the game plugin (`plugins/FlappyBird/flappy_bird.py`):

```python
from nexus.game import Game

class FlappyBird(Game):
    def __init__(self):
        super().__init__(
            name="FlappyBird",
            platform="browser",
            window_name="Flappy Bird"  # Browser tab title
        )
    
    def define_game_region(self):
        """Define the game area on screen"""
        # Return None for full window, or specific region
        return None  # Uses full window
    
    def define_actions(self):
        """Define possible actions"""
        return {
            "FLAP": "space",  # Space bar to flap
            "NOTHING": None   # Do nothing
        }
    
    def define_rewards(self):
        """Define reward regions for OCR"""
        return {
            "score": (100, 50, 200, 100),  # x, y, width, height
            "game_over": (400, 300, 200, 100)
        }
```

Test the game plugin:

```bash
# Launch game and test capture
nexus test game FlappyBird

# This will:
# - Find the game window
# - Capture a frame
# - Display the frame with regions
```

## Step 3: Create an Agent

Now let's create an AI agent to play the game:

```bash
# Generate agent plugin
nexus generate agent FlappyBirdAgent --type=dqn

# This creates:
# plugins/FlappyBirdAgent/
# ├── __init__.py
# ├── flappy_bird_agent.py
# └── manifest.yaml
```

Edit the agent (`plugins/FlappyBirdAgent/flappy_bird_agent.py`):

```python
from nexus.agents import DQNAgent
import numpy as np

class FlappyBirdAgent(DQNAgent):
    def __init__(self, game):
        super().__init__(
            game=game,
            learning_rate=0.001,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
    
    def process_frame(self, frame):
        """Preprocess game frame"""
        # Convert to grayscale and resize
        gray = self.to_grayscale(frame)
        resized = self.resize(gray, (84, 84))
        normalized = resized / 255.0
        return normalized
    
    def get_state(self, frame):
        """Extract game state from frame"""
        processed = self.process_frame(frame)
        
        # Simple state: bird position and pipes
        bird_y = self.detect_bird(processed)
        pipe_distance = self.detect_nearest_pipe(processed)
        
        return np.array([bird_y, pipe_distance])
    
    def get_reward(self, frame, info):
        """Calculate reward"""
        # Check if game over
        if self.is_game_over(frame):
            return -100
        
        # Small reward for surviving
        return 1
    
    def detect_bird(self, frame):
        """Detect bird position (simplified)"""
        # Use color detection or template matching
        # Returns Y position normalized 0-1
        return 0.5
    
    def detect_nearest_pipe(self, frame):
        """Detect nearest pipe distance (simplified)"""
        # Returns distance normalized 0-1
        return 0.5
    
    def is_game_over(self, frame):
        """Check if game over screen is visible"""
        # Use OCR or template matching
        return False
```

## Step 4: Train Your Agent

Start training your agent:

```bash
# Basic training
nexus train --game=FlappyBird --agent=FlappyBirdAgent --episodes=100

# With visualization
nexus train --game=FlappyBird --agent=FlappyBirdAgent --episodes=100 --visualize

# With specific settings
nexus train \
    --game=FlappyBird \
    --agent=FlappyBirdAgent \
    --episodes=1000 \
    --save-interval=100 \
    --log-interval=10
```

Monitor training progress:

```bash
# In another terminal, watch metrics
nexus monitor

# Or use TensorBoard
tensorboard --logdir=logs/
```

Training output example:
```
Episode 1/1000 | Reward: 5 | Epsilon: 1.00 | Loss: 0.000
Episode 10/1000 | Reward: 12 | Epsilon: 0.95 | Loss: 0.123
Episode 50/1000 | Reward: 45 | Epsilon: 0.78 | Loss: 0.089
Episode 100/1000 | Reward: 120 | Epsilon: 0.61 | Loss: 0.045
```

## Step 5: Watch It Play

After training, watch your agent play:

```bash
# Run in play mode
nexus play --game=FlappyBird --agent=FlappyBirdAgent

# With debug overlay
nexus play --game=FlappyBird --agent=FlappyBirdAgent --debug

# Record gameplay
nexus play --game=FlappyBird --agent=FlappyBirdAgent --record=gameplay.mp4
```

## Complete Example Script

Here's everything in a single Python script:

```python
# train_flappy_bird.py
from nexus import Game, DQNAgent, Trainer

# Define the game
class FlappyBird(Game):
    def __init__(self):
        super().__init__("FlappyBird", window_name="Flappy Bird")
    
    def define_actions(self):
        return ["FLAP", "NOTHING"]

# Define the agent
class FlappyBirdBot(DQNAgent):
    def __init__(self, game):
        super().__init__(game, learning_rate=0.001)
    
    def get_reward(self, frame, info):
        if info.get("game_over"):
            return -100
        return 1  # Reward for surviving

# Train
if __name__ == "__main__":
    game = FlappyBird()
    agent = FlappyBirdBot(game)
    
    trainer = Trainer(game, agent)
    trainer.train(episodes=1000)
    
    # Save the trained model
    agent.save("flappy_bird_model.pt")
    
    # Play with trained model
    agent.play(episodes=10)
```

Run it:
```bash
python train_flappy_bird.py
```

## Visual Debugger

Launch the visual debugger to see what your agent sees:

```bash
# Start visual debugger
nexus visual-debugger

# Connect to running agent
nexus visual-debugger --connect
```

Features:
- Real-time frame display
- Detection overlays
- Metrics graphs
- Input visualization
- Reward tracking

## Tips for Success

### 1. Start Simple
- Begin with simple games (Flappy Bird, Snake, Pong)
- Use basic rewards (survival, score)
- Gradually increase complexity

### 2. Preprocessing Matters
- Resize frames to reduce computation (84x84 is common)
- Convert to grayscale if color isn't important
- Normalize pixel values (0-1 range)

### 3. Reward Engineering
- Give immediate feedback (+1 for good, -1 for bad)
- Use sparse rewards carefully
- Consider reward shaping for complex games

### 4. Training Tips
- Start with high exploration (epsilon=1.0)
- Decrease exploration gradually
- Save checkpoints frequently
- Monitor loss and reward trends

### 5. Debug Effectively
- Use visual debugger to see what agent sees
- Log state values and actions
- Verify reward calculations
- Check frame capture quality

## Common Issues

### Game Not Detected
```bash
# List all windows
nexus list windows

# Specify exact window title
nexus test game FlappyBird --window="Flappy Bird - Google Chrome"
```

### Low FPS
```bash
# Reduce capture resolution
nexus config set capture.resize 320x240

# Skip frames
nexus config set capture.frame_skip 2
```

### Agent Not Learning
- Check reward function (is it providing signal?)
- Verify state representation (is it informative?)
- Adjust learning rate (try 0.0001 to 0.01)
- Increase training episodes

## Next Steps

Now that you have a working agent:

1. **Improve Your Agent**
   - Read [[Agent Development]] for advanced techniques
   - Try different algorithms (PPO, Rainbow DQN)
   - Implement better state representations

2. **Try Computer Vision**
   - Learn about [[Object Detection]]
   - Use [[Template Matching]] for game elements
   - Implement [[OCR]] for score reading

3. **Scale Up**
   - Train on multiple games simultaneously
   - Use [[Distributed Training]]
   - Implement [[Curriculum Learning]]

4. **Share Your Work**
   - Upload to [[Community Plugins]]
   - Create a tutorial
   - Share on Discord

## Resources

- [[Tutorial: Training a DQN Agent]] - Deep dive into DQN
- [[Tutorial: Computer Vision]] - Advanced frame processing
- [[Example: FPS Bot]] - First-person shooter example
- [[API Reference]] - Complete API documentation

---

<p align="center">
  <b>Congratulations! You've created your first AI agent! 🎉</b><br><br>
  <a href="https://github.com/neophrythe/nexus-ai/wiki/Agent-Development">Next: Agent Development →</a>
</p>