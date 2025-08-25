<div align="center">

# Nexus Game AI Framework

<img src="https://raw.githubusercontent.com/neophrythe/nexus-ai/main/logo.png" alt="Nexus Game AI Framework" width="500">

**Nexus - Game AI Development Framework**  
*Turn Any Game Into a Self-Playing AI Sandbox*

<a href="https://github.com/neophrythe/nexus-game-ai/wiki">Wiki</a> • 
<a href="https://discord.gg/nexus">Discord</a>

</div>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img src="https://img.shields.io/badge/license-AGPL%2FCommercial-red.svg">
  <img src="https://img.shields.io/badge/platform-windows%20%7C%20linux-lightgrey.svg">
  <img src="https://img.shields.io/badge/version-1.0.0-orange.svg">
</p>

## 🎮 What is Nexus?

Nexus is a modern game AI development framework that empowers developers to create intelligent game-playing agents. Whether you're interested in reinforcement learning, computer vision, or automated game testing, Nexus provides the tools you need to transform any game into an AI playground.

Built as a spiritual successor to SerpentAI, Nexus maintains full compatibility while introducing cutting-edge features and modern architectural improvements.

### ✨ Key Features

- **🎯 Universal Game Support** - Works with any game that runs on your computer
- **🤖 State-of-the-Art AI** - Includes Rainbow DQN, PPO, and modern vision models
- **👁️ Advanced Computer Vision** - YOLO v8, SAM, Vision Transformers, and OCR
- **🎬 Synchronized Recording** - Record gameplay with frame-perfect input capture
- **🔌 Plugin Architecture** - Extend functionality with hot-reloadable plugins
- **📊 Visual Debugging** - Real-time overlay system for debugging AI behavior
- **🚀 High Performance** - Optimized for 60+ FPS capture and processing
- **🔧 Developer Friendly** - Comprehensive API and CLI tools

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Windows 10/11 or Linux (Ubuntu 20.04+)
- NVIDIA GPU recommended (CUDA support for deep learning)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/neophrythe/nexus-game-ai.git
cd nexus-game-ai

# Install Nexus
pip install -e .

# Or install with all features
pip install -e .[full]
```

### Platform-Specific Requirements

#### Windows
```bash
pip install -e .[windows]
```

#### Linux
```bash
# Install system dependencies
sudo apt-get install python3-tk python3-dev xdotool

pip install -e .
```

## 🚀 Quick Start

### 1. Initialize Nexus

```bash
nexus setup
```

### 2. Create a Game Plugin

```bash
nexus generate game MyGame
```

### 3. Create an Agent

```bash
nexus generate agent MyAgent --type=dqn
```

### 4. Start Training

```bash
nexus train --game=MyGame --agent=MyAgent
```

### 5. Watch Your AI Play

```bash
nexus play --game=MyGame --agent=MyAgent
```

## 📖 Documentation

### Core Concepts

#### Game Plugins
Game plugins define how Nexus interacts with a specific game:

```python
from nexus.game import Game

class MyGame(Game):
    def __init__(self):
        super().__init__("MyGame", launch_cmd="mygame.exe")
        
    def define_game_region(self):
        return (0, 0, 1920, 1080)  # Full screen
        
    def define_actions(self):
        return ["UP", "DOWN", "LEFT", "RIGHT", "ACTION"]
```

#### Agent Development
Create intelligent agents using various approaches:

```python
from nexus.agents import DQNAgent

class MyGameAgent(DQNAgent):
    def __init__(self, game):
        super().__init__(game, learning_rate=0.001)
        
    def process_frame(self, frame):
        # Preprocess the game frame
        return self.resize_and_normalize(frame)
        
    def get_reward(self, frame):
        # Define your reward function
        score = self.extract_score(frame)
        return score - self.previous_score
```

#### Frame Processing Pipeline
Advanced frame processing with transformation chains:

```python
from nexus.vision import FrameTransformer

transformer = FrameTransformer()
transformer.add_resize((224, 224))
transformer.add_grayscale()
transformer.add_normalize()

processed_frame = transformer.transform(game_frame)
```

### Advanced Features

#### 🎥 Synchronized Recording
Record gameplay sessions with perfect frame-input synchronization:

```bash
nexus record --game=MyGame --output=gameplay.h5
```

#### 🔍 Visual Debugging
Enable real-time visual debugging overlays:

```bash
nexus debug --game=MyGame --show-detections --show-inputs
```

#### 📊 Experiment Tracking
Track training metrics with integrated MLOps tools:

```python
from nexus.analytics import ExperimentTracker

tracker = ExperimentTracker(backend="wandb")
tracker.log_metrics({"reward": 100, "loss": 0.5})
```

#### 🎮 Input Control
Human-like input simulation with Bézier curves:

```python
from nexus.input import InputController

controller = InputController(human_like=True)
controller.move_mouse_smooth(x=500, y=300, duration=0.5)
controller.click()
```

## 🔧 CLI Commands

```bash
# Setup and Configuration
nexus setup                    # Initial setup
nexus config                    # Configure Nexus

# Game Management  
nexus list games               # List available games
nexus launch <game>            # Launch a game
nexus capture <game>           # Test frame capture

# Agent Operations
nexus train <agent>            # Train an agent
nexus play <agent>             # Run agent in play mode
nexus evaluate <agent>         # Evaluate agent performance

# Development Tools
nexus generate game <name>     # Generate game plugin
nexus generate agent <name>    # Generate agent plugin
nexus visual-debugger          # Launch visual debugger
nexus record                   # Record gameplay

# Plugin Management
nexus plugin install <name>    # Install a plugin
nexus plugin list              # List installed plugins
nexus plugin remove <name>     # Remove a plugin
```

## 🏗️ Architecture

```
Nexus Game AI Framework
├── Capture System (MSS, DXCam, OpenCV)
├── Vision Pipeline (YOLO, SAM, OCR, CV)
├── Agent Framework (RL, Scripted, LLM)
├── Input Controller (PyAutoGUI, Native APIs)
├── Plugin System (Hot-reload, Sandboxed)
├── Analytics (W&B, MLflow, TensorBoard)
└── Visual Debugger (Qt-based GUI)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/neophrythe/nexus-game-ai.git
cd nexus-game-ai

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Run linting
flake8 nexus/
black nexus/
```

## 📚 Examples

### Training a DQN Agent on a Retro Game

```python
from nexus import Game, DQNAgent, Trainer

# Define the game
game = Game("RetroGame", window_name="RetroGame Window")

# Create agent
agent = DQNAgent(
    game=game,
    model="cnn",
    learning_rate=0.0001,
    batch_size=32
)

# Train
trainer = Trainer(game, agent)
trainer.train(episodes=1000)
```

### Object Detection in Games

```python
from nexus.vision import ObjectDetector

detector = ObjectDetector(model="yolov8")
game_frame = capture.grab_frame()

detections = detector.detect(game_frame)
for obj in detections:
    print(f"Found {obj.label} at {obj.bbox} with confidence {obj.confidence}")
```

### Creating a Scripted Bot

```python
from nexus.agents import ScriptedAgent

class MyBot(ScriptedAgent):
    def act(self, frame):
        # Detect enemies
        enemies = self.detect_enemies(frame)
        
        if enemies:
            # Aim at closest enemy
            closest = min(enemies, key=lambda e: e.distance)
            self.aim_at(closest.x, closest.y)
            self.shoot()
        else:
            # Patrol
            self.move_forward()
            self.look_around()
```

## 🔗 Ecosystem

- **[nexus-plugins](https://github.com/neophrythe/nexus-plugins)** - Community plugin repository
- **[nexus-models](https://github.com/neophrythe/nexus-models)** - Pre-trained models
- **[nexus-datasets](https://github.com/neophrythe/nexus-datasets)** - Game datasets
- **[nexus-tutorials](https://github.com/neophrythe/nexus-tutorials)** - Tutorial series

## 📊 Performance Benchmarks

| Feature | Performance |
|---------|------------|
| Frame Capture | 60+ FPS @ 1080p |
| Object Detection | 30+ FPS (YOLO v8) |
| Agent Decision Time | <10ms |
| Input Latency | <1ms |
| Memory Usage | ~500MB base |

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Low FPS during capture
```bash
# Switch to faster capture backend
nexus config set capture.backend dxcam  # Windows
nexus config set capture.backend mss     # Cross-platform
```

**Issue**: CUDA not detected
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Issue**: Game not detected
```bash
# Manually specify window
nexus capture --window-title "Exact Window Title"
```

## 📄 License

**⚠️ DUAL LICENSE - COMMERCIAL USE REQUIRES PAYMENT**

- **Non-Commercial Use:** AGPL-3.0 (free for personal, educational, and research use)
- **Commercial Use:** Requires paid commercial license - contact contact@digitalmanufacturinglabs.de

Commercial use includes: game bot services, SaaS offerings, proprietary software integration, or any for-profit use.

See [LICENSE](LICENSE) file for complete terms.

## 🙏 Acknowledgments

- Inspired by the original [SerpentAI](https://github.com/SerpentAI/SerpentAI) framework
- Built with love by the game AI community
- Special thanks to all contributors and supporters

## 📞 Support

- **Discord**: [Join our community](https://discord.gg/nexus)
- **Issues**: [GitHub Issues](https://github.com/neophrythe/nexus-game-ai/issues)
- **Wiki**: [Documentation Wiki](https://github.com/neophrythe/nexus-game-ai/wiki)
- **Email**: contact@digitalmanufacturinglabs.de

---

<p align="center">
  Made with ❤️ for the Game AI Community<br>
  <a href="https://github.com/neophrythe/nexus-game-ai">Star us on GitHub!</a>
</p>