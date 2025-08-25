# Game Plugins

Game plugins are the bridge between Nexus and the games you want to automate. This guide covers everything you need to know about creating and managing game plugins.

## Table of Contents
- [Overview](#overview)
- [Plugin Structure](#plugin-structure)
- [Creating a Game Plugin](#creating-a-game-plugin)
- [Game Detection](#game-detection)
- [Action Mapping](#action-mapping)
- [State Extraction](#state-extraction)
- [Advanced Features](#advanced-features)
- [Examples](#examples)

## Overview

A game plugin defines:
- How to find and launch the game
- The game window and play area
- Available actions and controls
- How to extract game state
- Reward calculation methods
- Game-specific optimizations

## Plugin Structure

```
plugins/MyGame/
├── __init__.py           # Package init
├── my_game.py           # Main game class
├── manifest.yaml        # Plugin metadata
├── sprites/            # Sprite templates (optional)
│   ├── player.png
│   ├── enemy.png
│   └── items.png
├── models/             # Trained models (optional)
│   └── detector.pt
└── config.yaml         # Game-specific config
```

### Manifest File

```yaml
# manifest.yaml
name: MyGame
version: 1.0.0
type: game
author: YourName
description: Plugin for MyGame

game:
  platform: pc  # pc, browser, mobile
  executable: mygame.exe
  window_title: "My Game"
  
requirements:
  - opencv-python
  - pytesseract

config:
  fps: 60
  resolution: [1920, 1080]
```

## Creating a Game Plugin

### Basic Game Class

```python
from nexus.game import Game
from nexus.game.registry import register_game

@register_game("MyGame")
class MyGame(Game):
    def __init__(self):
        super().__init__(
            name="MyGame",
            platform="pc",
            executable="C:/Games/MyGame/game.exe"
        )
        
        # Game-specific settings
        self.fps_target = 60
        self.resolution = (1920, 1080)
    
    def setup(self):
        """Initialize game-specific resources"""
        # Load sprites, models, etc.
        self.load_sprites()
        self.load_models()
    
    def validate_environment(self):
        """Check if game can run"""
        if not self.is_installed():
            raise GameNotFoundError("MyGame is not installed")
        return True
```

### Window Management

```python
def find_window(self):
    """Find game window"""
    # Try multiple methods
    windows = [
        self.find_by_title("My Game"),
        self.find_by_class("UnityWndClass"),
        self.find_by_process("mygame.exe")
    ]
    
    for window in windows:
        if window:
            return window
    
    return None

def define_game_region(self):
    """Define playable area"""
    # Full window
    return None
    
    # Specific region (x, y, width, height)
    return (100, 100, 1720, 880)
    
    # Dynamic detection
    window = self.get_window()
    if window:
        # Remove UI borders
        x, y, w, h = window.get_bounds()
        return (x + 10, y + 30, w - 20, h - 40)
```

## Game Detection

### Platform-Specific Detection

```python
class SteamGame(Game):
    def __init__(self, app_id):
        super().__init__(
            name="SteamGame",
            platform="steam",
            steam_app_id=app_id
        )
    
    def launch(self):
        """Launch via Steam"""
        import webbrowser
        webbrowser.open(f"steam://run/{self.steam_app_id}")
        
        # Wait for window
        return self.wait_for_window(timeout=30)

class BrowserGame(Game):
    def __init__(self, url):
        super().__init__(
            name="BrowserGame",
            platform="browser",
            url=url
        )
    
    def launch(self):
        """Open in browser"""
        import webbrowser
        webbrowser.open(self.url)
        
        # Find browser tab
        return self.find_browser_tab(self.url)
```

### Multi-Window Games

```python
def find_all_windows(self):
    """Find all game windows"""
    windows = {
        "main": self.find_by_title("Game - Main"),
        "map": self.find_by_title("Game - Map"),
        "inventory": self.find_by_title("Game - Inventory")
    }
    return windows

def capture_all_windows(self):
    """Capture from multiple windows"""
    frames = {}
    for name, window in self.windows.items():
        if window:
            frames[name] = self.capture_window(window)
    return frames
```

## Action Mapping

### Simple Actions

```python
def define_actions(self):
    """Define available actions"""
    return {
        # Movement
        "MOVE_UP": "w",
        "MOVE_DOWN": "s",
        "MOVE_LEFT": "a",
        "MOVE_RIGHT": "d",
        
        # Actions
        "JUMP": "space",
        "SHOOT": "mouse_left",
        "RELOAD": "r",
        "INTERACT": "e",
        
        # Special
        "NOTHING": None
    }
```

### Complex Actions

```python
def define_actions(self):
    """Define complex actions"""
    return {
        "MOVE": {
            "type": "continuous",
            "keys": ["w", "a", "s", "d"],
            "mouse": True
        },
        "AIM": {
            "type": "mouse_move",
            "relative": True,
            "sensitivity": 0.5
        },
        "BUILD": {
            "type": "sequence",
            "actions": ["b", "mouse_left", "mouse_left"]
        },
        "COMBO": {
            "type": "macro",
            "sequence": [
                ("key", "q", 0.1),
                ("key", "w", 0.1),
                ("key", "e", 0.2),
                ("mouse", "left", 0.1)
            ]
        }
    }
```

### Action Execution

```python
def execute_action(self, action, duration=None):
    """Execute game action"""
    if isinstance(action, str):
        # Simple key press
        self.input_controller.press_key(action, duration)
    
    elif isinstance(action, dict):
        action_type = action.get("type")
        
        if action_type == "mouse_move":
            x, y = action.get("x"), action.get("y")
            self.input_controller.move_mouse(x, y)
        
        elif action_type == "sequence":
            for sub_action in action.get("actions", []):
                self.execute_action(sub_action)
                time.sleep(0.1)
        
        elif action_type == "combo":
            self.execute_combo(action.get("keys", []))
```

## State Extraction

### Visual State

```python
def extract_state(self, frame):
    """Extract game state from frame"""
    state = {}
    
    # Object detection
    state["enemies"] = self.detect_enemies(frame)
    state["items"] = self.detect_items(frame)
    state["obstacles"] = self.detect_obstacles(frame)
    
    # OCR for text
    state["score"] = self.read_score(frame)
    state["health"] = self.read_health(frame)
    state["ammo"] = self.read_ammo(frame)
    
    # Color-based detection
    state["minimap"] = self.extract_minimap(frame)
    
    return state

def detect_enemies(self, frame):
    """Detect enemies using YOLO"""
    detections = self.detector.detect(frame, classes=["enemy"])
    return [
        {
            "bbox": det.bbox,
            "confidence": det.confidence,
            "distance": self.estimate_distance(det)
        }
        for det in detections
    ]
```

### Memory-Based State

```python
class StatefulGame(Game):
    def __init__(self):
        super().__init__("StatefulGame")
        self.state_history = deque(maxlen=10)
        self.action_history = deque(maxlen=10)
    
    def update_state(self, frame, action):
        """Track state over time"""
        current_state = self.extract_state(frame)
        
        # Add temporal information
        current_state["velocity"] = self.calculate_velocity(
            self.state_history[-1] if self.state_history else None,
            current_state
        )
        
        # Track patterns
        current_state["pattern"] = self.detect_pattern(
            self.state_history,
            current_state
        )
        
        self.state_history.append(current_state)
        self.action_history.append(action)
        
        return current_state
```

## Advanced Features

### Sprite Management

```python
def load_sprites(self):
    """Load sprite templates"""
    self.sprites = {}
    sprite_dir = Path(__file__).parent / "sprites"
    
    for sprite_file in sprite_dir.glob("*.png"):
        name = sprite_file.stem
        self.sprites[name] = cv2.imread(str(sprite_file))
    
    # Generate variations
    self.generate_sprite_variations()

def find_sprite(self, frame, sprite_name, threshold=0.8):
    """Find sprite in frame"""
    if sprite_name not in self.sprites:
        return None
    
    template = self.sprites[sprite_name]
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    
    locations = np.where(result >= threshold)
    return list(zip(locations[1], locations[0]))
```

### Game-Specific Optimizations

```python
def optimize_capture(self, frame):
    """Optimize frame for this game"""
    # Crop to play area
    frame = frame[100:980, 100:1820]
    
    # Reduce color space for retro games
    if self.is_retro:
        frame = self.quantize_colors(frame, n_colors=16)
    
    # Enhance contrast for dark games
    if self.is_dark:
        frame = self.enhance_contrast(frame)
    
    # Remove UI elements
    if self.ui_mask is not None:
        frame = cv2.bitwise_and(frame, frame, mask=self.ui_mask)
    
    return frame
```

### Event Detection

```python
def detect_events(self, frame, previous_frame=None):
    """Detect game events"""
    events = []
    
    # Death detection
    if self.is_death_screen(frame):
        events.append({"type": "death", "timestamp": time.time()})
    
    # Level completion
    if self.is_level_complete(frame):
        events.append({"type": "level_complete", "timestamp": time.time()})
    
    # Item pickup (via frame diff)
    if previous_frame is not None:
        diff = cv2.absdiff(frame, previous_frame)
        if self.detect_pickup_animation(diff):
            events.append({"type": "item_pickup", "timestamp": time.time()})
    
    # Audio cues (if available)
    if self.audio_enabled:
        audio_events = self.detect_audio_events()
        events.extend(audio_events)
    
    return events
```

## Examples

### FPS Game Plugin

```python
@register_game("CounterStrike")
class CounterStrike(Game):
    def __init__(self):
        super().__init__(
            name="Counter-Strike",
            platform="steam",
            steam_app_id=730
        )
        
        self.crosshair_position = (960, 540)
        self.fov = 90
    
    def define_actions(self):
        return {
            # Movement
            "FORWARD": "w",
            "BACKWARD": "s",
            "STRAFE_LEFT": "a",
            "STRAFE_RIGHT": "d",
            "JUMP": "space",
            "CROUCH": "ctrl",
            
            # Combat
            "SHOOT": "mouse_left",
            "AIM": "mouse_right",
            "RELOAD": "r",
            "SWITCH_WEAPON": "q",
            
            # Mouse aim (continuous)
            "AIM_AT": {
                "type": "mouse_move",
                "relative": False
            }
        }
    
    def get_reward(self, state, info):
        reward = 0
        
        # Kills
        reward += info.get("kills", 0) * 100
        
        # Deaths
        reward -= info.get("deaths", 0) * 50
        
        # Damage dealt
        reward += info.get("damage_dealt", 0) * 0.5
        
        # Objective
        if info.get("bomb_planted"):
            reward += 200
        if info.get("bomb_defused"):
            reward += 500
        
        return reward
```

### Strategy Game Plugin

```python
@register_game("AgeOfEmpires")
class AgeOfEmpires(Game):
    def __init__(self):
        super().__init__(
            name="Age of Empires",
            platform="pc"
        )
        
        self.resource_regions = {
            "food": (100, 10, 150, 30),
            "wood": (250, 10, 150, 30),
            "gold": (400, 10, 150, 30),
            "stone": (550, 10, 150, 30)
        }
    
    def extract_state(self, frame):
        state = super().extract_state(frame)
        
        # Read resources via OCR
        for resource, region in self.resource_regions.items():
            value = self.ocr_region(frame, region)
            state[f"resource_{resource}"] = int(value) if value else 0
        
        # Minimap analysis
        minimap = self.extract_minimap(frame)
        state["enemy_positions"] = self.analyze_minimap(minimap)
        
        # Unit detection
        state["units"] = self.detect_units(frame)
        state["buildings"] = self.detect_buildings(frame)
        
        return state
    
    def define_actions(self):
        return {
            # Camera
            "PAN_UP": {"type": "key", "key": "up"},
            "PAN_DOWN": {"type": "key", "key": "down"},
            "PAN_LEFT": {"type": "key", "key": "left"},
            "PAN_RIGHT": {"type": "key", "key": "right"},
            
            # Selection
            "SELECT": {"type": "mouse", "button": "left"},
            "BOX_SELECT": {"type": "drag", "button": "left"},
            "ADD_TO_SELECTION": {"type": "mouse", "button": "left", "modifier": "shift"},
            
            # Commands
            "MOVE": {"type": "mouse", "button": "right"},
            "ATTACK": {"type": "key", "key": "a"},
            "BUILD": {"type": "key", "key": "b"},
            "GATHER": {"type": "key", "key": "g"}
        }
```

### Platformer Plugin

```python
@register_game("SuperMario")
class SuperMario(Game):
    def __init__(self):
        super().__init__(
            name="Super Mario",
            platform="emulator"
        )
        
        self.mario_template = cv2.imread("sprites/mario.png")
    
    def extract_state(self, frame):
        # Simplified state for platformer
        state = {}
        
        # Find Mario
        mario_pos = self.find_mario(frame)
        if mario_pos:
            state["mario_x"] = mario_pos[0] / frame.shape[1]
            state["mario_y"] = mario_pos[1] / frame.shape[0]
        
        # Detect platforms
        platforms = self.detect_platforms(frame)
        state["platforms"] = platforms
        
        # Detect enemies
        enemies = self.detect_enemies(frame)
        state["enemies"] = enemies
        
        # Detect collectibles
        coins = self.detect_coins(frame)
        state["coins"] = coins
        
        return state
    
    def find_mario(self, frame):
        """Find Mario using template matching"""
        result = cv2.matchTemplate(
            frame, 
            self.mario_template,
            cv2.TM_CCOEFF_NORMED
        )
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.8:
            return max_loc
        return None
```

## Best Practices

### 1. Robust Window Detection
- Use multiple detection methods
- Handle window minimization/focus loss
- Verify window dimensions

### 2. Efficient State Extraction
- Cache expensive operations
- Use region-based processing
- Implement frame skipping for slow operations

### 3. Action Validation
- Verify actions are possible in current state
- Add cooldowns for actions
- Handle input lag

### 4. Error Handling
```python
def safe_extract_state(self, frame):
    try:
        return self.extract_state(frame)
    except Exception as e:
        self.logger.error(f"State extraction failed: {e}")
        return self.get_default_state()
```

### 5. Configuration
```python
# config.yaml
game:
  name: MyGame
  version: 1.0.0
  
capture:
  fps: 30
  region: [0, 0, 1920, 1080]
  
detection:
  confidence_threshold: 0.7
  nms_threshold: 0.4
  
performance:
  frame_skip: 2
  cache_sprites: true
```

## Debugging

### Visual Debugging
```python
def debug_frame(self, frame):
    """Add debug overlays"""
    debug_frame = frame.copy()
    
    # Draw regions
    for name, region in self.regions.items():
        x, y, w, h = region
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_frame, name, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw detections
    for detection in self.last_detections:
        self.draw_detection(debug_frame, detection)
    
    return debug_frame
```

### Logging
```python
import structlog

class MyGame(Game):
    def __init__(self):
        super().__init__("MyGame")
        self.logger = structlog.get_logger()
    
    def extract_state(self, frame):
        self.logger.debug("Extracting state", 
                         frame_shape=frame.shape)
        
        state = {}
        # ... extraction logic ...
        
        self.logger.debug("State extracted", 
                         state_keys=list(state.keys()))
        return state
```

## Next Steps

- Learn about [[Agent Development]] to create agents for your game
- Explore [[Computer Vision]] for advanced detection
- See [[Example Game Plugins]] for more examples
- Share your plugin in [[Community Plugins]]

---

<p align="center">
  <a href="https://github.com/neophrythe/nexus-ai/wiki/Agent-Development">Next: Agent Development →</a>
</p>