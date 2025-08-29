# Nexus AI Framework - WSL2 Setup Guide

## 🚀 Quick Start for WSL2

If you're running Nexus on WSL2 (Windows Subsystem for Linux 2), follow this guide for optimal setup.

### Prerequisites

1. **WSL2 installed** (Windows 10 version 2004+ or Windows 11)
2. **Python 3.10+** installed in WSL2
3. **X Server for GUI support** (optional, for GUI features)

### Automatic Installation

We provide an automated installer that handles WSL2-specific issues:

```bash
# Clone the repository
git clone https://github.com/neophrythe/nexus-ai.git
cd nexus-ai

# Run the WSL2 installer
python3 install_wsl2.py

# Activate the environment
source activate_nexus.sh

# Test the installation
nexus --help
```

### Manual Installation

If you prefer manual setup:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install with WSL2 detection (automatically excludes evdev)
pip install -e .

# 4. For full features (excluding problematic packages)
pip install -r requirements-wsl2.txt
```

## 🖥️ GUI Support Setup

For GUI features (visual debugger, etc.), you need an X server on Windows:

### Option 1: VcXsrv (Free)

1. Download [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
2. Install and launch with these settings:
   - Multiple windows
   - Start no client
   - **Disable access control** (important!)
3. In WSL2, set display:
   ```bash
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
   ```

### Option 2: X410 (Paid, from Microsoft Store)

1. Install X410 from Microsoft Store
2. Launch in "Windowed Apps" mode
3. WSL2 should auto-detect it

## 🎮 BlueStacks Integration

Nexus can control Android games via BlueStacks on WSL2:

```bash
# If BlueStacks is installed on Windows
# The installer will create an ADB wrapper automatically

# Test BlueStacks connection
nexus test bluestacks

# Launch Android game
nexus launch --platform=bluestacks --app=com.example.game
```

## ⚠️ WSL2 Limitations

Some features have limitations in WSL2:

| Feature | Status | Workaround |
|---------|--------|------------|
| Direct Input | ❌ Limited | Use pyautogui for basic input |
| evdev | ❌ Not supported | Mock module provided |
| Window Capture | ⚠️ Partial | Games must run in windowed mode |
| DXCam | ❌ Not supported | Uses MSS backend instead |
| Hardware Keys | ⚠️ Limited | Basic keyboard input only |
| GPU Acceleration | ✅ Supported | Requires WSL2 GPU drivers |

## 🔍 Platform Detection

Check your platform configuration:

```bash
# Run platform detection
python3 scripts/detect_platform.py

# Output will show:
# - WSL version (1 or 2)
# - X11 availability
# - CUDA support
# - Recommended requirements file
```

## 🛠️ Troubleshooting

### Issue: "No module named 'evdev'"

**Solution**: The installer creates a mock evdev module. If you still see errors:

```bash
# Reinstall with WSL2 mode
python3 install_wsl2.py
```

### Issue: GUI windows don't appear

**Solution**: Ensure X server is running on Windows and:

```bash
# Test X11 connection
sudo apt install x11-apps
xclock  # Should show a clock window

# If not working, check DISPLAY
echo $DISPLAY  # Should show IP:0.0
```

### Issue: Can't capture game windows

**Solution**: 
1. Run games in windowed mode (not fullscreen)
2. Use MSS backend:
   ```python
   from nexus.capture import CaptureManager
   capture = CaptureManager(backend="mss")
   ```

### Issue: Input commands don't work

**Solution**: WSL2 can't directly inject input to Windows apps. Options:
1. Use AutoHotkey on Windows side
2. Use BlueStacks for Android games
3. Run Linux-native games in WSL2

## 📊 Performance Tips

1. **Use WSL2 filesystem**: Store projects in `/home/` not `/mnt/c/`
2. **Enable GPU**: Install NVIDIA/AMD WSL2 GPU drivers
3. **Allocate RAM**: Edit `.wslconfig` in Windows home:
   ```ini
   [wsl2]
   memory=8GB
   processors=4
   ```

## 🔧 Advanced Configuration

Create `~/.nexus/wsl2_config.yaml`:

```yaml
platform: wsl2
capture:
  backend: mss
  fps_limit: 30
  
input:
  backend: pyautogui
  delay_ms: 50
  
display:
  x11_forward: true
  display_var: "auto"  # Auto-detect
  
features:
  gpu_acceleration: true
  mock_evdev: true
  
paths:
  windows_games: "/mnt/c/Program Files"
  steam_games: "/mnt/c/Program Files (x86)/Steam"
```

## 📚 Example Usage

```python
# Simple game automation on WSL2
from nexus import Game, CaptureManager
from nexus.input import InputController

# Initialize with WSL2-compatible settings
game = Game(
    name="MyGame",
    platform="wsl2",
    capture_backend="mss",
    input_backend="pyautogui"
)

# Capture game window
capture = game.get_capture_manager()
frame = capture.grab_frame()

# Control game (limited in WSL2)
input_ctrl = game.get_input_controller()
input_ctrl.move_mouse(500, 300)
input_ctrl.click()
```

## 🤝 Contributing

Help us improve WSL2 support! Report issues or contribute:

- [GitHub Issues](https://github.com/neophrythe/nexus-ai/issues)
- Tag with `wsl2` label
- Include output of `python3 scripts/detect_platform.py`

## 📞 Support

- **Discord**: [Join our community](https://discord.gg/nexus)
- **Wiki**: [WSL2 Documentation](https://github.com/neophrythe/nexus-ai/wiki/WSL2-Setup)

---

*Note: WSL2 support is continuously improving. Check for updates regularly!*