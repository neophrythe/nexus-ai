#!/usr/bin/env python3
"""
Universal Installer for Nexus Game AI Framework
Works on Windows, Linux, macOS, WSL1, and WSL2
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path


class NexusInstaller:
    """Universal installer for Nexus Game AI Framework"""
    
    def __init__(self):
        self.platform = self.detect_platform()
        self.python = sys.executable
        self.has_gpu = self.detect_gpu()
        self.install_dir = Path.cwd()
        
    def detect_platform(self):
        """Detect the current platform"""
        if sys.platform == "win32":
            return "windows"
        elif sys.platform == "darwin":
            print("⚠️ macOS is not supported (no CUDA support)")
            print("Please use Windows, Linux, or WSL2 for GPU-accelerated AI training")
            sys.exit(1)
        else:
            # Check for WSL
            try:
                with open("/proc/version", "r") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info:
                        # Detect WSL version
                        try:
                            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
                            kernel = result.stdout.strip().lower()
                            if 'wsl2' in kernel or kernel.startswith('5.'):
                                return "wsl2"
                            else:
                                return "wsl1"
                        except:
                            return "wsl2"  # Default to WSL2
            except:
                pass
            return "linux"
    
    def detect_gpu(self):
        """Detect if NVIDIA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def run_command(self, cmd, check=True, shell=True):
        """Run a command and handle errors"""
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, check=check)
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            if e.stderr:
                print(e.stderr)
            if check:
                raise
            return e
    
    def create_venv(self):
        """Create virtual environment"""
        print("\n📦 Creating virtual environment...")
        venv_path = self.install_dir / "venv"
        
        if venv_path.exists():
            print("Virtual environment already exists.")
            response = input("Do you want to recreate it? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(venv_path)
            else:
                return
        
        self.run_command(f"{self.python} -m venv venv")
        
        # Update pip
        if self.platform == "windows":
            pip = "venv\\Scripts\\pip"
        else:
            pip = "venv/bin/pip"
        
        self.run_command(f"{pip} install --upgrade pip setuptools wheel")
        print("✅ Virtual environment created")
    
    def install_system_deps(self):
        """Install system dependencies based on platform"""
        print(f"\n🔧 Installing system dependencies for {self.platform}...")
        
        if self.platform == "wsl2":
            print("Installing WSL2 dependencies...")
            deps_script = """#!/bin/bash
# WSL2 System Dependencies

echo "Installing system packages for GUI support..."
echo "Note: You'll need to enter your password for sudo"

# Update package list
sudo apt-get update

# Core dependencies
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl

# GUI dependencies for Qt
sudo apt-get install -y \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-randr0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libx11-xcb1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrender1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    libfontconfig1 \
    libdbus-1-3 \
    libegl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    mesa-utils \
    x11-apps \
    xauth

# OpenCV dependencies
sudo apt-get install -y \
    libopencv-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0

# OCR dependencies (optional)
sudo apt-get install -y tesseract-ocr || true

echo "✅ System dependencies installed"
"""
            
            # Write and execute script
            script_path = self.install_dir / "install_deps_wsl2.sh"
            script_path.write_text(deps_script)
            script_path.chmod(0o755)
            self.run_command(f"bash {script_path}", check=False)
            
        elif self.platform == "linux":
            print("Installing Linux dependencies...")
            # Similar to WSL2 but without X11 forwarding setup
            self.run_command("sudo apt-get update", check=False)
            self.run_command("sudo apt-get install -y python3-dev build-essential libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0", check=False)
                
        elif self.platform == "windows":
            print("Windows detected. No system dependencies needed.")
            print("Make sure you have Visual C++ Redistributables installed.")
    
    def install_python_deps(self):
        """Install Python dependencies"""
        print("\n📚 Installing Python dependencies...")
        
        if self.platform == "windows":
            pip = "venv\\Scripts\\pip"
        else:
            pip = "venv/bin/pip"
        
        # Core requirements
        core_deps = [
            "numpy>=1.24.0",
            "opencv-python>=4.8.0",
            "Pillow>=10.0.0",
            "click>=8.1.0",
            "structlog>=23.2.0",
            "pyyaml>=6.0",
            "toml>=0.10.2",
            "psutil>=5.9.0",
            "watchdog>=3.0.0",
            "tqdm>=4.62.0",
            "colorama>=0.4.4",
            "python-dotenv>=0.19.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "websockets>=12.0",
            "httpx>=0.25.0",
            "aiofiles>=23.2.0",
            "requests>=2.26.0",
            "packaging>=20.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "GitPython>=3.1.0",
            "tabulate>=0.9.0",
            "questionary>=1.10.0",
            "rich>=13.6.0",
            "h5py>=3.7.0",
            "GPUtil>=1.4.0",
        ]
        
        print("Installing core dependencies...")
        for dep in core_deps:
            self.run_command(f"{pip} install {dep}", check=False)
        
        # GUI dependencies
        print("Installing GUI dependencies...")
        self.run_command(f"{pip} install PyQt5>=5.15.0 pyqtgraph>=0.13.0", check=False)
        
        # Platform-specific
        if self.platform != "wsl2":
            # Install evdev only on native Linux
            if self.platform == "linux":
                self.run_command(f"{pip} install evdev>=1.6.0", check=False)
            elif self.platform == "windows":
                self.run_command(f"{pip} install pywin32>=306", check=False)
        else:
            # WSL2 specific
            self.run_command(f"{pip} install mss>=9.0.0 pyautogui>=0.9.54 pynput>=1.7.0", check=False)
        
        # Machine Learning packages
        print("Installing ML dependencies...")
        if self.has_gpu:
            print("Installing GPU-accelerated PyTorch...")
            self.run_command(f"{pip} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", check=False)
        else:
            print("Installing CPU-only PyTorch...")
            self.run_command(f"{pip} install torch torchvision torchaudio", check=False)
        
        # Additional ML packages
        self.run_command(f"{pip} install gymnasium>=0.29.0 gym>=0.21.0 stable-baselines3>=2.1.0 scikit-learn>=1.3.0 scipy>=1.11.0", check=False)
        
        # Optional packages (don't fail if they don't install)
        optional = [
            "pytesseract>=0.3.10",
            "easyocr>=1.7.0",
            "wandb>=0.13.0",
            "mlflow>=2.0.0",
            "tensorboard>=2.14.0",
        ]
        
        print("Installing optional dependencies...")
        for dep in optional:
            self.run_command(f"{pip} install {dep}", check=False)
        
        # Install package in development mode
        print("Installing Nexus in development mode...")
        self.run_command(f"{pip} install -e .", check=False)
        
        print("✅ Python dependencies installed")
    
    def setup_wsl2_gui(self):
        """Special setup for WSL2 GUI"""
        if self.platform != "wsl2":
            return
        
        print("\n🖥️ Setting up GUI for WSL2...")
        
        # Create WSL2 GUI helper script
        gui_script = """#!/bin/bash
# WSL2 GUI Setup Helper

# Detect WSL2 host IP
export WSL_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
export DISPLAY=$WSL_HOST_IP:0

# Qt Configuration
export QT_XCB_GL_INTEGRATION=none
export QT_QUICK_BACKEND=software
export XDG_RUNTIME_DIR=/tmp/runtime-$USER
mkdir -p $XDG_RUNTIME_DIR 2>/dev/null
chmod 700 $XDG_RUNTIME_DIR 2>/dev/null

echo "WSL2 GUI Environment Set:"
echo "  DISPLAY=$DISPLAY"
echo "  WSL_HOST_IP=$WSL_HOST_IP"

# Test X11 connection
if command -v xeyes >/dev/null 2>&1; then
    timeout 1 xeyes 2>/dev/null && echo "✅ X11 connection working!" || echo "⚠️ X11 connection failed. Is X server running on Windows?"
fi
"""
        
        script_path = self.install_dir / "setup_wsl2_display.sh"
        script_path.write_text(gui_script)
        script_path.chmod(0o755)
        
        # Create mock evdev for WSL2
        print("Creating evdev compatibility layer for WSL2...")
        evdev_mock = '''"""Mock evdev module for WSL2 compatibility."""

class InputDevice:
    def __init__(self, *args, **kwargs):
        self.name = "Mock Device"
        self.path = "/dev/input/mock"
        self.phys = "mock/device"
        self.info = type('info', (), {'bustype': 0, 'vendor': 0, 'product': 0, 'version': 0})()
        self.capabilities = {}
    
    def read_loop(self):
        return []
    
    def grab(self):
        pass
    
    def ungrab(self):
        pass
    
    def close(self):
        pass
    
    @property
    def fd(self):
        return -1

class UInput:
    def __init__(self, *args, **kwargs):
        self.device = InputDevice()
    
    def write(self, *args):
        pass
    
    def syn(self):
        pass
    
    def close(self):
        pass

class InputEvent:
    def __init__(self, sec=0, usec=0, type=0, code=0, value=0):
        self.sec = sec
        self.usec = usec
        self.type = type
        self.code = code
        self.value = value
        self.timestamp = sec + usec / 1000000

def list_devices():
    return []

def categorize(events):
    return {}

# Mock constants
class ecodes:
    EV_KEY = 1
    EV_REL = 2
    EV_ABS = 3
    EV_SYN = 0
    KEY_A = 30
    KEY_SPACE = 57
    BTN_LEFT = 272
    BTN_RIGHT = 273
    REL_X = 0
    REL_Y = 1
    ABS_X = 0
    ABS_Y = 1

# Make ecodes accessible directly
EV_KEY = ecodes.EV_KEY
EV_REL = ecodes.EV_REL
EV_ABS = ecodes.EV_ABS
EV_SYN = ecodes.EV_SYN
'''
        
        # Find site-packages directory
        venv_path = self.install_dir / "venv"
        for python_dir in (venv_path / "lib").glob("python*"):
            site_packages = python_dir / "site-packages"
            if site_packages.exists():
                evdev_file = site_packages / "evdev.py"
                evdev_file.write_text(evdev_mock)
                print(f"✅ Created evdev mock at {evdev_file}")
                break
        
        print("\n" + "="*60)
        print("WSL2 GUI SETUP INSTRUCTIONS:")
        print("="*60)
        print("\n1. Install X Server on Windows (if not already installed):")
        print("   Option A: VcXsrv (Free)")
        print("   - Download: https://sourceforge.net/projects/vcxsrv/")
        print("   - Run XLaunch with:")
        print("     • Multiple windows")
        print("     • Start no client")
        print("     • ✅ Disable access control")
        print("\n   Option B: X410 (Paid, from Microsoft Store)")
        print("\n2. Run this before starting GUI:")
        print("   source setup_wsl2_display.sh")
        print("\n3. Test with:")
        print("   nexus gui")
        print("="*60)
    
    def test_installation(self):
        """Test the installation"""
        print("\n🧪 Testing installation...")
        
        if self.platform == "windows":
            python = "venv\\Scripts\\python"
        else:
            python = "venv/bin/python"
        
        # Test import
        result = self.run_command(
            f'{python} -c "import nexus; print(f\'Nexus version: {{nexus.__version__}}\')"',
            check=False
        )
        
        if result.returncode == 0:
            print("✅ Nexus imported successfully")
        else:
            print("⚠️ Failed to import Nexus")
        
        # Test CLI
        if self.platform == "windows":
            nexus = "venv\\Scripts\\nexus"
        else:
            nexus = "venv/bin/nexus"
        
        result = self.run_command(f"{nexus} --version", check=False)
        
        if result.returncode == 0:
            print("✅ Nexus CLI working")
        else:
            print("⚠️ Nexus CLI not working")
    
    def run(self):
        """Run the complete installation"""
        print("="*60)
        print("NEXUS GAME AI FRAMEWORK - UNIVERSAL INSTALLER")
        print("="*60)
        print(f"Platform: {self.platform}")
        print(f"Python: {sys.version}")
        print(f"GPU: {'Detected' if self.has_gpu else 'Not detected'}")
        print(f"Directory: {self.install_dir}")
        print("="*60)
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            sys.exit(1)
        
        try:
            # Create virtual environment
            self.create_venv()
            
            # Install system dependencies
            self.install_system_deps()
            
            # Install Python dependencies
            self.install_python_deps()
            
            # WSL2 specific GUI setup
            if self.platform == "wsl2":
                self.setup_wsl2_gui()
            
            # Test installation
            self.test_installation()
            
            print("\n" + "="*60)
            print("✅ INSTALLATION COMPLETE!")
            print("="*60)
            print("\nTo activate the environment:")
            if self.platform == "windows":
                print("  .\\venv\\Scripts\\activate")
            else:
                print("  source venv/bin/activate")
            
            if self.platform == "wsl2":
                print("\nFor GUI support:")
                print("  source setup_wsl2_display.sh")
            
            print("\nTo test Nexus:")
            print("  nexus --help")
            print("  nexus doctor")
            print("  nexus gui")
            
        except Exception as e:
            print(f"\n❌ Installation failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    installer = NexusInstaller()
    installer.run()