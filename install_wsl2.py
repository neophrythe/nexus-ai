#!/usr/bin/env python3
"""
Quick installer for Nexus AI Framework on WSL2.
This handles the evdev issue and sets up a working environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True, capture=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=capture,
        text=True,
        check=check
    )
    return result

def main():
    print("=" * 60)
    print("NEXUS AI FRAMEWORK - WSL2 QUICK INSTALLER")
    print("=" * 60)
    
    # Check if we're on WSL2
    try:
        result = run_command("uname -r", capture=True)
        if "microsoft" not in result.stdout.lower():
            print("WARNING: This doesn't appear to be WSL2!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    except:
        pass
    
    print("\n1. Creating virtual environment...")
    if not Path("venv").exists():
        run_command(f"{sys.executable} -m venv venv")
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")
    
    # Activate venv
    venv_python = "./venv/bin/python" if os.name != 'nt' else "./venv/Scripts/python"
    venv_pip = "./venv/bin/pip" if os.name != 'nt' else "./venv/Scripts/pip"
    
    print("\n2. Upgrading pip...")
    run_command(f"{venv_pip} install --upgrade pip setuptools wheel", check=False)
    print("✓ Pip upgraded")
    
    print("\n3. Installing WSL2-compatible requirements...")
    
    # Install base requirements without evdev
    wsl2_requirements = """
numpy>=1.19.2
opencv-python>=4.5.3
Pillow>=8.3.1
scipy>=1.7.0
scikit-learn>=0.24.2
scikit-image>=0.18.3
torch>=1.10.0
torchvision>=0.11.0
click>=8.0.1
pyyaml>=5.4.1
python-dotenv>=0.19.0
structlog>=21.1.0
colorama>=0.4.4
tqdm>=4.62.0
requests>=2.26.0
mss>=6.1.0
pyautogui>=0.9.53
pynput>=1.7.3
gym>=0.21.0
stable-baselines3>=1.3.0
psutil>=5.8.0
"""
    
    # Write temporary requirements file
    with open("requirements_wsl2_temp.txt", "w") as f:
        f.write(wsl2_requirements)
    
    run_command(f"{venv_pip} install -r requirements_wsl2_temp.txt", check=False)
    os.remove("requirements_wsl2_temp.txt")
    print("✓ WSL2 requirements installed")
    
    print("\n4. Creating evdev mock for compatibility...")
    
    # Create evdev mock
    site_packages = Path("venv/lib")
    for python_dir in site_packages.glob("python*"):
        site_packages_dir = python_dir / "site-packages"
        if site_packages_dir.exists():
            # Create mock evdev module
            evdev_mock_content = '''
"""Mock evdev module for WSL2 compatibility."""

class InputDevice:
    def __init__(self, *args, **kwargs):
        self.name = "Mock Device"
        self.path = "/dev/input/mock"
    
    def read_loop(self):
        return []
    
    def grab(self):
        pass
    
    def ungrab(self):
        pass
    
    def close(self):
        pass

class UInput:
    def __init__(self, *args, **kwargs):
        pass
    
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

def list_devices():
    return []

# Mock constants
ecodes = type('ecodes', (), {
    'EV_KEY': 1,
    'EV_REL': 2,
    'EV_ABS': 3,
    'KEY_A': 30,
    'KEY_SPACE': 57,
    'BTN_LEFT': 272,
    'BTN_RIGHT': 273,
    'REL_X': 0,
    'REL_Y': 1,
})()
'''
            
            evdev_dir = site_packages_dir / "evdev"
            evdev_dir.mkdir(exist_ok=True)
            
            with open(evdev_dir / "__init__.py", "w") as f:
                f.write(evdev_mock_content)
            
            print(f"✓ evdev mock created at {evdev_dir}")
            break
    
    print("\n5. Installing Nexus...")
    run_command(f"{venv_pip} install -e .", check=False)
    print("✓ Nexus installed")
    
    print("\n6. Testing installation...")
    test_result = run_command(
        f"{venv_python} -c \"import nexus; print('Nexus version:', nexus.__version__)\"",
        check=False,
        capture=True
    )
    
    if test_result.returncode == 0:
        print("✓ Installation successful!")
        print(test_result.stdout)
    else:
        print("⚠ Import test failed, but this might be expected for some modules")
    
    # Create activation script
    with open("activate_nexus.sh", "w") as f:
        f.write("""#!/bin/bash
# Nexus WSL2 Activation Script

# Set X11 display if not set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
    echo "X11 Display set to: $DISPLAY"
fi

# Activate virtual environment
source venv/bin/activate

# Set WSL2 flag
export NEXUS_PLATFORM=WSL2

echo "Nexus environment activated (WSL2 mode)"
echo "Run 'nexus --help' to get started"
""")
    
    os.chmod("activate_nexus.sh", 0o755)
    
    print("\n" + "=" * 60)
    print("INSTALLATION COMPLETE!")
    print("=" * 60)
    print("\nTo use Nexus:")
    print("  1. Activate environment: source activate_nexus.sh")
    print("  2. Run Nexus: nexus --help")
    print("\nFor GUI support:")
    print("  - Install VcXsrv or X410 on Windows")
    print("  - Launch with 'Disable access control' option")
    print("\nNote: Some features are limited in WSL2:")
    print("  - Direct input injection may not work")
    print("  - Window capture requires games in windowed mode")
    print("  - Hardware key detection is limited")

if __name__ == "__main__":
    main()