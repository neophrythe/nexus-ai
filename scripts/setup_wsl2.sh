#!/bin/bash
# Nexus AI Framework - WSL2 Setup Script
# This script sets up Nexus for WSL2 environments

set -e

echo "=================================================="
echo "  NEXUS AI FRAMEWORK - WSL2 SETUP"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "ℹ $1"
}

# Detect if running in WSL2
detect_wsl2() {
    if grep -qi microsoft /proc/version && [[ $(uname -r) =~ ^[5-9]\. ]]; then
        return 0
    else
        return 1
    fi
}

# Check if running in WSL2
if ! detect_wsl2; then
    print_error "This script is designed for WSL2 environments only."
    print_info "Detected environment: $(uname -r)"
    exit 1
fi

print_success "WSL2 environment detected"

# Update package lists
print_info "Updating package lists..."
sudo apt-get update -qq

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt-get install -y -qq \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libopencv-dev \
    tesseract-ocr \
    xvfb \
    x11-apps \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxinerama-dev \
    libxi-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxtst-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    > /dev/null 2>&1

print_success "System dependencies installed"

# Check for X11 forwarding
print_info "Checking X11 forwarding..."
if [ -z "$DISPLAY" ]; then
    print_warning "X11 display not configured"
    print_info "Setting up X11 forwarding for WSL2..."
    
    # Get Windows host IP
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
    echo "export DISPLAY=$DISPLAY" >> ~/.bashrc
    
    print_info "X11 forwarding configured. You'll need:"
    print_info "  1. Install VcXsrv or X410 on Windows"
    print_info "  2. Launch it with 'Disable access control' option"
    print_info "  3. Restart your WSL2 terminal"
else
    print_success "X11 display configured: $DISPLAY"
fi

# Create virtual environment
print_info "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
print_success "Pip upgraded"

# Install WSL2-specific requirements
print_info "Installing WSL2-optimized dependencies..."
if [ -f "requirements-wsl2.txt" ]; then
    pip install -r requirements-wsl2.txt -q
    print_success "WSL2 requirements installed"
else
    print_warning "requirements-wsl2.txt not found, using default requirements"
    pip install -r requirements.txt -q
fi

# Handle evdev issue for WSL2
print_info "Handling evdev compatibility..."
# Instead of installing evdev (which won't work), we'll create a mock
cat > venv/lib/python*/site-packages/evdev_mock.py << 'EOF'
"""Mock evdev module for WSL2 compatibility."""

class InputDevice:
    def __init__(self, *args, **kwargs):
        pass
    
    def read_loop(self):
        return []
    
    def grab(self):
        pass
    
    def ungrab(self):
        pass

class UInput:
    def __init__(self, *args, **kwargs):
        pass
    
    def write(self, *args):
        pass
    
    def syn(self):
        pass

def list_devices():
    return []

class InputEvent:
    def __init__(self, *args, **kwargs):
        self.type = 0
        self.code = 0
        self.value = 0
EOF

# Create evdev package redirect
mkdir -p venv/lib/python*/site-packages/evdev
echo "from evdev_mock import *" > venv/lib/python*/site-packages/evdev/__init__.py
print_success "evdev compatibility layer installed"

# Setup ADB for BlueStacks support (if Windows ADB is available)
print_info "Checking for Windows ADB access..."
if [ -f "/mnt/c/Windows/System32/adb.exe" ] || [ -f "/mnt/c/Program Files/BlueStacks_nxt/adb.exe" ]; then
    print_success "Windows ADB found"
    print_info "Creating ADB wrapper..."
    
    cat > ~/adb_wrapper.sh << 'EOF'
#!/bin/bash
# Wrapper to use Windows ADB from WSL2
if [ -f "/mnt/c/Program Files/BlueStacks_nxt/adb.exe" ]; then
    "/mnt/c/Program Files/BlueStacks_nxt/adb.exe" "$@"
elif [ -f "/mnt/c/Windows/System32/adb.exe" ]; then
    /mnt/c/Windows/System32/adb.exe "$@"
else
    echo "Windows ADB not found"
    exit 1
fi
EOF
    
    chmod +x ~/adb_wrapper.sh
    sudo ln -sf ~/adb_wrapper.sh /usr/local/bin/adb
    print_success "ADB wrapper created"
else
    print_warning "Windows ADB not found. BlueStacks support will be limited."
fi

# Install Nexus
print_info "Installing Nexus AI Framework..."
pip install -e . -q
print_success "Nexus installed"

# Create WSL2 configuration file
print_info "Creating WSL2 configuration..."
cat > nexus_wsl2_config.yaml << EOF
# Nexus AI Framework - WSL2 Configuration
platform: wsl2
capture:
  backend: mss  # DXcam doesn't work in WSL2
  fps_limit: 30  # Lower FPS for stability
  
input:
  backend: pyautogui  # Limited functionality in WSL2
  use_windows_api: false
  
display:
  x11_forwarding: true
  display_var: "$DISPLAY"
  
paths:
  windows_home: "/mnt/c/Users/$USER"
  games_directory: "/mnt/c/Program Files"
  
features:
  evdev_support: false
  direct_input: false
  cuda_available: $(nvidia-smi > /dev/null 2>&1 && echo "true" || echo "false")
  
warnings:
  - "Some input features are limited in WSL2"
  - "Direct game window capture may not work for all games"
  - "Consider running games in windowed mode for better compatibility"
EOF
print_success "WSL2 configuration created"

# Test installation
print_info "Testing Nexus installation..."
if python -c "import nexus; print('Nexus version:', nexus.__version__)" 2>/dev/null; then
    print_success "Nexus imported successfully"
else
    print_error "Failed to import Nexus"
fi

# Create convenience scripts
print_info "Creating convenience scripts..."

# Create nexus launcher for WSL2
cat > nexus_wsl2.sh << 'EOF'
#!/bin/bash
# Nexus launcher for WSL2

# Ensure X11 forwarding is set up
if [ -z "$DISPLAY" ]; then
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
fi

# Activate virtual environment
source venv/bin/activate

# Run Nexus with WSL2 config
export NEXUS_PLATFORM=wsl2
nexus "$@"
EOF
chmod +x nexus_wsl2.sh
print_success "Launcher script created"

# Print summary
echo ""
echo "=================================================="
echo "  SETUP COMPLETE!"
echo "=================================================="
print_success "Nexus AI Framework is ready for WSL2"
echo ""
echo "Important notes:"
echo "  1. X11 Server: Install VcXsrv or X410 on Windows"
echo "  2. Launch X server with 'Disable access control'"
echo "  3. Use ./nexus_wsl2.sh instead of nexus command"
echo "  4. Games should run in windowed mode for best compatibility"
echo "  5. Some input features are limited in WSL2"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To test Nexus:"
echo "  ./nexus_wsl2.sh --help"
echo ""
echo "For GUI applications, ensure X server is running on Windows!"
echo "=================================================="