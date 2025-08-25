"""Platform Detection and Terminal Utilities - SerpentAI Compatible

Essential utilities for cross-platform support and terminal operations.
"""

import os
import sys
import platform
import subprocess
import shutil
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger()


def is_linux() -> bool:
    """Check if running on Linux - SerpentAI compatible"""
    return sys.platform.startswith('linux')


def is_windows() -> bool:
    """Check if running on Windows - SerpentAI compatible"""
    return sys.platform == 'win32'


def is_macos() -> bool:
    """Check if running on macOS - SerpentAI compatible"""
    return sys.platform == 'darwin'


def get_platform() -> str:
    """Get platform name"""
    if is_windows():
        return "Windows"
    elif is_linux():
        return "Linux"
    elif is_macos():
        return "macOS"
    else:
        return platform.system()


def get_platform_details() -> dict:
    """Get detailed platform information"""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'is_64bit': sys.maxsize > 2**32
    }


def clear_terminal() -> None:
    """Clear terminal screen - SerpentAI compatible"""
    if is_windows():
        os.system('cls')
    else:
        os.system('clear')


def display_serpent_logo() -> None:
    """Display Serpent/Nexus logo - SerpentAI compatible"""
    logo = """
    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    
    Game AI Framework - Modern Successor to SerpentAI
    """
    print(logo)


def get_terminal_size() -> Tuple[int, int]:
    """Get terminal size (columns, rows)"""
    try:
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except:
        return 80, 24  # Default size


def is_admin() -> bool:
    """Check if running with admin/root privileges"""
    if is_windows():
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except:
            return False
    else:
        return os.geteuid() == 0


def check_dependency(command: str) -> bool:
    """Check if a system dependency is available"""
    return shutil.which(command) is not None


def get_gpu_info() -> dict:
    """Get GPU information"""
    info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': []
    }
    
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(info['gpu_count'])]
    except ImportError:
        pass
    
    # Try nvidia-smi for more details
    if check_dependency('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) == 2 and parts[0] not in info['gpu_names']:
                        info['gpu_names'].append(f"{parts[0]} ({parts[1]})")
        except:
            pass
    
    return info


def get_memory_info() -> dict:
    """Get system memory information"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent
        }
    except ImportError:
        return {}


def ensure_admin() -> None:
    """Ensure running with admin privileges, restart if needed"""
    if not is_admin():
        if is_windows():
            # Restart with admin privileges
            import ctypes
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            sys.exit(0)
        else:
            print("This operation requires root privileges.")
            print(f"Please run: sudo {' '.join(sys.argv)}")
            sys.exit(1)


def get_home_directory() -> str:
    """Get user home directory"""
    return os.path.expanduser("~")


def get_config_directory() -> str:
    """Get platform-specific config directory"""
    if is_windows():
        return os.path.join(os.environ.get('APPDATA', ''), 'Nexus')
    elif is_macos():
        return os.path.expanduser("~/Library/Application Support/Nexus")
    else:
        return os.path.expanduser("~/.config/nexus")


def get_data_directory() -> str:
    """Get platform-specific data directory"""
    if is_windows():
        return os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Nexus')
    elif is_macos():
        return os.path.expanduser("~/Library/Nexus")
    else:
        return os.path.expanduser("~/.local/share/nexus")


def setup_directories() -> None:
    """Setup required directories"""
    directories = [
        get_config_directory(),
        get_data_directory(),
        os.path.join(get_data_directory(), 'plugins'),
        os.path.join(get_data_directory(), 'datasets'),
        os.path.join(get_data_directory(), 'models'),
        os.path.join(get_data_directory(), 'logs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logger.info(f"Directories setup complete")


def check_requirements() -> dict:
    """Check system requirements"""
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'os_supported': is_windows() or is_linux() or is_macos(),
        'dependencies': {}
    }
    
    # Check Python packages
    required_packages = [
        'numpy', 'opencv-python', 'pillow', 'mss', 
        'pyautogui', 'structlog', 'pyyaml'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            requirements['dependencies'][package] = True
        except ImportError:
            requirements['dependencies'][package] = False
    
    # Check system dependencies
    if is_linux():
        requirements['xdotool'] = check_dependency('xdotool')
        requirements['xvfb'] = check_dependency('Xvfb')
    
    return requirements


# Platform-specific imports and functions
if is_windows():
    def get_window_handles():
        """Get all window handles on Windows"""
        import win32gui
        handles = []
        
        def callback(hwnd, handles):
            if win32gui.IsWindowVisible(hwnd):
                handles.append(hwnd)
            return True
        
        win32gui.EnumWindows(callback, handles)
        return handles
        
elif is_linux():
    def get_window_ids():
        """Get all window IDs on Linux"""
        try:
            result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', ''],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except:
            pass
        return []