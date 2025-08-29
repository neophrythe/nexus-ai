#!/usr/bin/env python3
"""
Platform detection utility for Nexus AI Framework.
Detects the current platform and environment (Windows, Linux, WSL2, etc.)
"""

import platform
import subprocess
import os
import sys
from pathlib import Path


def is_wsl() -> bool:
    """Check if running on Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    
    # Check for WSL-specific indicators
    try:
        # Method 1: Check kernel version
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return True
    except:
        pass
    
    # Method 2: Check for WSL environment variable
    if os.environ.get('WSL_DISTRO_NAME'):
        return True
    
    # Method 3: Check for WSL interop
    if Path('/proc/sys/fs/binfmt_misc/WSLInterop').exists():
        return True
    
    return False


def is_wsl2() -> bool:
    """Check if running on WSL2 specifically."""
    if not is_wsl():
        return False
    
    try:
        # WSL2 uses a real Linux kernel (5.x or higher)
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        kernel_version = result.stdout.strip()
        if 'microsoft' in kernel_version.lower():
            # Extract major version
            version_parts = kernel_version.split('.')
            if len(version_parts) >= 1:
                major_version = int(version_parts[0])
                return major_version >= 5
    except:
        pass
    
    return False


def get_platform_info():
    """Get detailed platform information."""
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'is_wsl': is_wsl(),
        'is_wsl2': is_wsl2(),
        'is_windows': platform.system() == 'Windows',
        'is_linux': platform.system() == 'Linux',
        'is_native_linux': platform.system() == 'Linux' and not is_wsl(),
        'has_display': bool(os.environ.get('DISPLAY')),
        'has_cuda': False,
        'cuda_version': None
    }
    
    # Check for CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            info['has_cuda'] = True
            # Try to get CUDA version
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                output = result.stdout
                if 'release' in output:
                    # Extract version from output
                    lines = output.split('\n')
                    for line in lines:
                        if 'release' in line:
                            parts = line.split('release')[-1].strip().split(',')[0]
                            info['cuda_version'] = parts
                            break
    except:
        pass
    
    # WSL-specific info
    if info['is_wsl']:
        info['wsl_distro'] = os.environ.get('WSL_DISTRO_NAME', 'unknown')
        # Check if we can access Windows filesystem
        info['has_windows_access'] = Path('/mnt/c').exists()
        # Check if X11 forwarding is available
        info['has_x11'] = bool(os.environ.get('DISPLAY'))
    
    return info


def get_requirements_file():
    """Determine the appropriate requirements file based on platform."""
    info = get_platform_info()
    
    if info['is_wsl2']:
        return 'requirements-wsl2.txt'
    elif info['is_windows']:
        return 'requirements-windows.txt'
    elif info['is_native_linux']:
        return 'requirements-linux.txt'
    else:
        return 'requirements.txt'


def print_platform_info():
    """Print platform information in a readable format."""
    info = get_platform_info()
    
    print("=" * 60)
    print("NEXUS AI FRAMEWORK - PLATFORM DETECTION")
    print("=" * 60)
    print(f"System:           {info['system']}")
    print(f"Release:          {info['release']}")
    print(f"Machine:          {info['machine']}")
    print(f"Python Version:   {info['python_version']}")
    print("-" * 60)
    
    if info['is_wsl2']:
        print("Environment:      WSL2 (Windows Subsystem for Linux 2)")
        print(f"WSL Distro:       {info.get('wsl_distro', 'unknown')}")
        print(f"Windows Access:   {'Yes' if info.get('has_windows_access') else 'No'}")
        print(f"X11 Display:      {'Available' if info.get('has_x11') else 'Not configured'}")
    elif info['is_wsl']:
        print("Environment:      WSL1 (Windows Subsystem for Linux)")
    elif info['is_windows']:
        print("Environment:      Native Windows")
    elif info['is_native_linux']:
        print("Environment:      Native Linux")
    else:
        print("Environment:      Unknown")
    
    print("-" * 60)
    print(f"CUDA Available:   {'Yes' if info['has_cuda'] else 'No'}")
    if info['cuda_version']:
        print(f"CUDA Version:     {info['cuda_version']}")
    
    print("-" * 60)
    print(f"Recommended Requirements: {get_requirements_file()}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        import json
        print(json.dumps(get_platform_info(), indent=2))
    else:
        print_platform_info()