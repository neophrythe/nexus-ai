#!/usr/bin/env python3
"""
Nexus AI Framework - Installation Verification Script
Checks if all components are properly installed and working.
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class NexusVerifier:
    """Verify Nexus AI Framework installation."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []
        self.root_dir = Path(__file__).parent.parent
        
    def print_header(self, text: str):
        """Print a section header."""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}{text:^60}{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        
    def print_success(self, text: str):
        """Print success message."""
        print(f"{GREEN}✓{RESET} {text}")
        self.successes.append(text)
        
    def print_error(self, text: str):
        """Print error message."""
        print(f"{RED}✗{RESET} {text}")
        self.errors.append(text)
        
    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{YELLOW}⚠{RESET} {text}")
        self.warnings.append(text)
        
    def check_python_version(self) -> bool:
        """Check Python version."""
        self.print_header("Python Version Check")
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            self.print_success(f"Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.print_error(f"Python {version.major}.{version.minor} (require >= 3.9)")
            return False
    
    def check_core_dependencies(self) -> bool:
        """Check core Python dependencies."""
        self.print_header("Core Dependencies Check")
        
        core_deps = [
            'numpy', 'cv2', 'PIL', 'yaml', 'toml', 'fastapi', 
            'structlog', 'click', 'pydantic', 'psutil'
        ]
        
        all_ok = True
        for dep in core_deps:
            try:
                if dep == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif dep == 'PIL':
                    from PIL import Image
                    version = Image.__version__
                elif dep == 'yaml':
                    import yaml
                    version = yaml.__version__ if hasattr(yaml, '__version__') else 'installed'
                else:
                    module = importlib.import_module(dep)
                    version = getattr(module, '__version__', 'installed')
                self.print_success(f"{dep}: {version}")
            except ImportError as e:
                self.print_error(f"{dep}: Not installed - {str(e)}")
                all_ok = False
        
        return all_ok
    
    def check_nexus_modules(self) -> bool:
        """Check Nexus modules."""
        self.print_header("Nexus Modules Check")
        
        modules = [
            'nexus.core',
            'nexus.api',
            'nexus.agents',
            'nexus.sprites',
            'nexus.input',
            'nexus.window',
            'nexus.ocr',
            'nexus.analytics',
            'nexus.events',
            'nexus.datasets',
            'nexus.emulators',
            'nexus.gui',
            'nexus.cli'
        ]
        
        all_ok = True
        for module in modules:
            try:
                importlib.import_module(module)
                self.print_success(f"{module}")
            except ImportError as e:
                self.print_error(f"{module}: {str(e)}")
                all_ok = False
        
        return all_ok
    
    def check_plugins(self) -> bool:
        """Check installed plugins."""
        self.print_header("Plugins Check")
        
        plugins_dir = self.root_dir / 'plugins'
        if not plugins_dir.exists():
            self.print_warning("Plugins directory not found")
            return False
        
        plugins = list(plugins_dir.glob('*/plugin.yaml'))
        if not plugins:
            self.print_warning("No plugins found")
            return True
        
        for plugin_file in plugins:
            plugin_name = plugin_file.parent.name
            try:
                # Check if plugin has required files
                required_files = ['__init__.py', 'plugin.yaml']
                all_present = all(
                    (plugin_file.parent / f).exists() 
                    for f in required_files
                )
                if all_present:
                    self.print_success(f"Plugin: {plugin_name}")
                else:
                    self.print_warning(f"Plugin {plugin_name}: Missing files")
            except Exception as e:
                self.print_error(f"Plugin {plugin_name}: {str(e)}")
        
        return True
    
    def check_cli_commands(self) -> bool:
        """Check CLI commands."""
        self.print_header("CLI Commands Check")
        
        commands = [
            'nexus --version',
            'nexus --help',
            'nexus list games',
            'nexus list agents',
            'nexus list plugins'
        ]
        
        all_ok = True
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd.split(), 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.print_success(f"Command: {cmd}")
                else:
                    self.print_error(f"Command failed: {cmd}")
                    all_ok = False
            except FileNotFoundError:
                self.print_error(f"Nexus CLI not found: {cmd}")
                all_ok = False
            except subprocess.TimeoutExpired:
                self.print_warning(f"Command timeout: {cmd}")
            except Exception as e:
                self.print_error(f"Command error: {cmd} - {str(e)}")
                all_ok = False
        
        return all_ok
    
    def check_file_structure(self) -> bool:
        """Check project file structure."""
        self.print_header("File Structure Check")
        
        required_dirs = [
            'nexus',
            'nexus/core',
            'nexus/api',
            'nexus/agents',
            'nexus/sprites',
            'nexus/input',
            'nexus/window',
            'nexus/ocr',
            'nexus/analytics',
            'nexus/events',
            'nexus/datasets',
            'nexus/emulators',
            'nexus/gui',
            'nexus/cli',
            'plugins',
            'scripts',
            'configs'
        ]
        
        required_files = [
            'setup.py',
            'requirements.txt',
            'README.md',
            'Dockerfile',
            'docker-compose.yml'
        ]
        
        all_ok = True
        
        # Check directories
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                self.print_success(f"Directory: {dir_path}")
            else:
                self.print_error(f"Missing directory: {dir_path}")
                all_ok = False
        
        # Check files
        for file_path in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists() and full_path.is_file():
                self.print_success(f"File: {file_path}")
            else:
                self.print_error(f"Missing file: {file_path}")
                all_ok = False
        
        return all_ok
    
    def check_optional_features(self) -> bool:
        """Check optional features."""
        self.print_header("Optional Features Check")
        
        # GPU support
        try:
            import torch
            if torch.cuda.is_available():
                self.print_success(f"GPU support: CUDA {torch.version.cuda}")
            else:
                self.print_warning("GPU support: No CUDA devices found")
        except ImportError:
            self.print_warning("GPU support: PyTorch not installed")
        
        # GUI support
        try:
            from PyQt5 import QtCore
            self.print_success(f"GUI support: PyQt5 {QtCore.PYQT_VERSION_STR}")
        except ImportError:
            self.print_warning("GUI support: PyQt5 not installed")
        
        # OCR engines
        try:
            import easyocr
            self.print_success("OCR: EasyOCR installed")
        except ImportError:
            self.print_warning("OCR: EasyOCR not installed")
        
        # Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            r.ping()
            self.print_success("Redis: Connected")
        except:
            self.print_warning("Redis: Not available")
        
        # Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True)
            if result.returncode == 0:
                self.print_success(f"Docker: {result.stdout.decode().strip()}")
        except:
            self.print_warning("Docker: Not installed")
        
        return True
    
    def check_code_quality(self) -> bool:
        """Check code quality metrics."""
        self.print_header("Code Quality Check")
        
        # Count lines of code
        py_files = list(Path(self.root_dir / 'nexus').rglob('*.py'))
        total_lines = 0
        for file in py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        self.print_success(f"Total Python files: {len(py_files)}")
        self.print_success(f"Total lines of code: {total_lines:,}")
        
        # Check for common issues
        issues = {
            'placeholder': 0,
            'TODO': 0,
            'FIXME': 0,
            'XXX': 0,
            'pass': 0
        }
        
        for file in py_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in issues:
                        issues[pattern] += content.count(pattern)
            except:
                pass
        
        # Report findings
        if issues['placeholder'] > 0:
            self.print_warning(f"Placeholder code found: {issues['placeholder']} occurrences")
        else:
            self.print_success("No placeholder code found")
        
        if issues['TODO'] + issues['FIXME'] + issues['XXX'] > 0:
            total_todos = issues['TODO'] + issues['FIXME'] + issues['XXX']
            self.print_warning(f"TODOs found: {total_todos} items")
        else:
            self.print_success("No TODOs found")
        
        return True
    
    def generate_report(self):
        """Generate final verification report."""
        self.print_header("Verification Report")
        
        total_checks = len(self.successes) + len(self.errors) + len(self.warnings)
        
        print(f"\n{BOLD}Summary:{RESET}")
        print(f"  {GREEN}Passed:{RESET} {len(self.successes)}/{total_checks}")
        print(f"  {YELLOW}Warnings:{RESET} {len(self.warnings)}/{total_checks}")
        print(f"  {RED}Errors:{RESET} {len(self.errors)}/{total_checks}")
        
        if self.errors:
            print(f"\n{BOLD}{RED}Critical Issues:{RESET}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  • {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            print(f"\n{BOLD}{YELLOW}Warnings:{RESET}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  • {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        # Overall status
        print(f"\n{BOLD}Overall Status:{RESET}")
        if not self.errors:
            print(f"{GREEN}{BOLD}✓ Nexus AI Framework is 100% ready!{RESET}")
            print(f"{GREEN}All critical components are working correctly.{RESET}")
        elif len(self.errors) <= 3:
            print(f"{YELLOW}{BOLD}⚠ Nexus AI Framework is mostly ready{RESET}")
            print(f"{YELLOW}Minor issues detected. Please fix the errors above.{RESET}")
        else:
            print(f"{RED}{BOLD}✗ Nexus AI Framework needs attention{RESET}")
            print(f"{RED}Multiple issues detected. Please install missing dependencies.{RESET}")
        
        # Next steps
        print(f"\n{BOLD}Next Steps:{RESET}")
        if self.errors:
            print("1. Install missing dependencies: pip install -r requirements.txt")
            print("2. Fix any import errors in the modules")
            print("3. Run this verification again")
        else:
            print("1. Start the API: nexus api")
            print("2. Launch the GUI: nexus gui")
            print("3. Or use Docker: docker-compose up")
    
    def run(self):
        """Run all verification checks."""
        print(f"{BOLD}{BLUE}")
        print("=" * 60)
        print("NEXUS AI FRAMEWORK - INSTALLATION VERIFICATION")
        print("=" * 60)
        print(f"{RESET}")
        
        # Run all checks
        self.check_python_version()
        self.check_core_dependencies()
        self.check_nexus_modules()
        self.check_plugins()
        self.check_cli_commands()
        self.check_file_structure()
        self.check_optional_features()
        self.check_code_quality()
        
        # Generate report
        self.generate_report()
        
        # Return exit code
        return 0 if not self.errors else 1


if __name__ == "__main__":
    verifier = NexusVerifier()
    sys.exit(verifier.run())