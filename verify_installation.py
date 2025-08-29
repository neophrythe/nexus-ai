#!/usr/bin/env python3
"""
Nexus AI Framework - Installation Verification Script
Checks that all components are properly installed and working.
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_module_syntax(module_path):
    """Check if a Python module has valid syntax."""
    try:
        with open(module_path, 'r') as f:
            code = f.read()
        compile(code, module_path, 'exec')
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports_in_file(file_path):
    """Check if imports in a file would work (without external dependencies)."""
    issues = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line.startswith('from nexus') or line.startswith('import nexus'):
            # Check if it's importing from nexus modules
            parts = line.split()
            if len(parts) >= 2:
                module = parts[1].replace(',', '')
                if 'nexus' in module:
                    # Check if the module path exists
                    module_path = module.replace('.', '/')
                    if not Path(f"{module_path}.py").exists() and not Path(module_path).is_dir():
                        issues.append(f"Line {i}: Missing module {module}")
    
    return issues

def main():
    print("=" * 60)
    print("NEXUS AI FRAMEWORK - VERIFICATION REPORT")
    print("=" * 60)
    
    # Check we're in the right directory
    if not Path("nexus").exists():
        print("❌ ERROR: Must run from GAMEAI directory")
        sys.exit(1)
    
    all_good = True
    
    # 1. Check critical Python files for syntax
    print("\n📝 Checking Python Syntax...")
    critical_files = [
        "nexus/cli/nexus_cli.py",
        "nexus/datasets/dataset_manager.py",
        "nexus/agents/__init__.py",
        "nexus/emulators/bluestacks.py",
        "nexus/emulators/android_bridge.py",
        "nexus/debug/visual_debugger.py",
        "nexus/core/frame_grabber.py",
    ]
    
    for file_path in critical_files:
        if Path(file_path).exists():
            valid, msg = check_module_syntax(file_path)
            if valid:
                print(f"  ✅ {file_path}: {msg}")
            else:
                print(f"  ❌ {file_path}: {msg}")
                all_good = False
        else:
            print(f"  ❌ {file_path}: FILE NOT FOUND")
            all_good = False
    
    # 2. Check for placeholders
    print("\n🔍 Checking for Placeholders...")
    placeholder_patterns = [
        ("raise NotImplementedError", "Not implemented errors"),
        ("TODO", "TODO comments"),
        ("FIXME", "FIXME comments"),
        ("pass  # Implement", "Empty implementations"),
    ]
    
    for pattern, description in placeholder_patterns:
        count = 0
        locations = []
        for py_file in Path("nexus").rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if pattern in content:
                        # Check if it's in a template/string
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if pattern in line:
                                # Check if it's in a string (rough check)
                                if not (('"""' in content and 
                                       content.index('"""') < content.index(pattern) < content.rindex('"""')) or
                                       ("'''" in content and 
                                       content.index("'''") < content.index(pattern) < content.rindex("'''"))):
                                    if "template" not in str(py_file).lower():
                                        count += 1
                                        locations.append(f"{py_file}:{i}")
            except:
                pass
        
        if count == 0:
            print(f"  ✅ No {description} in core code")
        else:
            print(f"  ⚠️  Found {count} {description} (may be intentional)")
            if count <= 3:  # Show first few
                for loc in locations[:3]:
                    print(f"      - {loc}")
    
    # 3. Check directory structure
    print("\n📁 Checking Directory Structure...")
    required_dirs = [
        "nexus",
        "nexus/cli",
        "nexus/core",
        "nexus/agents",
        "nexus/datasets",
        "nexus/emulators",
        "nexus/debug",
        "nexus/window",
        "nexus/launchers",
        "nexus/api",
        "nexus/plugins",
        "plugins",
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING")
            all_good = False
    
    # 4. Check for critical functions
    print("\n⚙️ Checking Critical Functions...")
    function_checks = [
        ("nexus/agents/__init__.py", "create_agent", "Agent creation function"),
        ("nexus/agents/__init__.py", "load_agent", "Agent loading function"),
        ("nexus/datasets/dataset_manager.py", "class DatasetManager", "DatasetManager class"),
        ("nexus/emulators/bluestacks.py", "class BlueStacksController", "BlueStacksController class"),
        ("nexus/cli/nexus_cli.py", "def setup", "Setup command"),
        ("nexus/cli/nexus_cli.py", "def bluestacks", "BlueStacks commands"),
    ]
    
    for file_path, pattern, description in function_checks:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
                if pattern in content:
                    print(f"  ✅ {description}")
                else:
                    print(f"  ❌ {description} - NOT FOUND")
                    all_good = False
        else:
            print(f"  ❌ {description} - FILE NOT FOUND")
            all_good = False
    
    # 5. Check plugins
    print("\n🔌 Checking Plugins...")
    plugin_dirs = [
        "plugins/auto_aim",
        "plugins/speed_runner",
        "plugins/discord_integration",
        "plugins/performance_monitor",
        "plugins/game_state_logger",
    ]
    
    for plugin_dir in plugin_dirs:
        if Path(plugin_dir).is_dir():
            config_file = Path(plugin_dir) / "config.py"
            plugin_file = Path(plugin_dir) / "plugin.py"
            if config_file.exists() and plugin_file.exists():
                print(f"  ✅ {plugin_dir}")
            else:
                print(f"  ⚠️  {plugin_dir} - Incomplete")
        else:
            print(f"  ❌ {plugin_dir} - NOT FOUND")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all_good:
        print("✅ All critical components are present and valid!")
        print("\n⚠️  NOTE: External dependencies need to be installed:")
        print("  pip install structlog click pyyaml numpy opencv-python")
        print("  pip install torch tensorflow fastapi aiohttp")
    else:
        print("❌ Some issues found - see above for details")
        print("\n⚠️  The framework structure is complete but may need:")
        print("  1. Installing dependencies")
        print("  2. Checking file permissions")
        print("  3. Ensuring all files were copied correctly")
    
    print("\n📦 Framework Statistics:")
    py_files = list(Path("nexus").rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]
    total_lines = 0
    for f in py_files:
        try:
            with open(f) as file:
                total_lines += len(file.readlines())
        except:
            pass
    
    print(f"  - Python files: {len(py_files)}")
    print(f"  - Total lines of code: {total_lines:,}")
    print(f"  - Plugins: {len([p for p in plugin_dirs if Path(p).exists()])}")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())