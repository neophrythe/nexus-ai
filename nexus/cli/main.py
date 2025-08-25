"""Main CLI entry point for Nexus Game AI Framework"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexus.cli.nexus_cli import NexusCLI
from nexus.utils.platform import is_windows, is_linux, is_macos


def main():
    """Main entry point for 'nexus' command"""
    cli = NexusCLI()
    cli.run()


def serpent_compat():
    """SerpentAI compatibility entry point for 'serpent' command"""
    print("Running in SerpentAI compatibility mode...")
    print("Note: This is the Nexus Game AI Framework with SerpentAI compatibility")
    print()
    
    # Map serpent commands to nexus equivalents
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Map common SerpentAI commands
        serpent_to_nexus = {
            "setup": "setup",
            "grab_frames": "capture",
            "generate": "generate",
            "train": "train",
            "play": "play",
            "record": "record",
            "visual_debugger": "debug",
            "analytics": "analytics",
        }
        
        if command in serpent_to_nexus:
            sys.argv[1] = serpent_to_nexus[command]
    
    cli = NexusCLI()
    cli.run()


if __name__ == "__main__":
    main()