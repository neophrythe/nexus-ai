#!/usr/bin/env python3
"""
Quick start script for Nexus Game Automation Framework
"""

import asyncio
import sys
from pathlib import Path

# Add nexus to path
sys.path.insert(0, str(Path(__file__).parent))

from nexus.main import NexusRunner
from nexus.core import setup_logging, get_logger

async def main():
    """Main entry point"""
    
    # Setup logging
    setup_logging(level="INFO", console=True, json_format=False)
    logger = get_logger("nexus.quickstart")
    
    logger.info("Starting Nexus Game Automation Framework")
    
    # Create runner
    runner = NexusRunner()
    
    try:
        # Initialize framework
        await runner.initialize()
        
        # Run benchmark
        logger.info("Running capture benchmark...")
        results = await runner.benchmark_capture(duration=5)
        print(f"\nCapture Performance:")
        print(f"  Average FPS: {results['avg_fps']:.2f}")
        print(f"  Average Capture Time: {results['avg_capture_time_ms']:.2f}ms")
        
        # Test vision
        logger.info("\nTesting vision pipeline...")
        await runner.test_vision()
        print("Vision test complete - check vision_test.jpg for results")
        
        # List available plugins
        plugins = await runner.plugin_manager.discover_plugins()
        if plugins:
            print(f"\nAvailable Plugins:")
            for plugin in plugins:
                print(f"  - {plugin}")
        
        # Example: Run game with agent (if plugins available)
        if "example_game" in plugins:
            logger.info("\nRunning example game...")
            results = await runner.run_game("example_game", "example_agent", episodes=1)
            print(f"\nGame Results:")
            print(f"  Average Reward: {results['avg_reward']:.2f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    finally:
        # Cleanup
        await runner.cleanup()
    
    logger.info("Nexus shutdown complete")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)