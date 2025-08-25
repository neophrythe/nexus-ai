#!/usr/bin/env python3
"""
Nexus Game Automation Framework
Main entry point and runner
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import click
import structlog

# Add nexus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus.core import PluginManager, ConfigManager, setup_logging, get_logger
from nexus.capture import CaptureManager, CaptureBackendType
from nexus.vision import VisionPipeline
from nexus.agents import BaseAgent, AgentType
from nexus.environments import GameEnvironment
from nexus.training import Trainer, TrainingConfig
from nexus.input import PyAutoGUIController

logger = None  # Will be initialized after setup_logging


class NexusRunner:
    """Main runner for Nexus framework"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = ConfigManager(config_path)
        self.plugin_manager: Optional[PluginManager] = None
        self.capture_manager: Optional[CaptureManager] = None
        self.vision_pipeline: Optional[VisionPipeline] = None
        self.input_controller: Optional[PyAutoGUIController] = None
        self.current_game: Optional[GameEnvironment] = None
        self.current_agent: Optional[BaseAgent] = None
        
        # Setup logging
        setup_logging(
            level=self.config.get("logging.level", "INFO"),
            console=self.config.get("logging.console", True),
            file=self.config.get("logging.file"),
            json_format=self.config.get("logging.format", "json") == "json"
        )
        
        global logger
        logger = get_logger("nexus.main")
    
    async def initialize(self) -> None:
        """Initialize all components"""
        logger.info("Initializing Nexus Framework...")
        
        # Initialize plugin manager
        plugin_dirs = [Path(d) for d in self.config.get("nexus.plugin_dirs", ["plugins"])]
        self.plugin_manager = PluginManager(
            plugin_dirs,
            enable_hot_reload=self.config.get("plugins.hot_reload", True)
        )
        await self.plugin_manager.discover_plugins()
        
        # Initialize capture manager
        backend_type = CaptureBackendType(self.config.get("capture.backend", "dxcam"))
        self.capture_manager = CaptureManager(
            backend_type=backend_type,
            device_idx=self.config.get("capture.device_idx", 0),
            output_idx=self.config.get("capture.output_idx"),
            buffer_size=self.config.get("capture.buffer_size", 64)
        )
        await self.capture_manager.initialize()
        
        # Initialize vision pipeline
        self.vision_pipeline = VisionPipeline(
            enable_detection=True,
            enable_ocr=True,
            enable_templates=True,
            detection_model=self.config.get("vision.detection_model", "yolov8n.pt"),
            ocr_engine=self.config.get("vision.ocr_engine", "easyocr")
        )
        await self.vision_pipeline.initialize()
        
        # Initialize input controller
        self.input_controller = PyAutoGUIController(
            human_like=self.config.get("input.human_like", True),
            delay_range=tuple(self.config.get("input.delay_range", [0.05, 0.15]))
        )
        await self.input_controller.initialize()
        
        logger.info("Nexus Framework initialized successfully")
    
    async def load_game(self, game_name: str) -> GameEnvironment:
        """Load a game plugin"""
        logger.info(f"Loading game: {game_name}")
        
        game_plugin = await self.plugin_manager.load_plugin(game_name)
        
        # Create environment wrapper
        from nexus.environments import GameEnvironment
        
        class PluginGameEnvironment(GameEnvironment):
            def __init__(self, plugin, *args, **kwargs):
                self.plugin = plugin
                super().__init__(game_name, *args, **kwargs)
            
            def _build_observation_space(self):
                return self.plugin.get_observation_space()
            
            def _build_action_space(self):
                return self.plugin.get_action_space()
            
            def _get_observation(self):
                # Capture frame
                frame = asyncio.run(self.capture_manager.capture_frame())
                if frame:
                    return frame.data
                return None
            
            def _calculate_reward(self, observation, action):
                return 0  # Override in specific game
            
            def _is_terminated(self, observation):
                return False  # Override in specific game
            
            def _is_truncated(self, observation):
                return self.frame_count >= 10000
            
            def _execute_action(self, action):
                # Map action to input
                pass  # Override in specific game
            
            def _detect_game_phase(self, observation):
                from nexus.environments.base import GamePhase
                return GamePhase.PLAYING
        
        self.current_game = PluginGameEnvironment(
            game_plugin,
            capture_manager=self.capture_manager,
            input_controller=self.input_controller
        )
        
        return self.current_game
    
    async def load_agent(self, agent_name: str) -> BaseAgent:
        """Load an agent plugin"""
        logger.info(f"Loading agent: {agent_name}")
        
        agent_plugin = await self.plugin_manager.load_plugin(agent_name)
        self.current_agent = agent_plugin
        
        return self.current_agent
    
    async def run_game(self, 
                      game_name: str,
                      agent_name: str,
                      episodes: int = 1,
                      render: bool = False) -> Dict[str, Any]:
        """Run a game with an agent"""
        
        # Load game and agent
        game = await self.load_game(game_name)
        agent = await self.load_agent(agent_name)
        
        logger.info(f"Running {game_name} with {agent_name} for {episodes} episodes")
        
        results = {
            "episodes": [],
            "total_reward": 0,
            "avg_reward": 0,
            "best_reward": float('-inf'),
            "worst_reward": float('inf')
        }
        
        for episode in range(episodes):
            obs, info = game.reset()
            agent.reset()
            
            episode_reward = 0
            steps = 0
            
            while True:
                # Agent selects action
                action, _ = await agent.step(obs)
                
                # Execute action
                obs, reward, terminated, truncated, info = game.step(action)
                
                episode_reward += reward
                steps += 1
                
                if render:
                    game.render()
                
                if terminated or truncated:
                    break
            
            results["episodes"].append({
                "episode": episode,
                "reward": episode_reward,
                "steps": steps
            })
            
            results["total_reward"] += episode_reward
            results["best_reward"] = max(results["best_reward"], episode_reward)
            results["worst_reward"] = min(results["worst_reward"], episode_reward)
            
            logger.info(f"Episode {episode + 1}/{episodes} - Reward: {episode_reward:.2f}, Steps: {steps}")
        
        results["avg_reward"] = results["total_reward"] / episodes
        
        return results
    
    async def train_agent(self,
                         game_name: str,
                         agent_name: str,
                         config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """Train an agent on a game"""
        
        # Load game and agent
        game = await self.load_game(game_name)
        agent = await self.load_agent(agent_name)
        
        # Create trainer
        config = config or TrainingConfig()
        trainer = Trainer(agent, game, config)
        
        logger.info(f"Training {agent_name} on {game_name}")
        
        # Run training
        results = await trainer.train()
        
        return results
    
    async def benchmark_capture(self, duration: int = 10) -> Dict[str, Any]:
        """Benchmark capture performance"""
        logger.info(f"Running capture benchmark for {duration} seconds")
        
        results = await self.capture_manager.benchmark(duration)
        
        logger.info(f"Benchmark results: {results['avg_fps']:.2f} FPS, {results['avg_capture_time_ms']:.2f}ms avg capture time")
        
        return results
    
    async def test_vision(self) -> None:
        """Test vision pipeline"""
        logger.info("Testing vision pipeline")
        
        # Capture a frame
        frame = await self.capture_manager.capture_frame()
        
        if frame:
            # Process through vision pipeline
            result = await self.vision_pipeline.process(
                frame.data,
                detect_objects=True,
                detect_text=True
            )
            
            logger.info(f"Vision results: {len(result.objects)} objects, {len(result.text)} text, Processing time: {result.processing_time_ms:.2f}ms")
            
            # Create annotated frame
            annotated = self.vision_pipeline.create_annotated_frame(frame.data, result)
            
            # Save annotated frame
            import cv2
            cv2.imwrite("vision_test.jpg", annotated)
            logger.info("Annotated frame saved to vision_test.jpg")
    
    async def cleanup(self) -> None:
        """Cleanup all resources"""
        logger.info("Cleaning up...")
        
        if self.capture_manager:
            await self.capture_manager.cleanup()
        
        if self.plugin_manager:
            await self.plugin_manager.shutdown()
        
        logger.info("Cleanup complete")


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Nexus Game Automation Framework"""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--duration', '-d', default=10, help='Benchmark duration in seconds')
def benchmark(config: Optional[str], duration: int):
    """Benchmark capture performance"""
    async def run():
        runner = NexusRunner(Path(config) if config else None)
        await runner.initialize()
        await runner.benchmark_capture(duration)
        await runner.cleanup()
    
    asyncio.run(run())


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
def test_vision(config: Optional[str]):
    """Test vision pipeline"""
    async def run():
        runner = NexusRunner(Path(config) if config else None)
        await runner.initialize()
        await runner.test_vision()
        await runner.cleanup()
    
    asyncio.run(run())


@cli.command()
@click.argument('game')
@click.argument('agent')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--episodes', '-e', default=1, help='Number of episodes')
@click.option('--render', '-r', is_flag=True, help='Render game')
def play(game: str, agent: str, config: Optional[str], episodes: int, render: bool):
    """Play a game with an agent"""
    async def run():
        runner = NexusRunner(Path(config) if config else None)
        await runner.initialize()
        results = await runner.run_game(game, agent, episodes, render)
        
        print(f"\nResults:")
        print(f"  Episodes: {episodes}")
        print(f"  Average Reward: {results['avg_reward']:.2f}")
        print(f"  Best Reward: {results['best_reward']:.2f}")
        print(f"  Worst Reward: {results['worst_reward']:.2f}")
        
        await runner.cleanup()
    
    asyncio.run(run())


@cli.command()
@click.argument('game')
@click.argument('agent')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--episodes', '-e', default=1000, help='Training episodes')
@click.option('--save-freq', '-s', default=100, help='Save frequency')
def train(game: str, agent: str, config: Optional[str], episodes: int, save_freq: int):
    """Train an agent on a game"""
    async def run():
        runner = NexusRunner(Path(config) if config else None)
        await runner.initialize()
        
        training_config = TrainingConfig(
            episodes=episodes,
            save_frequency=save_freq
        )
        
        results = await runner.train_agent(game, agent, training_config)
        
        print(f"\nTraining Results:")
        print(f"  Total Episodes: {results['total_episodes']}")
        print(f"  Total Steps: {results['total_steps']}")
        print(f"  Training Time: {results['training_time']:.2f}s")
        print(f"  Best Reward: {results['best_reward']:.2f}")
        print(f"  Final Reward: {results['final_reward']:.2f}")
        print(f"  Average Reward: {results['avg_reward']:.2f}")
        
        await runner.cleanup()
    
    asyncio.run(run())


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--host', '-h', default='127.0.0.1', help='API host')
@click.option('--port', '-p', default=8000, help='API port')
def serve(config: Optional[str], host: str, port: int):
    """Start the web API server"""
    from nexus.api import create_app, run_server
    
    config_mgr = ConfigManager(Path(config) if config else None)
    app = create_app(config_mgr)
    run_server(app, host=host, port=port)


if __name__ == "__main__":
    cli()