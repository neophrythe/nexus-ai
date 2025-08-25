"""Complete CLI Interface for Nexus Game AI Framework"""

import click
import sys
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from tabulate import tabulate
import time

from nexus import __version__
from nexus.core import NexusCore
from nexus.plugins import EnhancedPluginManager
from nexus.launchers.game_launcher import GameLauncherFactory, LaunchConfig, LauncherType
from nexus.analytics.experiment_tracker import init_tracker
from nexus.window.window_controller import WindowController
from nexus.api.game_api import GameAPIFactory
from nexus.agents.agent_manager import AgentManager
from nexus.datasets.dataset_manager import DatasetManager, DatasetConfig
from nexus.profiling.performance_profiler import PerformanceProfiler

logger = structlog.get_logger()


@click.group()
@click.version_option(version=__version__)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Nexus Game AI Framework - Complete Game Automation Platform"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        config_path = Path(config)
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                ctx.obj['config'] = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path) as f:
                ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}
    
    ctx.obj['verbose'] = verbose
    
    if verbose:
        click.echo(f"Nexus Framework v{__version__}")


@cli.group()
def game():
    """Game management commands"""
    click.echo("Game management commands available:")


@game.command()
@click.argument('game_name')
@click.option('--launcher', '-l', type=click.Choice(['steam', 'executable', 'epic', 'web']), default='executable')
@click.option('--path', '-p', help='Game executable path')
@click.option('--app-id', '-a', help='Steam/Epic app ID')
@click.option('--url', '-u', help='Web game URL')
@click.option('--window-name', '-w', help='Window name to wait for')
@click.option('--args', multiple=True, help='Game launch arguments')
@click.pass_context
def launch(ctx, game_name, launcher, path, app_id, url, window_name, args):
    """Launch a game"""
    click.echo(f"Launching {game_name}...")
    
    # Create launch config
    config = LaunchConfig(
        launcher_type=LauncherType(launcher),
        game_path=path,
        app_id=app_id,
        url=url,
        window_name=window_name or game_name,
        arguments=list(args)
    )
    
    try:
        launcher = GameLauncherFactory.launch_game(config)
        click.echo(f"✓ Game launched successfully")
        
        if window_name:
            click.echo(f"Waiting for window: {window_name}")
            if launcher.wait_for_window():
                click.echo("✓ Window found")
            else:
                click.echo("✗ Window not found", err=True)
    
    except Exception as e:
        click.echo(f"✗ Failed to launch game: {e}", err=True)
        sys.exit(1)


@game.command()
@click.option('--filter', '-f', help='Filter by window name')
def list_windows(filter):
    """List all game windows"""
    controller = WindowController()
    windows = controller.list_windows()
    
    if filter:
        windows = [w for w in windows if filter.lower() in w.title.lower()]
    
    if not windows:
        click.echo("No windows found")
        return
    
    # Format as table
    table_data = []
    for w in windows:
        table_data.append([
            w.title[:50],
            f"{w.width}x{w.height}",
            f"({w.x}, {w.y})",
            "✓" if w.is_focused else "",
            w.process_name or "Unknown"
        ])
    
    headers = ["Window Title", "Size", "Position", "Focused", "Process"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@cli.group()
def plugin():
    """Plugin management commands"""
    click.echo("Plugin management commands available:")


@plugin.command()
@click.argument('source')
@click.option('--type', '-t', type=click.Choice(['local', 'git', 'url', 'registry']), default='registry')
@click.pass_context
def install(ctx, source, type):
    """Install a plugin"""
    click.echo(f"Installing plugin from {source}...")
    
    plugins_dir = Path.home() / ".nexus" / "plugins"
    manager = EnhancedPluginManager(plugins_dir)
    
    try:
        import asyncio
        plugin = asyncio.run(manager.install(source, type))
        click.echo(f"✓ Plugin {plugin.metadata.name} v{plugin.metadata.version} installed")
    except Exception as e:
        click.echo(f"✗ Failed to install plugin: {e}", err=True)
        sys.exit(1)


@plugin.command()
@click.argument('plugin_name')
@click.pass_context
def uninstall(ctx, plugin_name):
    """Uninstall a plugin"""
    click.echo(f"Uninstalling plugin {plugin_name}...")
    
    plugins_dir = Path.home() / ".nexus" / "plugins"
    manager = EnhancedPluginManager(plugins_dir)
    
    try:
        import asyncio
        success = asyncio.run(manager.uninstall(plugin_name))
        if success:
            click.echo(f"✓ Plugin {plugin_name} uninstalled")
        else:
            click.echo(f"✗ Plugin {plugin_name} not found", err=True)
    except Exception as e:
        click.echo(f"✗ Failed to uninstall plugin: {e}", err=True)
        sys.exit(1)


@plugin.command()
def list():
    """List installed plugins"""
    plugins_dir = Path.home() / ".nexus" / "plugins"
    manager = EnhancedPluginManager(plugins_dir)
    
    installed = manager.list_installed()
    
    if not installed:
        click.echo("No plugins installed")
        return
    
    table_data = []
    for name, plugin in installed.items():
        table_data.append([
            name,
            plugin.metadata.version,
            plugin.metadata.plugin_type,
            "✓" if plugin.is_active else "",
            plugin.metadata.author
        ])
    
    headers = ["Plugin", "Version", "Type", "Active", "Author"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@plugin.command()
@click.argument('plugin_type', type=click.Choice(['game', 'agent', 'vision', 'input']))
@click.argument('name')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def generate(plugin_type, name, output):
    """Generate a new plugin from template"""
    output_dir = Path(output) if output else Path.cwd()
    
    plugins_dir = Path.home() / ".nexus" / "plugins"
    manager = EnhancedPluginManager(plugins_dir)
    
    try:
        plugin_dir = manager.generate(plugin_type, name, output_dir)
        click.echo(f"✓ Generated {plugin_type} plugin at {plugin_dir}")
        click.echo(f"\nTo install: nexus plugin install {plugin_dir} --type local")
    except Exception as e:
        click.echo(f"✗ Failed to generate plugin: {e}", err=True)
        sys.exit(1)


@cli.group()
def agent():
    """Agent training and management"""
    click.echo("Agent training and management commands available:")


@agent.command()
@click.argument('agent_type', type=click.Choice(['dqn', 'rainbow', 'ppo', 'random']))
@click.argument('game_name')
@click.option('--config', '-c', type=click.Path(exists=True), help='Agent configuration file')
@click.option('--episodes', '-e', default=1000, help='Number of episodes')
@click.option('--checkpoint', '-cp', help='Load from checkpoint')
@click.option('--save-dir', '-s', type=click.Path(), help='Save directory')
@click.option('--visualize', '-v', is_flag=True, help='Visualize training')
@click.pass_context
def train(ctx, agent_type, game_name, config, episodes, checkpoint, save_dir, visualize):
    """Train an agent on a game"""
    click.echo(f"Training {agent_type} agent on {game_name}")
    
    # Initialize experiment tracker
    tracker = init_tracker("nexus", f"{game_name}_{agent_type}")
    
    # Load agent config
    agent_config = {}
    if config:
        with open(config) as f:
            agent_config = yaml.safe_load(f) if config.endswith('.yaml') else json.load(f)
    
    tracker.log_config({
        "agent_type": agent_type,
        "game": game_name,
        "episodes": episodes,
        **agent_config
    })
    
    # Create game API
    game_api = GameAPIFactory.create(game_name)
    
    # Create agent
    from nexus.agents import create_agent
    agent = create_agent(agent_type, game_api.get_observation_space(), 
                         game_api.get_action_space(), agent_config)
    
    if checkpoint:
        agent.load(checkpoint)
        click.echo(f"Loaded checkpoint: {checkpoint}")
    
    # Training loop
    with click.progressbar(range(episodes), label='Training') as bar:
        for episode in bar:
            obs = game_api.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, info = game_api.step(action)
                
                agent.remember(obs, action, reward, next_obs, done)
                agent.learn()
                
                obs = next_obs
                episode_reward += reward
                
                if visualize:
                    # Show frame
                    import cv2
                    frame = next_obs if hasattr(next_obs, 'shape') else None
                    if frame is not None and len(frame.shape) == 3:
                        cv2.imshow('Training Visualization', frame)
                        cv2.waitKey(1)
            
            # Log metrics
            tracker.log_metrics({
                "episode_reward": episode_reward,
                "epsilon": getattr(agent, 'epsilon', 0),
                "loss": getattr(agent, 'loss', 0)
            })
            
            # Save checkpoint
            if save_dir and episode % 100 == 0:
                save_path = Path(save_dir) / f"checkpoint_{episode}.pth"
                agent.save(save_path)
    
    tracker.finish()
    click.echo(f"✓ Training complete")


@agent.command()
@click.argument('agent_path')
@click.argument('game_name')
@click.option('--episodes', '-e', default=10, help='Number of evaluation episodes')
@click.option('--render', '-r', is_flag=True, help='Render gameplay')
@click.option('--record', is_flag=True, help='Record gameplay')
def evaluate(agent_path, game_name, episodes, render, record):
    """Evaluate a trained agent"""
    click.echo(f"Evaluating agent on {game_name}")
    
    # Load agent
    from nexus.agents import load_agent
    agent = load_agent(agent_path)
    
    # Create game API
    game_api = GameAPIFactory.create(game_name)
    
    rewards = []
    for episode in range(episodes):
        obs = game_api.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = game_api.step(action)
            episode_reward += reward
            
            if render:
                # Display frame
                import cv2
                frame = obs if hasattr(obs, 'shape') else None
                if frame is not None and len(frame.shape) == 3:
                    cv2.imshow('Agent Evaluation', frame)
                    cv2.waitKey(1)
        
        rewards.append(episode_reward)
        click.echo(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    click.echo(f"\nAverage Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


@cli.group()
def dataset():
    """Dataset management commands"""
    click.echo("Dataset management commands available:")


@dataset.command()
@click.argument('name')
@click.option('--source', '-s', type=click.Path(exists=True), help='Source directory')
@click.option('--format', '-f', type=click.Choice(['hdf5', 'zarr', 'lmdb', 'numpy']), default='hdf5')
@click.option('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15], help='Train/val/test split')
def create(name, source, format, split):
    """Create a new dataset"""
    click.echo(f"Creating dataset: {name}")
    
    config = DatasetConfig(
        name=name,
        format=format,
        train_ratio=split[0],
        val_ratio=split[1],
        test_ratio=split[2]
    )
    
    manager = DatasetManager(config)
    
    if source:
        # Load data from source
        source_path = Path(source)
        file_count = 0
        
        for file_path in source_path.rglob("*.png"):
            # Assume parent directory is label
            label = file_path.parent.name
            
            import cv2
            image = cv2.imread(str(file_path))
            manager.add_sample(image, label)
            file_count += 1
        
        click.echo(f"Loaded {file_count} samples")
    
    # Create splits
    manager.create_splits()
    
    # Save dataset
    output_path = Path.home() / ".nexus" / "datasets" / f"{name}.{format}"
    manager.save_dataset(str(output_path))
    
    click.echo(f"✓ Dataset saved to {output_path}")
    
    # Show statistics
    stats = manager.get_statistics()
    click.echo(f"\nDataset Statistics:")
    click.echo(f"  Total samples: {stats['total_samples']}")
    click.echo(f"  Unique labels: {len(stats['unique_labels'])}")
    click.echo(f"  Size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")


@cli.group()
def profile():
    """Performance profiling commands"""
    click.echo("Performance profiling commands available:")


@profile.command()
@click.option('--level', '-l', type=click.Choice(['minimal', 'standard', 'detailed', 'full']), default='standard')
@click.option('--duration', '-d', default=60, help='Profiling duration in seconds')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def start(level, duration, output):
    """Start performance profiling"""
    click.echo(f"Starting profiler (level: {level}, duration: {duration}s)")
    
    from nexus.profiling.performance_profiler import ProfileLevel
    
    profiler = PerformanceProfiler(ProfileLevel[level.upper()])
    profiler.start()
    
    try:
        with click.progressbar(range(duration), label='Profiling') as bar:
            for _ in bar:
                time.sleep(1)
                
                # Get snapshot
                snapshot = profiler.get_current_snapshot()
                
                # Show current stats
                if snapshot:
                    click.echo(f"\rFPS: {snapshot.fps:.1f} | "
                             f"CPU: {snapshot.cpu_percent:.1f}% | "
                             f"Memory: {snapshot.memory_mb:.1f} MB", nl=False)
    
    except KeyboardInterrupt:
        click.echo("\nProfiler stopped by user")
    
    finally:
        profiler.stop()
        
        if output:
            profiler.export_metrics(output)
            click.echo(f"\n✓ Profiling data saved to {output}")
        
        # Show summary
        summary = profiler.get_metrics_summary()
        click.echo("\nProfiling Summary:")
        click.echo(f"  Average FPS: {summary['fps']['mean']:.2f}")
        click.echo(f"  Average CPU: {summary['cpu_percent']['mean']:.1f}%")
        click.echo(f"  Peak Memory: {summary['memory_mb']['max']:.1f} MB")


@cli.command()
@click.option('--host', '-h', default='localhost', help='Server host')
@click.option('--port', '-p', default=8000, help='Server port')
@click.option('--debug', is_flag=True, help='Debug mode')
def serve(host, port, debug):
    """Start the Nexus API server"""
    click.echo(f"Starting Nexus API server on {host}:{port}")
    
    from nexus.api.server import create_app
    
    app = create_app(debug=debug)
    
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port, log_level="info" if debug else "error")
    except ImportError:
        click.echo("✗ uvicorn not installed. Install with: pip install uvicorn", err=True)
        sys.exit(1)


@cli.command()
@click.option('--browser', '-b', is_flag=True, help='Open in browser')
def dashboard(browser):
    """Launch the web dashboard"""
    click.echo("Starting Nexus Dashboard...")
    
    # Start API server in background
    import subprocess
    import webbrowser
    
    api_process = subprocess.Popen([sys.executable, "-m", "nexus.api.server"])
    
    # Start dashboard
    dashboard_dir = Path(__file__).parent.parent / "dashboard"
    
    if dashboard_dir.exists():
        subprocess.Popen(["npm", "start"], cwd=dashboard_dir)
        
        if browser:
            time.sleep(2)
            webbrowser.open("http://localhost:3000")
        
        click.echo("✓ Dashboard running at http://localhost:3000")
        click.echo("Press Ctrl+C to stop")
        
        try:
            api_process.wait()
        except KeyboardInterrupt:
            api_process.terminate()
    else:
        click.echo("✗ Dashboard not found. Run setup first.", err=True)


@cli.command()
def doctor():
    """Check system setup and dependencies"""
    click.echo("Running system diagnostics...\n")
    
    checks = []
    
    # Check Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python Version", py_version, sys.version_info >= (3, 8)))
    
    # Check core dependencies
    dependencies = [
        ("NumPy", "numpy"),
        ("OpenCV", "cv2"),
        ("PyTorch", "torch"),
        ("Pillow", "PIL"),
        ("Structlog", "structlog"),
    ]
    
    for name, module in dependencies:
        try:
            __import__(module)
            checks.append((name, "✓ Installed", True))
        except ImportError:
            checks.append((name, "✗ Not installed", False))
    
    # Check optional dependencies
    optional = [
        ("TensorFlow", "tensorflow"),
        ("Tesseract OCR", "pytesseract"),
        ("EasyOCR", "easyocr"),
        ("Weights & Biases", "wandb"),
        ("MLflow", "mlflow"),
    ]
    
    click.echo("Optional Dependencies:")
    for name, module in optional:
        try:
            __import__(module)
            click.echo(f"  {name}: ✓ Installed")
        except ImportError:
            click.echo(f"  {name}: ✗ Not installed")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(("CUDA", f"✓ Available ({torch.cuda.get_device_name(0)})", True))
        else:
            checks.append(("CUDA", "Not available", False))
    except:
        checks.append(("CUDA", "Cannot check", False))
    
    # Check window controller
    try:
        from nexus.window.window_controller import WindowController
        controller = WindowController()
        checks.append(("Window Controller", "✓ Working", True))
    except Exception as e:
        checks.append(("Window Controller", f"✗ Error: {e}", False))
    
    # Display results
    click.echo("\nSystem Check Results:")
    all_good = True
    for check, status, passed in checks:
        symbol = "✓" if passed else "✗"
        click.echo(f"  {symbol} {check}: {status}")
        if not passed:
            all_good = False
    
    if all_good:
        click.echo("\n✓ All checks passed! Nexus is ready to use.")
    else:
        click.echo("\n⚠ Some checks failed. Install missing dependencies for full functionality.")


@cli.command()
def interactive():
    """Start interactive Python session with Nexus loaded"""
    click.echo("Starting Nexus interactive session...")
    
    # Import everything
    import code
    import readline
    import rlcompleter
    
    # Enable tab completion
    readline.parse_and_bind("tab: complete")
    
    # Import Nexus modules
    namespace = {
        'nexus': __import__('nexus'),
        'np': __import__('numpy'),
        'cv2': __import__('cv2'),
        'Path': Path,
    }
    
    # Add convenience imports
    try:
        from nexus.core import NexusCore
        from nexus.window.window_controller import WindowController
        from nexus.api.game_api import GameAPIFactory
        from nexus.agents import create_agent
        from nexus.plugins import EnhancedPluginManager
        
        namespace.update({
            'NexusCore': NexusCore,
            'WindowController': WindowController,
            'GameAPIFactory': GameAPIFactory,
            'create_agent': create_agent,
            'PluginManager': EnhancedPluginManager,
        })
    except ImportError as e:
        click.echo(f"Warning: Could not import some modules: {e}")
    
    banner = """
    Nexus Game AI Framework - Interactive Session
    
    Available objects:
      nexus          - Main nexus module
      np            - NumPy
      cv2           - OpenCV
      Path          - pathlib.Path
      
    Available classes:
      NexusCore     - Core framework
      WindowController - Window management
      GameAPIFactory - Game API creation
      
    Type help(object) for documentation
    """
    
    code.interact(banner=banner, local=namespace)


def main():
    """Main entry point"""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()