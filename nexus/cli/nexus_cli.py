"""Complete CLI Interface for Nexus Game AI Framework"""

import click
import sys
import json
import yaml
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import structlog
from tabulate import tabulate
import time
import numpy as np

from nexus import __version__
# from nexus.core import NexusCore  # TODO: NexusCore class not implemented yet
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


@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstall')
@click.option('--dev', is_flag=True, help='Install in development mode')
def setup(force, dev):
    """Initial setup of Nexus Framework"""
    click.echo("Setting up Nexus Framework...")
    
    # Create directories
    nexus_dir = Path.home() / ".nexus"
    dirs_to_create = [
        nexus_dir,
        nexus_dir / "plugins",
        nexus_dir / "games",
        nexus_dir / "agents",
        nexus_dir / "datasets",
        nexus_dir / "configs",
        nexus_dir / "logs"
    ]
    
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        click.echo(f"✓ Created {directory}")
    
    # Create default configuration
    default_config = {
        "version": __version__,
        "paths": {
            "plugins": str(nexus_dir / "plugins"),
            "games": str(nexus_dir / "games"),
            "agents": str(nexus_dir / "agents"),
            "datasets": str(nexus_dir / "datasets"),
            "logs": str(nexus_dir / "logs")
        },
        "logging": {
            "level": "INFO",
            "format": "json"
        },
        "performance": {
            "max_cpu_percent": 80,
            "max_memory_mb": 2048
        }
    }
    
    config_file = nexus_dir / "config.yaml"
    if not config_file.exists() or force:
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        click.echo(f"✓ Created configuration file: {config_file}")
    
    # Install dependencies if in dev mode
    if dev:
        click.echo("Installing development dependencies...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        click.echo("✓ Installed in development mode")
    
    click.echo("\n✓ Setup complete! Run 'nexus doctor' to verify installation.")


@cli.command()
@click.option('--show', '-s', is_flag=True, help='Show current configuration')
@click.option('--edit', '-e', is_flag=True, help='Edit configuration in editor')
@click.option('--get', '-g', help='Get a specific configuration value')
@click.option('--set', help='Set a configuration value (key=value)')
def config(show, edit, get, set):
    """Manage Nexus configuration"""
    config_file = Path.home() / ".nexus" / "config.yaml"
    
    if not config_file.exists():
        click.echo("Configuration not found. Run 'nexus setup' first.", err=True)
        sys.exit(1)
    
    # Load config
    with open(config_file) as f:
        config_data = yaml.safe_load(f)
    
    if show:
        click.echo(yaml.dump(config_data, default_flow_style=False))
    
    elif edit:
        import os
        editor = os.environ.get('EDITOR', 'nano')
        subprocess.run([editor, str(config_file)])
        click.echo("✓ Configuration updated")
    
    elif get:
        # Navigate through nested keys
        keys = get.split('.')
        value = config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                click.echo(f"Key not found: {get}", err=True)
                sys.exit(1)
        click.echo(value)
    
    elif set:
        # Parse key=value
        if '=' not in set:
            click.echo("Invalid format. Use: --set key=value", err=True)
            sys.exit(1)
        
        key_path, value = set.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to parent and set value
        current = config_data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value to appropriate type
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass  # Keep as string
        
        current[keys[-1]] = value
        
        # Save config
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        click.echo(f"✓ Set {key_path} = {value}")
    
    else:
        click.echo("Use --show, --edit, --get, or --set")


@cli.group()
def game():
    """Game management commands"""
    pass


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


@game.command('list')
def list_games():
    """List all registered games"""
    games_dir = Path.home() / ".nexus" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    
    # Load games registry
    registry_file = games_dir / "registry.json"
    if not registry_file.exists():
        click.echo("No games registered. Use 'nexus game register' to add games.")
        return
    
    with open(registry_file) as f:
        games = json.load(f)
    
    if not games:
        click.echo("No games registered.")
        return
    
    # Format as table
    table_data = []
    for name, info in games.items():
        table_data.append([
            name,
            info.get('type', 'executable'),
            info.get('path', 'N/A')[:50],
            info.get('window_name', 'N/A'),
            "✓" if info.get('installed', False) else "✗"
        ])
    
    headers = ["Game", "Type", "Path", "Window", "Installed"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@game.command('register')
@click.argument('game_name')
@click.option('--path', '-p', help='Game executable path or package name for Android')
@click.option('--window-name', '-w', help='Window name pattern')
@click.option('--type', '-t', type=click.Choice(['steam', 'executable', 'epic', 'web', 'android']), default='executable')
@click.option('--package', help='Android package name (for BlueStacks/emulators)')
def register_game(game_name, path, window_name, type, package):
    """Register a game for use with Nexus"""
    games_dir = Path.home() / ".nexus" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing registry
    registry_file = games_dir / "registry.json"
    games = {}
    if registry_file.exists():
        with open(registry_file) as f:
            games = json.load(f)
    
    # Add new game
    if type == 'android':
        games[game_name] = {
            'type': type,
            'package': package or path,  # Package name
            'window_name': window_name or 'BlueStacks',
            'installed': True,
            'emulator': 'bluestacks'
        }
    else:
        games[game_name] = {
            'type': type,
            'path': path,
            'window_name': window_name or game_name,
            'installed': Path(path).exists() if type == 'executable' else True
        }
    
    # Save registry
    with open(registry_file, 'w') as f:
        json.dump(games, f, indent=2)
    
    click.echo(f"✓ Registered game: {game_name}")


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
    except Exception:
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


@cli.group()
def bluestacks():
    """BlueStacks Android emulator commands"""
    pass


@bluestacks.command('connect')
@click.option('--port', '-p', default=5555, help='ADB port')
def bluestacks_connect(port):
    """Connect to BlueStacks emulator"""
    from nexus.emulators.bluestacks import BlueStacksController, BlueStacksConfig
    
    config = BlueStacksConfig(adb_port=port)
    controller = BlueStacksController(config)
    
    if controller.adb_connected:
        click.echo(f"✓ Connected to BlueStacks on port {port}")
        
        # Show device info
        state = controller.get_game_state()
        click.echo(f"Device state: {'Running' if state['running'] else 'Not running'}")
        
        # List some installed games
        apps = controller.get_installed_apps()
        game_apps = [app for app in apps if any(keyword in app.lower() 
                     for keyword in ['game', 'play', 'clash', 'candy', 'pubg', 'cod'])]
        
        if game_apps:
            click.echo(f"\nDetected games:")
            for app in game_apps[:10]:
                click.echo(f"  - {app}")
    else:
        click.echo(f"✗ Failed to connect to BlueStacks", err=True)


@bluestacks.command('launch')
@click.argument('package_name')
def bluestacks_launch(package_name):
    """Launch Android game in BlueStacks"""
    from nexus.emulators.bluestacks import BlueStacksController
    
    controller = BlueStacksController()
    
    if not controller.adb_connected:
        click.echo("✗ Not connected to BlueStacks. Run 'nexus bluestacks connect' first", err=True)
        return
    
    click.echo(f"Launching {package_name}...")
    
    if controller.launch_app(package_name):
        click.echo(f"✓ Game launched successfully")
        
        # Wait for UI to load
        time.sleep(3)
        
        # Take screenshot
        screenshot = controller.get_screenshot()
        if screenshot is not None:
            click.echo(f"✓ Game is running (screenshot captured)")
    else:
        click.echo(f"✗ Failed to launch game", err=True)


@bluestacks.command('control')
@click.argument('action', type=click.Choice(['tap', 'swipe', 'text', 'back', 'home']))
@click.argument('params', nargs=-1)
def bluestacks_control(action, params):
    """Control BlueStacks game"""
    from nexus.emulators.bluestacks import BlueStacksController
    
    controller = BlueStacksController()
    
    if not controller.adb_connected:
        click.echo("✗ Not connected to BlueStacks", err=True)
        return
    
    if action == 'tap':
        if len(params) >= 2:
            x, y = int(params[0]), int(params[1])
            if controller.tap(x, y):
                click.echo(f"✓ Tapped at ({x}, {y})")
        else:
            click.echo("Usage: nexus bluestacks control tap X Y", err=True)
    
    elif action == 'swipe':
        if len(params) >= 4:
            x1, y1, x2, y2 = map(int, params[:4])
            duration = int(params[4]) if len(params) > 4 else 300
            if controller.swipe(x1, y1, x2, y2, duration):
                click.echo(f"✓ Swiped from ({x1},{y1}) to ({x2},{y2})")
        else:
            click.echo("Usage: nexus bluestacks control swipe X1 Y1 X2 Y2 [duration_ms]", err=True)
    
    elif action == 'text':
        text = ' '.join(params)
        if controller.send_text(text):
            click.echo(f"✓ Sent text: {text}")
    
    elif action == 'back':
        if controller.press_key('KEYCODE_BACK'):
            click.echo("✓ Pressed back button")
    
    elif action == 'home':
        if controller.press_key('KEYCODE_HOME'):
            click.echo("✓ Pressed home button")


@bluestacks.command('auto-play')
@click.argument('game_name')
@click.option('--agent', '-a', help='Agent to use for playing')
@click.option('--duration', '-d', default=300, help='Play duration in seconds')
def bluestacks_autoplay(game_name, agent, duration):
    """Auto-play Android game in BlueStacks"""
    from nexus.emulators.bluestacks import BlueStacksController
    from nexus.agents import load_agent
    
    controller = BlueStacksController()
    
    if not controller.adb_connected:
        # Try to start BlueStacks
        click.echo("Starting BlueStacks...")
        if not controller.start_bluestacks():
            click.echo("✗ Failed to start BlueStacks", err=True)
            return
    
    # Load game registry
    games_dir = Path.home() / ".nexus" / "games"
    registry_file = games_dir / "registry.json"
    
    if registry_file.exists():
        with open(registry_file) as f:
            games = json.load(f)
        
        if game_name in games:
            game_info = games[game_name]
            if game_info.get('type') == 'android':
                package = game_info.get('package')
                
                # Launch game
                click.echo(f"Launching {game_name} ({package})...")
                if not controller.launch_app(package):
                    click.echo("✗ Failed to launch game", err=True)
                    return
                
                time.sleep(5)  # Wait for game to load
                
                # Load agent if specified
                if agent:
                    click.echo(f"Loading agent: {agent}")
                    ai_agent = load_agent(agent)
                else:
                    # Use simple scripted agent
                    click.echo("Using default automation...")
                
                # Auto-play loop
                start_time = time.time()
                frame_count = 0
                
                with click.progressbar(length=duration, label='Auto-playing') as bar:
                    while time.time() - start_time < duration:
                        # Get screenshot
                        screenshot = controller.get_screenshot()
                        
                        if screenshot is not None:
                            frame_count += 1
                            
                            if agent and ai_agent:
                                # Get action from AI agent
                                action = ai_agent.act(screenshot)
                                # Execute action
                                # ... (action execution logic)
                            else:
                                # Simple automated actions
                                if frame_count % 30 == 0:  # Every 30 frames
                                    # Random tap
                                    import random
                                    x = random.randint(100, 1800)
                                    y = random.randint(100, 900)
                                    controller.tap(x, y)
                        
                        # Update progress
                        elapsed = time.time() - start_time
                        bar.update(min(1, elapsed))
                        
                        time.sleep(0.1)  # Small delay
                
                click.echo(f"\n✓ Auto-play complete. Processed {frame_count} frames")
            else:
                click.echo(f"✗ {game_name} is not an Android game", err=True)
        else:
            click.echo(f"✗ Game {game_name} not found. Register it first.", err=True)
    else:
        click.echo("✗ No games registered", err=True)


@cli.command()
@click.argument('game_name')
@click.option('--duration', '-d', default=60, help='Capture duration in seconds')
@click.option('--fps', default=30, help='Frames per second')
@click.option('--output', '-o', help='Output directory')
@click.option('--format', type=click.Choice(['frames', 'video', 'dataset']), default='frames')
def capture(game_name, duration, fps, output, format):
    """Capture game footage for dataset creation"""
    click.echo(f"Starting capture for {game_name}...")
    
    output_dir = Path(output) if output else Path.home() / ".nexus" / "captures" / game_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from nexus.window.window_controller import WindowController
    from nexus.core.frame_grabber import FrameGrabber
    
    # Find game window
    controller = WindowController()
    windows = controller.list_windows()
    game_window = None
    
    for w in windows:
        if game_name.lower() in w.title.lower():
            game_window = w
            break
    
    if not game_window:
        click.echo(f"✗ Game window not found: {game_name}", err=True)
        sys.exit(1)
    
    # Focus the window
    controller.focus_window(game_window)
    
    # Start capture
    grabber = FrameGrabber()
    frames_captured = 0
    start_time = time.time()
    
    with click.progressbar(range(duration * fps), label='Capturing') as bar:
        for i in bar:
            frame = grabber.grab_frame(game_window)
            
            if format == 'frames':
                # Save individual frames
                frame_path = output_dir / f"frame_{i:06d}.jpg"
                import cv2
                cv2.imwrite(str(frame_path), frame)
            
            frames_captured += 1
            
            # Maintain FPS
            elapsed = time.time() - start_time
            expected = (i + 1) / fps
            if elapsed < expected:
                time.sleep(expected - elapsed)
    
    click.echo(f"✓ Captured {frames_captured} frames to {output_dir}")


@cli.command()
@click.argument('agent_path')
@click.argument('game_name')
@click.option('--episodes', '-e', default=1, help='Number of episodes to play')
@click.option('--render', '-r', is_flag=True, help='Show gameplay')
@click.option('--record', is_flag=True, help='Record gameplay')
def play(agent_path, game_name, episodes, render, record):
    """Play a game with a trained agent"""
    click.echo(f"Loading agent from {agent_path}...")
    
    from nexus.agents import load_agent
    from nexus.api.game_api import GameAPIFactory
    
    # Load agent
    agent = load_agent(agent_path)
    click.echo(f"✓ Loaded agent: {agent.name}")
    
    # Create game API
    game_api = GameAPIFactory.create(game_name)
    
    for episode in range(episodes):
        click.echo(f"\nEpisode {episode + 1}/{episodes}")
        
        obs = game_api.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from agent
            action = agent.act(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, info = game_api.step(action)
            
            episode_reward += reward
            steps += 1
            
            if render:
                # Display current frame
                import cv2
                if hasattr(obs, 'shape') and len(obs.shape) == 3:
                    cv2.imshow(f'Nexus - {game_name}', obs)
                    cv2.waitKey(1)
        
        click.echo(f"Episode complete: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    click.echo("\n✓ Playback complete")


@cli.command()
@click.argument('type', type=click.Choice(['game-api', 'agent', 'plugin', 'config']))
@click.argument('name')
@click.option('--output', '-o', help='Output directory')
@click.option('--template', '-t', help='Template to use')
def generate(type, name, output, template):
    """Generate code from templates"""
    output_dir = Path(output) if output else Path.cwd() / name
    
    templates = {
        'game-api': 'game_api_template.py',
        'agent': 'agent_template.py',
        'plugin': 'plugin_template.py',
        'config': 'config_template.yaml'
    }
    
    click.echo(f"Generating {type}: {name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if type == 'game-api':
        # Generate game API implementation
        api_content = f'''"""
Game API implementation for {name}
"""

from nexus.api.game_api import BaseGameAPI
import numpy as np


class {name.replace("-", "").capitalize()}API(BaseGameAPI):
    """API for {name} game"""
    
    def __init__(self):
        super().__init__()
        self.observation_space = np.zeros((480, 640, 3))
        self.action_space = ['up', 'down', 'left', 'right', 'action']
    
    def reset(self):
        """Reset the game"""
        # TODO: Implement game reset
        return self.get_observation()
    
    def step(self, action):
        """Execute an action"""
        # TODO: Implement action execution
        obs = self.get_observation()
        reward = 0
        done = False
        info = {{}}
        return obs, reward, done, info
    
    def get_observation(self):
        """Get current game state"""
        # TODO: Implement observation capture
        return np.zeros((480, 640, 3))
'''
        
        with open(output_dir / f"{name}_api.py", 'w') as f:
            f.write(api_content)
        
        click.echo(f"✓ Generated game API at {output_dir}")
    
    elif type == 'plugin':
        # Use plugin manager to generate
        from nexus.plugins import EnhancedPluginManager
        
        plugins_dir = Path.home() / ".nexus" / "plugins"
        manager = EnhancedPluginManager(plugins_dir)
        
        plugin_dir = manager.generate('game', name, output_dir)
        click.echo(f"✓ Generated plugin at {plugin_dir}")
    
    elif type == 'config':
        # Generate configuration template
        config_content = {
            'name': name,
            'version': '1.0.0',
            'game': {
                'window_name': name,
                'resolution': [640, 480],
                'fps': 60
            },
            'agent': {
                'type': 'dqn',
                'learning_rate': 0.001,
                'batch_size': 32
            },
            'training': {
                'episodes': 1000,
                'save_interval': 100
            }
        }
        
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False)
        
        click.echo(f"✓ Generated config at {output_dir}")
    
    else:
        click.echo(f"Template generation for {type} not implemented yet", err=True)


@cli.command()
@click.option('--port', '-p', default=8080, help='Port for debugger')
@click.option('--host', '-h', default='localhost', help='Host for debugger')
def visual_debugger(port, host):
    """Launch visual debugging interface"""
    click.echo(f"Starting visual debugger on {host}:{port}")
    
    from nexus.debug.visual_debugger import VisualDebugger
    
    debugger = VisualDebugger(host=host, port=port)
    
    try:
        debugger.start()
        click.echo(f"✓ Visual debugger running at http://{host}:{port}")
        click.echo("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        click.echo("\nStopping debugger...")
        debugger.stop()
        click.echo("✓ Debugger stopped")


@cli.command()
def gui():
    """Launch the Nexus GUI (similar to SerpentAI's visual interface)"""
    click.echo("Launching Nexus GUI...")
    
    # Setup environment for WSL2
    import os
    import platform
    
    # Detect if we're on WSL2
    is_wsl2 = False
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                is_wsl2 = True
    except:
        pass
    
    if is_wsl2:
        # Set up X11 display for WSL2
        if not os.environ.get('DISPLAY'):
            try:
                with open("/etc/resolv.conf", "r") as f:
                    for line in f:
                        if "nameserver" in line:
                            ip = line.split()[1]
                            os.environ['DISPLAY'] = f"{ip}:0"
                            click.echo(f"WSL2 detected - setting DISPLAY={os.environ['DISPLAY']}")
                            break
            except:
                click.echo("Warning: Could not auto-detect WSL2 display. Set DISPLAY manually.")
        
        # Set Qt environment for WSL2
        os.environ['QT_XCB_GL_INTEGRATION'] = 'none'
        os.environ['QT_QUICK_BACKEND'] = 'software'
        os.environ.setdefault('XDG_RUNTIME_DIR', f'/tmp/runtime-{os.getuid()}')
        
        # Create runtime dir if needed
        runtime_dir = os.environ['XDG_RUNTIME_DIR']
        if not os.path.exists(runtime_dir):
            os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
    
    try:
        # Try to import from visual_debugger first
        from nexus.gui.visual_debugger import VisualDebugger
        debugger = VisualDebugger()
        
        # For PyQt5, we need to run the Qt application
        if hasattr(debugger, 'app'):
            import sys
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            debugger.window.show()
            sys.exit(app.exec_())
        else:
            debugger.run()
            
    except ImportError as e:
        # Fallback to main_window if it exists
        try:
            from nexus.gui.main_window import main
            main()
        except ImportError:
            click.echo(f"✗ GUI dependencies not installed: {e}", err=True)
            click.echo("\nTo use the GUI, install PyQt5:")
            click.echo("  pip install PyQt5")
            
            if is_wsl2:
                click.echo("\nFor WSL2, also ensure:")
                click.echo("  1. X server is running on Windows (VcXsrv or X410)")
                click.echo("  2. Run: ./setup_gui_wsl2.sh")
                click.echo("  3. DISPLAY is set correctly")
            
            click.echo("\nAlternatively, use the web-based visual debugger:")
            click.echo("  nexus visual-debugger")
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Failed to launch GUI: {e}", err=True)
        if "xcb" in str(e).lower():
            click.echo("\nQt platform plugin error detected!")
            click.echo("For WSL2, run: ./setup_gui_wsl2.sh")
            click.echo("Make sure X server is running on Windows")
        sys.exit(1)


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
        # from nexus.core import NexusCore  # TODO: NexusCore class not implemented yet
        from nexus.window.window_controller import WindowController
        from nexus.api.game_api import GameAPIFactory
        from nexus.agents import create_agent
        from nexus.plugins import EnhancedPluginManager
        
        namespace.update({
            # 'NexusCore': NexusCore,  # TODO: NexusCore class not implemented yet
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


class NexusCLI:
    """Wrapper class for Nexus CLI"""
    
    def __init__(self):
        pass
    
    def run(self):
        """Run the CLI"""
        try:
            cli(obj={})
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


if __name__ == "__main__":
    main()