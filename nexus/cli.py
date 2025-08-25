#!/usr/bin/env python3
import asyncio
import click
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

from nexus.core import PluginManager, ConfigManager, get_logger, setup_logging
from nexus.capture import CaptureManager, CaptureBackendType
from nexus import __version__

console = Console()
logger = get_logger("nexus.cli")


@click.group()
@click.version_option(version=__version__)
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """Nexus Game Automation Framework CLI"""
    ctx.ensure_object(dict)
    
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(level=log_level, console=True, json_format=False)
    
    config_path = Path(config) if config else None
    ctx.obj['config'] = ConfigManager(config_path)
    
    if debug:
        console.print(f"[yellow]Debug mode enabled[/yellow]")


@cli.command()
@click.pass_context
def info(ctx):
    """Show system information"""
    config = ctx.obj['config']
    
    table = Table(title="Nexus System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Version", __version__)
    table.add_row("Config File", str(config.config_path or "Using defaults"))
    table.add_row("Debug Mode", str(config.get("nexus.debug", False)))
    table.add_row("Plugin Dirs", ", ".join(config.get("nexus.plugin_dirs", [])))
    table.add_row("Capture Backend", config.get("capture.backend", "dxcam"))
    table.add_row("Vision Model", config.get("vision.detection_model", "yolov8"))
    table.add_row("API Enabled", str(config.get("api.enabled", True)))
    table.add_row("API Port", str(config.get("api.port", 8000)))
    
    console.print(table)


@cli.group()
@click.pass_context
def plugin(ctx):
    """Plugin management commands"""
    pass


@plugin.command('list')
@click.pass_context
def plugin_list(ctx):
    """List available plugins"""
    config = ctx.obj['config']
    plugin_dirs = [Path(d) for d in config.get("nexus.plugin_dirs", ["plugins"])]
    
    async def list_plugins():
        manager = PluginManager(plugin_dirs)
        plugins = await manager.discover_plugins()
        
        if not plugins:
            console.print("[yellow]No plugins found[/yellow]")
            return
        
        table = Table(title="Available Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="white")
        
        for plugin_name in plugins:
            manifest = manager.plugin_manifests[plugin_name]
            table.add_row(
                manifest.name,
                manifest.version,
                manifest.plugin_type.value,
                manifest.description[:50] + "..." if len(manifest.description) > 50 else manifest.description
            )
        
        console.print(table)
    
    asyncio.run(list_plugins())


@plugin.command('load')
@click.argument('name')
@click.pass_context
def plugin_load(ctx, name: str):
    """Load a plugin"""
    config = ctx.obj['config']
    plugin_dirs = [Path(d) for d in config.get("nexus.plugin_dirs", ["plugins"])]
    
    async def load_plugin():
        manager = PluginManager(plugin_dirs)
        await manager.discover_plugins()
        
        with console.status(f"Loading plugin {name}..."):
            try:
                plugin = await manager.load_plugin(name)
                console.print(f"[green]✓[/green] Plugin {name} loaded successfully")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load plugin {name}: {e}")
    
    asyncio.run(load_plugin())


@cli.group()
@click.pass_context
def capture(ctx):
    """Screen capture commands"""
    pass


@capture.command('test')
@click.option('--duration', '-d', default=5, help='Test duration in seconds')
@click.option('--region', '-r', type=str, help='Capture region (x,y,width,height)')
@click.pass_context
def capture_test(ctx, duration: int, region: Optional[str]):
    """Test screen capture performance"""
    config = ctx.obj['config']
    
    async def test_capture():
        backend_type = CaptureBackendType(config.get("capture.backend", "dxcam"))
        manager = CaptureManager(backend_type=backend_type)
        
        await manager.initialize()
        
        if region:
            try:
                x, y, w, h = map(int, region.split(','))
                manager.set_region_of_interest(x, y, w, h)
                console.print(f"Using region: ({x}, {y}, {w}, {h})")
            except ValueError:
                console.print("[red]Invalid region format. Use: x,y,width,height[/red]")
                return
        
        console.print(f"[cyan]Starting {duration}s capture test...[/cyan]")
        
        results = await manager.benchmark(duration=duration)
        
        table = Table(title="Capture Test Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Average FPS", f"{results['avg_fps']:.2f}")
        table.add_row("Total Frames", str(results['total_frames']))
        table.add_row("Avg Capture Time", f"{results['avg_capture_time_ms']:.2f} ms")
        table.add_row("Min Capture Time", f"{results['min_capture_time_ms']:.2f} ms")
        table.add_row("Max Capture Time", f"{results['max_capture_time_ms']:.2f} ms")
        table.add_row("Backend", results['backend'])
        
        console.print(table)
        
        await manager.cleanup()
    
    asyncio.run(test_capture())


@capture.command('info')
@click.pass_context
def capture_info(ctx):
    """Show capture device information"""
    config = ctx.obj['config']
    
    async def show_info():
        backend_type = CaptureBackendType(config.get("capture.backend", "dxcam"))
        manager = CaptureManager(backend_type=backend_type)
        
        await manager.initialize()
        info = manager.get_screen_info()
        
        table = Table(title="Capture Device Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Backend", info['backend'])
        table.add_row("Primary Resolution", f"{info['primary_resolution'][0]}x{info['primary_resolution'][1]}")
        
        if info['outputs']:
            console.print("\n[yellow]Available Outputs:[/yellow]")
            output_table = Table()
            output_table.add_column("Device", style="cyan")
            output_table.add_column("Output", style="green")
            output_table.add_column("Resolution", style="yellow")
            output_table.add_column("Primary", style="white")
            
            for output in info['outputs']:
                output_table.add_row(
                    str(output['device_idx']),
                    str(output['output_idx']),
                    f"{output['resolution'][0]}x{output['resolution'][1]}",
                    "Yes" if output['primary'] else "No"
                )
            
            console.print(output_table)
        
        await manager.cleanup()
    
    asyncio.run(show_info())


@cli.command()
@click.option('--host', '-h', default='127.0.0.1', help='API host')
@click.option('--port', '-p', default=8000, help='API port')
@click.pass_context
def serve(ctx, host: str, port: int):
    """Start the web API server"""
    config = ctx.obj['config']
    
    console.print(f"[cyan]Starting Nexus API server on {host}:{port}...[/cyan]")
    
    try:
        from nexus.api.server import create_app, run_server
        
        app = create_app(config)
        run_server(app, host=host, port=port)
    except ImportError:
        console.print("[red]API module not available. Please install API dependencies.[/red]")
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")


@cli.command()
@click.argument('game_name')
@click.option('--agent', '-a', default='scripted', help='Agent type to use')
@click.option('--episodes', '-e', default=1, help='Number of episodes to run')
@click.pass_context
def run(ctx, game_name: str, agent: str, episodes: int):
    """Run a game with an agent"""
    config = ctx.obj['config']
    
    async def run_game():
        plugin_dirs = [Path(d) for d in config.get("nexus.plugin_dirs", ["plugins"])]
        manager = PluginManager(plugin_dirs)
        
        await manager.discover_plugins()
        
        try:
            with console.status(f"Loading game plugin {game_name}..."):
                game_plugin = await manager.load_plugin(game_name)
            
            with console.status(f"Loading agent {agent}..."):
                agent_plugin = await manager.load_plugin(agent)
            
            console.print(f"[green]Starting {game_name} with {agent} agent[/green]")
            
            for episode in range(episodes):
                console.print(f"\n[cyan]Episode {episode + 1}/{episodes}[/cyan]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    asyncio.run(run_game())


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration"""
    config = ctx.obj['config']
    
    import yaml
    console.print(Panel(yaml.dump(config.to_dict(), default_flow_style=False), title="Current Configuration"))


@cli.command()
@click.argument('key')
@click.argument('value')
@click.pass_context
def config_set(ctx, key: str, value: str):
    """Set a configuration value"""
    config = ctx.obj['config']
    
    try:
        import ast
        parsed_value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed_value = value
    
    config.set(key, parsed_value)
    config.save()
    
    console.print(f"[green]✓[/green] Set {key} = {parsed_value}")


def main():
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()