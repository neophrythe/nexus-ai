import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
import yaml
import toml
import json
from dataclasses import dataclass, field
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = structlog.get_logger()


@dataclass
class ConfigSchema:
    nexus: Dict[str, Any] = field(default_factory=dict)
    capture: Dict[str, Any] = field(default_factory=dict)
    vision: Dict[str, Any] = field(default_factory=dict)
    agents: Dict[str, Any] = field(default_factory=dict)
    input: Dict[str, Any] = field(default_factory=dict)
    api: Dict[str, Any] = field(default_factory=dict)
    plugins: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)


class ConfigFileHandler(FileSystemEventHandler):
    
    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager
        
    def on_modified(self, event: FileModifiedEvent):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if path.name in ["config.yaml", "config.yml", "config.toml", "config.json"]:
            logger.info(f"Config file modified: {path}")
            self.config_manager.reload()


class ConfigManager:
    
    DEFAULT_CONFIG = {
        "nexus": {
            "version": "0.1.0",
            "debug": False,
            "auto_reload": True,
            "plugin_dirs": ["plugins"],
        },
        "capture": {
            "backend": "dxcam",
            "device_idx": 0,
            "output_idx": None,
            "fps": 60,
            "buffer_size": 64,
            "region": None,
        },
        "vision": {
            "detection_model": "yolov8",
            "ocr_engine": "easyocr",
            "confidence_threshold": 0.5,
            "gpu_enabled": True,
            "batch_size": 1,
        },
        "agents": {
            "default_type": "scripted",
            "max_buffer_size": 10000,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "input": {
            "backend": "pyautogui",
            "human_like": True,
            "delay_range": [0.05, 0.15],
            "mouse_speed": 1.0,
        },
        "api": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8000,
            "cors_enabled": True,
            "auth_enabled": False,
        },
        "plugins": {
            "auto_discovery": True,
            "hot_reload": True,
            "sandboxed": False,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "file": "nexus.log",
            "console": True,
            "rotation": "1 day",
        },
        "performance": {
            "max_cpu_percent": 80,
            "max_memory_mb": 4096,
            "gpu_memory_fraction": 0.5,
            "profiling_enabled": False,
            "thread_pool_size": 4,
        },
        "games": {
            "auto_detect": True,
            "window_title_patterns": [],
            "process_name_patterns": [],
        },
        "training": {
            "save_interval": 1000,
            "log_interval": 100,
            "checkpoint_dir": "checkpoints",
            "tensorboard_enabled": True,
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, auto_reload: bool = True):
        self.config_path = Path(config_path) if config_path else self._find_config_file()
        self.auto_reload = auto_reload
        self._config: Dict[str, Any] = {}
        self._observer: Optional[Observer] = None
        
        self.load()
        
        if self.auto_reload and self.config_path and self.config_path.exists():
            self._setup_file_watcher()
    
    def _find_config_file(self) -> Optional[Path]:
        search_paths = [
            Path.cwd(),
            Path.home() / ".nexus",
            Path("/etc/nexus") if os.name != 'nt' else Path.home() / "AppData" / "Roaming" / "nexus",
        ]
        
        config_names = ["config.yaml", "config.yml", "config.toml", "config.json", "nexus.yaml", "nexus.toml"]
        
        for path in search_paths:
            for name in config_names:
                config_file = path / name
                if config_file.exists():
                    logger.info(f"Found config file: {config_file}")
                    return config_file
        
        return None
    
    def _setup_file_watcher(self):
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        
        watch_dir = self.config_path.parent if self.config_path else Path.cwd()
        self._observer.schedule(handler, str(watch_dir), recursive=False)
        self._observer.start()
        
        logger.info(f"Config auto-reload enabled for {watch_dir}")
    
    def load(self) -> None:
        if not self.config_path or not self.config_path.exists():
            logger.warning("No config file found, using defaults")
            self._config = self.DEFAULT_CONFIG.copy()
            return
        
        try:
            suffix = self.config_path.suffix.lower()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f)
                elif suffix == '.toml':
                    self._config = toml.load(f)
                elif suffix == '.json':
                    self._config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
            
            self._config = self._merge_with_defaults(self._config)
            
            logger.info(f"Config loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._config = self.DEFAULT_CONFIG.copy()
    
    def reload(self) -> None:
        logger.info("Reloading configuration...")
        old_config = self._config.copy()
        
        try:
            self.load()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload config, keeping old config: {e}")
            self._config = old_config
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        def deep_merge(default: Dict, override: Dict) -> Dict:
            result = default.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return deep_merge(self.DEFAULT_CONFIG, config)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        logger.info(f"Config updated: {key} = {value}")
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            save_path = Path.cwd() / "config.yaml"
        
        try:
            suffix = save_path.suffix.lower()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
                elif suffix == '.toml':
                    toml.dump(self._config, f)
                elif suffix == '.json':
                    json.dump(self._config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {suffix}")
            
            logger.info(f"Config saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def get_section(self, section: str) -> Dict[str, Any]:
        return self._config.get(section, {})
    
    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section].update(values)
        logger.info(f"Config section '{section}' updated")
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration values"""
        errors = []
        
        if self.get("capture.fps", 60) < 1 or self.get("capture.fps", 60) > 240:
            errors.append("capture.fps must be between 1 and 240")
        
        if self.get("capture.buffer_size", 64) < 1:
            errors.append("capture.buffer_size must be at least 1")
        
        if self.get("vision.confidence_threshold", 0.5) < 0 or self.get("vision.confidence_threshold", 0.5) > 1:
            errors.append("vision.confidence_threshold must be between 0 and 1")
        
        if self.get("api.port", 8000) < 1 or self.get("api.port", 8000) > 65535:
            errors.append("api.port must be between 1 and 65535")
        
        if self.get("performance.max_cpu_percent", 80) < 1 or self.get("performance.max_cpu_percent", 80) > 100:
            errors.append("performance.max_cpu_percent must be between 1 and 100")
        
        if self.get("performance.thread_pool_size", 4) < 1:
            errors.append("performance.thread_pool_size must be at least 1")
        
        if self.get("training.save_interval", 1000) < 1:
            errors.append("training.save_interval must be at least 1")
        
        if self.get("training.log_interval", 100) < 1:
            errors.append("training.log_interval must be at least 1")
        
        valid = len(errors) == 0
        
        if not valid:
            for error in errors:
                logger.error(f"Config validation error: {error}")
        
        return valid, errors
    
    def export_env_vars(self) -> Dict[str, str]:
        """Export configuration as environment variables"""
        env_vars = {}
        
        def flatten_dict(d: Dict[str, Any], prefix: str = "NEXUS") -> None:
            for key, value in d.items():
                env_key = f"{prefix}_{key.upper()}"
                if isinstance(value, dict):
                    flatten_dict(value, env_key)
                else:
                    env_vars[env_key] = str(value)
        
        flatten_dict(self._config)
        return env_vars
    
    def load_from_env(self) -> None:
        """Load configuration values from environment variables"""
        for key, value in os.environ.items():
            if key.startswith("NEXUS_"):
                # Convert NEXUS_CAPTURE_FPS to capture.fps
                config_key = key[6:].lower().replace("_", ".")
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.isdigit():
                        value = int(value)
                    elif "." in value and value.replace(".", "", 1).isdigit():
                        value = float(value)
                except ValueError:
                    pass
                
                self.set(config_key, value)
        
        logger.info("Configuration loaded from environment variables")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Config file watcher stopped")
    
    def __del__(self):
        self.cleanup()


_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def set_config(config: ConfigManager) -> None:
    """Set the global configuration manager instance"""
    global _global_config
    _global_config = config


def create_default_config(path: Union[str, Path] = "config.yaml") -> None:
    """Create a default configuration file"""
    config_path = Path(path)
    
    if config_path.exists():
        logger.warning(f"Config file already exists: {config_path}")
        return
    
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create config manager with defaults and save
    config = ConfigManager()
    config.save(config_path)
    logger.info(f"Default configuration created: {config_path}")


class ProfileManager:
    """Manage configuration profiles for different games/scenarios"""
    
    def __init__(self, profiles_dir: Union[str, Path] = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.current_profile: Optional[str] = None
        self.base_config: Optional[ConfigManager] = None
    
    def create_profile(self, name: str, base_config: Optional[ConfigManager] = None) -> None:
        """Create a new configuration profile"""
        profile_path = self.profiles_dir / f"{name}.yaml"
        
        if profile_path.exists():
            raise ValueError(f"Profile '{name}' already exists")
        
        if base_config:
            base_config.save(profile_path)
        else:
            # Create with defaults
            config = ConfigManager()
            config.save(profile_path)
        
        logger.info(f"Created profile: {name}")
    
    def load_profile(self, name: str) -> ConfigManager:
        """Load a configuration profile"""
        profile_path = self.profiles_dir / f"{name}.yaml"
        
        if not profile_path.exists():
            raise ValueError(f"Profile '{name}' not found")
        
        config = ConfigManager(profile_path)
        self.current_profile = name
        logger.info(f"Loaded profile: {name}")
        return config
    
    def delete_profile(self, name: str) -> None:
        """Delete a configuration profile"""
        profile_path = self.profiles_dir / f"{name}.yaml"
        
        if not profile_path.exists():
            raise ValueError(f"Profile '{name}' not found")
        
        profile_path.unlink()
        
        if self.current_profile == name:
            self.current_profile = None
        
        logger.info(f"Deleted profile: {name}")
    
    def list_profiles(self) -> List[str]:
        """List available configuration profiles"""
        profiles = []
        for file in self.profiles_dir.glob("*.yaml"):
            profiles.append(file.stem)
        return sorted(profiles)
    
    def get_profile_info(self, name: str) -> Dict[str, Any]:
        """Get information about a profile"""
        profile_path = self.profiles_dir / f"{name}.yaml"
        
        if not profile_path.exists():
            raise ValueError(f"Profile '{name}' not found")
        
        stat = profile_path.stat()
        return {
            "name": name,
            "path": str(profile_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "is_current": self.current_profile == name
        }


class ConfigTemplateManager:
    """Manage configuration templates for different games/use cases"""
    
    TEMPLATES = {
        "fps_game": {
            "capture": {
                "fps": 120,
                "backend": "dxcam",
                "buffer_size": 32
            },
            "vision": {
                "detection_model": "yolov8",
                "confidence_threshold": 0.7
            },
            "agents": {
                "default_type": "dqn",
                "learning_rate": 0.0001
            },
            "input": {
                "human_like": False,
                "delay_range": [0.001, 0.005]
            }
        },
        
        "strategy_game": {
            "capture": {
                "fps": 30,
                "backend": "mss",
                "buffer_size": 16
            },
            "vision": {
                "detection_model": "yolov8",
                "confidence_threshold": 0.6,
                "ocr_engine": "tesseract"
            },
            "agents": {
                "default_type": "scripted",
                "max_buffer_size": 5000
            },
            "input": {
                "human_like": True,
                "delay_range": [0.1, 0.3],
                "mouse_speed": 0.5
            }
        },
        
        "mmo_game": {
            "capture": {
                "fps": 60,
                "backend": "dxcam",
                "buffer_size": 64
            },
            "vision": {
                "detection_model": "yolov8",
                "confidence_threshold": 0.5,
                "ocr_engine": "easyocr"
            },
            "agents": {
                "default_type": "ppo",
                "learning_rate": 0.001,
                "batch_size": 64
            },
            "input": {
                "human_like": True,
                "delay_range": [0.05, 0.15],
                "mouse_speed": 1.0
            }
        },
        
        "development": {
            "nexus": {
                "debug": True
            },
            "capture": {
                "fps": 30,
                "buffer_size": 16
            },
            "logging": {
                "level": "DEBUG",
                "console": True
            },
            "performance": {
                "profiling_enabled": True
            },
            "api": {
                "enabled": True,
                "port": 8000
            }
        },
        
        "production": {
            "nexus": {
                "debug": False
            },
            "logging": {
                "level": "INFO",
                "console": False,
                "file": "nexus_prod.log"
            },
            "performance": {
                "profiling_enabled": False,
                "max_cpu_percent": 70,
                "max_memory_mb": 8192
            },
            "api": {
                "enabled": False
            }
        }
    }
    
    @classmethod
    def create_from_template(cls, template_name: str, output_path: Union[str, Path]) -> ConfigManager:
        """Create a configuration from a template"""
        if template_name not in cls.TEMPLATES:
            available = ", ".join(cls.TEMPLATES.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        # Create base config and merge with template
        config = ConfigManager()
        template_config = cls.TEMPLATES[template_name]
        
        for section, values in template_config.items():
            config.update_section(section, values)
        
        # Save to specified path
        config.save(output_path)
        logger.info(f"Created configuration from template '{template_name}': {output_path}")
        
        return config
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List available configuration templates"""
        return list(cls.TEMPLATES.keys())
    
    @classmethod
    def get_template_info(cls, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        if template_name not in cls.TEMPLATES:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = cls.TEMPLATES[template_name]
        return {
            "name": template_name,
            "sections": list(template.keys()),
            "description": cls._get_template_description(template_name),
            "config": template
        }
    
    @classmethod
    def _get_template_description(cls, template_name: str) -> str:
        """Get description for a template"""
        descriptions = {
            "fps_game": "Optimized for fast-paced games requiring high FPS capture and quick reactions",
            "strategy_game": "Configured for strategy games with OCR support and human-like input",
            "mmo_game": "Balanced configuration for MMO games with moderate performance requirements",
            "development": "Development environment with debugging and profiling enabled",
            "production": "Production environment with optimal performance and minimal logging"
        }
        return descriptions.get(template_name, "No description available")