"""
Configuration schema for Game State Logger Plugin.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
from pathlib import Path


class LogLevel(Enum):
    """Logging detail levels."""
    MINIMAL = "minimal"      # Only critical events
    STANDARD = "standard"    # Normal logging
    DETAILED = "detailed"    # Detailed state tracking
    VERBOSE = "verbose"      # Everything including frames


class ExportFormat(Enum):
    """Data export formats."""
    JSON = "json"
    CSV = "csv"
    BINARY = "binary"        # Pickle format
    PARQUET = "parquet"      # Apache Parquet
    SQL = "sql"              # SQLite database
    HDF5 = "hdf5"           # Hierarchical Data Format


class CompressionType(Enum):
    """Compression methods."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class EventConfig:
    """Event logging configuration."""
    enabled: bool = True
    
    # Event types to log
    log_combat: bool = True
    log_movement: bool = True
    log_interaction: bool = True
    log_progression: bool = True
    log_system: bool = True
    log_custom: bool = True
    
    # Event filtering
    event_whitelist: List[str] = field(default_factory=list)  # If set, only log these
    event_blacklist: List[str] = field(default_factory=lambda: [
        'mouse_move', 'camera_rotate'  # High frequency events
    ])
    
    # Event categorization
    categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'combat': ['damage', 'kill', 'death', 'heal', 'buff', 'debuff'],
        'movement': ['jump', 'dash', 'teleport', 'fall', 'climb', 'swim'],
        'interaction': ['pickup', 'use', 'talk', 'buy', 'sell', 'craft'],
        'progression': ['level_up', 'achievement', 'quest_complete', 'unlock'],
        'system': ['save', 'load', 'pause', 'menu', 'settings_change']
    })
    
    # Event metadata
    include_timestamp: bool = True
    include_frame_number: bool = True
    include_stack_trace: bool = False
    include_game_time: bool = True
    
    # Batching
    batch_events: bool = True
    batch_size: int = 100
    batch_timeout: float = 1.0  # seconds
    
    def validate(self) -> List[str]:
        """Validate event configuration."""
        errors = []
        
        if self.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        if self.batch_timeout < 0:
            errors.append("batch_timeout must be positive")
        
        # Check for conflicting whitelist/blacklist
        if self.event_whitelist and self.event_blacklist:
            overlap = set(self.event_whitelist) & set(self.event_blacklist)
            if overlap:
                errors.append(f"Events in both whitelist and blacklist: {overlap}")
        
        return errors


@dataclass
class StateConfig:
    """State logging configuration."""
    enabled: bool = True
    
    # Snapshot settings
    snapshot_interval: float = 1.0  # seconds
    snapshot_on_change: bool = True  # Snapshot when significant change detected
    change_threshold: float = 0.1  # Threshold for significant change
    
    # State components
    log_player_state: bool = True
    log_world_state: bool = True
    log_ui_state: bool = True
    log_ai_state: bool = False
    log_physics_state: bool = False
    
    # Player state details
    player_state_fields: List[str] = field(default_factory=lambda: [
        'position', 'rotation', 'velocity', 'health', 'mana', 'stamina',
        'inventory', 'equipment', 'stats', 'buffs', 'debuffs'
    ])
    
    # World state details
    world_state_fields: List[str] = field(default_factory=lambda: [
        'time', 'weather', 'entities', 'npcs', 'objects', 'triggers'
    ])
    
    # State diffing
    enable_state_diff: bool = True  # Only log changes
    diff_ignore_fields: List[str] = field(default_factory=lambda: [
        'timestamp', 'frame_number'  # Always changing fields
    ])
    
    # Compression
    compress_states: bool = True
    state_compression: CompressionType = CompressionType.LZ4
    
    def validate(self) -> List[str]:
        """Validate state configuration."""
        errors = []
        
        if self.snapshot_interval < 0.01:
            errors.append("snapshot_interval must be at least 0.01 seconds")
        
        if not 0 <= self.change_threshold <= 1:
            errors.append("change_threshold must be between 0 and 1")
        
        return errors


@dataclass
class ActionConfig:
    """Action logging configuration."""
    enabled: bool = True
    
    # Action types
    log_input_actions: bool = True     # Keyboard, mouse, gamepad
    log_game_actions: bool = True      # In-game actions
    log_ui_actions: bool = True        # Menu interactions
    log_command_actions: bool = True   # Console commands
    
    # Input logging detail
    log_key_press: bool = True
    log_key_release: bool = False
    log_mouse_click: bool = True
    log_mouse_move: bool = False  # High frequency
    log_gamepad_input: bool = True
    
    # Action metadata
    include_input_state: bool = True   # Full input state at action time
    include_result: bool = True        # Action result/outcome
    include_context: bool = True       # Game context
    
    # Replay support
    record_for_replay: bool = True
    replay_timing_precision: float = 0.001  # seconds
    
    def validate(self) -> List[str]:
        """Validate action configuration."""
        errors = []
        
        if self.replay_timing_precision < 0.001:
            errors.append("replay_timing_precision must be at least 0.001 seconds")
        
        return errors


@dataclass
class FrameConfig:
    """Frame logging configuration."""
    enabled: bool = False  # Disabled by default (storage intensive)
    
    # Frame capture settings
    capture_interval: int = 60  # Capture every N frames
    capture_on_event: bool = True  # Capture frame on important events
    capture_events: List[str] = field(default_factory=lambda: [
        'death', 'achievement', 'boss_defeat', 'level_complete'
    ])
    
    # Frame processing
    resize_frames: bool = True
    frame_size: Tuple[int, int] = (640, 360)  # Resized dimensions
    frame_quality: int = 70  # JPEG quality (1-100)
    frame_format: str = "jpg"  # jpg, png, webp
    
    # Storage
    max_frames_in_memory: int = 300  # 5 seconds at 60 FPS
    frames_per_file: int = 1800  # 30 seconds at 60 FPS
    
    # Video generation
    generate_video: bool = False
    video_fps: int = 30
    video_codec: str = "h264"
    video_bitrate: str = "2M"
    
    def validate(self) -> List[str]:
        """Validate frame configuration."""
        errors = []
        
        if self.capture_interval < 1:
            errors.append("capture_interval must be at least 1")
        
        if not 1 <= self.frame_quality <= 100:
            errors.append("frame_quality must be between 1 and 100")
        
        valid_formats = ['jpg', 'png', 'webp']
        if self.frame_format not in valid_formats:
            errors.append(f"frame_format must be one of {valid_formats}")
        
        return errors


@dataclass
class StorageConfig:
    """Storage and export configuration."""
    # Directories
    data_directory: str = "~/.nexus/game_logs"
    temp_directory: Optional[str] = None  # Use system temp if None
    
    # Export settings
    export_format: ExportFormat = ExportFormat.JSON
    compression: CompressionType = CompressionType.GZIP
    compression_level: int = 6  # 1-9 for gzip/bzip2
    
    # Auto-export
    auto_export: bool = True
    export_interval: float = 60.0  # seconds
    export_on_buffer_full: bool = True
    buffer_size_threshold: float = 0.8  # Export when buffer 80% full
    
    # File management
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    split_large_files: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Retention
    retention_days: int = 30
    archive_old_logs: bool = True
    delete_after_archive: bool = False
    
    # Database settings (for SQL export)
    database_path: Optional[str] = None
    database_connection_string: Optional[str] = None
    create_indexes: bool = True
    
    def validate(self) -> List[str]:
        """Validate storage configuration."""
        errors = []
        
        if self.export_interval < 1:
            errors.append("export_interval must be at least 1 second")
        
        if not 0 < self.buffer_size_threshold <= 1:
            errors.append("buffer_size_threshold must be between 0 and 1")
        
        if not 1 <= self.compression_level <= 9:
            errors.append("compression_level must be between 1 and 9")
        
        if self.retention_days < 0:
            errors.append("retention_days must be non-negative")
        
        return errors


@dataclass
class AnalysisConfig:
    """Real-time analysis configuration."""
    enabled: bool = True
    
    # Pattern detection
    detect_patterns: bool = True
    pattern_window: float = 60.0  # seconds
    min_pattern_occurrences: int = 3
    
    # Anomaly detection
    detect_anomalies: bool = True
    anomaly_threshold: float = 3.0  # Standard deviations
    baseline_window: float = 300.0  # 5 minutes
    
    # Performance analysis
    track_performance: bool = True
    performance_metrics: List[str] = field(default_factory=lambda: [
        'actions_per_minute', 'events_per_second', 'state_changes_per_minute'
    ])
    
    # Behavior analysis
    track_behavior: bool = True
    behavior_categories: List[str] = field(default_factory=lambda: [
        'aggressive', 'defensive', 'exploratory', 'completionist'
    ])
    
    # Statistics
    calculate_statistics: bool = True
    statistics_interval: float = 10.0  # seconds
    rolling_window_size: int = 100  # samples
    
    def validate(self) -> List[str]:
        """Validate analysis configuration."""
        errors = []
        
        if self.pattern_window < 1:
            errors.append("pattern_window must be at least 1 second")
        
        if self.min_pattern_occurrences < 2:
            errors.append("min_pattern_occurrences must be at least 2")
        
        if self.anomaly_threshold < 0:
            errors.append("anomaly_threshold must be positive")
        
        return errors


@dataclass
class ReplayConfig:
    """Replay system configuration."""
    enabled: bool = True
    
    # Replay settings
    enable_replay: bool = True
    replay_buffer_size: int = 10000  # events
    replay_speed_options: List[float] = field(default_factory=lambda: [
        0.25, 0.5, 1.0, 2.0, 4.0, 8.0
    ])
    default_replay_speed: float = 1.0
    
    # Replay controls
    enable_seek: bool = True
    enable_pause: bool = True
    enable_step_forward: bool = True
    enable_step_backward: bool = True
    
    # Replay visualization
    show_replay_overlay: bool = True
    show_timeline: bool = True
    show_event_markers: bool = True
    highlight_important_events: bool = True
    
    # Export replay
    export_replay_video: bool = False
    replay_video_quality: str = "high"  # low, medium, high
    include_audio: bool = False
    
    def validate(self) -> List[str]:
        """Validate replay configuration."""
        errors = []
        
        if self.replay_buffer_size < 100:
            errors.append("replay_buffer_size must be at least 100")
        
        if self.default_replay_speed not in self.replay_speed_options:
            errors.append("default_replay_speed must be in replay_speed_options")
        
        valid_qualities = ['low', 'medium', 'high']
        if self.replay_video_quality not in valid_qualities:
            errors.append(f"replay_video_quality must be one of {valid_qualities}")
        
        return errors


@dataclass
class GameStateLoggerConfig:
    """Complete Game State Logger configuration."""
    
    # General settings
    enabled: bool = True
    log_level: LogLevel = LogLevel.STANDARD
    debug_mode: bool = False
    
    # Component configurations
    events: EventConfig = field(default_factory=EventConfig)
    states: StateConfig = field(default_factory=StateConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    
    # Buffer settings
    event_buffer_size: int = 10000
    state_buffer_size: int = 1000
    action_buffer_size: int = 5000
    frame_buffer_size: int = 300
    
    # Performance settings
    max_cpu_usage: float = 20.0  # Maximum CPU % for logging
    max_memory_usage: int = 500 * 1024 * 1024  # 500MB
    throttle_on_high_load: bool = True
    
    # Session settings
    auto_start_session: bool = True
    session_name_format: str = "session_{timestamp}"
    include_system_info: bool = True
    include_game_info: bool = True
    
    # Privacy settings
    anonymize_player_data: bool = False
    exclude_sensitive_data: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        'password', 'token', 'key', 'secret'
    ])
    
    @classmethod
    def from_file(cls, filepath: str) -> 'GameStateLoggerConfig':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load general settings
        if 'log_level' in data:
            config.log_level = LogLevel(data['log_level'])
        
        for key in ['enabled', 'debug_mode', 'event_buffer_size',
                    'state_buffer_size', 'action_buffer_size']:
            if key in data:
                setattr(config, key, data[key])
        
        # Load component configs
        if 'events' in data:
            for key, value in data['events'].items():
                if hasattr(config.events, key):
                    setattr(config.events, key, value)
        
        if 'states' in data:
            for key, value in data['states'].items():
                if hasattr(config.states, key):
                    if key == 'state_compression':
                        setattr(config.states, key, CompressionType(value))
                    else:
                        setattr(config.states, key, value)
        
        if 'storage' in data:
            for key, value in data['storage'].items():
                if hasattr(config.storage, key):
                    if key == 'export_format':
                        setattr(config.storage, key, ExportFormat(value))
                    elif key == 'compression':
                        setattr(config.storage, key, CompressionType(value))
                    else:
                        setattr(config.storage, key, value)
        
        return config
    
    def to_file(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'enabled': self.enabled,
            'log_level': self.log_level.value,
            'debug_mode': self.debug_mode,
            'event_buffer_size': self.event_buffer_size,
            'state_buffer_size': self.state_buffer_size,
            'action_buffer_size': self.action_buffer_size,
            'events': {
                'enabled': self.events.enabled,
                'batch_events': self.events.batch_events,
                'batch_size': self.events.batch_size
            },
            'states': {
                'enabled': self.states.enabled,
                'snapshot_interval': self.states.snapshot_interval,
                'state_compression': self.states.state_compression.value
            },
            'storage': {
                'data_directory': self.storage.data_directory,
                'export_format': self.storage.export_format.value,
                'compression': self.storage.compression.value,
                'auto_export': self.storage.auto_export
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        # Validate component configs
        errors.extend(self.events.validate())
        errors.extend(self.states.validate())
        errors.extend(self.actions.validate())
        errors.extend(self.frames.validate())
        errors.extend(self.storage.validate())
        errors.extend(self.analysis.validate())
        errors.extend(self.replay.validate())
        
        # Validate buffer sizes
        if self.event_buffer_size < 100:
            errors.append("event_buffer_size must be at least 100")
        
        if self.state_buffer_size < 10:
            errors.append("state_buffer_size must be at least 10")
        
        # Validate performance settings
        if not 0 < self.max_cpu_usage <= 100:
            errors.append("max_cpu_usage must be between 0 and 100")
        
        if self.max_memory_usage < 1024 * 1024:  # 1MB minimum
            errors.append("max_memory_usage must be at least 1MB")
        
        return errors
    
    def apply_preset(self, preset: str):
        """Apply configuration preset."""
        presets = {
            'minimal': {
                'log_level': LogLevel.MINIMAL,
                'events': {'enabled': True, 'batch_events': True},
                'states': {'enabled': False},
                'frames': {'enabled': False},
                'analysis': {'enabled': False}
            },
            'standard': {
                'log_level': LogLevel.STANDARD,
                'events': {'enabled': True},
                'states': {'enabled': True, 'snapshot_interval': 1.0},
                'frames': {'enabled': False},
                'analysis': {'enabled': True}
            },
            'detailed': {
                'log_level': LogLevel.DETAILED,
                'events': {'enabled': True, 'include_stack_trace': True},
                'states': {'enabled': True, 'snapshot_interval': 0.5},
                'actions': {'enabled': True, 'log_input_actions': True},
                'analysis': {'enabled': True, 'detect_patterns': True}
            },
            'training': {
                'log_level': LogLevel.VERBOSE,
                'events': {'enabled': True},
                'states': {'enabled': True, 'snapshot_interval': 0.1},
                'actions': {'enabled': True, 'record_for_replay': True},
                'frames': {'enabled': True, 'capture_interval': 30},
                'storage': {'export_format': ExportFormat.HDF5}
            },
            'replay': {
                'events': {'enabled': True},
                'actions': {'enabled': True, 'record_for_replay': True},
                'frames': {'enabled': True, 'capture_interval': 60},
                'replay': {'enabled': True, 'export_replay_video': True}
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        preset_data = presets[preset]
        
        # Apply log level
        if 'log_level' in preset_data:
            self.log_level = preset_data['log_level']
        
        # Apply component settings
        for component in ['events', 'states', 'actions', 'frames', 'storage', 'analysis', 'replay']:
            if component in preset_data:
                component_obj = getattr(self, component)
                for key, value in preset_data[component].items():
                    if isinstance(value, str) and key == 'export_format':
                        setattr(component_obj, key, ExportFormat(value))
                    else:
                        setattr(component_obj, key, value)