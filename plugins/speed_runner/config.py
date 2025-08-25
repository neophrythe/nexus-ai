"""
Configuration schema for Speed Runner Plugin.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path


class ComparisonMode(Enum):
    """Split comparison modes."""
    PERSONAL_BEST = "pb"
    WORLD_RECORD = "wr"
    SUM_OF_BEST = "sob"
    AVERAGE = "average"
    MEDIAN = "median"
    CUSTOM = "custom"
    NONE = "none"


class TimerFormat(Enum):
    """Timer display formats."""
    STANDARD = "standard"      # 00:00.00
    PRECISE = "precise"        # 00:00.000
    SECONDS = "seconds"        # 123.45
    HOURS = "hours"           # 0:00:00.0


class RouteOptimization(Enum):
    """Route optimization algorithms."""
    NONE = "none"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TWO_OPT = "two_opt"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    A_STAR = "a_star"


@dataclass
class DisplayConfig:
    """Display and overlay configuration."""
    show_overlay: bool = True
    overlay_position: str = "top_right"  # top_left, top_right, bottom_left, bottom_right
    overlay_opacity: float = 0.8
    overlay_scale: float = 1.0
    
    # Timer display
    show_timer: bool = True
    timer_format: TimerFormat = TimerFormat.STANDARD
    timer_precision: int = 2  # Decimal places
    show_pace: bool = True
    pace_colors: Dict[str, int] = field(default_factory=lambda: {
        'ahead': 0x00FF00,      # Green
        'behind': 0xFF0000,      # Red
        'gold': 0xFFD700,        # Gold
        'neutral': 0xFFFFFF      # White
    })
    
    # Splits display
    show_splits: bool = True
    max_splits_shown: int = 5
    show_split_comparison: bool = True
    show_possible_time_save: bool = True
    show_sum_of_best: bool = False
    
    # Route display
    show_route_info: bool = True
    show_checkpoint_progress: bool = True
    show_next_checkpoint: bool = True
    show_checkpoint_distance: bool = True
    
    # Ghost display
    show_ghost: bool = True
    ghost_transparency: float = 0.5
    ghost_color: int = 0x0000FF  # Blue
    show_ghost_trail: bool = False
    
    # Stats display
    show_speed: bool = True
    show_position: bool = False
    show_attempts_count: bool = True
    
    def validate(self) -> List[str]:
        """Validate display configuration."""
        errors = []
        
        valid_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        if self.overlay_position not in valid_positions:
            errors.append(f"overlay_position must be one of {valid_positions}")
        
        if not 0 <= self.overlay_opacity <= 1:
            errors.append("overlay_opacity must be between 0 and 1")
        
        if not 0.5 <= self.overlay_scale <= 2.0:
            errors.append("overlay_scale must be between 0.5 and 2.0")
        
        if not 0 <= self.timer_precision <= 3:
            errors.append("timer_precision must be between 0 and 3")
        
        if not 1 <= self.max_splits_shown <= 20:
            errors.append("max_splits_shown must be between 1 and 20")
        
        return errors


@dataclass
class AutoSplitterConfig:
    """Automatic splitting configuration."""
    enabled: bool = True
    
    # Detection methods
    use_position: bool = True
    use_image_recognition: bool = False
    use_memory_reading: bool = False
    use_audio_cues: bool = False
    
    # Position-based splitting
    checkpoint_radius: float = 50.0
    height_tolerance: float = 10.0
    require_velocity_threshold: bool = False
    velocity_threshold: float = 1.0
    
    # Image recognition
    template_matching_threshold: float = 0.8
    template_scale_range: Tuple[float, float] = (0.8, 1.2)
    max_template_rotation: float = 5.0  # degrees
    
    # Timing
    split_delay: float = 0.0  # Delay before splitting
    debounce_time: float = 1.0  # Prevent double splits
    
    # Checkpoint learning
    learning_mode: bool = False
    auto_detect_checkpoints: bool = True
    stationary_threshold: float = 5.0  # seconds
    position_variance_threshold: float = 100.0
    
    def validate(self) -> List[str]:
        """Validate auto-splitter configuration."""
        errors = []
        
        if not 1 <= self.checkpoint_radius <= 1000:
            errors.append("checkpoint_radius must be between 1 and 1000")
        
        if not 0 <= self.template_matching_threshold <= 1:
            errors.append("template_matching_threshold must be between 0 and 1")
        
        if not 0 <= self.split_delay <= 10:
            errors.append("split_delay must be between 0 and 10 seconds")
        
        return errors


@dataclass
class PracticeConfig:
    """Practice mode configuration."""
    enabled: bool = False
    
    # Save states
    enable_save_states: bool = True
    max_save_slots: int = 10
    auto_save_interval: float = 30.0  # seconds
    save_state_hotkeys: Dict[int, str] = field(default_factory=lambda: {
        1: 'F1', 2: 'F2', 3: 'F3', 4: 'F4', 5: 'F5',
        6: 'F6', 7: 'F7', 8: 'F8', 9: 'F9', 10: 'F10'
    })
    
    # Practice features
    infinite_attempts: bool = True
    disable_timer_save: bool = True
    show_segment_practice: bool = True
    auto_reset_on_completion: bool = False
    
    # IL (Individual Level) mode
    il_mode: bool = False
    il_start_checkpoint: int = 0
    il_end_checkpoint: int = -1
    
    # Ghost options
    practice_against_ghost: bool = True
    ghost_offset: float = 0.0  # Time offset for ghost
    multiple_ghosts: bool = False
    ghost_opacity_range: Tuple[float, float] = (0.3, 0.7)
    
    def validate(self) -> List[str]:
        """Validate practice configuration."""
        errors = []
        
        if not 1 <= self.max_save_slots <= 20:
            errors.append("max_save_slots must be between 1 and 20")
        
        if self.auto_save_interval < 0:
            errors.append("auto_save_interval must be positive")
        
        return errors


@dataclass
class RouteConfig:
    """Route management configuration."""
    # Route settings
    default_route: Optional[str] = None
    auto_load_route: bool = True
    
    # Categories
    categories: List[str] = field(default_factory=lambda: [
        "Any%", "100%", "Glitchless", "All Bosses", "Low%"
    ])
    default_category: str = "Any%"
    
    # Route optimization
    optimization_algorithm: RouteOptimization = RouteOptimization.NEAREST_NEIGHBOR
    optimize_on_creation: bool = False
    max_optimization_time: float = 5.0  # seconds
    
    # Route recording
    record_optimal_path: bool = True
    path_smoothing: bool = True
    path_simplification_tolerance: float = 5.0
    
    # Checkpoint settings
    default_checkpoint_radius: float = 50.0
    checkpoint_name_format: str = "Checkpoint {index}"
    auto_generate_splits: bool = True
    
    # Import/Export
    export_format: str = "json"  # json, csv, xml
    include_metadata: bool = True
    compress_exports: bool = False
    
    def validate(self) -> List[str]:
        """Validate route configuration."""
        errors = []
        
        if not self.categories:
            errors.append("At least one category must be defined")
        
        if self.default_category not in self.categories:
            errors.append("default_category must be in categories list")
        
        valid_formats = ['json', 'csv', 'xml']
        if self.export_format not in valid_formats:
            errors.append(f"export_format must be one of {valid_formats}")
        
        return errors


@dataclass
class AnalyticsConfig:
    """Analytics and statistics configuration."""
    enabled: bool = True
    
    # Data collection
    track_attempts: bool = True
    track_resets: bool = True
    track_gold_splits: bool = True
    track_pace: bool = True
    track_consistency: bool = True
    
    # Analysis
    calculate_success_rate: bool = True
    calculate_reset_patterns: bool = True
    identify_problem_segments: bool = True
    suggest_route_improvements: bool = True
    
    # History
    max_history_size: int = 1000
    compress_old_runs: bool = True
    archive_after_days: int = 30
    
    # Reporting
    generate_reports: bool = True
    report_interval: str = "weekly"  # daily, weekly, monthly
    include_graphs: bool = True
    export_to_csv: bool = False
    
    # Integration
    upload_to_leaderboard: bool = False
    leaderboard_api_url: Optional[str] = None
    sync_with_splits_io: bool = False
    
    def validate(self) -> List[str]:
        """Validate analytics configuration."""
        errors = []
        
        if self.max_history_size < 1:
            errors.append("max_history_size must be at least 1")
        
        valid_intervals = ['daily', 'weekly', 'monthly']
        if self.report_interval not in valid_intervals:
            errors.append(f"report_interval must be one of {valid_intervals}")
        
        return errors


@dataclass
class PositionTrackingConfig:
    """Position tracking configuration."""
    enabled: bool = True
    
    # Tracking method
    method: str = "memory"  # memory, ocr, image_recognition, api
    update_rate: float = 60.0  # Hz
    
    # Memory reading (if applicable)
    process_name: Optional[str] = None
    position_offsets: Optional[List[int]] = None
    coordinate_system: str = "xyz"  # xyz, xy, polar
    
    # OCR settings
    ocr_region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    ocr_preprocessing: List[str] = field(default_factory=lambda: [
        'grayscale', 'threshold', 'denoise'
    ])
    
    # Coordinate transformation
    scale_factor: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler angles
    
    # Filtering
    use_kalman_filter: bool = True
    smoothing_window: int = 5
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Velocity calculation
    calculate_velocity: bool = True
    velocity_smoothing: int = 3
    max_expected_speed: float = 1000.0  # units per second
    
    def validate(self) -> List[str]:
        """Validate position tracking configuration."""
        errors = []
        
        valid_methods = ['memory', 'ocr', 'image_recognition', 'api']
        if self.method not in valid_methods:
            errors.append(f"method must be one of {valid_methods}")
        
        if not 1 <= self.update_rate <= 240:
            errors.append("update_rate must be between 1 and 240 Hz")
        
        if self.smoothing_window < 1:
            errors.append("smoothing_window must be at least 1")
        
        return errors


@dataclass
class SpeedRunnerConfig:
    """Complete Speed Runner configuration."""
    
    # General settings
    enabled: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    display: DisplayConfig = field(default_factory=DisplayConfig)
    auto_splitter: AutoSplitterConfig = field(default_factory=AutoSplitterConfig)
    practice: PracticeConfig = field(default_factory=PracticeConfig)
    routes: RouteConfig = field(default_factory=RouteConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    position_tracking: PositionTrackingConfig = field(default_factory=PositionTrackingConfig)
    
    # Timer settings
    comparison_mode: ComparisonMode = ComparisonMode.PERSONAL_BEST
    start_on_reset: bool = True
    auto_start: bool = False
    auto_reset: bool = False
    
    # Hotkeys
    hotkeys: Dict[str, str] = field(default_factory=lambda: {
        'start_split': 'space',
        'reset': 'r',
        'skip_split': 's',
        'undo_split': 'backspace',
        'pause': 'p',
        'toggle_info': 'tab'
    })
    
    # Data storage
    data_directory: str = "~/.nexus/speedrun_data"
    backup_enabled: bool = True
    backup_interval: int = 24  # hours
    max_backups: int = 10
    
    # Network
    enable_race_mode: bool = False
    race_room_url: Optional[str] = None
    spectator_mode: bool = False
    
    @classmethod
    def from_file(cls, filepath: str) -> 'SpeedRunnerConfig':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load general settings
        for key in ['enabled', 'debug_mode', 'log_level', 'comparison_mode',
                    'data_directory', 'backup_enabled']:
            if key in data:
                if key == 'comparison_mode':
                    setattr(config, key, ComparisonMode(data[key]))
                else:
                    setattr(config, key, data[key])
        
        # Load component configs
        if 'display' in data:
            for key, value in data['display'].items():
                if hasattr(config.display, key):
                    setattr(config.display, key, value)
        
        if 'auto_splitter' in data:
            for key, value in data['auto_splitter'].items():
                if hasattr(config.auto_splitter, key):
                    setattr(config.auto_splitter, key, value)
        
        if 'practice' in data:
            for key, value in data['practice'].items():
                if hasattr(config.practice, key):
                    setattr(config.practice, key, value)
        
        if 'routes' in data:
            for key, value in data['routes'].items():
                if hasattr(config.routes, key):
                    if key == 'optimization_algorithm':
                        setattr(config.routes, key, RouteOptimization(value))
                    else:
                        setattr(config.routes, key, value)
        
        if 'analytics' in data:
            for key, value in data['analytics'].items():
                if hasattr(config.analytics, key):
                    setattr(config.analytics, key, value)
        
        if 'position_tracking' in data:
            for key, value in data['position_tracking'].items():
                if hasattr(config.position_tracking, key):
                    setattr(config.position_tracking, key, value)
        
        if 'hotkeys' in data:
            config.hotkeys.update(data['hotkeys'])
        
        return config
    
    def to_file(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'enabled': self.enabled,
            'debug_mode': self.debug_mode,
            'log_level': self.log_level,
            'comparison_mode': self.comparison_mode.value,
            'data_directory': self.data_directory,
            'backup_enabled': self.backup_enabled,
            'backup_interval': self.backup_interval,
            'display': {
                'show_overlay': self.display.show_overlay,
                'overlay_position': self.display.overlay_position,
                'show_timer': self.display.show_timer,
                'timer_format': self.display.timer_format.value,
                'show_splits': self.display.show_splits,
                'max_splits_shown': self.display.max_splits_shown
            },
            'auto_splitter': {
                'enabled': self.auto_splitter.enabled,
                'use_position': self.auto_splitter.use_position,
                'checkpoint_radius': self.auto_splitter.checkpoint_radius,
                'learning_mode': self.auto_splitter.learning_mode
            },
            'practice': {
                'enabled': self.practice.enabled,
                'enable_save_states': self.practice.enable_save_states,
                'max_save_slots': self.practice.max_save_slots
            },
            'routes': {
                'default_route': self.routes.default_route,
                'categories': self.routes.categories,
                'optimization_algorithm': self.routes.optimization_algorithm.value
            },
            'analytics': {
                'enabled': self.analytics.enabled,
                'track_attempts': self.analytics.track_attempts,
                'max_history_size': self.analytics.max_history_size
            },
            'position_tracking': {
                'enabled': self.position_tracking.enabled,
                'method': self.position_tracking.method,
                'update_rate': self.position_tracking.update_rate
            },
            'hotkeys': self.hotkeys
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        # Validate component configs
        errors.extend(self.display.validate())
        errors.extend(self.auto_splitter.validate())
        errors.extend(self.practice.validate())
        errors.extend(self.routes.validate())
        errors.extend(self.analytics.validate())
        errors.extend(self.position_tracking.validate())
        
        # Validate general settings
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        if self.backup_interval < 1:
            errors.append("backup_interval must be at least 1 hour")
        
        if self.max_backups < 1:
            errors.append("max_backups must be at least 1")
        
        return errors
    
    def apply_preset(self, preset: str):
        """Apply configuration preset."""
        presets = {
            'competitive': {
                'display': {
                    'timer_precision': 3,
                    'show_possible_time_save': True,
                    'show_sum_of_best': True
                },
                'auto_splitter': {
                    'enabled': True,
                    'use_position': True
                },
                'analytics': {
                    'enabled': True,
                    'identify_problem_segments': True
                }
            },
            'practice': {
                'practice': {
                    'enabled': True,
                    'enable_save_states': True,
                    'infinite_attempts': True
                },
                'display': {
                    'show_ghost': True,
                    'show_segment_practice': True
                }
            },
            'casual': {
                'display': {
                    'timer_precision': 1,
                    'show_splits': False
                },
                'auto_splitter': {
                    'enabled': False
                },
                'analytics': {
                    'enabled': False
                }
            },
            'marathon': {
                'display': {
                    'timer_format': TimerFormat.HOURS,
                    'show_pace': True
                },
                'practice': {
                    'enabled': False
                },
                'analytics': {
                    'track_consistency': True
                }
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        preset_data = presets[preset]
        
        # Apply display settings
        if 'display' in preset_data:
            for key, value in preset_data['display'].items():
                if isinstance(value, str) and key == 'timer_format':
                    setattr(self.display, key, TimerFormat(value))
                else:
                    setattr(self.display, key, value)
        
        # Apply other component settings
        for component in ['auto_splitter', 'practice', 'analytics']:
            if component in preset_data:
                component_obj = getattr(self, component)
                for key, value in preset_data[component].items():
                    setattr(component_obj, key, value)