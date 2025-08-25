"""
Configuration schema for Auto-Aim Assistant Plugin.

IMPORTANT: This plugin is for AI training and research purposes only.
Using aim assistance in online games violates terms of service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from pathlib import Path


class AimMode(Enum):
    """Aim assistance modes."""
    ASSIST = "assist"      # Smooth aim assistance
    LOCK = "lock"          # Target locking
    TRAINING = "training"  # Training mode (no actual input)
    DISABLED = "disabled"  # Completely disabled


class AimZone(Enum):
    """Target aim zones."""
    HEAD = "head"      # Aim for head (top 20%)
    NECK = "neck"      # Aim for neck (top 30%)
    CHEST = "chest"    # Aim for chest (center)
    BODY = "body"      # Aim for body center
    AUTO = "auto"      # Automatically choose best zone


class DetectionMethod(Enum):
    """Target detection methods."""
    COLOR = "color"          # Color-based detection
    MODEL = "model"          # ML model detection
    HYBRID = "hybrid"        # Both color and model
    TEMPLATE = "template"    # Template matching


@dataclass
class DetectionConfig:
    """Target detection configuration."""
    method: DetectionMethod = DetectionMethod.COLOR
    model_path: Optional[str] = None
    model_type: str = "yolov5"  # yolov5, yolov8, custom
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    
    # Color detection settings
    target_colors: List[str] = field(default_factory=lambda: ['red'])
    color_ranges: Dict[str, List[Tuple[int, int, int]]] = field(default_factory=lambda: {
        'red': [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
        'green': [(40, 100, 100), (80, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
        'yellow': [(20, 100, 100), (40, 255, 255)],
        'purple': [(130, 100, 100), (170, 255, 255)]
    })
    
    # Template matching settings
    template_paths: List[str] = field(default_factory=list)
    template_scale_range: Tuple[float, float] = (0.8, 1.2)
    template_threshold: float = 0.8
    
    # Detection area
    detection_area: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    min_target_size: Tuple[int, int] = (20, 20)  # min width, height
    max_target_size: Tuple[int, int] = (500, 500)  # max width, height
    
    def validate(self) -> List[str]:
        """Validate detection configuration."""
        errors = []
        
        if not 0 <= self.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.nms_threshold <= 1:
            errors.append("nms_threshold must be between 0 and 1")
        
        if self.method in [DetectionMethod.MODEL, DetectionMethod.HYBRID]:
            if not self.model_path:
                errors.append("model_path required for model-based detection")
            elif not Path(self.model_path).exists():
                errors.append(f"Model file not found: {self.model_path}")
        
        if self.method == DetectionMethod.TEMPLATE:
            if not self.template_paths:
                errors.append("template_paths required for template matching")
        
        return errors


@dataclass
class AimConfig:
    """Aim assistance configuration."""
    mode: AimMode = AimMode.TRAINING
    zone: AimZone = AimZone.CHEST
    
    # Aim settings
    fov_radius: int = 100  # Field of view in pixels
    smoothness: float = 5.0  # 1-20, higher = smoother
    prediction_enabled: bool = True
    prediction_time: float = 0.1  # Seconds to predict ahead
    
    # Movement settings
    max_speed: float = 500.0  # Max pixels per second
    acceleration: float = 1000.0  # Pixels per second squared
    deceleration_distance: int = 20  # Start slowing down within this distance
    
    # Humanization
    humanize: bool = True
    reaction_time: float = 0.15  # Human reaction time simulation
    overshoot_amount: float = 0.1  # Amount to overshoot target (0-1)
    shake_amount: float = 2.0  # Natural hand shake simulation
    
    # Target selection
    target_switch_delay: float = 0.5  # Delay before switching targets
    prefer_visible_targets: bool = True
    prefer_stationary_targets: bool = False
    
    # Activation
    activation_key: str = "right_mouse"  # Key to activate aim
    toggle_mode: bool = False  # Toggle vs hold
    auto_fire: bool = False  # Auto fire when on target
    auto_fire_delay: float = 0.1  # Delay before auto firing
    
    def validate(self) -> List[str]:
        """Validate aim configuration."""
        errors = []
        
        if not 10 <= self.fov_radius <= 500:
            errors.append("fov_radius must be between 10 and 500 pixels")
        
        if not 1 <= self.smoothness <= 20:
            errors.append("smoothness must be between 1 and 20")
        
        if not 0 <= self.prediction_time <= 1:
            errors.append("prediction_time must be between 0 and 1 second")
        
        if not 0 <= self.reaction_time <= 1:
            errors.append("reaction_time must be between 0 and 1 second")
        
        if self.auto_fire and self.mode != AimMode.TRAINING:
            errors.append("auto_fire only allowed in training mode")
        
        return errors


@dataclass
class RecoilConfig:
    """Recoil compensation configuration."""
    enabled: bool = False
    weapon: str = "ak47"  # Current weapon
    
    # Recoil patterns (x, y) offsets per shot
    patterns: Dict[str, List[Tuple[float, float]]] = field(default_factory=lambda: {
        'ak47': [(0, -3), (0, -5), (0, -7), (0, -8), (1, -8), (2, -7), (2, -6), (1, -5)],
        'm4a4': [(0, -2), (0, -3), (0, -4), (0, -4), (1, -4), (1, -3), (0, -3), (0, -2)],
        'm4a1': [(0, -2), (0, -3), (0, -3), (0, -3), (0, -3), (0, -2), (0, -2), (0, -2)],
        'mp9': [(0, -1), (0, -2), (1, -2), (1, -2), (0, -2), (-1, -2), (-1, -1), (0, -1)],
        'awp': [(0, 0)],  # No recoil pattern for AWP
        'deagle': [(0, -5), (0, -3), (0, -2)],
        'glock': [(0, -1), (0, -1), (0, -2), (0, -2), (1, -2), (0, -1)]
    })
    
    # Compensation settings
    multiplier: float = 1.0  # Strength multiplier
    randomness: float = 0.1  # Add randomness to pattern (0-1)
    delay_between_shots: float = 0.1  # Seconds between shots
    reset_time: float = 0.5  # Time to reset pattern after stopping
    
    # Advanced settings
    adaptive: bool = False  # Adapt to detected pattern
    learning_rate: float = 0.1  # How fast to adapt
    
    def validate(self) -> List[str]:
        """Validate recoil configuration."""
        errors = []
        
        if not 0 <= self.multiplier <= 2:
            errors.append("multiplier must be between 0 and 2")
        
        if not 0 <= self.randomness <= 1:
            errors.append("randomness must be between 0 and 1")
        
        if self.weapon not in self.patterns:
            errors.append(f"Unknown weapon: {self.weapon}")
        
        return errors


@dataclass
class TrainingConfig:
    """AI training configuration."""
    enabled: bool = True
    record_data: bool = True
    data_output_path: str = "./training_data"
    
    # Data collection
    record_frames: bool = True
    record_targets: bool = True
    record_actions: bool = True
    record_performance: bool = True
    
    # Training settings
    batch_size: int = 32
    save_interval: int = 100  # Save data every N frames
    max_samples: int = 10000  # Maximum samples to keep
    
    # Augmentation
    augment_data: bool = True
    augmentation_types: List[str] = field(default_factory=lambda: [
        'flip', 'rotate', 'scale', 'noise', 'brightness'
    ])
    
    # Labels
    label_format: str = "yolo"  # yolo, coco, pascal_voc
    class_names: List[str] = field(default_factory=lambda: [
        'enemy', 'teammate', 'hostage', 'bomb'
    ])
    
    def validate(self) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        if self.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        if self.save_interval < 1:
            errors.append("save_interval must be at least 1")
        
        valid_formats = ['yolo', 'coco', 'pascal_voc']
        if self.label_format not in valid_formats:
            errors.append(f"label_format must be one of {valid_formats}")
        
        return errors


@dataclass
class AutoAimConfig:
    """Complete Auto-Aim Assistant configuration."""
    
    # General settings
    enabled: bool = False  # Master switch
    debug_mode: bool = True
    show_overlay: bool = True
    
    # Screen settings
    screen_width: int = 1920
    screen_height: int = 1080
    game_window: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    
    # Component configurations
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    recoil: RecoilConfig = field(default_factory=RecoilConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Performance settings
    fps_limit: int = 60  # Process at most this many FPS
    cpu_limit: int = 50  # Maximum CPU usage percent
    gpu_enabled: bool = True  # Use GPU if available
    
    # Safety settings
    require_confirmation: bool = True  # Require confirmation to enable
    training_only: bool = True  # Only work in training mode
    watermark: bool = True  # Add "TRAINING MODE" watermark
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/auto_aim.log"
    log_actions: bool = True
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AutoAimConfig':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Parse sub-configurations
        config = cls()
        
        # General settings
        for key in ['enabled', 'debug_mode', 'show_overlay', 'screen_width',
                    'screen_height', 'fps_limit', 'training_only']:
            if key in data:
                setattr(config, key, data[key])
        
        # Detection config
        if 'detection' in data:
            detection_data = data['detection']
            config.detection = DetectionConfig(
                method=DetectionMethod(detection_data.get('method', 'color')),
                model_path=detection_data.get('model_path'),
                confidence_threshold=detection_data.get('confidence_threshold', 0.5),
                target_colors=detection_data.get('target_colors', ['red'])
            )
        
        # Aim config
        if 'aim' in data:
            aim_data = data['aim']
            config.aim = AimConfig(
                mode=AimMode(aim_data.get('mode', 'training')),
                zone=AimZone(aim_data.get('zone', 'chest')),
                fov_radius=aim_data.get('fov_radius', 100),
                smoothness=aim_data.get('smoothness', 5.0)
            )
        
        # Recoil config
        if 'recoil' in data:
            recoil_data = data['recoil']
            config.recoil = RecoilConfig(
                enabled=recoil_data.get('enabled', False),
                weapon=recoil_data.get('weapon', 'ak47'),
                multiplier=recoil_data.get('multiplier', 1.0)
            )
        
        # Training config
        if 'training' in data:
            training_data = data['training']
            config.training = TrainingConfig(
                enabled=training_data.get('enabled', True),
                record_data=training_data.get('record_data', True),
                data_output_path=training_data.get('data_output_path', './training_data')
            )
        
        return config
    
    def to_file(self, filepath: str):
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'enabled': self.enabled,
            'debug_mode': self.debug_mode,
            'show_overlay': self.show_overlay,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'fps_limit': self.fps_limit,
            'training_only': self.training_only,
            'detection': {
                'method': self.detection.method.value,
                'model_path': self.detection.model_path,
                'confidence_threshold': self.detection.confidence_threshold,
                'target_colors': self.detection.target_colors
            },
            'aim': {
                'mode': self.aim.mode.value,
                'zone': self.aim.zone.value,
                'fov_radius': self.aim.fov_radius,
                'smoothness': self.aim.smoothness,
                'prediction_enabled': self.aim.prediction_enabled
            },
            'recoil': {
                'enabled': self.recoil.enabled,
                'weapon': self.recoil.weapon,
                'multiplier': self.recoil.multiplier
            },
            'training': {
                'enabled': self.training.enabled,
                'record_data': self.training.record_data,
                'data_output_path': self.training.data_output_path
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate complete configuration."""
        errors = []
        
        # Validate screen settings
        if self.screen_width < 640 or self.screen_height < 480:
            errors.append("Screen resolution too small")
        
        if self.fps_limit < 1 or self.fps_limit > 240:
            errors.append("fps_limit must be between 1 and 240")
        
        # Validate sub-configurations
        errors.extend(self.detection.validate())
        errors.extend(self.aim.validate())
        errors.extend(self.recoil.validate())
        errors.extend(self.training.validate())
        
        # Safety checks
        if self.enabled and not self.training_only:
            errors.append("Non-training mode requires explicit safety override")
        
        return errors
    
    def apply_preset(self, preset: str):
        """Apply configuration preset."""
        presets = {
            'training': {
                'enabled': True,
                'training_only': True,
                'aim': {'mode': AimMode.TRAINING},
                'training': {'enabled': True, 'record_data': True}
            },
            'csgo': {
                'detection': {'target_colors': ['red', 'yellow']},
                'aim': {'zone': AimZone.HEAD, 'fov_radius': 50},
                'recoil': {'enabled': True}
            },
            'valorant': {
                'detection': {'target_colors': ['red', 'purple']},
                'aim': {'zone': AimZone.HEAD, 'smoothness': 8.0},
                'recoil': {'enabled': False}  # Valorant has less recoil
            },
            'overwatch': {
                'detection': {'target_colors': ['red']},
                'aim': {'zone': AimZone.BODY, 'prediction_enabled': True},
                'recoil': {'enabled': False}
            },
            'apex': {
                'detection': {'target_colors': ['red', 'yellow']},
                'aim': {'zone': AimZone.CHEST, 'prediction_time': 0.15},
                'recoil': {'enabled': True, 'adaptive': True}
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        preset_data = presets[preset]
        
        # Apply general settings
        for key in ['enabled', 'training_only']:
            if key in preset_data:
                setattr(self, key, preset_data[key])
        
        # Apply component settings
        if 'detection' in preset_data:
            for key, value in preset_data['detection'].items():
                setattr(self.detection, key, value)
        
        if 'aim' in preset_data:
            for key, value in preset_data['aim'].items():
                if isinstance(value, str) and key == 'mode':
                    setattr(self.aim, key, AimMode(value))
                elif isinstance(value, str) and key == 'zone':
                    setattr(self.aim, key, AimZone(value))
                else:
                    setattr(self.aim, key, value)
        
        if 'recoil' in preset_data:
            for key, value in preset_data['recoil'].items():
                setattr(self.recoil, key, value)
        
        if 'training' in preset_data:
            for key, value in preset_data['training'].items():
                setattr(self.training, key, value)