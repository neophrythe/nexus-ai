"""
Haptic Feedback System for Advanced Controller Vibration

Provides advanced haptic feedback patterns and effects for game controllers.
"""

import time
import threading
import math
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import structlog

from nexus.input.gamepad.gamepad_base import GamepadBase

logger = structlog.get_logger()


class HapticPattern(Enum):
    """Predefined haptic patterns."""
    PULSE = "pulse"
    WAVE = "wave"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    HEARTBEAT = "heartbeat"
    EXPLOSION = "explosion"
    RUMBLE = "rumble"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    NOTIFICATION = "notification"
    WARNING = "warning"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class HapticEffect:
    """Single haptic effect."""
    left_motor: float  # 0.0 to 1.0
    right_motor: float  # 0.0 to 1.0
    duration_ms: int
    delay_ms: int = 0


@dataclass
class HapticSequence:
    """Sequence of haptic effects."""
    name: str
    effects: List[HapticEffect]
    loop: bool = False
    loop_count: int = 1


class HapticFeedback:
    """
    Advanced haptic feedback system for game controllers.
    
    Features:
    - Predefined haptic patterns
    - Custom effect sequences
    - Dynamic haptic generation
    - Multi-controller support
    - Effect blending and layering
    - Audio-to-haptic conversion
    """
    
    # Predefined effect sequences
    PATTERNS = {
        HapticPattern.PULSE: [
            HapticEffect(0.8, 0.8, 100),
            HapticEffect(0.0, 0.0, 100)
        ],
        HapticPattern.WAVE: [
            HapticEffect(1.0, 0.0, 150),
            HapticEffect(0.0, 1.0, 150)
        ],
        HapticPattern.HEARTBEAT: [
            HapticEffect(0.7, 0.7, 100),
            HapticEffect(0.0, 0.0, 100),
            HapticEffect(1.0, 1.0, 100),
            HapticEffect(0.0, 0.0, 400)
        ],
        HapticPattern.EXPLOSION: [
            HapticEffect(1.0, 1.0, 200),
            HapticEffect(0.7, 0.7, 150),
            HapticEffect(0.4, 0.4, 100),
            HapticEffect(0.2, 0.2, 50)
        ],
        HapticPattern.CLICK: [
            HapticEffect(0.5, 0.5, 20)
        ],
        HapticPattern.DOUBLE_CLICK: [
            HapticEffect(0.5, 0.5, 20),
            HapticEffect(0.0, 0.0, 50),
            HapticEffect(0.5, 0.5, 20)
        ],
        HapticPattern.NOTIFICATION: [
            HapticEffect(0.3, 0.3, 100),
            HapticEffect(0.0, 0.0, 50),
            HapticEffect(0.3, 0.3, 100)
        ],
        HapticPattern.WARNING: [
            HapticEffect(0.6, 0.6, 150),
            HapticEffect(0.0, 0.0, 75),
            HapticEffect(0.6, 0.6, 150),
            HapticEffect(0.0, 0.0, 75),
            HapticEffect(0.6, 0.6, 150)
        ],
        HapticPattern.SUCCESS: [
            HapticEffect(0.4, 0.4, 100),
            HapticEffect(0.0, 0.0, 50),
            HapticEffect(0.7, 0.7, 200)
        ],
        HapticPattern.FAILURE: [
            HapticEffect(1.0, 1.0, 300),
            HapticEffect(0.0, 0.0, 100),
            HapticEffect(0.5, 0.5, 200)
        ]
    }
    
    def __init__(self, controller: GamepadBase):
        """
        Initialize haptic feedback system.
        
        Args:
            controller: Controller to provide feedback for
        """
        self.controller = controller
        
        # Active effects
        self.active_effects: List[threading.Thread] = []
        self.is_playing = False
        
        # Custom sequences
        self.custom_sequences: Dict[str, HapticSequence] = {}
        
        # Effect intensity
        self.global_intensity = 1.0
        self.left_intensity = 1.0
        self.right_intensity = 1.0
        
        # Audio-to-haptic settings
        self.audio_haptic_enabled = False
        self.audio_sensitivity = 1.0
        
        logger.info(f"Haptic feedback initialized for controller {controller.controller_id}")
    
    def play_pattern(self, pattern: HapticPattern, intensity: float = 1.0, repeat: int = 1):
        """
        Play a predefined haptic pattern.
        
        Args:
            pattern: Pattern to play
            intensity: Intensity multiplier (0.0-1.0)
            repeat: Number of times to repeat
        """
        if pattern not in self.PATTERNS:
            logger.warning(f"Unknown pattern: {pattern}")
            return
        
        effects = self.PATTERNS[pattern]
        self._play_effects(effects, intensity, repeat)
    
    def play_effect(self, left: float, right: float, duration_ms: int):
        """
        Play a simple haptic effect.
        
        Args:
            left: Left motor intensity (0.0-1.0)
            right: Right motor intensity (0.0-1.0)
            duration_ms: Duration in milliseconds
        """
        effect = HapticEffect(left, right, duration_ms)
        self._play_effects([effect], 1.0, 1)
    
    def play_sequence(self, sequence_name: str, intensity: float = 1.0):
        """
        Play a custom haptic sequence.
        
        Args:
            sequence_name: Name of sequence to play
            intensity: Intensity multiplier
        """
        if sequence_name not in self.custom_sequences:
            logger.warning(f"Unknown sequence: {sequence_name}")
            return
        
        sequence = self.custom_sequences[sequence_name]
        repeat = sequence.loop_count if sequence.loop else 1
        self._play_effects(sequence.effects, intensity, repeat)
    
    def create_sequence(self, name: str, effects: List[HapticEffect], 
                       loop: bool = False, loop_count: int = 1):
        """
        Create a custom haptic sequence.
        
        Args:
            name: Sequence name
            effects: List of effects
            loop: Whether to loop
            loop_count: Number of loops
        """
        sequence = HapticSequence(name, effects, loop, loop_count)
        self.custom_sequences[name] = sequence
        logger.info(f"Created haptic sequence: {name}")
    
    def generate_wave(self, frequency: float, amplitude: float, 
                     duration_ms: int, phase_shift: float = 0):
        """
        Generate a wave-based haptic effect.
        
        Args:
            frequency: Wave frequency in Hz
            amplitude: Wave amplitude (0.0-1.0)
            duration_ms: Duration in milliseconds
            phase_shift: Phase shift for stereo effect
        """
        effects = []
        sample_rate = 60  # Hz
        samples = int(duration_ms * sample_rate / 1000)
        
        for i in range(samples):
            t = i / sample_rate
            left = amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * frequency * t))
            right = amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * frequency * t + phase_shift))
            
            effects.append(HapticEffect(
                left, right, 
                int(1000 / sample_rate)  # Duration per sample
            ))
        
        self._play_effects(effects, 1.0, 1)
    
    def generate_noise(self, intensity: float, duration_ms: int):
        """
        Generate random noise haptic effect.
        
        Args:
            intensity: Noise intensity (0.0-1.0)
            duration_ms: Duration in milliseconds
        """
        import random
        
        effects = []
        samples = duration_ms // 20  # 20ms per sample
        
        for _ in range(samples):
            left = random.random() * intensity
            right = random.random() * intensity
            effects.append(HapticEffect(left, right, 20))
        
        self._play_effects(effects, 1.0, 1)
    
    def collision_feedback(self, impact_force: float, direction: float = 0):
        """
        Generate collision/impact feedback.
        
        Args:
            impact_force: Force of impact (0.0-1.0)
            direction: Direction in radians (0 = center, -pi/2 = left, pi/2 = right)
        """
        # Calculate stereo distribution based on direction
        left_intensity = impact_force * (0.5 - 0.5 * math.sin(direction))
        right_intensity = impact_force * (0.5 + 0.5 * math.sin(direction))
        
        # Create impact effect
        effects = [
            HapticEffect(left_intensity, right_intensity, 100),
            HapticEffect(left_intensity * 0.5, right_intensity * 0.5, 50),
            HapticEffect(left_intensity * 0.2, right_intensity * 0.2, 50)
        ]
        
        self._play_effects(effects, 1.0, 1)
    
    def weapon_feedback(self, weapon_type: str):
        """
        Generate weapon-specific feedback.
        
        Args:
            weapon_type: Type of weapon ('pistol', 'rifle', 'shotgun', etc.)
        """
        weapon_effects = {
            'pistol': [
                HapticEffect(0.0, 0.8, 50),
                HapticEffect(0.0, 0.3, 50)
            ],
            'rifle': [
                HapticEffect(0.3, 0.3, 30),
                HapticEffect(0.2, 0.2, 20),
                HapticEffect(0.1, 0.1, 20)
            ],
            'shotgun': [
                HapticEffect(1.0, 1.0, 100),
                HapticEffect(0.5, 0.5, 100)
            ],
            'laser': [
                HapticEffect(0.2, 0.2, 200)
            ],
            'melee': [
                HapticEffect(0.7, 0.0, 50),
                HapticEffect(0.0, 0.7, 50)
            ]
        }
        
        if weapon_type in weapon_effects:
            self._play_effects(weapon_effects[weapon_type], 1.0, 1)
        else:
            logger.warning(f"Unknown weapon type: {weapon_type}")
    
    def engine_rumble(self, rpm: float, max_rpm: float = 8000):
        """
        Generate engine rumble effect for racing games.
        
        Args:
            rpm: Current engine RPM
            max_rpm: Maximum engine RPM
        """
        intensity = min(1.0, rpm / max_rpm)
        frequency = 10 + (rpm / max_rpm) * 40  # 10-50 Hz
        
        # Low frequency rumble
        left = intensity * 0.3 + intensity * 0.2 * math.sin(time.time() * frequency)
        right = intensity * 0.3 + intensity * 0.2 * math.cos(time.time() * frequency)
        
        self.controller.vibrate(left, right, 100)
    
    def set_intensity(self, global_intensity: float = None, 
                     left: float = None, right: float = None):
        """
        Set haptic intensity.
        
        Args:
            global_intensity: Global intensity multiplier
            left: Left motor intensity
            right: Right motor intensity
        """
        if global_intensity is not None:
            self.global_intensity = max(0.0, min(1.0, global_intensity))
        if left is not None:
            self.left_intensity = max(0.0, min(1.0, left))
        if right is not None:
            self.right_intensity = max(0.0, min(1.0, right))
    
    def stop_all(self):
        """Stop all active haptic effects."""
        self.is_playing = False
        self.controller.vibrate(0, 0, 0)
        
        # Wait for threads to finish
        for thread in self.active_effects:
            if thread.is_alive():
                thread.join(timeout=0.1)
        
        self.active_effects.clear()
        logger.info("Stopped all haptic effects")
    
    def create_adaptive_trigger_effect(self, trigger: str, effect_type: str, **params):
        """
        Create adaptive trigger effect (PS5 DualSense only).
        
        Args:
            trigger: 'left' or 'right'
            effect_type: Type of effect
            **params: Effect parameters
        """
        # This would interface with PS5 controller's adaptive triggers
        # Placeholder for now
        logger.info(f"Adaptive trigger effect: {trigger} - {effect_type}")
    
    def audio_to_haptic(self, audio_data: bytes, sample_rate: int = 44100):
        """
        Convert audio to haptic feedback.
        
        Args:
            audio_data: Audio data bytes
            sample_rate: Audio sample rate
        """
        if not self.audio_haptic_enabled:
            return
        
        # Simplified audio analysis
        # In real implementation, would use FFT for frequency analysis
        import struct
        
        # Parse audio samples (assuming 16-bit mono)
        samples = []
        for i in range(0, len(audio_data) - 1, 2):
            sample = struct.unpack('<h', audio_data[i:i+2])[0] / 32768.0
            samples.append(abs(sample))
        
        # Create haptic from audio envelope
        window_size = sample_rate // 60  # 60 Hz haptic rate
        
        for i in range(0, len(samples), window_size):
            window = samples[i:i+window_size]
            if window:
                amplitude = sum(window) / len(window) * self.audio_sensitivity
                amplitude = min(1.0, amplitude)
                
                # Simple stereo distribution
                self.controller.vibrate(amplitude * 0.7, amplitude * 0.7, 16)
    
    # Private methods
    
    def _play_effects(self, effects: List[HapticEffect], intensity: float, repeat: int):
        """Play a list of haptic effects."""
        def play_thread():
            for _ in range(repeat):
                if not self.is_playing:
                    break
                
                for effect in effects:
                    if not self.is_playing:
                        break
                    
                    # Apply intensity modifiers
                    left = effect.left_motor * intensity * self.global_intensity * self.left_intensity
                    right = effect.right_motor * intensity * self.global_intensity * self.right_intensity
                    
                    # Apply delay if specified
                    if effect.delay_ms > 0:
                        time.sleep(effect.delay_ms / 1000.0)
                    
                    # Play effect
                    self.controller.vibrate(left, right, effect.duration_ms)
                    
                    # Wait for effect to complete
                    time.sleep(effect.duration_ms / 1000.0)
        
        self.is_playing = True
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
        self.active_effects.append(thread)
        
        # Clean up finished threads
        self.active_effects = [t for t in self.active_effects if t.is_alive()]
    
    def create_haptic_profile(self, name: str) -> Dict[str, Any]:
        """
        Create a haptic profile for different game scenarios.
        
        Args:
            name: Profile name
        
        Returns:
            Haptic profile configuration
        """
        profiles = {
            'racing': {
                'engine_enabled': True,
                'collision_intensity': 0.8,
                'surface_feedback': True,
                'gear_shift_feedback': True
            },
            'shooter': {
                'weapon_feedback': True,
                'explosion_intensity': 1.0,
                'footstep_feedback': False,
                'damage_feedback': True
            },
            'adventure': {
                'environment_feedback': True,
                'interaction_feedback': True,
                'combat_intensity': 0.6,
                'discovery_feedback': True
            },
            'silent': {
                'all_disabled': True
            }
        }
        
        return profiles.get(name, {})