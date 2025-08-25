"""Configuration and data validation utilities"""

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pathlib import Path
import yaml
import json
from dataclasses import dataclass
import structlog

from nexus.core.exceptions import ValidationError, ConfigError

logger = structlog.get_logger()


@dataclass
class ValidationRule:
    """Single validation rule"""
    field_path: str
    validator: Callable[[Any], bool]
    message: str
    required: bool = True
    default_value: Any = None


@dataclass
class ValidationResult:
    """Result of validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    fixed_values: Dict[str, Any]  # Values that were auto-corrected


class ConfigValidator:
    """Validates configuration dictionaries against rules"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.custom_validators: Dict[str, Callable] = {}
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules.append(rule)
    
    def add_custom_validator(self, name: str, validator: Callable[[Any], bool]):
        """Add a custom validator function"""
        self.custom_validators[name] = validator
    
    def validate(self, config: Dict[str, Any], fix_errors: bool = False) -> ValidationResult:
        """Validate configuration against all rules"""
        errors = []
        warnings = []
        fixed_values = {}
        
        for rule in self.rules:
            try:
                value = self._get_nested_value(config, rule.field_path)
                
                # Handle missing required fields
                if value is None and rule.required:
                    if rule.default_value is not None:
                        if fix_errors:
                            self._set_nested_value(config, rule.field_path, rule.default_value)
                            fixed_values[rule.field_path] = rule.default_value
                            warnings.append(f"Field '{rule.field_path}' missing, using default: {rule.default_value}")
                        else:
                            errors.append(f"Required field '{rule.field_path}' is missing")
                    else:
                        errors.append(f"Required field '{rule.field_path}' is missing")
                    continue
                
                # Skip validation for optional missing fields
                if value is None and not rule.required:
                    continue
                
                # Run validator
                if not rule.validator(value):
                    errors.append(f"Field '{rule.field_path}': {rule.message} (current value: {value})")
                
            except Exception as e:
                errors.append(f"Error validating field '{rule.field_path}': {e}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fixed_values=fixed_values
        )
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = path.split('.')
        current = config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set final value
        current[keys[-1]] = value


# Common validators
def is_positive_number(value: Any) -> bool:
    """Validate positive number"""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def is_non_negative_number(value: Any) -> bool:
    """Validate non-negative number"""
    try:
        return float(value) >= 0
    except (ValueError, TypeError):
        return False


def is_integer_range(min_val: int, max_val: int):
    """Validate integer within range"""
    def validator(value: Any) -> bool:
        try:
            int_val = int(value)
            return min_val <= int_val <= max_val
        except (ValueError, TypeError):
            return False
    return validator


def is_float_range(min_val: float, max_val: float):
    """Validate float within range"""
    def validator(value: Any) -> bool:
        try:
            float_val = float(value)
            return min_val <= float_val <= max_val
        except (ValueError, TypeError):
            return False
    return validator


def is_string_length(min_len: int = 0, max_len: int = None):
    """Validate string length"""
    def validator(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        length = len(value)
        return length >= min_len and (max_len is None or length <= max_len)
    return validator


def is_one_of(valid_values: List[Any]):
    """Validate value is one of allowed values"""
    def validator(value: Any) -> bool:
        return value in valid_values
    return validator


def is_regex_match(pattern: str):
    """Validate string matches regex pattern"""
    compiled_pattern = re.compile(pattern)
    
    def validator(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return bool(compiled_pattern.match(value))
    return validator


def is_valid_path(must_exist: bool = False):
    """Validate filesystem path"""
    def validator(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        
        try:
            path = Path(value)
            return not must_exist or path.exists()
        except Exception:
            return False
    return validator


def is_valid_ip_address(value: Any) -> bool:
    """Validate IP address (IPv4 or IPv6)"""
    if not isinstance(value, str):
        return False
    
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def is_valid_port(value: Any) -> bool:
    """Validate network port number"""
    try:
        port = int(value)
        return 1 <= port <= 65535
    except (ValueError, TypeError):
        return False


def is_boolean(value: Any) -> bool:
    """Validate boolean value"""
    return isinstance(value, bool)


def is_list_of(item_validator: Callable[[Any], bool]):
    """Validate list where all items pass validator"""
    def validator(value: Any) -> bool:
        if not isinstance(value, list):
            return False
        return all(item_validator(item) for item in value)
    return validator


def is_dict_with_keys(required_keys: List[str], optional_keys: List[str] = None):
    """Validate dictionary has required keys"""
    optional_keys = optional_keys or []
    
    def validator(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        
        # Check required keys
        for key in required_keys:
            if key not in value:
                return False
        
        # Check no extra keys
        allowed_keys = set(required_keys + optional_keys)
        return set(value.keys()).issubset(allowed_keys)
    
    return validator


def create_nexus_config_validator() -> ConfigValidator:
    """Create validator for Nexus configuration"""
    validator = ConfigValidator()
    
    # Core configuration
    validator.add_rule(ValidationRule(
        "nexus.version", is_string_length(1), "Version must be non-empty string", False, "0.1.0"
    ))
    validator.add_rule(ValidationRule(
        "nexus.debug", is_boolean, "Debug must be boolean", False, False
    ))
    
    # Capture configuration
    validator.add_rule(ValidationRule(
        "capture.backend", is_one_of(["dxcam", "mss"]), "Backend must be 'dxcam' or 'mss'", False, "dxcam"
    ))
    validator.add_rule(ValidationRule(
        "capture.fps", is_integer_range(1, 240), "FPS must be between 1 and 240", False, 60
    ))
    validator.add_rule(ValidationRule(
        "capture.buffer_size", is_positive_number, "Buffer size must be positive", False, 64
    ))
    
    # Vision configuration
    validator.add_rule(ValidationRule(
        "vision.confidence_threshold", is_float_range(0.0, 1.0), "Confidence threshold must be between 0.0 and 1.0", False, 0.5
    ))
    validator.add_rule(ValidationRule(
        "vision.batch_size", is_positive_number, "Batch size must be positive", False, 1
    ))
    
    # API configuration
    validator.add_rule(ValidationRule(
        "api.host", is_valid_ip_address, "Host must be valid IP address", False, "127.0.0.1"
    ))
    validator.add_rule(ValidationRule(
        "api.port", is_valid_port, "Port must be between 1 and 65535", False, 8000
    ))
    validator.add_rule(ValidationRule(
        "api.enabled", is_boolean, "Enabled must be boolean", False, True
    ))
    
    # Performance configuration
    validator.add_rule(ValidationRule(
        "performance.max_cpu_percent", is_integer_range(1, 100), "Max CPU percent must be between 1 and 100", False, 80
    ))
    validator.add_rule(ValidationRule(
        "performance.max_memory_mb", is_positive_number, "Max memory must be positive", False, 4096
    ))
    
    return validator


def validate_config(config: Dict[str, Any], fix_errors: bool = False) -> ValidationResult:
    """Validate Nexus configuration"""
    validator = create_nexus_config_validator()
    result = validator.validate(config, fix_errors)
    
    if result.errors:
        logger.error(f"Configuration validation failed: {len(result.errors)} errors")
        for error in result.errors:
            logger.error(f"  {error}")
    
    if result.warnings:
        logger.warning(f"Configuration validation warnings: {len(result.warnings)} warnings")
        for warning in result.warnings:
            logger.warning(f"  {warning}")
    
    return result


def validate_plugin_manifest(manifest_data: Dict[str, Any]) -> ValidationResult:
    """Validate plugin manifest"""
    validator = ConfigValidator()
    
    # Required fields
    validator.add_rule(ValidationRule(
        "name", is_string_length(1), "Name must be non-empty string"
    ))
    validator.add_rule(ValidationRule(
        "version", is_regex_match(r"^\d+\.\d+\.\d+"), "Version must be semantic version (x.y.z)"
    ))
    validator.add_rule(ValidationRule(
        "author", is_string_length(1), "Author must be non-empty string"
    ))
    validator.add_rule(ValidationRule(
        "description", is_string_length(1), "Description must be non-empty string"
    ))
    validator.add_rule(ValidationRule(
        "plugin_type", is_one_of(["game", "agent", "capture", "vision", "input", "processor", "extension"]), 
        "Plugin type must be valid type"
    ))
    validator.add_rule(ValidationRule(
        "entry_point", is_string_length(1), "Entry point must be non-empty string"
    ))
    
    # Optional fields
    validator.add_rule(ValidationRule(
        "dependencies", is_list_of(lambda x: isinstance(x, str)), "Dependencies must be list of strings", False, []
    ))
    validator.add_rule(ValidationRule(
        "min_nexus_version", is_regex_match(r"^\d+\.\d+\.\d+"), "Min Nexus version must be semantic version", False, "0.1.0"
    ))
    
    return validator.validate(manifest_data)


def validate_environment_vars() -> ValidationResult:
    """Validate environment variable configuration"""
    import os
    
    errors = []
    warnings = []
    
    # Check for common environment variables
    env_checks = [
        ("NEXUS_DEBUG", lambda v: v.lower() in ["true", "false"], "Must be 'true' or 'false'"),
        ("NEXUS_LOG_LEVEL", lambda v: v.upper() in ["DEBUG", "INFO", "WARNING", "ERROR"], "Must be valid log level"),
        ("NEXUS_CONFIG_PATH", lambda v: Path(v).exists(), "Config file must exist")
    ]
    
    for env_var, validator_func, message in env_checks:
        value = os.environ.get(env_var)
        if value is not None:
            try:
                if not validator_func(value):
                    errors.append(f"Environment variable {env_var}: {message} (current: {value})")
            except Exception as e:
                errors.append(f"Environment variable {env_var}: Validation error: {e}")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        fixed_values={}
    )


def sanitize_config_value(value: Any, expected_type: Type) -> Any:
    """Sanitize and convert config value to expected type"""
    if value is None:
        return value
    
    try:
        if expected_type == bool:
            if isinstance(value, str):
                return value.lower() in ["true", "1", "yes", "on"]
            return bool(value)
        
        elif expected_type == int:
            return int(float(value))  # Handle string floats
        
        elif expected_type == float:
            return float(value)
        
        elif expected_type == str:
            return str(value)
        
        elif expected_type == list:
            if isinstance(value, str):
                # Try to parse as JSON array or comma-separated
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return [item.strip() for item in value.split(",") if item.strip()]
            return list(value)
        
        else:
            return value
    
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to sanitize value {value} to type {expected_type}: {e}")
        return value