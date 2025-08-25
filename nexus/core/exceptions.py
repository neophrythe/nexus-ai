"""Custom exception classes for the Nexus framework"""

from typing import Any, Optional, Dict
import traceback


class NexusError(Exception):
    """Base exception for all Nexus-related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.traceback_info = traceback.format_exc()


class PluginError(NexusError):
    """Exception raised when plugin operations fail"""
    
    def __init__(self, plugin_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.plugin_name = plugin_name
        super().__init__(f"Plugin '{plugin_name}': {message}", details)


class CaptureError(NexusError):
    """Exception raised when screen capture operations fail"""
    
    def __init__(self, backend: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.backend = backend
        super().__init__(f"Capture backend '{backend}': {message}", details)


class VisionError(NexusError):
    """Exception raised when vision processing fails"""
    
    def __init__(self, operation: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        super().__init__(f"Vision operation '{operation}': {message}", details)


class AgentError(NexusError):
    """Exception raised when agent operations fail"""
    
    def __init__(self, agent_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.agent_name = agent_name
        super().__init__(f"Agent '{agent_name}': {message}", details)


class ConfigError(NexusError):
    """Exception raised when configuration is invalid"""
    
    def __init__(self, key: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.key = key
        super().__init__(f"Config '{key}': {message}", details)


class LauncherError(NexusError):
    """Exception raised when game launching fails"""
    
    def __init__(self, launcher_type: str, game_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.launcher_type = launcher_type
        self.game_name = game_name
        super().__init__(f"Launcher '{launcher_type}' for game '{game_name}': {message}", details)


class EnvironmentError(NexusError):
    """Exception raised when environment operations fail"""
    
    def __init__(self, environment_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.environment_name = environment_name
        super().__init__(f"Environment '{environment_name}': {message}", details)


class TrainingError(NexusError):
    """Exception raised when training operations fail"""
    
    def __init__(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.stage = stage
        super().__init__(f"Training stage '{stage}': {message}", details)


class APIError(NexusError):
    """Exception raised when API operations fail"""
    
    def __init__(self, endpoint: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.endpoint = endpoint
        super().__init__(f"API endpoint '{endpoint}': {message}", details)


class ResourceError(NexusError):
    """Exception raised when resource allocation/deallocation fails"""
    
    def __init__(self, resource_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.resource_type = resource_type
        super().__init__(f"Resource '{resource_type}': {message}", details)


class ValidationError(NexusError):
    """Exception raised when data validation fails"""
    
    def __init__(self, field: str, value: Any, message: str, details: Optional[Dict[str, Any]] = None):
        self.field = field
        self.value = value
        super().__init__(f"Validation error for '{field}' with value '{value}': {message}", details)


class TimeoutError(NexusError):
    """Exception raised when operations timeout"""
    
    def __init__(self, operation: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds}s", details)


class DependencyError(NexusError):
    """Exception raised when required dependencies are missing"""
    
    def __init__(self, dependency: str, message: str = None, details: Optional[Dict[str, Any]] = None):
        self.dependency = dependency
        default_msg = f"Required dependency '{dependency}' not found or incompatible"
        super().__init__(message or default_msg, details)


class PermissionError(NexusError):
    """Exception raised when permission/access is denied"""
    
    def __init__(self, resource: str, action: str, message: str = None, details: Optional[Dict[str, Any]] = None):
        self.resource = resource
        self.action = action
        default_msg = f"Permission denied for action '{action}' on resource '{resource}'"
        super().__init__(message or default_msg, details)


class InitializationError(NexusError):
    """Exception raised when component initialization fails"""
    
    def __init__(self, component: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.component = component
        super().__init__(f"Failed to initialize component '{component}': {message}", details)


class StateError(NexusError):
    """Exception raised when component is in invalid state for operation"""
    
    def __init__(self, component: str, current_state: str, required_state: str, details: Optional[Dict[str, Any]] = None):
        self.component = component
        self.current_state = current_state
        self.required_state = required_state
        super().__init__(
            f"Component '{component}' is in state '{current_state}', required state '{required_state}'",
            details
        )


# Utility functions for error handling

def handle_exception(exception: Exception, logger, context: str = "") -> NexusError:
    """Convert generic exceptions to Nexus exceptions with proper logging"""
    
    if isinstance(exception, NexusError):
        logger.error(f"{context}: {exception}")
        return exception
    
    # Convert common exceptions
    if isinstance(exception, FileNotFoundError):
        nexus_error = ResourceError("file", f"File not found: {exception}", {"original_error": str(exception)})
    elif isinstance(exception, PermissionError):
        nexus_error = PermissionError("filesystem", "access", str(exception), {"original_error": str(exception)})
    elif isinstance(exception, TimeoutError):
        nexus_error = TimeoutError("unknown", 0, {"original_error": str(exception)})
    elif isinstance(exception, ValueError):
        nexus_error = ValidationError("unknown", None, str(exception), {"original_error": str(exception)})
    elif isinstance(exception, ImportError):
        nexus_error = DependencyError("unknown", str(exception), {"original_error": str(exception)})
    else:
        nexus_error = NexusError(f"Unexpected error: {exception}", {"original_error": str(exception), "type": type(exception).__name__})
    
    logger.error(f"{context}: Converted exception to NexusError: {nexus_error}")
    return nexus_error


def create_error_context(component: str, operation: str, **kwargs) -> Dict[str, Any]:
    """Create standardized error context for debugging"""
    import time
    import os
    
    context = {
        "component": component,
        "operation": operation,
        "timestamp": time.time(),
        "pid": os.getpid(),
        **kwargs
    }
    
    return context


def log_performance_warning(logger, operation: str, duration: float, threshold: float = 1.0):
    """Log performance warnings for slow operations"""
    if duration > threshold:
        logger.warning(
            f"Performance warning: {operation} took {duration:.2f}s (threshold: {threshold:.2f}s)",
            extra={"operation": operation, "duration": duration, "threshold": threshold}
        )


class ErrorHandler:
    """Centralized error handling utility"""
    
    def __init__(self, logger, component_name: str):
        self.logger = logger
        self.component_name = component_name
        self.error_counts = {}
    
    def handle_error(self, error: Exception, operation: str, reraise: bool = True) -> Optional[NexusError]:
        """Handle an error with logging and optional reraising"""
        context = create_error_context(self.component_name, operation)
        nexus_error = handle_exception(error, self.logger, f"{self.component_name}.{operation}")
        
        # Track error frequency
        error_type = type(nexus_error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        if self.error_counts[error_type] > 5:
            self.logger.critical(
                f"High error frequency for {error_type} in {self.component_name}: {self.error_counts[error_type]} occurrences"
            )
        
        if reraise:
            raise nexus_error
        
        return nexus_error
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics for this handler"""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error counters"""
        self.error_counts.clear()