import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import structlog
from structlog import PrintLogger
from structlog.stdlib import LoggerFactory
from datetime import datetime
import json


class PerformanceProcessor:
    
    def __init__(self):
        self.timings = {}
    
    def __call__(self, logger, log_method, event_dict):
        if "duration_ms" in event_dict:
            event_name = event_dict.get("event", "unknown")
            duration = event_dict["duration_ms"]
            
            if event_name not in self.timings:
                self.timings[event_name] = []
            
            self.timings[event_name].append(duration)
            
            if len(self.timings[event_name]) > 100:
                self.timings[event_name].pop(0)
            
            avg_duration = sum(self.timings[event_name]) / len(self.timings[event_name])
            event_dict["avg_duration_ms"] = round(avg_duration, 2)
        
        return event_dict


class NexusLoggerFactory:
    
    def __init__(self, 
                 level: str = "INFO",
                 console_output: bool = True,
                 file_output: Optional[str] = None,
                 json_format: bool = True,
                 add_timestamp: bool = True,
                 performance_tracking: bool = True):
        
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.console_output = console_output
        self.file_output = file_output
        self.json_format = json_format
        self.add_timestamp = add_timestamp
        self.performance_tracking = performance_tracking
        
        self._setup_logging()
    
    def _setup_logging(self):
        processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
        ]
        
        if self.add_timestamp:
            processors.append(structlog.processors.TimeStamper(fmt="iso"))
        
        if self.performance_tracking:
            processors.append(PerformanceProcessor())
        
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ])
        
        if self.json_format:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        root_logger.handlers = []
        
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            root_logger.addHandler(console_handler)
        
        if self.file_output:
            file_path = Path(self.file_output)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5
            )
            file_handler.setLevel(self.level)
            root_logger.addHandler(file_handler)


class LogContext:
    
    def __init__(self, logger, **context):
        self.logger = logger
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger = self.logger.bind(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        
        if exc_type:
            self.logger.error(
                "Context failed",
                duration_ms=duration,
                exception=str(exc_val)
            )
        else:
            self.logger.info(
                "Context completed",
                duration_ms=duration
            )
        
        return False


class MetricsLogger:
    
    def __init__(self, logger):
        self.logger = logger
        self.metrics = {}
    
    def log_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, Any]] = None):
        self.metrics[name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
            "tags": tags or {}
        }
        
        self.logger.info(
            "Metric recorded",
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags
        )
    
    def log_counter(self, name: str, increment: int = 1, tags: Optional[Dict[str, Any]] = None):
        if name not in self.metrics:
            self.metrics[name] = {"value": 0, "tags": tags or {}}
        
        self.metrics[name]["value"] += increment
        
        self.logger.info(
            "Counter updated",
            counter_name=name,
            value=self.metrics[name]["value"],
            increment=increment,
            tags=tags
        )
    
    def log_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, Any]] = None):
        self.log_metric(f"{name}_duration", duration_ms, "ms", tags)
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self.metrics = {}
        self.logger.info("Metrics reset")


_logger_factory: Optional[NexusLoggerFactory] = None
_loggers: Dict[str, Any] = {}


def setup_logging(level: str = "INFO",
                  console: bool = True,
                  file: Optional[str] = None,
                  json_format: bool = True,
                  performance: bool = True):
    global _logger_factory
    
    _logger_factory = NexusLoggerFactory(
        level=level,
        console_output=console,
        file_output=file,
        json_format=json_format,
        performance_tracking=performance
    )


def get_logger(name: Optional[str] = None) -> Any:
    global _loggers, _logger_factory
    
    if _logger_factory is None:
        setup_logging()
    
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "nexus")
        else:
            name = "nexus"
    
    if name not in _loggers:
        _loggers[name] = structlog.get_logger(name)
    
    return _loggers[name]


def log_context(logger, **context) -> LogContext:
    return LogContext(logger, **context)


def get_metrics_logger(name: str = "metrics") -> MetricsLogger:
    logger = get_logger(name)
    return MetricsLogger(logger)


class ErrorHandler:
    
    def __init__(self, logger, reraise: bool = True):
        self.logger = logger
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                "Unhandled exception",
                exc_type=exc_type.__name__,
                exc_message=str(exc_val),
                exc_info=True
            )
            
            if not self.reraise:
                return True
        
        return False


def error_handler(logger, reraise: bool = True) -> ErrorHandler:
    return ErrorHandler(logger, reraise)