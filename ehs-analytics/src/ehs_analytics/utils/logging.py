"""
Comprehensive logging system for EHS Analytics platform.

This module provides structured logging capabilities with JSON formatting,
log rotation, context injection, and performance monitoring decorators.
Production-ready with integration points for monitoring tools.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass, asdict

import structlog


@dataclass
class LogContext:
    """Context information to be injected into log messages."""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    query_type: Optional[str] = None
    facility_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Formats log records as JSON with consistent field structure
    and proper timestamp formatting for production environments.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'funcName', 'lineno', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'message', 'exc_info',
                    'exc_text', 'stack_info'
                }
            }
            if extra_fields:
                log_entry.update(extra_fields)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """
    Logging filter that injects context information into log records.
    
    Adds request-scoped context like user_id, request_id, etc.
    to all log messages within the current context.
    """
    
    def __init__(self):
        super().__init__()
        self._context = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to the log record."""
        for key, value in self._context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, context: Union[Dict[str, Any], LogContext]):
        """Set the current context for logging."""
        if isinstance(context, LogContext):
            self._context = {k: v for k, v in asdict(context).items() if v is not None}
        elif isinstance(context, dict):
            self._context = context
    
    def clear_context(self):
        """Clear the current logging context."""
        self._context = {}
    
    def update_context(self, **kwargs):
        """Update specific context fields."""
        self._context.update(kwargs)


class EHSLogger:
    """
    Enhanced logger for EHS Analytics with structured logging capabilities.
    
    Provides methods for different types of EHS operations with automatic
    context injection and performance monitoring.
    """
    
    def __init__(self, name: str, context_filter: Optional[ContextFilter] = None):
        self.logger = structlog.get_logger(name)
        self.name = name
        self.context_filter = context_filter or ContextFilter()
    
    def bind(self, **kwargs) -> "EHSLogger":
        """Create a new logger with bound context."""
        bound_logger = self.logger.bind(**kwargs)
        return EHSLogger(self.name, self.context_filter)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.logger.critical(message, **kwargs)
    
    def query_start(self, query: str, query_type: str, **context):
        """Log the start of a query operation."""
        self.info(
            "Query operation started",
            event_type="query_start",
            query=query,
            query_type=query_type,
            **context
        )
    
    def query_end(self, query: str, query_type: str, duration_ms: float, success: bool, **context):
        """Log the end of a query operation."""
        self.info(
            "Query operation completed",
            event_type="query_end",
            query=query,
            query_type=query_type,
            duration_ms=duration_ms,
            success=success,
            **context
        )
    
    def retrieval_operation(self, strategy: str, results_count: int, duration_ms: float, **context):
        """Log a data retrieval operation."""
        self.info(
            "Data retrieval operation",
            event_type="retrieval",
            strategy=strategy,
            results_count=results_count,
            duration_ms=duration_ms,
            **context
        )
    
    def analysis_operation(self, analysis_type: str, confidence: float, duration_ms: float, **context):
        """Log an analysis operation."""
        self.info(
            "Analysis operation completed",
            event_type="analysis",
            analysis_type=analysis_type,
            confidence=confidence,
            duration_ms=duration_ms,
            **context
        )
    
    def recommendation_generated(self, recommendation_count: int, total_savings: float, **context):
        """Log recommendation generation."""
        self.info(
            "Recommendations generated",
            event_type="recommendation",
            recommendation_count=recommendation_count,
            total_savings=total_savings,
            **context
        )
    
    def security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log security-related events."""
        self.warning(
            f"Security event: {event_type}",
            event_type="security",
            security_event_type=event_type,
            severity=severity,
            details=details
        )
    
    def performance_metric(self, metric_name: str, value: float, unit: str, **context):
        """Log performance metrics."""
        self.info(
            f"Performance metric: {metric_name}",
            event_type="performance",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **context
        )


def performance_logger(include_args: bool = False, include_result: bool = False):
    """
    Decorator for logging function performance and execution details.
    
    Args:
        include_args: Whether to log function arguments
        include_result: Whether to log function return value
    
    Usage:
        @performance_logger(include_args=True)
        def my_function(arg1, arg2):
            return "result"
    """
    def decorator(func: Callable) -> Callable:
        logger = get_ehs_logger(f"{func.__module__}.{func.__name__}")
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            log_data = {
                "function": func_name,
                "event_type": "function_call"
            }
            
            if include_args and (args or kwargs):
                log_data.update({
                    "args": str(args) if args else None,
                    "kwargs": kwargs if kwargs else None
                })
            
            logger.debug("Function execution started", **log_data)
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                success_data = {
                    "function": func_name,
                    "event_type": "function_success",
                    "duration_ms": duration_ms,
                    "success": True
                }
                
                if include_result and result is not None:
                    success_data["result"] = str(result)
                
                logger.info("Function execution completed", **success_data)
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Function execution failed",
                    function=func_name,
                    event_type="function_error",
                    duration_ms=duration_ms,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            log_data = {
                "function": func_name,
                "event_type": "function_call"
            }
            
            if include_args and (args or kwargs):
                log_data.update({
                    "args": str(args) if args else None,
                    "kwargs": kwargs if kwargs else None
                })
            
            logger.debug("Function execution started", **log_data)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                success_data = {
                    "function": func_name,
                    "event_type": "function_success",
                    "duration_ms": duration_ms,
                    "success": True
                }
                
                if include_result and result is not None:
                    success_data["result"] = str(result)
                
                logger.info("Function execution completed", **success_data)
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Function execution failed",
                    function=func_name,
                    event_type="function_error",
                    duration_ms=duration_ms,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True
                )
                raise
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def log_context(**context):
    """
    Context manager for setting logging context within a block.
    
    Usage:
        with log_context(user_id="123", operation="query"):
            logger.info("This will include user_id and operation in the log")
    """
    # Get the global context filter if available
    root_logger = logging.getLogger()
    context_filter = None
    
    for handler in root_logger.handlers:
        for filter_obj in handler.filters:
            if isinstance(filter_obj, ContextFilter):
                context_filter = filter_obj
                break
    
    if context_filter:
        old_context = context_filter._context.copy()
        context_filter.update_context(**context)
        try:
            yield
        finally:
            context_filter._context = old_context
    else:
        yield


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    enable_rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = True,
    include_extra: bool = True
) -> ContextFilter:
    """
    Setup comprehensive logging configuration for EHS Analytics.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to ./logs)
        enable_file_logging: Enable logging to files
        enable_console_logging: Enable logging to console
        enable_rotation: Enable log file rotation
        max_bytes: Maximum size per log file
        backup_count: Number of backup files to keep
        json_format: Use JSON formatting for logs
        include_extra: Include extra fields in JSON logs
    
    Returns:
        ContextFilter instance for managing context
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=False) if not json_format else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create context filter
    context_filter = ContextFilter()
    
    # Setup formatters
    if json_format:
        formatter = StructuredFormatter(include_extra=include_extra)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Setup console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # Setup file handlers
    if enable_file_logging:
        # Main application log
        app_log_file = log_dir / "ehs_analytics.log"
        
        if enable_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                app_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(app_log_file)
        
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)
        
        # Error log (warnings and above)
        error_log_file = log_dir / "errors.log"
        
        if enable_rotation:
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            error_handler = logging.FileHandler(error_log_file)
        
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)
        
        # Query log (for query operations)
        query_log_file = log_dir / "queries.log"
        query_logger = logging.getLogger("ehs_analytics.queries")
        
        if enable_rotation:
            query_handler = logging.handlers.RotatingFileHandler(
                query_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            query_handler = logging.FileHandler(query_log_file)
        
        query_handler.setLevel(logging.INFO)
        query_handler.setFormatter(formatter)
        query_handler.addFilter(context_filter)
        query_logger.addHandler(query_handler)
        query_logger.setLevel(logging.INFO)
        
        # Performance log
        perf_log_file = log_dir / "performance.log"
        perf_logger = logging.getLogger("ehs_analytics.performance")
        
        if enable_rotation:
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            perf_handler = logging.FileHandler(perf_log_file)
        
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)
        perf_handler.addFilter(context_filter)
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
    
    return context_filter


def get_ehs_logger(name: str) -> EHSLogger:
    """
    Get an EHS-specific logger instance.
    
    Args:
        name: Logger name (typically module name)
    
    Returns:
        EHSLogger instance configured for EHS operations
    """
    # Get or create context filter
    root_logger = logging.getLogger()
    context_filter = None
    
    for handler in root_logger.handlers:
        for filter_obj in handler.filters:
            if isinstance(filter_obj, ContextFilter):
                context_filter = filter_obj
                break
    
    return EHSLogger(name, context_filter)


def configure_external_loggers():
    """Configure logging for external libraries to reduce noise."""
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_request_context(
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None
) -> LogContext:
    """
    Create a logging context for a request.
    
    Args:
        user_id: User identifier
        request_id: Request identifier (generates UUID if not provided)
        session_id: Session identifier
        component: Component name
        operation: Operation name
    
    Returns:
        LogContext instance
    """
    return LogContext(
        user_id=user_id,
        request_id=request_id or str(uuid.uuid4()),
        session_id=session_id,
        component=component,
        operation=operation,
        trace_id=str(uuid.uuid4())
    )


# Initialize logging when module is imported
_context_filter = None

def init_logging():
    """Initialize logging with default configuration."""
    global _context_filter
    if _context_filter is None:
        _context_filter = setup_logging()
        configure_external_loggers()

# Auto-initialize with basic config
init_logging()