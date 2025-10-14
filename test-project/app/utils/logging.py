"""
Logging configuration and utilities.

This module provides structured logging configuration with
JSON formatting, correlation IDs, and performance tracking.
"""

import logging
import sys
import json
import time
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar
import uuid

from app.core.config import get_settings

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)

# Global logger instance
_logger: Optional[logging.Logger] = None


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add correlation ID if available
        if correlation_id.get():
            log_data['correlation_id'] = correlation_id.get()
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            ]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        record.correlation_id = correlation_id.get()
        return True


def setup_logging(level: Optional[str] = None) -> None:
    """
    Setup application logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _logger
    
    settings = get_settings()
    log_level = level or settings.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Set formatter based on environment
    if settings.LOG_FORMAT == 'json':
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    
    # Add correlation ID filter
    console_handler.addFilter(CorrelationIDFilter())
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    _logger = logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id_value: Optional[str] = None) -> str:
    """
    Set correlation ID for request tracking.
    
    Args:
        correlation_id_value: Correlation ID value (generates new if None)
        
    Returns:
        Correlation ID
    """
    if correlation_id_value is None:
        correlation_id_value = str(uuid.uuid4())
    
    correlation_id.set(correlation_id_value)
    return correlation_id_value


def get_correlation_id() -> Optional[str]:
    """
    Get current correlation ID.
    
    Returns:
        Correlation ID or None
    """
    return correlation_id.get()


def log_performance(func_name: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics.
    
    Args:
        func_name: Function name
        duration: Duration in seconds
        **kwargs: Additional metrics
    """
    logger = get_logger("performance")
    logger.info(
        f"Performance: {func_name}",
        extra={
            "performance": True,
            "function": func_name,
            "duration_seconds": duration,
            **kwargs
        }
    )


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_id: Optional[int] = None,
    **kwargs
) -> None:
    """
    Log API request.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in seconds
        user_id: User ID (if authenticated)
        **kwargs: Additional request data
    """
    logger = get_logger("api")
    logger.info(
        f"API Request: {method} {path}",
        extra={
            "api_request": True,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_seconds": duration,
            "user_id": user_id,
            **kwargs
        }
    )


def log_database_query(
    query: str,
    duration: float,
    rows_affected: Optional[int] = None,
    **kwargs
) -> None:
    """
    Log database query.
    
    Args:
        query: SQL query
        duration: Query duration in seconds
        rows_affected: Number of rows affected
        **kwargs: Additional query data
    """
    logger = get_logger("database")
    logger.debug(
        f"Database Query: {query[:100]}...",
        extra={
            "database_query": True,
            "query": query,
            "duration_seconds": duration,
            "rows_affected": rows_affected,
            **kwargs
        }
    )


def log_security_event(
    event_type: str,
    user_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log security event.
    
    Args:
        event_type: Type of security event
        user_id: User ID (if applicable)
        ip_address: Client IP address
        user_agent: Client user agent
        **kwargs: Additional event data
    """
    logger = get_logger("security")
    logger.warning(
        f"Security Event: {event_type}",
        extra={
            "security_event": True,
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            **kwargs
        }
    )


def log_business_event(
    event_type: str,
    user_id: Optional[int] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log business event.
    
    Args:
        event_type: Type of business event
        user_id: User ID (if applicable)
        resource_type: Type of resource affected
        resource_id: ID of resource affected
        **kwargs: Additional event data
    """
    logger = get_logger("business")
    logger.info(
        f"Business Event: {event_type}",
        extra={
            "business_event": True,
            "event_type": event_type,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            **kwargs
        }
    )


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, func_name: str, **kwargs):
        self.func_name = func_name
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        log_performance(self.func_name, duration, **self.kwargs)


def performance_logger(func_name: str, **kwargs):
    """
    Decorator for performance logging.
    
    Args:
        func_name: Function name
        **kwargs: Additional metrics
        
    Returns:
        Decorator function
    """
    def decorator(func):
        async def async_wrapper(*args, **func_kwargs):
            with PerformanceLogger(func_name, **kwargs):
                return await func(*args, **func_kwargs)
        
        def sync_wrapper(*args, **func_kwargs):
            with PerformanceLogger(func_name, **kwargs):
                return func(*args, **func_kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
