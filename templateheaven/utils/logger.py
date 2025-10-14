"""
Logging utilities for Template Heaven.

This module provides structured logging functionality with support for
different log levels, colored output, and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_rich: Whether to use rich formatting (if available)
        
    Raises:
        ValueError: If invalid log level is provided
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set level
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if use_rich and RICH_AVAILABLE:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(level)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('git').setLevel(logging.WARNING)


class LoggerMixin:
    """
    Mixin class to add logging functionality to any class.
    
    Provides a logger property that returns a logger instance
    named after the class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = datetime.now() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time.total_seconds():.3f}s")
            return result
        except Exception as e:
            execution_time = datetime.now() - start_time
            logger.error(f"{func.__name__} failed after {execution_time.total_seconds():.3f}s: {e}")
            raise
    
    return wrapper
