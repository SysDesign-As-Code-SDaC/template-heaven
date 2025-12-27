"""
Logger Configuration

This module provides logging configuration and utilities for the AI model template.
It supports multiple log levels, file logging, and structured logging.
"""

import logging
import logging.handlers
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    This formatter outputs log records as JSON objects for better
    parsing and analysis by log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        # Create base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                             'pathname', 'filename', 'module', 'exc_info',
                             'exc_text', 'stack_info', 'lineno', 'funcName',
                             'created', 'msecs', 'relativeCreated', 'thread',
                             'threadName', 'processName', 'process', 'message']:
                    # Convert non-serializable objects to strings
                    try:
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry, ensure_ascii=False)


class ModelLogger:
    """
    Enhanced logger for AI model operations.

    This class provides specialized logging capabilities for machine learning
    workflows including training progress, metrics, and performance monitoring.
    """

    def __init__(self, name: str = "AIModel", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model logger.

        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup logger with handlers and formatters."""
        # Set log level
        level_str = self.config.get('level', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
        self.logger.setLevel(level)

        # Don't propagate to root logger
        self.logger.propagate = False

        # Console handler
        console_config = self.config.get('console', {})
        if console_config.get('enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)

            # Choose formatter
            if self.config.get('format', 'text') == 'json':
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    fmt=self.config.get('format_string',
                                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                    datefmt=self.config.get('date_format', '%Y-%m-%d %H:%M:%S')
                )

            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        file_config = self.config.get('file', {})
        if file_config.get('enabled', True):
            log_file = file_config.get('filename', 'logs/model.log')
            max_bytes = file_config.get('max_size', 10 * 1024 * 1024)  # 10MB
            backup_count = file_config.get('backup_count', 5)

            # Ensure log directory exists
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(level)

            # Use JSON formatter for file logs
            json_formatter = JSONFormatter()
            file_handler.setFormatter(json_formatter)

            self.logger.addHandler(file_handler)

    def log_training_start(self, config: Dict[str, Any]) -> None:
        """
        Log training start event.

        Args:
            config: Training configuration
        """
        self.logger.info("Training started", extra={
            "event": "training_start",
            "model_type": config.get("model", {}).get("type"),
            "framework": config.get("model", {}).get("framework"),
            "epochs": config.get("training", {}).get("epochs"),
            "batch_size": config.get("training", {}).get("batch_size")
        })

    def log_training_end(self, results: Dict[str, Any]) -> None:
        """
        Log training end event.

        Args:
            results: Training results
        """
        self.logger.info("Training completed", extra={
            "event": "training_end",
            "final_loss": results.get("final_loss"),
            "final_val_loss": results.get("final_val_loss"),
            "epochs_completed": results.get("epochs_completed"),
            "training_time": results.get("training_time")
        })

    def log_epoch_progress(self, epoch: int, metrics: Dict[str, float],
                          val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log epoch training progress.

        Args:
            epoch: Current epoch number
            metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        log_data = {
            "event": "epoch_end",
            "epoch": epoch,
            **metrics
        }

        if val_metrics:
            log_data.update({f"val_{k}": v for k, v in val_metrics.items()})

        self.logger.info(f"Epoch {epoch} completed", extra=log_data)

    def log_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Log model evaluation results.

        Args:
            results: Evaluation results
        """
        self.logger.info("Model evaluation completed", extra={
            "event": "evaluation",
            "task_type": results.get("metadata", {}).get("task_type"),
            "accuracy": results.get("accuracy"),
            "precision": results.get("precision"),
            "recall": results.get("recall"),
            "f1_score": results.get("f1_score"),
            "mse": results.get("mse"),
            "mae": results.get("mae"),
            "r2_score": results.get("r2_score")
        })

    def log_prediction_stats(self, stats: Dict[str, Any]) -> None:
        """
        Log prediction statistics.

        Args:
            stats: Prediction statistics
        """
        self.logger.info("Prediction statistics", extra={
            "event": "prediction_stats",
            **stats
        })

    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log error with context.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        self.logger.error(f"Error in {context}: {str(error)}", extra={
            "event": "error",
            "error_type": type(error).__name__,
            "context": context,
            "error_message": str(error)
        })

    def log_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics.

        Args:
            metrics: Performance metrics
        """
        self.logger.info("Performance metrics", extra={
            "event": "performance",
            **metrics
        })


def setup_logging(config: Optional[Dict[str, Any]] = None,
                 name: str = "AIModel") -> ModelLogger:
    """
    Setup logging for the AI model application.

    Args:
        config: Logging configuration
        name: Logger name

    Returns:
        Configured ModelLogger instance
    """
    return ModelLogger(name, config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Global logger instance
_default_logger = None


def get_default_logger() -> ModelLogger:
    """
    Get the default model logger.

    Returns:
        Default ModelLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = ModelLogger()
    return _default_logger


# Convenience logging functions
def log_training_start(config: Dict[str, Any]) -> None:
    """Log training start."""
    get_default_logger().log_training_start(config)


def log_training_end(results: Dict[str, Any]) -> None:
    """Log training end."""
    get_default_logger().log_training_end(results)


def log_evaluation_results(results: Dict[str, Any]) -> None:
    """Log evaluation results."""
    get_default_logger().log_evaluation_results(results)


def log_error(error: Exception, context: str = "") -> None:
    """Log error."""
    get_default_logger().log_error(error, context)
