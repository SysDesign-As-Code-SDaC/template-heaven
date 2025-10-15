"""
AI Model Template Source

This package contains the core components of the generic AI model template.
"""

from .model_factory import ModelFactory, create_model, register_model, list_model_types
from .trainer import Trainer, TrainingCallback, ModelCheckpoint, EarlyStopping
from .evaluator import ModelEvaluator
from .predictor import ModelPredictor, BatchPredictor
from .data_loader import (
    BaseDataLoader, CSVDataLoader, NumpyDataLoader, JSONDataLoader,
    DataLoaderFactory, load_and_preprocess_data
)
from .logger import ModelLogger, setup_logging, get_logger, log_training_start, log_training_end

__all__ = [
    # Model factory
    'ModelFactory', 'create_model', 'register_model', 'list_model_types',

    # Training
    'Trainer', 'TrainingCallback', 'ModelCheckpoint', 'EarlyStopping',

    # Evaluation
    'ModelEvaluator',

    # Prediction
    'ModelPredictor', 'BatchPredictor',

    # Data loading
    'BaseDataLoader', 'CSVDataLoader', 'NumpyDataLoader', 'JSONDataLoader',
    'DataLoaderFactory', 'load_and_preprocess_data',

    # Logging
    'ModelLogger', 'setup_logging', 'get_logger', 'log_training_start', 'log_training_end'
]
