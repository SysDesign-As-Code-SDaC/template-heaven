"""
Model Trainer

This module provides training orchestration and management for AI models.
It handles the training loop, callbacks, logging, and checkpointing.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
from datetime import datetime

from .model_factory import ModelFactory
from ..utils.config_utils import ConfigManager


class TrainingCallback:
    """
    Base class for training callbacks.

    Callbacks can be used to monitor training progress, save checkpoints,
    adjust learning rates, and perform other training-related tasks.
    """

    def on_training_start(self, trainer: 'Trainer', logs: Dict[str, Any] = None) -> None:
        """Called at the start of training."""
        pass

    def on_training_end(self, trainer: 'Trainer', logs: Dict[str, Any] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer: 'Trainer', batch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer: 'Trainer', batch: int, logs: Dict[str, Any] = None) -> None:
        """Called at the end of each batch."""
        pass


class ModelCheckpoint(TrainingCallback):
    """Callback for saving model checkpoints."""

    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min',
                 save_best_only: bool = True, save_weights_only: bool = False):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        # Check if this is the best value
        is_best = ((self.mode == 'min' and current_value < self.best_value) or
                  (self.mode == 'max' and current_value > self.best_value))

        if is_best or not self.save_best_only:
            self.best_value = current_value
            filepath = self.filepath.format(epoch=epoch, **logs)

            try:
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(filepath, weights_only=self.save_weights_only)
                trainer.logger.info(f"Saved checkpoint: {filepath}")
            except Exception as e:
                trainer.logger.error(f"Failed to save checkpoint: {e}")


class EarlyStopping(TrainingCallback):
    """Callback for early stopping based on metric monitoring."""

    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 10,
                 min_delta: float = 0.001, restore_best_weights: bool = True):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, Any] = None) -> None:
        logs = logs or {}
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        # Check if improvement
        if ((self.mode == 'min' and current_value < self.best_value - self.min_delta) or
            (self.mode == 'max' and current_value > self.best_value + self.min_delta)):
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = trainer.model._get_model_state()
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            trainer.logger.info(f"Early stopping at epoch {epoch}")

            if self.restore_best_weights and self.best_weights is not None:
                trainer.model._set_model_state(self.best_weights)
                trainer.logger.info("Restored best weights")


class Trainer:
    """
    Model trainer with comprehensive training orchestration.

    This class handles the complete training process including:
    - Model building and compilation
    - Training loop with callbacks
    - Validation and evaluation
    - Logging and monitoring
    - Checkpointing and early stopping
    """

    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        Initialize the trainer.

        Args:
            model: Model instance to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.history = {
            'loss': [],
            'val_loss': [],
            'metrics': [],
            'val_metrics': []
        }

        # Callbacks
        self.callbacks = []
        self._setup_callbacks()

        # Training parameters
        self.epochs = config.get('training', {}).get('epochs', 10)
        self.batch_size = config.get('training', {}).get('batch_size', 32)
        self.validation_freq = config.get('validation', {}).get('validation_freq', 1)

    def _setup_callbacks(self) -> None:
        """Setup training callbacks from configuration."""
        callbacks_config = self.config.get('callbacks', {})

        # Model checkpoint
        if callbacks_config.get('model_checkpoint', {}).get('enabled', False):
            checkpoint_config = callbacks_config['model_checkpoint']
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_config.get('filepath', 'models/checkpoints/model_{epoch:02d}.pkl'),
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                mode=checkpoint_config.get('mode', 'min'),
                save_best_only=checkpoint_config.get('save_best_only', True),
                save_weights_only=checkpoint_config.get('save_weights_only', False)
            )
            self.callbacks.append(checkpoint)

        # Early stopping
        early_stop_config = self.config.get('training', {}).get('early_stopping', {})
        if early_stop_config.get('enabled', False):
            early_stop = EarlyStopping(
                monitor=early_stop_config.get('monitor', 'val_loss'),
                mode=early_stop_config.get('mode', 'min'),
                patience=early_stop_config.get('patience', 10),
                min_delta=early_stop_config.get('min_delta', 0.001),
                restore_best_weights=early_stop_config.get('restore_best_weights', True)
            )
            self.callbacks.append(early_stop)

    def add_callback(self, callback: TrainingCallback) -> None:
        """
        Add a training callback.

        Args:
            callback: Callback instance to add
        """
        self.callbacks.append(callback)

    def train(self, X_train: Any, y_train: Any, X_val: Optional[Any] = None,
              y_val: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training arguments

        Returns:
            Training history and results
        """
        self.logger.info("Starting model training...")

        # Build model if not already built
        if not hasattr(self.model, '_model'):
            self.model.build()

        # Prepare data
        train_data = self._prepare_data(X_train, y_train, self.batch_size)
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = self._prepare_data(X_val, y_val, self.batch_size)

        # Training loop
        start_time = time.time()

        # Call training start callbacks
        self._call_callbacks('on_training_start')

        try:
            for epoch in range(self.epochs):
                if self.should_stop:
                    break

                self.current_epoch = epoch

                # Call epoch start callbacks
                self._call_callbacks('on_epoch_start', epoch=epoch)

                # Train for one epoch
                epoch_start_time = time.time()
                train_results = self._train_epoch(train_data)

                # Validation
                val_results = None
                if val_data is not None and epoch % self.validation_freq == 0:
                    val_results = self._validate_epoch(val_data)

                epoch_time = time.time() - epoch_start_time

                # Log progress
                self._log_epoch_progress(epoch, train_results, val_results, epoch_time)

                # Update history
                self._update_history(train_results, val_results)

                # Call epoch end callbacks
                logs = {**train_results}
                if val_results:
                    logs.update({f'val_{k}': v for k, v in val_results.items()})
                self._call_callbacks('on_epoch_end', epoch=epoch, logs=logs)

            # Call training end callbacks
            self._call_callbacks('on_training_end')

            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")

            return {
                'history': self.history,
                'training_time': training_time,
                'epochs_completed': self.current_epoch + 1,
                'final_metrics': self._get_final_metrics()
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e

    def _prepare_data(self, X: Any, y: Any, batch_size: int) -> List[Tuple[Any, Any]]:
        """
        Prepare data for training by creating batches.

        Args:
            X: Features
            y: Targets
            batch_size: Batch size

        Returns:
            List of (X_batch, y_batch) tuples
        """
        # Convert to numpy arrays if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        # Create batches
        batches = []
        n_samples = len(X)

        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            batches.append((X_batch, y_batch))

        return batches

    def _train_epoch(self, train_data: List[Tuple[Any, Any]]) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_data: List of training batches

        Returns:
            Training metrics for the epoch
        """
        epoch_loss = 0.0
        epoch_metrics = {}

        for batch_idx, (X_batch, y_batch) in enumerate(train_data):
            # Call batch start callbacks
            self._call_callbacks('on_batch_start', batch=batch_idx)

            # Train on batch
            batch_results = self.model.train(X_batch, y_batch)

            # Accumulate metrics
            if 'loss' in batch_results:
                epoch_loss += batch_results['loss']
            for key, value in batch_results.items():
                if key != 'loss':
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value

            self.global_step += 1

            # Call batch end callbacks
            self._call_callbacks('on_batch_end', batch=batch_idx, logs=batch_results)

        # Average metrics over batches
        n_batches = len(train_data)
        epoch_results = {'loss': epoch_loss / n_batches}
        epoch_results.update({k: v / n_batches for k, v in epoch_metrics.items()})

        return epoch_results

    def _validate_epoch(self, val_data: List[Tuple[Any, Any]]) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_data: List of validation batches

        Returns:
            Validation metrics for the epoch
        """
        val_metrics = {}

        for X_batch, y_batch in val_data:
            # Evaluate on batch
            batch_metrics = self.model.evaluate(X_batch, y_batch)

            # Accumulate metrics
            for key, value in batch_metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = 0.0
                val_metrics[key] += value

        # Average metrics over batches
        n_batches = len(val_data)
        return {k: v / n_batches for k, v in val_metrics.items()}

    def _log_epoch_progress(self, epoch: int, train_results: Dict[str, float],
                           val_results: Optional[Dict[str, float]], epoch_time: float) -> None:
        """Log training progress for an epoch."""
        log_msg = f"Epoch {epoch + 1}/{self.epochs} - "

        # Training metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_results.items()])
        log_msg += f"train: {{{metrics_str}}}"

        # Validation metrics
        if val_results:
            val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_results.items()])
            log_msg += f" - val: {{{val_metrics_str}}}"

        log_msg += f" - {epoch_time:.2f}s/epoch"

        self.logger.info(log_msg)

    def _update_history(self, train_results: Dict[str, float],
                       val_results: Optional[Dict[str, float]]) -> None:
        """Update training history."""
        self.history['loss'].append(train_results.get('loss', 0.0))

        if val_results:
            self.history['val_loss'].append(val_results.get('loss', 0.0))
            self.history['val_metrics'].append(val_results)
        else:
            self.history['val_loss'].append(None)
            self.history['val_metrics'].append({})

        self.history['metrics'].append(train_results)

    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final training metrics."""
        final_metrics = {}

        if self.history['loss']:
            final_metrics['final_loss'] = self.history['loss'][-1]

        if self.history['val_loss'] and self.history['val_loss'][-1] is not None:
            final_metrics['final_val_loss'] = self.history['val_loss'][-1]

        # Get latest metrics
        if self.history['metrics']:
            final_metrics.update(self.history['metrics'][-1])

        if self.history['val_metrics'] and self.history['val_metrics'][-1]:
            final_metrics.update({f'val_{k}': v for k, v in self.history['val_metrics'][-1].items()})

        return final_metrics

    def _call_callbacks(self, method_name: str, **kwargs) -> None:
        """Call a method on all callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    getattr(callback, method_name)(self, **kwargs)
                except Exception as e:
                    self.logger.error(f"Callback {callback.__class__.__name__} failed in {method_name}: {e}")

    def save_checkpoint(self, filepath: str, weights_only: bool = False) -> None:
        """
        Save a training checkpoint.

        Args:
            filepath: Path to save checkpoint
            weights_only: Whether to save only model weights
        """
        checkpoint_data = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_config': self.model.config,
            'training_config': self.config,
            'history': self.history,
            'timestamp': datetime.utcnow().isoformat()
        }

        if weights_only:
            # Save only model state
            checkpoint_data['model_state'] = self.model._get_model_state()
        else:
            # Save full model
            self.model.save(filepath.replace('.pkl', '_model.pkl'))

        # Save checkpoint metadata
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        self.logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a training checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        import pickle
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)

        self.current_epoch = checkpoint_data['epoch']
        self.global_step = checkpoint_data['global_step']
        self.history = checkpoint_data['history']

        if 'model_state' in checkpoint_data:
            self.model._set_model_state(checkpoint_data['model_state'])

        self.logger.info(f"Checkpoint loaded: {filepath}")
