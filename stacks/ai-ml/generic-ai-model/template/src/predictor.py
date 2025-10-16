"""
Model Predictor

This module provides inference capabilities for trained AI models including
batch processing, preprocessing, and result formatting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class ModelPredictor:
    """
    Model prediction and inference management.

    This class handles model inference with support for:
    - Single and batch predictions
    - Input preprocessing and validation
    - Result postprocessing and formatting
    - Performance monitoring
    - Error handling and fallbacks
    """

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor.

        Args:
            model: Trained model for predictions
            config: Prediction configuration
        """
        self.model = model
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Prediction statistics
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.error_count = 0

        # Configuration
        self.batch_size = self.config.get('batch_size', 32)
        self.max_batch_size = self.config.get('max_batch_size', 1000)
        self.timeout = self.config.get('timeout', 30.0)
        self.enable_monitoring = self.config.get('enable_monitoring', True)

        # Input validation
        self.input_validation = self.config.get('input_validation', True)
        self.expected_features = self.config.get('expected_features')

        # Output formatting
        self.output_format = self.config.get('output_format', 'default')
        self.include_probabilities = self.config.get('include_probabilities', False)
        self.include_confidence = self.config.get('include_confidence', False)

    def predict(self, X: Any, **kwargs) -> Dict[str, Any]:
        """
        Make predictions on input data.

        Args:
            X: Input features for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Prediction results with metadata
        """
        start_time = time.time()

        try:
            # Validate input
            if self.input_validation:
                self._validate_input(X)

            # Preprocess input
            X_processed = self._preprocess_input(X)

            # Make prediction
            predictions = self.model.predict(X_processed)

            # Postprocess results
            results = self._postprocess_predictions(predictions, X, **kwargs)

            # Update statistics
            prediction_time = time.time() - start_time
            self._update_statistics(prediction_time)

            if self.enable_monitoring:
                self.logger.debug(f"Prediction completed in {prediction_time:.4f}s")

            return results

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Prediction failed: {e}")
            return self._handle_prediction_error(e, X)

    def predict_batch(self, X_batch: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Make batch predictions.

        Args:
            X_batch: List of input features
            **kwargs: Additional prediction parameters

        Returns:
            List of prediction results
        """
        if not X_batch:
            return []

        # Check if batch processing is needed
        if len(X_batch) <= self.batch_size:
            # Process as single batch
            return [self.predict(X_batch, **kwargs)]

        # Split into smaller batches
        results = []
        for i in range(0, len(X_batch), self.batch_size):
            batch = X_batch[i:i + self.batch_size]
            batch_result = self.predict(batch, **kwargs)
            results.append(batch_result)

        return results

    def predict_async(self, X: Any, **kwargs) -> Any:
        """
        Make asynchronous predictions.

        Args:
            X: Input features for prediction
            **kwargs: Additional prediction parameters

        Returns:
            Future object for prediction results
        """
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.predict, X, **kwargs)

        # Clean up executor (optional, will be garbage collected)
        executor.shutdown(wait=False)

        return future

    def _validate_input(self, X: Any) -> None:
        """
        Validate input data.

        Args:
            X: Input data to validate

        Raises:
            ValueError: If input validation fails
        """
        if X is None:
            raise ValueError("Input data cannot be None")

        # Convert to numpy array for validation
        try:
            X_array = np.array(X)
        except Exception:
            raise ValueError("Input data must be convertible to numpy array")

        # Check dimensions
        if X_array.ndim == 0:
            raise ValueError("Input must be at least 1-dimensional")

        # Check feature count
        if self.expected_features:
            if X_array.ndim == 1:
                n_features = 1
            else:
                n_features = X_array.shape[1]

            if n_features != self.expected_features:
                raise ValueError(f"Expected {self.expected_features} features, got {n_features}")

        # Check for NaN/inf values
        if np.any(~np.isfinite(X_array)):
            raise ValueError("Input contains NaN or infinite values")

    def _preprocess_input(self, X: Any) -> Any:
        """
        Preprocess input data before prediction.

        Args:
            X: Raw input data

        Returns:
            Preprocessed input data
        """
        # Convert to numpy array
        X_processed = np.array(X)

        # Ensure 2D array
        if X_processed.ndim == 1:
            X_processed = X_processed.reshape(1, -1)

        # Apply any configured preprocessing
        preprocessing_config = self.config.get('preprocessing', {})

        # Scaling
        if preprocessing_config.get('scaling') == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)

        elif preprocessing_config.get('scaling') == 'min_max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(X_processed)

        # Handle categorical features if needed
        # (This would be more complex in practice)

        return X_processed

    def _postprocess_predictions(self, predictions: Any, original_input: Any, **kwargs) -> Dict[str, Any]:
        """
        Postprocess prediction results.

        Args:
            predictions: Raw model predictions
            original_input: Original input data
            **kwargs: Additional processing parameters

        Returns:
            Formatted prediction results
        """
        results = {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'timestamp': datetime.utcnow().isoformat(),
            'model_info': {
                'type': getattr(self.model, 'model_type', 'unknown'),
                'framework': getattr(self.model, 'framework', 'unknown')
            }
        }

        # Add probabilities if available and requested
        if self.include_probabilities and hasattr(predictions, 'shape') and predictions.shape[1] > 1:
            results['probabilities'] = predictions.tolist()

            # Add predicted classes for classification
            predicted_classes = np.argmax(predictions, axis=1)
            results['predicted_classes'] = predicted_classes.tolist()

            # Add confidence scores
            if self.include_confidence:
                confidence_scores = np.max(predictions, axis=1)
                results['confidence_scores'] = confidence_scores.tolist()

        # Add prediction metadata
        results['metadata'] = {
            'input_shape': np.array(original_input).shape,
            'output_shape': np.array(predictions).shape,
            'batch_size': len(original_input) if hasattr(original_input, '__len__') else 1
        }

        # Format output based on configuration
        if self.output_format == 'minimal':
            results = {'predictions': results['predictions']}
        elif self.output_format == 'detailed':
            # Add more detailed information
            results['processing_time'] = kwargs.get('processing_time', 0.0)
            results['model_version'] = getattr(self.model, 'metadata', {}).get('version', 'unknown')

        return results

    def _handle_prediction_error(self, error: Exception, input_data: Any) -> Dict[str, Any]:
        """
        Handle prediction errors gracefully.

        Args:
            error: The exception that occurred
            input_data: The input data that caused the error

        Returns:
            Error response
        """
        error_response = {
            'error': str(error),
            'error_type': type(error).__name__,
            'timestamp': datetime.utcnow().isoformat(),
            'input_shape': np.array(input_data).shape if input_data is not None else None,
            'success': False
        }

        # Add fallback predictions if configured
        if self.config.get('fallback_enabled', False):
            try:
                fallback_predictions = self._generate_fallback_predictions(input_data)
                error_response['fallback_predictions'] = fallback_predictions
            except Exception:
                pass  # Fallback failed, return error only

        return error_response

    def _generate_fallback_predictions(self, input_data: Any) -> Any:
        """
        Generate fallback predictions when model fails.

        Args:
            input_data: Input data for fallback prediction

        Returns:
            Fallback predictions
        """
        # Simple fallback: return zeros or means
        input_shape = np.array(input_data).shape

        if len(input_shape) == 1:
            # Single prediction
            return [0.0]
        else:
            # Batch prediction
            return [0.0] * input_shape[0]

    def _update_statistics(self, prediction_time: float) -> None:
        """
        Update prediction statistics.

        Args:
            prediction_time: Time taken for prediction
        """
        self.prediction_count += 1
        self.total_prediction_time += prediction_time

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get prediction statistics.

        Returns:
            Dictionary of prediction statistics
        """
        avg_time = self.total_prediction_time / self.prediction_count if self.prediction_count > 0 else 0.0

        return {
            'total_predictions': self.prediction_count,
            'total_errors': self.error_count,
            'average_prediction_time': avg_time,
            'total_prediction_time': self.total_prediction_time,
            'error_rate': self.error_count / self.prediction_count if self.prediction_count > 0 else 0.0
        }

    def reset_statistics(self) -> None:
        """Reset prediction statistics."""
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.error_count = 0

    def save_predictions(self, predictions: Dict[str, Any], filepath: str,
                        format: str = 'json') -> None:
        """
        Save predictions to file.

        Args:
            predictions: Prediction results to save
            filepath: Path to save predictions
            format: File format ('json', 'csv', 'pickle')
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(predictions)
            df.to_csv(filepath, index=False)
        elif format == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(predictions, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Predictions saved to: {filepath}")

    def load_predictions(self, filepath: str, format: str = 'json') -> Dict[str, Any]:
        """
        Load predictions from file.

        Args:
            filepath: Path to load predictions from
            format: File format

        Returns:
            Loaded predictions
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Predictions file not found: {filepath}")

        if format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format == 'csv':
            import pandas as pd
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        elif format == 'pickle':
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")


class BatchPredictor:
    """
    High-performance batch prediction handler.

    This class provides optimized batch processing capabilities
    with parallel execution and memory management.
    """

    def __init__(self, predictor: ModelPredictor, max_workers: int = 4):
        """
        Initialize batch predictor.

        Args:
            predictor: ModelPredictor instance
            max_workers: Maximum number of worker threads
        """
        self.predictor = predictor
        self.max_workers = max_workers
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict_large_batch(self, X_batch: List[Any], batch_size: Optional[int] = None,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Predict on a large batch of data with progress tracking.

        Args:
            X_batch: Large batch of input data
            batch_size: Size of individual prediction batches
            show_progress: Whether to show progress information

        Returns:
            List of prediction results
        """
        if batch_size is None:
            batch_size = self.predictor.batch_size

        results = []
        total_batches = len(X_batch) // batch_size + (1 if len(X_batch) % batch_size else 0)

        for i in range(0, len(X_batch), batch_size):
            batch = X_batch[i:i + batch_size]

            if show_progress:
                progress = (i // batch_size + 1) / total_batches * 100
                self.logger.info(".1f")

            batch_result = self.predictor.predict_batch(batch)
            results.extend(batch_result)

        return results

    def predict_parallel(self, X_batch: List[Any]) -> List[Dict[str, Any]]:
        """
        Predict on multiple inputs in parallel.

        Args:
            X_batch: List of input data

        Returns:
            List of prediction results
        """
        results = [None] * len(X_batch)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all prediction tasks
            future_to_index = {
                executor.submit(self.predictor.predict, X): i
                for i, X in enumerate(X_batch)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    self.logger.error(f"Prediction failed for item {index}: {e}")
                    results[index] = self.predictor._handle_prediction_error(e, X_batch[index])

        return results
