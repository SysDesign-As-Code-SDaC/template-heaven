"""
Base AI Model Interface

This module defines the abstract base class for all AI models in the template.
All model implementations must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime


class BaseModel(ABC):
    """
    Abstract base class for all AI models.

    This class defines the interface that all model implementations must follow.
    It provides common functionality for model management, serialization, and validation.

    Attributes:
        config (Dict[str, Any]): Model configuration dictionary
        logger (logging.Logger): Logger instance for the model
        metadata (Dict[str, Any]): Model metadata including version, creation date, etc.
        is_trained (bool): Flag indicating if the model has been trained
        model_type (str): Type of the model (e.g., 'neural_network', 'tree_model')
        framework (str): ML framework used (e.g., 'tensorflow', 'pytorch', 'sklearn')
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metadata = {
            "model_type": self.__class__.__name__,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "framework": config.get("framework", "unknown"),
            "config_hash": hash(json.dumps(config, sort_keys=True))
        }
        self.is_trained = False
        self.model_type = self._get_model_type()
        self.framework = config.get("framework", "unknown")

        # Validate configuration
        self._validate_config()

    @abstractmethod
    def _get_model_type(self) -> str:
        """
        Get the model type identifier.

        Returns:
            String identifier for the model type
        """
        pass

    def _validate_config(self) -> None:
        """
        Validate the model configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ["framework"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate framework
        supported_frameworks = ["tensorflow", "pytorch", "sklearn", "custom"]
        if self.framework not in supported_frameworks:
            self.logger.warning(f"Framework '{self.framework}' may not be fully supported")

    @abstractmethod
    def build(self) -> None:
        """
        Build the model architecture.

        This method should initialize the model with the specified configuration.
        Must be implemented by all concrete model classes.
        """
        pass

    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training history and metrics

        Raises:
            RuntimeError: If model is not built or training fails
        """
        if not hasattr(self, '_model'):
            raise RuntimeError("Model must be built before training. Call build() first.")

        self.logger.info("Starting model training...")
        start_time = datetime.utcnow()

        try:
            # Implement training logic in concrete classes
            training_result = self._train_implementation(X, y, **kwargs)

            # Update metadata
            self.metadata["trained_at"] = datetime.utcnow().isoformat()
            self.metadata["training_duration"] = (datetime.utcnow() - start_time).total_seconds()
            self.is_trained = True

            self.logger.info("Model training completed successfully")
            return training_result

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}") from e

    @abstractmethod
    def _train_implementation(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Implementation-specific training logic.

        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Make predictions using the trained model.

        Args:
            X: Input features for prediction

        Returns:
            Model predictions

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        try:
            predictions = self._predict_implementation(X)
            self.logger.debug(f"Made predictions for {len(X) if hasattr(X, '__len__') else 'input'} samples")
            return predictions
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    @abstractmethod
    def _predict_implementation(self, X: Any) -> Any:
        """
        Implementation-specific prediction logic.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Dictionary containing evaluation metrics

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        try:
            metrics = self._evaluate_implementation(X, y)
            self.logger.info(f"Model evaluation completed: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Model evaluation failed: {str(e)}") from e

    @abstractmethod
    def _evaluate_implementation(self, X: Any, y: Any) -> Dict[str, float]:
        """
        Implementation-specific evaluation logic.

        Args:
            X: Test features
            y: Test targets

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path where to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save model state
            model_data = {
                "model_state": self._get_model_state(),
                "config": self.config,
                "metadata": self.metadata,
                "is_trained": self.is_trained
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            raise RuntimeError(f"Model saving failed: {str(e)}") from e

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load a model from disk.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If loading fails
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            # Create model instance
            instance = cls(model_data["config"])

            # Restore model state
            instance._set_model_state(model_data["model_state"])
            instance.metadata = model_data["metadata"]
            instance.is_trained = model_data["is_trained"]

            instance.logger.info(f"Model loaded from {filepath}")
            return instance

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    @abstractmethod
    def _get_model_state(self) -> Any:
        """
        Get the current model state for serialization.

        Returns:
            Model state that can be serialized
        """
        pass

    @abstractmethod
    def _set_model_state(self, state: Any) -> None:
        """
        Set the model state from serialized data.

        Args:
            state: Previously serialized model state
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dictionary containing model metadata
        """
        return self.metadata.copy()

    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update a metadata field.

        Args:
            key: Metadata key to update
            value: New value for the metadata key
        """
        self.metadata[key] = value
        self.logger.debug(f"Updated metadata {key}: {value}")

    def validate_input(self, X: Any) -> bool:
        """
        Validate input data format.

        Args:
            X: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        try:
            self._validate_input_implementation(X)
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            return False

    def _validate_input_implementation(self, X: Any) -> None:
        """
        Implementation-specific input validation.

        Args:
            X: Input data to validate

        Raises:
            ValueError: If input is invalid
        """
        # Default implementation - can be overridden
        if X is None:
            raise ValueError("Input data cannot be None")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.

        Returns:
            Dictionary containing model summary information
        """
        return {
            "model_type": self.model_type,
            "framework": self.framework,
            "is_trained": self.is_trained,
            "config": self.config,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(type={self.model_type}, framework={self.framework}, trained={self.is_trained})"

    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (f"{self.__class__.__name__}("
                f"model_type='{self.model_type}', "
                f"framework='{self.framework}', "
                f"trained={self.is_trained}, "
                f"config_keys={list(self.config.keys())})")
