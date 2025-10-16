"""
Linear Model Implementation

This module provides linear model implementations using scikit-learn
and other frameworks based on configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base_model import BaseModel


class LinearModel(BaseModel):
    """
    Linear model implementation supporting multiple frameworks.

    This class provides a unified interface for linear models that can work
    with scikit-learn or other frameworks based on configuration.

    Attributes:
        framework (str): The framework to use ('sklearn', etc.)
        linear_config (Dict[str, Any]): Linear model configuration
        _model: Internal model object (framework-specific)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the linear model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)

        self.framework = config.get("framework", "sklearn")
        self.linear_config = config.get("linear_model", {})

        # Validate framework support
        supported_frameworks = ["sklearn"]
        if self.framework not in supported_frameworks:
            raise ValueError(f"Unsupported framework for linear models: {self.framework}. "
                           f"Supported: {supported_frameworks}")

        self.logger.info(f"Initialized linear model with {self.framework} framework")

    def _get_model_type(self) -> str:
        """Get the model type identifier."""
        return "linear_model"

    def build(self) -> None:
        """
        Build the linear model.

        This method constructs the linear model based on the framework
        and configuration.
        """
        try:
            if self.framework == "sklearn":
                self._build_sklearn_model()
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

            self.logger.info(f"Built {self.framework} linear model")

        except Exception as e:
            self.logger.error(f"Failed to build linear model: {str(e)}")
            raise RuntimeError(f"Linear model build failed: {str(e)}") from e

    def _build_sklearn_model(self) -> None:
        """Build scikit-learn linear model."""
        try:
            from sklearn.linear_model import (
                LogisticRegression, LinearRegression, Ridge, Lasso,
                SGDClassifier, SGDRegressor
            )
        except ImportError:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")

        model_type = self.linear_config.get("type", "logistic_regression")

        if model_type == "logistic_regression":
            self._model = LogisticRegression(**self.linear_config.get("params", {}))
        elif model_type == "linear_regression":
            self._model = LinearRegression(**self.linear_config.get("params", {}))
        elif model_type == "ridge":
            self._model = Ridge(**self.linear_config.get("params", {}))
        elif model_type == "lasso":
            self._model = Lasso(**self.linear_config.get("params", {}))
        elif model_type == "sgd_classifier":
            self._model = SGDClassifier(**self.linear_config.get("params", {}))
        elif model_type == "sgd_regressor":
            self._model = SGDRegressor(**self.linear_config.get("params", {}))
        else:
            raise ValueError(f"Unsupported sklearn linear model type: {model_type}")

    def _determine_task_type(self) -> str:
        """Determine if this is a classification or regression task."""
        model_type = self.linear_config.get("type", "logistic_regression")

        # Classification models
        classification_models = ["logistic_regression", "sgd_classifier"]

        if model_type in classification_models:
            return "classification"
        else:
            return "regression"

    def _train_implementation(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Implementation-specific training logic."""
        # Convert inputs to appropriate format
        X = self._prepare_input_data(X)
        y = self._prepare_target_data(y)

        # Train the model
        self._model.fit(X, y)

        # Get training information
        training_info = {
            "n_features": X.shape[1] if hasattr(X, 'shape') else len(X[0]),
            "n_samples": len(X),
            "model_type": type(self._model).__name__
        }

        # Get coefficients if available
        if hasattr(self._model, 'coef_'):
            training_info["coefficients"] = self._model.coef_.tolist()

        if hasattr(self._model, 'intercept_'):
            training_info["intercept"] = float(self._model.intercept_) if np.isscalar(self._model.intercept_) else self._model.intercept_.tolist()

        self.logger.info(f"Trained linear model on {training_info['n_samples']} samples")

        return {"training_info": training_info}

    def _prepare_input_data(self, X: Any) -> Any:
        """Prepare input data for training/prediction."""
        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def _prepare_target_data(self, y: Any) -> Any:
        """Prepare target data for training."""
        # Convert to numpy array if needed
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Ensure 1D array for most linear models
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        return y

    def _predict_implementation(self, X: Any) -> Any:
        """Implementation-specific prediction logic."""
        X = self._prepare_input_data(X)

        # Make predictions
        if hasattr(self._model, 'predict_proba') and self._determine_task_type() == "classification":
            # Return probabilities for classification
            predictions = self._model.predict_proba(X)
        else:
            # Return class predictions or regression values
            predictions = self._model.predict(X)

        return predictions

    def _evaluate_implementation(self, X: Any, y: Any) -> Dict[str, float]:
        """Implementation-specific evaluation logic."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )

        X = self._prepare_input_data(X)
        y = self._prepare_target_data(y)

        predictions = self._model.predict(X)
        task_type = self._determine_task_type()

        metrics = {}

        if task_type == "classification":
            # Classification metrics
            metrics["accuracy"] = accuracy_score(y, predictions)

            # For binary/multiclass classification
            try:
                metrics["precision"] = precision_score(y, predictions, average='weighted')
                metrics["recall"] = recall_score(y, predictions, average='weighted')
                metrics["f1_score"] = f1_score(y, predictions, average='weighted')
            except:
                # Handle cases where precision/recall might fail
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1_score"] = 0.0

        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(y, predictions)
            metrics["mae"] = mean_absolute_error(y, predictions)
            metrics["r2_score"] = r2_score(y, predictions)

        return metrics

    def _get_model_state(self) -> Any:
        """Get model state for serialization."""
        # For sklearn models, we can use pickle
        import pickle
        return pickle.dumps(self._model)

    def _set_model_state(self, state: Any) -> None:
        """Set model state from serialized data."""
        import pickle
        self._model = pickle.loads(state)

    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get model coefficients if available.

        Returns:
            Array of model coefficients, or None if not available
        """
        if hasattr(self._model, 'coef_'):
            return self._model.coef_
        return None

    def get_intercept(self) -> Optional[Union[float, np.ndarray]]:
        """
        Get model intercept if available.

        Returns:
            Model intercept value(s), or None if not available
        """
        if hasattr(self._model, 'intercept_'):
            return self._model.intercept_
        return None

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        if hasattr(self._model, 'get_params'):
            return self._model.get_params()
        return {}
