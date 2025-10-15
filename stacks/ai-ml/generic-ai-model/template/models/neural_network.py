"""
Neural Network Model Implementation

This module provides neural network model implementations that work with
multiple frameworks (TensorFlow, PyTorch) based on configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Neural network model implementation supporting multiple frameworks.

    This class provides a unified interface for neural networks that can work
    with TensorFlow/Keras, PyTorch, or other frameworks based on configuration.

    Attributes:
        framework (str): The framework to use ('tensorflow', 'pytorch', etc.)
        architecture (Dict[str, Any]): Neural network architecture configuration
        _model: Internal model object (framework-specific)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the neural network model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)

        self.framework = config.get("framework", "tensorflow")
        self.architecture = config.get("neural_network", {}).get("architecture", {})

        # Validate framework support
        supported_frameworks = ["tensorflow", "pytorch"]
        if self.framework not in supported_frameworks:
            raise ValueError(f"Unsupported framework for neural networks: {self.framework}. "
                           f"Supported: {supported_frameworks}")

        self.logger.info(f"Initialized neural network model with {self.framework} framework")

    def _get_model_type(self) -> str:
        """Get the model type identifier."""
        return "neural_network"

    def build(self) -> None:
        """
        Build the neural network architecture.

        This method constructs the neural network based on the framework
        and architecture configuration.
        """
        try:
            if self.framework == "tensorflow":
                self._build_tensorflow_model()
            elif self.framework == "pytorch":
                self._build_pytorch_model()
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")

            self.logger.info(f"Built {self.framework} neural network model")

        except Exception as e:
            self.logger.error(f"Failed to build neural network: {str(e)}")
            raise RuntimeError(f"Neural network build failed: {str(e)}") from e

    def _build_tensorflow_model(self) -> None:
        """Build TensorFlow/Keras model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")

        layers_config = self.architecture.get("layers", [])

        if not layers_config:
            raise ValueError("No layers specified in neural network architecture")

        # Build sequential model
        model = keras.Sequential()

        # Add layers
        for i, layer_config in enumerate(layers_config):
            layer_type = layer_config.get("type", "").lower()

            if layer_type == "dense":
                units = layer_config.get("units", 64)
                activation = layer_config.get("activation", "relu")

                # First layer needs input_shape
                if i == 0:
                    input_shape = layer_config.get("input_shape")
                    if input_shape is None:
                        raise ValueError("input_shape must be specified for the first layer")
                    model.add(keras.layers.Dense(units, activation=activation,
                                               input_shape=input_shape))
                else:
                    model.add(keras.layers.Dense(units, activation=activation))

                # Add dropout if specified
                dropout_rate = layer_config.get("dropout")
                if dropout_rate is not None:
                    model.add(keras.layers.Dropout(dropout_rate))

            elif layer_type == "conv2d":
                filters = layer_config.get("filters", 32)
                kernel_size = layer_config.get("kernel_size", (3, 3))
                activation = layer_config.get("activation", "relu")

                if i == 0:
                    input_shape = layer_config.get("input_shape")
                    if input_shape is None:
                        raise ValueError("input_shape must be specified for the first layer")
                    model.add(keras.layers.Conv2D(filters, kernel_size, activation=activation,
                                                input_shape=input_shape))
                else:
                    model.add(keras.layers.Conv2D(filters, kernel_size, activation=activation))

                # Add pooling if specified
                pool_size = layer_config.get("pool_size")
                if pool_size is not None:
                    model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Compile model
        optimizer_config = self.config.get("neural_network", {}).get("optimizer", "adam")
        loss_config = self.config.get("neural_network", {}).get("loss", "categorical_crossentropy")
        metrics_config = self.config.get("neural_network", {}).get("metrics", ["accuracy"])

        # Configure optimizer
        if isinstance(optimizer_config, str):
            optimizer = optimizer_config.lower()
        elif isinstance(optimizer_config, dict):
            optimizer_name = optimizer_config.get("name", "adam").lower()
            optimizer_params = {k: v for k, v in optimizer_config.items() if k != "name"}
            if optimizer_name == "adam":
                optimizer = tf.keras.optimizers.Adam(**optimizer_params)
            elif optimizer_name == "sgd":
                optimizer = tf.keras.optimizers.SGD(**optimizer_params)
            else:
                optimizer = optimizer_config
        else:
            optimizer = optimizer_config

        model.compile(optimizer=optimizer, loss=loss_config, metrics=metrics_config)
        self._model = model

    def _build_pytorch_model(self) -> None:
        """Build PyTorch model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        class PyTorchNeuralNetwork(nn.Module):
            """PyTorch neural network implementation."""

            def __init__(self, architecture_config):
                super().__init__()
                self.layers = nn.ModuleList()
                self.architecture_config = architecture_config

                # Build layers
                layers_config = architecture_config.get("layers", [])
                for i, layer_config in enumerate(layers_config):
                    layer_type = layer_config.get("type", "").lower()

                    if layer_type == "dense":
                        in_features = layer_config.get("input_shape", [64])[0] if i == 0 else prev_out_features
                        out_features = layer_config.get("units", 64)
                        self.layers.append(nn.Linear(in_features, out_features))

                        activation = layer_config.get("activation", "relu")
                        if activation == "relu":
                            self.layers.append(nn.ReLU())
                        elif activation == "sigmoid":
                            self.layers.append(nn.Sigmoid())
                        elif activation == "tanh":
                            self.layers.append(nn.Tanh())

                        dropout_rate = layer_config.get("dropout")
                        if dropout_rate is not None:
                            self.layers.append(nn.Dropout(dropout_rate))

                        prev_out_features = out_features

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        self._model = PyTorchNeuralNetwork(self.architecture)

    def _train_implementation(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Implementation-specific training logic."""
        if self.framework == "tensorflow":
            return self._train_tensorflow(X, y, **kwargs)
        elif self.framework == "pytorch":
            return self._train_pytorch(X, y, **kwargs)
        else:
            raise ValueError(f"Unsupported framework for training: {self.framework}")

    def _train_tensorflow(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Train TensorFlow model."""
        import tensorflow as tf

        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)
        validation_split = kwargs.get("validation_split", 0.2)
        verbose = kwargs.get("verbose", 1)

        # Convert to numpy arrays if needed
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        # Train model
        history = self._model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            **kwargs
        )

        return {
            "history": history.history,
            "epochs_trained": len(history.history['loss']),
            "final_loss": history.history['loss'][-1],
            "final_val_loss": history.history.get('val_loss', [None])[-1]
        }

    def _train_pytorch(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """Train PyTorch model."""
        import torch
        import torch.optim as optim
        import torch.nn.functional as F

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Configure optimizer
        optimizer_config = self.config.get("neural_network", {}).get("optimizer", "adam")
        if isinstance(optimizer_config, str):
            if optimizer_config.lower() == "adam":
                optimizer = optim.Adam(self._model.parameters())
            elif optimizer_config.lower() == "sgd":
                optimizer = optim.SGD(self._model.parameters())
            else:
                optimizer = optim.Adam(self._model.parameters())
        else:
            optimizer = optim.Adam(self._model.parameters())

        # Configure loss
        loss_config = self.config.get("neural_network", {}).get("loss", "mse")
        if loss_config == "mse":
            criterion = nn.MSELoss()
        elif loss_config == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)

        history = {"loss": []}

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(X_tensor) // batch_size + 1)
            history["loss"].append(avg_loss)
            self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return {
            "history": history,
            "epochs_trained": epochs,
            "final_loss": history["loss"][-1]
        }

    def _predict_implementation(self, X: Any) -> Any:
        """Implementation-specific prediction logic."""
        if self.framework == "tensorflow":
            return self._predict_tensorflow(X)
        elif self.framework == "pytorch":
            return self._predict_pytorch(X)
        else:
            raise ValueError(f"Unsupported framework for prediction: {self.framework}")

    def _predict_tensorflow(self, X: Any) -> Any:
        """Make predictions with TensorFlow model."""
        import numpy as np

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        predictions = self._model.predict(X)
        return predictions

    def _predict_pytorch(self, X: Any) -> Any:
        """Make predictions with PyTorch model."""
        import torch

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self._model(X_tensor)
            return predictions.numpy()

    def _evaluate_implementation(self, X: Any, y: Any) -> Dict[str, float]:
        """Implementation-specific evaluation logic."""
        if self.framework == "tensorflow":
            return self._evaluate_tensorflow(X, y)
        elif self.framework == "pytorch":
            return self._evaluate_pytorch(X, y)
        else:
            raise ValueError(f"Unsupported framework for evaluation: {self.framework}")

    def _evaluate_tensorflow(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate TensorFlow model."""
        import numpy as np

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y

        loss, accuracy = self._model.evaluate(X, y, verbose=0)
        return {"loss": float(loss), "accuracy": float(accuracy)}

    def _evaluate_pytorch(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate PyTorch model."""
        import torch
        import torch.nn.functional as F

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            outputs = self._model(X_tensor)

            # Calculate loss (using MSE as default)
            loss = F.mse_loss(outputs, y_tensor).item()

            # For simplicity, return basic metrics
            return {"loss": loss}

    def _get_model_state(self) -> Any:
        """Get model state for serialization."""
        if self.framework == "tensorflow":
            return self._model.get_weights()
        elif self.framework == "pytorch":
            import torch
            return {name: param.data.numpy() for name, param in self._model.named_parameters()}
        else:
            raise ValueError(f"Unsupported framework for state serialization: {self.framework}")

    def _set_model_state(self, state: Any) -> None:
        """Set model state from serialized data."""
        if self.framework == "tensorflow":
            self._model.set_weights(state)
        elif self.framework == "pytorch":
            import torch
            for name, param in self._model.named_parameters():
                if name in state:
                    param.data = torch.FloatTensor(state[name])
        else:
            raise ValueError(f"Unsupported framework for state deserialization: {self.framework}")
