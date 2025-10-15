"""
Model Factory

This module provides a factory pattern for creating AI models based on configuration.
It supports dynamic registration of new model types and automatic model instantiation.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import importlib
import inspect

from ..models.base_model import BaseModel


class ModelFactory:
    """
    Factory class for creating AI models based on configuration.

    This class implements the factory pattern to create different types of AI models
    based on configuration parameters. It supports dynamic registration of new model
    types and automatic discovery of model implementations.

    Attributes:
        _registry (Dict[str, Type[BaseModel]]): Registry of available model classes
        logger (logging.Logger): Logger instance for the factory
    """

    _registry: Dict[str, Type[BaseModel]] = {}
    _instance = None

    def __init__(self):
        """Initialize the model factory."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def get_instance(cls) -> 'ModelFactory':
        """
        Get singleton instance of the factory.

        Returns:
            Singleton ModelFactory instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._auto_register_models()
        return cls._instance

    def _auto_register_models(self) -> None:
        """
        Automatically register all model implementations found in the models directory.
        """
        try:
            models_dir = Path(__file__).parent.parent / "models"

            # Import all model modules
            for model_file in models_dir.glob("*.py"):
                if model_file.name.startswith("_"):
                    continue

                module_name = f"template.models.{model_file.stem}"
                try:
                    module = importlib.import_module(module_name)

                    # Find all classes that inherit from BaseModel
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, BaseModel) and
                            obj != BaseModel):
                            self.register_model(obj.__name__.lower(), obj)
                            self.logger.debug(f"Auto-registered model: {obj.__name__}")

                except ImportError as e:
                    self.logger.warning(f"Failed to import model module {module_name}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error processing model module {module_name}: {e}")

        except Exception as e:
            self.logger.warning(f"Auto-registration failed: {e}")

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model type with the factory.

        Args:
            model_type: String identifier for the model type
            model_class: Model class that inherits from BaseModel

        Raises:
            ValueError: If model_class doesn't inherit from BaseModel
            TypeError: If model_class is not a class
        """
        if not inspect.isclass(model_class):
            raise TypeError(f"model_class must be a class, got {type(model_class)}")

        if not issubclass(model_class, BaseModel):
            raise ValueError(f"model_class must inherit from BaseModel, got {model_class}")

        cls._registry[model_type] = model_class

        # Log registration
        logger = logging.getLogger(cls.__name__)
        logger.info(f"Registered model type: {model_type} -> {model_class.__name__}")

    @classmethod
    def unregister_model(cls, model_type: str) -> None:
        """
        Unregister a model type from the factory.

        Args:
            model_type: String identifier for the model type to remove
        """
        if model_type in cls._registry:
            del cls._registry[model_type]
            logger = logging.getLogger(cls.__name__)
            logger.info(f"Unregistered model type: {model_type}")

    @classmethod
    def list_model_types(cls) -> List[str]:
        """
        Get list of all registered model types.

        Returns:
            List of registered model type identifiers
        """
        return list(cls._registry.keys())

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """
        Get the model class for a given type.

        Args:
            model_type: String identifier for the model type

        Returns:
            Model class for the specified type

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._registry:
            available_types = cls.list_model_types()
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {available_types}")

        return cls._registry[model_type]

    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance based on configuration.

        Args:
            config: Configuration dictionary containing model parameters

        Returns:
            Instantiated model object

        Raises:
            ValueError: If model type is not specified or invalid
            RuntimeError: If model creation fails
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Extract model type
        model_config = config.get("model", {})
        model_type = model_config.get("type")

        if not model_type:
            raise ValueError("Model type must be specified in config['model']['type']")

        try:
            # Get model class
            model_class = cls.get_model_class(model_type)

            # Create model instance
            model = model_class(config)

            # Log creation
            logger = logging.getLogger(cls.__name__)
            logger.info(f"Created model: {model_type} ({model_class.__name__})")

            return model

        except Exception as e:
            logger = logging.getLogger(cls.__name__)
            logger.error(f"Failed to create model {model_type}: {e}")
            raise RuntimeError(f"Model creation failed: {e}") from e

    @classmethod
    def create_model_from_file(cls, config_path: str) -> BaseModel:
        """
        Create a model from a configuration file.

        Args:
            config_path: Path to the configuration file (YAML or JSON)

        Returns:
            Instantiated model object
        """
        import yaml
        import json

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

            return cls.create_model(config)

        except Exception as e:
            logger = logging.getLogger(cls.__name__)
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        Validate a model configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        if "model" not in config:
            errors.append("Missing 'model' section in configuration")
            return errors

        model_config = config["model"]

        if "type" not in model_config:
            errors.append("Missing 'model.type' in configuration")
            return errors

        model_type = model_config["type"]

        # Check if model type is registered
        if model_type not in cls._registry:
            available_types = cls.list_model_types()
            errors.append(f"Unknown model type '{model_type}'. Available: {available_types}")
            return errors

        # Get model class and validate its specific requirements
        try:
            model_class = cls.get_model_class(model_type)
            # Try to create model to validate configuration
            model = model_class(config)
            # If we get here, basic validation passed
        except Exception as e:
            errors.append(f"Configuration validation failed for {model_type}: {str(e)}")

        return errors

    @classmethod
    def get_model_info(cls, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered model type.

        Args:
            model_type: String identifier for the model type

        Returns:
            Dictionary containing model information, or None if not found
        """
        if model_type not in cls._registry:
            return None

        model_class = cls._registry[model_type]

        return {
            "type": model_type,
            "class_name": model_class.__name__,
            "module": model_class.__module__,
            "docstring": model_class.__doc__.strip() if model_class.__doc__ else "",
            "framework": getattr(model_class, 'framework', 'unknown')
        }

    @classmethod
    def list_available_models(cls) -> List[Dict[str, Any]]:
        """
        Get information about all registered model types.

        Returns:
            List of dictionaries containing model information
        """
        return [cls.get_model_info(model_type) for model_type in cls.list_model_types()]


# Convenience functions for easy access
def create_model(config: Dict[str, Any]) -> BaseModel:
    """
    Convenience function to create a model.

    Args:
        config: Model configuration dictionary

    Returns:
        Instantiated model object
    """
    return ModelFactory.create_model(config)


def register_model(model_type: str, model_class: Type[BaseModel]) -> None:
    """
    Convenience function to register a model type.

    Args:
        model_type: String identifier for the model type
        model_class: Model class to register
    """
    ModelFactory.register_model(model_type, model_class)


def list_model_types() -> List[str]:
    """
    Convenience function to list available model types.

    Returns:
        List of available model type identifiers
    """
    return ModelFactory.list_model_types()
