"""
Configuration Utilities

This module provides utilities for loading, validating, and managing
configuration files for the AI model template.
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigManager:
    """
    Configuration manager for loading and validating configuration files.

    This class provides a centralized way to manage configuration files
    with support for multiple formats, validation, and environment variable
    substitution.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "config"
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache = {}

    def load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            config_name: Name of the configuration file (without extension)
            use_cache: Whether to use cached configurations

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration file format is unsupported
        """
        cache_key = f"{config_name}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Try different file extensions
        config_path = None
        for ext in ['.yaml', '.yml', '.json']:
            potential_path = self.config_dir / f"{config_name}{ext}"
            if potential_path.exists():
                config_path = potential_path
                break

        if not config_path:
            available_files = list(self.config_dir.glob("*"))
            raise FileNotFoundError(
                f"Configuration file '{config_name}' not found in {self.config_dir}. "
                f"Available files: {[f.name for f in available_files]}"
            )

        # Determine format from extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            format_type = ConfigFormat.YAML
        elif config_path.suffix.lower() == '.json':
            format_type = ConfigFormat.JSON
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        # Load configuration
        config = self._load_file(config_path, format_type)

        # Process environment variables
        config = self._substitute_env_vars(config)

        # Cache the configuration
        if use_cache:
            self._cache[cache_key] = config.copy()

        self.logger.info(f"Loaded configuration: {config_name}")
        return config

    def _load_file(self, file_path: Path, format_type: ConfigFormat) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if format_type == ConfigFormat.YAML:
                    return yaml.safe_load(f)
                elif format_type == ConfigFormat.JSON:
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {file_path}: {e}")

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Substitute environment variables in configuration.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_var_in_string(config)
        else:
            return config

    def _substitute_env_var_in_string(self, text: str) -> str:
        """Substitute environment variables in a string."""
        import re

        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''

            return os.environ.get(var_name, default_value)

        # Replace ${VAR} or ${VAR:default} patterns
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, text)

    def save_config(self, config: Dict[str, Any], config_name: str,
                   format_type: ConfigFormat = ConfigFormat.YAML) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary to save
            config_name: Name of the configuration file (without extension)
            format_type: Format to save in
        """
        file_path = self.config_dir / f"{config_name}{format_type.value}"

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                elif format_type == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")

            self.logger.info(f"Saved configuration: {config_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to save configuration to {file_path}: {e}")

    def validate_config(self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> ConfigValidationResult:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration to validate
            schema: Validation schema (optional)

        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []

        # Basic validation
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return ConfigValidationResult(False, errors, warnings)

        # Required sections validation
        required_sections = ['model']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Model configuration validation
        if 'model' in config:
            model_config = config['model']

            # Validate model type
            if 'type' not in model_config:
                errors.append("model.type is required")
            else:
                valid_types = ['neural_network', 'tree_model', 'linear_model', 'ensemble_model', 'custom']
                if model_config['type'] not in valid_types:
                    errors.append(f"Invalid model type: {model_config['type']}. Valid types: {valid_types}")

            # Validate framework
            if 'framework' in model_config:
                framework = model_config['framework']
                model_type = model_config.get('type', '')

                # Framework validation based on model type
                if model_type == 'neural_network':
                    valid_frameworks = ['tensorflow', 'pytorch']
                elif model_type == 'tree_model':
                    valid_frameworks = ['sklearn', 'xgboost', 'lightgbm']
                elif model_type == 'linear_model':
                    valid_frameworks = ['sklearn']
                else:
                    valid_frameworks = ['custom']

                if framework not in valid_frameworks:
                    warnings.append(f"Framework '{framework}' may not be optimal for {model_type}. "
                                  f"Consider: {valid_frameworks}")

        # Data configuration validation
        if 'data' in config:
            data_config = config['data']
            # Add data validation logic here

        # Training configuration validation
        if 'training' in config:
            training_config = config['training']
            # Add training validation logic here

        return ConfigValidationResult(len(errors) == 0, errors, warnings)

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.

        Later configurations override earlier ones.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        merged = {}

        for config in configs:
            merged = self._deep_merge(merged, config)

        return merged

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_config_template(self, model_type: str) -> Dict[str, Any]:
        """
        Get a configuration template for a specific model type.

        Args:
            model_type: Type of model to get template for

        Returns:
            Configuration template dictionary
        """
        base_template = {
            "model": {
                "type": model_type,
                "framework": self._get_default_framework(model_type)
            }
        }

        # Add model-specific configuration
        if model_type == "neural_network":
            base_template["neural_network"] = {
                "architecture": {
                    "layers": [
                        {
                            "type": "dense",
                            "units": 64,
                            "activation": "relu"
                        }
                    ]
                },
                "optimizer": {"name": "adam", "learning_rate": 0.001},
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"]
            }
        elif model_type == "tree_model":
            base_template["tree_model"] = {
                "type": "random_forest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            }
        elif model_type == "linear_model":
            base_template["linear_model"] = {
                "type": "logistic_regression",
                "params": {
                    "C": 1.0,
                    "random_state": 42
                }
            }

        return base_template

    def _get_default_framework(self, model_type: str) -> str:
        """Get the default framework for a model type."""
        framework_map = {
            "neural_network": "tensorflow",
            "tree_model": "sklearn",
            "linear_model": "sklearn",
            "ensemble_model": "sklearn",
            "custom": "custom"
        }
        return framework_map.get(model_type, "custom")

    def list_available_configs(self) -> List[str]:
        """
        List all available configuration files.

        Returns:
            List of configuration file names (without extensions)
        """
        config_files = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            config_files.extend([f.stem for f in self.config_dir.glob(ext)])

        return sorted(list(set(config_files)))

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
        self.logger.info("Configuration cache cleared")


# Convenience functions
def load_config(config_name: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a configuration file.

    Args:
        config_name: Name of the configuration file
        config_dir: Directory containing configuration files

    Returns:
        Configuration dictionary
    """
    manager = ConfigManager(config_dir)
    return manager.load_config(config_name)


def save_config(config: Dict[str, Any], config_name: str,
               format_type: ConfigFormat = ConfigFormat.YAML,
               config_dir: Optional[str] = None) -> None:
    """
    Save a configuration to file.

    Args:
        config: Configuration dictionary
        config_name: Name of the configuration file
        format_type: Format to save in
        config_dir: Directory to save configuration in
    """
    manager = ConfigManager(config_dir)
    manager.save_config(config, config_name, format_type)


def validate_config(config: Dict[str, Any]) -> ConfigValidationResult:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        Validation result
    """
    manager = ConfigManager()
    return manager.validate_config(config)
