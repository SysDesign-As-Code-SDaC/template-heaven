"""
Tests for Model Factory

This module contains tests for the model factory functionality.
"""

import pytest
import tempfile
from pathlib import Path
import json

# Add template to path
import sys
template_dir = Path(__file__).parent.parent / "template"
sys.path.insert(0, str(template_dir))

from src.model_factory import ModelFactory, create_model, register_model, list_model_types


class TestModelFactory:
    """Test cases for ModelFactory."""

    def test_singleton_instance(self):
        """Test that ModelFactory is a singleton."""
        factory1 = ModelFactory.get_instance()
        factory2 = ModelFactory.get_instance()
        assert factory1 is factory2

    def test_list_model_types(self):
        """Test listing available model types."""
        model_types = list_model_types()
        assert isinstance(model_types, list)
        assert len(model_types) >= 3  # At least neural_network, tree_model, linear_model

    def test_register_model(self):
        """Test registering a new model type."""
        from models.base_model import BaseModel

        class DummyModel(BaseModel):
            def _get_model_type(self):
                return "dummy"

            def build(self):
                pass

            def _train_implementation(self, X, y, **kwargs):
                return {"loss": 0.5}

            def _predict_implementation(self, X):
                return [0.5] * len(X)

            def _evaluate_implementation(self, X, y):
                return {"accuracy": 0.8}

            def _get_model_state(self):
                return {"dummy": "state"}

            def _set_model_state(self, state):
                pass

        # Register model
        register_model("test_dummy", DummyModel)

        # Check if registered
        model_types = list_model_types()
        assert "test_dummy" in model_types

        # Test creation
        config = {"model": {"type": "test_dummy", "framework": "custom"}}
        model = create_model(config)
        assert isinstance(model, DummyModel)

    def test_create_neural_network_model(self):
        """Test creating a neural network model."""
        config = {
            "model": {
                "type": "neural_network",
                "framework": "tensorflow"
            },
            "neural_network": {
                "architecture": {
                    "layers": [
                        {
                            "type": "dense",
                            "units": 64,
                            "activation": "relu",
                            "input_shape": [10]
                        }
                    ]
                }
            }
        }

        model = create_model(config)
        assert model.model_type == "neural_network"
        assert model.framework == "tensorflow"

    def test_create_tree_model(self):
        """Test creating a tree-based model."""
        config = {
            "model": {
                "type": "tree_model",
                "framework": "sklearn"
            },
            "tree_model": {
                "type": "random_forest",
                "params": {"n_estimators": 10}
            }
        }

        model = create_model(config)
        assert model.model_type == "tree_model"
        assert model.framework == "sklearn"

    def test_create_linear_model(self):
        """Test creating a linear model."""
        config = {
            "model": {
                "type": "linear_model",
                "framework": "sklearn"
            },
            "linear_model": {
                "type": "logistic_regression"
            }
        }

        model = create_model(config)
        assert model.model_type == "linear_model"
        assert model.framework == "sklearn"

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        config = {
            "model": {
                "type": "invalid_model_type",
                "framework": "custom"
            }
        }

        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(config)

    def test_missing_model_type(self):
        """Test error handling for missing model type."""
        config = {
            "model": {
                "framework": "tensorflow"
            }
        }

        with pytest.raises(ValueError, match="Model type must be specified"):
            create_model(config)

    def test_create_model_from_file(self):
        """Test creating model from configuration file."""
        config = {
            "model": {
                "type": "tree_model",
                "framework": "sklearn"
            },
            "tree_model": {
                "type": "random_forest"
            }
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            model = ModelFactory.create_model_from_file(config_path)
            assert model.model_type == "tree_model"
        finally:
            Path(config_path).unlink()

    def test_validate_config(self):
        """Test configuration validation."""
        from src.model_factory import ModelFactory

        # Valid config
        valid_config = {
            "model": {
                "type": "neural_network",
                "framework": "tensorflow"
            }
        }

        errors = ModelFactory.validate_config(valid_config)
        assert len(errors) == 0

        # Invalid config - missing model section
        invalid_config = {
            "invalid": "config"
        }

        errors = ModelFactory.validate_config(invalid_config)
        assert len(errors) > 0
        assert "Missing required section: model" in errors[0]

    def test_get_model_info(self):
        """Test getting model information."""
        info = ModelFactory.get_model_info("neural_network")
        assert info is not None
        assert info["type"] == "neural_network"
        assert "class_name" in info

        # Test invalid model type
        info = ModelFactory.get_model_info("invalid_type")
        assert info is None

    def test_list_available_models(self):
        """Test listing all available models."""
        models = ModelFactory.list_available_models()
        assert isinstance(models, list)
        assert len(models) >= 3

        # Check structure
        for model in models:
            assert "type" in model
            assert "class_name" in model
            assert "docstring" in model
