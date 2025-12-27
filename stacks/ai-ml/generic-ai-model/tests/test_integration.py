"""
Integration Tests

This module contains integration tests for the complete AI model workflow.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json

# Add template to path
import sys
template_dir = Path(__file__).parent.parent / "template"
sys.path.insert(0, str(template_dir))

from src.model_factory import create_model
from src.trainer import Trainer
from src.evaluator import ModelEvaluator
from src.predictor import ModelPredictor


class TestIntegration:
    """Integration tests for the complete AI model workflow."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 4

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate binary target
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    @pytest.fixture
    def tree_model_config(self):
        """Configuration for tree-based model."""
        return {
            "model": {
                "type": "tree_model",
                "framework": "sklearn"
            },
            "tree_model": {
                "type": "random_forest",
                "params": {
                    "n_estimators": 10,
                    "random_state": 42
                }
            },
            "training": {
                "epochs": 1,  # Minimal for testing
                "batch_size": 32
            }
        }

    @pytest.fixture
    def linear_model_config(self):
        """Configuration for linear model."""
        return {
            "model": {
                "type": "linear_model",
                "framework": "sklearn"
            },
            "linear_model": {
                "type": "logistic_regression",
                "params": {
                    "random_state": 42,
                    "max_iter": 100
                }
            },
            "training": {
                "epochs": 1,
                "batch_size": 32
            }
        }

    def test_tree_model_workflow(self, sample_data, tree_model_config):
        """Test complete workflow with tree-based model."""
        X, y = sample_data

        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create model
        model = create_model(tree_model_config)
        assert model.model_type == "tree_model"

        # Build model
        model.build()

        # Train model
        trainer = Trainer(model, tree_model_config)
        results = trainer.train(X_train, y_train)

        assert "training_time" in results
        assert results["epochs_completed"] >= 1

        # Evaluate model
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate(model, X_test, y_test)

        assert "accuracy" in eval_results
        assert eval_results["accuracy"] >= 0.0
        assert eval_results["accuracy"] <= 1.0

        # Make predictions
        predictor = ModelPredictor(model)
        pred_results = predictor.predict(X_test[:5])

        assert "predictions" in pred_results
        assert len(pred_results["predictions"]) == 5

        # Test model saving and loading
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        try:
            # Save model
            model.save(model_path)

            # Load model
            loaded_model = create_model(tree_model_config)
            loaded_model.load(model_path)

            # Test loaded model
            loaded_pred = loaded_model.predict(X_test[:3])
            original_pred = model.predict(X_test[:3])

            np.testing.assert_array_almost_equal(
                loaded_pred, original_pred, decimal=5
            )

        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_linear_model_workflow(self, sample_data, linear_model_config):
        """Test complete workflow with linear model."""
        X, y = sample_data

        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create model
        model = create_model(linear_model_config)
        assert model.model_type == "linear_model"

        # Build model
        model.build()

        # Train model
        trainer = Trainer(model, linear_model_config)
        results = trainer.train(X_train, y_train)

        assert "training_time" in results

        # Evaluate model
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate(model, X_test, y_test)

        assert "accuracy" in eval_results

        # Test cross-validation
        cv_results = evaluator.cross_validate(model, X_test, y_test, cv_folds=3)
        assert "mean_scores" in cv_results
        assert cv_results["cv_folds"] == 3

    def test_model_comparison(self, sample_data, tree_model_config, linear_model_config):
        """Test model comparison functionality."""
        X, y = sample_data

        # Split data
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        models_results = []

        # Train and evaluate tree model
        tree_model = create_model(tree_model_config)
        tree_model.build()
        trainer = Trainer(tree_model, tree_model_config)
        trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        tree_results = evaluator.evaluate(tree_model, X_test, y_test)
        models_results.append(tree_results)

        # Train and evaluate linear model
        linear_model = create_model(linear_model_config)
        linear_model.build()
        trainer = Trainer(linear_model, linear_model_config)
        trainer.train(X_train, y_train)

        linear_results = evaluator.evaluate(linear_model, X_test, y_test)
        models_results.append(linear_results)

        # Compare models
        comparison = evaluator.compare_models(models_results, metric="accuracy")

        assert "best_model_index" in comparison
        assert "best_score" in comparison
        assert len(comparison["models"]) == 2

    def test_prediction_statistics(self, sample_data, tree_model_config):
        """Test prediction statistics tracking."""
        X, y = sample_data

        # Create and train model
        model = create_model(tree_model_config)
        model.build()
        trainer = Trainer(model, tree_model_config)
        trainer.train(X, y)

        # Create predictor
        predictor = ModelPredictor(model)

        # Make several predictions
        for i in range(5):
            predictor.predict(X[i:i+1])

        # Check statistics
        stats = predictor.get_statistics()

        assert stats["total_predictions"] == 5
        assert stats["total_errors"] == 0
        assert "average_prediction_time" in stats

        # Test error handling
        predictor.predict(None)  # This should cause an error
        stats_after_error = predictor.get_statistics()

        assert stats_after_error["total_errors"] >= 1

    def test_configuration_persistence(self, tree_model_config):
        """Test that model configuration is properly saved and loaded."""
        # Create model
        model = create_model(tree_model_config)

        # Save model with config
        with tempfile.NamedTemporaryFile(suffix='_config.json', delete=False) as f:
            config_path = f.name
        model_path = config_path.replace('_config.json', '.pkl')

        try:
            # Save config
            with open(config_path, 'w') as f:
                json.dump(tree_model_config, f)

            # Save model
            model.save(model_path)

            # Load config and model
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            loaded_model = create_model(loaded_config)
            loaded_model.load(model_path)

            # Verify configuration
            assert loaded_model.config == model.config
            assert loaded_model.model_type == model.model_type
            assert loaded_model.framework == model.framework

        finally:
            Path(config_path).unlink(missing_ok=True)
            Path(model_path).unlink(missing_ok=True)

    @pytest.mark.skipif(not hasattr(np, 'array'), reason="NumPy not available")
    def test_large_batch_prediction(self, tree_model_config):
        """Test prediction on large batches."""
        # Generate larger dataset
        np.random.seed(42)
        X = np.random.randn(1000, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Create and train model
        model = create_model(tree_model_config)
        model.build()
        trainer = Trainer(model, tree_model_config)
        trainer.train(X, y)

        # Test batch prediction
        predictor = ModelPredictor(model, {"batch_size": 100})
        results = predictor.predict_batch(X.tolist(), batch_size=100)

        # Should return multiple batch results
        assert len(results) > 1

        # Total predictions should match input size
        total_predictions = sum(len(result.get('predictions', [])) for result in results)
        assert total_predictions == len(X)
