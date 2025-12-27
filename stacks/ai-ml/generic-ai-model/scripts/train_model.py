#!/usr/bin/env python3
"""
Train AI Model Script

This script provides a command-line interface for training AI models
using the generic AI model template.
"""

import argparse
import sys
from pathlib import Path
import json

# Add template to path
template_dir = Path(__file__).parent.parent / "template"
sys.path.insert(0, str(template_dir))

from src.model_factory import create_model
from src.trainer import Trainer
from src.data_loader import load_and_preprocess_data
from src.logger import setup_logging
from utils.config_utils import load_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AI Model")
    parser.add_argument("--config", "-c", type=str, default="config/model_config.yaml",
                       help="Model configuration file")
    parser.add_argument("--data-config", type=str, default="config/data_config.yaml",
                       help="Data configuration file")
    parser.add_argument("--train-config", type=str, default="config/train_config.yaml",
                       help="Training configuration file")
    parser.add_argument("--output-dir", "-o", type=str, default="models",
                       help="Output directory for trained model")
    parser.add_argument("--experiment-name", type=str, default="experiment",
                       help="Experiment name for logging")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    try:
        # Setup logging
        log_config = {"level": "DEBUG" if args.verbose else "INFO"}
        logger = setup_logging(log_config, args.experiment_name)

        print("🚀 Starting AI Model Training")
        print("=" * 50)

        # Load configurations
        print("📋 Loading configurations...")
        model_config = load_config(args.config)
        data_config = load_config(args.data_config)
        train_config = load_config(args.train_config)

        # Merge configurations
        full_config = {
            "model": model_config.get("model", {}),
            "training": train_config.get("training", {}),
            "data": data_config
        }

        print(f"Model Type: {model_config['model']['type']}")
        print(f"Framework: {model_config['model']['framework']}")

        # Load and preprocess data
        print("📊 Loading and preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(data_config)

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Create model
        print("🏗️  Building model...")
        model = create_model(full_config)

        # Create trainer
        print("🎯 Setting up trainer...")
        trainer = Trainer(model, full_config)

        # Train model
        print("🚀 Training model...")
        logger.log_training_start(full_config)

        results = trainer.train(X_train, y_train, X_val, y_val)

        logger.log_training_end(results)

        # Evaluate on test set
        print("📈 Evaluating model...")
        test_metrics = model.evaluate(X_test, y_test)
        logger.log_evaluation_results(test_metrics)

        # Save model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{args.experiment_name}_model.pkl"
        config_path = output_dir / f"{args.experiment_name}_config.json"

        print("💾 Saving model...")
        model.save(model_path)

        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)

        # Print results
        print("\n" + "=" * 50)
        print("🎉 Training completed successfully!")
        print("=" * 50)
        print(f"Model saved to: {model_path}")
        print(f"Config saved to: {config_path}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Final epochs: {results['epochs_completed']}")

        if 'accuracy' in test_metrics:
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        if 'precision' in test_metrics:
            print(f"Test Precision: {test_metrics['precision']:.4f}")
        if 'recall' in test_metrics:
            print(f"Test Recall: {test_metrics['recall']:.4f}")
        if 'f1_score' in test_metrics:
            print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
        if 'r2_score' in test_metrics:
            print(f"Test R² Score: {test_metrics['r2_score']:.4f}")
        if 'mse' in test_metrics:
            print(f"Test MSE: {test_metrics['mse']:.4f}")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
