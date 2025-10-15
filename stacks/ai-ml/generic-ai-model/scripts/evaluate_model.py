#!/usr/bin/env python3
"""
Evaluate AI Model Script

This script provides a command-line interface for evaluating trained AI models.
"""

import argparse
import sys
from pathlib import Path
import json

# Add template to path
template_dir = Path(__file__).parent.parent / "template"
sys.path.insert(0, str(template_dir))

from src.model_factory import create_model
from src.evaluator import ModelEvaluator
from src.data_loader import load_and_preprocess_data
from src.logger import setup_logging
from utils.config_utils import load_config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate AI Model")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--data-config", type=str, default="config/data_config.yaml",
                       help="Data configuration file")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file for evaluation report")
    parser.add_argument("--cross-validation", action="store_true",
                       help="Perform cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--report", type=str,
                       help="Generate detailed HTML report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Setup logging
        log_config = {"level": "DEBUG" if args.verbose else "INFO"}
        logger = setup_logging(log_config, "model_evaluation")

        print("📊 Starting Model Evaluation")
        print("=" * 50)

        # Load model
        print(f"📦 Loading model from {args.model}...")
        model_config_path = args.model.replace('.pkl', '_config.json')

        if Path(model_config_path).exists():
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            model = create_model(model_config)
            model.load(Path(args.model))
        else:
            print("⚠️  Config file not found, attempting to load model directly...")
            # Try to load model directly (this may not work for all frameworks)
            raise NotImplementedError("Direct model loading without config not implemented")

        # Load and preprocess data
        print("📊 Loading evaluation data...")
        data_config = load_config(args.data_config)
        _, _, X_test, _, _, y_test = load_and_preprocess_data(data_config)

        print(f"Test samples: {len(X_test)}")

        # Create evaluator
        evaluator = ModelEvaluator()

        # Perform evaluation
        if args.cross_validation:
            print(f"🔄 Performing {args.cv_folds}-fold cross-validation...")
            cv_results = evaluator.cross_validate(model, X_test, y_test, args.cv_folds)

            print("\nCross-Validation Results:")
            print("-" * 30)
            for metric, scores in cv_results['test_scores'].items():
                mean_score = cv_results['mean_scores'][f'test_{metric}']
                std_score = cv_results['std_scores'][f'test_{metric}']
                print(".4f")

        else:
            print("📈 Evaluating model on test set...")
            results = evaluator.evaluate(model, X_test, y_test)

            # Display results
            print("\nEvaluation Results:")
            print("-" * 30)

            metadata = results.get('metadata', {})
            print(f"Task Type: {metadata.get('task_type', 'Unknown')}")
            print(f"Samples: {metadata.get('n_samples', 'Unknown')}")
            print(f"Features: {metadata.get('n_features', 'Unknown')}")
            print()

            # Display metrics based on task type
            task_type = metadata.get('task_type', 'unknown')

            if task_type in ['classification', 'multiclass']:
                metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            else:
                metrics_to_show = ['r2_score', 'mse', 'mae', 'rmse']

            for metric in metrics_to_show:
                if metric in results:
                    value = results[metric]
                    if isinstance(value, float):
                        print("10s")
                    else:
                        print("10s")

            # Show confusion matrix if available
            if 'confusion_matrix' in results:
                print("\nConfusion Matrix:")
                cm = results['confusion_matrix']
                for row in cm:
                    print(f"  {row}")

            # Show feature importance if available
            if 'feature_importance' in results:
                print("\nTop Features:")
                importance_data = results['feature_importance']
                top_features = importance_data.get('top_features', [])
                values = importance_data.get('values', [])

                for i, feature_idx in enumerate(top_features[:5]):
                    if feature_idx < len(values):
                        importance = values[feature_idx]
                        print("10s")

        # Generate report if requested
        if args.report:
            print(f"\n📄 Generating detailed report: {args.report}")
            report = evaluator.generate_report(results if 'results' in locals() else cv_results,
                                             args.report)
            print(f"Report saved to: {args.report}")

        # Save results if output specified
        if args.output:
            print(f"💾 Saving results to {args.output}")
            output_data = {
                'evaluation_type': 'cross_validation' if args.cross_validation else 'test_set',
                'results': results if 'results' in locals() else cv_results,
                'model_path': args.model,
                'data_config': str(args.data_config)
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
