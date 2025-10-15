#!/usr/bin/env python3
"""
Predict with AI Model Script

This script provides a command-line interface for making predictions
with trained AI models.
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd

# Add template to path
template_dir = Path(__file__).parent.parent / "template"
sys.path.insert(0, str(template_dir))

from src.model_factory import create_model
from src.predictor import ModelPredictor
from src.data_loader import CSVDataLoader
from src.logger import setup_logging


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make Predictions with AI Model")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input data file (CSV)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file for predictions")
    parser.add_argument("--format", "-f", choices=['json', 'csv'], default='json',
                       help="Output format")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for predictions")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    try:
        # Setup logging
        log_config = {"level": "DEBUG" if args.verbose else "INFO"}
        logger = setup_logging(log_config, "model_prediction")

        print("🔮 Starting Model Prediction")
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
            raise NotImplementedError("Direct model loading without config not implemented")

        # Load input data
        print(f"📊 Loading input data from {args.input}...")
        if not Path(args.input).exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")

        # Load data using pandas for flexibility
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")

        # Convert to numpy array (assuming all columns are features)
        X = df.values

        # Create predictor
        predictor_config = {
            'batch_size': args.batch_size,
            'enable_monitoring': True,
            'expected_features': X.shape[1] if len(X.shape) > 1 else 1
        }
        predictor = ModelPredictor(model, predictor_config)

        # Make predictions
        print("🔮 Making predictions...")
        if len(X) <= args.batch_size:
            # Single batch prediction
            results = predictor.predict(X)
            predictions = results['predictions']
        else:
            # Batch prediction
            batch_predictor = predictor.BatchPredictor(predictor, max_workers=4)
            batch_results = batch_predictor.predict_large_batch(X.tolist(), args.batch_size)
            # Combine results
            predictions = []
            for result in batch_results:
                if isinstance(result, dict) and 'predictions' in result:
                    predictions.extend(result['predictions'])

        print(f"Generated {len(predictions)} predictions")

        # Display sample predictions
        print("\nSample Predictions:")
        print("-" * 30)
        for i, pred in enumerate(predictions[:10]):
            if isinstance(pred, (int, float)):
                print("2d")
            elif isinstance(pred, list):
                print("2d")
            else:
                print("2d")

        if len(predictions) > 10:
            print(f"... and {len(predictions) - 10} more")

        # Get prediction statistics
        stats = predictor.get_statistics()
        print("
Prediction Statistics:")
        print("-" * 30)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Average time per prediction: {stats['average_prediction_time']:.4f}s")
        print(f"Error rate: {stats['error_rate']:.4f}")

        # Save predictions if output specified
        if args.output:
            print(f"\n💾 Saving predictions to {args.output}")

            if args.format == 'json':
                output_data = {
                    'model_path': args.model,
                    'input_file': args.input,
                    'predictions': predictions,
                    'statistics': stats,
                    'timestamp': predictor.predict([X[0]])['timestamp'] if len(X) > 0 else None
                }

                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)

            elif args.format == 'csv':
                # Create output dataframe
                output_df = df.copy()
                output_df['prediction'] = predictions
                output_df.to_csv(args.output, index=False)

            print(f"Predictions saved to: {args.output}")

        print("\n✅ Prediction completed successfully!")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
