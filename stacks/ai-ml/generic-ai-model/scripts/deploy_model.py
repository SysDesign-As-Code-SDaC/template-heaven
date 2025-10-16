#!/usr/bin/env python3
"""
Deploy AI Model Script

This script provides a command-line interface for deploying trained AI models
as web services or APIs.
"""

import argparse
import sys
from pathlib import Path
import json
import subprocess

# Add template to path
template_dir = Path(__file__).parent.parent / "template"
sys.path.insert(0, str(template_dir))

from src.model_factory import create_model
from src.predictor import ModelPredictor


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy AI Model")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--config", "-c", type=str, default="config/deploy_config.yaml",
                       help="Deployment configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind server to")
    parser.add_argument("--platform", choices=['fastapi', 'flask', 'docker', 'aws', 'gcp'],
                       default='fastapi', help="Deployment platform")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    try:
        print("🚀 Starting Model Deployment")
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

        if args.platform == 'fastapi':
            deploy_fastapi(model, args.host, args.port, args.verbose)
        elif args.platform == 'flask':
            deploy_flask(model, args.host, args.port, args.verbose)
        elif args.platform == 'docker':
            deploy_docker(model, args.verbose)
        elif args.platform == 'aws':
            deploy_aws(model, args.verbose)
        elif args.platform == 'gcp':
            deploy_gcp(model, args.verbose)
        else:
            raise ValueError(f"Unsupported platform: {args.platform}")

    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def deploy_fastapi(model, host: str, port: int, verbose: bool):
    """Deploy model using FastAPI."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        from pydantic import BaseModel
        from typing import List, Union
        import numpy as np

        print("🌐 Setting up FastAPI server...")

        app = FastAPI(
            title="AI Model API",
            description="Generic AI Model Deployment API",
            version="1.0.0"
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Create predictor
        predictor = ModelPredictor(model)

        # Pydantic models for request/response
        class PredictionRequest(BaseModel):
            data: List[List[Union[float, int]]]

        class PredictionResponse(BaseModel):
            predictions: List[Union[float, int, List[float]]]
            timestamp: str
            model_info: dict

        @app.get("/")
        async def root():
            return {"message": "AI Model API", "status": "running"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            try:
                # Convert input to numpy array
                X = np.array(request.data)

                # Make prediction
                result = predictor.predict(X)

                return PredictionResponse(
                    predictions=result['predictions'],
                    timestamp=result['timestamp'],
                    model_info=result['model_info']
                )

            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/stats")
        async def get_stats():
            stats = predictor.get_statistics()
            return stats

        print(f"🚀 Starting FastAPI server on {host}:{port}")
        print("📖 API Documentation available at: http://localhost:8000/docs"
        print("Press Ctrl+C to stop the server")

        uvicorn.run(app, host=host, port=port, log_level="info" if verbose else "warning")

    except ImportError:
        print("❌ FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)


def deploy_flask(model, host: str, port: int, verbose: bool):
    """Deploy model using Flask."""
    try:
        from flask import Flask, request, jsonify
        import numpy as np

        print("🌐 Setting up Flask server...")

        app = Flask(__name__)

        # Create predictor
        predictor = ModelPredictor(model)

        @app.route('/')
        def root():
            return {"message": "AI Model API", "status": "running"}

        @app.route('/health')
        def health():
            return {"status": "healthy"}

        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json()

                if not data or 'data' not in data:
                    return jsonify({"error": "Missing 'data' field"}), 400

                # Convert input to numpy array
                X = np.array(data['data'])

                # Make prediction
                result = predictor.predict(X)

                return jsonify(result)

            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @app.route('/stats')
        def get_stats():
            stats = predictor.get_statistics()
            return jsonify(stats)

        print(f"🚀 Starting Flask server on {host}:{port}")
        print("Press Ctrl+C to stop the server")

        app.run(host=host, port=port, debug=verbose)

    except ImportError:
        print("❌ Flask not installed. Install with: pip install flask")
        sys.exit(1)


def deploy_docker(model, verbose: bool):
    """Deploy model using Docker."""
    print("🐳 Docker deployment not implemented yet")
    print("To deploy with Docker:")
    print("1. Build Docker image: docker build -t ai-model .")
    print("2. Run container: docker run -p 8000:8000 ai-model")


def deploy_aws(model, verbose: bool):
    """Deploy model to AWS."""
    print("☁️  AWS deployment not implemented yet")
    print("To deploy to AWS SageMaker:")
    print("1. Create SageMaker model")
    print("2. Create endpoint configuration")
    print("3. Deploy endpoint")


def deploy_gcp(model, verbose: bool):
    """Deploy model to Google Cloud."""
    print("☁️  GCP deployment not implemented yet")
    print("To deploy to Google AI Platform:")
    print("1. Upload model to Cloud Storage")
    print("2. Create AI Platform model")
    print("3. Deploy model version")


if __name__ == "__main__":
    main()
