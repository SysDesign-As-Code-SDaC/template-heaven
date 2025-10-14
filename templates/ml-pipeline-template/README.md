# {{ project_name | title }} - ML Pipeline Template

{{ project_description }}

## 🚀 Features

This template provides a production-ready machine learning pipeline with:

- **🐍 Python 3.11+** - Latest Python with ML ecosystem support
- **📊 Data Processing** - Pandas, NumPy, Polars for data manipulation
- **🤖 ML Frameworks** - Scikit-learn, XGBoost, LightGBM, CatBoost
- **🧠 Deep Learning** - PyTorch, TensorFlow, Hugging Face Transformers
- **📈 Experiment Tracking** - MLflow, Weights & Biases integration
- **🔄 MLOps** - DVC for data versioning, Prefect for orchestration
- **🐳 Containerization** - Docker with GPU support
- **☁️ Cloud Ready** - AWS, GCP, Azure deployment configurations
- **📊 Monitoring** - Model performance monitoring and drift detection
- **🧪 Testing** - Comprehensive testing with pytest and Great Expectations
- **📝 Documentation** - Auto-generated docs with Sphinx
- **🔄 CI/CD** - GitHub Actions with model validation gates
- **🛡️ Security** - Data privacy and model security best practices

## 🛠️ Tech Stack

- **Core ML**: Python, Scikit-learn, XGBoost, PyTorch, TensorFlow
- **Data**: Pandas, NumPy, Polars, Dask, Apache Spark
- **MLOps**: MLflow, DVC, Prefect, Kubeflow
- **Cloud**: AWS SageMaker, GCP Vertex AI, Azure ML
- **Monitoring**: Evidently AI, WhyLabs, MLflow Model Registry
- **Testing**: pytest, Great Expectations, model validation
- **Deployment**: Docker, Kubernetes, serverless functions

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Git
- CUDA (for GPU support, optional)

### Installation

```bash
# Clone and setup
git clone <your-repo-url>
cd {{ project_name }}
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -e ".[dev]"

# Initialize DVC
dvc init

# Download sample data
dvc pull data/raw/sample_data.csv

# Run the pipeline
python -m src.pipeline.main

# Start MLflow UI
mlflow ui
```

### Docker

```bash
# Development with Docker Compose
docker-compose up --build

# GPU support
docker-compose -f docker-compose.gpu.yml up --build

# Production build
docker build -t {{ project_name }} .
docker run -p 5000:5000 {{ project_name }}
```

## 📁 Project Structure

```
{{ project_name }}/
├── data/                         # Data directory (DVC tracked)
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   ├── external/                # External data sources
│   └── models/                  # Trained models
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── ingestion.py         # Data ingestion
│   │   ├── preprocessing.py     # Data preprocessing
│   │   └── validation.py        # Data validation
│   ├── features/                # Feature engineering
│   │   ├── __init__.py
│   │   ├── extraction.py        # Feature extraction
│   │   ├── selection.py         # Feature selection
│   │   └── transformation.py    # Feature transformation
│   ├── models/                  # Model development
│   │   ├── __init__.py
│   │   ├── base.py              # Base model class
│   │   ├── training.py          # Model training
│   │   ├── evaluation.py        # Model evaluation
│   │   └── prediction.py        # Model prediction
│   ├── pipeline/                # ML pipeline orchestration
│   │   ├── __init__.py
│   │   ├── main.py              # Main pipeline
│   │   ├── stages.py            # Pipeline stages
│   │   └── config.py            # Pipeline configuration
│   ├── monitoring/              # Model monitoring
│   │   ├── __init__.py
│   │   ├── drift.py             # Data drift detection
│   │   ├── performance.py       # Performance monitoring
│   │   └── alerts.py            # Alerting system
│   ├── deployment/              # Model deployment
│   │   ├── __init__.py
│   │   ├── api.py               # REST API
│   │   ├── batch.py             # Batch inference
│   │   └── streaming.py         # Streaming inference
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── logging.py           # Logging configuration
│       ├── config.py            # Configuration management
│       └── helpers.py           # Helper functions
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
├── tests/                       # Test suite
│   ├── conftest.py
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── configs/                     # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── scripts/                     # Utility scripts
│   ├── setup_data.py
│   ├── train_model.py
│   └── deploy_model.py
├── docs/                        # Documentation
│   ├── source/
│   └── build/
├── docker/                      # Docker configurations
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
├── .github/                     # GitHub Actions
│   └── workflows/
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
├── dvc.yaml                    # DVC pipeline configuration
├── .dvc/                       # DVC metadata
├── mlflow/                     # MLflow tracking
├── Dockerfile                  # Container image
├── docker-compose.yml          # Local development
└── README.md                   # This file
```

## 🛠️ Development

### Available Scripts

```bash
# Data Management
dvc add data/raw/dataset.csv     # Add data to DVC
dvc push                         # Push data to remote storage
dvc pull                         # Pull data from remote storage
dvc repro                        # Reproduce pipeline

# Model Development
python -m src.pipeline.main      # Run full pipeline
python -m src.models.training    # Train model only
python -m src.models.evaluation  # Evaluate model only

# Testing
pytest                          # Run all tests
pytest tests/unit/              # Run unit tests
pytest tests/integration/       # Run integration tests
pytest --cov=src               # Run with coverage

# Code Quality
black src tests                 # Format code
isort src tests                 # Sort imports
flake8 src tests                # Lint code
mypy src                        # Type checking

# Documentation
sphinx-build docs/source docs/build  # Build docs
```

### Code Quality

```bash
# Run all quality checks
pre-commit run --all-files

# Or manually
black src tests && isort src tests && flake8 src tests && mypy src
```

### Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v

# Test with coverage
pytest --cov=src --cov-report=html

# Data validation tests
pytest tests/data/ -v
```

## 🔧 Configuration

### Pipeline Configuration

```yaml
# configs/pipeline_config.yaml
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  external_path: "data/external"
  
features:
  target_column: "target"
  categorical_columns: ["category1", "category2"]
  numerical_columns: ["feature1", "feature2", "feature3"]
  
model:
  name: "{{ project_name }}_model"
  algorithm: "xgboost"
  hyperparameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  cv_folds: 5
  test_size: 0.2
  
monitoring:
  drift_threshold: 0.1
  performance_threshold: 0.8
  alert_email: "alerts@{{ project_name }}.com"
```

### Environment Variables

Create a `.env` file:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME={{ project_name }}

# DVC
DVC_REMOTE_URL=s3://your-bucket/dvc-storage
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Model Registry
MODEL_REGISTRY_URI=http://localhost:5000
MODEL_STAGE=production

# Monitoring
EVIDENTLY_PROJECT_ID=your-project-id
WHYLABS_API_KEY=your-api-key

# Cloud (optional)
AWS_REGION=us-west-2
GCP_PROJECT_ID=your-project-id
AZURE_SUBSCRIPTION_ID=your-subscription-id
```

## 📊 Data Pipeline

### Data Ingestion

```python
# src/data/ingestion.py
import pandas as pd
import dvc.api
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    """Data ingestion class for loading data from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_data_path = config["data"]["raw_path"]
    
    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(f"{self.raw_data_path}/{filename}")
            logger.info(f"Loaded {len(data)} rows from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            raise
    
    def load_from_dvc(self, path: str) -> pd.DataFrame:
        """Load data from DVC tracked file."""
        try:
            data_path = dvc.api.get_url(path)
            data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(data)} rows from DVC path: {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from DVC path {path}: {e}")
            raise
    
    def load_from_database(self, query: str, connection_string: str) -> pd.DataFrame:
        """Load data from database."""
        try:
            data = pd.read_sql(query, connection_string)
            logger.info(f"Loaded {len(data)} rows from database")
            return data
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
```

### Data Preprocessing

```python
# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing class for cleaning and transforming data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing values and outliers."""
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Remove outliers
        data = self._remove_outliers(data)
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        logger.info(f"Data cleaning completed. Shape: {data.shape}")
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        categorical_cols = self.config["features"]["categorical_columns"]
        numerical_cols = self.config["features"]["numerical_columns"]
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # Fill numerical columns with median
        for col in numerical_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numerical_cols = self.config["features"]["numerical_columns"]
        
        for col in numerical_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = self.config["features"]["categorical_columns"]
        
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        return data
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        numerical_cols = self.config["features"]["numerical_columns"]
        
        if fit:
            data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
        else:
            data[numerical_cols] = self.scaler.transform(data[numerical_cols])
        
        return data
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        target_col = self.config["features"]["target_column"]
        test_size = self.config["evaluation"]["test_size"]
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
```

## 🤖 Model Development

### Model Training

```python
# src/models/training.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training class for training various ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_name = config["model"]["name"]
        self.algorithm = config["model"]["algorithm"]
        self.hyperparameters = config["model"]["hyperparameters"]
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train the model with the given data."""
        logger.info(f"Training {self.algorithm} model...")
        
        # Initialize model based on algorithm
        if self.algorithm == "random_forest":
            self.model = RandomForestClassifier(**self.hyperparameters)
        elif self.algorithm == "logistic_regression":
            self.model = LogisticRegression(**self.hyperparameters)
        elif self.algorithm == "xgboost":
            self.model = XGBClassifier(**self.hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        return self.model
    
    def train_with_mlflow(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Train model with MLflow tracking."""
        with mlflow.start_run(run_name=f"{self.model_name}_{self.algorithm}"):
            # Log parameters
            mlflow.log_params(self.hyperparameters)
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model
            from src.models.evaluation import ModelEvaluator
            evaluator = ModelEvaluator(self.config)
            metrics = evaluator.evaluate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Save model locally
            model_path = f"data/models/{self.model_name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            
            logger.info(f"Model training with MLflow completed. Run ID: {mlflow.active_run().info.run_id}")
            return mlflow.active_run().info.run_id
```

### Model Evaluation

```python
# src/models/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation class for assessing model performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = config["evaluation"]["metrics"]
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {}
        
        if "accuracy" in self.metrics:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        
        if "precision" in self.metrics:
            metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
        
        if "recall" in self.metrics:
            metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
        
        if "f1" in self.metrics:
            metrics["f1"] = f1_score(y_test, y_pred, average='weighted')
        
        # Log detailed results
        logger.info("Model evaluation completed:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def cross_validate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        cv_folds = self.config["evaluation"]["cv_folds"]
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        
        cv_metrics = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation completed. Mean: {cv_metrics['cv_mean']:.4f}, Std: {cv_metrics['cv_std']:.4f}")
        return cv_metrics
```

## 📊 MLOps Pipeline

### DVC Pipeline

```yaml
# dvc.yaml
stages:
  data_ingestion:
    cmd: python -m src.data.ingestion
    deps:
    - src/data/ingestion.py
    - configs/data_config.yaml
    outs:
    - data/raw/dataset.csv
    
  data_preprocessing:
    cmd: python -m src.data.preprocessing
    deps:
    - src/data/preprocessing.py
    - data/raw/dataset.csv
    - configs/data_config.yaml
    outs:
    - data/processed/train.csv
    - data/processed/test.csv
    
  feature_engineering:
    cmd: python -m src.features.extraction
    deps:
    - src/features/extraction.py
    - data/processed/train.csv
    - data/processed/test.csv
    - configs/feature_config.yaml
    outs:
    - data/processed/features_train.csv
    - data/processed/features_test.csv
    
  model_training:
    cmd: python -m src.models.training
    deps:
    - src/models/training.py
    - data/processed/features_train.csv
    - configs/model_config.yaml
    outs:
    - data/models/model.joblib
    - data/models/metrics.json
    
  model_evaluation:
    cmd: python -m src.models.evaluation
    deps:
    - src/models/evaluation.py
    - data/models/model.joblib
    - data/processed/features_test.csv
    - configs/evaluation_config.yaml
    outs:
    - data/models/evaluation_report.json
```

### MLflow Integration

```python
# src/pipeline/main.py
import mlflow
import mlflow.sklearn
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
import yaml
import logging

logger = logging.getLogger(__name__)

def main():
    """Main pipeline function."""
    # Load configuration
    with open("configs/pipeline_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set MLflow experiment
    mlflow.set_experiment(config["model"]["name"])
    
    with mlflow.start_run():
        # Data ingestion
        logger.info("Starting data ingestion...")
        ingestion = DataIngestion(config)
        data = ingestion.load_from_csv("dataset.csv")
        
        # Data preprocessing
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor(config)
        clean_data = preprocessor.clean_data(data)
        encoded_data = preprocessor.encode_categorical_features(clean_data)
        scaled_data = preprocessor.scale_features(encoded_data)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(scaled_data)
        
        # Model training
        logger.info("Starting model training...")
        trainer = ModelTrainer(config)
        model = trainer.train_model(X_train, y_train)
        
        # Model evaluation
        logger.info("Starting model evaluation...")
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        # Log everything to MLflow
        mlflow.log_params(config["model"]["hyperparameters"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
```

## 🚀 Deployment

### Docker Production

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "-m", "src.deployment.api"]
```

### API Deployment

```python
# src/deployment/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Load model
model = joblib.load("data/models/model.joblib")

app = FastAPI(title="{{ project_name | title }} API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the trained model."""
    try:
        # Convert to numpy array and reshape
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        # Determine confidence level
        if probability > 0.8:
            confidence = "high"
        elif probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

## 📊 Monitoring

### Model Performance Monitoring

```python
# src/monitoring/performance.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_metrics = None
        self.performance_threshold = config["monitoring"]["performance_threshold"]
    
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline performance metrics."""
        self.baseline_metrics = metrics
        logger.info("Baseline metrics set")
    
    def check_performance(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check current performance against baseline."""
        if self.baseline_metrics is None:
            logger.warning("No baseline metrics set")
            return {"status": "warning", "message": "No baseline set"}
        
        alerts = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                degradation = baseline_value - current_value
                
                if degradation > (1 - self.performance_threshold):
                    alerts.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "degradation": degradation
                    })
        
        if alerts:
            logger.warning(f"Performance degradation detected: {alerts}")
            return {"status": "alert", "alerts": alerts}
        else:
            logger.info("Performance is within acceptable range")
            return {"status": "ok", "message": "Performance is good"}
```

### Data Drift Detection

```python
# src/monitoring/drift.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data drift between training and production data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reference_data = None
        self.drift_threshold = config["monitoring"]["drift_threshold"]
    
    def set_reference_data(self, data: pd.DataFrame):
        """Set reference data for drift detection."""
        self.reference_data = data
        logger.info("Reference data set for drift detection")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in current data compared to reference data."""
        if self.reference_data is None:
            logger.warning("No reference data set")
            return {"status": "warning", "message": "No reference data set"}
        
        drift_results = {}
        
        # Check numerical columns
        numerical_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in current_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                drift_results[col] = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "drift_detected": ks_pvalue < self.drift_threshold
                }
        
        # Check for overall drift
        drift_detected = any(
            result["drift_detected"] for result in drift_results.values()
        )
        
        if drift_detected:
            logger.warning("Data drift detected")
            return {"status": "alert", "drift_detected": True, "details": drift_results}
        else:
            logger.info("No data drift detected")
            return {"status": "ok", "drift_detected": False, "details": drift_results}
```

## 🧪 Testing

### Unit Tests

```python
# tests/unit/test_data_preprocessing.py
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, np.nan],
        'feature2': ['A', 'B', 'A', 'C', 'B', 'A'],
        'target': [0, 1, 0, 1, 0, 1]
    })

@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "features": {
            "target_column": "target",
            "categorical_columns": ["feature2"],
            "numerical_columns": ["feature1"]
        },
        "evaluation": {
            "test_size": 0.2
        }
    }

def test_handle_missing_values(sample_data, config):
    """Test missing value handling."""
    preprocessor = DataPreprocessor(config)
    cleaned_data = preprocessor._handle_missing_values(sample_data)
    
    assert not cleaned_data['feature1'].isna().any()
    assert not cleaned_data['feature2'].isna().any()

def test_encode_categorical_features(sample_data, config):
    """Test categorical feature encoding."""
    preprocessor = DataPreprocessor(config)
    encoded_data = preprocessor.encode_categorical_features(sample_data)
    
    assert encoded_data['feature2'].dtype == 'int64'
    assert len(encoded_data['feature2'].unique()) == 3

def test_split_data(sample_data, config):
    """Test data splitting."""
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.split_data(sample_data)
    
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)
```

### Integration Tests

```python
# tests/integration/test_pipeline.py
import pytest
import pandas as pd
from src.pipeline.main import main
from src.data.ingestion import DataIngestion
from src.models.training import ModelTrainer

@pytest.fixture
def sample_dataset():
    """Create sample dataset for integration testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature3': np.random.uniform(0, 10, n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    return data

def test_full_pipeline(sample_dataset, tmp_path):
    """Test the complete ML pipeline."""
    # Save sample data
    data_path = tmp_path / "sample_data.csv"
    sample_dataset.to_csv(data_path, index=False)
    
    # This would test the full pipeline integration
    # Implementation depends on your specific pipeline structure
    pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `pre-commit run --all-files`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Standards

- **Type hints** required for all functions
- **Docstrings** for all public functions and classes
- **Tests** for all new functionality
- **Data validation** with Great Expectations
- **Model validation** before deployment

## 📄 License

This project is licensed under the {{ license }} License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn Team** for the ML library
- **MLflow Team** for the MLOps platform
- **DVC Team** for data version control
- **Template Heaven** for the ML pipeline template

---

**Built with ❤️ using Template Heaven ML Pipeline Template**

*This template provides a comprehensive foundation for building production-ready machine learning pipelines with MLOps best practices and monitoring capabilities.*
