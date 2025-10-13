# Cookiecutter Data Science Template

A logical, reasonably standardized, but flexible project structure for doing and sharing data science work.

## 🚀 Features

- **Standardized Project Structure** for data science projects
- **Environment Management** with conda/pip
- **Jupyter Notebooks** for exploration and analysis
- **Data Versioning** with DVC support
- **Model Tracking** with MLflow
- **Testing Framework** with pytest
- **Documentation** with Sphinx
- **CI/CD** with GitHub Actions
- **Docker** support for reproducible environments

## 📋 Prerequisites

- Python 3.8+
- Git
- Conda or pip
- Docker (optional)

## 🛠️ Quick Start

### 1. Create New Project

```bash
# Install cookiecutter
pip install cookiecutter

# Create project from template
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

### 2. Project Setup

```bash
cd your-project-name
conda env create -f environment.yml
conda activate your-project-name
```

### 3. Install Development Dependencies

```bash
pip install -e .
```

## 📁 Project Structure

```
├── data/
│   ├── external/          # Data from third party sources
│   ├── interim/           # Intermediate data that has been transformed
│   ├── processed/         # The final, canonical data sets for modeling
│   └── raw/              # The original, immutable data dump
├── docs/                 # Documentation
├── models/               # Trained and serialized models
├── notebooks/            # Jupyter notebooks for exploration
├── references/           # Data dictionaries, manuals, and papers
├── reports/              # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/          # Generated graphics and figures
├── src/                  # Source code for use in this project
│   ├── data/             # Scripts to download or generate data
│   ├── features/         # Scripts to turn raw data into features
│   ├── models/           # Scripts to train models and make predictions
│   └── visualization/    # Scripts to create exploratory and results oriented visualizations
├── tests/                # Unit tests
├── .gitignore
├── environment.yml       # Conda environment file
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md
```

## 🔧 Available Scripts

```bash
# Environment
conda env create -f environment.yml
conda activate your-project-name

# Development
pip install -e .
python -m pytest tests/
jupyter lab

# Data versioning
dvc add data/raw/dataset.csv
dvc push

# Model tracking
mlflow ui
python src/models/train_model.py

# Documentation
sphinx-build -b html docs/ docs/_build/html
```

## 🗄️ Data Management

### Raw Data

Place original, immutable data in `data/raw/`:

```bash
data/raw/
├── train.csv
├── test.csv
└── external/
    └── census_data.csv
```

### Processed Data

Store transformed data in `data/processed/`:

```python
# src/data/make_dataset.py
import pandas as pd
from pathlib import Path

def load_data(data_path: Path) -> pd.DataFrame:
    """Load raw data from CSV file."""
    return pd.read_csv(data_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.median())
    
    return df

if __name__ == "__main__":
    # Load and clean data
    raw_data = load_data(Path("data/raw/train.csv"))
    clean_data(raw_data).to_csv("data/processed/train_clean.csv", index=False)
```

## 🧪 Testing

```python
# tests/test_data.py
import pytest
import pandas as pd
from pathlib import Path
from src.data.make_dataset import load_data, clean_data

def test_load_data():
    """Test data loading function."""
    data_path = Path("data/raw/test_sample.csv")
    df = load_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_clean_data():
    """Test data cleaning function."""
    # Create sample data
    df = pd.DataFrame({
        'A': [1, 2, 2, 3, None],
        'B': [4, 5, 5, 6, 7]
    })
    
    cleaned_df = clean_data(df)
    assert len(cleaned_df) == 4  # Duplicate removed
    assert cleaned_df['A'].isna().sum() == 0  # Missing values filled
```

## 📊 Model Training

```python
# src/models/train_model.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

def train_model(X_train, y_train):
    """Train a random forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/processed/train_clean.csv")
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    with mlflow.start_run():
        model = train_model(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")
        
        # Save model
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "models/model.pkl")
```

## 📈 Visualization

```python
# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, save_path: str = None):
    """Plot feature importance from trained model."""
    importance = model.feature_importances_
    indices = importance.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project
RUN pip install -e .

# Expose port for Jupyter
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### MLflow Model Serving

```python
# serve_model.py
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

# Load model from MLflow
model = mlflow.sklearn.load_model("models:/your-model/1")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 📚 Learning Resources

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Jupyter Documentation](https://jupyter.org/documentation)

## 🔗 Upstream Source

- **Repository**: [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
- **Documentation**: [drivendata.github.io/cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)
- **License**: BSD-3-Clause
