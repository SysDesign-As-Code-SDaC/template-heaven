"""
Data Loader

This module provides data loading and preprocessing capabilities for the AI model template.
It supports various data formats and preprocessing pipelines.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
from abc import ABC, abstractmethod

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface that all data loaders must implement.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Args:
            config: Data loading configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any]:
        """
        Load data from the configured source.

        Returns:
            Tuple of (features, targets)
        """
        pass

    @abstractmethod
    def preprocess_data(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Preprocess the loaded data.

        Args:
            X: Raw features
            y: Raw targets

        Returns:
            Tuple of (processed_features, processed_targets)
        """
        pass

    def split_data(self, X: Any, y: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Split data into train/validation/test sets.

        Args:
            X: Features
            y: Targets

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        split_config = self.config.get('split', {})

        train_ratio = split_config.get('train_ratio', 0.7)
        val_ratio = split_config.get('val_ratio', 0.2)
        test_ratio = split_config.get('test_ratio', 0.1)
        random_state = split_config.get('random_state', 42)

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for data splitting")

        # First split: separate test set
        val_test_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_test_ratio, random_state=random_state,
            stratify=y if split_config.get('stratify', False) else None
        )

        # Second split: separate validation and test sets
        test_ratio_adjusted = test_ratio / val_test_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_adjusted, random_state=random_state,
            stratify=y_temp if split_config.get('stratify', False) else None
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


class CSVDataLoader(BaseDataLoader):
    """
    Data loader for CSV files.

    Supports loading data from CSV files with configurable preprocessing.
    """

    def load_data(self) -> Tuple[Any, Any]:
        """
        Load data from CSV file.

        Returns:
            Tuple of (features, targets)
        """
        file_path = self.config.get('train_path')
        if not file_path:
            raise ValueError("train_path must be specified in data config")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            # Load CSV data
            df = pd.read_csv(file_path)

            # Extract features and target
            target_column = self.config.get('target', {}).get('name', 'target')
            feature_columns = self.config.get('features', [])

            if not feature_columns:
                # Use all columns except target
                feature_columns = [col for col in df.columns if col != target_column]

            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")

            X = df[feature_columns].values
            y = df[target_column].values

            self.logger.info(f"Loaded {len(df)} samples with {len(feature_columns)} features")
            return X, y

        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            raise RuntimeError(f"CSV data loading failed: {e}") from e

    def preprocess_data(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Preprocess CSV data.

        Args:
            X: Raw features
            y: Raw targets

        Returns:
            Tuple of (processed_features, processed_targets)
        """
        preprocessing_config = self.config.get('preprocessing', {})

        # Convert to DataFrame for easier processing
        feature_columns = self.config.get('features', [])
        if feature_columns:
            X_df = pd.DataFrame(X, columns=feature_columns)
        else:
            X_df = pd.DataFrame(X)

        # Handle missing values
        missing_config = preprocessing_config.get('missing_values', {})
        strategy = missing_config.get('strategy', 'mean')

        if strategy == 'drop':
            X_df = X_df.dropna()
        else:
            # Fill missing values
            fill_value = missing_config.get('fill_value')
            if fill_value is not None:
                X_df = X_df.fillna(fill_value)
            else:
                # Use mean/median/mode based on strategy
                for col in X_df.columns:
                    if X_df[col].dtype in ['float64', 'int64']:
                        if strategy == 'mean':
                            X_df[col] = X_df[col].fillna(X_df[col].mean())
                        elif strategy == 'median':
                            X_df[col] = X_df[col].fillna(X_df[col].median())
                    else:
                        # For categorical, use mode
                        mode_value = X_df[col].mode()
                        if not mode_value.empty:
                            X_df[col] = X_df[col].fillna(mode_value[0])

        # Feature scaling
        scaling_config = preprocessing_config.get('scaling', {})
        scaling_method = scaling_config.get('method', 'none')

        if scaling_method == 'standard' and SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_df)
        elif scaling_method == 'min_max' and SKLEARN_AVAILABLE:
            feature_range = scaling_config.get('feature_range', [0, 1])
            scaler = MinMaxScaler(feature_range=feature_range)
            X_scaled = scaler.fit_transform(X_df)
        else:
            X_scaled = X_df.values

        # Handle target preprocessing if needed
        y_processed = y

        self.logger.info("Data preprocessing completed")
        return X_scaled, y_processed


class NumpyDataLoader(BaseDataLoader):
    """
    Data loader for NumPy arrays and .npy files.
    """

    def load_data(self) -> Tuple[Any, Any]:
        """
        Load data from NumPy files or arrays.

        Returns:
            Tuple of (features, targets)
        """
        # This is a simplified implementation
        # In practice, you'd load from .npy files or other sources
        raise NotImplementedError("NumPy data loader not fully implemented")

    def preprocess_data(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Preprocess NumPy data.

        Args:
            X: Raw features
            y: Raw targets

        Returns:
            Tuple of (processed_features, processed_targets)
        """
        # Basic preprocessing for NumPy arrays
        X_processed = np.array(X)
        y_processed = np.array(y)

        return X_processed, y_processed


class JSONDataLoader(BaseDataLoader):
    """
    Data loader for JSON files.
    """

    def load_data(self) -> Tuple[Any, Any]:
        """
        Load data from JSON file.

        Returns:
            Tuple of (features, targets)
        """
        file_path = self.config.get('train_path')
        if not file_path:
            raise ValueError("train_path must be specified in data config")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Assume data is a list of samples with features and target
            if isinstance(data, list) and len(data) > 0:
                # Extract features and target from each sample
                target_field = self.config.get('target', {}).get('name', 'target')
                feature_fields = self.config.get('features', [])

                features_list = []
                targets_list = []

                for sample in data:
                    if feature_fields:
                        features = [sample.get(field, 0) for field in feature_fields]
                    else:
                        # Use all fields except target
                        features = [v for k, v in sample.items() if k != target_field]

                    target = sample.get(target_field, 0)

                    features_list.append(features)
                    targets_list.append(target)

                X = np.array(features_list)
                y = np.array(targets_list)

                self.logger.info(f"Loaded {len(data)} samples from JSON")
                return X, y
            else:
                raise ValueError("JSON data must be a list of samples")

        except Exception as e:
            self.logger.error(f"Failed to load JSON data: {e}")
            raise RuntimeError(f"JSON data loading failed: {e}") from e

    def preprocess_data(self, X: Any, y: Any) -> Tuple[Any, Any]:
        """
        Preprocess JSON data.

        Args:
            X: Raw features
            y: Raw targets

        Returns:
            Tuple of (processed_features, processed_targets)
        """
        # Basic preprocessing - can be extended
        X_processed = np.array(X)
        y_processed = np.array(y)

        return X_processed, y_processed


class DataLoaderFactory:
    """
    Factory for creating data loaders based on configuration.
    """

    _loaders = {
        'csv': CSVDataLoader,
        'json': JSONDataLoader,
        'numpy': NumpyDataLoader,
    }

    @classmethod
    def create_loader(cls, config: Dict[str, Any]) -> BaseDataLoader:
        """
        Create a data loader based on configuration.

        Args:
            config: Data loading configuration

        Returns:
            Data loader instance
        """
        format_type = config.get('format', 'csv').lower()

        if format_type not in cls._loaders:
            available_formats = list(cls._loaders.keys())
            raise ValueError(f"Unsupported data format: {format_type}. "
                           f"Available formats: {available_formats}")

        loader_class = cls._loaders[format_type]
        return loader_class(config)

    @classmethod
    def register_loader(cls, format_type: str, loader_class: type) -> None:
        """
        Register a new data loader.

        Args:
            format_type: String identifier for the format
            loader_class: Data loader class
        """
        cls._loaders[format_type] = loader_class

    @classmethod
    def get_available_formats(cls) -> List[str]:
        """
        Get list of available data formats.

        Returns:
            List of supported format identifiers
        """
        return list(cls._loaders.keys())


def load_and_preprocess_data(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Convenience function to load and preprocess data.

    Args:
        config: Data configuration

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Create data loader
    loader = DataLoaderFactory.create_loader(config)

    # Load data
    X, y = loader.load_data()

    # Preprocess data
    X_processed, y_processed = loader.preprocess_data(X, y)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X_processed, y_processed)

    return X_train, X_val, X_test, y_train, y_val, y_test
