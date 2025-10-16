"""
Model Evaluator

This module provides comprehensive model evaluation capabilities including
various metrics, cross-validation, and performance analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
from datetime import datetime

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        confusion_matrix, classification_report, roc_auc_score,
        roc_curve, precision_recall_curve, average_precision_score
    )
    from sklearn.model_selection import cross_val_score, cross_validate
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and analysis.

    This class provides evaluation capabilities for different types of models
    and tasks (classification, regression) with support for cross-validation
    and detailed performance analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Some metrics will be limited.")

    def evaluate(self, model: Any, X: Any, y: Any, task_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            model: Trained model to evaluate
            X: Test features
            y: Test targets
            task_type: Type of task ('classification', 'regression', 'multiclass')

        Returns:
            Dictionary containing evaluation metrics and analysis
        """
        self.logger.info("Starting model evaluation...")

        # Determine task type if not provided
        if task_type is None:
            task_type = self._determine_task_type(y)

        # Get predictions
        predictions = model.predict(X)

        # Basic metrics
        metrics = self._calculate_basic_metrics(predictions, y, task_type)

        # Advanced metrics and analysis
        if task_type in ['classification', 'multiclass']:
            metrics.update(self._calculate_classification_metrics(predictions, y))
        elif task_type == 'regression':
            metrics.update(self._calculate_regression_metrics(predictions, y))

        # Model-specific metrics
        metrics.update(self._calculate_model_specific_metrics(model, X, y))

        # Metadata
        metrics['metadata'] = {
            'task_type': task_type,
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
            'evaluation_timestamp': datetime.utcnow().isoformat(),
            'model_type': getattr(model, 'model_type', 'unknown'),
            'framework': getattr(model, 'framework', 'unknown')
        }

        self.logger.info(f"Evaluation completed: {metrics.get('accuracy', metrics.get('r2_score', 'N/A'))}")
        return metrics

    def _determine_task_type(self, y: Any) -> str:
        """Determine the task type from target values."""
        y_array = np.array(y)

        # Check if classification
        unique_values = np.unique(y_array)
        n_unique = len(unique_values)

        if n_unique == 2:
            return 'classification'
        elif n_unique > 2 and n_unique <= 20:  # Arbitrary threshold for multiclass
            return 'multiclass'
        else:
            return 'regression'

    def _calculate_basic_metrics(self, predictions: Any, y_true: Any, task_type: str) -> Dict[str, float]:
        """Calculate basic evaluation metrics."""
        metrics = {}

        try:
            predictions = np.array(predictions)
            y_true = np.array(y_true)

            if task_type in ['classification', 'multiclass']:
                if SKLEARN_AVAILABLE:
                    # Accuracy
                    metrics['accuracy'] = accuracy_score(y_true, predictions)

                    # Precision, Recall, F1 (weighted for multiclass)
                    average_type = 'weighted' if task_type == 'multiclass' else 'binary'
                    metrics['precision'] = precision_score(y_true, predictions, average=average_type, zero_division=0)
                    metrics['recall'] = recall_score(y_true, predictions, average=average_type, zero_division=0)
                    metrics['f1_score'] = f1_score(y_true, predictions, average=average_type, zero_division=0)

            elif task_type == 'regression':
                if SKLEARN_AVAILABLE:
                    metrics['mse'] = mean_squared_error(y_true, predictions)
                    metrics['mae'] = mean_absolute_error(y_true, predictions)
                    metrics['r2_score'] = r2_score(y_true, predictions)
                    metrics['rmse'] = np.sqrt(metrics['mse'])

        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")

        return metrics

    def _calculate_classification_metrics(self, predictions: Any, y_true: Any) -> Dict[str, Any]:
        """Calculate detailed classification metrics."""
        metrics = {}

        if not SKLEARN_AVAILABLE:
            return metrics

        try:
            predictions = np.array(predictions)
            y_true = np.array(y_true)

            # Confusion matrix
            cm = confusion_matrix(y_true, predictions)
            metrics['confusion_matrix'] = cm.tolist()

            # Classification report
            report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
            metrics['classification_report'] = report

            # ROC AUC (for binary classification)
            if len(np.unique(y_true)) == 2:
                try:
                    # Get prediction probabilities if available
                    # This is a simplified approach - in practice, you'd need the model's predict_proba method
                    if hasattr(predictions, 'shape') and predictions.shape[1] > 1:
                        # Predictions are probabilities
                        y_prob = predictions[:, 1]
                    else:
                        # Convert to binary probabilities (simplified)
                        y_prob = predictions.astype(float)

                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

                    # ROC curve points
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    metrics['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    }

                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_true, y_prob)
                    metrics['precision_recall_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist()
                    }

                    metrics['average_precision'] = average_precision_score(y_true, y_prob)

                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC metrics: {e}")

        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {e}")

        return metrics

    def _calculate_regression_metrics(self, predictions: Any, y_true: Any) -> Dict[str, Any]:
        """Calculate detailed regression metrics."""
        metrics = {}

        try:
            predictions = np.array(predictions)
            y_true = np.array(y_true)

            # Additional regression metrics
            errors = predictions - y_true

            metrics['mean_error'] = float(np.mean(errors))
            metrics['median_error'] = float(np.median(errors))
            metrics['std_error'] = float(np.std(errors))
            metrics['min_error'] = float(np.min(errors))
            metrics['max_error'] = float(np.max(errors))

            # Percentage errors
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100
                metrics['mape'] = float(mape)

            # Residual analysis
            metrics['residual_analysis'] = {
                'skewness': float(self._calculate_skewness(errors)),
                'kurtosis': float(self._calculate_kurtosis(errors))
            }

        except Exception as e:
            self.logger.error(f"Error calculating regression metrics: {e}")

        return metrics

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def _calculate_model_specific_metrics(self, model: Any, X: Any, y: Any) -> Dict[str, Any]:
        """Calculate model-specific metrics."""
        metrics = {}

        try:
            # Feature importance (for tree-based models)
            if hasattr(model, 'get_feature_importances'):
                importance = model.get_feature_importances()
                if importance is not None:
                    metrics['feature_importance'] = {
                        'values': importance.tolist(),
                        'top_features': np.argsort(importance)[-10:][::-1].tolist()
                    }

            # Model coefficients (for linear models)
            if hasattr(model, 'get_coefficients'):
                coeffs = model.get_coefficients()
                if coeffs is not None:
                    metrics['coefficients'] = {
                        'values': coeffs.tolist() if hasattr(coeffs, 'tolist') else coeffs,
                        'abs_values': np.abs(coeffs).tolist() if hasattr(coeffs, 'tolist') else np.abs(coeffs)
                    }

            # Model parameters
            if hasattr(model, 'get_model_params'):
                params = model.get_model_params()
                if params:
                    metrics['model_parameters'] = params

        except Exception as e:
            self.logger.error(f"Error calculating model-specific metrics: {e}")

        return metrics

    def cross_validate(self, model: Any, X: Any, y: Any, cv_folds: int = 5,
                      scoring: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            scoring: List of scoring metrics

        Returns:
            Cross-validation results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for cross-validation")

        self.logger.info(f"Starting {cv_folds}-fold cross-validation...")

        try:
            # Default scoring metrics
            if scoring is None:
                task_type = self._determine_task_type(y)
                if task_type in ['classification', 'multiclass']:
                    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
                else:
                    scoring = ['neg_mean_squared_error', 'r2']

            # Perform cross-validation
            cv_results = cross_validate(
                estimator=model._model,  # Use the internal model
                X=X,
                y=y,
                cv=cv_folds,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )

            # Process results
            results = {
                'cv_folds': cv_folds,
                'test_scores': {},
                'train_scores': {},
                'mean_scores': {},
                'std_scores': {}
            }

            for metric in scoring:
                test_key = f'test_{metric}'
                train_key = f'train_{metric}'

                if test_key in cv_results:
                    results['test_scores'][metric] = cv_results[test_key].tolist()
                    results['mean_scores'][f'test_{metric}'] = float(np.mean(cv_results[test_key]))
                    results['std_scores'][f'test_{metric}'] = float(np.std(cv_results[test_key]))

                if train_key in cv_results:
                    results['train_scores'][metric] = cv_results[train_key].tolist()
                    results['mean_scores'][f'train_{metric}'] = float(np.mean(cv_results[train_key]))
                    results['std_scores'][f'train_{metric}'] = float(np.std(cv_results[train_key]))

            results['fit_times'] = cv_results['fit_time'].tolist()
            results['score_times'] = cv_results['score_time'].tolist()

            self.logger.info(f"Cross-validation completed: mean accuracy = {results['mean_scores'].get('test_accuracy', 'N/A')}")
            return results

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            raise RuntimeError(f"Cross-validation failed: {e}") from e

    def generate_report(self, evaluation_results: Dict[str, Any],
                       save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            evaluation_results: Results from evaluate() method
            save_path: Path to save the report (optional)

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Metadata
        metadata = evaluation_results.get('metadata', {})
        report_lines.append("MODEL INFORMATION:")
        report_lines.append(f"  Type: {metadata.get('model_type', 'Unknown')}")
        report_lines.append(f"  Framework: {metadata.get('framework', 'Unknown')}")
        report_lines.append(f"  Task Type: {metadata.get('task_type', 'Unknown')}")
        report_lines.append(f"  Samples: {metadata.get('n_samples', 'Unknown')}")
        report_lines.append(f"  Features: {metadata.get('n_features', 'Unknown')}")
        report_lines.append(f"  Evaluation Time: {metadata.get('evaluation_timestamp', 'Unknown')}")
        report_lines.append("")

        # Basic Metrics
        report_lines.append("PERFORMANCE METRICS:")
        task_type = metadata.get('task_type', 'unknown')

        if task_type in ['classification', 'multiclass']:
            metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        else:
            metrics_to_show = ['r2_score', 'mse', 'mae', 'rmse']

        for metric in metrics_to_show:
            if metric in evaluation_results:
                value = evaluation_results[metric]
                if isinstance(value, float):
                    report_lines.append(f"  {metric.upper()}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric.upper()}: {value}")

        report_lines.append("")

        # Confusion Matrix (for classification)
        if 'confusion_matrix' in evaluation_results:
            report_lines.append("CONFUSION MATRIX:")
            cm = evaluation_results['confusion_matrix']
            for row in cm:
                report_lines.append(f"  {row}")
            report_lines.append("")

        # Feature Importance (if available)
        if 'feature_importance' in evaluation_results:
            report_lines.append("TOP FEATURES:")
            importance_data = evaluation_results['feature_importance']
            top_features = importance_data.get('top_features', [])
            values = importance_data.get('values', [])

            for i, feature_idx in enumerate(top_features[:10]):
                if feature_idx < len(values):
                    importance = values[feature_idx]
                    report_lines.append(f"  Feature {feature_idx}: {importance:.4f}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        self._generate_recommendations(evaluation_results, report_lines)
        report_lines.append("")

        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        # Save report if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to: {save_path}")

        return report

    def _generate_recommendations(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Generate recommendations based on evaluation results."""
        task_type = results.get('metadata', {}).get('task_type', 'unknown')

        if task_type in ['classification', 'multiclass']:
            accuracy = results.get('accuracy', 0)
            if accuracy > 0.9:
                report_lines.append("  ✓ Excellent performance! Model is highly accurate.")
            elif accuracy > 0.8:
                report_lines.append("  ✓ Good performance. Consider fine-tuning for better results.")
            elif accuracy > 0.7:
                report_lines.append("  ⚠ Moderate performance. May need significant improvements.")
            else:
                report_lines.append("  ✗ Poor performance. Consider different model or extensive retraining.")

        elif task_type == 'regression':
            r2 = results.get('r2_score', 0)
            if r2 > 0.8:
                report_lines.append("  ✓ Excellent fit! Model explains most of the variance.")
            elif r2 > 0.6:
                report_lines.append("  ✓ Good fit. Consider feature engineering for improvements.")
            elif r2 > 0.3:
                report_lines.append("  ⚠ Moderate fit. May need better features or model selection.")
            else:
                report_lines.append("  ✗ Poor fit. Consider different model or feature engineering.")

        # General recommendations
        if 'roc_auc' in results and results['roc_auc'] < 0.7:
            report_lines.append("  ⚠ Consider addressing class imbalance or feature engineering.")

        if 'feature_importance' in results:
            importance_values = results['feature_importance'].get('values', [])
            if len(importance_values) > 0 and max(importance_values) < 0.1:
                report_lines.append("  ⚠ Features have low importance. Consider feature selection or engineering.")

    def compare_models(self, model_results: List[Dict[str, Any]],
                      metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.

        Args:
            model_results: List of evaluation result dictionaries
            metric: Metric to use for comparison

        Returns:
            Comparison results
        """
        comparison = {
            'metric': metric,
            'models': [],
            'best_model_index': None,
            'best_score': float('-inf') if metric in ['accuracy', 'f1_score', 'r2_score'] else float('inf')
        }

        for i, results in enumerate(model_results):
            model_info = {
                'index': i,
                'model_type': results.get('metadata', {}).get('model_type', 'Unknown'),
                'framework': results.get('metadata', {}).get('framework', 'Unknown'),
                'score': results.get(metric)
            }

            comparison['models'].append(model_info)

            if model_info['score'] is not None:
                is_better = ((metric in ['accuracy', 'f1_score', 'r2_score'] and
                             model_info['score'] > comparison['best_score']) or
                            (metric in ['mse', 'mae'] and
                             model_info['score'] < comparison['best_score']))

                if is_better:
                    comparison['best_score'] = model_info['score']
                    comparison['best_model_index'] = i

        return comparison
