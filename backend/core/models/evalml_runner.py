"""
LightAutoML integration for automated model selection

This module wraps LightAutoML to provide full AutoML capabilities:
- Automatic model selection from multiple algorithms
- Hyperparameter optimization
- Ensemble methods
- Support for tabular data (classification and regression)
- GPU support (optional)

Note: This module was originally designed for EvalML but has been
migrated to LightAutoML for better compatibility with modern
Python/NumPy/Pandas versions.
"""

from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from backend.core.utils.spark_pandas_bridge import spark_to_pandas_safe, pandas_to_spark
from backend.core.discovery import Problem, ProblemType
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

# Lazy import for lightautoml
_lama = None


def _get_lightautoml():
    """Lazy load LightAutoML"""
    global _lama
    if _lama is None:
        try:
            from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
            from lightautoml.tasks import Task
            _lama = {
                'TabularAutoML': TabularAutoML,
                'TabularUtilizedAutoML': TabularUtilizedAutoML,
                'Task': Task
            }
        except ImportError:
            raise ImportError(
                "LightAutoML is not installed. "
                "Install it with: pip install lightautoml"
            )
    return _lama


@dataclass
class AutoMLResult:
    """
    Results from LightAutoML search

    Attributes:
        best_model: Best performing model/automl instance
        best_score: Score of best model on validation set
        oof_predictions: Out-of-fold predictions (DataFrame)
        search_time: Total search time in seconds
        problem_type: Type of problem (binary, multiclass, regression)
        metric: Optimization metric used
        feature_importance: Feature importance from best model
        model_summary: Summary of models used in ensemble
    """
    best_model: Any
    best_score: float
    oof_predictions: pd.DataFrame
    search_time: float
    problem_type: str
    metric: str
    feature_importance: Optional[pd.DataFrame] = None
    model_summary: Optional[Dict[str, Any]] = None


class AutoMLRunner:
    """
    LightAutoML-based AutoML runner

    This class handles:
    - Conversion from PySpark to Pandas for LightAutoML
    - AutoML search with various configurations
    - Model evaluation and comparison
    - Feature importance extraction

    Example:
        runner = AutoMLRunner(spark, problem)

        # Run AutoML
        result = runner.run_automl(
            spark_df,
            timeout=300,  # 5 minutes
        )

        # Get best model info
        print(f"Best score: {result.best_score}")

        # Make predictions
        predictions = runner.predict(spark_df)
    """

    def __init__(
        self,
        spark: SparkSession,
        problem: Problem,
        max_rows_for_pandas: int = 500000,
        verbose: bool = True
    ):
        """
        Initialize the AutoMLRunner

        Args:
            spark: SparkSession instance
            problem: Problem definition
            max_rows_for_pandas: Maximum rows to convert to Pandas
            verbose: Whether to print progress information
        """
        self.spark = spark
        self.problem = problem
        self.max_rows = max_rows_for_pandas
        self.verbose = verbose
        self._automl = None
        self._feature_names = None
        self._task = None

    def _get_task_type(self) -> str:
        """Map problem type to LightAutoML task type"""
        if self.problem.type == getattr(ProblemType, "classification").value:
            return "binary"  # Will be updated to multiclass if needed
        else:
            return "reg"

    def _get_metric(self, task_type: str) -> str:
        """Get default metric for task type"""
        if task_type == "binary":
            return "auc"
        elif task_type == "multiclass":
            return "crossentropy"
        else:
            return "mse"

    def run_automl(
        self,
        spark_df: SparkDataFrame,
        feature_columns: Optional[List[str]] = None,
        timeout: int = 300,
        cpu_limit: int = 4,
        memory_limit: int = 16,
        metric: Optional[str] = None,
        reader_params: Optional[Dict[str, Any]] = None,
        use_algos: Optional[List[List[str]]] = None,
        skip_conn: bool = False,
        general_params: Optional[Dict[str, Any]] = None,
        quick_mode: bool = False
    ) -> AutoMLResult:
        """
        Run LightAutoML search

        Args:
            spark_df: PySpark DataFrame with features and target
            feature_columns: List of feature column names (None = all except target)
            timeout: Maximum search time in seconds
            cpu_limit: Number of CPUs to use
            memory_limit: Memory limit in GB
            metric: Optimization metric (None = auto based on task)
            reader_params: Custom reader parameters for data preprocessing
            use_algos: Algorithms to use as nested list (e.g., [['lgb'], ['lgb', 'cb']])
                       Options: 'lgb', 'cb', 'linear_l2'
            skip_conn: Skip connections in neural networks
            general_params: Additional general parameters
            quick_mode: If True, use fast settings (LightGBM only, fewer CV folds)

        Returns:
            AutoMLResult with search results
        """
        lama = _get_lightautoml()
        TabularAutoML = lama['TabularAutoML']
        Task = lama['Task']

        # Apply quick mode settings
        if quick_mode:
            logger.info("Quick mode enabled - using fast settings")
            timeout = min(timeout, 120)  # Cap at 2 minutes
            use_algos = [['lgb']]  # Only LightGBM
            if reader_params is None:
                reader_params = {}
            reader_params['cv'] = 3  # Fewer CV folds
            if general_params is None:
                general_params = {}
            general_params['use_algos'] = [['lgb']]

        logger.info("Starting LightAutoML search...")
        start_time = time.time()

        # Convert to Pandas
        pdf = spark_to_pandas_safe(spark_df, max_rows=self.max_rows, sample=True)

        # Prepare features and target
        target_col = self.problem.target

        if feature_columns is None:
            feature_columns = [c for c in pdf.columns if c != target_col]

        self._feature_names = feature_columns

        # Determine task type
        task_type = self._get_task_type()

        # Auto-detect multiclass
        if task_type == "binary" and pdf[target_col].nunique() > 2:
            task_type = "multiclass"
            logger.info(f"Detected multiclass classification with {pdf[target_col].nunique()} classes")

        # Set default metric
        if metric is None:
            metric = self._get_metric(task_type)

        logger.info(f"Task type: {task_type}, Metric: {metric}")

        # Create task
        task = Task(task_type, metric=metric)
        self._task = task

        # Configure AutoML
        automl_config = {
            'task': task,
            'timeout': timeout,
            'cpu_limit': cpu_limit,
            'memory_limit': memory_limit,
        }

        if reader_params:
            automl_config['reader_params'] = reader_params

        # Build general_params with use_algos if specified
        final_general_params = general_params.copy() if general_params else {}
        if use_algos:
            final_general_params['use_algos'] = use_algos

        if final_general_params:
            automl_config['general_params'] = final_general_params

        # Create AutoML instance
        automl = TabularAutoML(**automl_config)

        # Set roles for features
        roles = {
            'target': target_col,
        }

        # Fit AutoML
        train_data = pdf[feature_columns + [target_col]]
        oof_pred = automl.fit_predict(train_data, roles=roles, verbose=int(self.verbose))

        self._automl = automl

        # Get validation score
        if hasattr(oof_pred, 'data'):
            oof_df = pd.DataFrame(oof_pred.data, columns=['prediction'])
        else:
            oof_df = pd.DataFrame(oof_pred, columns=['prediction'])

        # Calculate score on OOF predictions
        from sklearn.metrics import (
            roc_auc_score, log_loss, accuracy_score,
            mean_squared_error, r2_score
        )

        y_true = pdf[target_col].values
        y_pred = oof_df['prediction'].values

        if task_type == "binary":
            try:
                best_score = roc_auc_score(y_true, y_pred)
            except Exception:
                best_score = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        elif task_type == "multiclass":
            try:
                best_score = -log_loss(y_true, y_pred)
            except Exception:
                best_score = 0.0
        else:
            best_score = r2_score(y_true, y_pred)

        # Get feature importance
        feature_importance = self._extract_feature_importance(automl)

        # Get model summary
        model_summary = self._get_model_summary(automl)

        search_time = time.time() - start_time

        logger.info(f"AutoML search completed in {search_time:.2f}s")
        logger.info(f"Best score ({metric}): {best_score:.4f}")

        return AutoMLResult(
            best_model=automl,
            best_score=best_score,
            oof_predictions=oof_df,
            search_time=search_time,
            problem_type=task_type,
            metric=metric,
            feature_importance=feature_importance,
            model_summary=model_summary
        )

    def _extract_feature_importance(self, automl) -> Optional[pd.DataFrame]:
        """Extract feature importance from LightAutoML model"""
        try:
            # Try to get feature importance from the automl object
            if hasattr(automl, 'get_feature_scores'):
                fi = automl.get_feature_scores()
                if fi is not None:
                    return pd.DataFrame({
                        'feature': fi.index if hasattr(fi, 'index') else range(len(fi)),
                        'importance': fi.values if hasattr(fi, 'values') else fi
                    }).sort_values('importance', ascending=False)

            # Alternative: try to get from reader
            if hasattr(automl, 'reader') and hasattr(automl.reader, 'used_features'):
                features = automl.reader.used_features
                return pd.DataFrame({
                    'feature': features,
                    'importance': [1.0 / len(features)] * len(features)
                })

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return None

    def _get_model_summary(self, automl) -> Dict[str, Any]:
        """Get summary of models used in the ensemble"""
        summary = {
            'n_models': 0,
            'models': [],
            'levels': 0
        }

        try:
            if hasattr(automl, 'levels'):
                summary['levels'] = len(automl.levels)

                for level_idx, level in enumerate(automl.levels):
                    for pipe in level:
                        model_info = {
                            'level': level_idx,
                            'name': type(pipe).__name__ if hasattr(pipe, '__name__') else str(type(pipe))
                        }
                        summary['models'].append(model_info)
                        summary['n_models'] += 1

        except Exception as e:
            logger.warning(f"Could not extract model summary: {e}")

        return summary

    def predict(
        self,
        spark_df: SparkDataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> SparkDataFrame:
        """
        Make predictions using the best model

        Args:
            spark_df: PySpark DataFrame with features
            feature_columns: Feature columns (None = use same as training)

        Returns:
            PySpark DataFrame with predictions
        """
        if self._automl is None:
            raise ValueError("No model available. Run run_automl first.")

        # Convert to Pandas
        pdf = spark_to_pandas_safe(spark_df, max_rows=None)

        # Use training feature columns if not specified
        if feature_columns is None:
            feature_columns = self._feature_names

        # Make predictions
        predictions = self._automl.predict(pdf[feature_columns])

        # Handle prediction output
        if hasattr(predictions, 'data'):
            pred_values = predictions.data
        else:
            pred_values = predictions

        # Add predictions to DataFrame
        if len(pred_values.shape) == 1:
            pdf['prediction'] = pred_values
        else:
            # Multi-column output (probabilities for multiclass)
            if pred_values.shape[1] == 1:
                pdf['prediction'] = pred_values[:, 0]
            else:
                pdf['prediction'] = pred_values[:, 1] if pred_values.shape[1] == 2 else pred_values.argmax(axis=1)
                for i in range(pred_values.shape[1]):
                    pdf[f'probability_class_{i}'] = pred_values[:, i]

        # Convert back to Spark
        return pandas_to_spark(pdf, self.spark)

    def evaluate(
        self,
        spark_df: SparkDataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on new data

        Args:
            spark_df: PySpark DataFrame with features and target
            feature_columns: Feature columns (None = use same as training)

        Returns:
            Dictionary of evaluation metrics
        """
        if self._automl is None:
            raise ValueError("No model available. Run run_automl first.")

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
        )
        from sklearn.preprocessing import LabelEncoder

        # Convert to Pandas
        pdf = spark_to_pandas_safe(spark_df, max_rows=None)

        if feature_columns is None:
            feature_columns = self._feature_names

        X = pdf[feature_columns]
        y_true_raw = pdf[self.problem.target].values

        # Make predictions
        predictions = self._automl.predict(X)
        if hasattr(predictions, 'data'):
            y_pred = predictions.data
        else:
            y_pred = predictions

        metrics = {}

        task_type = self._task.name if self._task else self._get_task_type()

        if task_type in ['binary', 'multiclass']:
            # Classification metrics
            if len(y_pred.shape) > 1:
                y_pred_class = (y_pred[:, 1] > 0.5).astype(int) if y_pred.shape[1] == 2 else y_pred.argmax(axis=1)
                y_pred_proba = y_pred[:, 1] if y_pred.shape[1] == 2 else y_pred
            else:
                y_pred_class = (y_pred > 0.5).astype(int)
                y_pred_proba = y_pred

            # Encode string labels to integers if needed
            if y_true_raw.dtype == object or isinstance(y_true_raw[0], str):
                le = LabelEncoder()
                y_true = le.fit_transform(y_true_raw)
                # Map desired_result to find positive class
                if self.problem.desired_result is not None:
                    try:
                        positive_class_idx = list(le.classes_).index(self.problem.desired_result)
                        # If desired result maps to 0, flip predictions
                        if positive_class_idx == 0:
                            y_pred_class = 1 - y_pred_class
                            y_pred_proba = 1 - y_pred_proba
                    except ValueError:
                        pass  # desired_result not in classes
            else:
                y_true = y_true_raw

            metrics['accuracy'] = accuracy_score(y_true, y_pred_class)

            try:
                metrics['precision'] = precision_score(y_true, y_pred_class, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred_class, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred_class, average='weighted', zero_division=0)
            except Exception:
                pass

            try:
                if task_type == 'binary':
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr')
            except Exception:
                pass
        else:
            # Regression metrics
            y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
            metrics['r2'] = r2_score(y_true, y_pred_flat)
            metrics['mse'] = mean_squared_error(y_true, y_pred_flat)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred_flat)

        return metrics

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of the trained model

        Returns:
            Dictionary with model details
        """
        if self._automl is None:
            raise ValueError("No model available. Run run_automl first.")

        return self._get_model_summary(self._automl)


# Backwards compatibility alias
EvalMLRunner = AutoMLRunner


def quick_automl(
    spark_df: SparkDataFrame,
    spark: SparkSession,
    problem: Problem,
    timeout: int = 120,
    feature_columns: Optional[List[str]] = None,
    quick_mode: bool = True
) -> Tuple[AutoMLResult, SparkDataFrame]:
    """
    Quick AutoML function for simple use cases

    Args:
        spark_df: PySpark DataFrame
        spark: SparkSession
        problem: Problem definition
        timeout: Maximum search time in seconds
        feature_columns: Feature columns (None = auto-detect)
        quick_mode: If True (default), use fast settings

    Returns:
        Tuple of (AutoMLResult, DataFrame with predictions)
    """
    runner = AutoMLRunner(spark, problem, verbose=False)

    result = runner.run_automl(
        spark_df,
        feature_columns=feature_columns,
        timeout=timeout,
        quick_mode=quick_mode
    )

    predictions = runner.predict(spark_df, feature_columns)

    return result, predictions
