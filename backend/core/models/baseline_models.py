"""
Baseline models for comparison

This module provides simple baseline models to compare against
more complex models and measure the impact of feature engineering.

Baseline models include:
- Logistic Regression (classification)
- Decision Tree (classification/regression)
- Linear Regression (regression)
- Naive baselines (majority class, mean prediction)
"""

from pyspark.sql import DataFrame
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator
)
from pyspark.ml.feature import VectorAssembler
from backend.core.discovery import Problem, ProblemType
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """
    Results from baseline model training

    Attributes:
        model_name: Name of the baseline model
        model_type: 'classification' or 'regression'
        metrics: Dictionary of evaluation metrics
        training_time: Time taken to train (seconds)
        predictions: DataFrame with predictions (optional)
        model: Trained model object
        feature_importances: Feature importance scores (if available)
    """
    model_name: str
    model_type: str
    metrics: Dict[str, float]
    training_time: float
    predictions: Optional[DataFrame] = None
    model: Optional[Any] = None
    feature_importances: Optional[Dict[str, float]] = None


class BaselineModels:
    """
    Collection of baseline models for comparison

    This class provides simple baseline models that serve as
    benchmarks for measuring the impact of:
    - Feature engineering
    - More complex models
    - AutoML approaches

    Example:
        baselines = BaselineModels(problem, train_df, test_df, feature_col='features')

        # Train all baselines
        results = baselines.train_all_baselines()

        # Compare results
        for result in results:
            print(f"{result.model_name}: {result.metrics}")
    """

    def __init__(
        self,
        problem: Problem,
        train_df: DataFrame,
        test_df: DataFrame,
        feature_col: str = "features",
        label_col: Optional[str] = None
    ):
        """
        Initialize baseline models

        Args:
            problem: Problem definition with target and type
            train_df: Training DataFrame
            test_df: Test DataFrame
            feature_col: Name of the feature vector column
            label_col: Name of the label column (defaults to problem.target)
        """
        self.problem = problem
        self.train_df = train_df
        self.test_df = test_df
        self.feature_col = feature_col
        self.label_col = label_col or problem.target
        self.is_classification = problem.type == getattr(ProblemType, "classification").value
        self._results: List[BaselineResult] = []

    def train_logistic_regression(
        self,
        max_iter: int = 100,
        reg_param: float = 0.0
    ) -> BaselineResult:
        """
        Train Logistic Regression baseline (classification only)

        Args:
            max_iter: Maximum iterations
            reg_param: Regularization parameter

        Returns:
            BaselineResult with metrics
        """
        if not self.is_classification:
            raise ValueError("Logistic Regression is only for classification problems")

        logger.info("Training Logistic Regression baseline...")
        start_time = time.time()

        # Train model
        lr = LogisticRegression(
            featuresCol=self.feature_col,
            labelCol=self.label_col,
            maxIter=max_iter,
            regParam=reg_param,
            predictionCol="prediction",
            probabilityCol="probability"
        )

        model = lr.fit(self.train_df)
        training_time = time.time() - start_time

        # Evaluate
        predictions = model.transform(self.test_df)
        metrics = self._evaluate_classification(predictions)

        result = BaselineResult(
            model_name="Logistic Regression",
            model_type="classification",
            metrics=metrics,
            training_time=training_time,
            predictions=predictions,
            model=model
        )

        self._results.append(result)
        logger.info(f"Logistic Regression trained in {training_time:.2f}s: {metrics}")

        return result

    def train_decision_tree(
        self,
        max_depth: int = 5,
        min_instances_per_node: int = 1
    ) -> BaselineResult:
        """
        Train Decision Tree baseline

        Args:
            max_depth: Maximum tree depth
            min_instances_per_node: Minimum instances per leaf

        Returns:
            BaselineResult with metrics
        """
        logger.info("Training Decision Tree baseline...")
        start_time = time.time()

        if self.is_classification:
            dt = DecisionTreeClassifier(
                featuresCol=self.feature_col,
                labelCol=self.label_col,
                maxDepth=max_depth,
                minInstancesPerNode=min_instances_per_node,
                predictionCol="prediction"
            )
        else:
            dt = DecisionTreeRegressor(
                featuresCol=self.feature_col,
                labelCol=self.label_col,
                maxDepth=max_depth,
                minInstancesPerNode=min_instances_per_node,
                predictionCol="prediction"
            )

        model = dt.fit(self.train_df)
        training_time = time.time() - start_time

        # Evaluate
        predictions = model.transform(self.test_df)

        if self.is_classification:
            metrics = self._evaluate_classification(predictions)
        else:
            metrics = self._evaluate_regression(predictions)

        # Get feature importances
        feature_importances = None
        if hasattr(model, 'featureImportances'):
            importances = model.featureImportances.toArray()
            feature_importances = {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

        result = BaselineResult(
            model_name="Decision Tree",
            model_type="classification" if self.is_classification else "regression",
            metrics=metrics,
            training_time=training_time,
            predictions=predictions,
            model=model,
            feature_importances=feature_importances
        )

        self._results.append(result)
        logger.info(f"Decision Tree trained in {training_time:.2f}s: {metrics}")

        return result

    def train_linear_regression(
        self,
        max_iter: int = 100,
        reg_param: float = 0.0
    ) -> BaselineResult:
        """
        Train Linear Regression baseline (regression only)

        Args:
            max_iter: Maximum iterations
            reg_param: Regularization parameter

        Returns:
            BaselineResult with metrics
        """
        if self.is_classification:
            raise ValueError("Linear Regression is only for regression problems")

        logger.info("Training Linear Regression baseline...")
        start_time = time.time()

        lr = LinearRegression(
            featuresCol=self.feature_col,
            labelCol=self.label_col,
            maxIter=max_iter,
            regParam=reg_param,
            predictionCol="prediction"
        )

        model = lr.fit(self.train_df)
        training_time = time.time() - start_time

        # Evaluate
        predictions = model.transform(self.test_df)
        metrics = self._evaluate_regression(predictions)

        result = BaselineResult(
            model_name="Linear Regression",
            model_type="regression",
            metrics=metrics,
            training_time=training_time,
            predictions=predictions,
            model=model
        )

        self._results.append(result)
        logger.info(f"Linear Regression trained in {training_time:.2f}s: {metrics}")

        return result

    def train_random_forest(
        self,
        num_trees: int = 20,
        max_depth: int = 5
    ) -> BaselineResult:
        """
        Train Random Forest baseline

        Args:
            num_trees: Number of trees
            max_depth: Maximum tree depth

        Returns:
            BaselineResult with metrics
        """
        logger.info("Training Random Forest baseline...")
        start_time = time.time()

        if self.is_classification:
            rf = RandomForestClassifier(
                featuresCol=self.feature_col,
                labelCol=self.label_col,
                numTrees=num_trees,
                maxDepth=max_depth,
                predictionCol="prediction"
            )
        else:
            from pyspark.ml.regression import RandomForestRegressor
            rf = RandomForestRegressor(
                featuresCol=self.feature_col,
                labelCol=self.label_col,
                numTrees=num_trees,
                maxDepth=max_depth,
                predictionCol="prediction"
            )

        model = rf.fit(self.train_df)
        training_time = time.time() - start_time

        # Evaluate
        predictions = model.transform(self.test_df)

        if self.is_classification:
            metrics = self._evaluate_classification(predictions)
        else:
            metrics = self._evaluate_regression(predictions)

        # Get feature importances
        feature_importances = None
        if hasattr(model, 'featureImportances'):
            importances = model.featureImportances.toArray()
            feature_importances = {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

        result = BaselineResult(
            model_name="Random Forest",
            model_type="classification" if self.is_classification else "regression",
            metrics=metrics,
            training_time=training_time,
            predictions=predictions,
            model=model,
            feature_importances=feature_importances
        )

        self._results.append(result)
        logger.info(f"Random Forest trained in {training_time:.2f}s: {metrics}")

        return result

    def train_naive_baseline(self) -> BaselineResult:
        """
        Train naive baseline (majority class or mean prediction)

        Returns:
            BaselineResult with metrics
        """
        logger.info("Calculating naive baseline...")
        start_time = time.time()

        if self.is_classification:
            # Majority class baseline
            from pyspark.sql import functions as F

            # Get majority class
            majority_class = self.train_df.groupBy(self.label_col) \
                .count() \
                .orderBy(F.desc('count')) \
                .first()[self.label_col]

            # Predict majority class for all
            predictions = self.test_df.withColumn("prediction", F.lit(float(majority_class)))

            metrics = self._evaluate_classification(predictions)
            model_name = f"Naive Baseline (Majority Class: {majority_class})"

        else:
            # Mean prediction baseline
            from pyspark.sql import functions as F

            mean_value = self.train_df.select(F.mean(self.label_col)).collect()[0][0]

            predictions = self.test_df.withColumn("prediction", F.lit(float(mean_value)))

            metrics = self._evaluate_regression(predictions)
            model_name = f"Naive Baseline (Mean: {mean_value:.4f})"

        training_time = time.time() - start_time

        result = BaselineResult(
            model_name=model_name,
            model_type="classification" if self.is_classification else "regression",
            metrics=metrics,
            training_time=training_time,
            predictions=predictions,
            model=None
        )

        self._results.append(result)
        logger.info(f"Naive baseline: {metrics}")

        return result

    def train_all_baselines(
        self,
        include_naive: bool = True,
        include_random_forest: bool = True
    ) -> List[BaselineResult]:
        """
        Train all applicable baseline models

        Args:
            include_naive: Include naive baseline
            include_random_forest: Include Random Forest

        Returns:
            List of BaselineResult objects
        """
        results = []

        # Naive baseline
        if include_naive:
            results.append(self.train_naive_baseline())

        # Decision Tree
        results.append(self.train_decision_tree())

        if self.is_classification:
            # Logistic Regression
            results.append(self.train_logistic_regression())
        else:
            # Linear Regression
            results.append(self.train_linear_regression())

        # Random Forest
        if include_random_forest:
            results.append(self.train_random_forest())

        return results

    def _evaluate_classification(self, predictions: DataFrame) -> Dict[str, float]:
        """Evaluate classification predictions"""
        metrics = {}

        # Accuracy
        accuracy_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="accuracy"
        )
        metrics['accuracy'] = accuracy_evaluator.evaluate(predictions)

        # F1 Score
        f1_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="f1"
        )
        metrics['f1'] = f1_evaluator.evaluate(predictions)

        # Precision
        precision_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="weightedPrecision"
        )
        metrics['precision'] = precision_evaluator.evaluate(predictions)

        # Recall
        recall_evaluator = MulticlassClassificationEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="weightedRecall"
        )
        metrics['recall'] = recall_evaluator.evaluate(predictions)

        # AUC-ROC (for binary classification)
        try:
            auc_evaluator = BinaryClassificationEvaluator(
                rawPredictionCol="probability" if "probability" in predictions.columns else "prediction",
                labelCol=self.label_col,
                metricName="areaUnderROC"
            )
            metrics['auc_roc'] = auc_evaluator.evaluate(predictions)
        except Exception:
            # May fail for multiclass or if probability column missing
            pass

        return metrics

    def _evaluate_regression(self, predictions: DataFrame) -> Dict[str, float]:
        """Evaluate regression predictions"""
        metrics = {}

        # RMSE
        rmse_evaluator = RegressionEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="rmse"
        )
        metrics['rmse'] = rmse_evaluator.evaluate(predictions)

        # MAE
        mae_evaluator = RegressionEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="mae"
        )
        metrics['mae'] = mae_evaluator.evaluate(predictions)

        # R2
        r2_evaluator = RegressionEvaluator(
            predictionCol="prediction",
            labelCol=self.label_col,
            metricName="r2"
        )
        metrics['r2'] = r2_evaluator.evaluate(predictions)

        return metrics

    def get_results(self) -> List[BaselineResult]:
        """Get all baseline results"""
        return self._results

    def get_best_baseline(self, metric: str = 'accuracy') -> BaselineResult:
        """
        Get the best performing baseline

        Args:
            metric: Metric to use for comparison

        Returns:
            Best BaselineResult
        """
        if not self._results:
            raise ValueError("No baselines have been trained yet")

        return max(self._results, key=lambda r: r.metrics.get(metric, 0))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all baseline results

        Returns:
            Dictionary mapping model names to metrics
        """
        return {
            result.model_name: result.metrics
            for result in self._results
        }
