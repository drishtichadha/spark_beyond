"""
Model comparison framework for measuring impact

This module provides tools to compare different models and feature sets
to measure the impact of feature engineering and model selection.

Key capabilities:
- Compare baseline vs advanced models
- Measure feature engineering impact
- Track experiments and results
- Generate comparison visualizations
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """
    Result from a single experiment

    Attributes:
        experiment_id: Unique identifier
        name: Descriptive name
        model_name: Name of the model used
        feature_set: Name of the feature set used
        metrics: Dictionary of evaluation metrics
        training_time: Time to train in seconds
        prediction_time: Time to predict in seconds
        num_features: Number of features used
        timestamp: When the experiment was run
        metadata: Additional metadata
    """
    experiment_id: str
    name: str
    model_name: str
    feature_set: str
    metrics: Dict[str, float]
    training_time: float
    prediction_time: Optional[float] = None
    num_features: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """
    Results from comparing multiple experiments

    Attributes:
        experiments: List of experiment results
        comparison_table: DataFrame comparing all experiments
        best_experiment: The best performing experiment
        improvements: Dictionary of improvement metrics
        primary_metric: Metric used for comparison
    """
    experiments: List[ExperimentResult]
    comparison_table: pd.DataFrame
    best_experiment: ExperimentResult
    improvements: Dict[str, Dict[str, float]]
    primary_metric: str


class ModelComparison:
    """
    Framework for comparing models and measuring impact

    This class enables:
    - Tracking multiple experiments
    - Comparing baselines vs advanced models
    - Measuring feature engineering impact
    - Generating comparison reports

    Example:
        comparison = ModelComparison(primary_metric='accuracy')

        # Add baseline result
        comparison.add_experiment(
            name="Baseline - Original Features",
            model_name="Decision Tree",
            feature_set="original",
            metrics={'accuracy': 0.75, 'f1': 0.72},
            training_time=5.2
        )

        # Add feature engineering result
        comparison.add_experiment(
            name="Engineered - Auto Features",
            model_name="XGBoost",
            feature_set="engineered",
            metrics={'accuracy': 0.85, 'f1': 0.83},
            training_time=45.0
        )

        # Get comparison
        result = comparison.get_comparison()
        print(result.comparison_table)
        print(f"Best model: {result.best_experiment.name}")
        print(f"Improvements: {result.improvements}")
    """

    def __init__(self, primary_metric: str = 'accuracy'):
        """
        Initialize the comparison framework

        Args:
            primary_metric: Primary metric for ranking experiments
        """
        self.primary_metric = primary_metric
        self._experiments: List[ExperimentResult] = []
        self._experiment_counter = 0

    def add_experiment(
        self,
        name: str,
        model_name: str,
        feature_set: str,
        metrics: Dict[str, float],
        training_time: float,
        prediction_time: Optional[float] = None,
        num_features: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentResult:
        """
        Add an experiment result

        Args:
            name: Descriptive name for the experiment
            model_name: Name of the model used
            feature_set: Name of the feature set ('original', 'engineered', etc.)
            metrics: Dictionary of evaluation metrics
            training_time: Time to train in seconds
            prediction_time: Time to predict in seconds
            num_features: Number of features used
            metadata: Additional metadata

        Returns:
            ExperimentResult object
        """
        self._experiment_counter += 1
        experiment_id = f"exp_{self._experiment_counter:03d}"

        result = ExperimentResult(
            experiment_id=experiment_id,
            name=name,
            model_name=model_name,
            feature_set=feature_set,
            metrics=metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            num_features=num_features,
            metadata=metadata or {}
        )

        self._experiments.append(result)
        logger.info(f"Added experiment: {name} ({experiment_id})")

        return result

    def add_baseline_result(self, baseline_result: Any) -> ExperimentResult:
        """
        Add a BaselineResult from the baseline_models module

        Args:
            baseline_result: BaselineResult object

        Returns:
            ExperimentResult object
        """
        return self.add_experiment(
            name=f"Baseline - {baseline_result.model_name}",
            model_name=baseline_result.model_name,
            feature_set="original",
            metrics=baseline_result.metrics,
            training_time=baseline_result.training_time
        )

    def add_automl_result(
        self,
        automl_result: Any,
        feature_set: str = "engineered"
    ) -> ExperimentResult:
        """
        Add an AutoMLResult from the evalml_runner module

        Args:
            automl_result: AutoMLResult object
            feature_set: Name of the feature set used

        Returns:
            ExperimentResult object
        """
        # Extract metrics from rankings
        metrics = {}
        if len(automl_result.rankings) > 0:
            best_row = automl_result.rankings.iloc[0]
            for col in automl_result.rankings.columns:
                if col not in ['id', 'pipeline_name', 'parameters']:
                    try:
                        metrics[col] = float(best_row[col])
                    except (ValueError, TypeError):
                        pass

        return self.add_experiment(
            name=f"AutoML - {automl_result.best_pipeline.name}",
            model_name=automl_result.best_pipeline.name,
            feature_set=feature_set,
            metrics=metrics,
            training_time=automl_result.search_time,
            metadata={
                'objective': automl_result.objective,
                'problem_type': automl_result.problem_type
            }
        )

    def get_comparison(self) -> ComparisonResult:
        """
        Get comparison of all experiments

        Returns:
            ComparisonResult with comparison table and improvements
        """
        if not self._experiments:
            raise ValueError("No experiments to compare")

        # Build comparison table
        rows = []
        for exp in self._experiments:
            row = {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'model': exp.model_name,
                'feature_set': exp.feature_set,
                'training_time': exp.training_time,
                'num_features': exp.num_features
            }
            row.update(exp.metrics)
            rows.append(row)

        comparison_df = pd.DataFrame(rows)

        # Sort by primary metric (descending)
        if self.primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                self.primary_metric, ascending=False
            )

        # Find best experiment
        best_exp = max(
            self._experiments,
            key=lambda x: x.metrics.get(self.primary_metric, 0)
        )

        # Calculate improvements
        improvements = self._calculate_improvements()

        return ComparisonResult(
            experiments=self._experiments,
            comparison_table=comparison_df,
            best_experiment=best_exp,
            improvements=improvements,
            primary_metric=self.primary_metric
        )

    def _calculate_improvements(self) -> Dict[str, Dict[str, float]]:
        """Calculate improvements between different feature sets and models"""
        improvements = {}

        # Group by feature set
        feature_sets = {}
        for exp in self._experiments:
            fs = exp.feature_set
            if fs not in feature_sets:
                feature_sets[fs] = []
            feature_sets[fs].append(exp)

        # Calculate feature engineering impact
        if 'original' in feature_sets and 'engineered' in feature_sets:
            # Best original vs best engineered
            best_original = max(
                feature_sets['original'],
                key=lambda x: x.metrics.get(self.primary_metric, 0)
            )
            best_engineered = max(
                feature_sets['engineered'],
                key=lambda x: x.metrics.get(self.primary_metric, 0)
            )

            original_score = best_original.metrics.get(self.primary_metric, 0)
            engineered_score = best_engineered.metrics.get(self.primary_metric, 0)

            if original_score > 0:
                pct_improvement = ((engineered_score - original_score) / original_score) * 100
            else:
                pct_improvement = 0

            improvements['feature_engineering'] = {
                'original_score': original_score,
                'engineered_score': engineered_score,
                'absolute_improvement': engineered_score - original_score,
                'percentage_improvement': pct_improvement,
                'original_model': best_original.model_name,
                'engineered_model': best_engineered.model_name
            }

        # Calculate baseline vs best improvement
        baselines = [e for e in self._experiments if 'Baseline' in e.name or 'Naive' in e.name]
        if baselines:
            worst_baseline = min(
                baselines,
                key=lambda x: x.metrics.get(self.primary_metric, 0)
            )
            best_overall = max(
                self._experiments,
                key=lambda x: x.metrics.get(self.primary_metric, 0)
            )

            baseline_score = worst_baseline.metrics.get(self.primary_metric, 0)
            best_score = best_overall.metrics.get(self.primary_metric, 0)

            if baseline_score > 0:
                pct_improvement = ((best_score - baseline_score) / baseline_score) * 100
            else:
                pct_improvement = 0

            improvements['vs_baseline'] = {
                'baseline_score': baseline_score,
                'best_score': best_score,
                'absolute_improvement': best_score - baseline_score,
                'percentage_improvement': pct_improvement,
                'baseline_model': worst_baseline.model_name,
                'best_model': best_overall.model_name
            }

        # Calculate model comparison (same feature set)
        for fs, exps in feature_sets.items():
            if len(exps) > 1:
                worst = min(exps, key=lambda x: x.metrics.get(self.primary_metric, 0))
                best = max(exps, key=lambda x: x.metrics.get(self.primary_metric, 0))

                worst_score = worst.metrics.get(self.primary_metric, 0)
                best_score = best.metrics.get(self.primary_metric, 0)

                if worst_score > 0:
                    pct_improvement = ((best_score - worst_score) / worst_score) * 100
                else:
                    pct_improvement = 0

                improvements[f'model_selection_{fs}'] = {
                    'worst_score': worst_score,
                    'best_score': best_score,
                    'absolute_improvement': best_score - worst_score,
                    'percentage_improvement': pct_improvement,
                    'worst_model': worst.model_name,
                    'best_model': best.model_name
                }

        return improvements

    def get_summary_report(self) -> str:
        """
        Generate a text summary report

        Returns:
            Formatted string report
        """
        result = self.get_comparison()

        lines = [
            "=" * 60,
            "MODEL COMPARISON REPORT",
            "=" * 60,
            f"Primary Metric: {self.primary_metric}",
            f"Total Experiments: {len(result.experiments)}",
            "",
            "EXPERIMENT RANKINGS",
            "-" * 40
        ]

        # Add ranking table
        for i, exp in enumerate(sorted(
            result.experiments,
            key=lambda x: x.metrics.get(self.primary_metric, 0),
            reverse=True
        )):
            score = exp.metrics.get(self.primary_metric, 0)
            lines.append(f"{i+1}. {exp.name}")
            lines.append(f"   Model: {exp.model_name}")
            lines.append(f"   Feature Set: {exp.feature_set}")
            lines.append(f"   {self.primary_metric}: {score:.4f}")
            lines.append(f"   Training Time: {exp.training_time:.2f}s")
            lines.append("")

        # Add improvements section
        lines.extend([
            "IMPACT ANALYSIS",
            "-" * 40
        ])

        for impact_name, impact_data in result.improvements.items():
            lines.append(f"\n{impact_name.upper().replace('_', ' ')}:")
            lines.append(f"  Improvement: {impact_data.get('absolute_improvement', 0):.4f}")
            lines.append(f"  Percentage: {impact_data.get('percentage_improvement', 0):.2f}%")

        lines.extend([
            "",
            "=" * 60,
            f"BEST MODEL: {result.best_experiment.name}",
            f"BEST SCORE: {result.best_experiment.metrics.get(self.primary_metric, 0):.4f}",
            "=" * 60
        ])

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all experiments as a DataFrame

        Returns:
            DataFrame with all experiment data
        """
        return self.get_comparison().comparison_table

    def to_json(self) -> str:
        """
        Export comparison results as JSON

        Returns:
            JSON string
        """
        result = self.get_comparison()

        data = {
            'primary_metric': result.primary_metric,
            'experiments': [
                {
                    'id': exp.experiment_id,
                    'name': exp.name,
                    'model': exp.model_name,
                    'feature_set': exp.feature_set,
                    'metrics': exp.metrics,
                    'training_time': exp.training_time,
                    'num_features': exp.num_features,
                    'timestamp': exp.timestamp.isoformat()
                }
                for exp in result.experiments
            ],
            'best_experiment': result.best_experiment.experiment_id,
            'improvements': result.improvements
        }

        return json.dumps(data, indent=2)

    def clear(self) -> None:
        """Clear all experiments"""
        self._experiments = []
        self._experiment_counter = 0
        logger.info("Cleared all experiments")


def create_comparison_from_results(
    baseline_results: List[Any],
    automl_result: Optional[Any] = None,
    xgboost_result: Optional[Dict[str, Any]] = None,
    primary_metric: str = 'accuracy'
) -> ComparisonResult:
    """
    Create a comparison from various result types

    Args:
        baseline_results: List of BaselineResult objects
        automl_result: Optional AutoMLResult object
        xgboost_result: Optional dict with XGBoost results
        primary_metric: Metric for comparison

    Returns:
        ComparisonResult
    """
    comparison = ModelComparison(primary_metric=primary_metric)

    # Add baseline results
    for result in baseline_results:
        comparison.add_baseline_result(result)

    # Add AutoML result
    if automl_result:
        comparison.add_automl_result(automl_result)

    # Add XGBoost result
    if xgboost_result:
        comparison.add_experiment(
            name="XGBoost - Engineered Features",
            model_name="XGBoost",
            feature_set="engineered",
            metrics=xgboost_result.get('metrics', {}),
            training_time=xgboost_result.get('training_time', 0)
        )

    return comparison.get_comparison()
