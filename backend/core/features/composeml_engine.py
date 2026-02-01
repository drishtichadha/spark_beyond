"""
ComposeML integration for prediction engineering

This module wraps composeml to automatically define prediction problems
and generate labels for time-series machine learning tasks.

Key capabilities:
- Automatic label generation for temporal prediction problems
- Cutoff time management to prevent data leakage
- Flexible labeling function definitions
- Support for various prediction windows
"""

from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F
from backend.core.utils.spark_pandas_bridge import spark_to_pandas_safe, pandas_to_spark
from typing import List, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Lazy import for composeml
_composeml = None


def _get_composeml():
    """Lazy load composeml"""
    global _composeml
    if _composeml is None:
        try:
            import composeml as cp
            _composeml = cp
        except ImportError:
            raise ImportError(
                "composeml is not installed. "
                "Install it with: pip install composeml"
            )
    return _composeml


@dataclass
class PredictionProblem:
    """
    Definition of a prediction problem

    Attributes:
        name: Name of the prediction problem
        description: Human-readable description
        target_column: Column to predict
        entity_column: Column identifying entities
        time_column: Timestamp column
        labeling_function: Function to generate labels
        window_size: Size of prediction window (e.g., "30 days")
        gap: Gap between feature cutoff and prediction start
        minimum_data: Minimum data required before making predictions
    """
    name: str
    description: str
    target_column: str
    entity_column: str
    time_column: str
    labeling_function: Optional[Callable] = None
    window_size: Optional[str] = None
    gap: Optional[str] = None
    minimum_data: Optional[str] = None


@dataclass
class LabelingResult:
    """
    Results from label generation

    Attributes:
        label_times: DataFrame with entity_id, cutoff_time, label
        label_column: Name of the label column
        num_labels: Number of labels generated
        label_distribution: Distribution of label values
        cutoff_times: List of unique cutoff times
    """
    label_times: pd.DataFrame
    label_column: str
    num_labels: int
    label_distribution: Dict[Any, int]
    cutoff_times: List


class ComposeMLEngine:
    """
    ComposeML-based prediction engineering engine

    This class handles:
    - Definition of prediction problems for time-series data
    - Automatic label generation with proper temporal handling
    - Cutoff time calculation to prevent data leakage
    - Integration with PySpark DataFrames

    Example:
        engine = ComposeMLEngine(spark)

        # Define a churn prediction problem
        problem = engine.define_problem(
            name="customer_churn_30d",
            entity_column="customer_id",
            time_column="transaction_date",
            window_size="30 days",
            labeling_function=lambda df: df['transaction_count'] == 0
        )

        # Generate labels
        result = engine.generate_labels(
            spark_df,
            problem,
            num_examples_per_entity=5
        )

        # Get cutoff times for feature engineering
        cutoff_times = result.label_times[['customer_id', 'cutoff_time']]
    """

    def __init__(
        self,
        spark: SparkSession,
        max_rows_for_pandas: int = 500000,
        verbose: bool = True
    ):
        """
        Initialize the ComposeMLEngine

        Args:
            spark: SparkSession instance
            max_rows_for_pandas: Maximum rows to convert to Pandas
            verbose: Whether to print progress information
        """
        self.spark = spark
        self.max_rows = max_rows_for_pandas
        self.verbose = verbose
        self._label_maker = None
        self._last_result = None

    def define_problem(
        self,
        name: str,
        entity_column: str,
        time_column: str,
        window_size: str = "30 days",
        gap: str = "0 days",
        minimum_data: str = "7 days",
        labeling_function: Optional[Callable] = None,
        description: str = ""
    ) -> PredictionProblem:
        """
        Define a prediction problem

        Args:
            name: Name for the prediction problem
            entity_column: Column identifying entities (e.g., customer_id)
            time_column: Timestamp column
            window_size: Size of prediction window (e.g., "30 days", "1 week")
            gap: Gap between features and prediction (to prevent leakage)
            minimum_data: Minimum historical data required
            labeling_function: Custom function to generate labels
            description: Human-readable description

        Returns:
            PredictionProblem configuration object
        """
        problem = PredictionProblem(
            name=name,
            description=description or f"Prediction problem: {name}",
            target_column=name,
            entity_column=entity_column,
            time_column=time_column,
            labeling_function=labeling_function,
            window_size=window_size,
            gap=gap,
            minimum_data=minimum_data
        )

        logger.info(f"Defined prediction problem: {name}")
        logger.info(f"  Entity: {entity_column}, Time: {time_column}")
        logger.info(f"  Window: {window_size}, Gap: {gap}")

        return problem

    def generate_labels(
        self,
        spark_df: SparkDataFrame,
        problem: PredictionProblem,
        num_examples_per_entity: int = -1,
        label_type: str = "discrete",
        verbose: bool = True
    ) -> LabelingResult:
        """
        Generate labels for a prediction problem

        Args:
            spark_df: PySpark DataFrame with time-series data
            problem: PredictionProblem definition
            num_examples_per_entity: Max labels per entity (-1 for all)
            label_type: "discrete" for classification, "continuous" for regression
            verbose: Whether to show progress

        Returns:
            LabelingResult with generated labels
        """
        cp = _get_composeml()

        logger.info(f"Generating labels for: {problem.name}")

        # Convert to Pandas
        pdf = spark_to_pandas_safe(spark_df, max_rows=self.max_rows, sample=True)

        # Ensure time column is datetime
        pdf[problem.time_column] = pd.to_datetime(pdf[problem.time_column])

        # Sort by entity and time
        pdf = pdf.sort_values([problem.entity_column, problem.time_column])

        # Create labeling function if not provided
        if problem.labeling_function is None:
            # Default: count records in window
            def default_labeling_function(df):
                return len(df)
            labeling_function = default_labeling_function
        else:
            labeling_function = problem.labeling_function

        # Create LabelMaker
        label_maker = cp.LabelMaker(
            target_dataframe_index=problem.entity_column,
            time_index=problem.time_column,
            labeling_function=labeling_function,
            window_size=problem.window_size
        )

        self._label_maker = label_maker

        # Generate labels
        label_times = label_maker.search(
            pdf,
            num_examples_per_instance=num_examples_per_entity,
            gap=problem.gap,
            minimum_data=problem.minimum_data,
            verbose=verbose
        )

        # Get label distribution
        label_col = label_times.columns[-1]  # Last column is the label
        label_distribution = label_times[label_col].value_counts().to_dict()

        result = LabelingResult(
            label_times=label_times,
            label_column=label_col,
            num_labels=len(label_times),
            label_distribution=label_distribution,
            cutoff_times=label_times['cutoff_time'].unique().tolist() if 'cutoff_time' in label_times.columns else []
        )

        self._last_result = result

        if verbose:
            logger.info(f"Generated {result.num_labels} labels")
            logger.info(f"Label distribution: {result.label_distribution}")

        return result

    def get_cutoff_times(
        self,
        result: Optional[LabelingResult] = None
    ) -> pd.DataFrame:
        """
        Get cutoff times DataFrame for feature engineering

        This DataFrame can be passed to featuretools or other
        feature engineering tools to ensure temporal validity.

        Args:
            result: LabelingResult (uses last result if not provided)

        Returns:
            DataFrame with entity_id and cutoff_time columns
        """
        if result is None:
            result = self._last_result

        if result is None:
            raise ValueError("No labeling result available. Call generate_labels first.")

        # Extract cutoff times
        if 'cutoff_time' in result.label_times.columns:
            cutoff_df = result.label_times[
                [result.label_times.index.name or 'id', 'cutoff_time']
            ].copy()
        else:
            # If no cutoff_time column, create from time index
            cutoff_df = result.label_times.reset_index()
            cutoff_df = cutoff_df[[cutoff_df.columns[0], 'time']].copy()
            cutoff_df.columns = ['id', 'cutoff_time']

        return cutoff_df

    def labels_to_spark(
        self,
        result: LabelingResult,
        original_spark_df: Optional[SparkDataFrame] = None,
        entity_column: Optional[str] = None
    ) -> SparkDataFrame:
        """
        Convert labels to PySpark DataFrame

        Args:
            result: LabelingResult from generate_labels
            original_spark_df: Optional original DataFrame to join with
            entity_column: Entity column name for joining

        Returns:
            PySpark DataFrame with labels
        """
        # Reset index to make entity a column
        label_df = result.label_times.reset_index()

        # Convert to Spark
        spark_labels = pandas_to_spark(label_df, self.spark)

        if original_spark_df is not None and entity_column:
            # Join with original on entity and cutoff time
            # This is useful for getting features at each cutoff point
            spark_labels = spark_labels.withColumnRenamed(
                result.label_column, 'label'
            )

        return spark_labels


# Pre-defined labeling functions for common use cases

def churn_labeling(window_df: pd.DataFrame, threshold: int = 0) -> int:
    """
    Churn labeling: 1 if no activity in window, 0 otherwise

    Args:
        window_df: DataFrame of records in the prediction window
        threshold: Activity threshold (default 0 = any activity)

    Returns:
        1 for churn, 0 for active
    """
    return 1 if len(window_df) <= threshold else 0


def total_spend_labeling(window_df: pd.DataFrame, amount_column: str = 'amount') -> float:
    """
    Total spend in prediction window

    Args:
        window_df: DataFrame of records in the prediction window
        amount_column: Column containing spend amounts

    Returns:
        Total spend in window
    """
    if amount_column in window_df.columns:
        return window_df[amount_column].sum()
    return 0.0


def purchase_probability_labeling(window_df: pd.DataFrame) -> int:
    """
    Binary label: will customer make a purchase in window?

    Args:
        window_df: DataFrame of records in the prediction window

    Returns:
        1 if any purchase, 0 otherwise
    """
    return 1 if len(window_df) > 0 else 0


def average_value_labeling(window_df: pd.DataFrame, value_column: str = 'value') -> float:
    """
    Average value in prediction window

    Args:
        window_df: DataFrame of records in the prediction window
        value_column: Column containing values

    Returns:
        Average value in window
    """
    if value_column in window_df.columns and len(window_df) > 0:
        return window_df[value_column].mean()
    return 0.0


def count_events_labeling(window_df: pd.DataFrame) -> int:
    """
    Count of events in prediction window

    Args:
        window_df: DataFrame of records in the prediction window

    Returns:
        Number of events
    """
    return len(window_df)


def create_threshold_labeling(
    column: str,
    threshold: float,
    aggregation: str = 'sum'
) -> Callable:
    """
    Create a labeling function that thresholds an aggregated value

    Args:
        column: Column to aggregate
        threshold: Threshold value
        aggregation: Aggregation function ('sum', 'mean', 'max', 'min', 'count')

    Returns:
        Labeling function
    """
    def labeling_function(window_df: pd.DataFrame) -> int:
        if column not in window_df.columns:
            return 0

        if aggregation == 'sum':
            value = window_df[column].sum()
        elif aggregation == 'mean':
            value = window_df[column].mean()
        elif aggregation == 'max':
            value = window_df[column].max()
        elif aggregation == 'min':
            value = window_df[column].min()
        elif aggregation == 'count':
            value = len(window_df)
        else:
            value = window_df[column].sum()

        return 1 if value >= threshold else 0

    return labeling_function


class PredictionProblemLibrary:
    """
    Library of common prediction problem templates
    """

    @staticmethod
    def customer_churn(
        entity_column: str = "customer_id",
        time_column: str = "timestamp",
        window_size: str = "30 days",
        gap: str = "0 days"
    ) -> PredictionProblem:
        """Create a customer churn prediction problem"""
        return PredictionProblem(
            name="customer_churn",
            description="Predict if customer will churn in the next window",
            target_column="will_churn",
            entity_column=entity_column,
            time_column=time_column,
            labeling_function=churn_labeling,
            window_size=window_size,
            gap=gap
        )

    @staticmethod
    def next_purchase(
        entity_column: str = "customer_id",
        time_column: str = "timestamp",
        window_size: str = "7 days"
    ) -> PredictionProblem:
        """Create a next purchase prediction problem"""
        return PredictionProblem(
            name="next_purchase",
            description="Predict if customer will make a purchase",
            target_column="will_purchase",
            entity_column=entity_column,
            time_column=time_column,
            labeling_function=purchase_probability_labeling,
            window_size=window_size
        )

    @staticmethod
    def customer_ltv(
        entity_column: str = "customer_id",
        time_column: str = "timestamp",
        amount_column: str = "amount",
        window_size: str = "90 days"
    ) -> PredictionProblem:
        """Create a customer lifetime value prediction problem"""

        def ltv_labeling(df):
            return total_spend_labeling(df, amount_column)

        return PredictionProblem(
            name="customer_ltv",
            description="Predict customer lifetime value",
            target_column="ltv",
            entity_column=entity_column,
            time_column=time_column,
            labeling_function=ltv_labeling,
            window_size=window_size
        )

    @staticmethod
    def event_frequency(
        entity_column: str = "entity_id",
        time_column: str = "timestamp",
        window_size: str = "7 days"
    ) -> PredictionProblem:
        """Create an event frequency prediction problem"""
        return PredictionProblem(
            name="event_frequency",
            description="Predict number of events in window",
            target_column="event_count",
            entity_column=entity_column,
            time_column=time_column,
            labeling_function=count_events_labeling,
            window_size=window_size
        )
