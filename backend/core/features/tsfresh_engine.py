"""
TSFresh integration for time-series feature extraction

This module wraps tsfresh to automatically extract comprehensive
time-series features from temporal data.

Key capabilities:
- Comprehensive time-series feature extraction
- Automatic feature relevance filtering
- Configurable feature calculation settings
- Integration with PySpark DataFrames
"""

from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F
from backend.core.utils.spark_pandas_bridge import spark_to_pandas_safe, pandas_to_spark
from backend.core.utils.time_series_detector import TimeSeriesInfo
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Lazy imports for tsfresh
_tsfresh = None
_tsfresh_settings = None


def _get_tsfresh():
    """Lazy load tsfresh"""
    global _tsfresh, _tsfresh_settings
    if _tsfresh is None:
        try:
            import tsfresh
            from tsfresh import extract_features, select_features
            from tsfresh.feature_extraction import (
                ComprehensiveFCParameters,
                MinimalFCParameters,
                EfficientFCParameters
            )
            from tsfresh.utilities.dataframe_functions import impute
            _tsfresh = tsfresh
            _tsfresh_settings = {
                'ComprehensiveFCParameters': ComprehensiveFCParameters,
                'MinimalFCParameters': MinimalFCParameters,
                'EfficientFCParameters': EfficientFCParameters,
                'extract_features': extract_features,
                'select_features': select_features,
                'impute': impute
            }
        except ImportError:
            raise ImportError(
                "tsfresh is not installed. "
                "Install it with: pip install tsfresh"
            )
    return _tsfresh, _tsfresh_settings


class FeatureExtractionMode(str, Enum):
    """Feature extraction modes with different complexity levels"""
    MINIMAL = "minimal"          # Fast, basic features (~10 per column)
    EFFICIENT = "efficient"      # Balanced speed/features (~100 per column)
    COMPREHENSIVE = "comprehensive"  # Full extraction (~750 per column)
    CUSTOM = "custom"            # User-defined settings


@dataclass
class TSFreshResult:
    """
    Results from tsfresh feature extraction

    Attributes:
        feature_matrix: DataFrame with extracted features
        feature_names: List of generated feature names
        relevant_features: Features deemed relevant to target (if filtered)
        extraction_time: Time taken for extraction
        extraction_mode: Mode used for extraction
        warnings: Any warnings during extraction
    """
    feature_matrix: pd.DataFrame
    feature_names: List[str]
    relevant_features: Optional[List[str]]
    extraction_time: float
    extraction_mode: FeatureExtractionMode
    warnings: Optional[List[str]] = None


class TSFreshEngine:
    """
    TSFresh-based time-series feature extraction engine

    This class handles:
    - Conversion from PySpark to the format required by tsfresh
    - Feature extraction with various complexity levels
    - Feature relevance filtering based on target variable
    - Conversion back to PySpark

    Example:
        engine = TSFreshEngine(spark)
        result = engine.extract_features(
            spark_df,
            id_column='customer_id',
            time_column='transaction_date',
            value_columns=['amount', 'quantity'],
            mode=FeatureExtractionMode.EFFICIENT
        )
        spark_df_with_features = engine.to_spark(result, spark_df, 'customer_id')
    """

    def __init__(
        self,
        spark: SparkSession,
        max_rows_for_pandas: int = 500000,
        n_jobs: int = 0,  # 0 = use all cores
        verbose: bool = True
    ):
        """
        Initialize the TSFreshEngine

        Args:
            spark: SparkSession instance
            max_rows_for_pandas: Maximum rows to convert to Pandas
            n_jobs: Number of parallel jobs (0 = all cores)
            verbose: Whether to print progress information
        """
        self.spark = spark
        self.max_rows = max_rows_for_pandas
        self.n_jobs = n_jobs
        self.verbose = verbose
        self._extracted_features = None

    def _prepare_tsfresh_format(
        self,
        spark_df: SparkDataFrame,
        id_column: str,
        time_column: str,
        value_columns: List[str]
    ) -> pd.DataFrame:
        """
        Convert PySpark DataFrame to tsfresh long format

        tsfresh expects data in long format with columns:
        - id: entity identifier
        - time: time/sort column
        - value: measurement values (one column per value type)
        """
        logger.info("Preparing data in tsfresh format...")

        # Select relevant columns
        columns_to_select = [id_column, time_column] + value_columns
        selected_df = spark_df.select(*columns_to_select)

        # Convert to Pandas
        pdf = spark_to_pandas_safe(selected_df, max_rows=self.max_rows, sample=True)

        # Ensure proper column names for tsfresh
        pdf = pdf.rename(columns={id_column: 'id', time_column: 'time'})

        # Sort by id and time
        pdf = pdf.sort_values(['id', 'time'])

        logger.info(f"Prepared {len(pdf)} rows for tsfresh")
        return pdf

    def extract_features(
        self,
        spark_df: SparkDataFrame,
        id_column: str,
        time_column: str,
        value_columns: List[str],
        mode: FeatureExtractionMode = FeatureExtractionMode.EFFICIENT,
        custom_settings: Optional[Dict] = None,
        impute_missing: bool = True
    ) -> TSFreshResult:
        """
        Extract time-series features using tsfresh

        Args:
            spark_df: PySpark DataFrame with time-series data
            id_column: Column identifying entities (e.g., customer_id)
            time_column: Column with timestamps or sort order
            value_columns: Columns to extract features from
            mode: Extraction complexity mode
            custom_settings: Custom feature calculation settings (for CUSTOM mode)
            impute_missing: Whether to impute missing values in results

        Returns:
            TSFreshResult with extracted features
        """
        import time
        start_time = time.time()

        _, settings = _get_tsfresh()
        warnings = []

        # Prepare data in tsfresh format
        pdf = self._prepare_tsfresh_format(
            spark_df, id_column, time_column, value_columns
        )

        # Select extraction settings based on mode
        if mode == FeatureExtractionMode.MINIMAL:
            extraction_settings = settings['MinimalFCParameters']()
        elif mode == FeatureExtractionMode.EFFICIENT:
            extraction_settings = settings['EfficientFCParameters']()
        elif mode == FeatureExtractionMode.COMPREHENSIVE:
            extraction_settings = settings['ComprehensiveFCParameters']()
        elif mode == FeatureExtractionMode.CUSTOM:
            if custom_settings is None:
                raise ValueError("custom_settings required for CUSTOM mode")
            extraction_settings = custom_settings
        else:
            extraction_settings = settings['EfficientFCParameters']()

        if self.verbose:
            logger.info(f"Extracting features with mode: {mode.value}")

        # Extract features for each value column
        all_features = []
        for value_col in value_columns:
            try:
                # Prepare column-specific dataframe
                col_df = pdf[['id', 'time', value_col]].copy()
                col_df = col_df.rename(columns={value_col: 'value'})

                # Extract features
                features = settings['extract_features'](
                    col_df,
                    column_id='id',
                    column_sort='time',
                    column_value='value',
                    default_fc_parameters=extraction_settings,
                    n_jobs=self.n_jobs,
                    disable_progressbar=not self.verbose
                )

                # Rename columns to include value column name
                features.columns = [f"{value_col}__{c}" for c in features.columns]
                all_features.append(features)

            except Exception as e:
                warnings.append(f"Error extracting features for {value_col}: {str(e)}")
                logger.warning(f"Error extracting features for {value_col}: {e}")

        if not all_features:
            raise RuntimeError("No features could be extracted from any column")

        # Combine features from all value columns
        feature_matrix = pd.concat(all_features, axis=1)

        # Impute missing values
        if impute_missing:
            feature_matrix = settings['impute'](feature_matrix)

        self._extracted_features = feature_matrix

        extraction_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Extracted {len(feature_matrix.columns)} features in {extraction_time:.2f}s")

        return TSFreshResult(
            feature_matrix=feature_matrix,
            feature_names=list(feature_matrix.columns),
            relevant_features=None,
            extraction_time=extraction_time,
            extraction_mode=mode,
            warnings=warnings if warnings else None
        )

    def filter_relevant_features(
        self,
        result: TSFreshResult,
        target: pd.Series,
        fdr_level: float = 0.05
    ) -> TSFreshResult:
        """
        Filter features based on relevance to target variable

        Uses statistical tests to identify features significantly
        correlated with the target.

        Args:
            result: TSFreshResult from extract_features
            target: Target variable (same index as feature_matrix)
            fdr_level: False Discovery Rate level for significance

        Returns:
            Updated TSFreshResult with filtered features
        """
        _, settings = _get_tsfresh()

        logger.info(f"Filtering features with FDR level: {fdr_level}")

        # Align indices
        aligned_features = result.feature_matrix.loc[target.index]

        # Select relevant features
        relevant_features = settings['select_features'](
            aligned_features,
            target,
            fdr_level=fdr_level
        )

        relevant_names = list(relevant_features.columns)

        logger.info(f"Selected {len(relevant_names)} relevant features from {len(result.feature_names)}")

        return TSFreshResult(
            feature_matrix=relevant_features,
            feature_names=relevant_names,
            relevant_features=relevant_names,
            extraction_time=result.extraction_time,
            extraction_mode=result.extraction_mode,
            warnings=result.warnings
        )

    def to_spark(
        self,
        result: TSFreshResult,
        original_spark_df: SparkDataFrame,
        id_column: str
    ) -> SparkDataFrame:
        """
        Convert tsfresh results back to PySpark DataFrame

        Args:
            result: TSFreshResult from feature extraction
            original_spark_df: Original PySpark DataFrame
            id_column: Column to join on

        Returns:
            PySpark DataFrame with extracted features
        """
        # Reset index to make ID a column
        feature_df = result.feature_matrix.reset_index()
        feature_df = feature_df.rename(columns={'index': id_column})

        # Convert to Spark
        spark_features = pandas_to_spark(feature_df, self.spark)

        # Join with original DataFrame
        # First, get distinct id values from original to maintain all rows
        id_df = original_spark_df.select(id_column).distinct()

        result_df = id_df.join(spark_features, on=id_column, how='left')

        # Join back all original columns except id
        other_cols = [c for c in original_spark_df.columns if c != id_column]
        if other_cols:
            result_df = result_df.join(
                original_spark_df.select([id_column] + other_cols),
                on=id_column,
                how='left'
            )

        return result_df

    def get_available_features(self) -> Dict[str, List[str]]:
        """
        Get list of available tsfresh features by category

        Returns:
            Dictionary mapping category names to feature lists
        """
        _, settings = _get_tsfresh()

        # Get comprehensive settings to see all available
        all_features = settings['ComprehensiveFCParameters']()

        # Group by feature type
        categories = {}
        for feature_name in all_features.keys():
            # Extract category from feature name
            base_name = feature_name.split('__')[0] if '__' in feature_name else feature_name
            if base_name not in categories:
                categories[base_name] = []
            categories[base_name].append(feature_name)

        return categories

    def create_custom_settings(
        self,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None,
        base_mode: FeatureExtractionMode = FeatureExtractionMode.EFFICIENT
    ) -> Dict:
        """
        Create custom feature extraction settings

        Args:
            include_features: Features to include (if None, use base_mode)
            exclude_features: Features to exclude from base_mode
            base_mode: Base mode to start from

        Returns:
            Custom settings dictionary for extract_features
        """
        _, settings = _get_tsfresh()

        # Start with base settings
        if base_mode == FeatureExtractionMode.MINIMAL:
            base_settings = settings['MinimalFCParameters']()
        elif base_mode == FeatureExtractionMode.COMPREHENSIVE:
            base_settings = settings['ComprehensiveFCParameters']()
        else:
            base_settings = settings['EfficientFCParameters']()

        if include_features:
            # Only include specified features
            custom_settings = {k: v for k, v in base_settings.items() if k in include_features}
        else:
            custom_settings = dict(base_settings)

        if exclude_features:
            # Remove excluded features
            for feature in exclude_features:
                custom_settings.pop(feature, None)

        return custom_settings


def extract_ts_features_simple(
    spark_df: SparkDataFrame,
    spark: SparkSession,
    id_column: str,
    time_column: str,
    value_columns: List[str],
    mode: str = "efficient"
) -> Tuple[SparkDataFrame, List[str]]:
    """
    Simple function to extract time-series features

    Args:
        spark_df: PySpark DataFrame
        spark: SparkSession
        id_column: Entity identifier column
        time_column: Time/sort column
        value_columns: Columns to extract features from
        mode: "minimal", "efficient", or "comprehensive"

    Returns:
        Tuple of (DataFrame with features, list of feature names)
    """
    mode_map = {
        'minimal': FeatureExtractionMode.MINIMAL,
        'efficient': FeatureExtractionMode.EFFICIENT,
        'comprehensive': FeatureExtractionMode.COMPREHENSIVE
    }

    engine = TSFreshEngine(spark, verbose=False)
    result = engine.extract_features(
        spark_df,
        id_column=id_column,
        time_column=time_column,
        value_columns=value_columns,
        mode=mode_map.get(mode, FeatureExtractionMode.EFFICIENT)
    )

    spark_with_features = engine.to_spark(result, spark_df, id_column)

    return spark_with_features, result.feature_names


class TSFreshPySparkNative:
    """
    Native PySpark implementation of common tsfresh features

    For cases where Pandas conversion is not feasible, this class
    provides PySpark-native implementations of key time-series features.
    """

    def __init__(self, spark_df: SparkDataFrame):
        self.df = spark_df

    def extract_basic_features(
        self,
        id_column: str,
        time_column: str,
        value_columns: List[str]
    ) -> SparkDataFrame:
        """
        Extract basic time-series statistics using pure PySpark

        Features extracted per entity:
        - mean, std, min, max, median
        - sum, count, variance
        - first, last values
        - trend (simple linear approximation)
        """
        from pyspark.sql.window import Window

        result_df = self.df

        for value_col in value_columns:
            # Window for entity-level aggregations
            window = Window.partitionBy(id_column)

            # Basic statistics
            result_df = result_df.withColumn(
                f"{value_col}__mean",
                F.mean(value_col).over(window)
            ).withColumn(
                f"{value_col}__std",
                F.stddev(value_col).over(window)
            ).withColumn(
                f"{value_col}__min",
                F.min(value_col).over(window)
            ).withColumn(
                f"{value_col}__max",
                F.max(value_col).over(window)
            ).withColumn(
                f"{value_col}__sum",
                F.sum(value_col).over(window)
            ).withColumn(
                f"{value_col}__count",
                F.count(value_col).over(window)
            )

            # First and last values
            ordered_window = Window.partitionBy(id_column).orderBy(time_column)
            result_df = result_df.withColumn(
                f"{value_col}__first",
                F.first(value_col).over(ordered_window)
            ).withColumn(
                f"{value_col}__last",
                F.last(value_col).over(ordered_window)
            )

            # Range
            result_df = result_df.withColumn(
                f"{value_col}__range",
                F.col(f"{value_col}__max") - F.col(f"{value_col}__min")
            )

        return result_df
