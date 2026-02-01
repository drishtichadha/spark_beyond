"""
Enhanced preprocessing module with feature-engine inspired transformations

This module implements PySpark-native versions of common feature-engine patterns:
- Missing value imputation (mean, median, mode, arbitrary)
- Outlier detection and treatment (IQR, Winsorization)
- Feature scaling (Standard, MinMax, Robust)
- Rare category handling
- Categorical encoding enhancements
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DoubleType
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import (
    StandardScaler, MinMaxScaler, RobustScaler,
    Imputer, VectorAssembler, StringIndexer
)
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from typing import List, Dict, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ImputationStrategy(str, Enum):
    """Imputation strategies for missing values"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    DROP = "drop"


class OutlierStrategy(str, Enum):
    """Strategies for handling outliers"""
    IQR_CAP = "iqr_cap"           # Cap at IQR bounds (Winsorization)
    IQR_REMOVE = "iqr_remove"     # Remove outlier rows
    ZSCORE_CAP = "zscore_cap"     # Cap at z-score bounds
    ZSCORE_REMOVE = "zscore_remove"  # Remove outlier rows


class ScalingStrategy(str, Enum):
    """Feature scaling strategies"""
    STANDARD = "standard"         # Zero mean, unit variance
    MINMAX = "minmax"            # Scale to [0, 1]
    ROBUST = "robust"            # Median and IQR based
    NONE = "none"                # No scaling


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing pipeline

    Attributes:
        imputation_strategy: Strategy for handling missing values
        imputation_value: Value for constant imputation
        outlier_strategy: Strategy for handling outliers
        outlier_threshold: Threshold for outlier detection (IQR multiplier or z-score)
        scaling_strategy: Strategy for feature scaling
        rare_category_threshold: Minimum frequency for rare category grouping
        rare_category_replacement: Replacement value for rare categories
    """
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    imputation_value: Optional[float] = None
    outlier_strategy: Optional[OutlierStrategy] = None
    outlier_threshold: float = 1.5
    scaling_strategy: ScalingStrategy = ScalingStrategy.NONE
    rare_category_threshold: float = 0.01  # 1% threshold
    rare_category_replacement: str = "RARE"


class EnhancedPreprocessor:
    """
    Enhanced preprocessing pipeline with feature-engine inspired transformations

    Example:
        config = PreprocessingConfig(
            imputation_strategy=ImputationStrategy.MEDIAN,
            outlier_strategy=OutlierStrategy.IQR_CAP,
            scaling_strategy=ScalingStrategy.ROBUST
        )
        preprocessor = EnhancedPreprocessor(df, config)
        processed_df = preprocessor.fit_transform(
            numerical_cols=['age', 'balance'],
            categorical_cols=['job', 'education']
        )
    """

    def __init__(self, df: DataFrame, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor

        Args:
            df: PySpark DataFrame to preprocess
            config: Preprocessing configuration
        """
        self.df = df
        self.config = config or PreprocessingConfig()
        self._fitted_params: Dict[str, Dict] = {}
        self._transformations_applied: List[str] = []

    def impute_missing_values(
        self,
        columns: List[str],
        strategy: Optional[ImputationStrategy] = None,
        constant_value: Optional[float] = None
    ) -> DataFrame:
        """
        Impute missing values in specified columns

        Args:
            columns: Columns to impute
            strategy: Imputation strategy (overrides config)
            constant_value: Value for constant imputation

        Returns:
            DataFrame with imputed values
        """
        strategy = strategy or self.config.imputation_strategy
        logger.info(f"Imputing missing values with strategy: {strategy}")

        if strategy == ImputationStrategy.DROP:
            result_df = self.df.dropna(subset=columns)
            self._transformations_applied.append(f"dropped_missing_{len(columns)}_cols")
            return result_df

        result_df = self.df
        imputation_params = {}

        for col in columns:
            col_type = dict(self.df.dtypes).get(col)

            # Skip non-existent columns
            if col not in self.df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue

            # Calculate imputation value based on strategy
            if strategy == ImputationStrategy.MEAN:
                impute_value = result_df.select(F.mean(col)).collect()[0][0]
            elif strategy == ImputationStrategy.MEDIAN:
                impute_value = result_df.select(
                    F.expr(f'percentile_approx({col}, 0.5)')
                ).collect()[0][0]
            elif strategy == ImputationStrategy.MODE:
                mode_result = result_df.groupBy(col).count() \
                    .orderBy(F.desc('count')) \
                    .first()
                impute_value = mode_result[col] if mode_result else None
            elif strategy == ImputationStrategy.CONSTANT:
                impute_value = constant_value or self.config.imputation_value or 0
            else:
                impute_value = None

            if impute_value is not None:
                result_df = result_df.withColumn(
                    col,
                    F.when(F.col(col).isNull(), F.lit(impute_value))
                    .otherwise(F.col(col))
                )
                imputation_params[col] = impute_value
                logger.debug(f"Imputed {col} with {impute_value}")

        self._fitted_params['imputation'] = imputation_params
        self._transformations_applied.append(f"imputed_{len(columns)}_cols")

        return result_df

    def handle_outliers(
        self,
        columns: List[str],
        strategy: Optional[OutlierStrategy] = None,
        threshold: Optional[float] = None
    ) -> DataFrame:
        """
        Handle outliers in specified numeric columns

        Args:
            columns: Numeric columns to process
            strategy: Outlier handling strategy (overrides config)
            threshold: Detection threshold (overrides config)

        Returns:
            DataFrame with outliers handled
        """
        strategy = strategy or self.config.outlier_strategy
        threshold = threshold or self.config.outlier_threshold

        if strategy is None:
            return self.df

        logger.info(f"Handling outliers with strategy: {strategy}, threshold: {threshold}")

        result_df = self.df
        outlier_params = {}

        for col in columns:
            # Skip non-numeric columns
            col_type = dict(self.df.dtypes).get(col)
            if col_type not in ['int', 'bigint', 'float', 'double', 'decimal']:
                continue

            if strategy in [OutlierStrategy.IQR_CAP, OutlierStrategy.IQR_REMOVE]:
                # Calculate IQR bounds
                quantiles = result_df.select(
                    F.expr(f'percentile_approx({col}, 0.25)').alias('q1'),
                    F.expr(f'percentile_approx({col}, 0.75)').alias('q3')
                ).collect()[0]

                q1, q3 = quantiles['q1'], quantiles['q3']
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                if strategy == OutlierStrategy.IQR_CAP:
                    # Winsorization - cap at bounds
                    result_df = result_df.withColumn(
                        col,
                        F.when(F.col(col) < lower_bound, lower_bound)
                        .when(F.col(col) > upper_bound, upper_bound)
                        .otherwise(F.col(col))
                    )
                else:  # IQR_REMOVE
                    result_df = result_df.filter(
                        (F.col(col) >= lower_bound) & (F.col(col) <= upper_bound)
                    )

                outlier_params[col] = {
                    'method': 'iqr',
                    'q1': q1, 'q3': q3,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

            elif strategy in [OutlierStrategy.ZSCORE_CAP, OutlierStrategy.ZSCORE_REMOVE]:
                # Calculate z-score bounds
                stats = result_df.select(
                    F.mean(col).alias('mean'),
                    F.stddev(col).alias('std')
                ).collect()[0]

                mean, std = stats['mean'], stats['std']
                if std and std > 0:
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std

                    if strategy == OutlierStrategy.ZSCORE_CAP:
                        result_df = result_df.withColumn(
                            col,
                            F.when(F.col(col) < lower_bound, lower_bound)
                            .when(F.col(col) > upper_bound, upper_bound)
                            .otherwise(F.col(col))
                        )
                    else:  # ZSCORE_REMOVE
                        result_df = result_df.filter(
                            (F.col(col) >= lower_bound) & (F.col(col) <= upper_bound)
                        )

                    outlier_params[col] = {
                        'method': 'zscore',
                        'mean': mean, 'std': std,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }

        self._fitted_params['outliers'] = outlier_params
        self._transformations_applied.append(f"outliers_{strategy.value}_{len(columns)}_cols")

        return result_df

    def group_rare_categories(
        self,
        columns: List[str],
        threshold: Optional[float] = None,
        replacement: Optional[str] = None
    ) -> DataFrame:
        """
        Group rare categories into a single category

        Args:
            columns: Categorical columns to process
            threshold: Minimum frequency threshold (0-1)
            replacement: Replacement value for rare categories

        Returns:
            DataFrame with rare categories grouped
        """
        threshold = threshold or self.config.rare_category_threshold
        replacement = replacement or self.config.rare_category_replacement

        logger.info(f"Grouping rare categories with threshold: {threshold}")

        result_df = self.df
        rare_params = {}

        for col in columns:
            # Calculate category frequencies
            total_count = result_df.count()
            category_counts = result_df.groupBy(col).count() \
                .withColumn('frequency', F.col('count') / total_count)

            # Find rare categories
            rare_categories = category_counts.filter(
                F.col('frequency') < threshold
            ).select(col).collect()

            rare_values = [row[col] for row in rare_categories]

            if rare_values:
                result_df = result_df.withColumn(
                    col,
                    F.when(F.col(col).isin(rare_values), replacement)
                    .otherwise(F.col(col))
                )
                rare_params[col] = {
                    'rare_categories': rare_values,
                    'replacement': replacement,
                    'count': len(rare_values)
                }
                logger.debug(f"Grouped {len(rare_values)} rare categories in {col}")

        self._fitted_params['rare_categories'] = rare_params
        self._transformations_applied.append(f"grouped_rare_{len(columns)}_cols")

        return result_df

    def scale_features(
        self,
        columns: List[str],
        strategy: Optional[ScalingStrategy] = None,
        output_suffix: str = "_scaled"
    ) -> DataFrame:
        """
        Scale numeric features

        Args:
            columns: Numeric columns to scale
            strategy: Scaling strategy (overrides config)
            output_suffix: Suffix for scaled column names

        Returns:
            DataFrame with scaled features
        """
        strategy = strategy or self.config.scaling_strategy

        if strategy == ScalingStrategy.NONE:
            return self.df

        logger.info(f"Scaling features with strategy: {strategy}")

        result_df = self.df
        scaling_params = {}

        for col in columns:
            output_col = f"{col}{output_suffix}"

            if strategy == ScalingStrategy.STANDARD:
                # Standard scaling (z-score normalization)
                stats = result_df.select(
                    F.mean(col).alias('mean'),
                    F.stddev(col).alias('std')
                ).collect()[0]

                mean, std = stats['mean'], stats['std']
                if std and std > 0:
                    result_df = result_df.withColumn(
                        output_col,
                        (F.col(col) - mean) / std
                    )
                    scaling_params[col] = {'mean': mean, 'std': std}

            elif strategy == ScalingStrategy.MINMAX:
                # Min-Max scaling to [0, 1]
                stats = result_df.select(
                    F.min(col).alias('min'),
                    F.max(col).alias('max')
                ).collect()[0]

                min_val, max_val = stats['min'], stats['max']
                range_val = max_val - min_val
                if range_val > 0:
                    result_df = result_df.withColumn(
                        output_col,
                        (F.col(col) - min_val) / range_val
                    )
                    scaling_params[col] = {'min': min_val, 'max': max_val}

            elif strategy == ScalingStrategy.ROBUST:
                # Robust scaling using median and IQR
                stats = result_df.select(
                    F.expr(f'percentile_approx({col}, 0.5)').alias('median'),
                    F.expr(f'percentile_approx({col}, 0.25)').alias('q1'),
                    F.expr(f'percentile_approx({col}, 0.75)').alias('q3')
                ).collect()[0]

                median = stats['median']
                iqr = stats['q3'] - stats['q1']
                if iqr > 0:
                    result_df = result_df.withColumn(
                        output_col,
                        (F.col(col) - median) / iqr
                    )
                    scaling_params[col] = {
                        'median': median,
                        'iqr': iqr,
                        'q1': stats['q1'],
                        'q3': stats['q3']
                    }

        self._fitted_params['scaling'] = scaling_params
        self._transformations_applied.append(f"scaled_{strategy.value}_{len(columns)}_cols")

        return result_df

    def encode_cyclical_features(
        self,
        column: str,
        period: int,
        output_prefix: Optional[str] = None
    ) -> DataFrame:
        """
        Encode cyclical features using sin/cos transformation

        Useful for time features like hour (24), day of week (7), month (12)

        Args:
            column: Column containing cyclical values
            period: Period of the cycle (e.g., 24 for hours, 7 for days)
            output_prefix: Prefix for output columns

        Returns:
            DataFrame with sin/cos encoded features
        """
        import math
        prefix = output_prefix or column

        logger.info(f"Encoding cyclical feature {column} with period {period}")

        result_df = self.df.withColumn(
            f"{prefix}_sin",
            F.sin(2 * math.pi * F.col(column) / period)
        ).withColumn(
            f"{prefix}_cos",
            F.cos(2 * math.pi * F.col(column) / period)
        )

        self._transformations_applied.append(f"cyclical_{column}")
        return result_df

    def create_polynomial_features(
        self,
        columns: List[str],
        degree: int = 2,
        interaction_only: bool = False
    ) -> DataFrame:
        """
        Create polynomial and interaction features

        Args:
            columns: Numeric columns to use
            degree: Maximum polynomial degree
            interaction_only: If True, only create interaction terms

        Returns:
            DataFrame with polynomial features
        """
        logger.info(f"Creating polynomial features (degree={degree})")

        result_df = self.df
        new_features = []

        # Polynomial terms for each column
        if not interaction_only:
            for col in columns:
                for d in range(2, degree + 1):
                    new_col = f"{col}_pow{d}"
                    result_df = result_df.withColumn(
                        new_col,
                        F.pow(F.col(col), d)
                    )
                    new_features.append(new_col)

        # Interaction terms between columns
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                new_col = f"{col1}_x_{col2}"
                result_df = result_df.withColumn(
                    new_col,
                    F.col(col1) * F.col(col2)
                )
                new_features.append(new_col)

        self._transformations_applied.append(f"polynomial_{len(new_features)}_features")
        return result_df

    def fit_transform(
        self,
        numerical_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        apply_imputation: bool = True,
        apply_outlier_handling: bool = True,
        apply_scaling: bool = False,
        apply_rare_grouping: bool = True
    ) -> DataFrame:
        """
        Apply full preprocessing pipeline

        Args:
            numerical_cols: List of numerical columns
            categorical_cols: List of categorical columns
            apply_imputation: Whether to impute missing values
            apply_outlier_handling: Whether to handle outliers
            apply_scaling: Whether to scale features
            apply_rare_grouping: Whether to group rare categories

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Running full preprocessing pipeline...")

        result_df = self.df

        # Step 1: Impute missing values
        if apply_imputation:
            result_df = EnhancedPreprocessor(result_df, self.config) \
                .impute_missing_values(numerical_cols, strategy=ImputationStrategy.MEDIAN)
            self._transformations_applied.extend(["impute_numerical"])

            if categorical_cols:
                preprocessor = EnhancedPreprocessor(result_df, self.config)
                result_df = preprocessor.impute_missing_values(
                    categorical_cols,
                    strategy=ImputationStrategy.MODE
                )
                self._transformations_applied.extend(["impute_categorical"])

        # Step 2: Handle outliers
        if apply_outlier_handling and self.config.outlier_strategy:
            preprocessor = EnhancedPreprocessor(result_df, self.config)
            result_df = preprocessor.handle_outliers(numerical_cols)
            self._fitted_params.update(preprocessor._fitted_params)

        # Step 3: Group rare categories
        if apply_rare_grouping and categorical_cols:
            preprocessor = EnhancedPreprocessor(result_df, self.config)
            result_df = preprocessor.group_rare_categories(categorical_cols)
            self._fitted_params.update(preprocessor._fitted_params)

        # Step 4: Scale features
        if apply_scaling:
            preprocessor = EnhancedPreprocessor(result_df, self.config)
            result_df = preprocessor.scale_features(numerical_cols)
            self._fitted_params.update(preprocessor._fitted_params)

        logger.info(f"Preprocessing complete. Applied: {self._transformations_applied}")
        return result_df

    def get_fitted_params(self) -> Dict[str, Dict]:
        """Get fitted parameters from all transformations"""
        return self._fitted_params

    def get_transformations_applied(self) -> List[str]:
        """Get list of applied transformations"""
        return self._transformations_applied


def create_preprocessing_pipeline(
    df: DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    config: Optional[PreprocessingConfig] = None
) -> tuple[DataFrame, Dict[str, Dict]]:
    """
    Convenience function to run preprocessing with default config

    Args:
        df: Input DataFrame
        numerical_cols: Numerical column names
        categorical_cols: Categorical column names
        config: Optional preprocessing config

    Returns:
        Tuple of (processed DataFrame, fitted parameters)
    """
    preprocessor = EnhancedPreprocessor(df, config)
    processed_df = preprocessor.fit_transform(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols
    )
    return processed_df, preprocessor.get_fitted_params()
