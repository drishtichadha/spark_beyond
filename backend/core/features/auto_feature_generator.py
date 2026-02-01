"""
Automatic Feature Generation for PySpark
This module provides automated feature engineering capabilities for PySpark DataFrames

Enhanced with:
- Polynomial and ratio features
- Cyclical encoding for periodic features
- Target encoding for high-cardinality categoricals
- Feature importance-based selection
- Integration hooks for featuretools and tsfresh
"""

from pyspark.sql import DataFrame, SparkSession
from backend.core.discovery import SchemaChecks, Problem
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType, StringType, DateType, TimestampType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class AutoFeatureGenerator:
    """
    Automated feature generation for PySpark DataFrames
    
    Features:
    - Numerical transformations (log, sqrt, polynomial)
    - Interaction features
    - Aggregation features (group-by statistics)
    - Binning/discretization
    - Date/time features
    - String features (length, word count, etc.)
    - Encoding (one-hot, label encoding)
    """
    
    def __init__(self, schema_checks: SchemaChecks, problem: Problem):
        
        self.schema_checks = schema_checks
        self.problem = problem
        self.dataframe = schema_checks.dataframe
        self.generated_features = []
        
    def generate_numerical_features(
        self,
        columns: Optional[List[str]] = None,
        transformations: List[str] = ['log', 'sqrt', 'square', 'cube']
    ) -> DataFrame:
        """
        Generate numerical transformations for specified columns
        
        Args:
            columns: List of columns to transform (None = all numeric columns)
            transformations: List of transformations to apply
        
        Returns:
            DataFrame with additional transformed features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, NumericType)]
        
        
        for col in columns:
            if 'log' in transformations:
                # Log transformation (handling zeros and negatives)
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_log",
                    F.log1p(F.when(F.col(col) > 0, F.col(col)).otherwise(0))
                )
                self.generated_features.append(f"{col}_log")
            
            if 'sqrt' in transformations:
                # Square root (handling negatives)
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_sqrt",
                    F.sqrt(F.when(F.col(col) >= 0, F.col(col)).otherwise(0))
                )
                self.generated_features.append(f"{col}_sqrt")
            
            if 'square' in transformations:
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_square",
                    F.pow(F.col(col), 2)
                )
                self.generated_features.append(f"{col}_square")
            
            if 'cube' in transformations:
                self.dataframe = self.dataframe.withColumn(
                    f"{col}_cube",
                    F.pow(F.col(col), 3)
                )
                self.generated_features.append(f"{col}_cube")
        
        return self.dataframe
    
    def generate_interaction_features(
        self,
        columns: Optional[List[str]] = None,
        max_interactions: int = 2
    ) -> DataFrame:
        """
        Generate interaction features (multiplication, division, addition)
        
        Args:
            columns: List of columns to use for interactions
            max_interactions: Maximum number of columns to interact
        
        Returns:
            DataFrame with interaction features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, NumericType)]
        
        
        # Pairwise interactions
        if max_interactions >= 2:
            for i, col1 in enumerate(columns):
                for col2 in columns[i+1:]:
                    # Multiplication
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_mult_{col2}",
                        F.col(col1) * F.col(col2)
                    )
                    self.generated_features.append(f"{col1}_mult_{col2}")
                    
                    # Division (handling division by zero)
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_div_{col2}",
                        F.when(F.col(col2) != 0, F.col(col1) / F.col(col2))
                        .otherwise(0)
                    )
                    self.generated_features.append(f"{col1}_div_{col2}")
                    
                    # Addition
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_add_{col2}",
                        F.col(col1) + F.col(col2)
                    )
                    self.generated_features.append(f"{col1}_add_{col2}")
                    
                    # Subtraction
                    self.dataframe = self.dataframe.withColumn(
                        f"{col1}_sub_{col2}",
                        F.col(col1) - F.col(col2)
                    )
                    self.generated_features.append(f"{col1}_sub_{col2}")
        
        return self.dataframe
    
    def generate_aggregation_features(
        self,
        group_by_cols: List[str],
        agg_cols: List[str],
        agg_functions: List[str] = ['mean', 'sum', 'min', 'max', 'stddev', 'count']
    ) -> DataFrame:
        """
        Generate aggregation features based on grouping
        
        Args:
            group_by_cols: Columns to group by
            agg_cols: Columns to aggregate
            agg_functions: Aggregation functions to apply
        
        Returns:
            DataFrame with aggregation features joined back
        """
        # Create aggregations
        agg_exprs = []
        feature_names = []
        
        for agg_col in agg_cols:
            for func in agg_functions:
                if func == 'mean':
                    agg_exprs.append(F.mean(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'sum':
                    agg_exprs.append(F.sum(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'min':
                    agg_exprs.append(F.min(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'max':
                    agg_exprs.append(F.max(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'stddev':
                    agg_exprs.append(F.stddev(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                elif func == 'count':
                    agg_exprs.append(F.count(agg_col).alias(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}"))
                
                feature_names.append(f"{agg_col}_{func}_by_{'_'.join(group_by_cols)}")
        
        # Perform aggregation
        agg_dataframe = self.dataframe.groupBy(group_by_cols).agg(*agg_exprs)
        
        # Join back to original DataFrame
        self.dataframe = self.dataframe.join(agg_dataframe, on=group_by_cols, how='left')
        
        self.generated_features.extend(feature_names)
        return self.dataframe
    
    def generate_binning_features(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 5
    ) -> DataFrame:
        """
        Generate binned/discretized features
        
        Args:
            columns: Columns to bin
            n_bins: Number of bins
        
        Returns:
            DataFrame with binned features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, NumericType)]
        
        
        for col in columns:
            # Calculate quantiles for binning
            quantiles = self.dataframe.approxQuantile(col, 
                [i/n_bins for i in range(n_bins + 1)], 0.01)
            
            # Ensure unique splits
            quantiles = sorted(list(set(quantiles)))
            
            if len(quantiles) > 2:
                # Create bucketizer
                bucketizer = Bucketizer(
                    splits=quantiles,
                    inputCol=col,
                    outputCol=f"{col}_binned"
                )
                
                self.dataframe = bucketizer.transform(self.dataframe)
                self.generated_features.append(f"{col}_binned")
        
        return self.dataframe
    
    def generate_datetime_features(
        self,
        columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Generate date/time features
        
        Args:
            columns: Date/timestamp columns
        
        Returns:
            DataFrame with datetime features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, (DateType, TimestampType))]
        
        
        for col in columns:
            # Extract components
            self.dataframe = self.dataframe.withColumn(f"{col}_year", F.year(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_month", F.month(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_day", F.dayofmonth(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_dayofweek", F.dayofweek(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_dayofyear", F.dayofyear(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_weekofyear", F.weekofyear(col))
            self.dataframe = self.dataframe.withColumn(f"{col}_quarter", F.quarter(col))
            
            # Is weekend
            self.dataframe = self.dataframe.withColumn(
                f"{col}_is_weekend",
                F.when(F.dayofweek(col).isin([1, 7]), 1).otherwise(0)
            )
            
            self.generated_features.extend([
                f"{col}_year", f"{col}_month", f"{col}_day",
                f"{col}_dayofweek", f"{col}_dayofyear",
                f"{col}_weekofyear", f"{col}_quarter", f"{col}_is_weekend"
            ])
        
        return self.dataframe
    
    def generate_string_features(
        self,
        columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Generate features from string columns
        
        Args:
            columns: String columns
        
        Returns:
            DataFrame with string-based features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields 
                      if isinstance(f.dataType, StringType)]
        
        
        for col in columns:
            # String length
            self.dataframe = self.dataframe.withColumn(
                f"{col}_length",
                F.length(F.col(col))
            )
            
            # Word count
            self.dataframe = self.dataframe.withColumn(
                f"{col}_word_count",
                F.size(F.split(F.col(col), " "))
            )
            
            # Number of unique characters
            self.dataframe = self.dataframe.withColumn(
                f"{col}_unique_chars",
                F.size(F.array_distinct(F.split(F.col(col), "")))
            )
            
            self.generated_features.extend([
                f"{col}_length", f"{col}_word_count", f"{col}_unique_chars"
            ])

        return self.dataframe

    def generate_polynomial_features(
        self,
        columns: Optional[List[str]] = None,
        degree: int = 3,
        include_bias: bool = False
    ) -> DataFrame:
        """
        Generate polynomial features up to specified degree

        Args:
            columns: Columns to transform (None = all numeric columns)
            degree: Maximum polynomial degree (2-5 recommended)
            include_bias: Whether to include bias term (constant 1)

        Returns:
            DataFrame with polynomial features
        """
        if columns is None:
            columns = [f.name for f in self.dataframe.schema.fields
                      if isinstance(f.dataType, NumericType)]

        logger.info(f"Generating polynomial features (degree={degree}) for {len(columns)} columns")

        if include_bias:
            self.dataframe = self.dataframe.withColumn("poly_bias", F.lit(1.0))
            self.generated_features.append("poly_bias")

        for col in columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_pow{d}"
                self.dataframe = self.dataframe.withColumn(
                    new_col,
                    F.pow(F.col(col), d)
                )
                self.generated_features.append(new_col)

        return self.dataframe

    def generate_ratio_features(
        self,
        column_pairs: Optional[List[Tuple[str, str]]] = None,
        add_inverse: bool = True
    ) -> DataFrame:
        """
        Generate ratio features between column pairs

        Useful for domain-specific ratios like:
        - balance / income (financial leverage)
        - contacts / duration (efficiency)

        Args:
            column_pairs: List of (numerator, denominator) tuples
                         If None, generates all pairwise ratios for numeric columns
            add_inverse: Also add inverse ratio (denominator/numerator)

        Returns:
            DataFrame with ratio features
        """
        if column_pairs is None:
            numeric_cols = [f.name for f in self.dataframe.schema.fields
                          if isinstance(f.dataType, NumericType)]
            column_pairs = [(c1, c2) for i, c1 in enumerate(numeric_cols)
                           for c2 in numeric_cols[i+1:]]

        logger.info(f"Generating {len(column_pairs)} ratio features")

        for col1, col2 in column_pairs:
            # Ratio col1/col2
            ratio_col = f"{col1}_ratio_{col2}"
            self.dataframe = self.dataframe.withColumn(
                ratio_col,
                F.when(F.col(col2) != 0, F.col(col1) / F.col(col2))
                .otherwise(F.lit(0.0))
            )
            self.generated_features.append(ratio_col)

            if add_inverse:
                # Inverse ratio col2/col1
                inv_ratio_col = f"{col2}_ratio_{col1}"
                self.dataframe = self.dataframe.withColumn(
                    inv_ratio_col,
                    F.when(F.col(col1) != 0, F.col(col2) / F.col(col1))
                    .otherwise(F.lit(0.0))
                )
                self.generated_features.append(inv_ratio_col)

        return self.dataframe

    def generate_cyclical_features(
        self,
        column: str,
        period: int,
        drop_original: bool = False
    ) -> DataFrame:
        """
        Encode cyclical features using sin/cos transformation

        Essential for periodic features like:
        - Hour of day (period=24)
        - Day of week (period=7)
        - Month (period=12)
        - Day of month (period=31)

        Args:
            column: Column containing cyclical values
            period: Period of the cycle
            drop_original: Whether to drop the original column

        Returns:
            DataFrame with sin/cos encoded features
        """
        logger.info(f"Encoding cyclical feature {column} with period {period}")

        sin_col = f"{column}_sin"
        cos_col = f"{column}_cos"

        self.dataframe = self.dataframe.withColumn(
            sin_col,
            F.sin(2 * math.pi * F.col(column) / period)
        ).withColumn(
            cos_col,
            F.cos(2 * math.pi * F.col(column) / period)
        )

        self.generated_features.extend([sin_col, cos_col])

        if drop_original:
            self.dataframe = self.dataframe.drop(column)

        return self.dataframe

    def generate_target_encoding(
        self,
        columns: List[str],
        target_col: str,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1
    ) -> DataFrame:
        """
        Generate target-encoded features for categorical columns

        Target encoding replaces categories with the mean target value,
        with smoothing to handle rare categories.

        Args:
            columns: Categorical columns to encode
            target_col: Target column name
            smoothing: Smoothing parameter (higher = more regularization)
            min_samples_leaf: Minimum samples for reliable estimate

        Returns:
            DataFrame with target-encoded features
        """
        logger.info(f"Generating target encoding for {len(columns)} columns")

        # Calculate global mean
        global_mean = self.dataframe.select(F.mean(target_col)).collect()[0][0]

        for col in columns:
            # Calculate category statistics
            cat_stats = self.dataframe.groupBy(col).agg(
                F.mean(target_col).alias("cat_mean"),
                F.count(target_col).alias("cat_count")
            )

            # Apply smoothing: smoothed_mean = (n * cat_mean + m * global_mean) / (n + m)
            # where m = smoothing parameter
            cat_stats = cat_stats.withColumn(
                "smoothed_mean",
                (F.col("cat_count") * F.col("cat_mean") + smoothing * global_mean) /
                (F.col("cat_count") + smoothing)
            )

            # Join back to original dataframe
            encoded_col = f"{col}_target_enc"
            cat_stats = cat_stats.select(col, F.col("smoothed_mean").alias(encoded_col))

            self.dataframe = self.dataframe.join(cat_stats, on=col, how="left")

            # Fill any nulls with global mean
            self.dataframe = self.dataframe.withColumn(
                encoded_col,
                F.coalesce(F.col(encoded_col), F.lit(global_mean))
            )

            self.generated_features.append(encoded_col)

        return self.dataframe

    def generate_lag_features(
        self,
        columns: List[str],
        order_col: str,
        partition_cols: Optional[List[str]] = None,
        lags: List[int] = [1, 2, 3, 7]
    ) -> DataFrame:
        """
        Generate lag features for time-series data

        Args:
            columns: Columns to create lags for
            order_col: Column to order by (timestamp or sequence)
            partition_cols: Columns to partition by (e.g., entity ID)
            lags: List of lag periods

        Returns:
            DataFrame with lag features
        """
        logger.info(f"Generating lag features for {len(columns)} columns with lags {lags}")

        # Define window specification
        if partition_cols:
            window_spec = Window.partitionBy(*partition_cols).orderBy(order_col)
        else:
            window_spec = Window.orderBy(order_col)

        for col in columns:
            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                self.dataframe = self.dataframe.withColumn(
                    lag_col,
                    F.lag(F.col(col), lag).over(window_spec)
                )
                self.generated_features.append(lag_col)

        return self.dataframe

    def generate_rolling_features(
        self,
        columns: List[str],
        order_col: str,
        partition_cols: Optional[List[str]] = None,
        windows: List[int] = [3, 7, 14],
        functions: List[str] = ['mean', 'std', 'min', 'max']
    ) -> DataFrame:
        """
        Generate rolling window features for time-series data

        Args:
            columns: Columns to compute rolling stats for
            order_col: Column to order by
            partition_cols: Columns to partition by
            windows: List of window sizes
            functions: Aggregation functions to apply

        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Generating rolling features for {len(columns)} columns")

        for col in columns:
            for window_size in windows:
                # Define window (rows between -window_size and -1, excluding current)
                if partition_cols:
                    window_spec = Window.partitionBy(*partition_cols) \
                        .orderBy(order_col) \
                        .rowsBetween(-window_size, -1)
                else:
                    window_spec = Window.orderBy(order_col) \
                        .rowsBetween(-window_size, -1)

                for func in functions:
                    feature_name = f"{col}_rolling_{func}_{window_size}"

                    if func == 'mean':
                        self.dataframe = self.dataframe.withColumn(
                            feature_name,
                            F.avg(F.col(col)).over(window_spec)
                        )
                    elif func == 'std':
                        self.dataframe = self.dataframe.withColumn(
                            feature_name,
                            F.stddev(F.col(col)).over(window_spec)
                        )
                    elif func == 'min':
                        self.dataframe = self.dataframe.withColumn(
                            feature_name,
                            F.min(F.col(col)).over(window_spec)
                        )
                    elif func == 'max':
                        self.dataframe = self.dataframe.withColumn(
                            feature_name,
                            F.max(F.col(col)).over(window_spec)
                        )
                    elif func == 'sum':
                        self.dataframe = self.dataframe.withColumn(
                            feature_name,
                            F.sum(F.col(col)).over(window_spec)
                        )

                    self.generated_features.append(feature_name)

        return self.dataframe

    def generate_diff_features(
        self,
        columns: List[str],
        order_col: str,
        partition_cols: Optional[List[str]] = None,
        periods: List[int] = [1]
    ) -> DataFrame:
        """
        Generate difference features (for trend analysis)

        Args:
            columns: Columns to compute differences for
            order_col: Column to order by
            partition_cols: Columns to partition by
            periods: Difference periods

        Returns:
            DataFrame with difference features
        """
        logger.info(f"Generating difference features for {len(columns)} columns")

        if partition_cols:
            window_spec = Window.partitionBy(*partition_cols).orderBy(order_col)
        else:
            window_spec = Window.orderBy(order_col)

        for col in columns:
            for period in periods:
                diff_col = f"{col}_diff_{period}"
                pct_change_col = f"{col}_pct_change_{period}"

                # Absolute difference
                self.dataframe = self.dataframe.withColumn(
                    diff_col,
                    F.col(col) - F.lag(F.col(col), period).over(window_spec)
                )
                self.generated_features.append(diff_col)

                # Percentage change
                self.dataframe = self.dataframe.withColumn(
                    pct_change_col,
                    F.when(
                        F.lag(F.col(col), period).over(window_spec) != 0,
                        (F.col(col) - F.lag(F.col(col), period).over(window_spec)) /
                        F.abs(F.lag(F.col(col), period).over(window_spec))
                    ).otherwise(F.lit(0.0))
                )
                self.generated_features.append(pct_change_col)

        return self.dataframe

    def generate_statistical_features(
        self,
        columns: List[str]
    ) -> DataFrame:
        """
        Generate statistical features across multiple columns

        Creates row-wise statistics like:
        - Mean across numeric columns
        - Std deviation
        - Min/Max
        - Number of zeros/nulls

        Args:
            columns: Numeric columns to aggregate

        Returns:
            DataFrame with statistical features
        """
        logger.info(f"Generating row-wise statistical features for {len(columns)} columns")

        # Row-wise mean
        self.dataframe = self.dataframe.withColumn(
            "row_mean",
            sum([F.col(c) for c in columns]) / len(columns)
        )
        self.generated_features.append("row_mean")

        # Row-wise max
        self.dataframe = self.dataframe.withColumn(
            "row_max",
            F.greatest(*[F.col(c) for c in columns])
        )
        self.generated_features.append("row_max")

        # Row-wise min
        self.dataframe = self.dataframe.withColumn(
            "row_min",
            F.least(*[F.col(c) for c in columns])
        )
        self.generated_features.append("row_min")

        # Row-wise range
        self.dataframe = self.dataframe.withColumn(
            "row_range",
            F.col("row_max") - F.col("row_min")
        )
        self.generated_features.append("row_range")

        # Count of zeros
        zero_count_expr = sum([
            F.when(F.col(c) == 0, 1).otherwise(0)
            for c in columns
        ])
        self.dataframe = self.dataframe.withColumn("row_zero_count", zero_count_expr)
        self.generated_features.append("row_zero_count")

        # Count of nulls
        null_count_expr = sum([
            F.when(F.col(c).isNull(), 1).otherwise(0)
            for c in columns
        ])
        self.dataframe = self.dataframe.withColumn("row_null_count", null_count_expr)
        self.generated_features.append("row_null_count")

        return self.dataframe

    def select_features_by_importance(
        self,
        feature_importance: Dict[str, float],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Select features based on importance scores

        Args:
            feature_importance: Dictionary of feature_name -> importance_score
            top_k: Select top K features (mutually exclusive with threshold)
            threshold: Select features with importance >= threshold

        Returns:
            List of selected feature names
        """
        if top_k and threshold:
            raise ValueError("Specify either top_k or threshold, not both")

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if top_k:
            selected = [f[0] for f in sorted_features[:top_k]]
        elif threshold:
            selected = [f[0] for f, imp in sorted_features if imp >= threshold]
        else:
            selected = [f[0] for f in sorted_features]

        logger.info(f"Selected {len(selected)} features from {len(feature_importance)}")
        return selected

    def get_feature_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all generated features

        Returns:
            Dictionary with feature metadata
        """
        summary = {
            'total_features': len(self.generated_features),
            'feature_names': self.generated_features,
            'features_by_type': {}
        }

        # Categorize features by prefix
        for feature in self.generated_features:
            if '_log' in feature:
                category = 'log_transform'
            elif '_sqrt' in feature:
                category = 'sqrt_transform'
            elif '_square' in feature or '_pow' in feature:
                category = 'polynomial'
            elif '_mult_' in feature or '_div_' in feature or '_ratio_' in feature:
                category = 'interaction'
            elif '_binned' in feature:
                category = 'binning'
            elif '_sin' in feature or '_cos' in feature:
                category = 'cyclical'
            elif '_lag_' in feature:
                category = 'lag'
            elif '_rolling_' in feature:
                category = 'rolling'
            elif '_diff_' in feature or '_pct_change_' in feature:
                category = 'difference'
            elif '_target_enc' in feature:
                category = 'target_encoding'
            elif any(x in feature for x in ['year', 'month', 'day', 'week', 'quarter']):
                category = 'datetime'
            elif any(x in feature for x in ['length', 'word_count', 'unique_chars']):
                category = 'string'
            else:
                category = 'other'

            if category not in summary['features_by_type']:
                summary['features_by_type'][category] = []
            summary['features_by_type'][category].append(feature)

        return summary

    def generate_all_features(
        self,
        include_numerical: bool = True,
        include_interactions: bool = False,
        include_binning: bool = True,
        include_datetime: bool = True,
        include_string: bool = True,
        numerical_columns: list = None,
        categorical_columns: list = None,
        datetime_columns: list = None
    ) -> DataFrame:
        """
        Generate all features automatically
        
        Args:
            include_numerical: Whether to include numerical transformations
            include_interactions: Whether to include interaction features
            include_binning: Whether to include binning features
            include_datetime: Whether to include datetime features
            include_string: Whether to include string features
        
        Returns:
            DataFrame with all generated features
        """
        
        if not numerical_columns:
            numerical_columns = self.schema_checks.get_typed_col("numerical")
        
        if not categorical_columns:
            categorical_columns = self.schema_checks.get_typed_col(col_type="categorical")
            categorical_columns.remove(self.problem.target)

        if not datetime_columns:
            datetime_columns = self.schema_checks.get_typed_col(col_type="datetime")



        
        if include_numerical:
            print("Generating numerical features...")
            self.dataframe = self.generate_numerical_features(numerical_columns)
        
        if include_interactions:
            print("Generating interaction features...")
            self.dataframe = self.generate_interaction_features(numerical_columns)
        
        if include_binning:
            print("Generating binning features...")
            self.dataframe = self.generate_binning_features(numerical_columns)
        
        if include_datetime:
            print("Generating datetime features...")
            self.dataframe = self.generate_datetime_features(datetime_columns)
        
        if include_string:
            print("Generating string features...")
            self.dataframe = self.generate_string_features(categorical_columns)
        
        print(f"Total features generated: {len(self.generated_features)}")
        return self.dataframe
    
    def get_generated_features(self) -> List[str]:
        """Return list of all generated feature names"""
        return self.generated_features


# Example usage
if __name__ == "__main__":
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("AutoFeatureGeneration") \
        .getOrCreate()
    
    # Create sample data
    data = [
        (1, 100.0, 50.0, "2023-01-15", "Category A", "Sample text here"),
        (2, 200.0, 75.0, "2023-02-20", "Category B", "Another sample"),
        (3, 150.0, 60.0, "2023-03-10", "Category A", "More text data"),
        (4, 300.0, 90.0, "2023-04-05", "Category C", "Final example text"),
    ]
    
    dataframe = spark.createDataFrame(
        data, 
        ["id", "value1", "value2", "date_col", "category", "text_col"]
    )
    
    # Convert date column
    dataframe = dataframe.withColumn("date_col", F.to_date("date_col"))
    
    # Initialize feature generator
    feature_gen = AutoFeatureGenerator(spark)
    
    # Generate all features
    dataframe_with_features = feature_gen.generate_all_features(
        dataframe,
        include_numerical=True,
        include_interactions=False,  # Set to True for interactions
        include_binning=True,
        include_datetime=True,
        include_string=True
    )
    
    # Show results
    print("\nOriginal columns:", dataframe.columns)
    print("\nNew columns:", dataframe_with_features.columns)
    print("\nGenerated features:", feature_gen.get_generated_features())
    
    dataframe_with_features.show(5, truncate=False)
    
    spark.stop()