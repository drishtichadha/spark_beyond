"""
PySpark <-> Pandas DataFrame conversion utilities

This module provides safe and efficient conversion between PySpark and Pandas DataFrames,
with automatic sampling for large datasets to avoid memory issues.
"""

from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def spark_to_pandas_safe(
    spark_df: SparkDataFrame,
    max_rows: Optional[int] = None,
    sample: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    """
    Safely convert PySpark DataFrame to Pandas with size checks and sampling

    Args:
        spark_df: PySpark DataFrame to convert
        max_rows: Maximum number of rows to convert (None = no limit)
        sample: If True, randomly sample when exceeding max_rows; if False, take first N rows
        seed: Random seed for sampling (for reproducibility)

    Returns:
        Pandas DataFrame

    Examples:
        # Convert small dataset completely
        pdf = spark_to_pandas_safe(spark_df)

        # Sample large dataset to 100K rows
        pdf = spark_to_pandas_safe(spark_df, max_rows=100000, sample=True)

        # Take first 50K rows
        pdf = spark_to_pandas_safe(spark_df, max_rows=50000, sample=False)
    """
    try:
        row_count = spark_df.count()
        logger.info(f"Converting PySpark DataFrame with {row_count:,} rows to Pandas")

        if max_rows and row_count > max_rows:
            logger.warning(
                f"Dataset has {row_count:,} rows, exceeding max_rows={max_rows:,}. "
                f"{'Sampling' if sample else 'Taking first'} {max_rows:,} rows."
            )

            if sample:
                # Random sampling to maintain distribution
                fraction = max_rows / row_count
                sampled_df = spark_df.sample(fraction=fraction, seed=seed)

                # Ensure we get close to max_rows (sampling is approximate)
                actual_count = sampled_df.count()
                if actual_count > max_rows * 1.1:  # Allow 10% overage
                    logger.info(f"Sample produced {actual_count:,} rows, limiting to {max_rows:,}")
                    return sampled_df.limit(max_rows).toPandas()

                logger.info(f"Sampled {actual_count:,} rows")
                return sampled_df.toPandas()
            else:
                # Take first N rows
                return spark_df.limit(max_rows).toPandas()

        # Convert entire dataset
        logger.info("Converting entire dataset to Pandas")
        return spark_df.toPandas()

    except Exception as e:
        logger.error(f"Error converting PySpark to Pandas: {str(e)}")
        raise


def pandas_to_spark(
    pandas_df: pd.DataFrame,
    spark: SparkSession,
    schema=None,
    validate_schema: bool = True
) -> SparkDataFrame:
    """
    Convert Pandas DataFrame back to PySpark with schema preservation

    Args:
        pandas_df: Pandas DataFrame to convert
        spark: SparkSession instance
        schema: Optional PySpark schema to use (None = infer)
        validate_schema: If True, validate data types match expected schema

    Returns:
        PySpark DataFrame

    Examples:
        # Simple conversion with schema inference
        spark_df = pandas_to_spark(pdf, spark)

        # Preserve original schema
        original_schema = spark_df.schema
        pdf = spark_to_pandas_safe(spark_df)
        # ... modify pdf ...
        spark_df_new = pandas_to_spark(pdf, spark, schema=original_schema)
    """
    try:
        logger.info(f"Converting Pandas DataFrame ({len(pandas_df):,} rows, {len(pandas_df.columns)} cols) to PySpark")

        if schema:
            logger.info("Using provided schema for conversion")
            spark_df = spark.createDataFrame(pandas_df, schema=schema)
        else:
            logger.info("Inferring schema from Pandas DataFrame")
            spark_df = spark.createDataFrame(pandas_df)

        if validate_schema and schema:
            # Verify schema matches
            actual_fields = {f.name: f.dataType for f in spark_df.schema.fields}
            expected_fields = {f.name: f.dataType for f in schema.fields}

            mismatches = []
            for col, expected_type in expected_fields.items():
                if col in actual_fields and str(actual_fields[col]) != str(expected_type):
                    mismatches.append(f"{col}: expected {expected_type}, got {actual_fields[col]}")

            if mismatches:
                logger.warning(f"Schema mismatches detected: {', '.join(mismatches)}")

        logger.info("Conversion to PySpark completed successfully")
        return spark_df

    except Exception as e:
        logger.error(f"Error converting Pandas to PySpark: {str(e)}")
        raise


def estimate_memory_usage(spark_df: SparkDataFrame) -> dict:
    """
    Estimate memory usage of a PySpark DataFrame when converted to Pandas

    Args:
        spark_df: PySpark DataFrame to analyze

    Returns:
        Dictionary with memory estimates
    """
    try:
        row_count = spark_df.count()
        col_count = len(spark_df.columns)

        # Sample a small subset to estimate memory per row
        sample_size = min(1000, row_count)
        sample_df = spark_df.limit(sample_size).toPandas()

        # Calculate memory usage of sample
        sample_memory_bytes = sample_df.memory_usage(deep=True).sum()
        memory_per_row = sample_memory_bytes / len(sample_df)

        # Estimate total memory needed
        estimated_memory_bytes = memory_per_row * row_count
        estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
        estimated_memory_gb = estimated_memory_mb / 1024

        return {
            'row_count': row_count,
            'col_count': col_count,
            'estimated_memory_bytes': int(estimated_memory_bytes),
            'estimated_memory_mb': round(estimated_memory_mb, 2),
            'estimated_memory_gb': round(estimated_memory_gb, 2),
            'memory_per_row_bytes': int(memory_per_row),
            'safe_for_pandas': estimated_memory_gb < 2.0,  # Conservative 2GB threshold
            'recommended_sample_size': int(2e9 / memory_per_row) if memory_per_row > 0 else None
        }

    except Exception as e:
        logger.error(f"Error estimating memory usage: {str(e)}")
        return {
            'error': str(e),
            'safe_for_pandas': False
        }


def get_conversion_strategy(spark_df: SparkDataFrame, max_memory_gb: float = 2.0) -> dict:
    """
    Recommend conversion strategy based on dataset size

    Args:
        spark_df: PySpark DataFrame to analyze
        max_memory_gb: Maximum memory to allow for Pandas conversion

    Returns:
        Dictionary with recommended strategy
    """
    memory_info = estimate_memory_usage(spark_df)

    if 'error' in memory_info:
        return {
            'strategy': 'keep_pyspark',
            'reason': 'Unable to estimate memory usage',
            'details': memory_info
        }

    estimated_gb = memory_info['estimated_memory_gb']

    if estimated_gb < max_memory_gb:
        return {
            'strategy': 'full_conversion',
            'reason': f'Dataset fits in memory ({estimated_gb:.2f} GB < {max_memory_gb} GB)',
            'max_rows': None,
            'details': memory_info
        }
    else:
        recommended_rows = memory_info.get('recommended_sample_size', 100000)
        return {
            'strategy': 'sample_conversion',
            'reason': f'Dataset too large ({estimated_gb:.2f} GB > {max_memory_gb} GB)',
            'max_rows': recommended_rows,
            'sample': True,
            'details': memory_info
        }


# Convenience function for automatic conversion
def auto_convert_to_pandas(
    spark_df: SparkDataFrame,
    max_memory_gb: float = 2.0,
    seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Automatically convert PySpark to Pandas with optimal strategy

    Args:
        spark_df: PySpark DataFrame to convert
        max_memory_gb: Maximum memory threshold for full conversion
        seed: Random seed for sampling

    Returns:
        Tuple of (Pandas DataFrame, strategy info dict)
    """
    strategy = get_conversion_strategy(spark_df, max_memory_gb)

    logger.info(f"Auto-conversion strategy: {strategy['strategy']} - {strategy['reason']}")

    if strategy['strategy'] == 'keep_pyspark':
        raise ValueError(
            "Dataset cannot be safely converted to Pandas. "
            "Use PySpark-native operations or reduce dataset size."
        )
    elif strategy['strategy'] == 'full_conversion':
        pdf = spark_to_pandas_safe(spark_df, max_rows=None, seed=seed)
    else:  # sample_conversion
        pdf = spark_to_pandas_safe(
            spark_df,
            max_rows=strategy['max_rows'],
            sample=strategy.get('sample', True),
            seed=seed
        )

    return pdf, strategy
