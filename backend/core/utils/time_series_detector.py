"""
Time-series pattern detection utilities

This module automatically detects time-series structures in datasets and recommends
appropriate feature engineering strategies.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from backend.core.discovery import SchemaChecks
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeSeriesFrequency(str, Enum):
    """Detected frequency of time-series data"""
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


@dataclass
class TimeSeriesInfo:
    """
    Information about detected time-series structure

    Attributes:
        is_time_series: Whether dataset has time-series structure
        time_column: Primary datetime column name
        entity_columns: Columns identifying entities (e.g., customer_id, product_id)
        frequency: Detected frequency of observations
        avg_records_per_entity: Average number of time points per entity
        date_range_days: Total date range in days
        has_regular_intervals: Whether observations are regularly spaced
        recommended_features: List of recommended time-series features
        warnings: List of data quality warnings
    """
    is_time_series: bool
    time_column: Optional[str] = None
    entity_columns: Optional[List[str]] = None
    frequency: TimeSeriesFrequency = TimeSeriesFrequency.UNKNOWN
    avg_records_per_entity: Optional[float] = None
    date_range_days: Optional[float] = None
    has_regular_intervals: bool = False
    recommended_features: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


def detect_time_series_structure(
    df: DataFrame,
    schema_checks: SchemaChecks,
    min_records_per_entity: int = 3,
    entity_id_patterns: List[str] = None
) -> TimeSeriesInfo:
    """
    Auto-detect if dataset has time-series structure and characteristics

    Args:
        df: PySpark DataFrame to analyze
        schema_checks: SchemaChecks instance with column type information
        min_records_per_entity: Minimum records per entity to consider as time-series
        entity_id_patterns: Column name patterns that might identify entities
                          (e.g., ['id', 'customer', 'user', 'account'])

    Returns:
        TimeSeriesInfo object with detection results

    Examples:
        schema_checks = SchemaChecks(df, problem)
        ts_info = detect_time_series_structure(df, schema_checks)

        if ts_info.is_time_series:
            print(f"Time-series detected with {ts_info.frequency} frequency")
            print(f"Recommended features: {ts_info.recommended_features}")
    """
    warnings = []
    metadata = {}

    # Default entity ID patterns if not provided
    if entity_id_patterns is None:
        entity_id_patterns = ['id', 'customer', 'user', 'account', 'product', 'store', 'entity']

    # Step 1: Check for datetime columns
    datetime_cols = schema_checks.datetime_cols

    if not datetime_cols:
        logger.info("No datetime columns found - not a time-series dataset")
        return TimeSeriesInfo(
            is_time_series=False,
            warnings=["No datetime columns detected in dataset"]
        )

    # Step 2: Identify primary time column
    time_column = _identify_primary_time_column(df, datetime_cols)
    logger.info(f"Primary time column identified: {time_column}")

    # Step 3: Identify entity columns (ID columns that group time-series)
    entity_columns = _identify_entity_columns(df, schema_checks, entity_id_patterns, time_column)
    logger.info(f"Entity columns identified: {entity_columns}")

    # Step 4: Analyze temporal structure
    temporal_analysis = _analyze_temporal_structure(df, time_column, entity_columns)
    metadata.update(temporal_analysis)

    # Step 5: Detect frequency
    frequency = _detect_frequency(temporal_analysis)

    # Step 6: Check if it's truly time-series (multiple records per entity over time)
    if entity_columns:
        avg_records = temporal_analysis.get('avg_records_per_entity', 0)
        is_time_series = avg_records >= min_records_per_entity
    else:
        # If no entity columns, check if there are multiple time points
        unique_dates = temporal_analysis.get('unique_time_points', 0)
        total_rows = df.count()
        is_time_series = unique_dates > 1 and total_rows / unique_dates >= min_records_per_entity

    if not is_time_series:
        warnings.append(
            f"Dataset has datetime column but insufficient temporal variation "
            f"(avg {temporal_analysis.get('avg_records_per_entity', 0):.1f} records per entity)"
        )

    # Step 7: Recommend time-series features
    recommended_features = _recommend_time_series_features(
        is_time_series,
        frequency,
        temporal_analysis,
        entity_columns
    )

    # Step 8: Check data quality
    quality_warnings = _check_time_series_quality(df, time_column, temporal_analysis)
    warnings.extend(quality_warnings)

    return TimeSeriesInfo(
        is_time_series=is_time_series,
        time_column=time_column,
        entity_columns=entity_columns,
        frequency=frequency,
        avg_records_per_entity=temporal_analysis.get('avg_records_per_entity'),
        date_range_days=temporal_analysis.get('date_range_days'),
        has_regular_intervals=temporal_analysis.get('has_regular_intervals', False),
        recommended_features=recommended_features,
        warnings=warnings if warnings else None,
        metadata=metadata
    )


def _identify_primary_time_column(df: DataFrame, datetime_cols: List[str]) -> str:
    """
    Identify the primary datetime column from multiple candidates

    Prefers columns with:
    1. Most unique values (higher granularity)
    2. Names suggesting primary time (date, time, timestamp, created_at, etc.)
    """
    if len(datetime_cols) == 1:
        return datetime_cols[0]

    # Priority keywords for time columns
    priority_keywords = ['timestamp', 'date', 'time', 'created', 'occurred', 'transaction']

    # Score each column
    scores = {}
    for col in datetime_cols:
        score = 0

        # Name-based scoring
        col_lower = col.lower()
        for keyword in priority_keywords:
            if keyword in col_lower:
                score += 10
                break

        # Uniqueness scoring (sample to avoid expensive count)
        unique_ratio = df.select(col).distinct().count() / df.count()
        score += unique_ratio * 5

        scores[col] = score

    # Return column with highest score
    primary_col = max(scores.items(), key=lambda x: x[1])[0]
    return primary_col


def _identify_entity_columns(
    df: DataFrame,
    schema_checks: SchemaChecks,
    entity_id_patterns: List[str],
    time_column: str
) -> Optional[List[str]]:
    """
    Identify columns that represent entities in the time-series

    Entity columns are typically:
    - Categorical or string columns
    - Low cardinality relative to dataset size
    - Match common ID patterns (id, customer_id, etc.)
    - Not the time column
    """
    entity_columns = []

    # Consider categorical columns
    categorical_cols = schema_checks.categorical_cols

    for col in categorical_cols:
        if col == time_column:
            continue

        col_lower = col.lower()

        # Check if column name matches entity patterns
        matches_pattern = any(pattern in col_lower for pattern in entity_id_patterns)

        if matches_pattern:
            # Verify it has reasonable cardinality (not too many unique values)
            total_rows = df.count()
            distinct_count = df.select(col).distinct().count()
            cardinality_ratio = distinct_count / total_rows

            # If cardinality is between 0.01% and 50% of rows, likely an entity ID
            if 0.0001 < cardinality_ratio < 0.5:
                entity_columns.append(col)
                logger.info(f"Entity column found: {col} (cardinality: {distinct_count:,})")

    return entity_columns if entity_columns else None


def _analyze_temporal_structure(
    df: DataFrame,
    time_column: str,
    entity_columns: Optional[List[str]]
) -> Dict[str, Any]:
    """
    Analyze temporal structure of the data
    """
    analysis = {}

    # Date range
    date_stats = df.agg(
        F.min(time_column).alias('min_date'),
        F.max(time_column).alias('max_date'),
        F.count(time_column).alias('total_records'),
        F.countDistinct(time_column).alias('unique_time_points')
    ).collect()[0]

    min_date = date_stats['min_date']
    max_date = date_stats['max_date']
    total_records = date_stats['total_records']
    unique_time_points = date_stats['unique_time_points']

    if min_date and max_date:
        date_range_days = (max_date - min_date).total_seconds() / (24 * 3600)
        analysis['date_range_days'] = date_range_days
        analysis['min_date'] = min_date
        analysis['max_date'] = max_date
    else:
        analysis['date_range_days'] = None

    analysis['total_records'] = total_records
    analysis['unique_time_points'] = unique_time_points

    # Entity-level analysis
    if entity_columns:
        # Calculate records per entity
        entity_counts = df.groupBy(*entity_columns).count()
        entity_stats = entity_counts.agg(
            F.avg('count').alias('avg_count'),
            F.min('count').alias('min_count'),
            F.max('count').alias('max_count'),
            F.stddev('count').alias('stddev_count')
        ).collect()[0]

        analysis['avg_records_per_entity'] = entity_stats['avg_count']
        analysis['min_records_per_entity'] = entity_stats['min_count']
        analysis['max_records_per_entity'] = entity_stats['max_count']
        analysis['stddev_records_per_entity'] = entity_stats['stddev_count']

        # Check regularity of intervals
        regularity_ratio = entity_stats['stddev_count'] / entity_stats['avg_count'] if entity_stats['avg_count'] > 0 else 1
        analysis['has_regular_intervals'] = regularity_ratio < 0.3  # Low variance = regular

    else:
        # No entities, analyze overall time distribution
        analysis['avg_records_per_entity'] = total_records / unique_time_points if unique_time_points > 0 else 0

    return analysis


def _detect_frequency(temporal_analysis: Dict[str, Any]) -> TimeSeriesFrequency:
    """
    Detect the frequency of time-series observations
    """
    date_range_days = temporal_analysis.get('date_range_days')
    total_records = temporal_analysis.get('total_records', 0)
    unique_time_points = temporal_analysis.get('unique_time_points', 0)

    if not date_range_days or date_range_days == 0:
        return TimeSeriesFrequency.UNKNOWN

    # Calculate average interval in days
    avg_interval_days = date_range_days / unique_time_points if unique_time_points > 0 else 0

    # Detect frequency based on average interval
    if avg_interval_days < 0.001:  # < ~1.5 minutes
        return TimeSeriesFrequency.SECONDS
    elif avg_interval_days < 0.05:  # < ~1 hour
        return TimeSeriesFrequency.MINUTES
    elif avg_interval_days < 0.5:  # < 12 hours
        return TimeSeriesFrequency.HOURLY
    elif avg_interval_days < 3:  # ~1-2 days
        return TimeSeriesFrequency.DAILY
    elif avg_interval_days < 14:  # ~1-2 weeks
        return TimeSeriesFrequency.WEEKLY
    elif avg_interval_days < 45:  # ~1 month
        return TimeSeriesFrequency.MONTHLY
    elif avg_interval_days < 120:  # ~3-4 months
        return TimeSeriesFrequency.QUARTERLY
    elif avg_interval_days < 400:  # ~1 year
        return TimeSeriesFrequency.YEARLY
    else:
        return TimeSeriesFrequency.IRREGULAR


def _recommend_time_series_features(
    is_time_series: bool,
    frequency: TimeSeriesFrequency,
    temporal_analysis: Dict[str, Any],
    entity_columns: Optional[List[str]]
) -> List[str]:
    """
    Recommend time-series features based on detected structure
    """
    if not is_time_series:
        return ['basic_datetime_features']

    recommended = []

    # Basic temporal features
    recommended.extend(['date_components', 'cyclical_encoding'])

    # Lag features based on frequency
    if frequency in [TimeSeriesFrequency.DAILY, TimeSeriesFrequency.WEEKLY]:
        recommended.append('lag_features_7d')
        recommended.append('rolling_mean_7d')
    elif frequency == TimeSeriesFrequency.MONTHLY:
        recommended.append('lag_features_3m')
        recommended.append('rolling_mean_3m')
    elif frequency in [TimeSeriesFrequency.HOURLY, TimeSeriesFrequency.MINUTES]:
        recommended.append('lag_features_24h')
        recommended.append('rolling_mean_24h')

    # Trend and seasonality
    if temporal_analysis.get('date_range_days', 0) > 365:
        recommended.append('yearly_seasonality')

    if frequency in [TimeSeriesFrequency.DAILY, TimeSeriesFrequency.HOURLY]:
        recommended.append('day_of_week_patterns')

    # tsfresh features for complex patterns
    if entity_columns and temporal_analysis.get('avg_records_per_entity', 0) >= 10:
        recommended.append('tsfresh_comprehensive')
    else:
        recommended.append('tsfresh_minimal')

    # Difference features for trend removal
    recommended.append('difference_features')

    return recommended


def _check_time_series_quality(
    df: DataFrame,
    time_column: str,
    temporal_analysis: Dict[str, Any]
) -> List[str]:
    """
    Check for common time-series data quality issues
    """
    warnings = []

    # Check for nulls in time column
    null_count = df.filter(F.col(time_column).isNull()).count()
    if null_count > 0:
        warnings.append(f"Time column '{time_column}' has {null_count:,} null values")

    # Check for duplicate timestamps (if no entities)
    unique_ratio = temporal_analysis['unique_time_points'] / temporal_analysis['total_records']
    if unique_ratio < 0.5:
        warnings.append(
            f"Low time point uniqueness ({unique_ratio:.1%}) - "
            f"may indicate multiple entities or duplicate timestamps"
        )

    # Check for large gaps in time
    date_range_days = temporal_analysis.get('date_range_days', 0)
    avg_records_per_day = temporal_analysis['total_records'] / date_range_days if date_range_days > 0 else 0

    if date_range_days > 30 and avg_records_per_day < 0.1:
        warnings.append(
            f"Sparse time-series detected: avg {avg_records_per_day:.2f} records/day over {date_range_days:.0f} days"
        )

    return warnings
