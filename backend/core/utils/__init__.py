"""
Utility modules for Spark Beyond ML Platform

This package contains utility functions for:
- DataFrame conversions between PySpark and Pandas
- Time-series pattern detection
- Helper functions for data processing
"""

from .spark_pandas_bridge import spark_to_pandas_safe, pandas_to_spark
from .time_series_detector import detect_time_series_structure, TimeSeriesInfo
from .common import process_col_names

__all__ = [
    'spark_to_pandas_safe',
    'pandas_to_spark',
    'detect_time_series_structure',
    'TimeSeriesInfo',
    'process_col_names'
]
