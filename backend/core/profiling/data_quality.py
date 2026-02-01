"""
PySpark-native data quality checks

This module provides data quality analysis without requiring Pandas conversion,
suitable for very large datasets.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, DateType, TimestampType
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """
    Data quality report results

    Attributes:
        row_count: Total number of rows
        column_count: Total number of columns
        missing_summary: Missing value statistics per column
        duplicate_count: Number of duplicate rows
        column_stats: Statistical summary per column
        quality_score: Overall quality score (0-100)
        issues: List of identified quality issues
        recommendations: List of recommended actions
    """
    row_count: int
    column_count: int
    missing_summary: Dict[str, Dict[str, Any]]
    duplicate_count: int
    column_stats: Dict[str, Dict[str, Any]]
    quality_score: float
    issues: List[Dict[str, str]] = field(default_factory=list)
    recommendations: List[Dict[str, str]] = field(default_factory=list)


class DataQualityChecker:
    """
    PySpark-native data quality checker for large datasets

    This class performs quality checks without converting to Pandas,
    making it suitable for datasets too large to fit in memory.

    Example:
        checker = DataQualityChecker(spark_df)
        report = checker.run_all_checks()
        print(f"Quality Score: {report.quality_score}")
        for issue in report.issues:
            print(f"  - {issue['column']}: {issue['issue']}")
    """

    def __init__(self, df: DataFrame):
        """
        Initialize the quality checker

        Args:
            df: PySpark DataFrame to analyze
        """
        self.df = df
        self._schema = df.schema
        self._row_count: Optional[int] = None

    @property
    def row_count(self) -> int:
        """Lazy evaluation of row count"""
        if self._row_count is None:
            self._row_count = self.df.count()
        return self._row_count

    def check_missing_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Check for missing values in all columns

        Returns:
            Dictionary with missing value stats per column
        """
        logger.info("Checking missing values...")
        missing_stats = {}

        # Get string columns for empty string check
        string_columns = {
            field.name for field in self._schema.fields
            if isinstance(field.dataType, StringType)
        }

        # Build aggregation expressions for all columns
        agg_exprs = []
        for col in self.df.columns:
            agg_exprs.append(
                F.count(F.when(F.col(col).isNull(), 1)).alias(f"{col}_null_count")
            )
            # Only check for empty strings on string columns
            if col in string_columns:
                agg_exprs.append(
                    F.count(F.when(F.col(col) == "", 1)).alias(f"{col}_empty_count")
                )

        # Execute single aggregation for all columns
        result = self.df.agg(*agg_exprs).collect()[0]

        for col in self.df.columns:
            null_count = result[f"{col}_null_count"]
            # Only get empty count for string columns
            empty_count = result[f"{col}_empty_count"] if col in string_columns else 0
            total_missing = null_count + empty_count

            missing_stats[col] = {
                'null_count': null_count,
                'empty_count': empty_count,
                'total_missing': total_missing,
                'missing_pct': (total_missing / self.row_count * 100) if self.row_count > 0 else 0
            }

        return missing_stats

    def check_duplicates(self, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows

        Args:
            subset: Columns to consider for duplicate detection (None = all columns)

        Returns:
            Number of duplicate rows
        """
        logger.info("Checking duplicates...")
        if subset:
            distinct_count = self.df.select(subset).distinct().count()
        else:
            distinct_count = self.df.distinct().count()

        return self.row_count - distinct_count

    def check_column_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistical summary for all columns

        Returns:
            Dictionary with stats per column
        """
        logger.info("Calculating column statistics...")
        stats = {}

        for field in self._schema.fields:
            col_name = field.name
            col_type = field.dataType

            col_stats = {
                'dtype': str(col_type),
                'distinct_count': self.df.select(col_name).distinct().count()
            }

            # Numeric column stats
            if isinstance(col_type, NumericType):
                numeric_stats = self.df.select(
                    F.mean(col_name).alias('mean'),
                    F.stddev(col_name).alias('std'),
                    F.min(col_name).alias('min'),
                    F.max(col_name).alias('max'),
                    F.expr(f'percentile_approx({col_name}, 0.5)').alias('median'),
                    F.expr(f'percentile_approx({col_name}, 0.25)').alias('q1'),
                    F.expr(f'percentile_approx({col_name}, 0.75)').alias('q3'),
                    F.skewness(col_name).alias('skewness'),
                    F.kurtosis(col_name).alias('kurtosis')
                ).collect()[0]

                col_stats.update({
                    'mean': numeric_stats['mean'],
                    'std': numeric_stats['std'],
                    'min': numeric_stats['min'],
                    'max': numeric_stats['max'],
                    'median': numeric_stats['median'],
                    'q1': numeric_stats['q1'],
                    'q3': numeric_stats['q3'],
                    'skewness': numeric_stats['skewness'],
                    'kurtosis': numeric_stats['kurtosis'],
                    'iqr': (numeric_stats['q3'] - numeric_stats['q1']) if numeric_stats['q3'] and numeric_stats['q1'] else None
                })

            # String/Categorical column stats
            elif isinstance(col_type, StringType):
                string_stats = self.df.select(
                    F.min(F.length(col_name)).alias('min_length'),
                    F.max(F.length(col_name)).alias('max_length'),
                    F.avg(F.length(col_name)).alias('avg_length')
                ).collect()[0]

                col_stats.update({
                    'min_length': string_stats['min_length'],
                    'max_length': string_stats['max_length'],
                    'avg_length': string_stats['avg_length'],
                    'cardinality_ratio': col_stats['distinct_count'] / self.row_count if self.row_count > 0 else 0
                })

            # Date/Timestamp column stats
            elif isinstance(col_type, (DateType, TimestampType)):
                date_stats = self.df.select(
                    F.min(col_name).alias('min_date'),
                    F.max(col_name).alias('max_date')
                ).collect()[0]

                col_stats.update({
                    'min_date': date_stats['min_date'],
                    'max_date': date_stats['max_date']
                })

            stats[col_name] = col_stats

        return stats

    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers in numeric columns

        Args:
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for zscore)

        Returns:
            Dictionary with outlier info per numeric column
        """
        logger.info(f"Detecting outliers using {method} method...")
        outliers = {}

        for field in self._schema.fields:
            if not isinstance(field.dataType, NumericType):
                continue

            col_name = field.name

            if method == 'iqr':
                # Calculate IQR bounds
                quantiles = self.df.select(
                    F.expr(f'percentile_approx({col_name}, 0.25)').alias('q1'),
                    F.expr(f'percentile_approx({col_name}, 0.75)').alias('q3')
                ).collect()[0]

                q1, q3 = quantiles['q1'], quantiles['q3']
                if q1 is not None and q3 is not None:
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr

                    outlier_count = self.df.filter(
                        (F.col(col_name) < lower_bound) | (F.col(col_name) > upper_bound)
                    ).count()

                    outliers[col_name] = {
                        'method': 'iqr',
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_count': outlier_count,
                        'outlier_pct': outlier_count / self.row_count * 100 if self.row_count > 0 else 0
                    }

            elif method == 'zscore':
                # Calculate z-score bounds
                stats = self.df.select(
                    F.mean(col_name).alias('mean'),
                    F.stddev(col_name).alias('std')
                ).collect()[0]

                mean, std = stats['mean'], stats['std']
                if mean is not None and std is not None and std > 0:
                    outlier_count = self.df.filter(
                        F.abs((F.col(col_name) - mean) / std) > threshold
                    ).count()

                    outliers[col_name] = {
                        'method': 'zscore',
                        'mean': mean,
                        'std': std,
                        'threshold': threshold,
                        'outlier_count': outlier_count,
                        'outlier_pct': outlier_count / self.row_count * 100 if self.row_count > 0 else 0
                    }

        return outliers

    def check_cardinality(self) -> Dict[str, Dict[str, Any]]:
        """
        Check cardinality of categorical columns

        Returns:
            Dictionary with cardinality info per column
        """
        logger.info("Checking cardinality...")
        cardinality = {}

        for field in self._schema.fields:
            if not isinstance(field.dataType, StringType):
                continue

            col_name = field.name
            distinct_count = self.df.select(col_name).distinct().count()

            # Get top categories
            top_categories = self.df.groupBy(col_name).count() \
                .orderBy(F.desc('count')) \
                .limit(10) \
                .collect()

            cardinality[col_name] = {
                'distinct_count': distinct_count,
                'cardinality_ratio': distinct_count / self.row_count if self.row_count > 0 else 0,
                'is_high_cardinality': distinct_count > 50,
                'is_potential_id': distinct_count / self.row_count > 0.9 if self.row_count > 0 else False,
                'top_categories': [
                    {'value': row[col_name], 'count': row['count']}
                    for row in top_categories
                ]
            }

        return cardinality

    def calculate_quality_score(
        self,
        missing_summary: Dict[str, Dict[str, Any]],
        duplicate_count: int,
        column_stats: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate overall data quality score (0-100)

        Args:
            missing_summary: Missing value statistics
            duplicate_count: Number of duplicate rows
            column_stats: Column statistics

        Returns:
            Quality score between 0 and 100
        """
        score = 100.0
        penalties = []

        # Penalty for missing values
        for col, stats in missing_summary.items():
            missing_pct = stats['missing_pct']
            if missing_pct > 50:
                penalties.append(('missing', col, 5))
            elif missing_pct > 20:
                penalties.append(('missing', col, 3))
            elif missing_pct > 5:
                penalties.append(('missing', col, 1))

        # Penalty for duplicates
        duplicate_pct = duplicate_count / self.row_count * 100 if self.row_count > 0 else 0
        if duplicate_pct > 20:
            penalties.append(('duplicates', 'dataset', 10))
        elif duplicate_pct > 5:
            penalties.append(('duplicates', 'dataset', 5))

        # Penalty for constant columns
        for col, stats in column_stats.items():
            if stats.get('distinct_count', 0) == 1:
                penalties.append(('constant', col, 2))

        # Apply penalties
        total_penalty = sum(p[2] for p in penalties)
        score = max(0, score - total_penalty)

        return round(score, 1)

    def identify_issues(
        self,
        missing_summary: Dict[str, Dict[str, Any]],
        column_stats: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Identify data quality issues

        Returns:
            List of issues with severity
        """
        issues = []

        for col, stats in missing_summary.items():
            missing_pct = stats['missing_pct']
            if missing_pct > 50:
                issues.append({
                    'column': col,
                    'issue': f"Very high missing values ({missing_pct:.1f}%)",
                    'severity': 'high'
                })
            elif missing_pct > 20:
                issues.append({
                    'column': col,
                    'issue': f"High missing values ({missing_pct:.1f}%)",
                    'severity': 'medium'
                })

        for col, stats in column_stats.items():
            # Constant columns
            if stats.get('distinct_count', 0) == 1:
                issues.append({
                    'column': col,
                    'issue': "Constant column (only one unique value)",
                    'severity': 'high'
                })

            # High skewness
            skewness = stats.get('skewness')
            if skewness is not None and abs(skewness) > 3:
                issues.append({
                    'column': col,
                    'issue': f"Highly skewed distribution (skewness: {skewness:.2f})",
                    'severity': 'low'
                })

            # Potential ID columns (very high cardinality)
            if stats.get('cardinality_ratio', 0) > 0.95:
                issues.append({
                    'column': col,
                    'issue': "Potential ID column (very high cardinality)",
                    'severity': 'low'
                })

        return issues

    def generate_recommendations(
        self,
        missing_summary: Dict[str, Dict[str, Any]],
        column_stats: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Generate preprocessing recommendations

        Returns:
            List of recommendations
        """
        recommendations = []

        for col, stats in missing_summary.items():
            missing_pct = stats['missing_pct']
            if missing_pct > 50:
                recommendations.append({
                    'column': col,
                    'action': f"Consider dropping column ({missing_pct:.1f}% missing)",
                    'priority': 'high'
                })
            elif missing_pct > 5:
                col_stats = column_stats.get(col, {})
                if 'mean' in col_stats:
                    recommendations.append({
                        'column': col,
                        'action': f"Impute with median ({missing_pct:.1f}% missing)",
                        'priority': 'medium'
                    })
                else:
                    recommendations.append({
                        'column': col,
                        'action': f"Impute with mode ({missing_pct:.1f}% missing)",
                        'priority': 'medium'
                    })

        for col, stats in column_stats.items():
            skewness = stats.get('skewness')
            if skewness is not None and abs(skewness) > 2:
                recommendations.append({
                    'column': col,
                    'action': "Apply log or Box-Cox transformation",
                    'priority': 'low'
                })

            if stats.get('distinct_count', 0) == 1:
                recommendations.append({
                    'column': col,
                    'action': "Remove constant column",
                    'priority': 'high'
                })

        return recommendations

    def run_all_checks(self) -> QualityReport:
        """
        Run all data quality checks

        Returns:
            Complete QualityReport
        """
        logger.info(f"Running data quality checks on {self.row_count:,} rows...")

        # Run checks
        missing_summary = self.check_missing_values()
        duplicate_count = self.check_duplicates()
        column_stats = self.check_column_stats()

        # Calculate quality score
        quality_score = self.calculate_quality_score(
            missing_summary, duplicate_count, column_stats
        )

        # Identify issues
        issues = self.identify_issues(missing_summary, column_stats)

        # Generate recommendations
        recommendations = self.generate_recommendations(missing_summary, column_stats)

        return QualityReport(
            row_count=self.row_count,
            column_count=len(self.df.columns),
            missing_summary=missing_summary,
            duplicate_count=duplicate_count,
            column_stats=column_stats,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
