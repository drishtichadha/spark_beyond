"""
Feature Insight Analyzer

Provides lift, support, and Relative Information Gain (RIG) analysis
for feature discovery and microsegment identification.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from backend.core.discovery import Problem, ProblemType, SchemaChecks

logger = logging.getLogger(__name__)


@dataclass
class FeatureInsight:
    """Represents an insight about a feature condition"""
    feature_name: str
    condition: str  # e.g., "degree_Pct >= 45.752"
    condition_type: str  # 'numeric_threshold', 'categorical', 'range'
    target_class: str
    lift: float  # How much better than baseline
    support: float  # Percentage of total population
    support_count: int  # Absolute count
    rig: float  # Relative Information Gain
    class_rate: float  # Rate of target class in this segment
    baseline_rate: float  # Baseline rate in population
    description: Optional[str] = None
    group: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Microsegment:
    """Represents a combination of feature conditions"""
    name: str
    conditions: List[Dict[str, Any]]
    target_class: str
    lift: float
    support: float
    support_count: int
    rig: float
    class_rate: float
    baseline_rate: float
    features_involved: List[str]
    description: Optional[str] = None


@dataclass
class InsightAnalysisResult:
    """Result of insight analysis"""
    insights: List[FeatureInsight]
    microsegments: List[Microsegment]
    baseline_rate: float
    total_count: int
    target_class: str
    summary: Dict[str, Any]


class FeatureInsightAnalyzer:
    """
    Analyzes features to discover insights with lift, support, and RIG metrics.

    Similar to SparkBeyond/DataRobot feature discovery capabilities.
    """

    def __init__(
        self,
        df: DataFrame,
        problem: Problem,
        schema_checks: SchemaChecks,
        n_bins: int = 10,
        min_support: float = 0.01,
        min_lift: float = 1.1
    ):
        """
        Initialize the analyzer.

        Args:
            df: PySpark DataFrame
            problem: Problem definition with target
            schema_checks: SchemaChecks instance
            n_bins: Number of bins for numeric features
            min_support: Minimum support threshold (fraction)
            min_lift: Minimum lift threshold
        """
        self.df = df
        self.problem = problem
        self.schema_checks = schema_checks
        self.n_bins = n_bins
        self.min_support = min_support
        self.min_lift = min_lift

        self._total_count = None
        self._baseline_rate = None
        self._target_class = None
        self._insights = []
        self._microsegments = []

    @property
    def total_count(self) -> int:
        if self._total_count is None:
            self._total_count = self.df.count()
        return self._total_count

    @property
    def baseline_rate(self) -> float:
        if self._baseline_rate is None:
            self._calculate_baseline()
        return self._baseline_rate

    @property
    def target_class(self) -> str:
        if self._target_class is None:
            self._calculate_baseline()
        return self._target_class

    def _calculate_baseline(self):
        """Calculate baseline rate for target class"""
        target_col = self.problem.target

        if self.problem.type == ProblemType.classification:
            # Use desired_result if specified, otherwise use most common class
            if self.problem.desired_result is not None:
                self._target_class = str(self.problem.desired_result)
                positive_count = self.df.filter(
                    F.col(target_col) == self.problem.desired_result
                ).count()
            else:
                # Find the minority class (more interesting for lift)
                class_counts = self.df.groupBy(target_col).count().collect()
                min_class = min(class_counts, key=lambda x: x['count'])
                self._target_class = str(min_class[target_col])
                positive_count = min_class['count']

            self._baseline_rate = positive_count / self.total_count
        else:
            # For regression, use above-median as "positive"
            median_val = self.df.approxQuantile(target_col, [0.5], 0.01)[0]
            self._target_class = f">{median_val:.2f}"
            positive_count = self.df.filter(F.col(target_col) > median_val).count()
            self._baseline_rate = positive_count / self.total_count

    def _calculate_entropy(self, p: float) -> float:
        """Calculate binary entropy"""
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)

    def _calculate_rig(self, segment_rate: float, segment_support: float) -> float:
        """
        Calculate Relative Information Gain (RIG)

        RIG measures how much information a feature condition provides
        about the target variable, normalized by the maximum possible gain.
        """
        baseline = self.baseline_rate

        # Entropy before split
        h_before = self._calculate_entropy(baseline)

        if h_before == 0:
            return 0

        # Entropy after split (weighted by support)
        h_segment = self._calculate_entropy(segment_rate)
        complement_rate = (baseline * self.total_count - segment_rate * segment_support * self.total_count) / \
                         ((1 - segment_support) * self.total_count) if segment_support < 1 else baseline
        complement_rate = max(0, min(1, complement_rate))  # Clip to valid range
        h_complement = self._calculate_entropy(complement_rate)

        h_after = segment_support * h_segment + (1 - segment_support) * h_complement

        # Information gain normalized by baseline entropy
        rig = (h_before - h_after) / h_before if h_before > 0 else 0
        return max(0, rig)

    def analyze_numeric_feature(
        self,
        feature_name: str,
        thresholds: Optional[List[float]] = None
    ) -> List[FeatureInsight]:
        """
        Analyze a numeric feature for insights.

        Args:
            feature_name: Name of the numeric column
            thresholds: Optional list of thresholds to test

        Returns:
            List of FeatureInsight objects
        """
        insights = []
        target_col = self.problem.target

        # Get feature statistics
        stats = self.df.select(
            F.min(feature_name).alias('min'),
            F.max(feature_name).alias('max'),
            F.avg(feature_name).alias('avg'),
            F.stddev(feature_name).alias('std')
        ).collect()[0]

        min_val, max_val = stats['min'], stats['max']

        if min_val is None or max_val is None or min_val == max_val:
            return insights

        # Generate thresholds if not provided
        if thresholds is None:
            # Use quantiles and evenly spaced thresholds
            quantiles = self.df.approxQuantile(feature_name,
                [i/self.n_bins for i in range(1, self.n_bins)], 0.01)
            thresholds = sorted(set(quantiles))

        # Test each threshold
        for threshold in thresholds:
            # Test >= threshold
            segment_df = self.df.filter(F.col(feature_name) >= threshold)
            segment_count = segment_df.count()

            if segment_count < self.min_support * self.total_count:
                continue

            # Calculate metrics
            if self.problem.type == ProblemType.classification:
                positive_count = segment_df.filter(
                    F.col(target_col) == self.problem.desired_result
                ).count()
            else:
                median_val = float(self._target_class.replace('>', ''))
                positive_count = segment_df.filter(F.col(target_col) > median_val).count()

            segment_rate = positive_count / segment_count if segment_count > 0 else 0
            support = segment_count / self.total_count
            lift = segment_rate / self.baseline_rate if self.baseline_rate > 0 else 0
            rig = self._calculate_rig(segment_rate, support)

            if lift >= self.min_lift:
                insights.append(FeatureInsight(
                    feature_name=feature_name,
                    condition=f"{feature_name} >= {threshold:.3f}",
                    condition_type='numeric_threshold',
                    target_class=self.target_class,
                    lift=lift,
                    support=support,
                    support_count=segment_count,
                    rig=rig,
                    class_rate=segment_rate,
                    baseline_rate=self.baseline_rate,
                    source=feature_name,
                    group="Numeric Features"
                ))

            # Test < threshold (complement)
            segment_df_lt = self.df.filter(F.col(feature_name) < threshold)
            segment_count_lt = segment_df_lt.count()

            if segment_count_lt >= self.min_support * self.total_count:
                if self.problem.type == ProblemType.classification:
                    positive_count_lt = segment_df_lt.filter(
                        F.col(target_col) == self.problem.desired_result
                    ).count()
                else:
                    positive_count_lt = segment_df_lt.filter(F.col(target_col) > median_val).count()

                segment_rate_lt = positive_count_lt / segment_count_lt if segment_count_lt > 0 else 0
                support_lt = segment_count_lt / self.total_count
                lift_lt = segment_rate_lt / self.baseline_rate if self.baseline_rate > 0 else 0
                rig_lt = self._calculate_rig(segment_rate_lt, support_lt)

                if lift_lt >= self.min_lift:
                    insights.append(FeatureInsight(
                        feature_name=feature_name,
                        condition=f"{feature_name} < {threshold:.3f}",
                        condition_type='numeric_threshold',
                        target_class=self.target_class,
                        lift=lift_lt,
                        support=support_lt,
                        support_count=segment_count_lt,
                        rig=rig_lt,
                        class_rate=segment_rate_lt,
                        baseline_rate=self.baseline_rate,
                        source=feature_name,
                        group="Numeric Features"
                    ))

        return insights

    def analyze_categorical_feature(self, feature_name: str) -> List[FeatureInsight]:
        """
        Analyze a categorical feature for insights.

        Args:
            feature_name: Name of the categorical column

        Returns:
            List of FeatureInsight objects
        """
        insights = []
        target_col = self.problem.target

        # Get value counts with target distribution
        if self.problem.type == ProblemType.classification:
            value_stats = self.df.groupBy(feature_name).agg(
                F.count('*').alias('count'),
                F.sum(F.when(F.col(target_col) == self.problem.desired_result, 1).otherwise(0)).alias('positive_count')
            ).collect()
        else:
            median_val = float(self._target_class.replace('>', ''))
            value_stats = self.df.groupBy(feature_name).agg(
                F.count('*').alias('count'),
                F.sum(F.when(F.col(target_col) > median_val, 1).otherwise(0)).alias('positive_count')
            ).collect()

        for row in value_stats:
            value = row[feature_name]
            count = row['count']
            positive_count = row['positive_count'] or 0

            if count < self.min_support * self.total_count:
                continue

            segment_rate = positive_count / count if count > 0 else 0
            support = count / self.total_count
            lift = segment_rate / self.baseline_rate if self.baseline_rate > 0 else 0
            rig = self._calculate_rig(segment_rate, support)

            if lift >= self.min_lift:
                insights.append(FeatureInsight(
                    feature_name=feature_name,
                    condition=f"{feature_name} = '{value}'",
                    condition_type='categorical',
                    target_class=self.target_class,
                    lift=lift,
                    support=support,
                    support_count=count,
                    rig=rig,
                    class_rate=segment_rate,
                    baseline_rate=self.baseline_rate,
                    source=feature_name,
                    group="Categorical Features"
                ))

        return insights

    def discover_microsegments(
        self,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100,
        progress_callback: Optional[callable] = None,
        batch_callback: Optional[callable] = None,
        batch_size: int = 10
    ) -> List[Microsegment]:
        """
        Discover microsegments (combinations of feature conditions).

        Args:
            max_depth: Maximum number of conditions to combine (2-5)
            top_n_features: Number of top features to consider for combinations
            max_microsegments: Maximum number of microsegments to return
            progress_callback: Optional callback for progress updates (progress_pct, message)
            batch_callback: Optional callback when a batch of microsegments is ready (microsegments_batch)
            batch_size: Number of microsegments to accumulate before calling batch_callback

        Returns:
            List of Microsegment objects
        """
        from itertools import combinations

        microsegments = []
        pending_batch = []

        # First, get top insights
        if not self._insights:
            self.analyze_all_features()

        # Sort by lift and take top features
        top_insights = sorted(self._insights, key=lambda x: x.lift, reverse=True)[:top_n_features]

        # Clamp max_depth to reasonable bounds
        max_depth = max(2, min(max_depth, 5))

        if len(top_insights) < 2:
            return microsegments

        target_col = self.problem.target

        # Calculate total combinations for progress tracking
        total_combinations = 0
        for depth in range(2, max_depth + 1):
            depth_limit = max(15, top_n_features - (depth - 2) * 10)
            n = min(len(top_insights), depth_limit)
            if n >= depth:
                # nCr = n! / (r! * (n-r)!)
                from math import comb
                total_combinations += min(comb(n, depth), 5000)

        combinations_processed = 0

        # Generate combinations for each depth level (2, 3, ..., max_depth)
        for depth in range(2, max_depth + 1):
            if len(top_insights) < depth:
                break

            logger.info(f"Discovering {depth}-way microsegments...")

            if progress_callback:
                progress_callback(
                    int((depth - 2) / (max_depth - 1) * 100),
                    f"Discovering {depth}-way combinations..."
                )

            # For higher depths, use fewer top insights to limit combinatorial explosion
            # depth 2: use all top_n_features
            # depth 3: use top 30
            # depth 4: use top 20
            # depth 5: use top 15
            depth_limit = max(15, top_n_features - (depth - 2) * 10)
            insights_for_depth = top_insights[:depth_limit]

            combinations_checked = 0
            max_combinations_per_depth = 5000  # Limit to avoid excessive computation

            for insight_combo in combinations(insights_for_depth, depth):
                combinations_checked += 1
                combinations_processed += 1

                if combinations_checked > max_combinations_per_depth:
                    logger.warning(f"Reached max combinations limit ({max_combinations_per_depth}) for depth {depth}")
                    break

                # Skip if any features are repeated
                feature_names = [i.feature_name for i in insight_combo]
                if len(feature_names) != len(set(feature_names)):
                    continue

                # Build combined filter
                combined_filter = self._insight_to_filter(insight_combo[0])
                for insight in insight_combo[1:]:
                    combined_filter = combined_filter & self._insight_to_filter(insight)

                combined_df = self.df.filter(combined_filter)
                combined_count = combined_df.count()

                if combined_count < self.min_support * self.total_count:
                    continue

                # Calculate metrics
                if self.problem.type == ProblemType.classification:
                    positive_count = combined_df.filter(
                        F.col(target_col) == self.problem.desired_result
                    ).count()
                else:
                    median_val = float(self._target_class.replace('>', ''))
                    positive_count = combined_df.filter(F.col(target_col) > median_val).count()

                segment_rate = positive_count / combined_count if combined_count > 0 else 0
                support = combined_count / self.total_count
                lift = segment_rate / self.baseline_rate if self.baseline_rate > 0 else 0
                rig = self._calculate_rig(segment_rate, support)

                # Check if combination is better than all individual insights
                max_individual_lift = max(i.lift for i in insight_combo)
                improvement_threshold = 1.1 + (depth - 2) * 0.05  # Higher threshold for deeper combinations

                if lift > max_individual_lift * improvement_threshold:
                    condition_str = " AND ".join(i.condition for i in insight_combo)
                    microsegment = Microsegment(
                        name=condition_str,
                        conditions=[
                            {'feature': i.feature_name, 'condition': i.condition}
                            for i in insight_combo
                        ],
                        target_class=self.target_class,
                        lift=lift,
                        support=support,
                        support_count=combined_count,
                        rig=rig,
                        class_rate=segment_rate,
                        baseline_rate=self.baseline_rate,
                        features_involved=feature_names,
                        description=f"{depth}-way combination"
                    )
                    microsegments.append(microsegment)
                    pending_batch.append(microsegment)

                    # Send batch if we have enough microsegments
                    if batch_callback and len(pending_batch) >= batch_size:
                        batch_callback(pending_batch.copy())
                        pending_batch.clear()

                # Update progress periodically
                if progress_callback and combinations_processed % 100 == 0:
                    progress_pct = min(99, int(combinations_processed / max(1, total_combinations) * 100))
                    progress_callback(
                        progress_pct,
                        f"Depth {depth}: checked {combinations_checked} combinations, found {len(microsegments)} microsegments"
                    )

            logger.info(f"Found {len([m for m in microsegments if len(m.features_involved) == depth])} microsegments at depth {depth}")

        # Send any remaining microsegments in the batch
        if batch_callback and pending_batch:
            batch_callback(pending_batch.copy())
            pending_batch.clear()

        # Sort by lift and limit results
        microsegments.sort(key=lambda x: x.lift, reverse=True)
        microsegments = microsegments[:max_microsegments]
        self._microsegments = microsegments

        if progress_callback:
            progress_callback(100, f"Complete! Found {len(microsegments)} microsegments")

        return microsegments

    def _insight_to_filter(self, insight: FeatureInsight):
        """Convert insight condition to PySpark filter expression"""
        if insight.condition_type == 'categorical':
            # Extract value from condition like "feature = 'value'"
            value = insight.condition.split("= '")[1].rstrip("'")
            return F.col(insight.feature_name) == value
        else:
            # Numeric threshold
            if '>=' in insight.condition:
                threshold = float(insight.condition.split('>= ')[1])
                return F.col(insight.feature_name) >= threshold
            elif '<' in insight.condition:
                threshold = float(insight.condition.split('< ')[1])
                return F.col(insight.feature_name) < threshold
            else:
                return F.lit(True)

    def analyze_all_features(
        self,
        include_numeric: bool = True,
        include_categorical: bool = True,
        include_engineered: bool = True
    ) -> List[FeatureInsight]:
        """
        Analyze all features in the dataset.

        Args:
            include_numeric: Include numeric features
            include_categorical: Include categorical features
            include_engineered: Include engineered features (binned, interactions, etc.)

        Returns:
            List of all FeatureInsight objects
        """
        insights = []
        target_col = self.problem.target

        if include_numeric:
            numeric_cols = self.schema_checks.numerical_cols
            numeric_cols = [c for c in numeric_cols if c != target_col]

            for col in numeric_cols:
                logger.info(f"Analyzing numeric feature: {col}")
                col_insights = self.analyze_numeric_feature(col)
                insights.extend(col_insights)

        if include_categorical:
            categorical_cols = self.schema_checks.categorical_cols
            categorical_cols = [c for c in categorical_cols if c != target_col]

            for col in categorical_cols:
                logger.info(f"Analyzing categorical feature: {col}")
                col_insights = self.analyze_categorical_feature(col)
                insights.extend(col_insights)

        # Analyze engineered features if present in the DataFrame
        if include_engineered:
            engineered_insights = self._analyze_engineered_features(target_col)
            insights.extend(engineered_insights)

        # Sort by lift
        insights.sort(key=lambda x: x.lift, reverse=True)
        self._insights = insights

        return insights

    def _analyze_engineered_features(self, target_col: str) -> List[FeatureInsight]:
        """
        Analyze engineered features (binned, interactions, transformations).

        Args:
            target_col: Target column name to exclude

        Returns:
            List of FeatureInsight objects for engineered features
        """
        insights = []
        all_cols = self.df.columns

        # Get original columns from schema_checks
        original_cols = set(
            self.schema_checks.numerical_cols +
            self.schema_checks.categorical_cols +
            self.schema_checks.boolean_cols +
            [target_col]
        )

        # Find engineered columns
        engineered_cols = [c for c in all_cols if c not in original_cols]

        if not engineered_cols:
            logger.info("No engineered features found in DataFrame")
            return insights

        logger.info(f"Found {len(engineered_cols)} engineered features to analyze")

        for col in engineered_cols:
            # Determine feature type and analyze accordingly
            col_lower = col.lower()

            # Binned features (categorical-like)
            if '_binned' in col_lower or '_bin' in col_lower:
                logger.info(f"Analyzing binned feature: {col}")
                col_insights = self._analyze_binned_feature(col)
                insights.extend(col_insights)

            # Interaction features (numeric)
            elif any(op in col_lower for op in ['_mult_', '_div_', '_add_', '_sub_']):
                logger.info(f"Analyzing interaction feature: {col}")
                col_insights = self._analyze_numeric_engineered_feature(col, "Interaction Features")
                insights.extend(col_insights)

            # Transformation features (numeric)
            elif any(transform in col_lower for transform in ['_log', '_sqrt', '_square', '_cube']):
                logger.info(f"Analyzing transformation feature: {col}")
                col_insights = self._analyze_numeric_engineered_feature(col, "Transformation Features")
                insights.extend(col_insights)

            # Ratio features (numeric)
            elif '_ratio' in col_lower or '_pct' in col_lower:
                logger.info(f"Analyzing ratio feature: {col}")
                col_insights = self._analyze_numeric_engineered_feature(col, "Ratio Features")
                insights.extend(col_insights)

            # Lag/rolling features for time series (numeric)
            elif any(ts in col_lower for ts in ['_lag_', '_rolling_', '_diff_']):
                logger.info(f"Analyzing time-series feature: {col}")
                col_insights = self._analyze_numeric_engineered_feature(col, "Time-Series Features")
                insights.extend(col_insights)

            # Datetime extracted features
            elif any(dt in col_lower for dt in ['_year', '_month', '_day', '_hour', '_dayofweek', '_quarter']):
                logger.info(f"Analyzing datetime feature: {col}")
                # These are typically low-cardinality numeric/categorical
                col_insights = self._analyze_datetime_extracted_feature(col)
                insights.extend(col_insights)

            # Other numeric engineered features
            else:
                # Check if it's numeric
                col_type = dict(self.df.dtypes).get(col, 'string')
                if col_type in ['int', 'bigint', 'float', 'double', 'decimal']:
                    logger.info(f"Analyzing other numeric engineered feature: {col}")
                    col_insights = self._analyze_numeric_engineered_feature(col, "Other Engineered Features")
                    insights.extend(col_insights)

        return insights

    def _analyze_binned_feature(self, feature_name: str) -> List[FeatureInsight]:
        """
        Analyze a binned feature as categorical.

        Args:
            feature_name: Name of the binned column

        Returns:
            List of FeatureInsight objects
        """
        insights = []
        target_col = self.problem.target

        # Get distinct bin values
        distinct_values = self.df.select(feature_name).distinct().collect()

        for row in distinct_values:
            bin_value = row[feature_name]
            if bin_value is None:
                continue

            # Filter for this bin
            segment_df = self.df.filter(F.col(feature_name) == bin_value)
            segment_count = segment_df.count()

            if segment_count < self.min_support * self.total_count:
                continue

            # Calculate metrics
            if self.problem.type == ProblemType.classification:
                positive_count = segment_df.filter(
                    F.col(target_col) == self.problem.desired_result
                ).count()
            else:
                median_val = float(self._target_class.replace('>', ''))
                positive_count = segment_df.filter(F.col(target_col) > median_val).count()

            segment_rate = positive_count / segment_count if segment_count > 0 else 0
            support = segment_count / self.total_count
            lift = segment_rate / self.baseline_rate if self.baseline_rate > 0 else 0
            rig = self._calculate_rig(segment_rate, support)

            if lift >= self.min_lift:
                insights.append(FeatureInsight(
                    feature_name=feature_name,
                    condition=f"{feature_name} = '{bin_value}'",
                    condition_type='categorical',
                    target_class=self.target_class,
                    lift=lift,
                    support=support,
                    support_count=segment_count,
                    rig=rig,
                    class_rate=segment_rate,
                    baseline_rate=self.baseline_rate,
                    source=feature_name,
                    group="Binned Features"
                ))

        return insights

    def _analyze_numeric_engineered_feature(
        self,
        feature_name: str,
        group: str = "Engineered Features"
    ) -> List[FeatureInsight]:
        """
        Analyze a numeric engineered feature.

        Args:
            feature_name: Name of the numeric column
            group: Group name for categorization

        Returns:
            List of FeatureInsight objects
        """
        insights = []
        target_col = self.problem.target

        # Check for nulls and get stats
        non_null_df = self.df.filter(F.col(feature_name).isNotNull())
        non_null_count = non_null_df.count()

        if non_null_count < self.min_support * self.total_count:
            return insights

        stats = non_null_df.select(
            F.min(feature_name).alias('min'),
            F.max(feature_name).alias('max')
        ).collect()[0]

        min_val, max_val = stats['min'], stats['max']

        if min_val is None or max_val is None or min_val == max_val:
            return insights

        # Use fewer bins for engineered features to reduce computation
        try:
            quantiles = non_null_df.approxQuantile(
                feature_name,
                [0.25, 0.5, 0.75],
                0.05
            )
            thresholds = sorted(set(quantiles))
        except Exception as e:
            logger.warning(f"Could not compute quantiles for {feature_name}: {e}")
            return insights

        # Test each threshold
        for threshold in thresholds:
            # Test >= threshold
            segment_df = self.df.filter(F.col(feature_name) >= threshold)
            segment_count = segment_df.count()

            if segment_count < self.min_support * self.total_count:
                continue

            # Calculate metrics
            if self.problem.type == ProblemType.classification:
                positive_count = segment_df.filter(
                    F.col(target_col) == self.problem.desired_result
                ).count()
            else:
                median_val = float(self._target_class.replace('>', ''))
                positive_count = segment_df.filter(F.col(target_col) > median_val).count()

            segment_rate = positive_count / segment_count if segment_count > 0 else 0
            support = segment_count / self.total_count
            lift = segment_rate / self.baseline_rate if self.baseline_rate > 0 else 0
            rig = self._calculate_rig(segment_rate, support)

            if lift >= self.min_lift:
                insights.append(FeatureInsight(
                    feature_name=feature_name,
                    condition=f"{feature_name} >= {threshold:.3f}",
                    condition_type='numeric_threshold',
                    target_class=self.target_class,
                    lift=lift,
                    support=support,
                    support_count=segment_count,
                    rig=rig,
                    class_rate=segment_rate,
                    baseline_rate=self.baseline_rate,
                    source=feature_name,
                    group=group
                ))

        return insights

    def _analyze_datetime_extracted_feature(self, feature_name: str) -> List[FeatureInsight]:
        """
        Analyze datetime extracted features (year, month, day, etc.).

        Args:
            feature_name: Name of the datetime-derived column

        Returns:
            List of FeatureInsight objects
        """
        insights = []
        target_col = self.problem.target

        # These are typically low-cardinality, so treat as categorical
        distinct_count = self.df.select(feature_name).distinct().count()

        if distinct_count <= 31:  # Likely a day, month, hour, etc.
            # Analyze as categorical
            value_stats = self.df.groupBy(feature_name).agg(
                F.count('*').alias('count'),
                F.sum(
                    F.when(
                        F.col(target_col) == self.problem.desired_result, 1
                    ).otherwise(0)
                ).alias('positive_count')
            ).collect()

            for row in value_stats:
                value = row[feature_name]
                if value is None:
                    continue

                count = row['count']
                positive_count = row['positive_count'] or 0

                if count < self.min_support * self.total_count:
                    continue

                segment_rate = positive_count / count if count > 0 else 0
                support = count / self.total_count
                lift = segment_rate / self.baseline_rate if self.baseline_rate > 0 else 0
                rig = self._calculate_rig(segment_rate, support)

                if lift >= self.min_lift:
                    insights.append(FeatureInsight(
                        feature_name=feature_name,
                        condition=f"{feature_name} = {value}",
                        condition_type='categorical',
                        target_class=self.target_class,
                        lift=lift,
                        support=support,
                        support_count=count,
                        rig=rig,
                        class_rate=segment_rate,
                        baseline_rate=self.baseline_rate,
                        source=feature_name,
                        group="Datetime Features"
                    ))
        else:
            # High cardinality, analyze as numeric
            insights = self._analyze_numeric_engineered_feature(feature_name, "Datetime Features")

        return insights

    def get_analysis_result(
        self,
        discover_microsegments: bool = True,
        max_depth: int = 2,
        include_engineered: bool = True
    ) -> InsightAnalysisResult:
        """
        Get complete analysis result.

        Args:
            discover_microsegments: Whether to discover microsegments
            max_depth: Maximum depth for microsegment discovery
            include_engineered: Whether to include engineered features

        Returns:
            InsightAnalysisResult object
        """
        if not self._insights:
            self.analyze_all_features(include_engineered=include_engineered)

        if discover_microsegments and not self._microsegments:
            self.discover_microsegments(max_depth=max_depth)

        # Generate summary
        summary = {
            'total_insights': len(self._insights),
            'total_microsegments': len(self._microsegments),
            'avg_lift': np.mean([i.lift for i in self._insights]) if self._insights else 0,
            'max_lift': max([i.lift for i in self._insights]) if self._insights else 0,
            'avg_rig': np.mean([i.rig for i in self._insights]) if self._insights else 0,
            'top_features': list(set([i.feature_name for i in self._insights[:10]])),
        }

        return InsightAnalysisResult(
            insights=self._insights,
            microsegments=self._microsegments,
            baseline_rate=self.baseline_rate,
            total_count=self.total_count,
            target_class=self.target_class,
            summary=summary
        )

    def to_dataframe(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Convert insights to a pandas DataFrame.

        Args:
            top_n: Return only top N insights (by lift)

        Returns:
            pandas DataFrame with insights
        """
        if not self._insights:
            self.analyze_all_features()

        insights = self._insights[:top_n] if top_n else self._insights

        data = []
        for insight in insights:
            data.append({
                'Feature': insight.feature_name,
                'Condition': insight.condition,
                'Class': insight.target_class,
                'Lift': f"x{insight.lift:.2f}",
                'Lift_Value': insight.lift,
                'Support': f"{insight.support*100:.1f}%",
                'Support_Count': f"{insight.support_count:,}",
                'Support_Value': insight.support,
                'RIG': f"{insight.rig:.3f}",
                'RIG_Value': insight.rig,
                'Class_Rate': f"{insight.class_rate*100:.1f}%",
                'Baseline_Rate': f"{insight.baseline_rate*100:.1f}%",
                'Group': insight.group,
                'Source': insight.source
            })

        return pd.DataFrame(data)

    def plot_lift_support_scatter(
        self,
        top_n: int = 50,
        highlight_microsegments: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Create a scatter plot of Lift vs Support (similar to the image).

        Args:
            top_n: Number of top insights to plot
            highlight_microsegments: Highlight microsegments differently
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not self._insights:
            self.analyze_all_features()

        # Prepare data
        insights = self._insights[:top_n]

        lifts = [i.lift for i in insights]
        supports = [i.support * 100 for i in insights]  # Convert to percentage
        rigs = [i.rig for i in insights]
        labels = [i.condition[:30] + '...' if len(i.condition) > 30 else i.condition for i in insights]

        # Create figure with dark theme (like the image)
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=figsize)

        # Set background color
        ax.set_facecolor('#1a1f2e')
        fig.set_facecolor('#1a1f2e')

        # Create scatter plot
        scatter = ax.scatter(
            supports, lifts,
            c=rigs,
            cmap='YlOrRd',
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5
        )

        # Add colorbar for RIG
        cbar = plt.colorbar(scatter, ax=ax, label='RIG')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        # Add optimal line (hyperbola: lift * support = constant)
        support_range = np.linspace(max(0.1, min(supports)), max(supports), 100)
        # Optimal curve where lift * support is maximized
        optimal_constant = np.percentile([l * s for l, s in zip(lifts, supports)], 75)
        optimal_lifts = optimal_constant / support_range
        ax.plot(support_range, optimal_lifts, '--', color='#00d4ff', alpha=0.5, label='Optimal line')

        # Highlight microsegments if available
        if highlight_microsegments and self._microsegments:
            micro_lifts = [m.lift for m in self._microsegments[:10]]
            micro_supports = [m.support * 100 for m in self._microsegments[:10]]
            ax.scatter(
                micro_supports, micro_lifts,
                c='#ffcc00',
                s=150,
                marker='*',
                edgecolors='white',
                linewidths=1,
                label='Microsegments',
                zorder=5
            )

        # Labels and styling
        ax.set_xlabel('Support (%)', fontsize=12, color='white')
        ax.set_ylabel('Lift', fontsize=12, color='white')
        ax.set_title(f'Feature Insights: Lift vs Support\nTarget: {self.target_class} (Baseline: {self.baseline_rate*100:.1f}%)',
                    fontsize=14, color='white')

        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(loc='upper right', facecolor='#2a2f3e', edgecolor='white')

        # Add grid
        ax.grid(True, alpha=0.2, color='white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1f2e')
            print(f"Lift-Support scatter plot saved to '{save_path}'")

        plt.close()
        return fig

    def plot_top_insights(
        self,
        top_n: int = 20,
        metric: str = 'lift',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Create a horizontal bar chart of top insights.

        Args:
            top_n: Number of insights to show
            metric: Metric to sort by ('lift', 'rig', 'support')
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not self._insights:
            self.analyze_all_features()

        # Sort by metric
        if metric == 'lift':
            sorted_insights = sorted(self._insights, key=lambda x: x.lift, reverse=True)[:top_n]
            values = [i.lift for i in sorted_insights]
            xlabel = 'Lift'
        elif metric == 'rig':
            sorted_insights = sorted(self._insights, key=lambda x: x.rig, reverse=True)[:top_n]
            values = [i.rig for i in sorted_insights]
            xlabel = 'Relative Information Gain (RIG)'
        else:
            sorted_insights = sorted(self._insights, key=lambda x: x.support, reverse=True)[:top_n]
            values = [i.support * 100 for i in sorted_insights]
            xlabel = 'Support (%)'

        labels = [f"{i.condition}" for i in sorted_insights]

        # Truncate long labels
        labels = [l[:40] + '...' if len(l) > 40 else l for l in labels]

        # Create figure with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#1a1f2e')
        fig.set_facecolor('#1a1f2e')

        # Create horizontal bar chart
        colors = plt.cm.YlOrRd([i.rig / max(i.rig for i in sorted_insights) for i in sorted_insights])

        bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels
        for i, (bar, insight) in enumerate(zip(bars, sorted_insights)):
            width = bar.get_width()
            ax.text(
                width + 0.02 * max(values),
                bar.get_y() + bar.get_height()/2,
                f'x{insight.lift:.2f} | {insight.support*100:.0f}% | RIG:{insight.rig:.3f}',
                va='center',
                fontsize=8,
                color='white'
            )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=12, color='white')
        ax.set_title(f'Top {top_n} Feature Insights by {metric.upper()}\nTarget: {self.target_class}',
                    fontsize=14, color='white')

        ax.tick_params(colors='white')
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1f2e')
            print(f"Top insights plot saved to '{save_path}'")

        plt.close()
        return fig

    def display_insights_table(self, top_n: int = 20) -> pd.DataFrame:
        """
        Display insights in a formatted table (for Jupyter/Streamlit).

        Args:
            top_n: Number of insights to display

        Returns:
            Styled pandas DataFrame
        """
        df = self.to_dataframe(top_n)

        # Select display columns
        display_cols = ['Condition', 'Class', 'Lift', 'Support', 'Support_Count', 'RIG', 'Class_Rate']

        return df[display_cols]


def quick_insight_analysis(
    df: DataFrame,
    problem: Problem,
    schema_checks: SchemaChecks,
    top_n: int = 20,
    plot: bool = True,
    include_engineered: bool = True
) -> Tuple[pd.DataFrame, Optional[InsightAnalysisResult]]:
    """
    Quick function for insight analysis.

    Args:
        df: PySpark DataFrame
        problem: Problem definition
        schema_checks: SchemaChecks instance
        top_n: Number of top insights to return
        plot: Whether to generate plots
        include_engineered: Whether to include engineered features in analysis

    Returns:
        Tuple of (insights DataFrame, full result)
    """
    analyzer = FeatureInsightAnalyzer(df, problem, schema_checks)
    result = analyzer.get_analysis_result(include_engineered=include_engineered)

    if plot:
        analyzer.plot_lift_support_scatter(save_path='insight_lift_support.png')
        analyzer.plot_top_insights(save_path='insight_top_features.png')

    return analyzer.to_dataframe(top_n), result
