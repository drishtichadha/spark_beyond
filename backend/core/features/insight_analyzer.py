"""
Feature Insight Analyzer

Provides lift, support, and Relative Information Gain (RIG) analysis
for feature discovery and microsegment identification.

Optimized for production use with:
- Vectorized aggregations instead of filter-count loops
- Batch quantile computation across all features
- Single-pass microsegment evaluation using conditional aggregation
- Intelligent DataFrame caching with lifecycle management
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType, DecimalType
from pyspark.storagelevel import StorageLevel
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from backend.core.discovery import Problem, ProblemType, SchemaChecks

logger = logging.getLogger(__name__)

# Numeric types for schema detection
NUMERIC_TYPES = (DoubleType, IntegerType, LongType, FloatType, DecimalType)


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

    Performance optimizations:
    - Uses vectorized aggregations instead of filter-count loops
    - Computes all numeric thresholds in a single pass per feature type
    - Microsegment discovery uses conditional aggregation, not combinatorial filters
    - Intelligent caching with automatic cleanup
    """

    def __init__(
        self,
        df: DataFrame,
        problem: Problem,
        schema_checks: SchemaChecks,
        n_bins: int = 10,
        min_support: float = 0.01,
        min_lift: float = 1.1,
        cache_df: bool = True
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
            cache_df: Whether to cache the DataFrame for repeated operations
        """
        self.df = df
        self.problem = problem
        self.schema_checks = schema_checks
        self.n_bins = n_bins
        self.min_support = min_support
        self.min_lift = min_lift
        self._cache_df = cache_df
        self._df_cached = False

        self._total_count = None
        self._baseline_rate = None
        self._target_class = None
        self._positive_count = None
        self._insights = []
        self._microsegments = []

        # Pre-computed data for optimization
        self._quantile_cache: Dict[str, List[float]] = {}
        self._categorical_stats_cache: Dict[str, List[Dict]] = {}
        self._target_col_expr = None  # Cached target expression

    def _ensure_cached(self) -> None:
        """Cache DataFrame if caching is enabled and not already cached."""
        if self._cache_df and not self._df_cached:
            self.df = self.df.cache()
            self._df_cached = True
            logger.debug("DataFrame cached for insight analysis")

    def _uncache(self) -> None:
        """Uncache DataFrame to free memory."""
        if self._df_cached:
            self.df.unpersist()
            self._df_cached = False
            logger.debug("DataFrame uncached")

    @contextmanager
    def _cached_context(self):
        """Context manager for automatic cache management."""
        try:
            self._ensure_cached()
            yield
        finally:
            pass  # Keep cache for subsequent operations; call cleanup() explicitly

    def cleanup(self) -> None:
        """Release all cached resources. Call when done with analysis."""
        self._uncache()
        self._quantile_cache.clear()
        self._categorical_stats_cache.clear()
        logger.info("Insight analyzer resources cleaned up")

    @property
    def total_count(self) -> int:
        if self._total_count is None:
            self._ensure_cached()
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

    def _get_positive_indicator(self) -> F.Column:
        """
        Get a column expression that evaluates to 1 for positive class, 0 otherwise.
        Cached for reuse across multiple aggregations.
        """
        if self._target_col_expr is not None:
            return self._target_col_expr

        target_col = self.problem.target

        if self.problem.type == ProblemType.classification:
            # Ensure baseline is calculated first to set target_class
            _ = self.target_class
            self._target_col_expr = F.when(
                F.col(target_col) == self.problem.desired_result, 1
            ).otherwise(0)
        else:
            # For regression, use above-median as "positive"
            median_val = float(self._target_class.replace('>', ''))
            self._target_col_expr = F.when(
                F.col(target_col) > median_val, 1
            ).otherwise(0)

        return self._target_col_expr

    def _calculate_baseline(self):
        """
        Calculate baseline rate for target class.
        Optimized to compute total_count and positive_count in a single aggregation.
        """
        self._ensure_cached()
        target_col = self.problem.target

        if self.problem.type == ProblemType.classification:
            # Use desired_result if specified, otherwise use minority class
            if self.problem.desired_result is not None:
                self._target_class = str(self.problem.desired_result)
                # Single aggregation for both counts
                result = self.df.agg(
                    F.count('*').alias('total'),
                    F.sum(F.when(F.col(target_col) == self.problem.desired_result, 1).otherwise(0)).alias('positive')
                ).collect()[0]
                self._total_count = result['total']
                self._positive_count = result['positive']
            else:
                # Find the minority class (more interesting for lift)
                class_counts = self.df.groupBy(target_col).count().collect()
                min_class = min(class_counts, key=lambda x: x['count'])
                self._target_class = str(min_class[target_col])
                self._positive_count = min_class['count']
                self._total_count = sum(row['count'] for row in class_counts)

            self._baseline_rate = self._positive_count / self._total_count
        else:
            # For regression, use above-median as "positive"
            median_val = self.df.approxQuantile(target_col, [0.5], 0.01)[0]
            self._target_class = f">{median_val:.2f}"
            # Single aggregation for both counts
            result = self.df.agg(
                F.count('*').alias('total'),
                F.sum(F.when(F.col(target_col) > median_val, 1).otherwise(0)).alias('positive')
            ).collect()[0]
            self._total_count = result['total']
            self._positive_count = result['positive']
            self._baseline_rate = self._positive_count / self._total_count

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

    def _compute_quantiles_batch(self, feature_names: List[str]) -> Dict[str, List[float]]:
        """
        Compute quantiles for multiple numeric features in a batch.
        Uses a single pass over the data with percentile_approx.

        Args:
            feature_names: List of numeric column names

        Returns:
            Dict mapping feature name to list of quantile values
        """
        if not feature_names:
            return {}

        # Check cache first
        uncached_features = [f for f in feature_names if f not in self._quantile_cache]

        if uncached_features:
            self._ensure_cached()
            percentiles = [i / self.n_bins for i in range(1, self.n_bins)]

            # Build aggregation expressions for all features at once
            agg_exprs = []
            for feature in uncached_features:
                for i, p in enumerate(percentiles):
                    agg_exprs.append(
                        F.percentile_approx(F.col(feature), p, 10000).alias(f"{feature}_p{i}")
                    )

            # Single aggregation for all quantiles
            if agg_exprs:
                result = self.df.agg(*agg_exprs).collect()[0]

                # Parse results back into per-feature quantile lists
                for feature in uncached_features:
                    quantiles = []
                    for i in range(len(percentiles)):
                        val = result[f"{feature}_p{i}"]
                        if val is not None:
                            quantiles.append(val)
                    self._quantile_cache[feature] = sorted(set(quantiles))

        return {f: self._quantile_cache.get(f, []) for f in feature_names}

    def analyze_numeric_feature(
        self,
        feature_name: str,
        thresholds: Optional[List[float]] = None,
        group: str = "Numeric Features"
    ) -> List[FeatureInsight]:
        """
        Analyze a numeric feature for insights using vectorized aggregation.

        Instead of running separate filter().count() for each threshold,
        this method computes all threshold statistics in a SINGLE aggregation
        using conditional sums.

        Args:
            feature_name: Name of the numeric column
            thresholds: Optional list of thresholds to test
            group: Group name for categorization

        Returns:
            List of FeatureInsight objects
        """
        self._ensure_cached()
        insights = []

        # Get feature statistics and validate
        stats = self.df.select(
            F.min(feature_name).alias('min'),
            F.max(feature_name).alias('max')
        ).collect()[0]

        min_val, max_val = stats['min'], stats['max']

        if min_val is None or max_val is None or min_val == max_val:
            return insights

        # Generate thresholds if not provided (use cached quantiles)
        if thresholds is None:
            if feature_name in self._quantile_cache:
                thresholds = self._quantile_cache[feature_name]
            else:
                quantile_result = self._compute_quantiles_batch([feature_name])
                thresholds = quantile_result.get(feature_name, [])

        if not thresholds:
            return insights

        # OPTIMIZATION: Compute all threshold stats in a SINGLE aggregation
        # Instead of N separate filter().count() calls, use conditional sums
        positive_indicator = self._get_positive_indicator()
        total_count = self.total_count
        baseline = self.baseline_rate
        min_support_count = self.min_support * total_count

        agg_exprs = []
        for i, threshold in enumerate(thresholds):
            # Count for >= threshold
            agg_exprs.append(
                F.sum(F.when(F.col(feature_name) >= threshold, 1).otherwise(0)).alias(f"cnt_gte_{i}")
            )
            # Positive count for >= threshold
            agg_exprs.append(
                F.sum(F.when(F.col(feature_name) >= threshold, positive_indicator).otherwise(0)).alias(f"pos_gte_{i}")
            )
            # Count for < threshold
            agg_exprs.append(
                F.sum(F.when(F.col(feature_name) < threshold, 1).otherwise(0)).alias(f"cnt_lt_{i}")
            )
            # Positive count for < threshold
            agg_exprs.append(
                F.sum(F.when(F.col(feature_name) < threshold, positive_indicator).otherwise(0)).alias(f"pos_lt_{i}")
            )

        # SINGLE aggregation call for all thresholds
        result = self.df.agg(*agg_exprs).collect()[0]

        # Process results
        for i, threshold in enumerate(thresholds):
            # Process >= threshold
            segment_count = result[f"cnt_gte_{i}"] or 0
            if segment_count >= min_support_count:
                positive_count = result[f"pos_gte_{i}"] or 0
                segment_rate = positive_count / segment_count if segment_count > 0 else 0
                support = segment_count / total_count
                lift = segment_rate / baseline if baseline > 0 else 0
                rig = self._calculate_rig(segment_rate, support)

                if lift >= self.min_lift:
                    insights.append(FeatureInsight(
                        feature_name=feature_name,
                        condition=f"{feature_name} >= {threshold:.3f}",
                        condition_type='numeric_threshold',
                        target_class=self.target_class,
                        lift=lift,
                        support=support,
                        support_count=int(segment_count),
                        rig=rig,
                        class_rate=segment_rate,
                        baseline_rate=baseline,
                        source=feature_name,
                        group=group
                    ))

            # Process < threshold
            segment_count_lt = result[f"cnt_lt_{i}"] or 0
            if segment_count_lt >= min_support_count:
                positive_count_lt = result[f"pos_lt_{i}"] or 0
                segment_rate_lt = positive_count_lt / segment_count_lt if segment_count_lt > 0 else 0
                support_lt = segment_count_lt / total_count
                lift_lt = segment_rate_lt / baseline if baseline > 0 else 0
                rig_lt = self._calculate_rig(segment_rate_lt, support_lt)

                if lift_lt >= self.min_lift:
                    insights.append(FeatureInsight(
                        feature_name=feature_name,
                        condition=f"{feature_name} < {threshold:.3f}",
                        condition_type='numeric_threshold',
                        target_class=self.target_class,
                        lift=lift_lt,
                        support=support_lt,
                        support_count=int(segment_count_lt),
                        rig=rig_lt,
                        class_rate=segment_rate_lt,
                        baseline_rate=baseline,
                        source=feature_name,
                        group=group
                    ))

        return insights

    def analyze_numeric_features_batch(
        self,
        feature_names: List[str],
        group: str = "Numeric Features"
    ) -> List[FeatureInsight]:
        """
        Analyze multiple numeric features in an optimized batch.

        This method pre-computes quantiles for all features in one pass,
        then analyzes each feature. For very large feature sets, this is
        significantly faster than analyzing features one-by-one.

        Args:
            feature_names: List of numeric column names
            group: Group name for categorization

        Returns:
            List of FeatureInsight objects for all features
        """
        if not feature_names:
            return []

        self._ensure_cached()

        # Pre-compute all quantiles in one pass
        self._compute_quantiles_batch(feature_names)

        # Analyze each feature (uses cached quantiles)
        insights = []
        for feature in feature_names:
            try:
                feature_insights = self.analyze_numeric_feature(feature, group=group)
                insights.extend(feature_insights)
            except Exception as e:
                logger.warning(f"Error analyzing numeric feature {feature}: {e}")

        return insights

    def analyze_categorical_feature(
        self,
        feature_name: str,
        group: str = "Categorical Features"
    ) -> List[FeatureInsight]:
        """
        Analyze a categorical feature for insights.

        Already optimized - uses single groupBy aggregation for all category values.
        Added caching for repeated analysis.

        Args:
            feature_name: Name of the categorical column
            group: Group name for categorization

        Returns:
            List of FeatureInsight objects
        """
        self._ensure_cached()
        insights = []

        # Check cache first
        if feature_name in self._categorical_stats_cache:
            value_stats = self._categorical_stats_cache[feature_name]
        else:
            # Single groupBy aggregation for all category values
            positive_indicator = self._get_positive_indicator()
            value_stats = self.df.groupBy(feature_name).agg(
                F.count('*').alias('count'),
                F.sum(positive_indicator).alias('positive_count')
            ).collect()
            # Cache for potential reuse
            self._categorical_stats_cache[feature_name] = value_stats

        total_count = self.total_count
        baseline = self.baseline_rate
        min_support_count = self.min_support * total_count

        for row in value_stats:
            value = row[feature_name]
            count = row['count']
            positive_count = row['positive_count'] or 0

            if count < min_support_count:
                continue

            segment_rate = positive_count / count if count > 0 else 0
            support = count / total_count
            lift = segment_rate / baseline if baseline > 0 else 0
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
                    baseline_rate=baseline,
                    source=feature_name,
                    group=group
                ))

        return insights

    def analyze_categorical_features_batch(
        self,
        feature_names: List[str]
    ) -> List[FeatureInsight]:
        """
        Analyze multiple categorical features.

        Args:
            feature_names: List of categorical column names

        Returns:
            List of FeatureInsight objects for all features
        """
        insights = []
        for feature in feature_names:
            try:
                feature_insights = self.analyze_categorical_feature(feature)
                insights.extend(feature_insights)
            except Exception as e:
                logger.warning(f"Error analyzing categorical feature {feature}: {e}")
        return insights

    def discover_microsegments(
        self,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100,
        progress_callback: Optional[callable] = None,
        batch_callback: Optional[callable] = None,
        batch_size: int = 10,
        combinations_per_batch: int = 50
    ) -> List[Microsegment]:
        """
        Discover microsegments (combinations of feature conditions).

        OPTIMIZED: Uses batched conditional aggregation instead of individual
        filter().count() calls. Evaluates multiple combinations in each Spark job.

        Args:
            max_depth: Maximum number of conditions to combine (2-5)
            top_n_features: Number of top features to consider for combinations
            max_microsegments: Maximum number of microsegments to return
            progress_callback: Optional callback for progress updates (progress_pct, message)
            batch_callback: Optional callback when a batch of microsegments is ready
            batch_size: Number of microsegments to accumulate before calling batch_callback
            combinations_per_batch: Number of combinations to evaluate per Spark job

        Returns:
            List of Microsegment objects
        """
        from itertools import combinations
        from math import comb

        self._ensure_cached()
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

        positive_indicator = self._get_positive_indicator()
        total_count = self.total_count
        baseline = self.baseline_rate
        min_support_count = self.min_support * total_count

        # Calculate total combinations for progress tracking
        total_combinations = 0
        for depth in range(2, max_depth + 1):
            depth_limit = max(15, top_n_features - (depth - 2) * 10)
            n = min(len(top_insights), depth_limit)
            if n >= depth:
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
            depth_limit = max(15, top_n_features - (depth - 2) * 10)
            insights_for_depth = top_insights[:depth_limit]

            max_combinations_per_depth = 5000

            # Generate all valid combinations for this depth
            valid_combos = []
            for insight_combo in combinations(insights_for_depth, depth):
                if len(valid_combos) >= max_combinations_per_depth:
                    logger.warning(f"Reached max combinations limit ({max_combinations_per_depth}) for depth {depth}")
                    break

                # Skip if any features are repeated
                feature_names = [i.feature_name for i in insight_combo]
                if len(feature_names) != len(set(feature_names)):
                    continue

                valid_combos.append(insight_combo)

            # OPTIMIZATION: Process combinations in batches using single aggregation
            for batch_start in range(0, len(valid_combos), combinations_per_batch):
                batch_combos = valid_combos[batch_start:batch_start + combinations_per_batch]

                # Build aggregation expressions for all combinations in this batch
                agg_exprs = []
                combo_filters = []

                for i, insight_combo in enumerate(batch_combos):
                    # Build combined filter expression
                    combined_filter = self._insight_to_filter(insight_combo[0])
                    for insight in insight_combo[1:]:
                        combined_filter = combined_filter & self._insight_to_filter(insight)

                    combo_filters.append(combined_filter)

                    # Count matching rows
                    agg_exprs.append(
                        F.sum(F.when(combined_filter, 1).otherwise(0)).alias(f"cnt_{i}")
                    )
                    # Count positive in matching rows
                    agg_exprs.append(
                        F.sum(F.when(combined_filter, positive_indicator).otherwise(0)).alias(f"pos_{i}")
                    )

                # SINGLE aggregation for entire batch
                result = self.df.agg(*agg_exprs).collect()[0]

                # Process results for each combination
                for i, insight_combo in enumerate(batch_combos):
                    combinations_processed += 1

                    segment_count = result[f"cnt_{i}"] or 0
                    if segment_count < min_support_count:
                        continue

                    positive_count = result[f"pos_{i}"] or 0
                    segment_rate = positive_count / segment_count if segment_count > 0 else 0
                    support = segment_count / total_count
                    lift = segment_rate / baseline if baseline > 0 else 0
                    rig = self._calculate_rig(segment_rate, support)

                    # Check if combination is better than all individual insights
                    max_individual_lift = max(ins.lift for ins in insight_combo)
                    improvement_threshold = 1.1 + (depth - 2) * 0.05

                    if lift > max_individual_lift * improvement_threshold:
                        feature_names = [ins.feature_name for ins in insight_combo]
                        condition_str = " AND ".join(ins.condition for ins in insight_combo)
                        microsegment = Microsegment(
                            name=condition_str,
                            conditions=[
                                {'feature': ins.feature_name, 'condition': ins.condition}
                                for ins in insight_combo
                            ],
                            target_class=self.target_class,
                            lift=lift,
                            support=support,
                            support_count=int(segment_count),
                            rig=rig,
                            class_rate=segment_rate,
                            baseline_rate=baseline,
                            features_involved=feature_names,
                            description=f"{depth}-way combination"
                        )
                        microsegments.append(microsegment)
                        pending_batch.append(microsegment)

                        # Send batch if we have enough microsegments
                        if batch_callback and len(pending_batch) >= batch_size:
                            batch_callback(pending_batch.copy())
                            pending_batch.clear()

                # Update progress after each aggregation batch
                if progress_callback:
                    progress_pct = min(99, int(combinations_processed / max(1, total_combinations) * 100))
                    progress_callback(
                        progress_pct,
                        f"Depth {depth}: checked {combinations_processed} combinations, found {len(microsegments)} microsegments"
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

        OPTIMIZED: Uses batch processing for numeric features (pre-computes
        all quantiles in a single pass) and parallel feature analysis.

        Args:
            include_numeric: Include numeric features
            include_categorical: Include categorical features
            include_engineered: Include engineered features (binned, interactions, etc.)

        Returns:
            List of all FeatureInsight objects
        """
        self._ensure_cached()
        insights = []
        target_col = self.problem.target

        if include_numeric:
            numeric_cols = self.schema_checks.numerical_cols
            numeric_cols = [c for c in numeric_cols if c != target_col]

            if numeric_cols:
                logger.info(f"Analyzing {len(numeric_cols)} numeric features (batch mode)...")
                # OPTIMIZATION: Batch analysis pre-computes all quantiles in one pass
                numeric_insights = self.analyze_numeric_features_batch(numeric_cols)
                insights.extend(numeric_insights)
                logger.info(f"Found {len(numeric_insights)} insights from numeric features")

        if include_categorical:
            categorical_cols = self.schema_checks.categorical_cols
            categorical_cols = [c for c in categorical_cols if c != target_col]

            if categorical_cols:
                logger.info(f"Analyzing {len(categorical_cols)} categorical features...")
                categorical_insights = self.analyze_categorical_features_batch(categorical_cols)
                insights.extend(categorical_insights)
                logger.info(f"Found {len(categorical_insights)} insights from categorical features")

        # Analyze engineered features if present in the DataFrame
        if include_engineered:
            engineered_insights = self._analyze_engineered_features(target_col)
            insights.extend(engineered_insights)
            logger.info(f"Found {len(engineered_insights)} insights from engineered features")

        # Sort by lift
        insights.sort(key=lambda x: x.lift, reverse=True)
        self._insights = insights

        logger.info(f"Total insights discovered: {len(insights)}")
        return insights

    def _analyze_engineered_features(self, target_col: str) -> List[FeatureInsight]:
        """
        Analyze engineered features (binned, interactions, transformations).

        OPTIMIZED: Groups engineered features by type and processes them in batches.

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

        # OPTIMIZATION: Group features by type for batch processing
        binned_cols = []
        numeric_by_group: Dict[str, List[str]] = {
            "Interaction Features": [],
            "Transformation Features": [],
            "Ratio Features": [],
            "Time-Series Features": [],
            "Datetime Features": [],
            "Other Engineered Features": []
        }

        # Get column types once
        col_types = dict(self.df.dtypes)

        for col in engineered_cols:
            col_lower = col.lower()

            # Binned features (categorical-like)
            if '_binned' in col_lower or '_bin' in col_lower:
                binned_cols.append(col)

            # Interaction features (numeric)
            elif any(op in col_lower for op in ['_mult_', '_div_', '_add_', '_sub_']):
                numeric_by_group["Interaction Features"].append(col)

            # Transformation features (numeric)
            elif any(transform in col_lower for transform in ['_log', '_sqrt', '_square', '_cube']):
                numeric_by_group["Transformation Features"].append(col)

            # Ratio features (numeric)
            elif '_ratio' in col_lower or '_pct' in col_lower:
                numeric_by_group["Ratio Features"].append(col)

            # Lag/rolling features for time series (numeric)
            elif any(ts in col_lower for ts in ['_lag_', '_rolling_', '_diff_']):
                numeric_by_group["Time-Series Features"].append(col)

            # Datetime extracted features
            elif any(dt in col_lower for dt in ['_year', '_month', '_day', '_hour', '_dayofweek', '_quarter']):
                numeric_by_group["Datetime Features"].append(col)

            # Other numeric engineered features
            else:
                col_type = col_types.get(col, 'string')
                if col_type in ['int', 'bigint', 'float', 'double', 'decimal']:
                    numeric_by_group["Other Engineered Features"].append(col)

        # Process binned features (treat as categorical)
        if binned_cols:
            logger.info(f"Analyzing {len(binned_cols)} binned features...")
            for col in binned_cols:
                col_insights = self.analyze_categorical_feature(col, group="Binned Features")
                insights.extend(col_insights)

        # Process numeric features by group (using batch quantile computation)
        for group_name, cols in numeric_by_group.items():
            if cols:
                logger.info(f"Analyzing {len(cols)} {group_name}...")
                # Pre-compute quantiles for this group
                self._compute_quantiles_batch(cols)
                for col in cols:
                    try:
                        col_insights = self.analyze_numeric_feature(col, group=group_name)
                        insights.extend(col_insights)
                    except Exception as e:
                        logger.warning(f"Error analyzing engineered feature {col}: {e}")

        return insights

    # NOTE: _analyze_binned_feature and _analyze_numeric_engineered_feature
    # have been replaced by the optimized analyze_categorical_feature and
    # analyze_numeric_feature methods which use vectorized aggregations.

    def _analyze_datetime_extracted_feature(self, feature_name: str) -> List[FeatureInsight]:
        """
        Analyze datetime extracted features (year, month, day, etc.).

        OPTIMIZED: Uses existing optimized methods for categorical/numeric analysis.

        Args:
            feature_name: Name of the datetime-derived column

        Returns:
            List of FeatureInsight objects
        """
        # These are typically low-cardinality, so treat as categorical
        distinct_count = self.df.select(feature_name).distinct().count()

        if distinct_count <= 31:  # Likely a day, month, hour, etc.
            # Use optimized categorical analysis
            return self.analyze_categorical_feature(feature_name, group="Datetime Features")
        else:
            # High cardinality, use optimized numeric analysis
            return self.analyze_numeric_feature(feature_name, group="Datetime Features")

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
