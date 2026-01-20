"""
Spark Service - Manages Spark session and ML pipeline state

This service manages the ML pipeline state for each user session. It supports
both singleton mode (for backward compatibility) and session-aware mode with
Redis persistence.
"""
import sys
import os

# Add parent directory to path for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyspark.sql import SparkSession, DataFrame
from typing import Optional, Dict, Any, List, Tuple, Literal, TYPE_CHECKING
import logging

from backend.core.utils import process_col_names
from backend.core.discovery import Problem, SchemaChecks, ProblemType, ColumnTypes
from backend.core.features.auto_feature_generator import AutoFeatureGenerator
from backend.core.features.process import PreProcessVariables
from backend.core.features.feature_selector import FeatureSelector
from backend.core.profiling.data_quality import DataQualityChecker
from backend.core.profiling.ydata_profiler import DataProfiler, quick_profile
from backend.core.features.insight_analyzer import FeatureInsightAnalyzer, InsightAnalysisResult
from backend.core.models.model_comparison import ModelComparison

if TYPE_CHECKING:
    from backend.core.session import SessionManager, SessionState, FileStorage

logger = logging.getLogger(__name__)

# Global Spark session (shared across all sessions for efficiency)
_global_spark: Optional[SparkSession] = None


def get_global_spark() -> SparkSession:
    """Get or create the global Spark session."""
    global _global_spark
    if _global_spark is None:
        _global_spark = SparkSession.builder \
            .master("local[*]") \
            .appName("Spark Tune API") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        _global_spark.sparkContext.setLogLevel("ERROR")
    return _global_spark


class InsightCache:
    """Cache for insight analysis results to avoid repeated expensive calculations"""

    def __init__(self):
        self._cached_result: Optional[InsightAnalysisResult] = None
        self._cached_insights_list: Optional[List[Dict[str, Any]]] = None
        self._cached_microsegments_list: Optional[List[Dict[str, Any]]] = None
        self._cache_df_id: Optional[int] = None  # Track which DataFrame was used
        self._cache_problem_id: Optional[str] = None  # Track problem configuration
        # Track microsegment parameters
        self._cache_max_depth: Optional[int] = None
        self._cache_top_n_features: Optional[int] = None
        self._cache_max_microsegments: Optional[int] = None

    def is_valid(
        self,
        df_id: int,
        problem_id: str,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100
    ) -> bool:
        """Check if cache is valid for the given DataFrame, problem, and microsegment params"""
        return (
            self._cached_result is not None and
            self._cache_df_id == df_id and
            self._cache_problem_id == problem_id and
            self._cache_max_depth == max_depth and
            self._cache_top_n_features == top_n_features and
            self._cache_max_microsegments == max_microsegments
        )

    def set(
        self,
        result: InsightAnalysisResult,
        insights_list: List[Dict[str, Any]],
        microsegments_list: List[Dict[str, Any]],
        df_id: int,
        problem_id: str,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100
    ):
        """Store analysis results in cache"""
        self._cached_result = result
        self._cached_insights_list = insights_list
        self._cached_microsegments_list = microsegments_list
        self._cache_df_id = df_id
        self._cache_problem_id = problem_id
        self._cache_max_depth = max_depth
        self._cache_top_n_features = top_n_features
        self._cache_max_microsegments = max_microsegments

    def get_filtered(
        self,
        min_support: float,
        min_lift: float,
        top_n: int = 30
    ) -> Dict[str, Any]:
        """Get cached results filtered by min_support and min_lift"""
        if self._cached_result is None or self._cached_insights_list is None:
            raise ValueError("Cache is empty")

        # Filter insights by min_support and min_lift
        filtered_insights = [
            insight for insight in self._cached_insights_list
            if insight['Support_Value'] >= min_support and insight['Lift_Value'] >= min_lift
        ]

        # Sort by lift and take top N
        filtered_insights = sorted(
            filtered_insights,
            key=lambda x: x['Lift_Value'],
            reverse=True
        )[:top_n]

        # Filter microsegments similarly (return all, pagination handled separately)
        filtered_microsegments = [
            m for m in self._cached_microsegments_list
            if m['support'] >= min_support and m['lift'] >= min_lift
        ]

        return {
            "target_class": self._cached_result.target_class,
            "baseline_rate": self._cached_result.baseline_rate,
            "total_count": self._cached_result.total_count,
            "insights": filtered_insights,
            "microsegments": filtered_microsegments[:10],  # Default limit for backward compat
            "microsegments_total": len(filtered_microsegments),
            "summary": {
                **self._cached_result.summary,
                "filtered_insights": len(filtered_insights),
                "filtered_microsegments": len(filtered_microsegments)
            }
        }

    def get_microsegments_paginated(
        self,
        min_support: float = 0.01,
        min_lift: float = 1.1,
        page: int = 1,
        page_size: int = 20,
        sort_by: Literal['lift', 'support', 'rig', 'support_count'] = 'lift',
        sort_order: Literal['asc', 'desc'] = 'desc'
    ) -> Dict[str, Any]:
        """Get paginated microsegments with sorting"""
        if self._cached_result is None or self._cached_microsegments_list is None:
            raise ValueError("Cache is empty - run analysis first")

        # Filter microsegments
        filtered = [
            m for m in self._cached_microsegments_list
            if m['support'] >= min_support and m['lift'] >= min_lift
        ]

        # Sort
        reverse = sort_order == 'desc'
        filtered = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=reverse)

        # Paginate
        total = len(filtered)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = filtered[start_idx:end_idx]

        return {
            "microsegments": page_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "sort": {
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }

    def invalidate(self):
        """Clear the cache"""
        self._cached_result = None
        self._cached_insights_list = None
        self._cached_microsegments_list = None
        self._cache_df_id = None
        self._cache_problem_id = None
        self._cache_max_depth = None
        self._cache_top_n_features = None
        self._cache_max_microsegments = None


class SparkService:
    """
    Service to manage Spark session and ML pipeline state.

    Supports both:
    - Singleton mode (backward compatibility): Use spark_service global instance
    - Session-aware mode: Use get_spark_service() with session_id
    """

    _instance = None  # For singleton backward compatibility

    def __new__(cls, session_id: Optional[str] = None, **kwargs):
        # Only use singleton for backward compatibility when no session_id provided
        if session_id is None:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
        # For session-aware mode, always create new instance
        return super().__new__(cls)

    def __init__(
        self,
        session_id: Optional[str] = None,
        session_manager: Optional["SessionManager"] = None,
        file_storage: Optional["FileStorage"] = None,
    ):
        # Skip if already initialized (singleton mode)
        if hasattr(self, "_initialized") and self._initialized and session_id is None:
            return

        self._session_id = session_id
        self._session_manager = session_manager
        self._file_storage = file_storage

        self._df: Optional[DataFrame] = None
        self._problem: Optional[Problem] = None
        self._schema_checker: Optional[SchemaChecks] = None
        self._df_with_features: Optional[DataFrame] = None
        self._transformed_df: Optional[DataFrame] = None
        self._feature_names: Optional[List[str]] = None
        self._feature_output_col: Optional[str] = None
        self._feature_map: Optional[Dict[str, str]] = None
        self._feature_selector: Optional[FeatureSelector] = None
        self._metrics: Dict[str, Any] = {}
        self._model_comparison: Optional[ModelComparison] = None
        self._profile_report: Optional[Dict[str, Any]] = None

        # Cache for expensive insight analysis
        self._insight_cache = InsightCache()

        # Track if state has been loaded from storage
        self._state_loaded = False

        self._initialized = True

    @property
    def session_id(self) -> Optional[str]:
        """Get the session ID for this service instance."""
        return self._session_id

    @property
    def spark(self) -> SparkSession:
        """Get the global Spark session."""
        return get_global_spark()

    @property
    def df(self) -> Optional[DataFrame]:
        return self._df

    @property
    def problem(self) -> Optional[Problem]:
        return self._problem

    @property
    def schema_checker(self) -> Optional[SchemaChecks]:
        return self._schema_checker

    @property
    def feature_selector(self) -> Optional[FeatureSelector]:
        return self._feature_selector

    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Load CSV data into Spark DataFrame"""
        try:
            self._df = process_col_names(self.spark.read.options(
                header=True,
                inferSchema='True',
                delimiter=','
            ).csv(file_path))

            # Reset downstream state
            self._problem = None
            self._schema_checker = None
            self._df_with_features = None
            self._transformed_df = None
            self._feature_selector = None
            self._metrics = {}

            # Invalidate insight cache when data changes
            self._insight_cache.invalidate()

            return self.get_dataset_info()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def get_dataset_info(self) -> Optional[Dict[str, Any]]:
        """Get basic dataset information"""
        if self._df is None:
            return None

        return {
            "rows": self._df.count(),
            "columns": len(self._df.columns),
            "column_names": self._df.columns,
            "column_types": {col: dtype for col, dtype in self._df.dtypes}
        }

    def _cast_to_timestamp(self, df: DataFrame, column: str) -> Tuple[DataFrame, bool]:
        """
        Attempt to cast a string column to timestamp.

        Returns:
            Tuple of (DataFrame with cast column, success boolean)
        """
        from pyspark.sql import functions as F

        col_type = dict(df.dtypes).get(column)

        # Already a datetime type
        if col_type in ColumnTypes.datetime.value:
            return df, True

        # Not a string, can't auto-cast
        if col_type not in ColumnTypes.categorical.value:
            return df, False

        # Try common date formats
        date_formats = [
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-dd",
            "dd/MM/yyyy",
            "MM/dd/yyyy",
            "dd-MM-yyyy",
            "MM-dd-yyyy",
            "yyyy/MM/dd",
            "dd/MM/yyyy HH:mm:ss",
            "MM/dd/yyyy HH:mm:ss",
        ]

        # First try automatic parsing (works for ISO formats)
        try:
            test_df = df.withColumn(
                f"_test_{column}",
                F.to_timestamp(F.col(column))
            )
            # Check if conversion produced valid results (not all nulls)
            non_null_count = test_df.filter(F.col(f"_test_{column}").isNotNull()).count()
            original_non_null = df.filter(F.col(column).isNotNull()).count()

            if non_null_count > 0 and non_null_count >= original_non_null * 0.9:  # 90% success rate
                result_df = df.withColumn(column, F.to_timestamp(F.col(column)))
                logger.info(f"Successfully cast column '{column}' to timestamp using automatic parsing")
                return result_df, True
        except Exception as e:
            logger.debug(f"Automatic timestamp parsing failed: {e}")

        # Try explicit formats
        for fmt in date_formats:
            try:
                test_df = df.withColumn(
                    f"_test_{column}",
                    F.to_timestamp(F.col(column), fmt)
                )
                non_null_count = test_df.filter(F.col(f"_test_{column}").isNotNull()).count()
                original_non_null = df.filter(F.col(column).isNotNull()).count()

                if non_null_count > 0 and non_null_count >= original_non_null * 0.9:
                    result_df = df.withColumn(column, F.to_timestamp(F.col(column), fmt))
                    logger.info(f"Successfully cast column '{column}' to timestamp using format '{fmt}'")
                    return result_df, True
            except Exception:
                continue

        logger.warning(f"Could not auto-cast column '{column}' to timestamp")
        return df, False

    def set_problem(self, target: str, problem_type: str, desired_result: Optional[str] = None, date_column: Optional[str] = None) -> Dict[str, Any]:
        """Set problem definition and validate schema"""
        if self._df is None:
            raise ValueError("No data loaded. Load data first.")

        # Auto-cast date column if it's a string
        if date_column is not None:
            if date_column not in self._df.columns:
                raise ValueError(f"Date column '{date_column}' not found in dataset. Available columns: {self._df.columns}")

            col_type = dict(self._df.dtypes).get(date_column)
            if col_type not in ColumnTypes.datetime.value:
                logger.info(f"Date column '{date_column}' has type '{col_type}', attempting to cast to timestamp...")
                self._df, success = self._cast_to_timestamp(self._df, date_column)
                if not success:
                    raise ValueError(
                        f"Date column '{date_column}' has type '{col_type}' and could not be automatically "
                        f"converted to a timestamp. Please ensure the column contains valid date/time values "
                        f"in a recognizable format (e.g., 'yyyy-MM-dd', 'dd/MM/yyyy', etc.)"
                    )

        self._problem = Problem(
            target=target,
            type=problem_type,
            desired_result=desired_result,
            date_column=date_column
        )

        self._schema_checker = SchemaChecks(
            dataframe=self._df,
            problem=self._problem
        )

        schema_info = self._schema_checker.check()

        return {
            "categorical": [
                {
                    "name": col["col_name"],
                    "distinct_count": col["description"]["count_distinct"],
                    "count": col["description"]["count"],
                    "null_count": col["description"]["null_count"]
                }
                for col in schema_info["categorical"]
            ],
            "numerical": [
                {
                    "name": col["col_name"],
                    "mean": col["description"]["mean"],
                    "std": col["description"]["std"],
                    "min": col["description"]["min"],
                    "max": col["description"]["max"],
                    "null_count": col["description"]["null_count"]
                }
                for col in schema_info["numerical"]
            ],
            "boolean": [
                {"name": col["col_name"]}
                for col in schema_info["boolean"]
            ]
        }

    def run_quality_check(self) -> Dict[str, Any]:
        """Run data quality checks"""
        if self._df is None:
            raise ValueError("No data loaded")

        checker = DataQualityChecker(self._df)
        report = checker.run_all_checks()

        return {
            "quality_score": report.quality_score,
            "row_count": report.row_count,
            "column_count": report.column_count,
            "duplicate_count": report.duplicate_count,
            "issues": report.issues,
            "recommendations": report.recommendations
        }

    def generate_features(
        self,
        include_numerical: bool = True,
        include_interactions: bool = True,
        include_binning: bool = True,
        include_datetime: bool = True,
        include_string: bool = False
    ) -> Dict[str, Any]:
        """Generate features using AutoFeatureGenerator"""
        if self._schema_checker is None:
            raise ValueError("Problem not defined. Set problem first.")

        feature_gen = AutoFeatureGenerator(
            schema_checks=self._schema_checker,
            problem=self._problem
        )

        self._df_with_features = feature_gen.generate_all_features(
            include_numerical=include_numerical,
            include_interactions=include_interactions,
            include_binning=include_binning,
            include_datetime=include_datetime,
            include_string=include_string
        )

        # Invalidate insight cache since features changed
        self._insight_cache.invalidate()

        # Categorize generated features
        original_cols = set(self._df.columns)
        new_cols = [c for c in self._df_with_features.columns if c not in original_cols]

        categories = {
            'transformations': [f for f in new_cols if any(x in f for x in ['_log', '_sqrt', '_square', '_cube'])],
            'interactions': [f for f in new_cols if any(x in f for x in ['_mult_', '_div_', '_add_', '_sub_'])],
            'binned': [f for f in new_cols if '_binned' in f],
            'other': []
        }
        categories['other'] = [f for f in new_cols if f not in categories['transformations']
                              and f not in categories['interactions'] and f not in categories['binned']]

        return {
            "original_features": len(original_cols),
            "total_features": len(self._df_with_features.columns),
            "generated_features": len(new_cols),
            "feature_categories": {k: len(v) for k, v in categories.items()},
            "sample_features": {k: v[:10] for k, v in categories.items()}
        }

    def preprocess_features(
        self,
        imputation_strategy: str = "median",
        handle_outliers: bool = False,
        outlier_strategy: Optional[str] = None,
        outlier_threshold: float = 1.5,
        apply_scaling: bool = False,
        scaling_strategy: Optional[str] = None,
        group_rare: bool = False,
        rare_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Preprocess features for model training"""
        if self._df_with_features is None:
            raise ValueError("Features not generated. Generate features first.")

        from backend.core.features.preprocessing_enhanced import (
            EnhancedPreprocessor, PreprocessingConfig,
            ImputationStrategy as ImpStrategy, OutlierStrategy as OutStrategy, ScalingStrategy as ScaleStrategy
        )

        df_to_process = self._df_with_features
        numerical_cols = self._schema_checker.get_typed_col("numerical")
        categorical_cols = self._schema_checker.get_typed_col("categorical")

        # Build preprocessing config
        config = PreprocessingConfig(
            imputation_strategy=ImpStrategy(imputation_strategy),
            outlier_strategy=OutStrategy(outlier_strategy) if handle_outliers and outlier_strategy else None,
            outlier_threshold=outlier_threshold,
            scaling_strategy=ScaleStrategy(scaling_strategy) if apply_scaling and scaling_strategy else ScaleStrategy.NONE,
            rare_category_threshold=rare_threshold
        )

        # Apply enhanced preprocessing
        preprocessor = EnhancedPreprocessor(df_to_process, config)

        if numerical_cols:
            df_to_process = preprocessor.impute_missing_values(
                numerical_cols,
                strategy=ImpStrategy(imputation_strategy)
            )

        if handle_outliers and numerical_cols and outlier_strategy:
            preprocessor_outliers = EnhancedPreprocessor(df_to_process, config)
            df_to_process = preprocessor_outliers.handle_outliers(numerical_cols)

        if group_rare and categorical_cols:
            preprocessor_rare = EnhancedPreprocessor(df_to_process, config)
            df_to_process = preprocessor_rare.group_rare_categories(categorical_cols)

        if apply_scaling and numerical_cols and scaling_strategy:
            preprocessor_scale = EnhancedPreprocessor(df_to_process, config)
            df_to_process = preprocessor_scale.scale_features(numerical_cols)

        # Standard Spark ML preprocessing
        pre_process_variables = PreProcessVariables(
            dataframe=df_to_process,
            problem=self._problem,
            schema_checks=self._schema_checker
        )

        self._transformed_df, self._feature_names, self._feature_output_col, self._feature_map = \
            pre_process_variables.process()

        return {
            "encoded_features": len(self._feature_names),
            "total_dimensions": len(self._transformed_df.columns),
            "feature_vector_col": self._feature_output_col,
            "feature_names": self._feature_names[:20]  # Sample
        }

    def train_model(
        self,
        train_split: float = 0.8,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        num_rounds: int = 100
    ) -> Dict[str, Any]:
        """Train XGBoost model"""
        if self._transformed_df is None:
            raise ValueError("Features not preprocessed. Preprocess features first.")

        self._feature_selector = FeatureSelector(
            problem=self._problem,
            transformed_df=self._transformed_df,
            feature_names=self._feature_names,
            feature_col=self._feature_output_col,
            feature_idx_name_mapping=self._feature_map,
            train_split=train_split
        )

        self._feature_selector.train_model()

        train_metrics = self._feature_selector.evaluate(train=True)
        test_metrics = self._feature_selector.evaluate(train=False)

        self._metrics = {
            "train": train_metrics,
            "test": test_metrics
        }

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get feature importance from trained model"""
        if self._feature_selector is None or self._feature_selector.xgb_model is None:
            raise ValueError("Model not trained")

        importance_list = self._feature_selector.get_feature_importances()
        return [
            {"feature": feature, "importance": importance}
            for feature, importance in importance_list[:top_n]
        ]

    def get_probability_impact(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get probability impact analysis"""
        if self._feature_selector is None or self._feature_selector.xgb_model is None:
            raise ValueError("Model not trained")

        impact_df = self._feature_selector.get_probability_impact_summary()
        return impact_df.head(top_n).to_dict('records')

    def _get_cache_identifiers(self) -> Tuple[int, str]:
        """Get identifiers for cache validation"""
        df_to_analyze = self._df_with_features if self._df_with_features is not None else self._df
        df_id = id(df_to_analyze) if df_to_analyze is not None else 0
        problem_id = f"{self._problem.target}_{self._problem.type}_{self._problem.desired_result}" if self._problem else ""
        return df_id, problem_id

    def _compute_and_cache_insights(
        self,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100
    ) -> None:
        """Compute insights with very low thresholds and cache for later filtering"""
        if self._schema_checker is None:
            raise ValueError("Problem not defined")

        df_to_analyze = self._df_with_features if self._df_with_features is not None else self._df

        logger.info(f"Computing insight analysis (max_depth={max_depth}, top_n_features={top_n_features}, max_microsegments={max_microsegments})...")

        # Use very low thresholds to capture all potential insights
        # These will be filtered client-side later
        analyzer = FeatureInsightAnalyzer(
            df=df_to_analyze,
            problem=self._problem,
            schema_checks=self._schema_checker,
            min_support=0.001,  # Very low to capture all
            min_lift=1.0       # Include everything >= 1.0
        )

        result = analyzer.get_analysis_result(
            discover_microsegments=True,
            max_depth=max_depth
        )

        # Re-discover microsegments with the user-specified parameters
        # (get_analysis_result uses default max_depth=2, so we need to call again if different)
        if max_depth != 2 or top_n_features != 50 or max_microsegments != 100:
            analyzer.discover_microsegments(
                max_depth=max_depth,
                top_n_features=top_n_features,
                max_microsegments=max_microsegments
            )
            # Update result with new microsegments
            result = analyzer.get_analysis_result(discover_microsegments=False)

        # Convert all insights to dict format for caching
        insights_list = []
        for insight in result.insights:
            insights_list.append({
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

        # Convert microsegments to dict format
        microsegments_list = [
            {
                "name": m.name,
                "conditions": [c['condition'] for c in m.conditions],
                "lift": m.lift,
                "support": m.support,
                "support_count": m.support_count,
                "rig": m.rig,
                "depth": len(m.features_involved),
                "features_involved": m.features_involved,
                "description": m.description
            }
            for m in result.microsegments
        ]

        # Cache the results
        df_id, problem_id = self._get_cache_identifiers()
        self._insight_cache.set(
            result=result,
            insights_list=insights_list,
            microsegments_list=microsegments_list,
            df_id=df_id,
            problem_id=problem_id,
            max_depth=max_depth,
            top_n_features=top_n_features,
            max_microsegments=max_microsegments
        )

        logger.info(f"Cached {len(insights_list)} insights and {len(microsegments_list)} microsegments")

    def get_insight_analysis(
        self,
        min_support: float = 0.01,
        min_lift: float = 1.1,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100
    ) -> Dict[str, Any]:
        """
        Get feature insight analysis with lift, support, RIG.

        Uses caching to avoid expensive recalculation on every request.
        Results are computed once with low thresholds and then filtered.

        Args:
            min_support: Minimum support threshold for filtering
            min_lift: Minimum lift threshold for filtering
            max_depth: Maximum depth for microsegment combinations (2-5)
            top_n_features: Number of top features to consider for microsegments
            max_microsegments: Maximum number of microsegments to return
        """
        if self._schema_checker is None:
            raise ValueError("Problem not defined")

        df_id, problem_id = self._get_cache_identifiers()

        # Check if we need to recompute (including microsegment params)
        if not self._insight_cache.is_valid(
            df_id, problem_id, max_depth, top_n_features, max_microsegments
        ):
            self._compute_and_cache_insights(
                max_depth=max_depth,
                top_n_features=top_n_features,
                max_microsegments=max_microsegments
            )

        # Return filtered results from cache
        return self._insight_cache.get_filtered(
            min_support=min_support,
            min_lift=min_lift,
            top_n=30
        )

    def get_microsegments_paginated(
        self,
        min_support: float = 0.01,
        min_lift: float = 1.1,
        page: int = 1,
        page_size: int = 20,
        sort_by: Literal['lift', 'support', 'rig', 'support_count'] = 'lift',
        sort_order: Literal['asc', 'desc'] = 'desc'
    ) -> Dict[str, Any]:
        """
        Get paginated microsegments with sorting.

        Requires that insight analysis has been run first (cache populated).

        Args:
            min_support: Minimum support threshold for filtering
            min_lift: Minimum lift threshold for filtering
            page: Page number (1-indexed)
            page_size: Number of items per page
            sort_by: Field to sort by (lift, support, rig, support_count)
            sort_order: Sort direction (asc, desc)
        """
        if self._schema_checker is None:
            raise ValueError("Problem not defined")

        # Delegate to cache
        return self._insight_cache.get_microsegments_paginated(
            min_support=min_support,
            min_lift=min_lift,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )

    def discover_microsegments_streaming(
        self,
        min_support: float = 0.01,
        min_lift: float = 1.1,
        max_depth: int = 3,
        top_n_features: int = 50,
        max_microsegments: int = 100,
        progress_callback: Optional[callable] = None,
        batch_callback: Optional[callable] = None,
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Discover microsegments with streaming updates via callbacks.

        This method runs the analysis and provides real-time updates via callbacks,
        allowing WebSocket streaming to the frontend.

        Args:
            min_support: Minimum support threshold
            min_lift: Minimum lift threshold
            max_depth: Maximum depth for combinations
            top_n_features: Number of top features to consider
            max_microsegments: Maximum microsegments to return
            progress_callback: Callback for progress updates (progress_pct, message)
            batch_callback: Callback when batch of microsegments is ready (microsegments_list)
            batch_size: Number of microsegments per batch

        Returns:
            Dict with total count and status
        """
        if self._schema_checker is None:
            raise ValueError("Problem not defined")

        df_to_analyze = self._df_with_features if self._df_with_features is not None else self._df

        logger.info(f"Starting streaming microsegment discovery (max_depth={max_depth})...")

        # Create analyzer with low thresholds to capture all
        analyzer = FeatureInsightAnalyzer(
            df=df_to_analyze,
            problem=self._problem,
            schema_checks=self._schema_checker,
            min_support=min_support,
            min_lift=min_lift
        )

        # First analyze all features (needed before microsegment discovery)
        if progress_callback:
            progress_callback(0, "Analyzing individual features...")

        analyzer.analyze_all_features()

        if progress_callback:
            progress_callback(10, f"Found {len(analyzer._insights)} individual insights. Starting microsegment discovery...")

        # Wrapper to convert Microsegment objects to dicts for JSON serialization
        def batch_wrapper(microsegments_batch):
            if batch_callback:
                batch_dicts = [
                    {
                        "name": m.name,
                        "conditions": [c['condition'] for c in m.conditions],
                        "lift": m.lift,
                        "support": m.support,
                        "support_count": m.support_count,
                        "rig": m.rig,
                        "depth": len(m.features_involved),
                        "features_involved": m.features_involved,
                        "description": m.description
                    }
                    for m in microsegments_batch
                ]
                batch_callback(batch_dicts)

        # Progress wrapper to offset for feature analysis phase
        def progress_wrapper(pct, msg):
            if progress_callback:
                # Scale progress: 10-100 range (feature analysis takes 0-10)
                adjusted_pct = 10 + int(pct * 0.9)
                progress_callback(adjusted_pct, msg)

        # Run microsegment discovery with callbacks
        microsegments = analyzer.discover_microsegments(
            max_depth=max_depth,
            top_n_features=top_n_features,
            max_microsegments=max_microsegments,
            progress_callback=progress_wrapper,
            batch_callback=batch_wrapper,
            batch_size=batch_size
        )

        # Update cache with results
        result = analyzer.get_analysis_result(discover_microsegments=False)

        # Convert all insights to dict format for caching
        insights_list = []
        for insight in result.insights:
            insights_list.append({
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

        # Convert microsegments to dict format
        microsegments_list = [
            {
                "name": m.name,
                "conditions": [c['condition'] for c in m.conditions],
                "lift": m.lift,
                "support": m.support,
                "support_count": m.support_count,
                "rig": m.rig,
                "depth": len(m.features_involved),
                "features_involved": m.features_involved,
                "description": m.description
            }
            for m in microsegments
        ]

        # Cache the results
        df_id, problem_id = self._get_cache_identifiers()
        self._insight_cache.set(
            result=result,
            insights_list=insights_list,
            microsegments_list=microsegments_list,
            df_id=df_id,
            problem_id=problem_id,
            max_depth=max_depth,
            top_n_features=top_n_features,
            max_microsegments=max_microsegments
        )

        logger.info(f"Streaming discovery complete: {len(microsegments)} microsegments")

        return {
            "total": len(microsegments),
            "insights_count": len(insights_list),
            "status": "complete"
        }

    def get_shap_analysis(self, sample_size: int = 500) -> Dict[str, Any]:
        """Get SHAP analysis results"""
        if self._feature_selector is None or self._feature_selector.xgb_model is None:
            raise ValueError("Model not trained")

        shap_results = self._feature_selector.get_shap_analysis(
            sample_size=sample_size,
            plot=False
        )

        return {
            "feature_importance": shap_results['feature_importance'].head(20).to_dict('records')
        }

    def reset(self):
        """Reset all state"""
        self._df = None
        self._problem = None
        self._schema_checker = None
        self._df_with_features = None
        self._transformed_df = None
        self._feature_names = None
        self._feature_output_col = None
        self._feature_map = None
        self._feature_selector = None
        self._metrics = {}
        self._model_comparison: Optional[ModelComparison] = None
        self._profile_report: Optional[Dict[str, Any]] = None
        self._insight_cache.invalidate()

    def run_data_profile(self, minimal: bool = True, max_rows: int = 50000) -> Dict[str, Any]:
        """Run comprehensive data profiling using ydata-profiling"""
        if self._df is None:
            raise ValueError("No data loaded")

        try:
            # Use quick_profile for a fast summary
            profile_result = quick_profile(self._df, max_rows=max_rows)

            # Store for later reference
            self._profile_report = profile_result

            return {
                "summary": profile_result.get('summary', {}),
                "missing_values": profile_result.get('missing_values', {}),
                "alerts": profile_result.get('alerts', []),
                "sample_info": profile_result.get('sample_info', {}),
                "recommendations": profile_result.get('recommendations', []),
                "variables_count": len(profile_result.get('summary', {}).get('types', {}))
            }
        except ImportError as e:
            logger.warning(f"ydata-profiling not available, using basic profiler: {e}")
            return self._basic_profile()
        except Exception as e:
            logger.error(f"Error running data profile: {e}")
            # Fallback to basic profiling if ydata-profiling fails
            return self._basic_profile()

    def _basic_profile(self) -> Dict[str, Any]:
        """Basic profiling fallback when ydata-profiling is not available"""
        if self._df is None:
            return {}

        from pyspark.sql import functions as F
        from collections import Counter

        def to_python_type(val):
            """Convert numpy/spark types to native Python types for JSON serialization"""
            if val is None:
                return None
            if hasattr(val, 'item'):  # numpy types
                return val.item()
            return val

        columns_info = []
        for col_name, col_type in self._df.dtypes:
            col_stats = self._df.select(
                F.count(col_name).alias('count'),
                F.countDistinct(col_name).alias('distinct'),
                F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias('nulls')
            ).collect()[0]

            count_val = to_python_type(col_stats['count'])
            nulls_val = to_python_type(col_stats['nulls'])

            columns_info.append({
                'name': col_name,
                'type': col_type,
                'count': count_val,
                'distinct': to_python_type(col_stats['distinct']),
                'nulls': nulls_val,
                'missing_pct': float(nulls_val / count_val * 100) if count_val > 0 else 0.0
            })

        total_rows = int(self._df.count())
        total_cols = int(len(self._df.columns))
        missing_total = int(sum(c['nulls'] for c in columns_info))

        # Count column types properly
        type_counts = {k: int(v) for k, v in Counter(c['type'] for c in columns_info).items()}

        return {
            "summary": {
                "n_rows": total_rows,
                "n_columns": total_cols,
                "missing_cells": missing_total,
                "missing_cells_pct": float(missing_total / (total_rows * total_cols) * 100) if total_rows > 0 else 0.0,
                "types": type_counts
            },
            "missing_values": {c['name']: float(c['missing_pct']) for c in columns_info if c['missing_pct'] > 0},
            "alerts": [],
            "sample_info": {"sampled": False, "original_rows": total_rows},
            "recommendations": [],
            "columns_info": columns_info
        }

    def compare_base_vs_engineered_features(self) -> Dict[str, Any]:
        """
        Compare model performance between base features and engineered features.
        Trains XGBoost on both feature sets and returns comparison metrics.
        """
        if self._df is None:
            raise ValueError("No data loaded")
        if self._problem is None:
            raise ValueError("Problem not defined")
        if self._schema_checker is None:
            raise ValueError("Schema not validated")

        from backend.core.models.baseline_models import BaselineModels
        import time

        self._model_comparison = ModelComparison(primary_metric='accuracy')
        results = {
            "base_features": {},
            "engineered_features": {},
            "improvements": {},
            "comparison_table": []
        }

        # --- Train on BASE features ---
        logger.info("Training models on base features...")

        # Preprocess base features
        base_preprocessor = PreProcessVariables(
            dataframe=self._df,
            problem=self._problem,
            schema_checks=self._schema_checker
        )
        base_transformed, base_feature_names, base_feature_col, base_feature_map = base_preprocessor.process()

        # Split data
        base_train, base_test = base_transformed.randomSplit([0.8, 0.2], seed=42)

        # Train baseline models on base features
        base_baselines = BaselineModels(
            problem=self._problem,
            train_df=base_train,
            test_df=base_test,
            feature_col=base_feature_col,
            label_col=self._problem.target
        )

        # Train Decision Tree on base features
        start_time = time.time()
        dt_base_result = base_baselines.train_decision_tree(max_depth=5)
        dt_base_time = time.time() - start_time

        results["base_features"]["decision_tree"] = {
            "metrics": dt_base_result.metrics,
            "training_time": dt_base_time,
            "num_features": len(base_feature_names)
        }

        self._model_comparison.add_experiment(
            name="Decision Tree - Base Features",
            model_name="Decision Tree",
            feature_set="original",
            metrics=dt_base_result.metrics,
            training_time=dt_base_time,
            num_features=len(base_feature_names)
        )

        # Train XGBoost on base features
        base_selector = FeatureSelector(
            problem=self._problem,
            transformed_df=base_transformed,
            feature_names=base_feature_names,
            feature_col=base_feature_col,
            feature_idx_name_mapping=base_feature_map,
            train_split=0.8
        )

        start_time = time.time()
        base_selector.train_model()
        xgb_base_time = time.time() - start_time
        xgb_base_metrics = base_selector.evaluate(train=False)

        results["base_features"]["xgboost"] = {
            "metrics": xgb_base_metrics,
            "training_time": xgb_base_time,
            "num_features": len(base_feature_names)
        }

        self._model_comparison.add_experiment(
            name="XGBoost - Base Features",
            model_name="XGBoost",
            feature_set="original",
            metrics=xgb_base_metrics,
            training_time=xgb_base_time,
            num_features=len(base_feature_names)
        )

        # --- Train on ENGINEERED features ---
        if self._df_with_features is not None and self._transformed_df is not None:
            logger.info("Training models on engineered features...")

            # Split engineered data
            eng_train, eng_test = self._transformed_df.randomSplit([0.8, 0.2], seed=42)

            # Train baseline models on engineered features
            eng_baselines = BaselineModels(
                problem=self._problem,
                train_df=eng_train,
                test_df=eng_test,
                feature_col=self._feature_output_col,
                label_col=self._problem.target
            )

            # Train Decision Tree on engineered features
            start_time = time.time()
            dt_eng_result = eng_baselines.train_decision_tree(max_depth=5)
            dt_eng_time = time.time() - start_time

            results["engineered_features"]["decision_tree"] = {
                "metrics": dt_eng_result.metrics,
                "training_time": dt_eng_time,
                "num_features": len(self._feature_names)
            }

            self._model_comparison.add_experiment(
                name="Decision Tree - Engineered Features",
                model_name="Decision Tree",
                feature_set="engineered",
                metrics=dt_eng_result.metrics,
                training_time=dt_eng_time,
                num_features=len(self._feature_names)
            )

            # Train XGBoost on engineered features
            eng_selector = FeatureSelector(
                problem=self._problem,
                transformed_df=self._transformed_df,
                feature_names=self._feature_names,
                feature_col=self._feature_output_col,
                feature_idx_name_mapping=self._feature_map,
                train_split=0.8
            )

            start_time = time.time()
            eng_selector.train_model()
            xgb_eng_time = time.time() - start_time
            xgb_eng_metrics = eng_selector.evaluate(train=False)

            results["engineered_features"]["xgboost"] = {
                "metrics": xgb_eng_metrics,
                "training_time": xgb_eng_time,
                "num_features": len(self._feature_names)
            }

            self._model_comparison.add_experiment(
                name="XGBoost - Engineered Features",
                model_name="XGBoost",
                feature_set="engineered",
                metrics=xgb_eng_metrics,
                training_time=xgb_eng_time,
                num_features=len(self._feature_names)
            )

        # Get comparison results
        comparison_result = self._model_comparison.get_comparison()
        results["improvements"] = comparison_result.improvements
        results["comparison_table"] = comparison_result.comparison_table.to_dict('records')
        results["best_model"] = {
            "name": comparison_result.best_experiment.name,
            "metrics": comparison_result.best_experiment.metrics,
            "feature_set": comparison_result.best_experiment.feature_set
        }

        return results

    def get_model_comparison_summary(self) -> Dict[str, Any]:
        """Get summary of model comparison results"""
        if self._model_comparison is None:
            raise ValueError("No comparison has been run yet. Run compare_base_vs_engineered_features first.")

        comparison_result = self._model_comparison.get_comparison()

        return {
            "total_experiments": len(comparison_result.experiments),
            "primary_metric": comparison_result.primary_metric,
            "best_model": {
                "name": comparison_result.best_experiment.name,
                "model": comparison_result.best_experiment.model_name,
                "feature_set": comparison_result.best_experiment.feature_set,
                "score": comparison_result.best_experiment.metrics.get(comparison_result.primary_metric, 0)
            },
            "improvements": comparison_result.improvements,
            "comparison_table": comparison_result.comparison_table.to_dict('records'),
            "summary_report": self._model_comparison.get_summary_report()
        }

    # =========================================================================
    # Session Persistence Methods
    # =========================================================================

    async def save_state(self) -> bool:
        """
        Save current state to Redis and filesystem.

        Returns True if successful.
        """
        if not self._session_id or not self._session_manager or not self._file_storage:
            logger.debug("Session persistence not configured, skipping save")
            return False

        from backend.core.session import SessionState
        from backend.core.session.serializers import SchemaChecksSerializer, ModelComparisonSerializer

        try:
            # Create session state object
            state = SessionState(session_id=self._session_id)

            # Set pipeline flags
            state.has_data = self._df is not None
            state.has_problem = self._problem is not None
            state.has_features = self._df_with_features is not None
            state.has_preprocessed = self._transformed_df is not None
            state.has_model = (
                self._feature_selector is not None
                and self._feature_selector.xgb_model is not None
            )

            # Save problem definition
            if self._problem:
                state.problem = {
                    "target": self._problem.target,
                    "type": self._problem.type,
                    "desired_result": self._problem.desired_result,
                    "date_column": self._problem.date_column,
                }

            # Save schema summary
            if self._schema_checker:
                state.schema_summary = SchemaChecksSerializer.serialize_summary(
                    self._schema_checker
                )

            # Save metrics
            if self._metrics:
                state.metrics = self._metrics

            # Save profile report
            if self._profile_report:
                state.profile_report = self._profile_report

            # Save feature metadata
            state.feature_names = self._feature_names
            state.feature_output_col = self._feature_output_col
            state.feature_idx_name_mapping = self._feature_map

            # Save model comparison
            if self._model_comparison:
                state.model_comparison = ModelComparisonSerializer.serialize(
                    self._model_comparison
                )

            # Save DataFrames to filesystem
            if self._df is not None:
                state.data_file_path = self._file_storage.save_dataframe(
                    self._session_id, self._df, "original_df"
                )
                state.row_count = self._df.count()
                state.column_count = len(self._df.columns)
                state.columns = self._df.columns

            if self._df_with_features is not None:
                state.features_file_path = self._file_storage.save_dataframe(
                    self._session_id, self._df_with_features, "features_df"
                )

            if self._transformed_df is not None:
                state.transformed_file_path = self._file_storage.save_dataframe(
                    self._session_id, self._transformed_df, "transformed_df"
                )

            # Save model to filesystem
            if self._feature_selector and self._feature_selector.xgb_model:
                state.model_path = self._file_storage.save_model(
                    self._session_id, self._feature_selector.xgb_model, "xgb_model"
                )

            # Save state to Redis
            await self._session_manager.save_session(state)

            # Save insight cache to Redis
            if self._insight_cache._cached_result is not None:
                await self._session_manager.save_insights(
                    self._session_id,
                    self._insight_cache._cached_insights_list or [],
                    self._insight_cache._cached_microsegments_list or [],
                )

            logger.info(f"Saved session state for {self._session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return False

    async def load_state(self) -> bool:
        """
        Load state from Redis and filesystem.

        Returns True if state was loaded successfully.
        """
        if not self._session_id or not self._session_manager or not self._file_storage:
            logger.debug("Session persistence not configured, skipping load")
            return False

        if self._state_loaded:
            return True

        try:
            # Get session state from Redis
            state = await self._session_manager.get_session(self._session_id)
            if state is None:
                logger.debug(f"No session state found for {self._session_id}")
                return False

            # Restore problem definition
            if state.problem:
                self._problem = Problem(
                    target=state.problem.get("target"),
                    type=state.problem.get("type"),
                    desired_result=state.problem.get("desired_result"),
                    date_column=state.problem.get("date_column"),
                )

            # Restore metrics
            if state.metrics:
                self._metrics = state.metrics

            # Restore profile report
            if state.profile_report:
                self._profile_report = state.profile_report

            # Restore feature metadata
            self._feature_names = state.feature_names
            self._feature_output_col = state.feature_output_col
            self._feature_map = state.feature_idx_name_mapping

            # Restore model comparison
            if state.model_comparison:
                from backend.core.session.serializers import ModelComparisonSerializer
                self._model_comparison = ModelComparisonSerializer.deserialize(
                    state.model_comparison, ModelComparison
                )

            # Load DataFrames from filesystem (lazy - only load what we need)
            if state.data_file_path:
                self._df = self._file_storage.load_dataframe(
                    self._session_id, "original_df", self.spark
                )

            # Reconstruct schema checker if we have data and problem
            if self._df is not None and self._problem is not None:
                self._schema_checker = SchemaChecks(
                    dataframe=self._df, problem=self._problem
                )

            if state.features_file_path:
                self._df_with_features = self._file_storage.load_dataframe(
                    self._session_id, "features_df", self.spark
                )

            if state.transformed_file_path:
                self._transformed_df = self._file_storage.load_dataframe(
                    self._session_id, "transformed_df", self.spark
                )

            # Load model from filesystem
            if state.model_path and self._transformed_df is not None:
                from xgboost.spark import SparkXGBClassifierModel, SparkXGBRegressorModel

                model_class = (
                    SparkXGBClassifierModel
                    if self._problem and self._problem.type == "classification"
                    else SparkXGBRegressorModel
                )
                model = self._file_storage.load_model(
                    self._session_id, model_class, "xgb_model"
                )
                if model:
                    # Reconstruct FeatureSelector with loaded model
                    self._feature_selector = FeatureSelector(
                        problem=self._problem,
                        transformed_df=self._transformed_df,
                        feature_names=self._feature_names or [],
                        feature_col=self._feature_output_col or "features",
                        feature_idx_name_mapping=self._feature_map or {},
                        train_split=0.8,
                    )
                    self._feature_selector._xgb_model = model

            # Load insight cache from Redis
            insights, microsegments = await self._session_manager.get_insights(
                self._session_id
            )
            if insights is not None and microsegments is not None:
                # Partially restore cache for filtering
                self._insight_cache._cached_insights_list = insights
                self._insight_cache._cached_microsegments_list = microsegments

            self._state_loaded = True
            logger.info(f"Loaded session state for {self._session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return {
            "has_data": self._df is not None,
            "has_problem": self._problem is not None,
            "has_features": self._df_with_features is not None,
            "has_preprocessed": self._transformed_df is not None,
            "has_model": self._feature_selector is not None and self._feature_selector.xgb_model is not None,
            "session_id": self._session_id,
            "dataset_info": self.get_dataset_info() if self._df else None,
            "problem": {
                "target": self._problem.target,
                "type": self._problem.type,
                "desired_result": self._problem.desired_result
            } if self._problem else None
        }


# Global singleton instance (for backward compatibility)
spark_service = SparkService()


# Session-aware service factory
_session_services: Dict[str, SparkService] = {}


async def get_spark_service(
    session_id: str,
    session_manager: "SessionManager",
    file_storage: Optional["FileStorage"] = None,
) -> SparkService:
    """
    Get or create a SparkService instance for a session.

    This function maintains a cache of SparkService instances per session
    to avoid re-loading state on every request.

    Args:
        session_id: The session identifier
        session_manager: SessionManager instance for persistence
        file_storage: Optional FileStorage instance (uses session_manager's if not provided)

    Returns:
        SparkService instance for the session
    """
    global _session_services

    # Check if we already have a service for this session
    if session_id in _session_services:
        service = _session_services[session_id]
        # Ensure session manager is set (in case it was created before)
        if service._session_manager is None:
            service._session_manager = session_manager
            service._file_storage = file_storage or session_manager.file_storage
        return service

    # Create new service for this session
    fs = file_storage or session_manager.file_storage
    service = SparkService(
        session_id=session_id,
        session_manager=session_manager,
        file_storage=fs,
    )

    # Try to load existing state
    await service.load_state()

    # Cache the service
    _session_services[session_id] = service

    return service


def clear_session_service(session_id: str):
    """Remove a session's service from the cache."""
    global _session_services
    if session_id in _session_services:
        del _session_services[session_id]
