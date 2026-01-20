"""
Serializers for complex objects that need to be stored in Redis.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InsightCacheSerializer:
    """
    Serializer for InsightCache objects.

    Converts between InsightCache instances and JSON-compatible dictionaries
    that can be stored in Redis.
    """

    @staticmethod
    def serialize(cache) -> Dict[str, Any]:
        """
        Serialize an InsightCache to a JSON-compatible dict.

        Args:
            cache: InsightCache instance

        Returns:
            Dictionary that can be JSON-serialized
        """
        if cache is None or cache._cached_result is None:
            return {"empty": True}

        try:
            return {
                "empty": False,
                "insights_list": cache._cached_insights_list or [],
                "microsegments_list": cache._cached_microsegments_list or [],
                "df_id": cache._cache_df_id,
                "problem_id": cache._cache_problem_id,
                "max_depth": cache._cache_max_depth,
                "top_n_features": cache._cache_top_n_features,
                "max_microsegments": cache._cache_max_microsegments,
                "result_summary": {
                    "target_class": cache._cached_result.target_class,
                    "baseline_rate": cache._cached_result.baseline_rate,
                    "total_count": cache._cached_result.total_count,
                    "summary": cache._cached_result.summary
                    if hasattr(cache._cached_result, "summary")
                    else {},
                },
            }
        except Exception as e:
            logger.error(f"Failed to serialize InsightCache: {e}")
            return {"empty": True, "error": str(e)}

    @staticmethod
    def deserialize(data: Dict[str, Any], insight_cache_class) -> Any:
        """
        Deserialize an InsightCache from a JSON dict.

        Args:
            data: Dictionary from JSON
            insight_cache_class: The InsightCache class to instantiate

        Returns:
            Reconstructed InsightCache instance
        """
        cache = insight_cache_class()

        if data.get("empty", True):
            return cache

        try:
            # Reconstruct minimal InsightAnalysisResult for cache validation
            from backend.core.features.insight_analyzer import InsightAnalysisResult

            result_summary = data.get("result_summary", {})
            result = InsightAnalysisResult(
                insights=[],  # Full insights stored separately
                microsegments=[],  # Full microsegments stored separately
                baseline_rate=result_summary.get("baseline_rate", 0.0),
                total_count=result_summary.get("total_count", 0),
                target_class=result_summary.get("target_class", ""),
                summary=result_summary.get("summary", {}),
            )

            cache._cached_result = result
            cache._cached_insights_list = data.get("insights_list", [])
            cache._cached_microsegments_list = data.get("microsegments_list", [])
            cache._cache_df_id = data.get("df_id")
            cache._cache_problem_id = data.get("problem_id")
            cache._cache_max_depth = data.get("max_depth", 3)
            cache._cache_top_n_features = data.get("top_n_features", 50)
            cache._cache_max_microsegments = data.get("max_microsegments", 100)

            return cache
        except Exception as e:
            logger.error(f"Failed to deserialize InsightCache: {e}")
            return insight_cache_class()


class SchemaChecksSerializer:
    """
    Serializer for SchemaChecks summary information.

    Extracts key information from SchemaChecks for session state storage.
    """

    @staticmethod
    def serialize_summary(schema_checks) -> Dict[str, Any]:
        """
        Extract summary information from SchemaChecks.

        Args:
            schema_checks: SchemaChecks instance

        Returns:
            Dictionary with schema summary
        """
        if schema_checks is None:
            return {}

        try:
            return {
                "categorical_vars": list(schema_checks.categorical_vars or []),
                "numerical_vars": list(schema_checks.numerical_vars or []),
                "datetime_vars": list(schema_checks.datetime_vars or []),
                "all_columns": list(schema_checks.all_columns or []),
                "target_column": schema_checks.target_column
                if hasattr(schema_checks, "target_column")
                else None,
            }
        except Exception as e:
            logger.error(f"Failed to serialize SchemaChecks: {e}")
            return {}


class ModelComparisonSerializer:
    """
    Serializer for ModelComparison tracking data.
    """

    @staticmethod
    def serialize(comparison) -> Dict[str, Any]:
        """
        Serialize ModelComparison to dict.

        Args:
            comparison: ModelComparison instance

        Returns:
            Dictionary that can be JSON-serialized
        """
        if comparison is None:
            return {}

        try:
            return {
                "primary_metric": comparison.primary_metric,
                "experiments": [
                    {
                        "name": exp.name,
                        "model_name": exp.model_name,
                        "feature_set": exp.feature_set,
                        "metrics": exp.metrics,
                        "training_time": exp.training_time,
                        "feature_count": exp.feature_count,
                        "timestamp": exp.timestamp.isoformat()
                        if hasattr(exp, "timestamp") and exp.timestamp
                        else None,
                    }
                    for exp in (comparison.experiments or [])
                ],
            }
        except Exception as e:
            logger.error(f"Failed to serialize ModelComparison: {e}")
            return {}

    @staticmethod
    def deserialize(data: Dict[str, Any], model_comparison_class) -> Any:
        """
        Deserialize ModelComparison from dict.

        Args:
            data: Dictionary from JSON
            model_comparison_class: The ModelComparison class

        Returns:
            Reconstructed ModelComparison instance
        """
        if not data:
            return None

        try:
            comparison = model_comparison_class(
                primary_metric=data.get("primary_metric", "accuracy")
            )

            for exp_data in data.get("experiments", []):
                comparison.add_experiment(
                    name=exp_data.get("name", ""),
                    model_name=exp_data.get("model_name", ""),
                    feature_set=exp_data.get("feature_set", ""),
                    metrics=exp_data.get("metrics", {}),
                    training_time=exp_data.get("training_time"),
                    feature_count=exp_data.get("feature_count"),
                )

            return comparison
        except Exception as e:
            logger.error(f"Failed to deserialize ModelComparison: {e}")
            return None
