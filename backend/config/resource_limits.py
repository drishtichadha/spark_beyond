"""
Resource limits and quotas for the ML platform.

These limits prevent resource exhaustion and ensure fair usage across users.
"""
from typing import Dict, Any
from dataclasses import dataclass
import os


@dataclass
class ResourceLimits:
    """Resource limits for data loading and processing"""

    # File size limits
    max_file_size_mb: int = 500  # Maximum CSV file size in MB
    max_file_size_bytes: int = 500 * 1024 * 1024  # 500MB

    # Row count limits
    default_max_rows: int = 1_000_000  # Default max rows per dataset
    absolute_max_rows: int = 10_000_000  # Hard limit across all tiers

    # Data profiling limits
    max_profile_rows: int = 100_000  # Maximum rows for data profiling
    min_profile_rows: int = 1  # Minimum rows for profiling

    # Feature generation limits
    max_generated_features: int = 1000  # Maximum generated features
    max_feature_interactions: int = 500  # Maximum interaction features

    # Model training limits
    max_training_time_seconds: int = 3600  # 1 hour max training time
    max_automl_time_seconds: int = 600  # 10 minutes max for AutoML

    # Memory limits (per user session)
    spark_driver_memory_gb: int = 4  # Spark driver memory per session

    # Concurrent operation limits
    max_concurrent_sessions: int = 100  # Maximum concurrent user sessions
    max_concurrent_trainings: int = 10  # Maximum concurrent model trainings

    def to_dict(self) -> Dict[str, Any]:
        """Convert limits to dictionary"""
        return {
            "max_file_size_mb": self.max_file_size_mb,
            "default_max_rows": self.default_max_rows,
            "absolute_max_rows": self.absolute_max_rows,
            "max_profile_rows": self.max_profile_rows,
            "max_generated_features": self.max_generated_features,
            "max_training_time_seconds": self.max_training_time_seconds,
            "max_automl_time_seconds": self.max_automl_time_seconds,
        }


# Production limits (default)
PRODUCTION_LIMITS = ResourceLimits(
    max_file_size_mb=500,
    default_max_rows=1_000_000,
    absolute_max_rows=10_000_000,
    max_profile_rows=100_000,
    max_generated_features=1000,
    max_feature_interactions=500,
    max_training_time_seconds=3600,
    max_automl_time_seconds=600,
    spark_driver_memory_gb=4,
    max_concurrent_sessions=100,
    max_concurrent_trainings=10,
)

# Development limits (more relaxed for testing)
DEVELOPMENT_LIMITS = ResourceLimits(
    max_file_size_mb=1000,
    default_max_rows=5_000_000,
    absolute_max_rows=10_000_000,
    max_profile_rows=200_000,
    max_generated_features=2000,
    max_feature_interactions=1000,
    max_training_time_seconds=7200,
    max_automl_time_seconds=1200,
    spark_driver_memory_gb=8,
    max_concurrent_sessions=50,
    max_concurrent_trainings=5,
)

# Enterprise limits (for paid tiers)
ENTERPRISE_LIMITS = ResourceLimits(
    max_file_size_mb=2000,
    default_max_rows=5_000_000,
    absolute_max_rows=10_000_000,
    max_profile_rows=500_000,
    max_generated_features=5000,
    max_feature_interactions=2000,
    max_training_time_seconds=14400,  # 4 hours
    max_automl_time_seconds=3600,  # 1 hour
    spark_driver_memory_gb=16,
    max_concurrent_sessions=500,
    max_concurrent_trainings=50,
)


def get_resource_limits(tier: str = None) -> ResourceLimits:
    """
    Get resource limits based on environment or tier.

    Args:
        tier: User tier (production, development, enterprise)
              If None, uses RESOURCE_TIER environment variable

    Returns:
        ResourceLimits instance
    """
    if tier is None:
        tier = os.getenv("RESOURCE_TIER", "production").lower()

    limits_map = {
        "production": PRODUCTION_LIMITS,
        "development": DEVELOPMENT_LIMITS,
        "enterprise": ENTERPRISE_LIMITS,
    }

    return limits_map.get(tier, PRODUCTION_LIMITS)


# Global limits instance (can be overridden per-request)
DEFAULT_LIMITS = get_resource_limits()
