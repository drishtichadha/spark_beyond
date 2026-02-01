"""Configuration module for the ML platform"""
from .resource_limits import (
    ResourceLimits,
    get_resource_limits,
    DEFAULT_LIMITS,
    PRODUCTION_LIMITS,
    DEVELOPMENT_LIMITS,
    ENTERPRISE_LIMITS,
)

__all__ = [
    "ResourceLimits",
    "get_resource_limits",
    "DEFAULT_LIMITS",
    "PRODUCTION_LIMITS",
    "DEVELOPMENT_LIMITS",
    "ENTERPRISE_LIMITS",
]
