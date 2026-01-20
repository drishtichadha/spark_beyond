"""
Session management module for Spark Tune.

Provides Redis-based session persistence with filesystem storage for large objects.
"""

from backend.core.session.redis_client import create_redis_pool, get_redis
from backend.core.session.session_state import SessionState
from backend.core.session.file_storage import FileStorage
from backend.core.session.session_manager import SessionManager
from backend.core.session.serializers import InsightCacheSerializer

__all__ = [
    "create_redis_pool",
    "get_redis",
    "SessionState",
    "FileStorage",
    "SessionManager",
    "InsightCacheSerializer",
]
