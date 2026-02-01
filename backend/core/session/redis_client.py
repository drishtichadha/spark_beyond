"""
Redis client utilities for session management.
"""

import os
import logging
from typing import Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Global Redis pool
_redis_pool: Optional[redis.Redis] = None


def get_redis_url() -> str:
    """Get Redis URL from environment or default."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


async def create_redis_pool() -> Optional[redis.Redis]:
    """
    Create and return a Redis connection pool.

    Returns None if Redis is unavailable.
    """
    global _redis_pool

    if _redis_pool is not None:
        return _redis_pool

    redis_url = get_redis_url()
    logger.info(f"Connecting to Redis at {redis_url}")

    try:
        _redis_pool = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        await _redis_pool.ping()
        logger.info("Redis connection established")
        return _redis_pool
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. Using in-memory fallback.")
        _redis_pool = None
        return None


async def close_redis_pool():
    """Close the Redis connection pool."""
    global _redis_pool

    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None
        logger.info("Redis connection closed")


def get_redis() -> Optional[redis.Redis]:
    """Get the current Redis connection pool (dependency injection helper)."""
    return _redis_pool


async def check_redis_health() -> bool:
    """Check if Redis is available and responding."""
    if _redis_pool is None:
        return False
    try:
        await _redis_pool.ping()
        return True
    except Exception:
        return False
