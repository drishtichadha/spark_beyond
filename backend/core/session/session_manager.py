"""
Session manager for coordinating Redis and filesystem storage.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis

from backend.core.session.file_storage import FileStorage
from backend.core.session.session_state import SessionState

logger = logging.getLogger(__name__)

# Default TTL for sessions: 24 hours
DEFAULT_TTL = 86400


class SessionManager:
    """
    Manages session persistence using Redis and filesystem storage.

    - Metadata and small JSON objects are stored in Redis
    - Large objects (DataFrames, models) are stored on filesystem
    - Provides in-memory fallback when Redis is unavailable
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        file_storage: Optional[FileStorage] = None,
        fallback_enabled: bool = True,
        ttl: int = DEFAULT_TTL,
    ):
        """
        Initialize SessionManager.

        Args:
            redis_client: Async Redis client
            file_storage: FileStorage instance for large objects
            fallback_enabled: Whether to use in-memory fallback
            ttl: Session TTL in seconds (default 24 hours)
        """
        self.redis = redis_client
        self.file_storage = file_storage or FileStorage()
        self.fallback_enabled = fallback_enabled
        self.ttl = ttl

        # In-memory fallback store
        self._memory_store: Dict[str, SessionState] = {}
        self._redis_available = redis_client is not None

    def _key(self, session_id: str, suffix: str = "metadata") -> str:
        """Generate Redis key for a session."""
        return f"spark_tune:session:{session_id}:{suffix}"

    async def _check_redis_health(self) -> bool:
        """Check if Redis is available."""
        if self.redis is None:
            return False
        try:
            await self.redis.ping()
            self._redis_available = True
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            self._redis_available = False
            return False

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state by ID.

        Tries Redis first, falls back to in-memory if unavailable.

        Args:
            session_id: Session identifier

        Returns:
            SessionState if found, None otherwise
        """
        # Try Redis first
        if await self._check_redis_health():
            try:
                data = await self.redis.hgetall(self._key(session_id))
                if data:
                    # Extend TTL on access
                    await self._extend_ttl(session_id)
                    state = SessionState.from_redis_dict(data)
                    # Also update memory cache
                    if self.fallback_enabled:
                        self._memory_store[session_id] = state
                    return state
            except Exception as e:
                logger.error(f"Redis get_session failed: {e}")

        # Fallback to memory
        if self.fallback_enabled and session_id in self._memory_store:
            logger.debug(f"Using in-memory fallback for session {session_id}")
            state = self._memory_store[session_id]
            state.update_last_accessed()
            return state

        return None

    async def save_session(self, state: SessionState) -> bool:
        """
        Save session state.

        Args:
            state: SessionState to save

        Returns:
            True if successful
        """
        state.update_last_accessed()

        # Try Redis first
        if await self._check_redis_health():
            try:
                key = self._key(state.session_id)
                await self.redis.hset(key, mapping=state.to_redis_dict())
                await self.redis.expire(key, self.ttl)
                logger.debug(f"Saved session to Redis: {state.session_id}")

                # Also save to memory as backup
                if self.fallback_enabled:
                    self._memory_store[state.session_id] = state
                return True
            except Exception as e:
                logger.error(f"Redis save_session failed: {e}")

        # Fallback to memory only
        if self.fallback_enabled:
            logger.warning(f"Saving session to memory fallback: {state.session_id}")
            self._memory_store[state.session_id] = state
            return True

        return False

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely (Redis + filesystem).

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        success = True

        # Delete from Redis
        if await self._check_redis_health():
            try:
                # Delete all keys for this session
                keys = [
                    self._key(session_id, "metadata"),
                    self._key(session_id, "insights"),
                    self._key(session_id, "microsegments"),
                ]
                await self.redis.delete(*keys)
            except Exception as e:
                logger.error(f"Failed to delete session from Redis: {e}")
                success = False

        # Delete from memory
        if session_id in self._memory_store:
            del self._memory_store[session_id]

        # Delete filesystem
        if not self.file_storage.delete_session_files(session_id):
            success = False

        return success

    async def _extend_ttl(self, session_id: str):
        """Extend TTL for all session keys."""
        if not self._redis_available:
            return

        try:
            keys = [
                self._key(session_id, "metadata"),
                self._key(session_id, "insights"),
                self._key(session_id, "microsegments"),
            ]
            for key in keys:
                if await self.redis.exists(key):
                    await self.redis.expire(key, self.ttl)
        except Exception as e:
            logger.error(f"Failed to extend TTL: {e}")

    async def save_insights(
        self, session_id: str, insights: list, microsegments: list
    ) -> bool:
        """
        Save insight analysis results.

        Args:
            session_id: Session identifier
            insights: List of insight dictionaries
            microsegments: List of microsegment dictionaries

        Returns:
            True if successful
        """
        if await self._check_redis_health():
            try:
                # Store insights
                await self.redis.set(
                    self._key(session_id, "insights"),
                    json.dumps(insights),
                    ex=self.ttl,
                )
                # Store microsegments
                await self.redis.set(
                    self._key(session_id, "microsegments"),
                    json.dumps(microsegments),
                    ex=self.ttl,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to save insights: {e}")

        return False

    async def get_insights(
        self, session_id: str
    ) -> tuple[Optional[list], Optional[list]]:
        """
        Get cached insight analysis results.

        Args:
            session_id: Session identifier

        Returns:
            Tuple of (insights_list, microsegments_list), or (None, None) if not cached
        """
        if await self._check_redis_health():
            try:
                insights_json = await self.redis.get(self._key(session_id, "insights"))
                microsegments_json = await self.redis.get(
                    self._key(session_id, "microsegments")
                )

                insights = json.loads(insights_json) if insights_json else None
                microsegments = (
                    json.loads(microsegments_json) if microsegments_json else None
                )

                return insights, microsegments
            except Exception as e:
                logger.error(f"Failed to get insights: {e}")

        return None, None

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        if await self._check_redis_health():
            try:
                return await self.redis.exists(self._key(session_id)) > 0
            except Exception:
                pass

        return session_id in self._memory_store

    async def create_session(self, session_id: str) -> SessionState:
        """
        Create a new session.

        Args:
            session_id: Session identifier

        Returns:
            New SessionState instance
        """
        state = SessionState(session_id=session_id)
        await self.save_session(state)
        return state

    async def sync_memory_to_redis(self) -> int:
        """
        Sync in-memory sessions back to Redis when it becomes available.

        Returns:
            Number of sessions synced
        """
        if not self._memory_store:
            return 0

        if not await self._check_redis_health():
            return 0

        synced = 0
        logger.info(f"Syncing {len(self._memory_store)} sessions to Redis")

        for session_id, state in list(self._memory_store.items()):
            try:
                key = self._key(session_id)
                await self.redis.hset(key, mapping=state.to_redis_dict())
                await self.redis.expire(key, self.ttl)
                synced += 1
            except Exception as e:
                logger.error(f"Failed to sync session {session_id}: {e}")

        return synced

    async def get_all_session_ids(self) -> list[str]:
        """Get all active session IDs."""
        session_ids = set()

        # From Redis
        if await self._check_redis_health():
            try:
                pattern = "spark_tune:session:*:metadata"
                async for key in self.redis.scan_iter(pattern):
                    # Extract session_id from key
                    parts = key.split(":")
                    if len(parts) >= 3:
                        session_ids.add(parts[2])
            except Exception as e:
                logger.error(f"Failed to scan Redis for sessions: {e}")

        # From memory
        session_ids.update(self._memory_store.keys())

        return list(session_ids)

    async def cleanup_expired_sessions(self) -> int:
        """
        Cleanup orphaned filesystem directories.

        Returns:
            Number of sessions cleaned up
        """
        valid_sessions = set(await self.get_all_session_ids())
        return self.file_storage.cleanup_orphaned_sessions(valid_sessions)

    def get_memory_session_count(self) -> int:
        """Get number of sessions in memory fallback."""
        return len(self._memory_store)

    @property
    def is_redis_available(self) -> bool:
        """Check if Redis is currently available."""
        return self._redis_available
