"""
Session middleware for FastAPI.

Extracts session ID from request headers and attaches it to request state.
"""

import logging
import re
import uuid
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# Header name for session ID
SESSION_HEADER = "X-Session-ID"

# UUID v4 regex pattern for validation
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID v4."""
    return bool(UUID_PATTERN.match(value))


def generate_session_id() -> str:
    """Generate a new UUID v4 session ID."""
    return str(uuid.uuid4())


class SessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that handles session ID extraction and generation.

    - Extracts session ID from X-Session-ID header
    - Generates new session ID if not provided or invalid
    - Attaches session_id to request.state for use in route handlers
    - Adds session ID to response headers
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request and attach session information."""
        # Skip session handling for health checks and static files
        if request.url.path in ("/health", "/", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Extract session ID from header
        session_id = request.headers.get(SESSION_HEADER)
        is_new_session = False

        # Validate and potentially generate new session ID
        if session_id and is_valid_uuid(session_id):
            logger.debug(f"Using existing session: {session_id}")
        else:
            if session_id:
                logger.warning(f"Invalid session ID received: {session_id}")
            session_id = generate_session_id()
            is_new_session = True
            logger.info(f"Generated new session: {session_id}")

        # Attach to request state for route handlers
        request.state.session_id = session_id
        request.state.is_new_session = is_new_session

        # Call the next middleware/handler
        response = await call_next(request)

        # Add session ID to response headers
        response.headers[SESSION_HEADER] = session_id
        if is_new_session:
            response.headers["X-Session-New"] = "true"

        return response


def get_session_id(request: Request) -> str:
    """
    Dependency to get session ID from request state.

    Usage in route handlers:
        @router.get("/endpoint")
        async def handler(session_id: str = Depends(get_session_id)):
            ...
    """
    return getattr(request.state, "session_id", None)


def get_session_id_from_query(
    request: Request, session_id: Optional[str] = None
) -> str:
    """
    Get session ID from query parameter or request state.

    Useful for WebSocket connections where headers may not be accessible.

    Usage:
        @app.websocket("/ws")
        async def websocket_handler(
            websocket: WebSocket,
            session_id: str = Depends(get_session_id_from_query)
        ):
            ...
    """
    # First try query parameter
    if session_id and is_valid_uuid(session_id):
        return session_id

    # Fall back to request state (if middleware ran)
    state_session = getattr(request.state, "session_id", None)
    if state_session:
        return state_session

    # Generate new if nothing found
    return generate_session_id()
