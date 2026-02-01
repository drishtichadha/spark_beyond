"""
Spark Tune - FastAPI Backend
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import json
import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from backend.routes import data, features, models, insights
from backend.middleware.session_middleware import SessionMiddleware, is_valid_uuid, generate_session_id
from backend.core.session import (
    create_redis_pool,
    SessionManager,
    FileStorage,
)
from backend.core.session.redis_client import close_redis_pool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    logger.info("Starting Spark Tune API...")

    # Initialize Redis connection
    redis_client = await create_redis_pool()

    # Initialize file storage
    file_storage = FileStorage(base_path="./data/sessions")

    # Initialize session manager
    session_manager = SessionManager(
        redis_client=redis_client,
        file_storage=file_storage,
        fallback_enabled=True,
    )

    # Store in app state for access in routes
    app.state.session_manager = session_manager
    app.state.file_storage = file_storage

    # Start background task for Redis health monitoring
    async def redis_health_monitor():
        """Periodically check Redis health and sync memory sessions."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            try:
                if session_manager.is_redis_available:
                    synced = await session_manager.sync_memory_to_redis()
                    if synced > 0:
                        logger.info(f"Synced {synced} sessions to Redis")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis health monitor error: {e}")

    health_task = asyncio.create_task(redis_health_monitor())

    logger.info("Spark Tune API started successfully")
    yield

    # Cleanup
    logger.info("Shutting down Spark Tune API...")
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    await close_redis_pool()
    logger.info("Spark Tune API shutdown complete")


app = FastAPI(
    title="Spark Tune API",
    description="ML Feature Discovery Platform API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - configured via environment for security
# In production, set CORS_ORIGINS environment variable to comma-separated list of allowed origins
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")

# Resource limits configuration
# Set RESOURCE_TIER environment variable to: production, development, or enterprise
RESOURCE_TIER = os.getenv("RESOURCE_TIER", "production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Explicit methods (no wildcard)
    allow_headers=["Content-Type", "Authorization", "X-Session-ID"],  # Explicit headers (no wildcard)
    expose_headers=["X-Session-ID", "X-Session-New"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Session middleware (must be added after CORS)
app.add_middleware(SessionMiddleware)

# Validation error handler for better error messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.error(f"Validation error for {request.url}: {errors}")
    error_messages = []
    for error in errors:
        loc = " -> ".join(str(l) for l in error["loc"])
        error_messages.append(f"{loc}: {error['msg']}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "Validation error",
            "detail": error_messages,
            "errors": errors
        }
    )


# Include routers
app.include_router(data.router)
app.include_router(features.router)
app.include_router(models.router)
app.include_router(insights.router)


# WebSocket for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()

# Thread pool for running blocking Spark operations
executor = ThreadPoolExecutor(max_workers=2)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/microsegments")
async def websocket_microsegments(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for streaming microsegment discovery.

    Client sends: {"action": "start", "params": {...}}
    Server sends:
        - {"type": "progress", "progress": 0-100, "message": "..."}
        - {"type": "batch", "microsegments": [...], "total_found": N}
        - {"type": "complete", "total": N}
        - {"type": "error", "message": "..."}
    """
    from backend.services.spark_service import get_spark_service

    # Validate or generate session ID
    if not session_id or not is_valid_uuid(session_id):
        session_id = generate_session_id()
        logger.warning(f"WebSocket: Generated new session ID: {session_id}")

    await websocket.accept()
    logger.info(f"WebSocket microsegments connection established (session: {session_id})")

    # Get session-aware spark service
    session_manager = websocket.app.state.session_manager
    spark_service = await get_spark_service(session_id, session_manager)

    # Message queue for thread-safe communication
    message_queue: asyncio.Queue = asyncio.Queue()
    is_cancelled = False

    async def send_queued_messages():
        """Coroutine to send messages from the queue"""
        while True:
            try:
                message = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                if message is None:  # Sentinel to stop
                    break
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                break

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            action = data.get("action")

            if action == "start":
                params = data.get("params", {})

                min_support = params.get("min_support", 0.01)
                min_lift = params.get("min_lift", 1.1)
                max_depth = params.get("max_depth", 3)
                top_n_features = params.get("top_n_features", 50)
                max_microsegments = params.get("max_microsegments", 100)

                # Send initial status
                await websocket.send_json({
                    "type": "status",
                    "status": "starting",
                    "message": "Starting microsegment discovery..."
                })

                # Capture the event loop before running in executor
                loop = asyncio.get_running_loop()

                # Run the discovery in a thread to avoid blocking
                def run_discovery():
                    try:
                        return spark_service.discover_microsegments_streaming(
                            min_support=min_support,
                            min_lift=min_lift,
                            max_depth=max_depth,
                            top_n_features=top_n_features,
                            max_microsegments=max_microsegments,
                            progress_callback=lambda p, m: asyncio.run_coroutine_threadsafe(
                                message_queue.put({"type": "progress", "progress": p, "message": m}),
                                loop
                            ),
                            batch_callback=lambda batch: asyncio.run_coroutine_threadsafe(
                                message_queue.put({"type": "batch", "microsegments": batch}),
                                loop
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error in discovery: {e}")
                        return {"error": str(e)}

                # Start message sender task
                sender_task = asyncio.create_task(send_queued_messages())

                # Run discovery in thread pool
                result = await loop.run_in_executor(executor, run_discovery)

                # Signal sender to stop
                await message_queue.put(None)
                await sender_task

                if "error" in result:
                    await websocket.send_json({
                        "type": "error",
                        "message": result["error"]
                    })
                else:
                    await websocket.send_json({
                        "type": "complete",
                        "total": result.get("total", 0),
                        "message": f"Discovery complete! Found {result.get('total', 0)} microsegments"
                    })

            elif action == "cancel":
                is_cancelled = True
                await websocket.send_json({
                    "type": "cancelled",
                    "message": "Discovery cancelled"
                })

            elif action == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket microsegments connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


@app.get("/")
async def root():
    return {
        "name": "Spark Tune API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
