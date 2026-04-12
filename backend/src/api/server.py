from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.api.routes.cameras import router as cameras_router
from src.api.routes.persons import router as persons_router
from src.api.routes.evidence import router as evidence_router
from src.api.routes.alerts import router as alerts_router
from src.api.routes.incidents import router as incidents_router
from src.api.ws_manager import manager
from src.utils.s3_client import ensure_bucket_exists


# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages startup and shutdown events for the application."""
    # Startup: ensure MinIO bucket exists
    ensure_bucket_exists()
    yield


# Initialize the application with enterprise-grade metadata
app = FastAPI(
    title="Residential Security System API",
    description="Core backend for video ingestion, AI evaluation, and incident management.",
    version="1.0.0",
    docs_url="/api/docs",   # Custom route for Swagger UI documentation
    redoc_url="/api/redoc",  # Alternative route for ReDoc documentation
    lifespan=lifespan,
)

# --- Register Routers ---
app.include_router(cameras_router, prefix="/api")
app.include_router(persons_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(incidents_router, prefix="/api")


# --- WebSocket Endpoint ---
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert notifications.
    Frontend connects here to receive push updates when:
    - A new alert is created
    - An alert status changes (ACKNOWLEDGED, RESOLVED)
    - Alert counts are updated

    Connect from frontend: ws://localhost:8000/ws/alerts
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive - listen for any client messages
            # (heartbeat, ping, etc.) but we don't need to process them
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/health", tags=["System"])
async def health_check():
    """
    System health check endpoint.
    Used by load balancers or Docker Healthchecks to verify the API is alive and responsive.
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Core Backend API",
            "version": "1.0.0"
        },
        status_code=200
    )