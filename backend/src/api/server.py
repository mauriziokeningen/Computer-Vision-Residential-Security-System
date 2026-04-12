from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.cameras import router as cameras_router
from src.api.routes.persons import router as persons_router
from src.api.routes.evidence import router as evidence_router
from src.api.routes.alerts import router as alerts_router
from src.api.routes.incidents import router as incidents_router
from src.api.ws_manager import manager
from src.database.session import init_db
from src.utils.s3_client import ensure_bucket_exists

logger = logging.getLogger("APIServer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages startup and shutdown events for the application."""
    init_db()

    try:
        ensure_bucket_exists()
    except Exception as exc:
        logger.warning(
            "MinIO is not available during startup. The API will continue, "
            "but evidence storage endpoints may fail until MinIO is up. Error: %s",
            exc,
        )

    yield


app = FastAPI(
    title="Residential Security System API",
    description="Core backend for video ingestion, AI evaluation, and incident management.",
    version="1.0.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras_router, prefix="/api")
app.include_router(persons_router, prefix="/api")
app.include_router(evidence_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(incidents_router, prefix="/api")


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/health", tags=["System"])
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Core Backend API",
            "version": "1.0.1"
        },
        status_code=200
    )
