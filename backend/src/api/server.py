from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.routes.cameras import router as cameras_router
from src.api.routes.persons import router as persons_router
from src.api.routes.evidence import router as evidence_router
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