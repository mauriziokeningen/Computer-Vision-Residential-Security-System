"""
REST endpoints for Camera management (CRUD + local laptop webcam stream).
Uses SQLAlchemy ORM with the session provided by src.database.session.
"""
import asyncio
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.api.schemas import CameraCreate, CameraUpdate, CameraResponse
from src.database.session import get_db
from src.database.models import Camera
from src.services.local_camera_stream import local_camera_stream_service

router = APIRouter(prefix="/cameras", tags=["Cameras"])

LOCAL_CAMERA_LOCATION = "Laptop webcam"
LOCAL_CAMERA_SOURCE = "local://0"


@router.post("/local-webcam/ensure", response_model=CameraResponse, status_code=200)
async def ensure_local_webcam(db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.ip_address == LOCAL_CAMERA_SOURCE).first()

    if camera is None:
        camera = Camera(
            location=LOCAL_CAMERA_LOCATION,
            ip_address=LOCAL_CAMERA_SOURCE,
            status="ACTIVE",
        )
        db.add(camera)
    else:
        camera.location = LOCAL_CAMERA_LOCATION
        camera.status = "ACTIVE"

    db.commit()
    db.refresh(camera)
    return camera


@router.get("/local-webcam/status")
async def local_webcam_status():
    return local_camera_stream_service.get_status()


@router.post("/local-webcam/stop")
async def stop_local_webcam():
    local_camera_stream_service.stop()
    return {"status": "stopped"}


@router.get("/local-webcam/stream")
async def local_webcam_stream(
    source: int = Query(0, ge=0, le=10, description="Local webcam device index"),
):
    """
    MJPEG stream owned by the backend.
    Reuses a single shared capture session instead of reopening the camera on each request.
    """
    status = local_camera_stream_service.get_status()
    if not status["running"]:
        try:
            await asyncio.to_thread(local_camera_stream_service.start, source, 4.0)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return StreamingResponse(
        local_camera_stream_service.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@router.post("/", response_model=CameraResponse, status_code=201)
async def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    new_camera = Camera(
        location=camera.location,
        ip_address=camera.ip_address,
        status=camera.status,
    )
    try:
        db.add(new_camera)
        db.commit()
        db.refresh(new_camera)
        return new_camera
    except IntegrityError as e:
        db.rollback()
        detail = str(e.orig)
        if "ip_address" in detail:
            raise HTTPException(status_code=409, detail=f"IP address '{camera.ip_address}' is already registered.")
        elif "location" in detail:
            raise HTTPException(status_code=409, detail=f"Location '{camera.location}' is already registered.")
        raise HTTPException(status_code=409, detail="A camera with this data already exists.")


@router.get("/", response_model=List[CameraResponse])
async def list_cameras(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of cameras to return"),
    offset: int = Query(0, ge=0, description="Number of cameras to skip"),
    db: Session = Depends(get_db),
):
    query = db.query(Camera)
    if status:
        query = query.filter(Camera.status == status)
    return query.order_by(Camera.location).offset(offset).limit(limit).all()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: UUID, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")
    return camera


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: UUID, camera: CameraUpdate, db: Session = Depends(get_db)):
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")

    update_data = camera.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update.")

    for key, value in update_data.items():
        setattr(db_camera, key, value)

    try:
        db.commit()
        db.refresh(db_camera)
        return db_camera
    except IntegrityError as e:
        db.rollback()
        detail = str(e.orig)
        if "ip_address" in detail:
            raise HTTPException(status_code=409, detail="This IP address is already registered to another camera.")
        elif "location" in detail:
            raise HTTPException(status_code=409, detail="This location is already registered to another camera.")
        raise HTTPException(status_code=409, detail="Update conflicts with existing data.")


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: UUID, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")

    try:
        db.delete(camera)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete camera {camera_id}: it has associated incident records. "
                   f"Remove or reassign those records first.",
        )