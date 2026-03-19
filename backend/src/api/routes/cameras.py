"""
REST endpoints for Camera management (CRUD).
Uses SQLAlchemy ORM with the session provided by src.database.session.
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.api.schemas import CameraCreate, CameraUpdate, CameraResponse
from src.database.session import get_db
from src.database.models import Camera

router = APIRouter(prefix="/cameras", tags=["Cameras"])

# CREATE

@router.post("/", response_model=CameraResponse, status_code=201)
async def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """
    Register a new camera in the system.
    Returns the created camera with its auto-generated UUID.
    """
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


# ==============================================================================
# READ (List + Detail)
# ==============================================================================
@router.get("/", response_model=List[CameraResponse])
async def list_cameras(
    status: Optional[str] = Query(None, description="Filter by status (e.g., ACTIVE, INACTIVE, MAINTENANCE)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of cameras to return"),
    offset: int = Query(0, ge=0, description="Number of cameras to skip"),
    db: Session = Depends(get_db),
):
    """
    Retrieve all cameras, with optional filtering by status.
    Supports pagination via limit/offset.
    """
    query = db.query(Camera)
    if status:
        query = query.filter(Camera.status == status)
    return query.order_by(Camera.location).offset(offset).limit(limit).all()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: UUID, db: Session = Depends(get_db)):
    """
    Retrieve a single camera by its UUID.
    """
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")
    return camera


# ==============================================================================
# UPDATE (Partial — PATCH)
# ==============================================================================
@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: UUID, camera: CameraUpdate, db: Session = Depends(get_db)):
    """
    Partially update a camera's fields.
    Only the provided fields will be modified (PATCH semantics).
    """
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


# ==============================================================================
# DELETE
# ==============================================================================
@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: UUID, db: Session = Depends(get_db)):
    """
    Remove a camera from the system.
    Returns 204 No Content on success.
    Fails with 409 if the camera has associated incident records.
    """
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