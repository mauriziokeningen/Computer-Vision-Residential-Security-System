"""
REST endpoints for Camera management (CRUD).
Follows the Residential Security System schema: cameras(id, location, ip_address, status).
"""
from uuid import UUID
from typing import List

from fastapi import APIRouter, HTTPException, Query
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors

from src.api.schemas import CameraCreate, CameraUpdate, CameraResponse
from src.utils.database import get_connection

router = APIRouter(prefix="/cameras", tags=["Cameras"])



# CREATE

@router.post("/", response_model=CameraResponse, status_code=201)
async def create_camera(camera: CameraCreate):
    """
    Register a new camera in the system.
    Returns the created camera with its auto-generated UUID.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO cameras (location, ip_address, status)
                    VALUES (%s, %s, %s)
                    RETURNING id, location, ip_address, status
                    """,
                    (camera.location, camera.ip_address, camera.status),
                )
                new_camera = cur.fetchone()
                conn.commit()
                return new_camera
            except pg_errors.UniqueViolation as e:
                conn.rollback()
                # Determine which field caused the conflict
                detail = str(e)
                if "ip_address" in detail:
                    raise HTTPException(status_code=409, detail=f"IP address '{camera.ip_address}' is already registered.")
                elif "location" in detail:
                    raise HTTPException(status_code=409, detail=f"Location '{camera.location}' is already registered.")
                raise HTTPException(status_code=409, detail="A camera with this data already exists.")


# READ (List + Detail)

@router.get("/", response_model=List[CameraResponse])
async def list_cameras(
    status: str | None = Query(None, description="Filter by status (e.g., ACTIVE, INACTIVE, MAINTENANCE)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of cameras to return"),
    offset: int = Query(0, ge=0, description="Number of cameras to skip"),
):
    """
    Retrieve all cameras, with optional filtering by status.
    Supports pagination via limit/offset.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if status:
                cur.execute(
                    "SELECT id, location, ip_address, status FROM cameras WHERE status = %s ORDER BY location LIMIT %s OFFSET %s",
                    (status, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT id, location, ip_address, status FROM cameras ORDER BY location LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            return cur.fetchall()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: UUID):
    """
    Retrieve a single camera by its UUID.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, location, ip_address, status FROM cameras WHERE id = %s",
                (str(camera_id),),
            )
            camera = cur.fetchone()
            if not camera:
                raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")
            return camera



# UPDATE (Partial — PATCH)

@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: UUID, camera: CameraUpdate):
    """
    Partially update a camera's fields.
    Only the provided fields will be modified (PATCH semantics).
    """
    # Build dynamic SET clause from non-null fields
    update_data = camera.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided for update.")

    set_clause = ", ".join(f"{key} = %s" for key in update_data)
    values = list(update_data.values()) + [str(camera_id)]

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(
                    f"UPDATE cameras SET {set_clause} WHERE id = %s RETURNING id, location, ip_address, status",
                    values,
                )
                updated = cur.fetchone()
                if not updated:
                    conn.rollback()
                    raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")
                conn.commit()
                return updated
            except pg_errors.UniqueViolation as e:
                conn.rollback()
                detail = str(e)
                if "ip_address" in detail:
                    raise HTTPException(status_code=409, detail="This IP address is already registered to another camera.")
                elif "location" in detail:
                    raise HTTPException(status_code=409, detail="This location is already registered to another camera.")
                raise HTTPException(status_code=409, detail="Update conflicts with existing data.")



# DELETE

@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: UUID):
    """
    Remove a camera from the system.
    Returns 204 No Content on success.
    Fails with 409 if the camera has associated incident records.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("DELETE FROM cameras WHERE id = %s RETURNING id", (str(camera_id),))
                deleted = cur.fetchone()
                if not deleted:
                    conn.rollback()
                    raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found.")
                conn.commit()
            except pg_errors.ForeignKeyViolation:
                conn.rollback()
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot delete camera {camera_id}: it has associated incident records. "
                           f"Remove or reassign those records first.",
                )