"""
REST endpoints for evidence file management.
Handles uploading incident evidence (video clips, images) to MinIO
and generating temporary URLs for viewing.
"""
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

from src.utils.s3_client import (
    upload_incident_clip,
    get_presigned_url,
    list_incident_files,
)

router = APIRouter(prefix="/evidence", tags=["Evidence"])


#Response Schemas 
class UploadResponse(BaseModel):
    object_name: str
    message: str


class FileInfo(BaseModel):
    object_name: str
    size: int
    last_modified: str = None
    content_type: str = None


class PresignedUrlResponse(BaseModel):
    url: str
    expires_in: str


# UPLOAD

@router.post("/upload/{incident_id}/{camera_id}", response_model=UploadResponse, status_code=201)
async def upload_evidence(
    incident_id: str,
    camera_id: str,
    file: UploadFile = File(...),
):
    """Upload an evidence file for an incident."""
    try:
        file_data = await file.read()

        content_type = file.content_type or "application/octet-stream"
        filename = file.filename or "evidence"

        object_name = upload_incident_clip(
            file_data=file_data,
            incident_id=incident_id,
            camera_id=camera_id,
            filename=filename,
            content_type=content_type,
        )

        return UploadResponse(
            object_name=object_name,
            message=f"Evidence uploaded successfully for incident {incident_id}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload evidence: {str(e)}",
        )



# GET PRESIGNED URL
@router.get("/url", response_model=PresignedUrlResponse)
async def get_evidence_url(
    object_name: str = Query(..., description="The object path in MinIO"),
    expires_hours: int = Query(1, ge=1, le=24, description="URL validity in hours"),
):
    """Generate a temporary access URL for a file."""
    from datetime import timedelta

    try:
        url = get_presigned_url(
            object_name=object_name,
            expires=timedelta(hours=expires_hours),
        )
        return PresignedUrlResponse(
            url=url,
            expires_in=f"{expires_hours} hour(s)",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate URL: {str(e)}",
        )



# LIST FILES
@router.get("/incident/{incident_id}", response_model=List[FileInfo])
async def list_evidence(incident_id: str):
    """List all files for an incident."""
    try:
        files = list_incident_files(incident_id=incident_id)
        return files
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list evidence: {str(e)}",
        )