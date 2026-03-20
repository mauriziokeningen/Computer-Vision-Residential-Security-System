"""
MinIO S3 client for uploading files and generating temporary access URLs.
"""
import os
import io
import logging
from datetime import timedelta, datetime
from typing import Optional

from minio import Minio
from minio.error import S3Error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("S3Client")

# Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Default bucket
EVIDENCE_BUCKET = os.getenv("MINIO_EVIDENCE_BUCKET", "incident-evidence")

# Lazy singleton client
_client: Optional[Minio] = None


def _get_client() -> Minio:
    """Return a singleton MinIO client."""
    global _client
    if _client is None:
        _client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        logger.info(
            f"MinIO client initialized at {MINIO_ENDPOINT} "
            f"(secure={MINIO_SECURE})"
        )
    return _client


def ensure_bucket_exists(bucket_name: str = EVIDENCE_BUCKET) -> None:
    """Create the bucket if it does not exist."""
    client = _get_client()
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' created successfully.")
        else:
            logger.info(f"Bucket '{bucket_name}' already exists.")
    except S3Error as e:
        logger.error(f"Failed to ensure bucket '{bucket_name}': {e}")
        raise


def upload_file(
    file_data: bytes,
    object_name: str,
    content_type: str = "application/octet-stream",
    bucket_name: str = EVIDENCE_BUCKET,
) -> str:
    """Upload a file to MinIO and return its object name."""
    
    client = _get_client()
    try:
        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(file_data),
            length=len(file_data),
            content_type=content_type,
        )
        logger.info(
            f"Uploaded '{object_name}' to '{bucket_name}' "
            f"({len(file_data)} bytes, {content_type})"
        )
        return object_name
    except S3Error as e:
        logger.error(f"Failed to upload '{object_name}': {e}")
        raise


def upload_incident_clip(
    file_data: bytes,
    incident_id: str,
    camera_id: str,
    filename: str = "clip.mp4",
    content_type: str = "video/mp4",
) -> str:
    """Upload a file grouped by incident and camera."""
    date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
    object_name = (
        f"{date_prefix}/incident_{incident_id}/"
        f"camera_{camera_id}/{filename}"
    )
    return upload_file(
        file_data=file_data,
        object_name=object_name,
        content_type=content_type,
    )


def get_presigned_url(
    object_name: str,
    bucket_name: str = EVIDENCE_BUCKET,
    expires: timedelta = timedelta(hours=1),
) -> str:
    """Generate a temporary URL to access a file."""
    client = _get_client()
    try:
        url = client.presigned_get_object(
            bucket_name=bucket_name,
            object_name=object_name,
            expires=expires,
        )
        logger.info(
            f"Generated presigned URL for '{object_name}' "
            f"(expires in {expires})"
        )
        return url
    except S3Error as e:
        logger.error(
            f"Failed to generate presigned URL for '{object_name}': {e}"
        )
        raise


def list_incident_files(
    incident_id: str,
    bucket_name: str = EVIDENCE_BUCKET,
) -> list:
    """List all files for a given incident."""
    client = _get_client()
    prefix = f"incident_{incident_id}/"
    files = []

    try:
        objects = client.list_objects(
            bucket_name=bucket_name,
            prefix=prefix,
            recursive=True,
        )
        for obj in objects:
            files.append({
                "object_name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified.isoformat()
                if obj.last_modified else None,
                "content_type": obj.content_type,
            })
        logger.info(
            f"Found {len(files)} files for incident '{incident_id}'"
        )
    except S3Error as e:
        logger.error(
            f"Failed to list files for incident '{incident_id}': {e}"
        )
        raise

    return files