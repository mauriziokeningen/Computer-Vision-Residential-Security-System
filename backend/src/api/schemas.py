"""
Pydantic schemas for the Residential Security System.
Separates input validation (Create/Update) from output serialization (Response).
"""
from uuid import UUID
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# --- CAMERA SCHEMAS ---

class CameraCreate(BaseModel):
    """Schema for creating a new camera. All fields required."""
    location: str = Field(
        ..., min_length=1, max_length=250, 
        json_schema_extra={"examples": ["Lobby principal - Edificio A"]}
    )
    ip_address: str = Field(
        ..., min_length=7, max_length=250, 
        json_schema_extra={"examples": ["192.168.1.100"]}
    )
    status: str = Field(
        ..., min_length=1, max_length=100, 
        json_schema_extra={"examples": ["ACTIVE"]}
    )


class CameraUpdate(BaseModel):
    """Schema for partial camera updates. All fields optional."""
    location: Optional[str] = Field(None, min_length=1, max_length=250)
    ip_address: Optional[str] = Field(None, min_length=7, max_length=250)
    status: Optional[str] = Field(None, min_length=1, max_length=100)


class CameraResponse(BaseModel):
    """Schema for API responses. Maps directly to the 'cameras' table."""
    id: UUID
    location: str
    ip_address: str
    status: str

    model_config = {"from_attributes": True}


# --- PERSON SCHEMAS ---

class PersonBase(BaseModel):
    full_name: str = Field(
        ..., 
        description="Full name",
        json_schema_extra={"example": "Mauricio"}
    )
    person_type: str = Field(
        ..., 
        description="Tipo (RESIDENT, VISITOR, STAFF)",
        json_schema_extra={"example": "RESIDENT"}
    )


class PersonCreate(PersonBase):
    pass


class PersonResponse(PersonBase):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class EnrollmentResponse(BaseModel):
    """Schema for the response after a successful biometric extraction."""
    person_id: UUID
    status: str
    faces_processed: int
    message: str


# --- ALERT SCHEMAS ---

class AlertCreate(BaseModel):
    """Schema for creating a new alert."""
    incident_id: Optional[UUID] = Field(None, description="UUID of the related incident")
    message: str = Field(
        ..., min_length=1, 
        json_schema_extra={"examples": ["Arma detectada en Lobby principal"]}
    )


class AlertStatusUpdate(BaseModel):
    """Schema for updating an alert's status."""
    status: str = Field(..., description="New status: ACKNOWLEDGED or RESOLVED")


class AlertResponse(BaseModel):
    """Schema for API responses. Maps directly to the 'alerts' table."""
    id: UUID
    incident_id: Optional[UUID]
    message: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]

    model_config = {"from_attributes": True}