"""
Pydantic schemas for the Camera entity.
Separates input validation (Create/Update) from output serialization (Response).
"""
from uuid import UUID
from pydantic import BaseModel, Field


class CameraCreate(BaseModel):
    """Schema for creating a new camera. All fields required."""
    location: str = Field(..., min_length=1, max_length=250, examples=["Lobby principal - Edificio A"])
    ip_address: str = Field(..., min_length=7, max_length=250, examples=["192.168.1.100"])
    status: str = Field(..., min_length=1, max_length=100, examples=["ACTIVE"])


class CameraUpdate(BaseModel):
    """Schema for partial camera updates. All fields optional."""
    location: str | None = Field(None, min_length=1, max_length=250)
    ip_address: str | None = Field(None, min_length=7, max_length=250)
    status: str | None = Field(None, min_length=1, max_length=100)


class CameraResponse(BaseModel):
    """Schema for API responses. Maps directly to the 'cameras' table."""
    id: UUID
    location: str
    ip_address: str
    status: str

    model_config = {"from_attributes": True}