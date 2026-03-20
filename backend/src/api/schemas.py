"""
Pydantic schemas for the Residential Security System.
Separates input validation (Create/Update) from output serialization (Response).
"""
from uuid import UUID
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

# ==========================================
# CAMERA SCHEMAS 
# ==========================================
class CameraCreate(BaseModel):
    """Schema for creating a new camera. All fields required."""
    location: str = Field(..., min_length=1, max_length=250, examples=["Lobby principal - Edificio A"])
    ip_address: str = Field(..., min_length=7, max_length=250, examples=["192.168.1.100"])
    status: str = Field(..., min_length=1, max_length=100, examples=["ACTIVE"])

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

# ==========================================
# PERSON SCHEMAS
# ==========================================
# Reemplaza SOLO la parte de PERSON SCHEMAS en tu schemas.py
class PersonBase(BaseModel):
    full_name: str = Field(..., example="Mauricio", description="Full name")
    person_type: str = Field(..., example="RESIDENT", description="Tipo (RESIDENT, VISITOR, STAFF)")

class PersonCreate(PersonBase):
    pass

class PersonResponse(PersonBase):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}