"""
SQLAlchemy ORM models for the Residential Security System.
Maps database tables to Python classes.
"""
import uuid
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from src.database.session import Base

# --- CAMERA MODEL ---
class Camera(Base):
    """ORM model for the 'cameras' table."""
    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location = Column(String(250), unique=True, nullable=False)
    ip_address = Column(String(250), unique=True, nullable=False)
    status = Column(String(100), nullable=False)

# --- PERSON MODEL ---
class Person(Base):
    """
    Database model representing an enrolled person in the security system.
    """
    __tablename__ = "persons"

    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(100), nullable=False)
    person_type = Column(String(50), nullable=False)
    building = Column(String(100), nullable=True)
    apartment = Column(String(100), nullable=True)
    phone = Column(String(20), unique=True, nullable=True)
    email = Column(String(100), unique=True, nullable=True)
    valid_from = Column(DateTime, nullable=True)
    valid_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    face_embedding = Column(String, nullable=True) # We leave it as a string for FastAPI response, but it will store the embedding vector.