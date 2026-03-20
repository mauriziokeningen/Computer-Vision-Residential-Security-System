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

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True, nullable=False)
    face_encoding = Column(String, nullable=True) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())