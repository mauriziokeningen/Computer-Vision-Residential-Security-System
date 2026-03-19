"""
SQLAlchemy ORM models for the Residential Security System.
Maps database tables to Python classes.
"""
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.database.session import Base


class Camera(Base):
    """ORM model for the 'cameras' table."""
    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    location = Column(String(250), unique=True, nullable=False)
    ip_address = Column(String(250), unique=True, nullable=False)
    status = Column(String(100), nullable=False)