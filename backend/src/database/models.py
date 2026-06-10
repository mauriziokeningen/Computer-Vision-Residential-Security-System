"""
SQLAlchemy ORM models for the Residential Security System.
Maps database tables to Python classes.
"""
import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
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
    """ORM model for the 'persons' table."""
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
    face_embedding = Column(Vector(512), nullable=True)

    __table_args__ = (
        Index(
            'idx_persons_face_embedding_hnsw',
            'face_embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'face_embedding': 'vector_cosine_ops'}
        ),
    )


# --- INCIDENT MODEL ---
class Incident(Base):
    """ORM model for the 'incidents' table."""
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, server_default=func.now())
    incident_metadata = Column(JSONB, nullable=False)


# --- ALERT MODEL ---
class Alert(Base):
    """ORM model for the 'alerts' table (notification inbox for frontend)."""
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=True)
    message = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, server_default="UNREAD")
    created_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)
