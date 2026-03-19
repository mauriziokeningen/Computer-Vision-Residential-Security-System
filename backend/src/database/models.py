from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from src.database.session import Base

class Person(Base):
    """
    Database model representing an enrolled person in the security system.
    """
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    
    # The person's full name (indexed for faster searches)
    name = Column(String(255), index=True, nullable=False)
    
    # We will store the AI generated face embedding as a JSON string or path for now.
    # When integrating ArcFace, this will hold the vector data.
    face_encoding = Column(String, nullable=True) 
    
    # Auto-generates the timestamp when the person is registered
    created_at = Column(DateTime(timezone=True), server_default=func.now())