from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# 1. Base Schema: Properties shared across all Person schemas
class PersonBase(BaseModel):
    name: str = Field(..., example="Mauricio", description="Full name of the enrolled person")

# 2. Create Schema: Properties required from the client to create a Person
class PersonCreate(PersonBase):
    # Currently we only need the name. 
    # Later, we will add the face vector array here when integrating ArcFace.
    pass

# 3. Response Schema: Properties returned to the client (Frontend)
class PersonResponse(PersonBase):
    id: int
    face_encoding: Optional[str] = None
    created_at: datetime

    class Config:
        # Crucial for FastAPI: Tells Pydantic to read data from SQLAlchemy ORM objects
        from_attributes = True