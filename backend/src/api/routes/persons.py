from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List

from src.database.session import get_db
from src.database.models import Person
from src.api.schemas import PersonCreate, PersonResponse, EnrollmentResponse
from src.services.face_processor import FaceProcessorService

# Create the router
router = APIRouter(
    prefix="/persons",
    tags=["Enrollment"]
)

# Initialize the AI Service
# Loading this at the module level ensures models are loaded into RAM/VRAM exactly once.
face_processor = FaceProcessorService()


@router.post("/", response_model=PersonResponse)
def create_person(person: PersonCreate, db: Session = Depends(get_db)):
    """
    Step 1: Register a new person (Metadata only).
    """
    # 1. Create the database object. 
    # Notice we removed the dummy_vector. The face_embedding will default to NULL 
    # until the biometric enrollment endpoint is called.
    db_person = Person(
        full_name=person.full_name, 
        person_type=person.person_type
    )
    
    # 2. Save to PostgreSQL
    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    
    return db_person


@router.get("/", response_model=list[PersonResponse])
def get_persons(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all enrolled persons. 
    """
    persons = db.query(Person).all()
    return persons


@router.post("/{person_id}/enroll", response_model=EnrollmentResponse, status_code=status.HTTP_200_OK)
async def enroll_biometrics(
    person_id: UUID,
    files: List[UploadFile] = File(..., description="Upload 1 to 3 images of the person's face."),
    db: Session = Depends(get_db)
):
    """
    Step 2: Biometric Enrollment Pipeline.
    Receives up to 3 facial images, extracts 512-d embeddings using ArcFace,
    calculates the master vector, and saves it to the PostgreSQL database.
    """
    # 1. Payload Validation
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed per enrollment.")

    # 2. Verify Person exists
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person with ID {person_id} not found.")

    embeddings = []

    # 3. Process each image
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid image format (JPEG/PNG).")
        
        try:
            image_bytes = await file.read()
            vector = face_processor.extract_face_embedding(image_bytes)
            embeddings.append(vector)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Image {file.filename} rejected: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal AI processing error: {str(e)}")
        finally:
            await file.close()

    # 4. Mathematical Aggregation
    try:
        master_vector = face_processor.calculate_master_vector(embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate master vector: {str(e)}")

    # 5. Database Commit
    try:
        # Convert numpy array to standard Python list for pgvector
        person.face_embedding = master_vector.tolist()
        db.commit()
        db.refresh(person)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database transaction failed: {str(e)}")

    return EnrollmentResponse(
        person_id=person.id,
        status="SUCCESS",
        faces_processed=len(embeddings),
        message="Biometric profile successfully generated and linked."
    )