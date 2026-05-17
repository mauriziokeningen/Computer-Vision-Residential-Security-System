from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List

from src.database.session import get_db
from src.database.models import Person
from src.api.schemas import PersonCreate, PersonResponse, EnrollmentResponse
from src.services.face_processor import FaceProcessorService

router = APIRouter(
    prefix="/persons",
    tags=["Enrollment"]
)

face_processor = None

def get_face_processor():
    """Lazy loader for the AI service."""
    global face_processor
    if face_processor is None:
        face_processor = FaceProcessorService()
    return face_processor


@router.post("/", response_model=PersonResponse)
def create_person(person: PersonCreate, db: Session = Depends(get_db)):
    db_person = Person(
        full_name=person.full_name,
        person_type=person.person_type,
        building=person.building,
        apartment=person.apartment,
        phone=person.phone,
        email=person.email,
        valid_from=person.valid_from,
        valid_until=person.valid_until,
    )

    db.add(db_person)
    db.commit()
    db.refresh(db_person)

    return db_person


@router.get("/", response_model=list[PersonResponse])
def get_persons(db: Session = Depends(get_db)):
    persons = db.query(Person).all()
    return persons


@router.post("/{person_id}/enroll", response_model=EnrollmentResponse, status_code=status.HTTP_200_OK)
async def enroll_biometrics(
    person_id: UUID,
    files: List[UploadFile] = File(..., description="Upload exactly 3 images of the person's face."),
    db: Session = Depends(get_db)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 images are required per enrollment.")

    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail=f"Person with ID {person_id} not found.")
    
    ai_service = get_face_processor()

    embeddings = []

    for file in files:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid image format (JPEG/PNG).")

        try:
            image_bytes = await file.read()
            vector = ai_service.extract_face_embedding(image_bytes)
            embeddings.append(vector)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Image {file.filename} rejected: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal AI processing error: {str(e)}")
        finally:
            await file.close()

    try:
        master_vector = ai_service.calculate_master_vector(embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate master vector: {str(e)}")

    try:
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