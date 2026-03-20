from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database.session import get_db
from src.database.models import Person
from src.api.schemas import PersonCreate, PersonResponse

# Creamos el enrutador para agrupar todos los endpoints relacionados con "personas"
router = APIRouter(
    prefix="/persons",
    tags=["Enrollment"]
)

@router.post("/", response_model=PersonResponse)
def enroll_person(person: PersonCreate, db: Session = Depends(get_db)):
    """
    Registra a una nueva persona en el sistema.
    """

    # Truco de Python: [0.0] * 512 crea una lista de 512 ceros automáticamente.
    # Luego str() lo convierte en texto para que PostgreSQL lo acepte.
    dummy_vector = str([0.0] * 512)

    # 1. Creamos el objeto del modelo de base de datos usando los datos validados del esquema
    db_person = Person(
        full_name=person.full_name, 
        person_type=person.person_type,
        # OJO: En tu SQL face_embedding dice "NOT NULL", le pasamos un texto dummy para que no explote
        face_embedding=dummy_vector
    )
    
    # 2. Lo añadimos a la sesión y guardamos en PostgreSQL
    db.add(db_person)
    db.commit()
    db.refresh(db_person) # Refrescamos para obtener el ID autogenerado
    
    # 3. Retornamos el objeto (FastAPI y Pydantic lo convertirán a JSON automáticamente)
    return db_person


@router.get("/", response_model=list[PersonResponse])
def get_persons(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all enrolled persons. 
    """

    persons = db.query(Person).all()

    return persons
