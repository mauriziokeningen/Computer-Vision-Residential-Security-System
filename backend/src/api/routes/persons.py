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
    # 1. Creamos el objeto del modelo de base de datos usando los datos validados del esquema
    db_person = Person(name=person.name)
    
    # 2. Lo añadimos a la sesión y guardamos en PostgreSQL
    db.add(db_person)
    db.commit()
    db.refresh(db_person) # Refrescamos para obtener el ID autogenerado
    
    # 3. Retornamos el objeto (FastAPI y Pydantic lo convertirán a JSON automáticamente)
    return db_person