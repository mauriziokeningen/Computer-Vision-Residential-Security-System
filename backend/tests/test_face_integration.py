import pytest
import numpy as np
from sqlalchemy import text, select
from src.database.session import SessionLocal
from src.database.models import Person
from src.modules.face.inference import _find_closest_match_in_db, MAX_ALLOWED_DISTANCE

def _generate_normalized_embedding():
    """Generates a random 512-D L2-normalized vector (Valid ArcFace signature)."""
    vec = np.random.randn(512)
    return (vec / np.linalg.norm(vec)).tolist()

@pytest.fixture(scope="module")
def setup_test_db():
    """Fixture to ensure a clean database connection state for testing."""
    db = SessionLocal()
    # Clear potential remnants from previous test executions
    db.execute(text("DELETE FROM persons WHERE full_name LIKE 'Test Resident%';"))
    db.commit()
    yield db
    # Final post-testing cleanup loop
    db.execute(text("DELETE FROM persons WHERE full_name LIKE 'Test Resident%';"))
    db.commit()
    db.close()

def test_database_hnsw_index_exists(setup_test_db):
    """Verifies that the HNSW graph index is physically created inside PostgreSQL."""
    query = text("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'persons' AND indexname = 'idx_persons_face_embedding_hnsw';
    """)
    result = setup_test_db.execute(query).fetchone()
    assert result is not None, " The HNSW index does not exist in the database schema."

def test_closest_match_known_resident(setup_test_db):
    """Verifies that a registered resident is accurately recognized (Distance <= 0.61)."""
    db = setup_test_db
    embedding_target = _generate_normalized_embedding()
    
    # Register a temporary test resident
    new_person = Person(full_name="Test Resident Alpha", person_type="RESIDENT", face_embedding=embedding_target)
    db.add(new_person)
    db.commit()

    # Execute the database search by passing the exact matching embedding signature
    name, distance = _find_closest_match_in_db(np.array(embedding_target))
    
    assert name == "Test Resident Alpha"
    assert distance < MAX_ALLOWED_DISTANCE
    assert distance >= 0.0

def test_closest_match_unknown_person(setup_test_db):
    """Ensures that unmapped or orthogonal vectors correctly trigger the 'unknown_person' boundary gate."""
    db = setup_test_db
    # Insert a dummy base resident record to ensure the database is not empty and the HNSW index is active
    vec_base = _generate_normalized_embedding()
    new_person = Person(full_name="Test Resident Beta", person_type="RESIDENT", face_embedding=vec_base)
    db.add(new_person)
    db.commit()

    # Generate a completely separate random vector signature to simulate an intruder threat
    intruder_vec = _generate_normalized_embedding()
    
    name, distance = _find_closest_match_in_db(np.array(intruder_vec))
    
    # If the cosine similarity distance exceeds the 0.61 cutoff, the Gatekeeper must flag it as anonymous
    if distance > MAX_ALLOWED_DISTANCE:
        assert name == "unknown_person"