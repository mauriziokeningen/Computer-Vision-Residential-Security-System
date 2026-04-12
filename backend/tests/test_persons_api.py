import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch

# --- PRE-IMPORT PATCHING ---
# Prevent real AI models from loading during test collection
with patch("src.services.face_processor.FaceAnalysis"):
    from src.api.server import app

# ---------------------------------------------------------
# FIX: Manual TestClient initialization
# ---------------------------------------------------------
# We use a context manager inside the test OR 
# initialize without keywords to bypass the Starlette/httpx bug
# ---------------------------------------------------------

@patch("src.api.routes.persons.face_processor")
def test_complete_enrollment_pipeline(mock_face_processor):
    """
    End-to-End test for the biometric enrollment pipeline.
    Phase 1: Metadata creation.
    Phase 2: Biometric extraction simulation and pgvector persistence.
    """
    
    # Initialize client locally inside the test to be safe
    with TestClient(app) as client:
        
        # ---------------------------------------------------------
        # MOCK CONFIGURATION
        # ---------------------------------------------------------
        # Return a 512-d unit vector
        fake_vector = np.ones(512, dtype=np.float32)
        
        mock_face_processor.extract_face_embedding.return_value = fake_vector
        mock_face_processor.calculate_master_vector.return_value = fake_vector

        # ---------------------------------------------------------
        # PHASE 1: Create Person
        # ---------------------------------------------------------
        response_create = client.post("/api/persons/", json={
            "full_name": "Automated Test User",
            "person_type": "STAFF"
        })
        
        assert response_create.status_code == 200, f"Person creation failed: {response_create.text}"
        
        person_data = response_create.json()
        assert "id" in person_data
        person_id = person_data["id"]

        # ---------------------------------------------------------
        # PHASE 2: Biometric Enrollment
        # ---------------------------------------------------------
        fake_image_bytes = b"fake_jpeg_binary_data"
        
        # Simulate multipart/form-data upload
        files = [
            ("files", ("test_face_1.jpg", fake_image_bytes, "image/jpeg")),
            ("files", ("test_face_2.jpg", fake_image_bytes, "image/jpeg"))
        ]

        response_enroll = client.post(f"/api/persons/{person_id}/enroll", files=files)

        # ---------------------------------------------------------
        # PHASE 3: Validations
        # ---------------------------------------------------------
        assert response_enroll.status_code == 200, f"Enrollment failed: {response_enroll.text}"
        
        enroll_data = response_enroll.json()
        assert enroll_data["status"] == "SUCCESS"
        assert enroll_data["faces_processed"] == 2
        
        # Ensure the AI service was called correctly
        assert mock_face_processor.extract_face_embedding.call_count == 2
        mock_face_processor.calculate_master_vector.assert_called_once()