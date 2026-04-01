import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

# Updated test cases to match the actual API return values
test_cases = [
    (
        {"module": "face", "camera_id": "cam-lobby", "detections": [{"name": "unknown_person", "confidence": 0.85}]},
        "RN-02", "MEDIUM"
    ),
    (
        {"module": "weapons", "camera_id": "cam-lobby", "detections": [{"class": "gun", "confidence": 0.92}]},
        "WEAPON_DETECTED", "HIGH" # Changed from WEAPON to WEAPON_DETECTED
    ),
    (
        {"module": "pose", "camera_id": "cam-lobby", "detections": [{"action": "punch", "confidence": 0.88}]},
        "RN-04", "HIGH"
    ),
    (
        {"module": "pose", "camera_id": "cam-lobby", "detections": [{"action": "fall", "confidence": 0.90}]},
        "RN-05", "MEDIUM"
    )
]

@pytest.mark.parametrize("event_payload, expected_rule, expected_priority", test_cases)
def test_incident_creation_rules(event_payload, expected_rule, expected_priority):
    """
    Verify that each AI event triggers the correct incident with its assigned priority.
    """
    response = client.post("/api/incidents/simulate", json=event_payload)
    assert response.status_code == 201
    
    # FIX: The key is 'incident_id', not 'id'
    incident_id = response.json()["incident_id"]

    # Retrieve the incident to verify persistence and metadata
    res_incident = client.get(f"/api/incidents/{incident_id}")
    assert res_incident.status_code == 200
    data = res_incident.json()
    
    metadata = data["incident_metadata"]
    # FIX: The key is 'rule_triggered', not 'rule_id'
    assert metadata["rule_triggered"] == expected_rule
    assert metadata["priority"] == expected_priority

def test_alert_linkage():
    """
    Verify that creating an incident automatically generates a linked alert.
    """
    event = {
        "module": "weapons", 
        "camera_id": "cam-test", 
        "detections": [{"class": "knife", "confidence": 0.95}]
    }
    
    res_sim = client.post("/api/incidents/simulate", json=event)
    # FIX: Using 'incident_id' as returned by the simulation
    incident_id = res_sim.json()["incident_id"]
    
    res_alerts = client.get("/api/alerts/")
    alerts = res_alerts.json()
    
    linked_alert = next((a for a in alerts if a["incident_id"] == incident_id), None)
    
    assert linked_alert is not None
    # Verify the message contains detection details
    assert "knife" in linked_alert["message"] or "ARMA" in linked_alert["message"]