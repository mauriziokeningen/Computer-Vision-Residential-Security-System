"""
REST endpoints for Incident management and event simulation.
Provides query endpoints for the frontend and a simulation endpoint
for testing the rule engine without running the AI modules.
Integrates WebSocket notifications for real-time push to frontend.
"""
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from src.database.session import get_db
from src.database.models import Incident, Alert
from src.api.notifications import (
    notify_new_alert,
    notify_alert_count_update,
)

router = APIRouter(prefix="/incidents", tags=["Incidents"])


# --- Schemas ---
class IncidentResponse(BaseModel):
    id: UUID
    created_at: datetime
    incident_metadata: dict

    model_config = {"from_attributes": True}


class EventSimulation(BaseModel):
    """Schema for simulating an AI module event via REST."""
    module: str = Field(
        ...,
        description="AI module: face, weapons, or pose",
        examples=["face"]
    )
    camera_id: str = Field(
        default="main_camera",
        examples=["cam-lobby-01"]
    )
    detections: list = Field(
        ...,
        description="List of detections from the AI module",
        examples=[[{
            "name": "unknown_person",
            "confidence": 0.85,
            "bbox": {"x": 100, "y": 50, "w": 200, "h": 300}
        }]]
    )


class SimulationResponse(BaseModel):
    incident_id: Optional[UUID]
    alert_message: Optional[str]
    priority: str
    rule_triggered: str


# QUERY ENDPOINTS (for frontend)

@router.get("/", response_model=List[IncidentResponse])
async def list_incidents(
    priority: Optional[str] = Query(None, description="Filter by priority: LOW, MEDIUM, HIGH, CRITICAL"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all incidents, ordered by most recent first."""
    query = db.query(Incident)
    if priority:
        query = query.filter(
            Incident.incident_metadata["priority"].astext == priority
        )
    return query.order_by(Incident.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: UUID, db: Session = Depends(get_db)):
    """Get a single incident by UUID."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")
    return incident


# EVENT SIMULATION (for testing without AI modules)

@router.post("/simulate", response_model=SimulationResponse, status_code=201)
async def simulate_event(event: EventSimulation, db: Session = Depends(get_db)):
    """
    Simulate an AI module event to test the rule engine.
    Creates incident + alert and pushes WebSocket notification.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    camera_id = event.camera_id

    incident_id = None
    alert_message = None
    priority = "NONE"
    rule_triggered = "NONE"

    if event.module == "face":
        for detection in event.detections:
            name = detection.get("name", "unknown_person")
            confidence = detection.get("confidence", 0.0)
            conf_pct = confidence * 100

            if name == "unknown_person":
                priority = "MEDIUM"
                rule_triggered = "RN-02"
                alert_message = (
                    f"Persona desconocida detectada en {camera_id} "
                    f"({timestamp}) - Confianza: {conf_pct:.1f}%"
                )

    elif event.module == "weapons":
        for detection in event.detections:
            weapon_class = detection.get("class", "unknown")
            confidence = detection.get("confidence", 0.0)
            conf_pct = confidence * 100

            priority = "HIGH"
            rule_triggered = "WEAPON_DETECTED"
            alert_message = (
                f"ARMA DETECTADA: {weapon_class} en {camera_id} "
                f"({timestamp}) - Confianza: {conf_pct:.1f}%"
            )

    elif event.module == "pose":
        aggressive_actions = {"punch", "kick", "push", "fight", "struggle",
                              "golpe", "patada", "empujon", "pelea", "forcejeo"}
        fall_actions = {"fall", "caida"}

        for detection in event.detections:
            action = detection.get("action", "unknown")
            confidence = detection.get("confidence", 0.0)
            conf_pct = confidence * 100

            if action.lower() in aggressive_actions:
                priority = "HIGH"
                rule_triggered = "RN-04"
                alert_message = (
                    f"AGRESION DETECTADA: {action} en {camera_id} "
                    f"({timestamp}) - Confianza: {conf_pct:.1f}%"
                )
            elif action.lower() in fall_actions:
                priority = "MEDIUM"
                rule_triggered = "RN-05"
                alert_message = (
                    f"CAIDA DETECTADA en {camera_id} "
                    f"({timestamp}) - Confianza: {conf_pct:.1f}%"
                )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown module: {event.module}")

    # Create incident and alert if a rule was triggered
    if alert_message:
        metadata = {
            "rule_triggered": rule_triggered,
            "priority": priority,
            "module": event.module,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "detections": event.detections,
        }

        incident = Incident(incident_metadata=metadata)
        db.add(incident)
        db.commit()
        db.refresh(incident)
        incident_id = incident.id

        alert = Alert(
            incident_id=incident.id,
            message=alert_message,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        # Push WebSocket notifications
        await notify_new_alert(
            alert_id=alert.id,
            incident_id=incident.id,
            message=alert_message,
            status=alert.status,
            created_at=alert.created_at,
        )

        # Broadcast updated counts
        unread = db.query(Alert).filter(Alert.status == "UNREAD").count()
        acknowledged = db.query(Alert).filter(Alert.status == "ACKNOWLEDGED").count()
        resolved = db.query(Alert).filter(Alert.status == "RESOLVED").count()
        await notify_alert_count_update(unread, acknowledged, resolved)

    return SimulationResponse(
        incident_id=incident_id,
        alert_message=alert_message,
        priority=priority,
        rule_triggered=rule_triggered,
    )