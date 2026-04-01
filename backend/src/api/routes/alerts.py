"""
REST endpoints for Alert state management.
Handles viewing, filtering, and updating alert statuses.
Alerts are created by the system (incident engine) and managed by security personnel.
"""
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.api.schemas import AlertCreate, AlertStatusUpdate, AlertResponse
from src.database.session import get_db
from src.database.models import Alert

router = APIRouter(prefix="/alerts", tags=["Alerts"])

# Valid status transitions
VALID_STATUSES = {"UNREAD", "ACKNOWLEDGED", "RESOLVED"}


# ==============================================================================
# CREATE
# ==============================================================================
@router.post("/", response_model=AlertResponse, status_code=201)
async def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    """
    Create a new alert in the system.
    Typically called by the incident rule engine, not manually.
    """
    new_alert = Alert(
        incident_id=alert.incident_id,
        message=alert.message,
    )
    db.add(new_alert)
    
    try:
        db.commit()
        db.refresh(new_alert)
        return new_alert
    except IntegrityError:
        db.rollback()  # Limpia la sesión para que no se corrompa
        raise HTTPException(
            status_code=400,
            detail=f"El incidente proporcionado ({alert.incident_id}) no existe en el sistema."
        )


# ==============================================================================
# READ (List + Detail)
# ==============================================================================
@router.get("/", response_model=List[AlertResponse])
async def list_alerts(
    status: Optional[str] = Query(
        None,
        description="Filter by status: UNREAD, ACKNOWLEDGED, RESOLVED"
    ),
    limit: int = Query(50, ge=1, le=200, description="Max alerts to return"),
    offset: int = Query(0, ge=0, description="Number of alerts to skip"),
    db: Session = Depends(get_db),
):
    """
    Retrieve all alerts, ordered by most recent first.
    Optionally filter by status.
    """
    query = db.query(Alert)
    if status:
        if status not in VALID_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        query = query.filter(Alert.status == status)
    return query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/count")
async def count_alerts(
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """
    Returns the count of alerts, optionally filtered by status.
    Useful for the frontend badge (e.g., '5 unread alerts').
    """
    query = db.query(Alert)
    if status:
        if status not in VALID_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
            )
        query = query.filter(Alert.status == status)
    return {"count": query.count(), "status_filter": status}


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: UUID, db: Session = Depends(get_db)):
    """
    Retrieve a single alert by its UUID.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return alert


# ==============================================================================
# UPDATE STATUS
# ==============================================================================
@router.patch("/{alert_id}/status", response_model=AlertResponse)
async def update_alert_status(
    alert_id: UUID,
    update: AlertStatusUpdate,
    db: Session = Depends(get_db),
):
    """
    Update the status of an alert.
    Valid transitions: UNREAD -> ACKNOWLEDGED -> RESOLVED.
    When resolved, the resolved_at timestamp is set automatically.
    """
    if update.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{update.status}'. Must be one of: {', '.join(VALID_STATUSES)}"
        )

    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    # Validate status transition
    if alert.status == "RESOLVED":
        raise HTTPException(
            status_code=409,
            detail="Cannot change status of a resolved alert."
        )

    if alert.status == "ACKNOWLEDGED" and update.status == "UNREAD":
        raise HTTPException(
            status_code=409,
            detail="Cannot revert an acknowledged alert back to unread."
        )

    alert.status = update.status

    # Auto-set resolved_at when status changes to RESOLVED
    if update.status == "RESOLVED":
        alert.resolved_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(alert)
    return alert