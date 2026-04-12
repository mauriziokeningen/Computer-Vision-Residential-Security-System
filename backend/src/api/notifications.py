from typing import Optional
from uuid import UUID
from datetime import datetime, timezone


async def notify_new_alert(
    alert_id: UUID,
    incident_id: Optional[UUID],
    message: str,
    status: str,
    created_at: datetime,
) -> None:
    """Broadcast a new alert notification to all connected clients."""
    from src.api.ws_manager import manager

    await manager.broadcast({
        "event_type": "NEW_ALERT",
        "data": {
            "alert_id": str(alert_id),
            "incident_id": str(incident_id) if incident_id else None,
            "message": message,
            "status": status,
            "created_at": created_at.isoformat(),
        }
    })


async def notify_alert_status_change(
    alert_id: UUID,
    old_status: str,
    new_status: str,
    resolved_at: Optional[datetime] = None,
) -> None:
    """Broadcast an alert status change to all connected clients."""
    from src.api.ws_manager import manager

    await manager.broadcast({
        "event_type": "ALERT_STATUS_CHANGED",
        "data": {
            "alert_id": str(alert_id),
            "old_status": old_status,
            "new_status": new_status,
            "resolved_at": resolved_at.isoformat() if resolved_at else None,
        }
    })


async def notify_alert_count_update(
    unread_count: int,
    acknowledged_count: int,
    resolved_count: int,
) -> None:
    #Broadcast updated alert counts to all connected clients (for badge updates).
    from src.api.ws_manager import manager

    await manager.broadcast({
        "event_type": "ALERT_COUNT_UPDATE",
        "data": {
            "unread": unread_count,
            "acknowledged": acknowledged_count,
            "resolved": resolved_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    })