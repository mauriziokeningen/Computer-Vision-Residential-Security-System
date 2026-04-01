"""
Incident Rule Engine for the Residential Security System.
Receives JSON events from AI modules via ZeroMQ, evaluates business rules, 
and creates incidents, alerts, and evidence records in the database.
"""
import zmq
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RuleEngine")

RECEIVER_PORT = "tcp://127.0.0.1:5556"

# --- Priority Levels ---
PRIORITY_LOW = "LOW"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_HIGH = "HIGH"
PRIORITY_CRITICAL = "CRITICAL"


def _get_db_session():
    """Creates a database session for the orchestrator process."""
    from src.database.session import SessionLocal
    return SessionLocal()


def _create_incident(db, event: Dict[str, Any], rule_triggered: str, priority: str) -> Any:
    """
    Creates an incident record in the database.
    Returns the created incident.
    """
    from src.database.models import Incident

    metadata = {
        "rule_triggered": rule_triggered,
        "priority": priority,
        "module": event.get("module", "unknown"),
        "camera_id": event.get("camera_id", "unknown"),
        "timestamp": event.get("timestamp", datetime.utcnow().isoformat()),
        "detections": event.get("detections", []),
    }

    incident = Incident(incident_metadata=metadata)
    db.add(incident)
    db.commit()
    db.refresh(incident)

    logger.info(f"Incident created: {incident.id} (Rule: {rule_triggered}, Priority: {priority})")
    return incident


def _create_alert(db, incident_id, message: str) -> Any:
    """Creates an alert linked to an incident."""
    from src.database.models import Alert

    alert = Alert(
        incident_id=incident_id,
        message=message,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)

    logger.info(f"Alert created: {alert.id} -> {message}")
    return alert


def _save_evidence(incident_id: str, camera_id: str, frame_data: Optional[bytes] = None) -> Optional[str]:
    """
    Saves evidence to MinIO if frame data is available.
    Returns the object path or None.
    """
    if not frame_data:
        return None

    try:
        from src.utils.s3_client import upload_incident_clip

        object_name = upload_incident_clip(
            file_data=frame_data,
            incident_id=str(incident_id),
            camera_id=camera_id,
            filename=f"frame_{datetime.utcnow().strftime('%H%M%S')}.jpg",
            content_type="image/jpeg",
        )
        logger.info(f"Evidence saved: {object_name}")
        return object_name
    except Exception as e:
        logger.error(f"Failed to save evidence: {e}")
        return None


# RULE EVALUATORS (RN-01 - RN-07

def _evaluate_face_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Evaluates facial recognition events.
    RN-01: Access authorized if cosine similarity >= threshold
    RN-02: Unknown person if no match found
    Returns detection summary for compound event evaluation.
    """
    detections = event.get("detections", [])
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")
    frame_data = event.get("frame_data")

    face_summary = {"unknown_detected": False, "known_names": []}

    for detection in detections:
        name = detection.get("name", "unknown_person")
        confidence = detection.get("confidence", 0.0)
        conf_pct = confidence * 100

        if name == "unknown_person":
            face_summary["unknown_detected"] = True

            # RN-02: Unknown person detected
            incident = _create_incident(db, event, "RN-02", PRIORITY_MEDIUM)
            _create_alert(
                db, incident.id,
                f"Persona desconocida detectada en {camera_id} "
                f"({timestamp}) - Confianza: {conf_pct:.1f}%"
            )
            _save_evidence(incident.id, camera_id, frame_data)

            logger.warning(
                f"[SECURITY ALERT] Unknown person at {timestamp} "
                f"on {camera_id} (Confidence: {conf_pct:.1f}%)"
            )
        else:
            face_summary["known_names"].append(name)

            # RN-01: Access authorized
            logger.info(
                f"[ACCESS GRANTED] Resident: {name} at {timestamp} "
                f"on {camera_id} (Similarity: {conf_pct:.1f}%)"
            )

    return face_summary


def _evaluate_weapon_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Evaluates weapon detection events.
    Creates a HIGH priority incident when a gun or knife is detected.
    """
    detections = event.get("detections", [])
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")
    frame_data = event.get("frame_data")

    weapon_summary = {"weapon_detected": False, "weapons": []}

    for detection in detections:
        weapon_class = detection.get("class", "unknown")
        confidence = detection.get("confidence", 0.0)
        conf_pct = confidence * 100

        weapon_summary["weapon_detected"] = True
        weapon_summary["weapons"].append(weapon_class)

        incident = _create_incident(db, event, "WEAPON_DETECTED", PRIORITY_HIGH)
        _create_alert(
            db, incident.id,
            f"ARMA DETECTADA: {weapon_class} en {camera_id} "
            f"({timestamp}) - Confianza: {conf_pct:.1f}%"
        )
        _save_evidence(incident.id, camera_id, frame_data)

        logger.warning(
            f"[WEAPON ALERT] {weapon_class} detected at {timestamp} "
            f"on {camera_id} (Confidence: {conf_pct:.1f}%)"
        )

    return weapon_summary


def _evaluate_pose_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Evaluates body language / pose events.
    RN-04: Sustained aggression alert
    RN-05: Fall detection (assistential alert)
    """
    detections = event.get("detections", [])
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")
    frame_data = event.get("frame_data")

    pose_summary = {"aggression_detected": False, "fall_detected": False, "actions": []}

    for detection in detections:
        action = detection.get("action", "unknown")
        confidence = detection.get("confidence", 0.0)
        conf_pct = confidence * 100

        pose_summary["actions"].append(action)

        aggressive_actions = {"punch", "kick", "push", "fight", "struggle",
                              "golpe", "patada", "empujon", "pelea", "forcejeo"}
        fall_actions = {"fall", "caida"}

        if action.lower() in aggressive_actions:
            pose_summary["aggression_detected"] = True

            # RN-04: Sustained aggression
            incident = _create_incident(db, event, "RN-04", PRIORITY_HIGH)
            _create_alert(
                db, incident.id,
                f"AGRESION DETECTADA: {action} en {camera_id} "
                f"({timestamp}) - Confianza: {conf_pct:.1f}%"
            )
            _save_evidence(incident.id, camera_id, frame_data)

            logger.warning(
                f"[AGGRESSION ALERT] {action} at {timestamp} "
                f"on {camera_id} (Confidence: {conf_pct:.1f}%)"
            )

        elif action.lower() in fall_actions:
            pose_summary["fall_detected"] = True

            # RN-05: Fall detection
            incident = _create_incident(db, event, "RN-05", PRIORITY_MEDIUM)
            _create_alert(
                db, incident.id,
                f"CAIDA DETECTADA en {camera_id} "
                f"({timestamp}) - Confianza: {conf_pct:.1f}%"
            )
            _save_evidence(incident.id, camera_id, frame_data)

            logger.warning(
                f"[FALL ALERT] Fall detected at {timestamp} "
                f"on {camera_id} (Confidence: {conf_pct:.1f}%)"
            )

    return pose_summary


def _evaluate_compound_event(
    db,
    event: Dict[str, Any],
    face_summary: Optional[Dict],
    weapon_summary: Optional[Dict],
    pose_summary: Optional[Dict],
) -> None:
    """
    Evaluates compound events by combining signals from multiple modules.
    RN-06: Critical alert = unknown person + (aggression OR weapon)
    RN-07: Assistential alert = known resident + fall
    """
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")
    frame_data = event.get("frame_data")

    unknown = face_summary and face_summary.get("unknown_detected", False)
    weapon = weapon_summary and weapon_summary.get("weapon_detected", False)
    aggression = pose_summary and pose_summary.get("aggression_detected", False)
    fall = pose_summary and pose_summary.get("fall_detected", False)
    known_names = face_summary.get("known_names", []) if face_summary else []

    # RN-06: Critical alert (unknown + weapon or aggression)
    if unknown and (weapon or aggression):
        threats = []
        if weapon:
            threats.extend(weapon_summary.get("weapons", []))
        if aggression:
            threats.extend(pose_summary.get("actions", []))

        incident = _create_incident(db, event, "RN-06", PRIORITY_CRITICAL)
        _create_alert(
            db, incident.id,
            f"ALERTA CRITICA: Persona desconocida con amenaza activa "
            f"({', '.join(threats)}) en {camera_id} ({timestamp})"
        )
        _save_evidence(incident.id, camera_id, frame_data)

        logger.critical(
            f"[CRITICAL] Unknown person + active threat at {timestamp} on {camera_id}"
        )

    # RN-07: Assistential alert (known resident + fall)
    if known_names and fall:
        incident = _create_incident(db, event, "RN-07", PRIORITY_MEDIUM)
        _create_alert(
            db, incident.id,
            f"ALERTA ASISTENCIAL: Residente {', '.join(known_names)} "
            f"detecto caida en {camera_id} ({timestamp})"
        )
        _save_evidence(incident.id, camera_id, frame_data)

        logger.warning(
            f"[ASSISTENTIAL] Resident fall detected at {timestamp} on {camera_id}"
        )


# EVENT ACCUMULATOR (for compound detection within time window)

class EventAccumulator:
    """
    Accumulates events from different modules within a time window
    to enable compound event evaluation.
    """
    def __init__(self, window_seconds: float = 2.0):
        self.window = window_seconds
        self.events: Dict[str, Dict] = {}
        self.last_reset = time.time()

    def add(self, module: str, summary: Dict) -> None:
        self.events[module] = summary

    def should_evaluate(self) -> bool:
        return (time.time() - self.last_reset) >= self.window

    def get_summaries(self):
        return (
            self.events.get("face"),
            self.events.get("weapons"),
            self.events.get("pose"),
        )

    def reset(self):
        self.events.clear()
        self.last_reset = time.time()


# MAIN ORCHESTRATOR LOOP

def start_orchestrator() -> None:
    """
    Receives JSON events from all AI modules via ZeroMQ and evaluates
    business rules. Creates incidents, alerts, and saves evidence.
    """
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(RECEIVER_PORT)

    logger.info(f"Rule engine started. Waiting for events on {RECEIVER_PORT}")

    accumulator = EventAccumulator(window_seconds=2.0)

    while True:
        try:
            event = receiver.recv_json()
            module = event.get("module")
            db = _get_db_session()

            try:
                # Route to the appropriate evaluator
                if module == "face":
                    summary = _evaluate_face_event(db, event)
                    accumulator.add("face", summary)

                elif module == "weapons":
                    summary = _evaluate_weapon_event(db, event)
                    accumulator.add("weapons", summary)

                elif module == "pose":
                    summary = _evaluate_pose_event(db, event)
                    accumulator.add("pose", summary)

                else:
                    logger.warning(f"Unknown module: {module}")

                # Evaluate compound events after time window
                if accumulator.should_evaluate():
                    face_s, weapon_s, pose_s = accumulator.get_summaries()
                    if any([face_s, weapon_s, pose_s]):
                        _evaluate_compound_event(
                            db, event, face_s, weapon_s, pose_s
                        )
                    accumulator.reset()

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error processing event: {e}")