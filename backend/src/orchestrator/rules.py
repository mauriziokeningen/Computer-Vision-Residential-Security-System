"""
Incident Rule Engine & Orchestrator.
Acts as the central Sink Node in the IPC architecture. Consumes stateless events 
from AI workers via ZeroMQ (PULL), applies temporal state (debouncing), 
and executes side-effects (DB writes, S3 uploads, WebSocket alerts).
"""
import zmq
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RuleEngine")

# --- Boundary Contract ---
# ASSUMPTION: Upstream AI workers MUST connect via zmq.PUSH. 
# This orchestrator uses zmq.PULL to act as a load-balanced sink.
RECEIVER_PORT = "tcp://127.0.0.1:5556"

PRIORITY_LOW = "LOW"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_HIGH = "HIGH"
PRIORITY_CRITICAL = "CRITICAL"

# --- Architectural Trade-off: Local Memory vs External Cache ---
# We use a native Python dict for O(1) state tracking instead of Redis. 
# TRADE-OFF: State is lost on container restart. This is acceptable for physical security 
# (a reboot should immediately trigger fresh alerts). It saves ~2-5ms of network I/O per frame, 
# preventing the ZeroMQ PULL socket from backing up.
COOLDOWN_PERIODS = {
    "RN-02": 15.0,            
    "WEAPON_DETECTED": 10.0,  
    "RN-04": 15.0,            
    "RN-05": 30.0,            
    "RN-06": 10.0,            
    "RN-07": 30.0             
}

last_incident_times = {}

def _check_cooldown(camera_id: str, rule_id: str) -> bool:
    current_time = time.time()
    cache_key = f"{camera_id}_{rule_id}"
    last_time = last_incident_times.get(cache_key, 0)
    
    if (current_time - last_time) >= COOLDOWN_PERIODS.get(rule_id, 10.0):
        last_incident_times[cache_key] = current_time
        return True
    return False

def _get_db_session():
    # TECH DEBT: Instantiating a new session per event is expensive.
    # For V2 scaling (>5 cameras), we must implement a SQLAlchemy Connection Pool 
    # or pass a persistent session generator to avoid exhausting DB connections.
    from src.database.session import SessionLocal
    return SessionLocal()

def _create_incident(db, event: Dict[str, Any], rule_triggered: str, priority: str) -> Any:
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
    logger.debug(f"Incident created: {incident.id} (Rule: {rule_triggered}, Priority: {priority})")
    return incident

def _create_alert(db, incident_id, message: str) -> Any:
    from src.database.models import Alert
    alert = Alert(incident_id=incident_id, message=message)
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.debug(f"Alert created: {alert.id} -> {message}")
    return alert

def _save_evidence(incident_id: str, camera_id: str, frame_data=None) -> Optional[str]:
    if not frame_data:
        return None
    try:
        from src.utils.s3_client import upload_incident_clip

        # If frame_data is base64 string, decode it to bytes
        if isinstance(frame_data, str):
            import base64
            frame_data = base64.b64decode(frame_data)

        # TECH DEBT: Synchronous network I/O.
        # Uploading to MinIO/S3 blocks the main ZMQ event loop. If the network degrades,
        # the IPC bus will back up. V2 must offload this to a Celery background worker.
        object_name = upload_incident_clip(
            file_data=frame_data,
            incident_id=str(incident_id),
            camera_id=camera_id,
            filename=f"frame_{datetime.utcnow().strftime('%H%M%S')}.jpg",
            content_type="image/jpeg",
        )
        logger.debug(f"Evidence saved: {object_name}")
        return object_name
    except Exception as e:
        logger.error(f"Failed to save evidence: {e}")
        return None

def _evaluate_face_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

            if _check_cooldown(camera_id, "RN-02"):
                incident = _create_incident(db, event, "RN-02", PRIORITY_MEDIUM)
                _create_alert(db, incident.id, f"Persona desconocida detectada en {camera_id} ({timestamp}) - Confianza: {conf_pct:.1f}%")
                _save_evidence(incident.id, camera_id, frame_data)
                logger.warning(f"[SECURITY ALERT] Unknown person at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")
        else:
            face_summary["known_names"].append(name)
            logger.info(f"[ACCESS GRANTED] Resident: {name} at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

    return face_summary

def _evaluate_weapon_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

        if _check_cooldown(camera_id, "WEAPON_DETECTED"):
            incident = _create_incident(db, event, "WEAPON_DETECTED", PRIORITY_HIGH)
            _create_alert(db, incident.id, f"ARMA DETECTADA: {weapon_class} en {camera_id} ({timestamp}) - Confianza: {conf_pct:.1f}%")
            _save_evidence(incident.id, camera_id, frame_data)
            logger.warning(f"[WEAPON ALERT] {weapon_class} detected at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

    return weapon_summary

def _evaluate_pose_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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

        aggressive_actions = {"punch", "kick", "push", "fight", "struggle", "golpe", "patada", "empujon", "pelea", "forcejeo"}
        fall_actions = {"fall", "caida"}

        if action.lower() in aggressive_actions:
            pose_summary["aggression_detected"] = True
            
            if _check_cooldown(camera_id, "RN-04"):
                incident = _create_incident(db, event, "RN-04", PRIORITY_HIGH)
                _create_alert(db, incident.id, f"AGRESION DETECTADA: {action} en {camera_id} ({timestamp}) - Confianza: {conf_pct:.1f}%")
                _save_evidence(incident.id, camera_id, frame_data)
                logger.warning(f"[AGGRESSION ALERT] {action} at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

        elif action.lower() in fall_actions:
            pose_summary["fall_detected"] = True
            
            if _check_cooldown(camera_id, "RN-05"):
                incident = _create_incident(db, event, "RN-05", PRIORITY_MEDIUM)
                _create_alert(db, incident.id, f"CAIDA DETECTADA en {camera_id} ({timestamp}) - Confianza: {conf_pct:.1f}%")
                _save_evidence(incident.id, camera_id, frame_data)
                logger.warning(f"[FALL ALERT] Fall detected at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

    return pose_summary

def _evaluate_compound_event(db, event: Dict[str, Any], face_summary: Optional[Dict], weapon_summary: Optional[Dict], pose_summary: Optional[Dict]) -> None:
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")
    frame_data = event.get("frame_data")

    unknown = face_summary and face_summary.get("unknown_detected", False)
    weapon = weapon_summary and weapon_summary.get("weapon_detected", False)
    aggression = pose_summary and pose_summary.get("aggression_detected", False)
    fall = pose_summary and pose_summary.get("fall_detected", False)
    known_names = face_summary.get("known_names", []) if face_summary else []

    if unknown and (weapon or aggression):
        if _check_cooldown(camera_id, "RN-06"):
            threats = []
            if weapon: threats.extend(weapon_summary.get("weapons", []))
            if aggression: threats.extend(pose_summary.get("actions", []))

            incident = _create_incident(db, event, "RN-06", PRIORITY_CRITICAL)
            _create_alert(db, incident.id, f"ALERTA CRITICA: Persona desconocida con amenaza activa ({', '.join(threats)}) en {camera_id} ({timestamp})")
            _save_evidence(incident.id, camera_id, frame_data)
            logger.critical(f"[CRITICAL] Unknown person + active threat at {timestamp} on {camera_id}")

    if known_names and fall:
        if _check_cooldown(camera_id, "RN-07"):
            incident = _create_incident(db, event, "RN-07", PRIORITY_MEDIUM)
            _create_alert(db, incident.id, f"ALERTA ASISTENCIAL: Residente {', '.join(known_names)} detecto caida en {camera_id} ({timestamp})")
            _save_evidence(incident.id, camera_id, frame_data)
            logger.warning(f"[ASSISTENTIAL] Resident fall detected at {timestamp} on {camera_id}")

class EventAccumulator:
    """
    Temporal Synchronization Buffer.
    Different AI models (Face, Pose) process frames at different latencies. 
    This buffer captures events within a small temporal window to accurately 
    evaluate cross-module compound rules (e.g., Threat + Unknown Face).
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
        return self.events.get("face"), self.events.get("weapons"), self.events.get("pose")

    def reset(self):
        self.events.clear()
        self.last_reset = time.time()

def start_orchestrator() -> None:
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

                if accumulator.should_evaluate():
                    face_s, weapon_s, pose_s = accumulator.get_summaries()
                    if any([face_s, weapon_s, pose_s]):
                        _evaluate_compound_event(db, event, face_s, weapon_s, pose_s)
                    accumulator.reset()

            finally:
                # CHESTERTON'S FENCE: Always close the DB session in the finally block.
                # Failing to release this connection back to the OS will cause a PostgreSQL
                # connection pool exhaustion (FATAL: sorry, too many clients already) in minutes.
                db.close()

        except Exception as e:
            logger.error(f"Error processing event: {e}")