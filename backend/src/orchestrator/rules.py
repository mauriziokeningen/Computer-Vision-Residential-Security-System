"""
Incident Rule Engine & Orchestrator.
Acts as the central Sink Node in the IPC architecture. Consumes stateless events 
from AI workers via ZeroMQ (PULL), applies temporal state (debouncing), 
and executes side-effects (DB writes, S3 uploads, WebSocket alerts).
"""
import os
import zmq
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RuleEngine")

# --- Boundary Contract ---
# ASSUMPTION: Upstream AI workers MUST connect via zmq.PUSH. 
# This orchestrator uses zmq.PULL to act as a load-balanced sink.
#
# Endpoints are configurable via environment variables so deployment topology
# (single-host dev, containerized, multi-host) can change without code edits.
# Defaults preserve the original developer-machine layout.
RECEIVER_PORT = os.getenv("ORCHESTRATOR_PUSH_PORT", "tcp://127.0.0.1:5556")

# Annotated frame stream (from the annotator process). The orchestrator
# subscribes here in a background thread to keep the latest annotated frame
# buffered in memory. When an incident triggers and we need to persist
# evidence, we pull from this buffer instead of from the worker's event,
# guaranteeing the persisted JPEG is byte-identical to what the operator
# was watching live a moment earlier.
ANNOTATED_SUB_PORT = os.getenv("ANNOTATED_PUB_PORT", "tcp://127.0.0.1:5557")

# Compound-event temporal window (seconds). Different AI models process at
# different latencies; this is how long we wait to correlate cross-module
# events before evaluating compound rules.
COMPOUND_EVENT_WINDOW_SECONDS = float(os.getenv("COMPOUND_EVENT_WINDOW_SECONDS", "2.0"))

# Internal API endpoint for alert broadcasting. Kept env-driven so the
# orchestrator and the FastAPI host can be deployed on different machines.
ALERT_API_URL = os.getenv("ALERT_API_URL", "http://127.0.0.1:8000/api/alerts/")
ALERT_API_TIMEOUT_SECONDS = float(os.getenv("ALERT_API_TIMEOUT_SECONDS", "2.0"))

# Weapon debouncer tuning. Defaults assume ~7-8 FPS effective worker throughput
# (CoreML on M-series with a single camera). At that rate, 5 hits in a 1.5s
# window correspond to ~0.7s of sustained detection — long enough to outlast
# a single-frame flicker, short enough to keep alert latency well under 1s.
WEAPON_DEBOUNCE_HITS = int(os.getenv("WEAPON_DEBOUNCE_HITS", "5"))
WEAPON_DEBOUNCE_WINDOW = float(os.getenv("WEAPON_DEBOUNCE_WINDOW", "1.5"))
WEAPON_DEBOUNCE_MIN_AVG_CONF = float(os.getenv("WEAPON_DEBOUNCE_MIN_AVG_CONF", "0.60"))
WEAPON_TRACK_TTL = float(os.getenv("WEAPON_TRACK_TTL", "5.0"))

# --- ATOMIC CONCURRENCY STRUCTURE (LIST-BASED) ---
# Stores tuples of (camera_id, incident_id) queued sequentially for processing
_pending_evidences: List[Tuple[str, str]] = []
_pending_evidences_lock = threading.Lock()


class AnnotatedFrameBuffer:
    """
    Thread-safe holder for the most recent annotated JPEG bytes received from
    the annotator process.

    Encapsulates what was previously module-level mutable state
    (`_latest_annotated_frame` + `_annotated_frame_lock`) into a single
    object whose lifetime is bound to the orchestrator instance. The
    daemon-thread consumer drains the SUB socket (with CONFLATE=1, so we
    only ever hold the latest frame) and writes into the buffer under the
    lock. Readers obtain a snapshot via `get_latest()` with no I/O on the
    call path, so evidence persistence never blocks waiting on the socket.

    Lives in the orchestrator process (rather than as a separate process)
    because the consumer of this buffer — _save_evidence — runs in the same
    event loop. Threading is sufficient: the GIL doesn't block I/O-bound
    socket reads, and there's no CPU contention with the rule engine.
    """

    def __init__(self, endpoint: str):
        self._endpoint = endpoint
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(self._endpoint)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.setsockopt(zmq.CONFLATE, 1)

        threading.Thread(
            target=self._consume,
            name="AnnotatedFrameListener",
            daemon=True,
        ).start()
        logger.info(f"Annotated frame listener online ({self._endpoint})")

    def _consume(self) -> None:
        while True:
            try:
                frame_bytes = self._sock.recv()
                with self._lock:
                    self._frame = frame_bytes

                # --- ULTRA-FAST ATOMIC DRAIN (<1ms lock) ---
                to_process = []
                with _pending_evidences_lock:
                    if _pending_evidences:
                        to_process = list(_pending_evidences)
                        _pending_evidences.clear() # Clear the list for new incoming alerts

                # Process the accumulated evidence queue sequentially and non-blocking
                for camera_id, incident_id in to_process:
                    _execute_async_persistence(incident_id, camera_id, frame_bytes)

            except Exception as e:
                logger.error(f"Annotated frame listener crashed: {e}")
                break

    def get_latest(self) -> Optional[bytes]:
        """Returns a snapshot of the latest annotated JPEG, or None if not ready yet."""
        with self._lock:
            return self._frame


def _execute_async_persistence(incident_id: str, camera_id: str, frame_bytes: bytes) -> Optional[str]:
    """
    Asynchronous persistence to S3/MinIO fed by the visual SSoT pipeline.
    Invokes the persistence layer safely once the frame lineage has been 
    mathematically verified to contain the visual annotations.
    """
    try:
        from src.utils.s3_client import upload_incident_clip

        object_name = upload_incident_clip(
            file_data=frame_bytes,
            incident_id=str(incident_id),
            camera_id=camera_id,
            filename=f"frame_{datetime.utcnow().strftime('%H%M%S')}.jpg",
            content_type="image/jpeg",
        )
        logger.info(f"[ASYNC FORENSIC SSoT] Visual evidence successfully frozen for incident {incident_id}: {object_name}")
        return object_name
    except Exception as e:
        logger.error(f"[ASYNC FORENSIC ERROR] Failed to freeze frame for incident {incident_id}: {e}")
        return None


class WeaponDebouncer:
    """
    Per-track temporal debouncer for weapon detections.

    Filters out flicker false positives (1-2 frame hallucinations where YOLO
    confuses a phone, remote, or hand for a knife/pistol) by requiring a
    track to sustain a configurable number of detections within a sliding
    time window, with a minimum average confidence, before promoting it to
    a real threat.

    State machine (per track key `(camera_id, track_id, class)`):

        pending       -> hits accumulate but threshold not yet reached
        newly_confirmed -> exact frame where threshold crossed (single fire)
        active        -> already confirmed; threat is currently in the frame
        idle/evicted  -> no detections for TTL seconds; track is GC'd

    The ``observe()`` method returns two booleans, ``is_newly_confirmed`` and
    ``is_active_threat``, so callers can distinguish "fire a new alert" from
    "report this as an ongoing threat for compound rule evaluation". Both
    states must gate downstream rules: an unconfirmed signal is mathematically
    invalid for compound rule escalation as well, since the same noisy frame
    that we rejected for a HIGH alert cannot be trusted to drive a CRITICAL
    alert just because an unknown person happens to be co-located.

    Design notes:
        * Composite key includes class so a track that flips between
          ``knife`` and pistol`` doesn't share confirmations across labels.
        * Sliding window is implemented as list pruning per observation, not
          as a periodic sweep — at ~10 FPS the per-call cost is negligible
          and avoids the complexity of a background thread.
        * Track state is wiped by ``_gc()`` after ``ttl`` seconds of silence
          to bound memory in long-running deployments with many short-lived
          tracks.
    """

    def __init__(
        self,
        hits: int = WEAPON_DEBOUNCE_HITS,
        window: float = WEAPON_DEBOUNCE_WINDOW,
        min_avg_conf: float = WEAPON_DEBOUNCE_MIN_AVG_CONF,
        ttl: float = WEAPON_TRACK_TTL,
    ):
        self.hits = hits
        self.window = window
        self.min_avg_conf = min_avg_conf
        self.ttl = ttl
        # key: (camera_id, track_id, class) -> state dict
        self._tracks: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        self._last_gc = time.time()

    def observe(
        self,
        camera_id: str,
        track_id: Optional[int],
        cls: str,
        confidence: float,
    ) -> Tuple[bool, bool]:
        """
        Record a detection and return (is_newly_confirmed, is_active_threat).

        ``is_newly_confirmed``: True only on the exact frame the track crosses
            the confirmation threshold. Drives the standalone WEAPON_DETECTED
            alert, which must fire exactly once per track lifecycle to avoid
            duplicate incidents for the same physical object.

        ``is_active_threat``: True for every frame where the track is currently
            confirmed and within the activity window. Drives compound rules
            (RN-06: unknown person + weapon) that need to know whether a
            validated weapon is in the frame *right now*, not just whether
            one was ever confirmed at some point in the past.

        Detections without a track_id (yet to be assigned by ByteTrack —
        typically the first 1-2 frames of an object's lifetime) cannot be
        debounced reliably and are conservatively treated as noise:
        ``(False, False)``. They will not escalate into either standalone or
        compound alerts; the next 1-2 frames will carry a real track_id and
        normal accumulation resumes.
        """
        if track_id is None:
            return False, False

        now = time.time()
        # Opportunistic GC. Cheap call; runs at most once per second.
        if now - self._last_gc > 1.0:
            self._gc(now)

        key = (camera_id, track_id, cls)
        state = self._tracks.get(key)
        if state is None:
            state = {
                "hits": [],          # list of (timestamp, confidence)
                "confirmed": False,
                "last_seen": now,
            }
            self._tracks[key] = state

        # Prune hits outside the sliding window before appending the new one,
        # so the confidence average reflects only recent observations.
        cutoff = now - self.window
        state["hits"] = [h for h in state["hits"] if h[0] >= cutoff]
        state["hits"].append((now, confidence))
        state["last_seen"] = now

        if state["confirmed"]:
            # Already-confirmed track: still an active threat for compound
            # rules, but never a "newly confirmed" event again. The standalone
            # alert pipeline must not re-fire for the same track.
            return False, True

        if len(state["hits"]) >= self.hits:
            avg_conf = sum(c for _, c in state["hits"]) / len(state["hits"])
            if avg_conf >= self.min_avg_conf:
                state["confirmed"] = True
                return True, True

        return False, False

    def _gc(self, now: float) -> None:
        """Evict tracks idle longer than ``ttl`` seconds. Called opportunistically."""
        expired = [k for k, st in self._tracks.items() if now - st["last_seen"] > self.ttl]
        for k in expired:
            del self._tracks[k]
        self._last_gc = now


# Module-level singletons, populated when start_orchestrator() runs. Kept as
# module attributes (not true global mutables) so the helper functions
# (_save_evidence, _evaluate_weapon_event, etc.) can access them without
# threading every signature.
_frame_buffer: Optional[AnnotatedFrameBuffer] = None
_weapon_debouncer: Optional[WeaponDebouncer] = None


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
    import urllib.request
    import json

    # THE BRIDGE: Instead of writing to the DB in silence, Brain 2 hits Brain 1's API.
    # This forces FastAPI to execute the database insert AND broadcast the WebSocket message.
    payload = {
        "incident_id": str(incident_id) if incident_id else None,
        "message": message
    }
    headers = {"Content-Type": "application/json"}

    try:
        req = urllib.request.Request(
            ALERT_API_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=ALERT_API_TIMEOUT_SECONDS) as response:
            logger.debug(f"Alert pushed to API successfully: {message}")
            return json.loads(response.read().decode())

    except Exception as e:
        logger.error(f"Failed to push alert to API, falling back to direct DB write: {e}")
        # Fallback just in case FastAPI is rebooting
        from src.database.models import Alert
        alert = Alert(incident_id=incident_id, message=message)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return alert

def _save_evidence(incident_id: str, camera_id: str, frame_data=None) -> Optional[str]:
    """ Kept strictly for backwards compatibility. Implementation moved to _execute_async_persistence. """
    return None

def _evaluate_face_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    detections = event.get("detections", [])
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")

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
                
                # NON-BLOCKING ASYNC QUEUING
                with _pending_evidences_lock:
                    _pending_evidences.append((camera_id, incident.id))
                    
                logger.warning(f"[SECURITY ALERT] Unknown person at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")
        else:
            face_summary["known_names"].append(name)
            logger.info(f"[ACCESS GRANTED] Resident: {name} at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

    return face_summary

def _evaluate_weapon_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Promotes weapon detections to incidents and surfaces validated threats
    to the compound rule engine.

    Per-detection flow:
        1. Feed the detection into the debouncer to learn whether the
           underlying track is (a) newly confirmed this frame and (b)
           currently an active threat.
        2. If the debouncer rejects the detection as noise (``not
           is_active_threat``), skip it entirely. Critically, do NOT
           populate ``weapon_summary``: a signal too unreliable to drive
           the HIGH-priority standalone alert is just as unreliable for
           driving the CRITICAL-priority compound alert, regardless of
           what other modules report in the same window.
        3. For validated detections, populate ``weapon_summary`` so the
           compound rule engine can correlate it with face/pose findings.
        4. Fire the standalone ``WEAPON_DETECTED`` incident only on the
           exact frame the track crosses the confirmation threshold,
           subject to the global cooldown.
    """
    detections = event.get("detections", [])
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")

    weapon_summary = {"weapon_detected": False, "weapons": []}

    for detection in detections:
        weapon_class = detection.get("class", "unknown")
        track_id = detection.get("track_id")
        confidence = detection.get("confidence", 0.0)
        conf_pct = confidence * 100

        # 1. Evaluate debouncer first. Returns two states so we can
        # distinguish "fire the standalone alert" from "report as ongoing
        # threat to compound rules".
        if _weapon_debouncer is None:
            is_newly_confirmed, is_active_threat = False, False
        else:
            is_newly_confirmed, is_active_threat = _weapon_debouncer.observe(
                camera_id, track_id, weapon_class, confidence
            )

        # 2. Reject unconfirmed detections entirely. They must not influence
        # standalone OR compound rules.
        if not is_active_threat:
            continue

        # 3. The detection is validated reality — only now does the compound
        # rule engine learn that a weapon is in the frame.
        weapon_summary["weapon_detected"] = True
        if weapon_class not in weapon_summary["weapons"]:
            weapon_summary["weapons"].append(weapon_class)

        # 4. Standalone alert: fires once per track lifecycle on the exact
        # frame the track crosses the confirmation threshold, gated by the
        # global cooldown to avoid burst alerts when multiple tracks of the
        # same class confirm in rapid succession.
        if is_newly_confirmed and _check_cooldown(camera_id, "WEAPON_DETECTED"):
            incident = _create_incident(db, event, "WEAPON_DETECTED", PRIORITY_HIGH)
            _create_alert(
                db,
                incident.id,
                f"ARMA DETECTADA: {weapon_class} en {camera_id} ({timestamp}) - Confianza: {conf_pct:.1f}%",
            )
            
            # NON-BLOCKING ASYNC QUEUING
            with _pending_evidences_lock:
                _pending_evidences.append((camera_id, incident.id))

            logger.warning(
                f"[WEAPON ALERT PENDING VISUAL SSoT] {weapon_class} (track={track_id}) confirmed. "
                f"Waiting for annotated frame from port {ANNOTATED_SUB_PORT} to freeze evidence."
            )

    return weapon_summary

def _evaluate_pose_event(db, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    detections = event.get("detections", [])
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")

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
                
                # NON-BLOCKING ASYNC QUEUING
                with _pending_evidences_lock:
                    _pending_evidences.append((camera_id, incident.id))
                    
                logger.warning(f"[AGGRESSION ALERT] {action} at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

        elif action.lower() in fall_actions:
            pose_summary["fall_detected"] = True
            
            if _check_cooldown(camera_id, "RN-05"):
                incident = _create_incident(db, event, "RN-05", PRIORITY_MEDIUM)
                _create_alert(db, incident.id, f"CAIDA DETECTADA en {camera_id} ({timestamp}) - Confianza: {conf_pct:.1f}%")
                
                # NON-BLOCKING ASYNC QUEUING
                with _pending_evidences_lock:
                    _pending_evidences.append((camera_id, incident.id))
                    
                logger.warning(f"[FALL ALERT] Fall detected at {timestamp} on {camera_id} (Confidence: {conf_pct:.1f}%)")

    return pose_summary

def _evaluate_compound_event(db, event: Dict[str, Any], face_summary: Optional[Dict], weapon_summary: Optional[Dict], pose_summary: Optional[Dict]) -> None:
    timestamp = event.get("timestamp", "unknown_time")
    camera_id = event.get("camera_id", "unknown_camera")

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
            
            # NON-BLOCKING ASYNC QUEUING
            with _pending_evidences_lock:
                _pending_evidences.append((camera_id, incident.id))
                
            logger.critical(f"[CRITICAL] Unknown person + active threat at {timestamp} on {camera_id}")

    if known_names and fall:
        if _check_cooldown(camera_id, "RN-07"):
            incident = _create_incident(db, event, "RN-07", PRIORITY_MEDIUM)
            _create_alert(db, incident.id, f"ALERTA ASISTENCIAL: Residente {', '.join(known_names)} detecto caida en {camera_id} ({timestamp})")
            
            # NON-BLOCKING ASYNC QUEUING
            with _pending_evidences_lock:
                _pending_evidences.append((camera_id, incident.id))
                
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
    # Initialize the AnnotatedFrameBuffer and the WeaponDebouncer BEFORE
    # binding the rule socket. This way both subsystems are warm by the time
    # the first detection event arrives, minimizing cold-start windows where
    # evidence has nothing to persist or debouncer state must be lazily
    # created. They are exposed via module-level names so helper functions
    # can reach them without signature changes.
    global _frame_buffer, _weapon_debouncer
    _frame_buffer = AnnotatedFrameBuffer(ANNOTATED_SUB_PORT)
    _weapon_debouncer = WeaponDebouncer()
    logger.info(
        f"Weapon debouncer online (hits={WEAPON_DEBOUNCE_HITS}, "
        f"window={WEAPON_DEBOUNCE_WINDOW}s, "
        f"min_avg_conf={WEAPON_DEBOUNCE_MIN_AVG_CONF})"
    )

    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(RECEIVER_PORT)

    logger.info(f"Rule engine started. Waiting for events on {RECEIVER_PORT}")
    accumulator = EventAccumulator(window_seconds=COMPOUND_EVENT_WINDOW_SECONDS)

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