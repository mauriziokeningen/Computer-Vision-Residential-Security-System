#!/bin/bash
# apply-review-fixes.sh
#
# Writes the six files that change in response to the review on PR #65.
# Only writes files. Does not run git, does not stage, does not commit.
# Inspect with `git diff` and run any verification before committing.
#
# Run from the repo root (the folder that contains 'backend/' and 'frontend/').

set -e

if [ ! -d "backend" ]; then
  echo "ERROR: run this from the repo root (where 'backend/' folder lives)."
  exit 1
fi

echo "[1/2] Ensuring directory structure exists..."
mkdir -p backend/scripts
mkdir -p backend/src/modules/weapons
mkdir -p backend/src/modules/face
mkdir -p backend/src/orchestrator
mkdir -p backend/src/annotator
mkdir -p backend/src/services

echo "[2/2] Writing changed files..."

cat > "backend/scripts/export_weapons_to_coreml.py" << 'REVIEW_EOF_MARKER_X9P3'
"""
Export a YOLO model to a deployable inference format.

Default behavior exports the weapon-detection model (best2.pt) to CoreML
with FP16 at imgsz=640 — the configuration validated for our Apple
Silicon target. Arguments are exposed so the same script can be reused
for other models (face, future modules) and other formats (ONNX,
TensorRT, etc.) without editing source.

Usage examples
--------------
Default (weapons → CoreML FP16):
    python scripts/export_weapons_to_coreml.py

Custom weights:
    python scripts/export_weapons_to_coreml.py \\
        --weights research/models/object_detection/weights/best2.pt

Higher input resolution (must match training resolution to preserve accuracy):
    python scripts/export_weapons_to_coreml.py --imgsz 1280

Different format:
    python scripts/export_weapons_to_coreml.py --format onnx

Notes on the defaults
---------------------
- ``half=True`` exports in FP16. Apple Neural Engine runs natively in
  FP16; it is essentially a free speedup with no perceptible accuracy
  loss for object detection.
- ``imgsz=640`` matches the inference resolution used by the weapons
  worker. If the model was trained at a different resolution (e.g.
  1280), pass --imgsz to match training to recover full accuracy.
- ``nms`` defaults to True for parity with the original script. YOLOv10
  is end-to-end and ignores this flag (with a benign warning); older
  YOLO families respect it.
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


# Repository-relative defaults so the script works regardless of
# checkout location, as long as the repo layout is preserved.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_WEIGHTS = (
    ROOT_DIR / "research" / "models" / "object_detection" / "weights" / "best2.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO model for accelerated inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the source PyTorch weights (.pt).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="coreml",
        choices=["coreml", "onnx", "torchscript", "tflite", "engine"],
        help="Target export format.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size; must match training resolution for best accuracy.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export with FP16 weights (recommended for Apple Neural Engine).",
    )
    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bake NMS into the exported graph. YOLOv10 is end-to-end and ignores this.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"ERROR: Source weights not found at {args.weights}", file=sys.stderr)
        return 1

    print(f"Loading {args.weights}...")
    model = YOLO(str(args.weights))

    print(
        f"Exporting to {args.format} "
        f"(half={args.half}, imgsz={args.imgsz}, nms={args.nms})..."
    )
    print("This takes 1-3 minutes on Apple Silicon.")
    output_path = model.export(
        format=args.format,
        half=args.half,
        imgsz=args.imgsz,
        nms=args.nms,
    )

    print(f"\nDone. Exported package: {output_path}")
    print("Next: just run `python main.py`. The worker will detect and use it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

REVIEW_EOF_MARKER_X9P3

cat > "backend/src/modules/weapons/inference.py" << 'REVIEW_EOF_MARKER_X9P3'
"""
Weapon Detection Worker.

Subscribes to the video stream via ZeroMQ, runs YOLO inference, and publishes:
    - Detection metadata to the orchestrator (PUSH, port 5556) for rule evaluation.
    - Detection metadata to the annotator (PUB, port 5558) for visual rendering.

Frame data is intentionally absent from both outputs; frame rendering is the
annotator's responsibility, and evidence persistence consumes the annotated
stream from the orchestrator's annotated-frame buffer.

Model loading strategy:
    Prefers `best2.mlpackage` (CoreML, Apple Silicon-accelerated) over
    `best2.pt` (PyTorch, CPU-only on Mac without CUDA). The CoreML export
    is produced by `scripts/export_weapons_to_coreml.py` and yields ~3-5x
    lower inference latency on M-series chips by routing convolutions to
    the Neural Engine. If the .mlpackage is missing the worker silently
    falls back to .pt, so the system always boots regardless of export state.

    Platform gating: CoreML is only attempted when the host is actually
    macOS on Apple Silicon. On Linux/Windows or Intel Macs the worker
    falls back to PyTorch even if a `.mlpackage` happens to exist on
    disk — loading an Apple-only artifact on a non-Apple host raises a
    fatal error inside the model loader, so platform detection is a
    hard gate.
"""
import os
import platform
import zmq
import time
import logging
import numpy as np
import cv2
from typing import Optional, Tuple
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WeaponInference")

# Network endpoints. Sourced from the environment so deployment topology
# (containers, multi-host, multi-camera) can change without code edits.
# Defaults preserve the original developer-machine layout.
VIDEO_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")
ORCHESTRATOR_PUSH_PORT = os.getenv("ORCHESTRATOR_PUSH_PORT", "tcp://127.0.0.1:5556")
ANNOTATOR_PUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")
MODULE_NAME = "weapons"
CAMERA_ID = os.getenv("CAMERA_ID", "main_camera")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
WEIGHTS_DIR = ROOT_DIR / "research" / "models" / "object_detection" / "weights"
COREML_WEIGHTS = WEIGHTS_DIR / "best2.mlpackage"
PYTORCH_WEIGHTS = WEIGHTS_DIR / "best2.pt"

CONFIDENCE_THRESHOLD = float(os.getenv("WEAPON_CONFIDENCE_THRESHOLD", "0.50"))
THREAT_CLASSES = {"knife", "pistol"}


def _is_apple_silicon() -> bool:
    """
    Returns True only on macOS running on an arm64 chip (M1/M2/M3/M4...).

    Used as a hard gate before attempting to load a CoreML `.mlpackage`.
    Loading an Apple-only artifact on Linux or Intel raises an opaque
    error from inside coremltools / Ultralytics, which we'd rather
    avoid by detecting the unsupported host upfront.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _resolve_model_path() -> Tuple[str, str]:
    """
    Returns (path_to_load, backend_label_for_logs).

    CoreML export wins when present AND the host is Apple Silicon. The
    backend label is logged on startup so it's obvious from stdout
    which path is in use after a deploy or a fresh checkout.
    """
    if COREML_WEIGHTS.exists() and _is_apple_silicon():
        return str(COREML_WEIGHTS), "CoreML (Apple Silicon accelerated)"

    if COREML_WEIGHTS.exists() and not _is_apple_silicon():
        logger.info(
            "CoreML weights present but host is not Apple Silicon "
            f"(system={platform.system()}, machine={platform.machine()}); "
            "falling back to PyTorch."
        )

    if PYTORCH_WEIGHTS.exists():
        return str(PYTORCH_WEIGHTS), "PyTorch (CPU)"
    raise FileNotFoundError(
        f"No weights found at {COREML_WEIGHTS} or {PYTORCH_WEIGHTS}"
    )


def _decode_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
    """
    Deserializes the IPC byte payload into an OpenCV-compatible BGR matrix.

    Returns None if the bytes are corrupt or unreadable, allowing the
    caller to skip the frame instead of letting a downstream cv2/YOLO
    call raise on a malformed buffer.
    """
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)


def start_weapon_model() -> None:
    context = zmq.Context()

    video_receiver = context.socket(zmq.SUB)
    video_receiver.connect(VIDEO_SUB_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1)

    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(ORCHESTRATOR_PUSH_PORT)

    annotator_publisher = context.socket(zmq.PUB)
    annotator_publisher.connect(ANNOTATOR_PUB_PORT)

    try:
        model_path, backend_label = _resolve_model_path()
    except FileNotFoundError as e:
        logger.critical(str(e))
        return

    logger.info(f"Loading weapon detection model: {model_path}")
    logger.info(f"Backend: {backend_label}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    logger.info("Running warmup inference...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    logger.info("Weapon model ready. Listening for video stream...")

    last_idle_log = 0.0
    # Aggregated counter for decode failures so we surface persistent
    # corruption without flooding logs on a single bad packet.
    decode_failures = 0

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)
            if frame is None:
                decode_failures += 1
                # Log every 30th failure to flag persistent issues without
                # spamming the journal on transient single-frame corruption.
                if decode_failures % 30 == 1:
                    logger.warning(
                        f"Frame decode returned None ({decode_failures} total); "
                        "skipping inference for this frame."
                    )
                continue

            t0 = time.time()
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            infer_ms = (time.time() - t0) * 1000.0

            detections_payload = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]

                    if cls_name not in THREAT_CLASSES:
                        continue

                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    detections_payload.append({
                        "class": cls_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    })

            # Only publish on positive detections. Empty publishes would clear
            # the annotator's TTL buffer prematurely; absence of a publish lets
            # bboxes age out naturally if the model has a flicker frame.
            if detections_payload:
                annotator_publisher.send_json({
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "detections": detections_payload,
                })

                # Frame data intentionally omitted: the orchestrator pulls the
                # annotated frame from its own SUB buffer, so live view and
                # stored evidence are pixel-identical.
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "detections": detections_payload,
                }
                result_sender.send_json(payload)

                for d in detections_payload:
                    logger.warning(
                        f"[WEAPON DETECTED] {d['class']} "
                        f"conf={d['confidence']*100:.1f}% "
                        f"infer={infer_ms:.0f}ms"
                    )
            else:
                # Lightweight idle heartbeat for measuring inference latency
                # over time (catches thermal throttling).
                now = time.time()
                if now - last_idle_log > 2.0:
                    logger.info(f"[idle] infer={infer_ms:.0f}ms")
                    last_idle_log = now

        except Exception as e:
            logger.debug(f"Inference cycle error: {e}")

REVIEW_EOF_MARKER_X9P3

cat > "backend/src/modules/face/inference.py" << 'REVIEW_EOF_MARKER_X9P3'
"""
Biometric Inference Worker Service.

This module acts as an isolated microservice within the distributed IPC architecture, 
responsible for real-time facial detection, embedding extraction, and identity verification.
It publishes detection metadata to:
    - The orchestrator (PUSH, port 5556) for rule evaluation.
    - The annotator (PUB, port 5558) for visual rendering.

Frame data is no longer carried on either output; the orchestrator obtains the
annotated frame from its own SUB buffer when persisting evidence, ensuring that
live view and stored evidence display the exact same pixels.

Architectural Decisions & Trade-offs:
* Hardware Isolation: Consumes frames via ZeroMQ SUB sockets instead of interacting 
  with /dev/video0. This respects the AI ingestion node as the Single Source of Truth 
  (SSoT) for hardware mutex locks.
* Vector Search (pgvector): Utilizes native PostgreSQL vector operations for nearest-neighbor 
  searches. We explicitly rejected in-memory vector indices (like FAISS) to prevent 
  state synchronization issues across distributed worker nodes, trading a negligible 
  latency increase for strict ACID compliance.
* No frame muxing in the worker: drawing was historically tempting to do here, but
  it would create two divergent visual paths (live view vs evidence). The annotator
  is now the single source of visual truth.
"""
import os
import cv2
import zmq
import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import text

from src.services.face_processor import FaceProcessorService
from src.database.session import SessionLocal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceInference")

# --- System Integration Constants ---
# Endpoints sourced from the environment so deployment topology can change
# without code edits. Defaults preserve the original developer-machine layout.
VIDEO_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")
ORCHESTRATOR_PUSH_PORT = os.getenv("ORCHESTRATOR_PUSH_PORT", "tcp://127.0.0.1:5556")
ANNOTATOR_PUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")
MODULE_NAME = "face"
CAMERA_ID = os.getenv("CAMERA_ID", "main_camera")

# Boundary Warning: In pgvector, the <=> operator calculates Cosine Distance 
# (0.0 is a mathematically perfect match, 1.0 is completely orthogonal).
# A threshold of 0.40 guarantees a >60% mathematical similarity, aggressively minimizing 
# false positives at the risk of slightly higher false negatives (which is preferable in physical security).
MAX_ALLOWED_DISTANCE = float(os.getenv("FACE_MAX_ALLOWED_DISTANCE", "0.40"))


def _decode_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
    """
    Deserializes the IPC byte payload into an OpenCV-compatible BGR matrix.

    Returns None if the bytes are corrupt or unreadable, allowing the caller
    to skip the frame instead of letting a downstream call raise on a
    malformed buffer.

    Args:
        frame_bytes (bytes): The raw byte array transmitted over ZeroMQ.

    Returns:
        Optional[np.ndarray]: A multi-dimensional array representing the image
        frame, or None if decoding failed.
    """
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)


def _find_closest_match_in_db(embedding: np.ndarray) -> Tuple[str, float]:
    """
    Executes a high-speed nearest-neighbor search against the PostgreSQL vector index.

    Args:
        embedding (np.ndarray): The 512-dimensional L2-normalized face vector.

    Returns:
        Tuple[str, float]: The database identifier (name) and the calculated cosine distance.
                           Returns "unknown_person" if the distance exceeds the security threshold.
    """
    db = SessionLocal()
    try:
        vector_list = embedding.tolist()
          # Native SQL utilizing the pgvector <=> operator forces the database engine 
        # to execute the nearest-neighbor calculation, keeping the Python worker stateless.
        query = text("""
            SELECT full_name, (face_embedding <=> :vector) AS distance
            FROM persons
            WHERE face_embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
        """)
        result = db.execute(query, {"vector": str(vector_list)}).fetchone()
        if result:
            name, distance = result
            # Strict security gate enforcement
            if distance <= MAX_ALLOWED_DISTANCE:
                return name, float(distance)
            return "unknown_person", float(distance)
        return "unknown_person", 1.0
    except Exception as e:
        logger.error(f"Database vector search failed: {e}")
        return "unknown_person", 1.0
    finally:
        db.close()


def start_face_model() -> None:
    """
    Initializes the AI process, establishes IPC pipelines, and enters the infinite polling loop.
    
    Constraints:
        Designed as an isolated multiprocess target. Do not call this synchronously 
        within an ASGI event loop.
    """
    context = zmq.Context()
    # Establish read-only ingestion pipeline
    video_receiver = context.socket(zmq.SUB)
    video_receiver.connect(VIDEO_SUB_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1)

   # Establish write-only orchestration pipeline
    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(ORCHESTRATOR_PUSH_PORT)

    annotator_publisher = context.socket(zmq.PUB)
    annotator_publisher.connect(ANNOTATOR_PUB_PORT)

    logger.info("Initializing FaceProcessorService (InsightFace)...")
    try:
        # Instantiating the AI service dynamically claims VRAM. 
        # Failure here indicates hardware resource exhaustion or missing CUDA libraries.

        ai_service = FaceProcessorService()
        logger.info("Face module loaded into VRAM. Listening for video stream...")
    except Exception as e:
        logger.critical(f"FATAL: Could not load AI models into memory: {e}")
        return

    # Aggregated counter for decode failures so we can surface persistent
    # corruption without flooding the journal on a single bad packet.
    decode_failures = 0

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)
            if frame is None:
                decode_failures += 1
                # Log every 30th failure to flag persistent issues without
                # spamming the logs on a transient corrupt frame.
                if decode_failures % 30 == 1:
                    logger.warning(
                        f"Frame decode returned None ({decode_failures} total); "
                        "skipping inference for this frame."
                    )
                continue

            detections_payload = []

            faces = ai_service.app.get(frame)

            for face in faces:
                box = face.bbox.astype(int)
                x, y, x2, y2 = box[0], box[1], box[2], box[3]
                w, h = x2 - x, y2 - y
                
                live_embedding = face.normed_embedding
                name, distance = _find_closest_match_in_db(live_embedding)

                detections_payload.append({
                    "name": name,
                    "confidence": round(1.0 - distance, 4),
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                })

            # Always publish to the annotator (even empty), so that bboxes age out
            # cleanly via the TTL buffer when faces leave the scene.
            annotator_publisher.send_json({
                "camera_id": CAMERA_ID,
                "module": MODULE_NAME,
                "detections": detections_payload,
            })

            if detections_payload:
                # Frame data is intentionally absent: the orchestrator's annotated
                # frame buffer (subscribed to the annotator) provides the evidence
                # image, guaranteeing visual parity with the live feed.
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "detections": detections_payload
                }
                result_sender.send_json(payload)

        except Exception as e:
            # Catching generic exceptions prevents a single bad frame matrix from killing the entire worker.
            logger.debug(f"Inference cycle error: {e}")

REVIEW_EOF_MARKER_X9P3

cat > "backend/src/orchestrator/rules.py" << 'REVIEW_EOF_MARKER_X9P3'
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
from typing import Dict, Any, List, Optional
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
            except Exception as e:
                logger.error(f"Annotated frame listener crashed: {e}")
                break

    def get_latest(self) -> Optional[bytes]:
        """Returns a snapshot of the latest annotated JPEG, or None if not ready yet."""
        with self._lock:
            return self._frame


# Module-level singleton, populated when start_orchestrator() runs. Kept as a
# module attribute (not a true global mutable) so the helper functions
# (_save_evidence, etc.) can access it without threading every signature.
_frame_buffer: Optional[AnnotatedFrameBuffer] = None


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
    """
    Persists the latest annotated frame (from the annotator's output stream)
    as evidence for the given incident.

    The `frame_data` parameter is kept in the signature for backwards
    compatibility but is no longer used: workers stopped sending frame bytes
    to the orchestrator since the rendering moved to the annotator. Evidence
    is sourced from the in-process buffer fed by AnnotatedFrameBuffer.

    If the annotator hasn't produced a frame yet (e.g. cold start, or the
    annotator process is down), this returns None and skips persistence
    rather than uploading a raw, unannotated frame — visual consistency
    between live view and evidence is the explicit guarantee of this
    architecture.
    """
    if _frame_buffer is None:
        logger.warning(
            f"AnnotatedFrameBuffer not initialized when persisting incident "
            f"{incident_id}; evidence will not be saved."
        )
        return None

    frame_bytes = _frame_buffer.get_latest()
    if frame_bytes is None:
        logger.warning(
            f"No annotated frame available for incident {incident_id}; "
            "skipping evidence upload (annotator may still be warming up)."
        )
        return None

    try:
        from src.utils.s3_client import upload_incident_clip

        # TECH DEBT: Synchronous network I/O.
        # Uploading to MinIO/S3 blocks the main ZMQ event loop. If the network degrades,
        # the IPC bus will back up. V2 must offload this to a Celery background worker.
        object_name = upload_incident_clip(
            file_data=frame_bytes,
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
    # Initialize the AnnotatedFrameBuffer BEFORE binding the rule socket. This
    # way the buffer is already populating by the time the first detection
    # event arrives, minimizing the cold-start window where _save_evidence
    # would have nothing to persist. The buffer is exposed via a module-level
    # name so the helper functions can reach it without signature changes.
    global _frame_buffer
    _frame_buffer = AnnotatedFrameBuffer(ANNOTATED_SUB_PORT)

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

REVIEW_EOF_MARKER_X9P3

cat > "backend/src/annotator/process.py" << 'REVIEW_EOF_MARKER_X9P3'
"""
Frame Annotator Service.

Single source of truth for visual overlays. Subscribes to:
    1. Raw frames from the ingestion publisher (port 5555, CONFLATE -> always latest).
    2. Detection metadata from inference workers (port 5558, no conflate -> all events).

For every raw frame received, draws all currently-active detections (TTL window)
on top and republishes the annotated frame on port 5557. Both the live MJPEG
feed (LocalCameraStreamService) and the orchestrator's evidence buffer
subscribe to that single annotated stream, guaranteeing pixel-identical visuals
between live view and persisted incident evidence.

Architectural decisions:
    - One process, one polling loop. No threads inside the annotator: zmq.Poller
      handles both inputs in a single event loop. Simpler than locks, and the
      drawing throughput on a single core is more than enough for ~20 FPS at 720p.
    - Detection TTL (DEFAULT_DETECTION_TTL_SECONDS) is the max age a detection
      stays "active" on the frame. Beyond that it's dropped. This handles the
      asymmetry between fast inference (bbox arrives quickly) and the absence of
      a follow-up detection (e.g. weapon left the frame): without TTL the bbox
      would freeze on the last position forever.
    - Pass-through path: when no detections are active, the raw JPEG is forwarded
      as-is without re-encoding. Saves one decode + one encode per frame in the
      common case.
"""
import os
import zmq
import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Any
from dataclasses import dataclass, field

from src.utils.draw import draw_module_detections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Annotator")

# Endpoints sourced from the environment with the current localhost
# topology as defaults. Keeps a multi-host or containerized deployment
# configurable without code changes.
INGEST_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")            # raw frames from stream.py
DETECTIONS_SUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")    # detection metadata from workers
ANNOTATED_PUB_PORT = os.getenv("ANNOTATED_PUB_PORT", "tcp://127.0.0.1:5557")     # annotated frames out

# How long a detection stays drawn on the frame after the last time it was reported.
# At ~20 FPS inference on weapons + ~30 FPS on face, 0.5s comfortably covers the
# next inference cycle from the slowest module. Tuned downward will cause flicker;
# tuned upward will leave stale boxes lingering when objects exit the scene.
DEFAULT_DETECTION_TTL_SECONDS = float(os.getenv("ANNOTATOR_TTL_SECONDS", "0.5"))

# JPEG quality for the republished annotated frame. Matches the workers' previous
# evidence quality (75) so that the live feed and stored evidence are visually
# indistinguishable.
JPEG_QUALITY = int(os.getenv("ANNOTATOR_JPEG_QUALITY", "75"))


@dataclass
class _DetectionEntry:
    """A detection set tagged with its expiration timestamp."""
    expires_at: float
    module: str
    detections: List[Dict[str, Any]]


class DetectionBuffer:
    """
    TTL store keyed by camera_id, then by module.

    We store the latest detection set per (camera, module) instead of accumulating
    a stream of events. Reason: each new detection event from a worker represents
    the worker's current view of the world for that camera — the previous one is
    stale by definition. Accumulating would draw duplicated/jittering boxes.
    """
    def __init__(self, ttl_seconds: float = DEFAULT_DETECTION_TTL_SECONDS):
        self.ttl = ttl_seconds
        # camera_id -> module -> _DetectionEntry
        self._items: Dict[str, Dict[str, _DetectionEntry]] = {}

    def update(self, camera_id: str, module: str, detections: List[Dict[str, Any]]) -> None:
        cam_bucket = self._items.setdefault(camera_id, {})
        cam_bucket[module] = _DetectionEntry(
            expires_at=time.time() + self.ttl,
            module=module,
            detections=detections,
        )

    def get_active(self, camera_id: str) -> List[_DetectionEntry]:
        """Returns all non-expired entries for the given camera. Prunes as it goes."""
        now = time.time()
        cam_bucket = self._items.get(camera_id, {})
        active = [e for e in cam_bucket.values() if e.expires_at > now]
        # Prune expired in place
        if len(active) != len(cam_bucket):
            self._items[camera_id] = {e.module: e for e in active}
        return active


def _decode(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def start_annotator() -> None:
    context = zmq.Context()

    # Raw frames in. CONFLATE: we only ever care about the latest frame, so the
    # annotator never lags behind the camera even if drawing takes longer than
    # the inter-frame interval.
    raw_sub = context.socket(zmq.SUB)
    raw_sub.connect(INGEST_SUB_PORT)
    raw_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    raw_sub.setsockopt(zmq.CONFLATE, 1)

    # Detections in. NO CONFLATE: every detection event matters; if we drop one,
    # we miss drawing a class on the current frames. Workers throttle naturally
    # (inference latency ~30-50ms), so the queue never grows out of control.
    det_sub = context.socket(zmq.SUB)
    det_sub.bind(DETECTIONS_SUB_PORT)
    det_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Annotated frames out.
    annotated_pub = context.socket(zmq.PUB)
    annotated_pub.bind(ANNOTATED_PUB_PORT)

    # Brief warmup so subscribers (LocalCameraStreamService, orchestrator listener)
    # have time to connect before the first message lands. ZMQ slow-joiner pattern.
    time.sleep(0.2)

    poller = zmq.Poller()
    poller.register(raw_sub, zmq.POLLIN)
    poller.register(det_sub, zmq.POLLIN)

    buffer = DetectionBuffer()
    logger.info(
        f"Annotator online. raw={INGEST_SUB_PORT}  "
        f"detections={DETECTIONS_SUB_PORT}  out={ANNOTATED_PUB_PORT}"
    )

    while True:
        try:
            socks = dict(poller.poll(timeout=100))

            # Drain detection events first — they're cheap, low-volume, and we
            # want the buffer up to date before we draw the next frame.
            if det_sub in socks:
                while True:
                    try:
                        meta = det_sub.recv_json(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    buffer.update(
                        camera_id=meta.get("camera_id", "unknown"),
                        module=meta.get("module", "unknown"),
                        detections=meta.get("detections", []),
                    )

            # Process the latest raw frame, if any.
            if raw_sub in socks:
                frame_bytes = raw_sub.recv()
                # SSoT camera assumption: today the system runs with a single camera.
                # When multi-camera lands, the ingestion publisher must include
                # camera_id in the message envelope and we key the buffer accordingly.
                camera_id = "main_camera"

                active = buffer.get_active(camera_id)

                if not active:
                    # Pass-through: no decode/encode roundtrip, just forward the
                    # JPEG bytes. Common case during quiet periods.
                    annotated_pub.send(frame_bytes)
                    continue

                frame = _decode(frame_bytes)
                if frame is None:
                    continue

                # Draw all active modules on the same buffer. Order is stable
                # but not enforced: face green/orange, weapons red, pose yellow/red.
                for entry in active:
                    draw_module_detections(frame, entry.module, entry.detections)

                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    annotated_pub.send(jpeg.tobytes())

        except Exception as e:
            # Catching generic exceptions keeps a single bad frame from killing
            # the whole live view. Symptomatic logging only.
            logger.debug(f"Annotator cycle error: {e}")

REVIEW_EOF_MARKER_X9P3

cat > "backend/src/services/local_camera_stream.py" << 'REVIEW_EOF_MARKER_X9P3'
import os
import zmq
import time
import threading
from typing import Optional


# Annotated stream endpoint. Defaults to localhost:5557 to preserve the
# original developer-machine layout. Override with ANNOTATED_PUB_PORT
# env var when deploying with a different topology.
ANNOTATED_STREAM_ENDPOINT = os.getenv("ANNOTATED_PUB_PORT", "tcp://127.0.0.1:5557")


class LocalCameraStreamService:
    """
    Decoupled Video Ingestion Subscriber Service.

    This service operates as the boundary between the AI inference worker and the FastAPI 
    client stream. By acting as a downstream ZeroMQ subscriber, it eliminates the OS-level 
    hardware mutex collisions that occur when multiple processes attempt to bind to the 
    same physical capture device (e.g., /dev/video0). The AI inference node is maintained 
    as the Single Source of Truth (SSoT) for all hardware interaction.

    Architectural Decisions & Trade-offs:
    * Inter-Process Communication (IPC): ZeroMQ was selected over broker-based pub/sub 
      (e.g., Redis, RabbitMQ) to route data directly socket-to-socket. This prevents memory 
      bloat and intermediate disk I/O when processing high-throughput (60MB/s) video payloads.
    * Resilience: Designed as a singleton initialized once per ASGI worker lifecycle. It 
      utilizes non-blocking sockets to prevent thread starvation if the upstream publisher 
      terminates unexpectedly.
    * Performance: By consuming pre-compressed JPEG byte arrays over TCP, this service 
      completely eliminates matrix serialization overhead on the web server, acting purely 
      as a zero-cost byte forwarder.
    """

    def __init__(self):
        # Initializes the underlying C-level thread pool for ZMQ sockets.
        self._context = zmq.Context()
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_frame: Optional[bytes] = None
        
        # Concurrency control: Enforces thread-safe memory access across the ZMQ polling 
        # daemon and concurrent ASGI worker threads executing HTTP requests.
        self._lock = threading.Lock()

    def start(self, source_index: int = 0, warmup_timeout: float = 4.0) -> None:
        """
        Provisions the TCP socket and spawns the daemon reader thread.

        Args:
            source_index (int): Maintained for interface compatibility; overridden by network topology.
            warmup_timeout (float): Execution threshold for initial stream acquisition.

        Constraints:
            Implements ZMQ_CONFLATE set to 1. Video streams generate data faster than standard 
            network clients consume it. Rather than queueing frames (which causes temporal latency 
            and OOM crashes), conflation enforces an O(1) memory bound by actively overwriting 
            unread frames. We explicitly sacrifice sequential completeness for real-time accuracy.
        """
        with self._lock:
            if self._running:
                return
            self._running = True

        # Subscribe to the annotated stream (port 5557) instead of the raw
        # ingestion stream. The annotator process draws bounding boxes,
        # labels, and confidences on top of the raw frames; consuming from
        # 5557 means the live MJPEG feed shown to the operator already has
        # the visual overlay baked in. When no detections are active, the
        # annotator passes the raw JPEG through unchanged.
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(ANNOTATED_STREAM_ENDPOINT)

        # An empty subscription filter assumes total consumption of the target port.
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1) 

        # Delegates blocking network I/O to a background thread to preserve the asyncio event loop.
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Gracefully terminates the polling daemon and releases the IPC network socket.
        """
        with self._lock:
            self._running = False
            
        if self._socket:
            self._socket.close()
            self._socket = None

    def _reader_loop(self) -> None:
        """
        Dedicated network polling daemon.

        Boundary Warnings:
            This implementation relies strictly on zmq.NOBLOCK. A standard blocking recv() 
            will permanently deadlock the background thread if the upstream AI publisher crashes.
            The exception handler yields the Global Interpreter Lock (GIL) to prevent CPU starvation.
        """
        while self._running and self._socket:
            try:
                frame = self._socket.recv(zmq.NOBLOCK)
                with self._lock:
                    self._last_frame = frame
            except zmq.Again:
                # time.sleep(0.01) is required here for backoff. Yields the GIL.
                time.sleep(0.01) 
            except Exception:
                break

    def get_status(self) -> dict:
        """
        Diagnostic probe utilized by the API for readiness checks.

        Returns:
            dict: Internal state mapping containing execution status and frame availability.
        """
        with self._lock:
            return {
                "running": self._running,
                "has_frame": self._last_frame is not None,
            }

    def mjpeg_generator(self):
        """
        Constructs an HTTP Multipart / MJPEG compliant data stream.

        Yields:
            bytes: Encoded HTTP MJPEG boundary strings wrapping the raw JPEG payload.

        Intent:
            MJPEG (multipart/x-mixed-replace) enables native HTML <img> tag rendering without 
            relying on heavy client-side JavaScript decoders (e.g., WebRTC, Canvas).
        """
        try:
            while True:
                with self._lock:
                    frame = self._last_frame
                    running = self._running

                if not running:
                    break

                if frame is None:
                    # Prevents infinite looping and CPU spikes during upstream frame drops.
                    time.sleep(0.05)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
                )
                
                # Regulates output to ~20 FPS. Protects downstream clients from rendering exhaustion.
                time.sleep(0.05)
        except GeneratorExit:
            return


local_camera_stream_service = LocalCameraStreamService()

REVIEW_EOF_MARKER_X9P3

echo ""
echo "Done. Files written:"
echo "  - backend/scripts/export_weapons_to_coreml.py    (argparse)"
echo "  - backend/src/modules/weapons/inference.py       (Darwin check + env vars + None frame, debug block removed)"
echo "  - backend/src/modules/face/inference.py          (env vars + None frame)"
echo "  - backend/src/orchestrator/rules.py              (env vars + AnnotatedFrameBuffer class)"
echo "  - backend/src/annotator/process.py               (env vars)"
echo "  - backend/src/services/local_camera_stream.py    (env var)"
