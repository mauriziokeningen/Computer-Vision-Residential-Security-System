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
from src.database.models import Person

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
MAX_ALLOWED_DISTANCE = float(os.getenv("FACE_MAX_ALLOWED_DISTANCE", "0.61"))


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
    Optimized for zero-copy memory and minimal network I/O.
    """

    db = SessionLocal()
    try:
        vector_list = embedding.tolist()

        #1. Configuration of hyperparameter HNSW for this transaction (Latency Optimization)
        db.execute(text("SET LOCAL hnsw.ef_search = 32;"))

        #2. Define the calculated distance column
        #This let us extract the calculation that PostgreSQL already did, avoiding recalculating in Python.
        distance_col = Person.face_embedding.cosine_distance(vector_list).label("distance")
        
        #3. Strict Projection: We only extract the name and the distance.
        #We never bring the 512D vector back by the network in the critical route.
        result = db.query(Person.full_name, distance_col)\
            .filter(Person.face_embedding.is_not(None))\
            .order_by(distance_col)\
            .limit(1)\
            .first()
        
        if result:
            full_name, distance = result

            #Strict validation of the security Gatekeeper
            if distance <= MAX_ALLOWED_DISTANCE:
                return full_name, float(distance)
            return "unknown_person", float(distance)
        
        return "unknown_person", 1.0
    
    except Exception as e:
        logger.error(f"Database vector search failed: {e}")
        return "unknown_person", 1.0
    finally:
        # This does an implicit rollback in the transaction
        # Cleaning the "SET LOCAL" securely for the connections pool
        db.close()


def start_face_model() -> None:
    """
    Initializes the AI process, establishes IPC pipelines, and enters the infinite polling loop.
    
    Constraints:
        Designed as an isolated multiprocess target. Do not call this synchronously 
        within an ASGI event loop.
    """
    context = zmq.Context()

    logger.info("Initializing FaceProcessorService (InsightFace)...")
    try:
        # Instantiating the AI service dynamically claims VRAM. 
        # Failure here indicates hardware resource exhaustion or missing CUDA libraries.

        ai_service = FaceProcessorService()
        logger.info("Face module loaded into VRAM. Listening for video stream...")
    except Exception as e:
        logger.critical(f"FATAL: Could not load AI models into memory: {e}")
        return

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

    logger.info("Face module online and synchronized. Enterring polling loop...")
    # Aggregated counter for decode failures so we can surface persistent
    # corruption without flooding the journal on a single bad packet.
    decode_failures = 0

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)

            t0 = time.time() 
            faces = ai_service.app.get(frame)
            infer_ms = (time.time() - t0) * 1000.0 

            if faces:
                logger.info(f"🏀 [FACE INFERENCE] Processed frame. infer={infer_ms:.0f}ms")

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

