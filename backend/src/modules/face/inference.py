"""
Biometric Inference Worker Service.

This module acts as an isolated microservice within the distributed IPC architecture, 
responsible for real-time facial detection, embedding extraction, and identity verification.

Architectural Decisions & Trade-offs:
* Hardware Isolation: Consumes frames via ZeroMQ SUB sockets instead of interacting 
  with /dev/video0. This respects the AI ingestion node as the Single Source of Truth 
  (SSoT) for hardware mutex locks.
* Vector Search (pgvector): Utilizes native PostgreSQL vector operations for nearest-neighbor 
  searches. We explicitly rejected in-memory vector indices (like FAISS) to prevent 
  state synchronization issues across distributed worker nodes, trading a negligible 
  latency increase for strict ACID compliance.
* Payload Contract Parity: Enforces a strict data contract with the downstream 
  Orchestrator by providing a compressed Base64 image payload upon detection, guaranteeing 
  MinIO evidence persistence.
"""
import cv2
import zmq
import time
import base64
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy import text

from src.services.face_processor import FaceProcessorService
from src.database.session import SessionLocal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceInference")

# --- System Integration Constants ---
SUBSCRIBER_PORT = "tcp://127.0.0.1:5555"
PUBLISHER_PORT = "tcp://127.0.0.1:5556"
MODULE_NAME = "face"
CAMERA_ID = "main_camera"

# Boundary Warning: In pgvector, the <=> operator calculates Cosine Distance 
# (0.0 is a mathematically perfect match, 1.0 is completely orthogonal).
# A threshold of 0.40 guarantees a >60% mathematical similarity, aggressively minimizing 
# false positives at the risk of slightly higher false negatives (which is preferable in physical security).
MAX_ALLOWED_DISTANCE = 0.40 


def _decode_frame(frame_bytes: bytes) -> np.ndarray:
    """
    Deserializes the IPC byte payload into an OpenCV-compatible BGR matrix.

    Args:
        frame_bytes (bytes): The raw byte array transmitted over ZeroMQ.

    Returns:
        np.ndarray: A multi-dimensional array representing the image frame.
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
        # CRITICAL: Connection pool exhaustion will occur if this lock is not released.
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
    video_receiver.connect(SUBSCRIBER_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1)

    # Establish write-only orchestration pipeline
    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(PUBLISHER_PORT)

    logger.info("Initializing FaceProcessorService (InsightFace)...")
    try:
        # Instantiating the AI service dynamically claims VRAM. 
        # Failure here indicates hardware resource exhaustion or missing CUDA libraries.
        ai_service = FaceProcessorService()
        logger.info("Face module loaded into VRAM. Listening for video stream...")
    except Exception as e:
        logger.critical(f"FATAL: Could not load AI models into memory: {e}")
        return

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)
            detections_payload = []

            # Execute unified detection and alignment forward pass
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

            if detections_payload:
                # System Design Constraint: Memory Management
                # Raw BGR matrices at 720p consume ~2.7MB. Encoding this directly to Base64 
                # saturates the ZeroMQ IPC bus and triggers OOM crashes in the Orchestrator. 
                # We aggressively compress to JPEG (Quality 75) first, reducing the payload to ~40KB.
                success, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                
                frame_b64 = ""
                if success:
                    frame_b64 = base64.b64encode(jpeg_buffer.tobytes()).decode('utf-8')

                # Fulfils the strict JSON data contract expected by the Orchestrator for MinIO persistence.
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "frame_data": frame_b64, 
                    "detections": detections_payload
                }
                result_sender.send_json(payload)

        except Exception as e:
            # Catching generic exceptions prevents a single bad frame matrix from killing the entire worker.
            logger.debug(f"Inference cycle error: {e}")