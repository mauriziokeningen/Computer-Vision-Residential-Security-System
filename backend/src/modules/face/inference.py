import cv2
import zmq
import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy import text

# Import our custom AI Service and Database Session
from src.services.face_processor import FaceProcessorService
from src.database.session import SessionLocal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceInference")

# --- ZeroMQ Configuration ---
SUBSCRIBER_PORT = "tcp://127.0.0.1:5555"
PUBLISHER_PORT = "tcp://127.0.0.1:5556"
MODULE_NAME = "face"
CAMERA_ID = "main_camera"

# --- Security Configuration ---
# In pgvector, the <=> operator calculates Cosine Distance (0.0 is perfect match, 1.0 is orthogonal).
# A distance of 0.40 means we require at least 60% mathematical similarity to grant access.
MAX_ALLOWED_DISTANCE = 0.40 


def _decode_frame(frame_bytes: bytes) -> np.ndarray:
    """Decodes a byte array into an OpenCV BGR image."""
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)


def _find_closest_match_in_db(embedding: np.ndarray) -> Tuple[str, float]:
    """
    Executes a high-speed nearest-neighbor search in PostgreSQL using pgvector.
    Returns the person's name and the cosine distance.
    """
    db = SessionLocal()
    try:
        # Convert the numpy array to a standard Python list for SQL formatting
        vector_list = embedding.tolist()
        
        # Native SQL using the pgvector <=> (cosine distance) operator
        query = text("""
            SELECT full_name, (face_embedding <=> :vector) AS distance
            FROM persons
            WHERE face_embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
        """)
        
        # Execute query and fetch the absolute closest face
        result = db.execute(query, {"vector": str(vector_list)}).fetchone()
        
        if result:
            name, distance = result
            
            # Strict security gate: Ensure the match is close enough
            if distance <= MAX_ALLOWED_DISTANCE:
                return name, float(distance)
            else:
                return "unknown_person", float(distance)
        else:
            # The database is completely empty
            return "unknown_person", 1.0
            
    except Exception as e:
        logger.error(f"Database vector search failed: {e}")
        return "unknown_person", 1.0
    finally:
        # CRITICAL: Always release the connection back to the pool
        db.close()


def start_face_model() -> None:
    """Listens to the video stream, runs InsightFace inference, and searches pgvector."""
    context = zmq.Context()
    
    # Setup ZeroMQ Receiver (Video Stream)
    video_receiver = context.socket(zmq.SUB)
    video_receiver.connect(SUBSCRIBER_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1)  # Zero-Lag mode: only keep the newest frame

    # Setup ZeroMQ Sender (Results to Orchestrator)
    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(PUBLISHER_PORT)

    # 1. INIT: Load the AI Model into VRAM (Singleton)
    logger.info("Initializing FaceProcessorService (InsightFace)...")
    try:
        ai_service = FaceProcessorService()
        logger.info("Face module loaded into VRAM. Listening for video stream...")
    except Exception as e:
        logger.critical(f"FATAL: Could not load AI models into memory: {e}")
        return

    # 2. INFERENCE LOOP
    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)
            detections_payload = []

            # A. Extract Faces using InsightFace (Handles detection and alignment natively)
            faces = ai_service.app.get(frame)

            # B. Process each face found in the frame
            for face in faces:
                # InsightFace returns the bounding box as [x1, y1, x2, y2]
                box = face.bbox.astype(int)
                x, y, x2, y2 = box[0], box[1], box[2], box[3]
                w, h = x2 - x, y2 - y
                
                # Retrieve the 512-d normalized L2 embedding
                live_embedding = face.normed_embedding

                # C. Sub-second Vector Database Search
                name, distance = _find_closest_match_in_db(live_embedding)

                # D. Build the payload for the Orchestrator
                detections_payload.append({
                    "name": name,
                    "confidence": round(1.0 - distance, 4), # Convert distance back to a % confidence
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                })

            # 3. SEND RESULTS TO ORCHESTRATOR (Only if faces were found)
            if detections_payload:
                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "detections": detections_payload
                }
                result_sender.send_json(payload)

        except Exception as e:
            logger.debug(f"Inference cycle error: {e}")