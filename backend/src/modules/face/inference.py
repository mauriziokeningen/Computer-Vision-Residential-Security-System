import os
import cv2
import zmq
import time
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

# ==========================================
#    VRAM OPTIMIZATION: TENSORFLOW HARD LIMIT
# ==========================================
# IMPORTANTE: TensorFlow debe importarse y limitarse ANTES de DeepFace
import tensorflow as tf

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restringe a TensorFlow para que use estrictamente 4096 MB (4GB)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
except RuntimeError as e:
    print(f"VRAM Restriction Failed: {e}")

# Ahora sí, importamos DeepFace de forma segura
from deepface import DeepFace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceInference")

# --- ZeroMQ Configuration ---
SUBSCRIBER_PORT = "tcp://127.0.0.1:5555"
PUBLISHER_PORT = "tcp://127.0.0.1:5556"
MODULE_NAME = "face"
CAMERA_ID = "main_camera"

# --- AI Model Configuration ---
FACIAL_MODEL = "ArcFace"
DETECTOR_BACKEND = "opencv" # Usamos OpenCV para detección rápida, aunque no es tan precisa como YOLOv8
SIMILARITY_THRESHOLD = 0.40
REFERENCES_PATH = "mis_fotos" # Temporary fallback if DB is not ready

def _simulate_database_fetch() -> Dict[str, np.ndarray]:
    # Esto busca la carpeta 'mis_fotos' en la RAÍZ de donde estás ejecutando main.py
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, REFERENCES_PATH)
    
    logger.info(f"Checking images at: {full_path}")
    
    if not os.path.exists(full_path):
        logger.warning(f"FOLDER NOT FOUND: {full_path}")
        return {}
    
    """
    Simulates fetching pre-computed embeddings from a Vector Database (e.g., PostgreSQL + pgvector).
    In a real scenario, this function runs a SQL SELECT query.
    For now, it calculates them once from the local folder to simulate the DB state.
    """
    logger.info("Connecting to Vector Database... (Simulated)")
    db_embeddings = {}
    
    if not os.path.exists(REFERENCES_PATH):
        logger.warning(f"Database simulation failed: Folder '{REFERENCES_PATH}' not found.")
        return db_embeddings

    image_paths = [os.path.join(REFERENCES_PATH, f) for f in os.listdir(REFERENCES_PATH) if f.lower().endswith(('.jpg','.png'))]
    if not image_paths:
        return db_embeddings

    # Simulate generating the vector to store in DB
    generated_embeddings = []
    for img_path in image_paths:
        try:
            emb_info = DeepFace.represent(
                img_path=img_path, 
                model_name=FACIAL_MODEL, 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=True
            )
            generated_embeddings.append(emb_info[0]['embedding'])
        except Exception as e:
            logger.debug(f"Could not extract face from {img_path}: {e}")
            
    if generated_embeddings:
        # Assuming the folder belongs to 'Mauricio' for this mock DB
        db_embeddings["Mauricio"] = np.mean(generated_embeddings, axis=0)
        logger.info("Successfully loaded known embeddings from database.")
        
    return db_embeddings

def _find_best_match(live_embedding: List[float], known_db: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Compares the live face embedding against the database using Cosine Similarity."""
    best_name = "unknown_person"
    best_similarity = 0.0
    live_np = np.asarray(live_embedding)
    
    for name, ref_np in known_db.items():
        dot_product = np.dot(live_np, ref_np)
        norm_live = np.linalg.norm(live_np)
        norm_ref = np.linalg.norm(ref_np)
        
        if norm_live > 0 and norm_ref > 0:
            similarity = dot_product / (norm_live * norm_ref)
            if similarity > best_similarity:
                best_similarity = similarity
                if similarity >= SIMILARITY_THRESHOLD:
                    best_name = name
                    
    return best_name, best_similarity

def _decode_frame(frame_bytes: bytes) -> np.ndarray:
    """Decodes a byte array into an OpenCV BGR image."""
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

def start_face_model() -> None:
    """Listens to the video stream, runs ArcFace inference, and sends JSON results."""
    context = zmq.Context()
    
    video_receiver = context.socket(zmq.SUB)
    video_receiver.connect(SUBSCRIBER_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1) # Zero-Lag

    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(PUBLISHER_PORT)

    # 1. INIT: Load Database and warm up the AI model into VRAM
    logger.info("Initializing Face Module. Loading models into VRAM...")
    known_database = _simulate_database_fetch()
    
    if not known_database:
        logger.warning("No known faces in database. Every face will be 'unknown_person'.")

    # Warm-up inference (Forces Keras/PyTorch to allocate VRAM before the loop)
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        DeepFace.represent(img_path=dummy_img, model_name=FACIAL_MODEL, enforce_detection=False)
        logger.info(f"Model {FACIAL_MODEL} successfully loaded into VRAM.")
    except Exception as e:
        logger.error(f"Failed to warm up model: {e}")

    logger.info("Face module ready. Listening for video streams...")

    # 2. INFERENCE LOOP
    while True:
        frame_bytes = video_receiver.recv()
        #logger.info("Frame received by AI module...") # Descomenta esto para ver si la IA recibe datos
        frame = _decode_frame(frame_bytes)
        detections_payload = []

        try:
            # A. Extract Faces (Uses YOLOv8 Backend for speed)
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )

            if len(faces) > 0 and faces[0].get('confidence', 0) > 0:
                logger.info(f"¡CARA DETECTADA! Procesando identidad...")

            # B. Process each face found in the frame
            for face_obj in faces:
                if face_obj.get('confidence', 0) == 0:
                    continue
                
                facial_area = face_obj['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                # Validate crop coordinates
                if w > 0 and h > 0 and y+h <= frame.shape[0] and x+w <= frame.shape[1]:
                    face_img = frame[y:y+h, x:x+w]
                    
                    # C. Generate Embedding (Detector is skipped because we already cropped it)
                    emb_results = DeepFace.represent(
                        img_path=face_img,
                        model_name=FACIAL_MODEL,
                        detector_backend="skip",
                        enforce_detection=False
                    )

                    if emb_results:
                        current_embedding = emb_results[0]['embedding']
                        
                        # D. Query the Database
                        name, score = _find_best_match(current_embedding, known_database)
                        
                        # E. Build the detailed JSON payload for this specific face
                        detections_payload.append({
                            "name": name,
                            "confidence": round(float(score), 4),
                            "bbox": {"x": x, "y": y, "w": w, "h": h}
                        })

        except ValueError:
            # DeepFace throws ValueError when no faces are detected. We just ignore it and continue.
            pass
        except Exception as e:
            logger.debug(f"Inference warning: {e}")

        # 3. SEND RESULTS TO ORCHESTRATOR (Only if faces were found)
        if detections_payload:
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "camera_id": CAMERA_ID,
                "module": MODULE_NAME,
                "detections": detections_payload
            }
            result_sender.send_json(payload)