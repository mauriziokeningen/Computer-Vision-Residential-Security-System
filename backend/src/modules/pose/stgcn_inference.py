"""
Pose / Aggression Detection Worker.

Subscribes to the video stream via ZeroMQ, extracts 3D skeletons using MediaPipe, 
maintains a temporal rolling buffer, and runs the ONNX ST-GCN model.
Publishes:
    - Detection metadata to the orchestrator (PUSH, port 5556)
    - Detection metadata to the annotator (PUB, port 5558)
"""
import os
import zmq
import time
import logging
import numpy as np
import cv2
from collections import deque
from pathlib import Path
from typing import Optional

import mediapipe as mp
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoseInference")

VIDEO_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")
ORCHESTRATOR_PUSH_PORT = os.getenv("ORCHESTRATOR_PUSH_PORT", "tcp://127.0.0.1:5556")
ANNOTATOR_PUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")
MODULE_NAME = "pose"
CAMERA_ID = os.getenv("CAMERA_ID", "main_camera")

# Use 'fight' to trigger the existing RN-04 aggressive_actions rule in orchestrator
THREAT_ACTION_LABEL = "fight" 
CONFIDENCE_THRESHOLD = float(os.getenv("POSE_CONFIDENCE_THRESHOLD", "0.60"))

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
ONNX_WEIGHTS = ROOT_DIR / "backend" / "src" / "modules" / "pose" / "weights" / "stgcn_pose.onnx"

def _decode_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

def _normalize_skeleton(seq_array: np.ndarray) -> np.ndarray:
    """Decoupled ST-GCN centering logic."""
    norm_seq = np.zeros_like(seq_array)
    for f in range(seq_array.shape[0]):
        valid_mask = (seq_array[f, :, 3] > 0.1)
        if not np.any(valid_mask): continue
        
        left_hip, right_hip = seq_array[f, 23], seq_array[f, 24]
        if left_hip[3] > 0.1 and right_hip[3] > 0.1:
            pelvis = (left_hip[:3] + right_hip[:3]) / 2.0
        else:
            pelvis = np.mean(seq_array[f][valid_mask][:, :3], axis=0)
            
        norm_seq[f, valid_mask, :3] = seq_array[f, valid_mask, :3] - pelvis
        norm_seq[f, valid_mask, 3] = seq_array[f, valid_mask, 3]
        
    max_spread = np.max(np.abs(norm_seq[:, :, :3]))
    if max_spread > 1e-5:
        norm_seq[:, :, :3] = norm_seq[:, :, :3] / max_spread
    return norm_seq

def start_pose_model() -> None:
    context = zmq.Context()

    video_receiver = context.socket(zmq.SUB)
    video_receiver.connect(VIDEO_SUB_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1)

    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(ORCHESTRATOR_PUSH_PORT)

    annotator_publisher = context.socket(zmq.PUB)
    annotator_publisher.connect(ANNOTATOR_PUB_PORT)

    if not ONNX_WEIGHTS.exists():
        logger.critical(f"FATAL: ONNX model not found at {ONNX_WEIGHTS}")
        return

    logger.info(f"Loading ST-GCN ONNX model: {ONNX_WEIGHTS}")
    ort_session = ort.InferenceSession(str(ONNX_WEIGHTS), providers=['CPUExecutionProvider'])
    
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sequence_buffer = deque(maxlen=15)
    logger.info("Pose model ready. Listening for video stream...")

    decode_failures = 0
    last_idle_log = 0.0

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)
            if frame is None:
                decode_failures += 1
                continue

            t0 = time.time()
            
            # 1. Extract MediaPipe Skeleton
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(image_rgb)
            
            current_skeleton = np.zeros((33, 4))
            bbox = None
            
            if results.pose_landmarks:
                h, w, _ = frame.shape
                x_coords, y_coords = [], []
                
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    current_skeleton[i] = [lm.x, lm.y, lm.z, lm.visibility]
                    if lm.visibility > 0.1:
                        x_coords.append(lm.x * w)
                        y_coords.append(lm.y * h)
                
                if x_coords and y_coords:
                    # Pad the bounding box slightly
                    bbox = [int(min(x_coords)-20), int(min(y_coords)-20), int(max(x_coords)+20), int(max(y_coords)+20)]
                    
            sequence_buffer.append(current_skeleton)

            detections_payload = []

            # 2. Run ST-GCN inference if buffer is full and person is visible
            if len(sequence_buffer) == 15:
                seq_array = np.array(sequence_buffer)
                
                if np.sum(seq_array[:, :, 3]) > 2.0 and bbox is not None:
                    norm_seq = _normalize_skeleton(seq_array)
                    
                    # Prepare ONNX input: (1, 4, 15, 33)
                    onnx_input = np.expand_dims(np.transpose(norm_seq, (2, 0, 1)), axis=0).astype(np.float32)
                    
                    logits = ort_session.run(None, {'skeleton_sequence': onnx_input})[0]
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                    threat_prob = float(probs[0][1])

                    # Only publish if it breaches the confidence threshold
                    if threat_prob > CONFIDENCE_THRESHOLD:
                        detections_payload.append({
                            "action": THREAT_ACTION_LABEL,
                            "confidence": round(threat_prob, 4),
                            "bbox": bbox
                        })

            infer_ms = (time.time() - t0) * 1000.0

            if detections_payload:
                annotator_publisher.send_json({
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "detections": detections_payload,
                })

                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "detections": detections_payload,
                }
                result_sender.send_json(payload)

                logger.warning(f"[AGGRESSION DETECTED] conf={threat_prob*100:.1f}% infer={infer_ms:.0f}ms")
            else:
                now = time.time()
                if now - last_idle_log > 5.0:
                    logger.info(f"[idle] infer={infer_ms:.0f}ms")
                    last_idle_log = now

        except Exception as e:
            logger.debug(f"Pose Inference cycle error: {e}")