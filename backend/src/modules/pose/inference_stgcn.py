"""
Pose & Behavior Detection Worker Service.

Cascaded Pipeline:
    L1: YOLO11-Pose extracts 17 COCO joints per frame.
    L2: A sliding temporal window (deque) captures 15 frames of movement.
    L3: ST-GCN analyzes the coordinate matrix and outputs an action class.

Publishes detection metadata to orchestrator (5556) and annotator (5558).
"""
import os
import zmq
import time
import cv2
import numpy as np
import logging
import torch
from collections import deque
from ultralytics import YOLO

from src.modules.pose.analysis_stgcn import PoseInferenceService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoseWorker")

VIDEO_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")
ORCHESTRATOR_PUSH_PORT = os.getenv("ORCHESTRATOR_PUSH_PORT", "tcp://127.0.0.1:5556")
ANNOTATOR_PUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")
MODULE_NAME = "pose"
CAMERA_ID = os.getenv("CAMERA_ID", "main_camera")

def _decode_frame(frame_bytes: bytes):
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

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

    # 1. Load L1 (YOLO) and L2 (ST-GCN)
    logger.info("Initializing YOLO-Pose (L1) and ST-GCN (L2)...")
    try:
        yolo_eyes = YOLO("yolo11n-pose.pt")
        stgcn_brain = PoseInferenceService("stgcn_v1_production.ckpt")
    except Exception as e:
        logger.critical(f"FATAL: Could not load Pose models: {e}")
        return

    # 2. Temporal Buffer (15 frames)
    # Using a dictionary to track multiple people would be ideal, but for MVP
    # we track the largest/most confident person in the frame.
    pose_buffer = deque(maxlen=15)
    
    logger.info("Pose module ready. Listening for video stream...")
    decode_failures = 0

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)
            
            if frame is None:
                decode_failures += 1
                continue

            # Run YOLO-Pose
            results = yolo_eyes(frame, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")
            detections_payload = []

            for result in results:
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    # 1. Grab the raw frame dimensions (Height, Width)
                    h, w = result.orig_shape 
                    
                    # 2. Grab the primary subject
                    subject_kpts = result.keypoints.data[0].cpu().numpy() # Shape: (17, 3)
                    subject_bbox = result.boxes.xyxy[0].cpu().numpy().astype(int).tolist()
                    
                    # 3. Extract X and Y
                    coords_2d = subject_kpts[:, :2].copy() 
                    
                    # --- THE FIX: NORMALIZE TO [0, 1] ---
                    # Divide X by width, divide Y by height
                    coords_2d[:, 0] = coords_2d[:, 0] / w  
                    coords_2d[:, 1] = coords_2d[:, 1] / h  
                    
                    pose_buffer.append((coords_2d, subject_bbox))

            # If we have captured a full 15-frame movement, ask the ST-GCN
            if len(pose_buffer) == 15:
                # Extract just the coordinates from the buffer tuples
                sequence_coords = [item[0] for item in pose_buffer]
                latest_bbox = pose_buffer[-1][1] # Get bounding box of the latest frame

                # --- 1. ADD THIS LINE ---
                logger.info("⏱️ [DEBUG] 15 frames captured! Sending to ST-GCN...")
                
                # Analyze the movement
                action, confidence = stgcn_brain.predict_sequence(sequence_coords)

                logger.warning(f"🥊 [DEBUG] ST-GCN Output -> Action: {action} | Confidence: {confidence*100:.1f}%")

                # Only trigger orchestrator rules if it's an aggressive action
                if action != "neutral":
                    x1, y1, x2, y2 = latest_bbox
                    detections_payload.append({
                        "action": action,
                        "confidence": round(confidence, 4),
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    # Optional: Clear the buffer after a positive hit so it doesn't 
                    # double-trigger on frames 16, 17, 18 of the same punch.
                    pose_buffer.clear()

            # Broadcast to UI and Orchestrator
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

        except Exception as e:
            logger.error(f"Pose Inference cycle error: {e}")