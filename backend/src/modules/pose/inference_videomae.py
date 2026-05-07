"""
Pose & Action Detection Worker (VideoMAE v2).

Subscribes to the video stream via ZeroMQ, maintains a rolling 16-frame buffer,
and runs parameter-efficient spatio-temporal inference. Publishes:
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
import torch
from torchvision.transforms import Resize, Normalize, InterpolationMode

# Import your custom VideoMAE Wrapper
# Adjust the import path based on where you placed model.py
from src.modules.pose.model import TT2ResidentialVideoMAE 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoseInference")

# --- Network Topology ---
VIDEO_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")
ORCHESTRATOR_PUSH_PORT = os.getenv("ORCHESTRATOR_PUSH_PORT", "tcp://127.0.0.1:5556")
ANNOTATOR_PUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")
MODULE_NAME = "pose"
CAMERA_ID = os.getenv("CAMERA_ID", "main_camera")

# --- Model Configuration ---
CONFIDENCE_THRESHOLD = float(os.getenv("POSE_CONFIDENCE_THRESHOLD", "0.50"))
NUM_FRAMES = 16
TARGET_SIZE = 224
TEMPORAL_STRIDE = 2

def _decode_frame(frame_bytes: bytes) -> np.ndarray:
    """Deserializes the IPC byte payload into an OpenCV BGR matrix."""
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

    # 1. Hardware Detection Check
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"NVIDIA GPU DETECTED: {gpu_name}")
    else:
        logger.warning("CUDA NOT FOUND. Running on slow CPU. Check your 'torch' version and CUDA drivers.")

    logger.info(f"Loading VideoMAE backbone to {device}...")
    
    try:
        model = TT2ResidentialVideoMAE(num_classes=2)
        weights_path = os.path.join("src", "modules", "pose", "weights", "tt2_v1_final.ckpt")

        if os.path.exists(weights_path):
            logger.info(f"Loading weights from {weights_path}...")
            checkpoint = torch.load(weights_path, map_location=device)
            
            # 1. Get the state dict from Lightning's wrapper
            raw_state_dict = checkpoint['state_dict']
            
            # 2. Advanced key cleaning & Architecture Bridge
            clean_state_dict = {}
            for k, v in raw_state_dict.items():
                new_key = k
                
                # Strip the Lightning wrapper prefix
                if new_key.startswith('model.'):
                    new_key = new_key[6:]
                
                # --- ARCHITECTURE BRIDGE ---
                # Map the old Hugging Face Q/V Linear biases to the new standalone Parameter biases
                new_key = new_key.replace("attention.attention.query.base_layer.bias", "attention.attention.q_bias")
                new_key = new_key.replace("attention.attention.value.base_layer.bias", "attention.attention.v_bias")
                
                # The current architecture has no k_bias. Discard it to prevent metadata bloat.
                if "attention.attention.key.bias" in new_key:
                    continue
                    
                clean_state_dict[new_key] = v

            # 3. Load with strict=False
            msg = model.load_state_dict(clean_state_dict, strict=False)
            
            # Log what happened so we are sure the important parts (encoder/classifier) loaded
            logger.info(f"Load Result: Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
            
            if len(msg.missing_keys) > 50: # Arbitrary high number indicating a total mismatch
                logger.error("WARNING: Too many missing keys! The model architecture might not match the weights.")
            else:
                logger.info("Weight loading successful.")

        else:
            logger.warning(f"Weights not found at {weights_path}")
            return
        
        model.to(device)
        model.eval()
        
    except Exception as e:
        logger.critical(f"Failed to load VideoMAE model: {e}")
        return

    # SOTA Normalization matching the training pipeline
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = Resize((TARGET_SIZE, TARGET_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True)
    
    # Rolling buffer for the spatio-temporal window
    frame_buffer = deque(maxlen=NUM_FRAMES)
    decode_failures = 0

    logger.info("Pose model ready. Accumulating initial frame buffer...")

    frame_count = 0

    with torch.no_grad():
        while True:
            try:
                frame_bytes = video_receiver.recv()
                frame_bgr = _decode_frame(frame_bytes)
                
                if frame_bgr is None:
                    decode_failures += 1
                    continue

                frame_count += 1

                # OpenCV yields BGR. Model was trained on Decord (RGB). CRITICAL conversion.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                #Temporal Striding
                #Only add every 2nd frame to the memory buffer.
                #This stretches the 16-frame window across 3.2 seconds of real time, allowing the model to capture slower actions like pushing or prolonged fights.
                if frame_count % TEMPORAL_STRIDE != 0:
                    continue
                
                frame_buffer.append(frame_rgb)

                # Wait until we have a full 16-frame temporal window
                if len(frame_buffer) < NUM_FRAMES:
                    continue

                t0 = time.time()
                
                # --- Tensor Construction ---
                # 1. Stack frames to (T, H, W, C)
                stacked_frames = np.stack(frame_buffer)
                
                # 2. To Tensor & Permute to (T, C, H, W)
                frames_tensor = torch.from_numpy(stacked_frames).permute(0, 3, 1, 2).float() / 255.0
                frames_tensor = frames_tensor.to(device)
                
                # 3. Spatial transforms
                frames_tensor = resize(frames_tensor)
                frames_tensor = normalize(frames_tensor)
                
                # 4. Final format for VideoMAE: (Batch, Channels, Frames, Height, Width) -> (1, 3, 16, 224, 224)
                frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

                # --- Inference ---
                logits = model(pixel_values=frames_tensor)
                probs = torch.softmax(logits, dim=1)
                threat_prob = probs[0][1].item()
                infer_ms = (time.time() - t0) * 1000.0

                logger.info(f"[HEARTBEAT] Infer: {infer_ms:.0f}ms | Threat Prob: {threat_prob*100:.2f}%")

                detections_payload = []

                # Class 1 = Threat (Punch/Slap, Kicking, Pushing)
                if threat_prob >= CONFIDENCE_THRESHOLD:
                    # VideoMAE is a global classifier, so we create a bounding box 
                    # that covers the entire frame to ensure the Annotator draws it.
                    h, w, _ = frame_bgr.shape
                    
                    detections_payload.append({
                        "action": "fight",           # Triggers RN-04 in your Orchestrator
                        "confidence": round(threat_prob, 4),
                        "bbox": [10, 10, w-10, h-10] # Full-screen bounding box
                    })

                # Publish visual overlays to Annotator
                if detections_payload:
                    annotator_publisher.send_json({
                        "camera_id": CAMERA_ID,
                        "module": MODULE_NAME,
                        "detections": detections_payload,
                    })

                    # Publish logical event to Orchestrator Rule Engine
                    payload = {
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "camera_id": CAMERA_ID,
                        "module": MODULE_NAME,
                        "detections": detections_payload,
                    }
                    result_sender.send_json(payload)

                    logger.warning(
                        f"[THREAT DETECTED] Action=fight "
                        f"conf={threat_prob*100:.1f}% "
                        f"infer={infer_ms:.0f}ms"
                    )

            except Exception as e:
                logger.error(f"Inference cycle error: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting Pose Inference Worker...")
    start_pose_model()