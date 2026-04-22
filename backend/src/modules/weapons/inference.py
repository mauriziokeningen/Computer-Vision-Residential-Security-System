"""
Weapon Detection Worker.
Subscribes to the video stream via ZeroMQ, runs YOLO inference,
and pushes detections to the Orchestrator.
"""
import zmq
import time
import base64
import logging
import numpy as np
import cv2
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WeaponInference")

SUBSCRIBER_PORT = "tcp://127.0.0.1:5555"
PUBLISHER_PORT = "tcp://127.0.0.1:5556"
MODULE_NAME = "weapons"
CAMERA_ID = "main_camera"

MODEL_WEIGHTS = "../../research/models/object_detection/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.50

# Only report actual threats, skip the generic "object" class
THREAT_CLASSES = {"knife", "pistol"}


def _decode_frame(frame_bytes: bytes) -> np.ndarray:
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)


def start_weapon_model() -> None:
    context = zmq.Context()

    # Subscribe to video stream
    video_receiver = context.socket(zmq.SUB)
    video_receiver.connect(SUBSCRIBER_PORT)
    video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
    video_receiver.setsockopt(zmq.CONFLATE, 1)

    # Push results to Orchestrator
    result_sender = context.socket(zmq.PUSH)
    result_sender.connect(PUBLISHER_PORT)

    # Load YOLO model
    logger.info(f"Loading weapon detection model: {MODEL_WEIGHTS}")
    try:
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    # Warmup pass
    logger.info("Running warmup inference...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    logger.info("Weapon model ready. Listening for video stream...")

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)

            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
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

            if detections_payload:
                _, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_b64 = base64.b64encode(jpeg_buffer.tobytes()).decode('utf-8')

                payload = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "camera_id": CAMERA_ID,
                    "module": MODULE_NAME,
                    "frame_data": frame_b64,
                    "detections": detections_payload,
                }
                result_sender.send_json(payload)

                for d in detections_payload:
                    logger.warning(
                        f"[WEAPON DETECTED] {d['class']} "
                        f"(Confidence: {d['confidence']*100:.1f}%)"
                    )

        except Exception as e:
            logger.debug(f"Inference cycle error: {e}")