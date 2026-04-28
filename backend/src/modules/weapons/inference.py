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
"""
import zmq
import time
import logging
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WeaponInference")

VIDEO_SUB_PORT = "tcp://127.0.0.1:5555"
ORCHESTRATOR_PUSH_PORT = "tcp://127.0.0.1:5556"
ANNOTATOR_PUB_PORT = "tcp://127.0.0.1:5558"
MODULE_NAME = "weapons"
CAMERA_ID = "main_camera"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
WEIGHTS_DIR = ROOT_DIR / "research" / "models" / "object_detection" / "weights"
COREML_WEIGHTS = WEIGHTS_DIR / "best2.mlpackage"
PYTORCH_WEIGHTS = WEIGHTS_DIR / "best2.pt"

CONFIDENCE_THRESHOLD = 0.50
THREAT_CLASSES = {"knife", "pistol"}


def _resolve_model_path() -> tuple[str, str]:
    """
    Returns (path_to_load, backend_label_for_logs).

    CoreML export wins when present. The backend label is logged on startup
    so it's obvious from stdout which path is in use after a deploy or a
    fresh checkout.
    """
    if COREML_WEIGHTS.exists():
        return str(COREML_WEIGHTS), "CoreML (Apple Silicon accelerated)"
    if PYTORCH_WEIGHTS.exists():
        return str(PYTORCH_WEIGHTS), "PyTorch (CPU)"
    raise FileNotFoundError(
        f"No weights found at {COREML_WEIGHTS} or {PYTORCH_WEIGHTS}"
    )


def _decode_frame(frame_bytes: bytes) -> np.ndarray:
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

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)

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

