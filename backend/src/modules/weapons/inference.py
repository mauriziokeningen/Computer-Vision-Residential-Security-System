"""
Weapon Detection Worker — DEBUG INSTRUMENTED.

Same logic as the production worker, but additionally:
    - Times every YOLO inference call (logged in ms).
    - When a positive detection fires, saves the EXACT frame the model saw
      to /tmp/weapon_debug/, named with timestamp + confidence.

This lets us answer the question: when the system reports a "pistol" while
no pistol is in front of the camera, is the model:
    (a) processing a stale frame (pipeline lag), or
    (b) firing a false positive on the current frame?

Inspect /tmp/weapon_debug/*.jpg after reproducing. If the saved frames show
the weapon: it's pipeline lag. If they don't: it's the model.

Once we have the answer, this instrumentation can be removed.
"""
import os
import zmq
import time
import logging
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WeaponInference")

VIDEO_SUB_PORT = "tcp://127.0.0.1:5555"            # raw frames in
ORCHESTRATOR_PUSH_PORT = "tcp://127.0.0.1:5556"    # rule events out (PUSH/PULL)
ANNOTATOR_PUB_PORT = "tcp://127.0.0.1:5558"        # detection metadata out (PUB/SUB)
MODULE_NAME = "weapons"
CAMERA_ID = "main_camera"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
MODEL_WEIGHTS = str(ROOT_DIR / "research" / "models" / "object_detection" / "weights" / "best2.pt")
CONFIDENCE_THRESHOLD = 0.50

THREAT_CLASSES = {"knife", "pistol"}

# --- DEBUG ----------------------------------------------------------------
DEBUG_FRAME_DIR = "/tmp/weapon_debug"
os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
# --------------------------------------------------------------------------


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

    logger.info(f"Loading weapon detection model: {MODEL_WEIGHTS}")
    try:
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return

    logger.info("Running warmup inference...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    logger.info(f"Weapon model ready. DEBUG frames -> {DEBUG_FRAME_DIR}")

    while True:
        try:
            frame_bytes = video_receiver.recv()
            frame = _decode_frame(frame_bytes)

            # --- DEBUG: time the inference ---
            t0 = time.time()
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            infer_ms = (time.time() - t0) * 1000.0
            # ---------------------------------

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
                # --- DEBUG: persist the exact frame the model saw ---
                # Filename format: HHMMSS_mmm_class_confXX.jpg
                # Sorting by name = sorting by capture time.
                ms = int(time.time() * 1000) % 1000
                ts = time.strftime("%H%M%S") + f"_{ms:03d}"
                top = max(detections_payload, key=lambda d: d["confidence"])
                debug_name = f"{ts}_{top['class']}_conf{int(top['confidence']*100)}.jpg"
                cv2.imwrite(
                    os.path.join(DEBUG_FRAME_DIR, debug_name),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                # -----------------------------------------------------

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

                for d in detections_payload:
                    logger.warning(
                        f"[WEAPON DETECTED] {d['class']} "
                        f"conf={d['confidence']*100:.1f}% "
                        f"infer={infer_ms:.0f}ms saved={debug_name}"
                    )
            else:
                # Periodic timing log even without detections, so we can see
                # whether inference latency drifts up over time (thermal throttling).
                # Print roughly once per second to avoid log spam.
                now = time.time()
                last = getattr(start_weapon_model, "_last_idle_log", 0.0)
                if now - last > 1.0:
                    logger.info(f"[idle] infer={infer_ms:.0f}ms")
                    start_weapon_model._last_idle_log = now

        except Exception as e:
            logger.debug(f"Inference cycle error: {e}")