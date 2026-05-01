"""
Frame Annotator Service.

Single source of truth for visual overlays. Subscribes to:
    1. Raw frames from the ingestion publisher (port 5555, CONFLATE -> always latest).
    2. Detection metadata from inference workers (port 5558, no conflate -> all events).

For every raw frame received, draws all currently-active detections (TTL window)
on top and republishes the annotated frame on port 5557. Both the live MJPEG
feed (LocalCameraStreamService) and the orchestrator's evidence buffer
subscribe to that single annotated stream, guaranteeing pixel-identical visuals
between live view and persisted incident evidence.

Architectural decisions:
    - One process, one polling loop. No threads inside the annotator: zmq.Poller
      handles both inputs in a single event loop. Simpler than locks, and the
      drawing throughput on a single core is more than enough for ~20 FPS at 720p.
    - Detection TTL (DEFAULT_DETECTION_TTL_SECONDS) is the max age a detection
      stays "active" on the frame. Beyond that it's dropped. This handles the
      asymmetry between fast inference (bbox arrives quickly) and the absence of
      a follow-up detection (e.g. weapon left the frame): without TTL the bbox
      would freeze on the last position forever.
    - Pass-through path: when no detections are active, the raw JPEG is forwarded
      as-is without re-encoding. Saves one decode + one encode per frame in the
      common case.
"""
import os
import zmq
import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Any
from dataclasses import dataclass, field

from src.utils.draw import draw_module_detections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Annotator")

# Endpoints sourced from the environment with the current localhost
# topology as defaults. Keeps a multi-host or containerized deployment
# configurable without code changes.
INGEST_SUB_PORT = os.getenv("VIDEO_SUB_PORT", "tcp://127.0.0.1:5555")            # raw frames from stream.py
DETECTIONS_SUB_PORT = os.getenv("ANNOTATOR_PUB_PORT", "tcp://127.0.0.1:5558")    # detection metadata from workers
ANNOTATED_PUB_PORT = os.getenv("ANNOTATED_PUB_PORT", "tcp://127.0.0.1:5557")     # annotated frames out

# How long a detection stays drawn on the frame after the last time it was reported.
# At ~20 FPS inference on weapons + ~30 FPS on face, 0.5s comfortably covers the
# next inference cycle from the slowest module. Tuned downward will cause flicker;
# tuned upward will leave stale boxes lingering when objects exit the scene.
DEFAULT_DETECTION_TTL_SECONDS = float(os.getenv("ANNOTATOR_TTL_SECONDS", "0.5"))

# JPEG quality for the republished annotated frame. Matches the workers' previous
# evidence quality (75) so that the live feed and stored evidence are visually
# indistinguishable.
JPEG_QUALITY = int(os.getenv("ANNOTATOR_JPEG_QUALITY", "75"))


@dataclass
class _DetectionEntry:
    """A detection set tagged with its expiration timestamp."""
    expires_at: float
    module: str
    detections: List[Dict[str, Any]]


class DetectionBuffer:
    """
    TTL store keyed by camera_id, then by module.

    We store the latest detection set per (camera, module) instead of accumulating
    a stream of events. Reason: each new detection event from a worker represents
    the worker's current view of the world for that camera — the previous one is
    stale by definition. Accumulating would draw duplicated/jittering boxes.
    """
    def __init__(self, ttl_seconds: float = DEFAULT_DETECTION_TTL_SECONDS):
        self.ttl = ttl_seconds
        # camera_id -> module -> _DetectionEntry
        self._items: Dict[str, Dict[str, _DetectionEntry]] = {}

    def update(self, camera_id: str, module: str, detections: List[Dict[str, Any]]) -> None:
        cam_bucket = self._items.setdefault(camera_id, {})
        cam_bucket[module] = _DetectionEntry(
            expires_at=time.time() + self.ttl,
            module=module,
            detections=detections,
        )

    def get_active(self, camera_id: str) -> List[_DetectionEntry]:
        """Returns all non-expired entries for the given camera. Prunes as it goes."""
        now = time.time()
        cam_bucket = self._items.get(camera_id, {})
        active = [e for e in cam_bucket.values() if e.expires_at > now]
        # Prune expired in place
        if len(active) != len(cam_bucket):
            self._items[camera_id] = {e.module: e for e in active}
        return active


def _decode(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def start_annotator() -> None:
    context = zmq.Context()

    # Raw frames in. CONFLATE: we only ever care about the latest frame, so the
    # annotator never lags behind the camera even if drawing takes longer than
    # the inter-frame interval.
    raw_sub = context.socket(zmq.SUB)
    raw_sub.connect(INGEST_SUB_PORT)
    raw_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    raw_sub.setsockopt(zmq.CONFLATE, 1)

    # Detections in. NO CONFLATE: every detection event matters; if we drop one,
    # we miss drawing a class on the current frames. Workers throttle naturally
    # (inference latency ~30-50ms), so the queue never grows out of control.
    det_sub = context.socket(zmq.SUB)
    det_sub.bind(DETECTIONS_SUB_PORT)
    det_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    # Annotated frames out.
    annotated_pub = context.socket(zmq.PUB)
    annotated_pub.bind(ANNOTATED_PUB_PORT)

    # Brief warmup so subscribers (LocalCameraStreamService, orchestrator listener)
    # have time to connect before the first message lands. ZMQ slow-joiner pattern.
    time.sleep(0.2)

    poller = zmq.Poller()
    poller.register(raw_sub, zmq.POLLIN)
    poller.register(det_sub, zmq.POLLIN)

    buffer = DetectionBuffer()
    logger.info(
        f"Annotator online. raw={INGEST_SUB_PORT}  "
        f"detections={DETECTIONS_SUB_PORT}  out={ANNOTATED_PUB_PORT}"
    )

    while True:
        try:
            socks = dict(poller.poll(timeout=100))

            # Drain detection events first — they're cheap, low-volume, and we
            # want the buffer up to date before we draw the next frame.
            if det_sub in socks:
                while True:
                    try:
                        meta = det_sub.recv_json(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    buffer.update(
                        camera_id=meta.get("camera_id", "unknown"),
                        module=meta.get("module", "unknown"),
                        detections=meta.get("detections", []),
                    )

            # Process the latest raw frame, if any.
            if raw_sub in socks:
                frame_bytes = raw_sub.recv()
                # SSoT camera assumption: today the system runs with a single camera.
                # When multi-camera lands, the ingestion publisher must include
                # camera_id in the message envelope and we key the buffer accordingly.
                camera_id = "main_camera"

                active = buffer.get_active(camera_id)

                if not active:
                    # Pass-through: no decode/encode roundtrip, just forward the
                    # JPEG bytes. Common case during quiet periods.
                    annotated_pub.send(frame_bytes)
                    continue

                frame = _decode(frame_bytes)
                if frame is None:
                    continue

                # Draw all active modules on the same buffer. Order is stable
                # but not enforced: face green/orange, weapons red, pose yellow/red.
                for entry in active:
                    draw_module_detections(frame, entry.module, entry.detections)

                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    annotated_pub.send(jpeg.tobytes())

        except Exception as e:
            # Catching generic exceptions keeps a single bad frame from killing
            # the whole live view. Symptomatic logging only.
            logger.debug(f"Annotator cycle error: {e}")

