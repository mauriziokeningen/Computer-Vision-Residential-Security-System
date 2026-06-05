"""
Video Ingestion Service.
Acts as the edge-node publisher in the IPC architecture. 
Captures raw hardware video feeds, applies lossy compression, and broadcasts
frames via ZeroMQ PUB socket to downstream AI inference workers.
"""
import cv2
import zmq
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VideoIngestion")

# --- Boundary Contract ---
# ASSUMPTION: Downstream subscribers MUST implement `zmq.CONFLATE` to drop stale frames.
# This is a fire-and-forget PUB socket; it does not guarantee delivery and will not wait for slow consumers.
PUBLISHER_PORT = "tcp://127.0.0.1:5555"
CAMERA_SOURCE = 0

# --- Trade-offs & Tuning ---
# WHY 10 FPS? Throttling at the source prevents buffer bloat in the ZeroMQ bus.
# Real-time physical security rarely requires >10 FPS, making higher framerates an unnecessary CPU tax.
FPS_LIMIT = 10      
FRAME_DELAY = 1.0 / FPS_LIMIT

# WHY JPEG 75? Raw BGR frames (640x480x3) consume ~1MB each, saturating the IPC bus at 10MB/s.
# H.264 encoding introduces unacceptable latency. JPEG @ 75 reduces payload to ~40KB per frame 
# while maintaining sufficient structural fidelity for InsightFace cosine distance matching.
JPEG_QUALITY = 75   

def start_ingestion() -> None:
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    try:
        socket.bind(PUBLISHER_PORT)
    except zmq.ZMQError as e:
        logger.error(f"Could not bind to port {PUBLISHER_PORT}: {e}. Is another process running?")
        return

    # TECH DEBT: cv2.VideoCapture is synchronous and blocking. 
    # If scaling to multi-camera (>4 feeds), this module must be refactored to use 
    # asynchronous GStreamer pipelines or dedicated multiprocessing queues to avoid GIL bottlenecks.
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    if not cap.isOpened():
        logger.error(f"FATAL: Could not open camera source {CAMERA_SOURCE}. "
                     "Hardware access denied or device disconnected.")
        return

    logger.info(f"Camera successfully opened. Broadcasting on {PUBLISHER_PORT}...")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                # CHESTERTON'S FENCE: Do NOT remove this sleep. 
                # If the USB hardware bus drops a frame or disconnects, an instant `continue` loop 
                # will cause a 100% CPU spike and trigger OS driver panics. The 1s delay allows the bus to reset.
                logger.error("Hardware stream interrupted. Retrying in 1s...")
                time.sleep(1)
                continue
            
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            if success:
                socket.send(buffer.tobytes())
                frame_count += 1
                
                # Telemetry heartbeat (Hidden in production via INFO log level config)
                if frame_count % 100 == 0:
                    logger.debug(f"Stream status: {frame_count} frames broadcasted.")
            
            time.sleep(FRAME_DELAY)

    except Exception as e:
        logger.error(f"Unexpected error in Ingestion module: {e}")
    finally:
        cap.release()
        socket.close()
        logger.info("Video ingestion stopped and hardware released.")

if __name__ == "__main__":
    start_ingestion()