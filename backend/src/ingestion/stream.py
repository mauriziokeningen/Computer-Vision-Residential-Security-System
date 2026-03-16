import cv2
import zmq
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VideoIngestion")

# --- Configuration ---
CAMERA_SOURCE = 0  # 0 es la cámara predeterminada
PUBLISHER_PORT = "tcp://127.0.0.1:5555"
FPS_LIMIT = 10      # Bajamos a 20 FPS para reducir carga en el bus de ZeroMQ
FRAME_DELAY = 1.0 / FPS_LIMIT
JPEG_QUALITY = 75   # Calidad óptima para balancear nitidez y peso de red

def start_ingestion() -> None:
    """Reads the camera stream and broadcasts frames via ZeroMQ."""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    
    try:
        socket.bind(PUBLISHER_PORT)
    except zmq.ZMQError as e:
        logger.error(f"Could not bind to port {PUBLISHER_PORT}: {e}. Is another process running?")
        return

    # Intentamos abrir la cámara
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    # VALIDACIÓN CRÍTICA: ¿La cámara realmente abrió?
    if not cap.isOpened():
        logger.error(f"FATAL: Could not open camera source {CAMERA_SOURCE}. "
                     "Check if it's connected or used by another app (Zoom, Teams, etc.)")
        return

    logger.info(f"Camera successfully opened. Broadcasting on {PUBLISHER_PORT}...")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame. Retrying in 1s...")
                time.sleep(1)
                continue
            
            # Codificación a JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if success:
                # Enviamos los bytes puros por el socket
                socket.send(buffer.tobytes())
                frame_count += 1
                
                # Cada 100 frames imprimimos un "latido" para saber que sigue vivo
                if frame_count % 100 == 0:
                    logger.info(f"Stream status: {frame_count} frames broadcasted.")
            
            time.sleep(FRAME_DELAY)

    except Exception as e:
        logger.error(f"Unexpected error in Ingestion module: {e}")
    finally:
        cap.release()
        socket.close()
        logger.info("Video ingestion stopped and camera released.")

if __name__ == "__main__":
    start_ingestion()