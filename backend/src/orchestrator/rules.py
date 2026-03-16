import zmq
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Orchestrator")

RECEIVER_PORT = "tcp://127.0.0.1:5556"

def _evaluate_face_event(event: Dict[str, Any]) -> None:
    """Evaluates business rules specifically for facial recognition events."""
    detections: List[Dict[str, Any]] = event.get("detections", [])
    timestamp: str = event.get("timestamp", "unknown_time")
    camera_id: str = event.get("camera_id", "unknown_camera")

    if not detections:
        return

    # Iterate through all faces found in the current frame
    for detection in detections:
        name = detection.get("name", "unknown_person")
        confidence = detection.get("confidence", 0.0)
        
        # Convert confidence to a readable percentage
        conf_pct = confidence * 100

        if name == "unknown_person":
            logger.warning(
                f"[SECURITY ALERT] Unknown person detected at {timestamp} "
                f"on {camera_id} (Confidence: {conf_pct:.1f}%)"
            )
        else:
            logger.info(
                f"[ACCESS GRANTED] Resident recognized: {name} at {timestamp} "
                f"on {camera_id} (Similarity: {conf_pct:.1f}%)"
            )

def start_orchestrator() -> None:
    """Receives JSON events from all AI modules and evaluates business rules."""
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(RECEIVER_PORT)

    logger.info(f"Rule engine started. Waiting for events on {RECEIVER_PORT}")

    while True:
        try:
            event = receiver.recv_json()
            module = event.get("module")
            
            # Router logic: Send the event to the appropriate rule evaluator
            if module == "face":
                _evaluate_face_event(event)
            # El día de mañana, aquí agregarás: elif module == "weapons": _evaluate_weapon_event(event)
            
        except Exception as e:
            logger.error(f"Error processing incoming event: {e}")