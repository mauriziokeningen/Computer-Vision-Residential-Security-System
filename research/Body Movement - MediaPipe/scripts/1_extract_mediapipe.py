import cv2
import mediapipe as mp
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MediaPipe_Extractor")

def extract_mediapipe_poses(output_file: str, max_frames: int = 1800, fps_target: int = 30):
    """
    Extracts 33 skeletal joints in 3D (X, Y, Z) using MediaPipe.
    Enforces a strict FPS lock to guarantee temporal consistency for the ST-GCN.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # model_complexity=1 is a great balance of speed and accuracy for webcams
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam.")
        return

    extracted_data = {
        "resolution": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "frames": []
    }

    # --- 3 SECOND COUNTDOWN ---
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f"GET READY: {i}", (150, 250), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 165, 255), 4)
        cv2.imshow("MediaPipe Capture", frame)
        cv2.waitKey(1000)

    logger.info(f"STARTING CAPTURE: {output_file} | Target: {max_frames} frames at {fps_target} FPS")
    
    frame_count = 0
    frame_time = 1.0 / fps_target
    last_capture_time = time.time()

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        current_time = time.time()
        
        # --- STRICT FPS LOCK ---
        # We only process a frame if exactly 1/30th of a second has passed.
        # This guarantees physical velocity is perfectly preserved for the ST-GCN.
        if current_time - last_capture_time >= frame_time:
            last_capture_time = current_time
            
            # MediaPipe requires RGB images
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            frame_skeleton = []
            
            if results.pose_landmarks:
                # Draw the skeleton on the screen so you can see it working
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract all 33 joints: [X, Y, Z, Visibility]
                for lm in results.pose_landmarks.landmark:
                    frame_skeleton.append([lm.x, lm.y, lm.z, lm.visibility])
            
            # Append to data (even if empty, to prevent the Time-Warp bug!)
            extracted_data["frames"].append({
                "frame_id": frame_count,
                "skeleton": frame_skeleton 
            })
            
            frame_count += 1
            
            # UI Feedback
            cv2.putText(frame, f"REC: {frame_count} / {max_frames}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow("MediaPipe Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    out_path = Path("data/custom_poses") / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(extracted_data, f)
    logger.info(f"Saved dataset to {out_path}")

if __name__ == "__main__":
    # INSTRUCTIONS:
    # 1. Run this script once to record 60 seconds of NEUTRAL actions.
    #    (Uncomment the line below, run script, wave, sit, stand idle)
     extract_mediapipe_poses("class_0_neutral.json", max_frames=1800)
    
    # 2. Change the filename, uncomment the line below, and run AGAIN for THREAT actions.
    #    (Throw punches, kicks, and aggressive shoves at the camera)
    #extract_mediapipe_poses("class_1_threat.json", max_frames=1800)