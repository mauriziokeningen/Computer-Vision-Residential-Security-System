import cv2
import json
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO11_Pose_Extractor")

def extract_poses(source: int | str = 0, output_file: str = "class_0_idle.json", max_frames: int = 3000):
    """
    Extracts 17 COCO-format joints using YOLO11-Pose.
    Run this while standing idle or walking around to build your "Class 0: Neutral" dataset.
    """
    # Use the Nano or Small YOLO11-Pose model for maximum real-time speed
    logger.info("Loading YOLO11-Pose-Nano...")
    model = YOLO("yolo11n-pose.pt")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return

    extracted_data = {
        "label": "Neutral",
        "resolution": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "frames": []
    }

    logger.info(f"Starting extraction. Target: {max_frames} frames.")
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference (device='cuda' utilizes the 4070 Ti SUPER automatically)
        results = model(frame, verbose=False, device='cuda')
        
        frame_skeletons = []
        for result in results:
            if result.keypoints is not None:
                # Extract the (x, y, confidence) for all 17 COCO joints
                keypoints = result.keypoints.data.cpu().numpy() 
                for person in keypoints:
                    # Filter out low-confidence skeletons
                    if person[:, 2].mean() > 0.5:
                        frame_skeletons.append(person.tolist())

        if frame_skeletons:
            extracted_data["frames"].append({
                "frame_id": frame_count,
                "skeletons": frame_skeletons
            })

        frame_count += 1
        
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{max_frames} frames...")

    cap.release()
    
    # Save the lightweight mathematical representation
    out_path = Path("data/custom_poses") / output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(extracted_data, f)
        
    logger.info(f"Extraction complete. Saved to {out_path}")

if __name__ == "__main__":
    # Record 5 minutes (3000 frames at 10fps) of you doing nothing for your "Background" class[cite: 2]
    extract_poses(source=0, output_file="class_0_idle_1.json", max_frames=3000)