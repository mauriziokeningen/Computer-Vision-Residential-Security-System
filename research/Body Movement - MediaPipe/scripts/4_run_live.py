import cv2
import time
import torch
import numpy as np
import os
import mediapipe as mp
from collections import deque
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.system import TT2SkeletonSystem
from src.data.dataset import SkeletonDataset

# 1. Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. Load ST-GCN Checkpoint
ckpt_paths = list(Path("mlruns").rglob("*.ckpt"))
if not ckpt_paths:
    print("ERROR: No checkpoints found. Did you train the model?")
    sys.exit()

latest_ckpt = max(ckpt_paths, key=os.path.getctime)
print(f"Loading Checkpoint: {latest_ckpt}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TT2SkeletonSystem.load_from_checkpoint(latest_ckpt).to(device)
model.eval()

# --- THE FIX: Decoupled Normalizer ---
class LiveNormalizer(SkeletonDataset):
    def __init__(self):
        pass # Bypass the file loading entirely!

normalizer = LiveNormalizer()
# -------------------------------------

def run_live():
    cap = cv2.VideoCapture(0)
    # Rolling buffer of 15 frames (33 joints, 4 channels)
    sequence_buffer = deque(maxlen=15)
    
    FPS_TARGET = 30
    frame_time = 1.0 / FPS_TARGET
    last_process_time = time.time()

    print("Live Inference Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        
        if current_time - last_process_time >= frame_time:
            actual_fps = 1.0 / (current_time - last_process_time + 1e-6)
            last_process_time = current_time
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            current_skeleton = np.zeros((33, 4))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    current_skeleton[i] = [lm.x, lm.y, lm.z, lm.visibility]
            
            sequence_buffer.append(current_skeleton)

            if len(sequence_buffer) == 15:
                seq_array = np.array(sequence_buffer)
                
                # Check if we actually see a person (prevent ghost detections)
                if np.sum(seq_array[:, :, 3]) > 2.0:
                    seq_array = np.nan_to_num(seq_array, nan=0.0)
                    
                    norm_seq = normalizer._normalize_and_center(seq_array)
                    
                    tensor_in = torch.from_numpy(norm_seq).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = model(tensor_in)
                        probs = torch.softmax(logits, dim=1)[0]
                        threat_prob = probs[1].item()

                    label = "THREAT" if threat_prob > 0.45 else "Neutral"
                    color = (0, 0, 255) if label == "THREAT" else (0, 255, 0)
                    
                    cv2.rectangle(frame, (0,0), (450, 70), (0,0,0), -1)
                    cv2.putText(frame, f"{label}: {threat_prob:.2f}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("TT2 3D-STGCN Security Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()