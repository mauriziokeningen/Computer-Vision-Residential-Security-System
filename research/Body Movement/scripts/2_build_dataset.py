import os
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split # NEW

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Dataset_Builder")

NUM_FRAMES = 15 
THREAT_ACTIONS = {"A050", "A051", "A052"} 
DECOY_ACTIONS = {"A007", "A008", "A009", "A023", "A027", "A031", "A040"} 

def parse_ntu_skeleton(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if not lines: return None

    frame_count = int(lines[0].strip())
    frames_data = []
    line_idx = 1
    
    for _ in range(frame_count):
        if line_idx >= len(lines): break
        body_count = int(lines[line_idx].strip())
        line_idx += 1
        
        if body_count == 0:
            frames_data.append(np.zeros((25, 2))) # FIX: Only 2 dimensions
            continue
            
        line_idx += 1 
        joint_count = int(lines[line_idx].strip())
        line_idx += 1
        
        joints = []
        for _ in range(joint_count):
            coords = lines[line_idx].strip().split()
            # FIX: Only grab X and Y. Ignore Z (Depth)
            joints.append([float(coords[0]), float(coords[1])]) 
            line_idx += 1
            
        frames_data.append(joints)
        
        for _ in range(1, body_count):
            line_idx += 1
            j_count = int(lines[line_idx].strip())
            line_idx += 1 + j_count

    return np.array(frames_data)

def map_ntu_to_coco(ntu_keypoints: np.ndarray) -> np.ndarray:
    if ntu_keypoints is None or len(ntu_keypoints) == 0: return None
    num_frames = ntu_keypoints.shape[0]
    coco = np.zeros((num_frames, 17, 2), dtype=np.float32) # FIX: 2 channels

    for f in range(num_frames):
        coco[f, 0] = (ntu_keypoints[f, 3] + ntu_keypoints[f, 2]) / 2.0 
        coco[f, 1:5] = ntu_keypoints[f, 3] 
        coco[f, 5] = ntu_keypoints[f, 4] 
        coco[f, 6] = ntu_keypoints[f, 8] 
        coco[f, 7] = ntu_keypoints[f, 5] 
        coco[f, 8] = ntu_keypoints[f, 9] 
        coco[f, 9] = ntu_keypoints[f, 6] 
        coco[f, 10] = ntu_keypoints[f, 10] 
        coco[f, 11] = ntu_keypoints[f, 12] 
        coco[f, 12] = ntu_keypoints[f, 16] 
        coco[f, 13] = ntu_keypoints[f, 13] 
        coco[f, 14] = ntu_keypoints[f, 17] 
        coco[f, 15] = ntu_keypoints[f, 14] 
        coco[f, 16] = ntu_keypoints[f, 18] 

    return coco

def chunk_sequence(sequence: np.ndarray, window_size: int = 15) -> list:
    chunks = []
    total_frames = sequence.shape[0]
    step = window_size // 2 
    for start in range(0, total_frames - window_size + 1, step):
        chunks.append(sequence[start:start + window_size])
    return chunks

def build_dataset():
    ntu_dir = Path("data/raw_ntu/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons")
    custom_dir = Path("data/custom_poses")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_data, y_data = [], []

    # 1. NTU RGB+D
    if ntu_dir.exists():
        for file_path in ntu_dir.glob("*.skeleton"):
            filename = file_path.name
            action_code = filename[filename.find('A'):filename.find('A')+4]
            
            if action_code in THREAT_ACTIONS: label = 1
            elif action_code in DECOY_ACTIONS: label = 0
            else: continue 

            ntu_array = parse_ntu_skeleton(str(file_path))
            coco_array = map_ntu_to_coco(ntu_array)
            
            if coco_array is not None and coco_array.shape[0] >= NUM_FRAMES:
                chunks = chunk_sequence(coco_array, NUM_FRAMES)
                X_data.extend(chunks)
                y_data.extend([label] * len(chunks))

    # 2. YOLO CUSTOM POSES
    if custom_dir.exists():
        for file_path in custom_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            sequence = []
            for frame_info in data["frames"]:
                if frame_info["skeletons"]:
                    # FIX: Slice to grab only [X, Y] and discard [Confidence]
                    sequence.append(np.array(frame_info["skeletons"][0])[:, :2])
            
            custom_array = np.array(sequence)
            if custom_array.shape[0] >= NUM_FRAMES:
                chunks = chunk_sequence(custom_array, NUM_FRAMES)
                X_data.extend(chunks)
                y_data.extend([0] * len(chunks))

    # 3. SAFETY CHECK BEFORE SAVING
    if len(X_data) == 0:
        logger.error("FATAL: No data was found!")
        logger.error(f"Check 1: Are there .skeleton files in {ntu_dir.absolute()}?")
        logger.error(f"Check 2: Are there .json files in {custom_dir.absolute()}?")
        return # Exit early instead of crashing

    X_array = np.array(X_data)
    y_array = np.array(y_data)
    
    logger.info(f"Successfully loaded {len(X_array)} total chunks.")

    # Split the data 80/20 for Train/Validation
    X_train, X_val, y_train, y_val = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    logger.info(f"Saved splits to {out_dir}! Train size: {len(X_train)} | Val size: {len(X_val)}")

if __name__ == "__main__":
    build_dataset()