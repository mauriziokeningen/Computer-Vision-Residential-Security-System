import os
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MediaPipe_Builder")

NUM_FRAMES = 15

def chunk_sequence(sequence: np.ndarray, window_size: int = 15) -> list:
    chunks = []
    total_frames = sequence.shape[0]
    step = window_size // 2 
    for start in range(0, total_frames - window_size + 1, step):
        chunks.append(sequence[start:start + window_size])
    return chunks

def process_json(file_path: Path, label: int):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not data["frames"]:
        return []
        
    # Get the last frame_id to know the exact physical time elapsed
    max_frame_id = data["frames"][-1]["frame_id"]
    
    # Initialize a blank timeline (X, Y, Z, Visibility) for 33 joints
    sequence = np.zeros((max_frame_id + 1, 33, 4), dtype=np.float32)
    
    # Impute the MediaPipe data into the exact correct physical time slots
    for frame_info in data["frames"]:
        f_idx = frame_info["frame_id"]
        if frame_info["skeleton"]:
            sequence[f_idx] = np.array(frame_info["skeleton"])
            
    if sequence.shape[0] >= NUM_FRAMES:
        return chunk_sequence(sequence, NUM_FRAMES)
    return []

def build_dataset():
    data_dir = Path("data/custom_poses")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_data, y_data = [], []

    # Map files to labels
    files_to_process = {
        "class_0_neutral.json": 0,
        "class_1_threat.json": 1
    }

    for filename, label in files_to_process.items():
        file_path = data_dir / filename
        if file_path.exists():
            chunks = process_json(file_path, label)
            X_data.extend(chunks)
            y_data.extend([label] * len(chunks))
            logger.info(f"Loaded {len(chunks)} chunks from {filename}")
        else:
            logger.warning(f"Missing file: {file_path}")

    if not X_data:
        logger.error("FATAL: No data found. Run the extractor script first!")
        return

    X_array = np.array(X_data)
    y_array = np.array(y_data)

    X_train, X_val, y_train, y_val = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    logger.info(f"Saved splits to {out_dir}! Train: {len(X_train)} | Val: {len(X_val)}")

if __name__ == "__main__":
    build_dataset()