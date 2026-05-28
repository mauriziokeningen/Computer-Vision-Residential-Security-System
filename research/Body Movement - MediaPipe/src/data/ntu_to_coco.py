import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NTU_COCO_Mapper")

def map_ntu_to_coco(ntu_keypoints: np.ndarray) -> np.ndarray:
    """
    Transforms an NTU RGB+D skeleton (25 joints) to a COCO skeleton (17 joints).
    Input shape expected: (Num_Frames, 25_Joints, 3_Coords) -> (x, y, confidence)
    Output shape: (Num_Frames, 17_Joints, 3_Coords)
    """
    num_frames = ntu_keypoints.shape[0]
    coco_keypoints = np.zeros((num_frames, 17, 3), dtype=np.float32)

    # ---------------------------------------------------------
    # The NTU (25) -> COCO (17) Industry Standard Mapping[cite: 2]
    # Note: NTU arrays are 0-indexed in Python, so NTU joint 1 is index 0.
    # ---------------------------------------------------------
    
    for f in range(num_frames):
        # 1. Averaging: Calculate Nose (COCO #0) by averaging NTU Head (#4) and Neck (#3)[cite: 2]
        coco_keypoints[f, 0] = (ntu_keypoints[f, 3] + ntu_keypoints[f, 2]) / 2.0
        
        # Eyes and Ears (COCO 1-4) - NTU doesn't have exact facial joints, 
        # so we map them closely to the Head joint (index 3) to prevent NaN errors.
        coco_keypoints[f, 1] = ntu_keypoints[f, 3] # L Eye
        coco_keypoints[f, 2] = ntu_keypoints[f, 3] # R Eye
        coco_keypoints[f, 3] = ntu_keypoints[f, 3] # L Ear
        coco_keypoints[f, 4] = ntu_keypoints[f, 3] # R Ear

        # 2. Direct Mapping: Arms map 1:1[cite: 2]
        # NTU #5,6,7 (Arm) -> YOLO #5,7,9 (Shoulder, Elbow, Wrist)[cite: 2]
        coco_keypoints[f, 5] = ntu_keypoints[f, 4]   # L Shoulder
        coco_keypoints[f, 6] = ntu_keypoints[f, 8]   # R Shoulder
        coco_keypoints[f, 7] = ntu_keypoints[f, 5]   # L Elbow
        coco_keypoints[f, 8] = ntu_keypoints[f, 9]   # R Elbow
        coco_keypoints[f, 9] = ntu_keypoints[f, 6]   # L Wrist
        coco_keypoints[f, 10] = ntu_keypoints[f, 10] # R Wrist

        # 3. Direct Mapping: Legs map 1:1[cite: 2]
        coco_keypoints[f, 11] = ntu_keypoints[f, 12] # L Hip
        coco_keypoints[f, 12] = ntu_keypoints[f, 16] # R Hip
        coco_keypoints[f, 13] = ntu_keypoints[f, 13] # L Knee
        coco_keypoints[f, 14] = ntu_keypoints[f, 17] # R Knee
        coco_keypoints[f, 15] = ntu_keypoints[f, 14] # L Ankle
        coco_keypoints[f, 16] = ntu_keypoints[f, 18] # R Ankle

        # 4. Hips: COCO requires Mid-Hip math, using NTU Spine-Base (#1) as anchor[cite: 2]
        # In COCO, hip mid-point isn't an explicit joint, but the spine base acts as 
        # the geometric center for scaling and normalizing the body later[cite: 2].
        
    return coco_keypoints

# Example Usage:
if __name__ == "__main__":
    # Simulating a loaded NTU sequence of 100 frames
    dummy_ntu_data = np.random.rand(100, 25, 3) 
    
    logger.info(f"Input NTU Shape: {dummy_ntu_data.shape}")
    coco_data = map_ntu_to_coco(dummy_ntu_data)
    logger.info(f"Output COCO Shape: {coco_data.shape}")
    logger.info("Successfully converted 25 joints down to 17 COCO joints.")