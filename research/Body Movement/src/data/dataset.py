import torch
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, is_training: bool = False):
        self.X = np.load(data_path)  # Shape: (Samples, 15 Frames, 17 Joints, 2 Coords)
        self.y = np.load(label_path) 
        self.is_training = is_training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 1. Grab the skeleton sequence
        skeleton = self.X[idx].copy() # Use .copy() to avoid modifying the original array in memory
        label = self.y[idx]

        # 2. Geometric Data Augmentation (Only if training)
        if self.is_training:
            skeleton = self._apply_augmentations(skeleton)

        # 3. Format for ST-GCN: (Channels, Frames, Joints) -> (2, 15, 17)
        tensor_data = torch.from_numpy(skeleton).float()
        tensor_data = tensor_data.permute(2, 0, 1) 

        return tensor_data, torch.tensor(label, dtype=torch.long)

    def _apply_augmentations(self, skeleton):
        """Applies SOTA Skeletal Augmentations to prevent overfitting."""
        
        # A. VIEWPOINT ROTATION (+/- 15 degrees)
        if np.random.rand() < 0.5:
            angle = np.radians(np.random.uniform(-15, 15))
            c, s = np.cos(angle), np.sin(angle)
            # 2D Rotation Matrix
            R = np.array(((c, -s), (s, c)))
            # Apply rotation to the X, Y coordinates
            skeleton = np.dot(skeleton, R.T)

        # B. TEMPORAL SCALING (Speed up / Slow down by 20%)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            orig_frames = skeleton.shape[0]
            new_frames = int(orig_frames * scale)
            
            if new_frames != orig_frames:
                # Interpolate frames
                indices = np.linspace(0, orig_frames - 1, new_frames).astype(int)
                scaled = skeleton[indices]
                
                # Crop or Pad back to exactly 15 frames
                if new_frames > orig_frames:
                    start = (new_frames - orig_frames) // 2
                    skeleton = scaled[start:start+orig_frames]
                else:
                    pad_len = orig_frames - new_frames
                    padding = np.repeat(scaled[-1:], pad_len, axis=0) # Pad with the last frame
                    skeleton = np.concatenate([scaled, padding], axis=0)

        # C. JOINT DROPOUT (Simulating occlusion / camera missing a limb)
        if np.random.rand() < 0.3:
            # COCO indices -> Left Arm: 5,7,9 | Right Arm: 6,8,10 | Left Leg: 11,13,15 | Right Leg: 12,14,16
            limbs = [[5,7,9], [6,8,10], [11,13,15], [12,14,16]]
            dropped_limb = limbs[np.random.randint(0, len(limbs))]
            # Set the coordinates of the dropped limb to 0
            skeleton[:, dropped_limb, :] = 0.0

        return skeleton