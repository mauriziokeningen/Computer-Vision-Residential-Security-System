import torch
import numpy as np
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, data_path: str, label_path: str, is_training: bool = False):
        self.X = np.load(data_path)  
        self.y = np.load(label_path)
        self.is_training = is_training

    def __len__(self):
        return len(self.X)

    def _normalize_and_center(self, skeleton):
        norm_skel = np.zeros_like(skeleton)
        
        for f in range(skeleton.shape[0]):
            valid_mask = (skeleton[f, :, 3] > 0.1) # Use Visibility channel to check if real
            if not np.any(valid_mask): continue
                
            # UPGRADE: MediaPipe Hips are 23 and 24
            left_hip, right_hip = skeleton[f, 23], skeleton[f, 24]
            if left_hip[3] > 0.1 and right_hip[3] > 0.1:
                pelvis = (left_hip[:3] + right_hip[:3]) / 2.0 # X, Y, Z
            else:
                pelvis = np.mean(skeleton[f][valid_mask][:, :3], axis=0)
            
            # Center X, Y, Z. Leave Visibility (index 3) untouched.
            norm_skel[f, valid_mask, :3] = skeleton[f, valid_mask, :3] - pelvis
            norm_skel[f, valid_mask, 3] = skeleton[f, valid_mask, 3] 
            
        # Scale only X, Y, Z coordinates
        max_spread = np.max(np.abs(norm_skel[:, :, :3]))
        if max_spread > 1e-5:
            norm_skel[:, :, :3] = norm_skel[:, :, :3] / max_spread
            
        return norm_skel

    def __getitem__(self, idx):
        skeleton = self.X[idx].copy()
        label = self.y[idx]

        # FIX: The Antidote. Purge all NaNs and Infs from the NTU dataset
        skeleton = np.nan_to_num(skeleton, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to (0,0) before augmenting
        skeleton = self._normalize_and_center(skeleton)

        if self.is_training:
            skeleton = self._apply_augmentations(skeleton)

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
            
            # FIX: Apply rotation ONLY to the X, Y coordinates (channels 0 and 1).
            # Leave the Confidence score (channel 2) completely alone.
            skeleton[:, :, :2] = np.dot(skeleton[:, :, :2], R.T)

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