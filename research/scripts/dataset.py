import os
import logging
from pathlib import Path
from typing import Tuple, List, Set

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, Normalize, InterpolationMode
from decord import VideoReader, cpu

# =============================================================================
# [INFRASTRUCTURE] Enterprise Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TT2_Ingestion_Engine")

class TT2SecurityDataset(Dataset):
    """
    [ARCHITECTURE] 
    High-Performance PyTorch Dataset for the TT2 Security System.
    Translates NTU RGB+D .avi files into 5D Spatio-Temporal Tensors.
    """
    def __init__(
        self, 
        data_dir: str, 
        num_frames: int = 16, 
        target_size: int = 224
    ) -> None:
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.target_size = target_size
        
        # [SECURITY LOGIC] Threat = 1, Hard Negative = 0
        # A050: Punch/Slap, A051: Kicking, A052: Pushing
        self.threat_classes: Set[str] = {"A050", "A051", "A052"}
        
        # [I/O] Glob all video assets. In production, this would hit an S3 bucket or DB.
        self.video_paths: List[Path] = list(self.data_dir.glob("*.avi"))
        if not self.video_paths:
            logger.error(f"FATAL: No .avi files found in {self.data_dir}. Vault is empty.")
            raise FileNotFoundError(f"Vault empty at {self.data_dir}")

        # [SOTA] Pre-allocate Transformer Normalization Math (ImageNet Statistics)
        # VideoMAE v2 backbone requires these exact mean/std values.
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # [SOTA] Deterministic resizing for Transformer Patch Embeddings
        self.resize = Resize(
            (self.target_size, self.target_size), 
            interpolation=InterpolationMode.BILINEAR, 
            antialias=True
        )

        logger.info(f"Engine Online. Mapping {len(self.video_paths)} assets to Spacetime Tensors.")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = str(self.video_paths[idx])
        filename = os.path.basename(video_path)

        # ---------------------------------------------------------
        # 1. LABEL EXTRACTION
        # ---------------------------------------------------------
        action_idx = filename.find('A')
        action_class = filename[action_idx:action_idx+4]
        label = 1 if action_class in self.threat_classes else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        # ---------------------------------------------------------
        # 2. FAULT-TOLERANT DECODING
        # ---------------------------------------------------------
        try:
            # [PERFORMANCE] Decord uses a C++ backend, bypassing the Python GIL.
            # ctx=cpu(0) keeps extraction off the GPU until the tensor is fully built.
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames < self.num_frames:
                raise ValueError(f"Video too short: {total_frames} frames.")

            # [TEMPORAL SAMPLING] Extract evenly spaced frames to capture full action physics.
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames_np = vr.get_batch(indices).asnumpy() # Shape: (T, H, W, C)

        except Exception as e:
            # [FAULT TOLERANCE] Never crash the dataloader on a bad file. 
            # Log the corruption and return a dummy tensor so the training loop survives.
            logger.warning(f"Corrupted asset detected: {filename}. Yielding zero-tensor. Error: {str(e)}")
            dummy_frames = torch.zeros((3, self.num_frames, self.target_size, self.target_size))
            return dummy_frames, label_tensor

        # ---------------------------------------------------------
        # 3. HIGH-DIMENSIONAL TENSOR TRANSFORMATION
        # ---------------------------------------------------------
        # Convert to Tensor and permute to (T, C, H, W) for Torchvision transforms
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

        # Apply deterministic spatial sizing (T, C, 224, 224)
        frames_tensor = self.resize(frames_tensor)
        
        # Apply standardization math to prevent exploding gradients
        frames_tensor = self.normalize(frames_tensor)

        # Final permutation to (C, T, H, W) -> The exact shape VideoMAE expects
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)

        return frames_tensor, label_tensor


# =============================================================================
# [EXECUTION] Tracer Bullet Payload
# =============================================================================
if __name__ == "__main__":
    # Ensure we point to the exact DVC-tracked vault
    vault_path = Path.cwd() / "data" / "raw" / "body" / "NTU_RGB_D"
    
    try:
        logger.info("Initializing TT2 Ingestion Engine...")
        dataset = TT2SecurityDataset(data_dir=str(vault_path))
        
        # [PERFORMANCE] num_workers=2 parallelizes disk I/O, pin_memory=True speeds up GPU transfer
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        
        logger.info("Firing hardware-accelerated Tracer Bullet...")
        video_batch, label_batch = next(iter(dataloader))
        
        logger.info("=" * 60)
        logger.info("PYTORCH INGESTION: VERIFIED SOTA")
        logger.info(f"Target Architecture:  VideoMAE v2")
        logger.info(f"Output Tensor Shape:  {video_batch.shape} [C, T, H, W]")
        logger.info(f"Tensor Data Range:    Min: {video_batch.min():.2f}, Max: {video_batch.max():.2f}")
        logger.info(f"Ground Truth Labels:  {label_batch.tolist()} (1=Threat, 0=Decoy)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"System Failure during ingestion test: {str(e)}")