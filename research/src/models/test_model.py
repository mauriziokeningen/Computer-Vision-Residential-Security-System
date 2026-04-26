import os
import logging
import torch
import pytest

from model import TT2ResidentialVideoMAE

# Set logging for the test output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("TracerBullet")

def test_videomae_initialization_and_forward_pass():
    """
    [TRACER BULLET]
    Validates model initialization, head replacement, tensor permutation, 
    and output logit dimensions without triggering CUDA OOM.
    """
    # 1. Initialization
    logger.info("Initializing TT2ResidentialVideoMAE...")
    model = TT2ResidentialVideoMAE(num_classes=2)
    
    # 2. Hardware Allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() # Set to evaluation mode for testing
    
    # 3. Synthetic Payload (Mimicking TT2SecurityDataset output)
    # Shape: [Batch=4, Channels=3, Frames=16, Height=224, Width=224]
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 16, 224, 224).to(device)
    
    # 4. Execution
    logger.info(f"Firing {dummy_input.shape} tensor into architecture...")
    with torch.no_grad(): # Prevent graph caching to save VRAM during test
        logits = model(pixel_values=dummy_input)
        
    # 5. Assertions (The core of SOTA testing)
    assert logits is not None, "FATAL: Model returned None."
    assert logits.shape == (batch_size, 2), f"FATAL: Expected shape (4, 2), got {logits.shape}"
    
    # 6. Memory Audit (Check how hard we hit the RTX 4090)
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        logger.info(f"VRAM Allocated: {vram_allocated:.2f} MB")
        
    logger.info("✅ Tracer Bullet Success: Architecture is Sound.")

if __name__ == "__main__":
    # Allows running the test directly without pytest for a quick sanity check
    test_videomae_initialization_and_forward_pass()