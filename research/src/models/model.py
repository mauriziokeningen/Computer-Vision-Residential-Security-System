import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

# =============================================================================
# [INFRASTRUCTURE] Enterprise Logging
# =============================================================================
logger = logging.getLogger(__name__)

class TT2ResidentialVideoMAE(nn.Module):
    """
    [ARCHITECTURE]
    SOTA Wrapper for VideoMAE v2, customized for Binary Threat Detection.
    Encapsulates Hugging Face internals to provide a clean PyTorch nn.Module API.
    """
    def __init__(
        self, 
        model_name: str = "MCG-NJU/videomae-base", 
        num_classes: int = 2,
        freeze_backbone: bool = False
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        logger.info(f"Mounting backbone: {self.model_name} | Target Classes: {self.num_classes}")
        
        # 1. Load configuration and override the classification head
        config = VideoMAEConfig.from_pretrained(
            self.model_name, 
            num_labels=self.num_classes
        )
        
        # 2. Initialize model with the overridden head (ignores the 400-class mismatch)
        self.backbone = VideoMAEForVideoClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # 3. Optional: Freeze backbone for Linear Probing / Transfer Learning
        if freeze_backbone:
            logger.info("Freezing VideoMAE backbone. Only the classification head will train.")
            for param in self.backbone.videomae.parameters():
                param.requires_grad = False

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes the forward pass, handling defensive tensor reshaping.
        
        Expected Input from Dataset:  [Batch, Channels, Frames, Height, Width]
        Required by Hugging Face:     [Batch, Frames, Channels, Height, Width]
        """
        # [DEFENSIVE ENGINEERING] Auto-permute if the tensor matches the dataset output
        if pixel_values.shape[1] == 3 and pixel_values.shape[2] > 3:
            # Shift from (B, C, T, H, W) to (B, T, C, H, W)
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
            
        # Execute forward pass
        outputs = self.backbone(pixel_values=pixel_values, labels=labels)
        
        # In a custom nn.Module, it's best practice to return raw logits 
        # and handle the loss function in the training loop/Lightning module.
        return outputs.logits

    @property
    def device(self) -> torch.device:
        """Helper to dynamically fetch the device the model is currently on."""
        return next(self.parameters()).device