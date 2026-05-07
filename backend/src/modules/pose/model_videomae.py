import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig
from peft import LoraConfig, get_peft_model

# =============================================================================
# [INFRASTRUCTURE] Enterprise Logging
# =============================================================================
logger = logging.getLogger(__name__)

class TT2ResidentialVideoMAE(nn.Module):
    """
    [ARCHITECTURE]
    SOTA Wrapper for VideoMAE (v1) Base, customized for Binary Threat Detection.
    Equipped with LoRA (Parameter-Efficient Fine-Tuning) to guarantee
    RTX 4090 VRAM compliance by keeping trainable parameters under 5%.
    """
    def __init__(
        self, 
        model_name: str = "MCG-NJU/videomae-base", 
        num_classes: int = 2,
        r: int = 16,
        alpha: int = 32
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
        
        # 2. Initialize base model (ignores the 400-class mismatch)
        self.base_model = VideoMAEForVideoClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # 3. Configure LoRA Matrix Injections
        # Target the 'query' and 'value' projections in the attention mechanism
        peft_config = LoraConfig( 
            r=r, 
            lora_alpha=alpha, 
            target_modules=["query", "value"], 
            lora_dropout=0.1,
            modules_to_save=["classifier"] # CRITICAL: Keep our new binary head trainable
        )

        # 4. Wrap the architecture with PEFT
        logger.info(f"Injecting LoRA Adapters (Rank={r}, Alpha={alpha})...")
        self.model = get_peft_model(self.base_model, peft_config)
        
        # 5. Hardware Audit
        self._print_trainable_parameters()

    def _print_trainable_parameters(self) -> None:
        """Calculates and logs the percentage of active weights."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        efficiency = 100 * trainable_params / all_params
        
        logger.info("=" * 60)
        logger.info("PEFT HARDWARE AUDIT")
        logger.info(f"Trainable Parameters: {trainable_params:,d}")
        logger.info(f"Total Parameters:     {all_params:,d}")
        logger.info(f"VRAM Efficiency:      {efficiency:.2f}% active")
        logger.info("=" * 60)

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
            
        # Execute forward pass through the PEFT-wrapped model
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        return outputs.logits

    @property
    def device(self) -> torch.device:
        """Helper to dynamically fetch the device the model is currently on."""
        return next(self.parameters()).device