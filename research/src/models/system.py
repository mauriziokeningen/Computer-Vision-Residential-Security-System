import logging
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import our SOTA, LoRA-optimized architecture
from src.models.model import TT2ResidentialVideoMAE

# =============================================================================
# [INFRASTRUCTURE] Enterprise Logging
# =============================================================================
logger = logging.getLogger(__name__)

class TT2SecuritySystem(pl.LightningModule):
    """
    [ARCHITECTURE] PyTorch Lightning Engine for Residential Threat Detection.
    Encapsulates the model, weighted loss functions, and SOTA optimization strategies.
    Designed specifically to handle the TT2 Binary Imbalance problem.
    """
    def __init__(
        self, 
        model_name: str = "MCG-NJU/videomae-base",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        threat_weight: float = 3.0, # Assigns 3x more mathematical importance to threats
        epochs: int = 50
    ):
        super().__init__()
        
        # [TELEMETRY] Saves hyperparameters to the checkpoint and automatically streams to MLflow
        self.save_hyperparameters() 
        
        # 1. Mount the LoRA-optimized Brain
        logger.info(f"Initializing TT2 Security System with backbone: {model_name}")
        self.model = TT2ResidentialVideoMAE(
            model_name=self.hparams.model_name, 
            num_classes=2
        )
        
        # 2. Define the Loss Function (Addressing the Class Imbalance)
        # Security is imbalanced. Class 0: Decoy (Weight 1.0), Class 1: Threat (Weight 3.0)
        # This prevents the network from achieving high accuracy by just guessing "Decoy" every time.
        class_weights = torch.tensor([1.0, self.hparams.threat_weight])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """The forward pass is cleanly delegated to our LoRA wrapper."""
        return self.model(pixel_values)

    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy for SOTA telemetry
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # self.log automatically hooks into our MLFlow logger
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # sync_dist=True guarantees accurate logging across multi-GPU environments
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        """
        [SOTA] AdamW + Cosine Annealing is the industry standard for Vision Transformers.
        """
        # [CRITICAL HARDWARE OPTIMIZATION]
        # We explicitly filter out the frozen backbone weights. If we pass the whole model
        # to AdamW, it allocates memory for all 86M parameters, defeating the purpose of LoRA.
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        optimizer = AdamW(
            trainable_params, 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        # Smoothly decrease the learning rate to settle into local minima
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs) 
        
        return [optimizer], [scheduler]