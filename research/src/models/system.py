import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.loggers import MLFlowLogger

# SOTA Metrics
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall, 
    BinaryF1Score,
    BinaryAveragePrecision,
    BinaryConfusionMatrix
)

from src.models.model import TT2ResidentialVideoMAE

logger = logging.getLogger(__name__)

class TT2SecuritySystem(pl.LightningModule):
    def __init__(
        self, 
        model_name: str = "MCG-NJU/videomae-base",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        threat_weight: float = 3.0, 
        epochs: int = 50
    ):
        super().__init__()
        self.save_hyperparameters() 
        
        logger.info(f"Initializing TT2 Security System with backbone: {model_name}")
        self.model = TT2ResidentialVideoMAE(model_name=self.hparams.model_name, num_classes=2)
        
        # [CRITICAL FIX: Device Synchronization]
        # register_buffer ensures this tensor moves to the RTX 4090 automatically
        self.register_buffer(
            "class_weights", 
            torch.tensor([1.0, self.hparams.threat_weight], dtype=torch.float32)
        )

        # =====================================================================
        # [SOTA TELEMETRY] The Audit Suite
        # =====================================================================
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        
        # [L5 Metrics] Threshold-independent evaluation
        self.val_pr_auc = BinaryAveragePrecision()
        self.val_conf_matrix = BinaryConfusionMatrix()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    def training_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        
        # [CRITICAL FIX] Apply device-aware weights dynamically
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        logits = self(videos)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        probs = torch.softmax(logits, dim=1)[:, 1] 
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.val_pr_auc(probs, labels)
        self.val_conf_matrix(preds, labels)
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, prog_bar=True, sync_dist=True)
        self.log("val/recall", self.val_recall, prog_bar=True, sync_dist=True)
        self.log("val/precision", self.val_precision, prog_bar=False, sync_dist=True)
        self.log("val/f1", self.val_f1, prog_bar=False, sync_dist=True)
        self.log("val/pr_auc", self.val_pr_auc, prog_bar=False, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        """Generates and logs a Confusion Matrix image."""
        cm = self.val_conf_matrix.compute().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                    xticklabels=["Decoy", "Threat"], 
                    yticklabels=["Decoy", "Threat"])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - Epoch {self.current_epoch}')
        
        # [CRITICAL FIX: SOLID Principle / Defensive Programming]
        # Ensure we only call MLflow-specific methods if the logger is actually MLflow
        if self.logger and isinstance(self.logger, MLFlowLogger):
            mlflow_logger = self.logger.experiment
            run_id = self.logger.run_id
            mlflow_logger.log_figure(run_id, fig, f"confusion_matrices/epoch_{self.current_epoch}.png")
            
        plt.close(fig) 
        self.val_conf_matrix.reset() 

    def configure_optimizers(self):
        # [CRITICAL FIX: Memory Safety] Cast generator to list
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        optimizer = AdamW(trainable_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs) 
        return [optimizer], [scheduler]