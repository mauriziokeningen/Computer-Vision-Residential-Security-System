import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. ADD THE NEW METRICS HERE
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from .model import TT2SkeletonClassifier

class TT2SkeletonSystem(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TT2SkeletonClassifier(num_classes=2)
        
        # 2. INITIALIZE METRICS FOR FAANG-LEVEL MONITORING
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision() # NEW
        self.val_recall = BinaryRecall()       # NEW

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        skeletons, labels = batch
        logits = self(skeletons)
        loss = F.cross_entropy(logits, labels)
        
        self.train_acc(torch.argmax(logits, dim=1), labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        skeletons, labels = batch
        logits = self(skeletons)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        # 3. UPDATE ALL METRICS
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        # 4. LOG ALL METRICS TO MLFLOW
        # Setting on_epoch=True ensures we get a smooth curve per epoch
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True) 
        self.log("val/recall", self.val_recall, on_epoch=True)       
        
        return loss

    def configure_optimizers(self):
        # AdamW and Cosine Decay are SOTA for fast convergence
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]