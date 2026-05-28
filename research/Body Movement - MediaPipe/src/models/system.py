import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from .model import TT2SkeletonClassifier

class TT2SkeletonSystem(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TT2SkeletonClassifier(num_classes=2)
        
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision() 
        self.val_recall = BinaryRecall()       

        # OPTIMIZATION: Register the class weights as a buffer so it is only allocated 
        # in GPU memory once, rather than every batch.
        self.register_buffer("class_weights", torch.tensor([1.0, 3.0]))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        skeletons, labels = batch
        logits = self(skeletons)
        
        # Apply the pre-allocated weights
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        self.train_acc(torch.argmax(logits, dim=1), labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        skeletons, labels = batch
        logits = self(skeletons)
        
        # Apply the pre-allocated weights
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True) 
        self.log("val/recall", self.val_recall, on_epoch=True)       
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]