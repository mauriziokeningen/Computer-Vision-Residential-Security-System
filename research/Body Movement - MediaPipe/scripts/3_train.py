import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import MLFlowLogger # <--- NEW IMPORT
from lightning.pytorch.callbacks import EarlyStopping # <--- 1. NEW IMPORT

from src.data.dataset import SkeletonDataset
from src.models.system import TT2SkeletonSystem

from scripts.evaluate import evaluate
 

def main():
    # 1. Load the mapped 17-joint data
    train_dataset = SkeletonDataset(
        data_path="data/processed/X_train.npy", 
        label_path="data/processed/y_train.npy", 
        is_training=True
    )
    val_dataset = SkeletonDataset(
        data_path="data/processed/X_val.npy", 
        label_path="data/processed/y_val.npy", 
        is_training=False
    )

    # 2. Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 3. Initialize ST-GCN System
    model = TT2SkeletonSystem(learning_rate=0.001)

    # 4. Initialize MLflow Logger
    # This creates a local tracking database in an "mlruns" folder
    mlf_logger = MLFlowLogger(
        experiment_name="TT2_Skeletal_Tracking",
        tracking_uri="file:./mlruns" 
    )

    # DEFINE THE EARLY STOPPING CALLBACK
    early_stop_callback = EarlyStopping(
        monitor="val/loss", # Watch the validation loss
        patience=10,        # Stop if it doesn't improve for 10 epochs
        mode="min"
    )

    # 5. Train!
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="gpu",
        devices=1,
        logger=mlf_logger, # <--- TELL LIGHTNING TO USE MLFLOW
        callbacks=[early_stop_callback], # <--- ADD EARLY STOPPING CALLBACK
        gradient_clip_val=1.0 # <--- GRADIENT CLIPPING TO PREVENT EXPLOSIONS   
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    evaluate()

if __name__ == "__main__":
    main()