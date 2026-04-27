import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

# Import our architectural components
# Adjust the import paths if your directory structure differs slightly
from src.data.dataset import TT2SecurityDataset
from src.models.system import TT2SecuritySystem

# =============================================================================
# [INFRASTRUCTURE] Enterprise Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("TT2_Master_Controller")

def main():
    # 1. Hardware Verification
    if not torch.cuda.is_available():
        logger.error("FATAL: CUDA not detected. The RTX 4090 is offline.")
        return
    logger.info(f"Target Hardware Acquired: {torch.cuda.get_device_name(0)}")

    # 2. Vault Path Definition
    # Update this path to exactly where your NTU_RGB_D .avi files live
    vault_path = Path.cwd() / "data" / "raw" / "body" / "NTU_RGB_D"
    if not vault_path.exists():
        logger.error(f"FATAL: Data vault not found at {vault_path}.")
        return

    # 3. Spin Up the Data Engine
    logger.info("Igniting High-Performance Data Engine...")
    full_dataset = TT2SecurityDataset(data_dir=str(vault_path))
    
    # Standard 80/20 Train-Validation Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Dataloaders: batch_size=4 is VRAM safe, num_workers=4 maximizes SSD bandwidth
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize the LoRA-optimized System
    logger.info("Booting TT2 Nervous System...")
    model = TT2SecuritySystem(learning_rate=1e-4)

    # 5. Initialize SOTA Telemetry (MLflow)
    logger.info("Connecting to MLflow Tracking Server...")
    mlflow_logger = MLFlowLogger(
        experiment_name="TT2_Pilot_Run",
        # [SOTA FIX] Upgraded from local files to a relational database
        tracking_uri="sqlite:///mlflow.db"
    )

    # 6. SOTA Automated Guardrails
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="tt2-pilot-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode="min"
    )

    # =========================================================================
    # 7. THE PILOT RUN CONFIGURATION
    # =========================================================================
    logger.info("Initializing PyTorch Lightning Trainer (PILOT MODE)...")
    trainer = pl.Trainer(
        max_epochs=2,                 # Restrict to 2 full passes
        accelerator="gpu",
        devices=1,
        precision="16-mixed",         # [SOTA] Double VRAM capacity via Mixed Precision
        limit_train_batches=0.05,     # <--- 5% PILOT CONSTRAINT
        limit_val_batches=0.05,       # <--- 5% PILOT CONSTRAINT
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1           # Force logging often since we only have 5% of data
    )

    # 8. Engage!
    logger.info("=" * 60)
    logger.info("INITIATING 5% PILOT RUN")
    logger.info("=" * 60)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    logger.info("Pilot Run Complete. Run 'mlflow ui' in your terminal to view telemetry.")

if __name__ == "__main__":
    # Workaround for multiprocessing on some Linux environments
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    main()