import os
import logging
from pathlib import Path
import torch
import git
import mlflow
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl

# [SOTA TELEMETRY] Import the advanced hardware and learning rate monitors
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    DeviceStatsMonitor,
    LearningRateMonitor
)

from lightning.pytorch.loggers import MLFlowLogger

import subprocess

# Import our architectural components
# Adjust the import paths if your directory structure differs slightly
from src.data.dataset import TT2SecurityDataset
from src.models.system import TT2SecuritySystem

# =============================================================================
# [INFRASTRUCTURE] Enterprise Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("TT2_Master_Controller")

def get_git_hash():
    """[SOTA TELEMETRY] Data Provenance: Fetches the current commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "unknown_hash"

def get_dvc_status():
    """[SOTA TELEMETRY] Data Provenance: Audits DVC to ensure data isn't secretly mutating."""
    try:
        # Runs 'dvc status -q'. If data is modified but uncommitted, it returns text.
        result = subprocess.run(["dvc", "status", "-q"], capture_output=True, text=True)
        if result.returncode == 0 and not result.stdout.strip():
            return "clean"
        return "DIRTY_UNCOMMITTED_DATA"
    except FileNotFoundError:
        return "dvc_cli_not_found"

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

    # 3. Spin Up the Data Engine (Cross-Subject Split)
    logger.info("Parsing data vault for Cross-Subject (X-Sub) Split...")
    all_files = list(vault_path.glob("*.avi"))
    
    # Official NTU RGB+D Cross-Subject Training IDs
    train_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
    
    train_files = []
    val_files = []
    
    for file_path in all_files:
        filename = file_path.name
        # Extract the 3-digit subject ID following 'P' (e.g., S001C001P003R001A050 -> 003)
        try:
            subject_idx = filename.find('P') + 1
            subject_id = int(filename[subject_idx:subject_idx+3])
            
            if subject_id in train_subjects:
                train_files.append(file_path)
            else:
                val_files.append(file_path)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse subject ID from {filename}. Skipping.")
            
    logger.info(f"X-Sub Split Complete: {len(train_files)} Train assets, {len(val_files)} Validation assets.")

    # Instantiate datasets using explicit file lists instead of globbing the whole directory
    train_dataset = TT2SecurityDataset(video_paths=train_files)
    val_dataset = TT2SecurityDataset(video_paths=val_files)
    
    # Dataloaders: batch_size=4 is VRAM safe, num_workers=4 maximizes SSD bandwidth
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize the LoRA-optimized System
    logger.info("Booting TT2 Nervous System...")
    model = TT2SecuritySystem(learning_rate=1e-4)

    # 5. Initialize SOTA Telemetry (MLflow)
    logger.info("Connecting to MLflow Tracking Server...")
    
    # [SOTA TELEMETRY] Hardware Monitoring (Power, vRAM, Temperature)
    mlflow.enable_system_metrics_logging()

    mlflow_logger = MLFlowLogger(
        experiment_name="TT2_Production_Run",
        # [SOTA FIX] Upgraded from local files to a relational database
        tracking_uri="sqlite:///mlflow.db"
    )

    # [SOTA TELEMETRY] Log Data Provenance (Git Hash) so we can reproduce this exact run
    mlflow_logger.log_hyperparams({"git_commit_hash": get_git_hash()})

    # 6. SOTA Automated Guardrails
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="tt2-prod-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=3,
        mode="min"
    )

    # [SOTA TELEMETRY] Initialize Pulse Monitors
    device_monitor = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # =========================================================================
    # 7. THE PILOT RUN CONFIGURATION
    # =========================================================================
    logger.info("INITIATING FULL PRODUCTION RUN")
    trainer = pl.Trainer(
        max_epochs=50,                 # <--- INCREASED TO 50
        accelerator="gpu",
        devices=1,
        precision="16-mixed",         # [SOTA] Double VRAM capacity via Mixed Precision
        logger=mlflow_logger,
        callbacks=[
            checkpoint_callback, 
            early_stop_callback,
            RichProgressBar(),
            device_monitor, 
            lr_monitor
        ],
        log_every_n_steps=10          # Relaxed logging frequency for speed
    )

    # 8. Engage!
    logger.info("=" * 60)
    logger.info("INITIATING FULL PRODUCTION RUN")
    logger.info("=" * 60)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    logger.info("Production Run Complete. Run 'mlflow ui' in your terminal to view telemetry.")

if __name__ == "__main__":
    # Workaround for multiprocessing on some Linux environments
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    main()