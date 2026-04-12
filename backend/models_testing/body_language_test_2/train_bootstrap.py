from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG
from src.features import feature_size
from src.io_utils import save_checkpoint
from src.model import TemporalActionNet
from src.synthetic_data import CLASS_NAMES, make_dataset


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    x, y = make_dataset(n_per_class=args.samples_per_class, seq_len=CONFIG.sequence_length, seed=args.seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TemporalActionNet(input_size=feature_size(), num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += xb.size(0)

        model.eval()
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_seen += xb.size(0)
        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        val_acc = val_correct / max(val_seen, 1)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    out_path = Path(args.output)
    save_checkpoint(
        path=out_path,
        model=model.cpu(),
        input_size=feature_size(),
        class_names=CLASS_NAMES,
        sequence_length=CONFIG.sequence_length,
        meta={
            "trained_on": "synthetic_motion_bootstrap",
            "best_val_acc": float(best_acc),
            "epochs": int(args.epochs),
            "samples_per_class": int(args.samples_per_class),
            "seed": int(args.seed),
        },
    )
    print(f"Saved checkpoint to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bootstrap checkpoint on synthetic motion data.")
    parser.add_argument("--output", type=str, default=str(CONFIG.checkpoint_path))
    parser.add_argument("--samples-per-class", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)
