from __future__ import annotations

import argparse
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


def load_npz_dataset(root: Path, sequence_length: int, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for path in sorted(root.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        x = data["x"].astype(np.float32)
        label = str(data["y"])
        if label not in label_to_idx:
            continue
        if x.shape[0] < sequence_length:
            pad = np.repeat(x[-1:], sequence_length - x.shape[0], axis=0)
            x = np.concatenate([x, pad], axis=0)
        elif x.shape[0] > sequence_length:
            center = x.shape[0] // 2
            start = max(0, center - sequence_length // 2)
            x = x[start:start + sequence_length]
        xs.append(x)
        ys.append(label_to_idx[label])
    if not xs:
        raise RuntimeError(f"No .npz sequences found under {root}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model on prepared public skeleton sequences.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/public_sequences"))
    parser.add_argument("--output", type=Path, default=CONFIG.models_root / "public_data_temporal_action_net.pt")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    class_names = CONFIG.class_names
    x, y = load_npz_dataset(args.data_dir, CONFIG.sequence_length, class_names)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=args.batch_size, num_workers=0)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TemporalActionNet(input_size=feature_size(), num_classes=len(class_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += xb.size(0)
        val_acc = correct / max(total, 1)
        print(f"epoch={epoch:02d} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training failed.")
    model.load_state_dict(best_state)
    save_checkpoint(
        args.output,
        model.cpu(),
        input_size=feature_size(),
        class_names=class_names,
        sequence_length=CONFIG.sequence_length,
        meta={
            "trained_on": str(args.data_dir),
            "best_val_acc": float(best_acc),
        },
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
