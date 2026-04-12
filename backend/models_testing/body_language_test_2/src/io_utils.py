from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .model import TemporalActionNet


def save_checkpoint(
    path: Path,
    model: TemporalActionNet,
    input_size: int,
    class_names: list[str],
    sequence_length: int,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "input_size": input_size,
        "class_names": class_names,
        "sequence_length": sequence_length,
        "meta": meta,
    }
    torch.save(bundle, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[TemporalActionNet, dict[str, Any]]:
    bundle = torch.load(path, map_location=device)
    model = TemporalActionNet(input_size=int(bundle["input_size"]), num_classes=len(bundle["class_names"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle
