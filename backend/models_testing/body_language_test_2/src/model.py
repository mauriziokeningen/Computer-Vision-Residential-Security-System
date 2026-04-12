from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


class TemporalActionNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return self.head(x)


@dataclass(slots=True)
class CheckpointBundle:
    state_dict: Dict
    input_size: int
    class_names: list[str]
    sequence_length: int
    meta: Dict
