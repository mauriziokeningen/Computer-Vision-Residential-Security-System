from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(slots=True)
class AppConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = project_root / "data"
    raw_root: Path = data_root / "raw"
    processed_root: Path = data_root / "processed"
    models_root: Path = project_root / "models"
    exports_root: Path = project_root / "exports"

    class_names: List[str] = field(
        default_factory=lambda: [
            "neutral",
            "punch",
            "kick",
            "push",
            "wave",
            "fall",
        ]
    )
    sequence_length: int = 40
    min_confidence: float = 0.60
    smoothing_alpha: float = 0.35
    vote_window: int = 7
    webcam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    checkpoint_path: Path = models_root / "bootstrap_temporal_action_net.pt"


CONFIG = AppConfig()
