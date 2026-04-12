from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import math
import numpy as np


POSE_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
]

KEY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


@dataclass(slots=True)
class Landmark3D:
    x: float
    y: float
    z: float
    visibility: float = 1.0


def _distance(a: Landmark3D, b: Landmark3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle(a: Landmark3D, b: Landmark3D, c: Landmark3D) -> float:
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float32)
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float32)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.arccos(cosang))


def mp_landmarks_to_struct(landmarks: Sequence) -> list[Landmark3D]:
    return [
        Landmark3D(
            x=float(lm.x),
            y=float(lm.y),
            z=float(getattr(lm, "z", 0.0)),
            visibility=float(getattr(lm, "visibility", 1.0)),
        )
        for lm in landmarks
    ]


def build_feature_vector(
    current: Sequence[Landmark3D],
    previous: Sequence[Landmark3D] | None = None,
) -> np.ndarray:
    if len(current) < 33:
        raise ValueError("Expected 33 landmarks.")

    lhip, rhip = current[23], current[24]
    lsho, rsho = current[11], current[12]
    hip_center = Landmark3D(
        x=(lhip.x + rhip.x) * 0.5,
        y=(lhip.y + rhip.y) * 0.5,
        z=(lhip.z + rhip.z) * 0.5,
        visibility=min(lhip.visibility, rhip.visibility),
    )
    shoulder_width = max(_distance(lsho, rsho), 1e-4)
    torso_height = max(_distance(current[0], hip_center), 1e-4)

    coords: list[float] = []
    vels: list[float] = []
    prev = previous if previous is not None and len(previous) >= 33 else None

    for idx in range(33):
        lm = current[idx]
        coords.extend([
            (lm.x - hip_center.x) / shoulder_width,
            (lm.y - hip_center.y) / shoulder_width,
            (lm.z - hip_center.z) / shoulder_width,
        ])
        if prev is None:
            vels.extend([0.0, 0.0, 0.0])
        else:
            plm = prev[idx]
            vels.extend([
                (lm.x - plm.x) / shoulder_width,
                (lm.y - plm.y) / shoulder_width,
                (lm.z - plm.z) / shoulder_width,
            ])

    engineered = [
        hip_center.y,
        current[0].y,
        current[15].y,
        current[16].y,
        current[27].y,
        current[28].y,
        current[15].x,
        current[16].x,
        current[27].x,
        current[28].x,
        (current[0].y - hip_center.y) / torso_height,
        (current[11].y - current[12].y) / shoulder_width,
        (current[15].y - current[11].y) / shoulder_width,
        (current[16].y - current[12].y) / shoulder_width,
        (current[27].y - current[23].y) / shoulder_width,
        (current[28].y - current[24].y) / shoulder_width,
        _angle(current[11], current[13], current[15]),
        _angle(current[12], current[14], current[16]),
        _angle(current[23], current[25], current[27]),
        _angle(current[24], current[26], current[28]),
        min(lm.visibility for lm in current),
        float(np.mean([lm.visibility for lm in current])),
    ]
    return np.asarray(coords + vels + engineered, dtype=np.float32)


def feature_size() -> int:
    return 33 * 3 + 33 * 3 + 22


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=(0, 1), keepdims=True)
    std = sequence.std(axis=(0, 1), keepdims=True) + 1e-6
    return (sequence - mean) / std
