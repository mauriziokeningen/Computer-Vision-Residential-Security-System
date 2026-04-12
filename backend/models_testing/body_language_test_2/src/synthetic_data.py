from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import random
import numpy as np

from .features import Landmark3D, build_feature_vector, feature_size


CLASS_NAMES = ["neutral", "punch", "kick", "push", "wave", "fall"]


def _base_pose() -> list[Landmark3D]:
    pts = [Landmark3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    # Head / face
    pts[0] = Landmark3D(0.50, 0.18, -0.03, 1.0)
    for i in [1,2,3,4,5,6,7,8,9,10]:
        pts[i] = Landmark3D(0.50, 0.19 + 0.003 * (i % 2), -0.02, 0.9)
    # Upper body
    pts[11] = Landmark3D(0.43, 0.30, 0.00, 1.0)
    pts[12] = Landmark3D(0.57, 0.30, 0.00, 1.0)
    pts[13] = Landmark3D(0.40, 0.42, 0.02, 1.0)
    pts[14] = Landmark3D(0.60, 0.42, 0.02, 1.0)
    pts[15] = Landmark3D(0.39, 0.56, 0.05, 1.0)
    pts[16] = Landmark3D(0.61, 0.56, 0.05, 1.0)
    for i in [17,19,21]:
        pts[i] = Landmark3D(0.37, 0.58 + 0.01 * (i - 17), 0.06, 0.9)
    for i in [18,20,22]:
        pts[i] = Landmark3D(0.63, 0.58 + 0.01 * (i - 18), 0.06, 0.9)
    # Lower body
    pts[23] = Landmark3D(0.46, 0.54, 0.00, 1.0)
    pts[24] = Landmark3D(0.54, 0.54, 0.00, 1.0)
    pts[25] = Landmark3D(0.46, 0.72, 0.02, 1.0)
    pts[26] = Landmark3D(0.54, 0.72, 0.02, 1.0)
    pts[27] = Landmark3D(0.46, 0.92, 0.03, 1.0)
    pts[28] = Landmark3D(0.54, 0.92, 0.03, 1.0)
    pts[29] = Landmark3D(0.45, 0.97, 0.04, 0.8)
    pts[30] = Landmark3D(0.55, 0.97, 0.04, 0.8)
    pts[31] = Landmark3D(0.44, 0.98, 0.04, 0.8)
    pts[32] = Landmark3D(0.56, 0.98, 0.04, 0.8)
    return pts


def _copy_pose(pose: Sequence[Landmark3D]) -> list[Landmark3D]:
    return [Landmark3D(p.x, p.y, p.z, p.visibility) for p in pose]


def _jitter(pose: list[Landmark3D], sigma: float = 0.005) -> None:
    for p in pose:
        p.x += random.gauss(0.0, sigma)
        p.y += random.gauss(0.0, sigma)
        p.z += random.gauss(0.0, sigma * 0.6)


def _arm_chain(pose: list[Landmark3D], side: str, shoulder_shift=(0.0,0.0), elbow_shift=(0.0,0.0), wrist_shift=(0.0,0.0), z_shift=0.0) -> None:
    if side == "left":
        s, e, w = 11, 13, 15
    else:
        s, e, w = 12, 14, 16
    pose[s].x += shoulder_shift[0]
    pose[s].y += shoulder_shift[1]
    pose[e].x += elbow_shift[0]
    pose[e].y += elbow_shift[1]
    pose[w].x += wrist_shift[0]
    pose[w].y += wrist_shift[1]
    pose[w].z += z_shift


def _leg_chain(pose: list[Landmark3D], side: str, hip_shift=(0.0,0.0), knee_shift=(0.0,0.0), ankle_shift=(0.0,0.0), z_shift=0.0) -> None:
    if side == "left":
        h, k, a = 23, 25, 27
    else:
        h, k, a = 24, 26, 28
    pose[h].x += hip_shift[0]
    pose[h].y += hip_shift[1]
    pose[k].x += knee_shift[0]
    pose[k].y += knee_shift[1]
    pose[a].x += ankle_shift[0]
    pose[a].y += ankle_shift[1]
    pose[a].z += z_shift


def _make_sequence(label: str, length: int) -> np.ndarray:
    prev = None
    feats: list[np.ndarray] = []
    dominant_side = random.choice(["left", "right"])
    direction = random.choice([-1.0, 1.0])

    for t in range(length):
        phase = t / max(length - 1, 1)
        pose = _copy_pose(_base_pose())
        _jitter(pose, sigma=0.004 + 0.002 * random.random())

        if label == "neutral":
            breathe = 0.01 * math.sin(2 * math.pi * phase * (1.0 + random.random()))
            pose[11].y += breathe
            pose[12].y += breathe
            pose[23].y += breathe * 0.5
            pose[24].y += breathe * 0.5

        elif label == "wave":
            side = dominant_side
            arc = math.sin(2 * math.pi * (1.5 + random.random()) * phase)
            raise_amt = 0.18 + 0.08 * phase
            _arm_chain(
                pose,
                side,
                shoulder_shift=(0.0, -0.03),
                elbow_shift=(0.03 * direction, -0.10),
                wrist_shift=(0.10 * arc * direction, -raise_amt),
                z_shift=-0.02,
            )

        elif label == "punch":
            side = dominant_side
            thrust = 1.0 / (1.0 + math.exp(-20 * (phase - 0.45)))
            retract = 1.0 - 0.35 * max(0.0, phase - 0.75) / 0.25
            thrust *= retract
            _arm_chain(
                pose,
                side,
                elbow_shift=(0.05 * direction, -0.03),
                wrist_shift=(0.18 * direction + 0.14 * thrust * direction, -0.02 - 0.05 * thrust),
                z_shift=-0.18 * thrust,
            )
            pose[0].x -= 0.02 * direction * thrust
            pose[11].x -= 0.015 * direction * thrust
            pose[12].x -= 0.015 * direction * thrust

        elif label == "kick":
            side = dominant_side
            lift = math.sin(math.pi * min(1.0, phase * 1.2))
            extend = 1.0 / (1.0 + math.exp(-18 * (phase - 0.45)))
            _leg_chain(
                pose,
                side,
                knee_shift=(0.04 * direction, -0.14 * lift),
                ankle_shift=(0.12 * direction + 0.10 * extend * direction, -0.28 * lift),
                z_shift=-0.10 * extend,
            )
            pose[0].x -= 0.03 * direction * lift
            pose[23].x -= 0.015 * direction * lift
            pose[24].x -= 0.015 * direction * lift

        elif label == "push":
            drive = 1.0 / (1.0 + math.exp(-18 * (phase - 0.40)))
            _arm_chain(pose, "left", elbow_shift=(0.02, -0.03), wrist_shift=(0.08, -0.12), z_shift=-0.15 * drive)
            _arm_chain(pose, "right", elbow_shift=(-0.02, -0.03), wrist_shift=(-0.08, -0.12), z_shift=-0.15 * drive)
            pose[11].x += 0.01
            pose[12].x -= 0.01
            pose[0].y += 0.01 * drive
            pose[23].x += 0.01
            pose[24].x -= 0.01

        elif label == "fall":
            drop = 1.0 / (1.0 + math.exp(-16 * (phase - 0.55)))
            tilt = 0.35 * drop * direction
            for idx in range(33):
                pose[idx].y += 0.38 * drop
                pose[idx].x += tilt * (pose[idx].y - 0.55)
            pose[0].y += 0.12 * drop
            pose[15].y += 0.08 * drop
            pose[16].y += 0.08 * drop
            pose[27].y -= 0.03 * drop
            pose[28].y -= 0.03 * drop
            pose[23].visibility = 0.9
            pose[24].visibility = 0.9
        else:
            raise ValueError(f"Unknown label: {label}")

        current_feat = build_feature_vector(pose, prev)
        feats.append(current_feat)
        prev = pose

    return np.stack(feats).astype(np.float32)


def make_dataset(n_per_class: int, seq_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for label_idx, label in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            xs.append(_make_sequence(label, seq_len))
            ys.append(label_idx)
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return x, y
