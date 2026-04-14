from __future__ import annotations

import math
import random
import numpy as np

from .features import Landmark3D, build_feature_vector

CLASS_NAMES = ["neutral", "punch", "kick", "push", "wave", "fall"]


def _base_pose() -> list[Landmark3D]:
    pts = [Landmark3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    pts[0] = Landmark3D(0.50, 0.18, -0.03, 1.0)
    for i in [1,2,3,4,5,6,7,8,9,10]:
        pts[i] = Landmark3D(0.50, 0.19 + 0.003 * (i % 2), -0.02, 0.9)
    pts[11] = Landmark3D(0.43, 0.30, 0.00, 1.0)
    pts[12] = Landmark3D(0.57, 0.30, 0.00, 1.0)
    pts[13] = Landmark3D(0.40, 0.42, 0.02, 1.0)
    pts[14] = Landmark3D(0.60, 0.42, 0.02, 1.0)
    pts[15] = Landmark3D(0.39, 0.56, 0.05, 1.0)
    pts[16] = Landmark3D(0.61, 0.56, 0.05, 1.0)
    for i in [17, 19, 21]:
        pts[i] = Landmark3D(0.37, 0.58 + 0.01 * (i - 17), 0.06, 0.9)
    for i in [18, 20, 22]:
        pts[i] = Landmark3D(0.63, 0.58 + 0.01 * (i - 18), 0.06, 0.9)
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


def _copy_pose(pose):
    return [Landmark3D(p.x, p.y, p.z, p.visibility) for p in pose]


def _jitter(pose, sigma: float = 0.005) -> None:
    for p in pose:
        p.x += random.gauss(0.0, sigma)
        p.y += random.gauss(0.0, sigma)
        p.z += random.gauss(0.0, sigma * 0.7)


def _arm_chain(pose, side: str, shoulder=(0.0, 0.0), elbow=(0.0, 0.0), wrist=(0.0, 0.0), z_shift=0.0):
    s, e, w = (11, 13, 15) if side == "left" else (12, 14, 16)
    pose[s].x += shoulder[0]; pose[s].y += shoulder[1]
    pose[e].x += elbow[0]; pose[e].y += elbow[1]
    pose[w].x += wrist[0]; pose[w].y += wrist[1]
    pose[w].z += z_shift


def _leg_chain(pose, side: str, hip=(0.0, 0.0), knee=(0.0, 0.0), ankle=(0.0, 0.0), z_shift=0.0):
    h, k, a = (23, 25, 27) if side == "left" else (24, 26, 28)
    pose[h].x += hip[0]; pose[h].y += hip[1]
    pose[k].x += knee[0]; pose[k].y += knee[1]
    pose[a].x += ankle[0]; pose[a].y += ankle[1]
    pose[a].z += z_shift


def _neutral_variant(pose, phase: float, direction: float) -> None:
    mode = random.choice(["breathe", "one_hand_small", "both_hands_chest", "lean", "step"])
    if mode == "breathe":
        breathe = 0.01 * math.sin(2 * math.pi * phase * (1.0 + random.random()))
        for idx in [11, 12, 23, 24]:
            pose[idx].y += breathe * (1.0 if idx < 23 else 0.5)
    elif mode == "one_hand_small":
        side = random.choice(["left", "right"])
        arc = 0.03 * math.sin(2 * math.pi * phase)
        _arm_chain(pose, side, elbow=(0.01 * direction, -0.02), wrist=(0.05 * direction + arc, -0.07), z_shift=-0.01)
    elif mode == "both_hands_chest":
        _arm_chain(pose, "left", elbow=(0.01, -0.01), wrist=(0.05, -0.05), z_shift=-0.01)
        _arm_chain(pose, "right", elbow=(-0.01, -0.01), wrist=(-0.05, -0.05), z_shift=-0.01)
    elif mode == "lean":
        lean = 0.03 * math.sin(math.pi * phase)
        for idx in range(33):
            pose[idx].x += lean * direction
    else:
        lift = 0.06 * math.sin(math.pi * phase)
        side = random.choice(["left", "right"])
        _leg_chain(pose, side, knee=(0.01 * direction, -0.04 * lift), ankle=(0.04 * direction, -0.08 * lift), z_shift=-0.01)


def _make_sequence(label: str, length: int) -> np.ndarray:
    prev = None
    feats = []
    dominant_side = random.choice(["left", "right"])
    direction = random.choice([-1.0, 1.0])
    push_bias = random.choice([-1.0, 1.0])

    for t in range(length):
        phase = t / max(length - 1, 1)
        pose = _copy_pose(_base_pose())
        _jitter(pose, sigma=0.0035 + 0.002 * random.random())

        if label == "neutral":
            _neutral_variant(pose, phase, direction)

        elif label == "wave":
            side = dominant_side
            arc = math.sin(2 * math.pi * (2.2 + 0.4 * random.random()) * phase)
            _arm_chain(
                pose,
                side,
                shoulder=(0.0, -0.03),
                elbow=(0.02 * direction, -0.10),
                wrist=(0.20 * arc * direction, -0.20 - 0.05 * phase),
                z_shift=-0.02,
            )
            other = "right" if side == "left" else "left"
            _arm_chain(pose, other, elbow=(0.01, -0.01), wrist=(0.03, -0.02), z_shift=0.0)

        elif label == "push":
            drive = 1.0 / (1.0 + math.exp(-18 * (phase - 0.40)))
            center_pull = 0.10 * (1.0 - drive)
            _arm_chain(pose, "left", elbow=(0.04, -0.02), wrist=(0.16 + center_pull, -0.14), z_shift=-0.20 * drive)
            _arm_chain(pose, "right", elbow=(-0.04, -0.02), wrist=(-0.16 - center_pull, -0.14), z_shift=-0.20 * drive)
            pose[11].x += 0.02 * push_bias
            pose[12].x -= 0.02 * push_bias
            pose[23].x += 0.01 * push_bias
            pose[24].x -= 0.01 * push_bias

        elif label == "punch":
            side = dominant_side
            burst = 1.0 / (1.0 + math.exp(-28 * (phase - 0.40)))
            retract = 1.0 - max(0.0, phase - 0.78) / 0.22
            burst *= max(retract, 0.65)
            _arm_chain(
                pose,
                side,
                shoulder=(0.0, -0.01),
                elbow=(0.06 * direction, -0.04),
                wrist=(0.12 * direction + 0.20 * burst * direction, -0.02 - 0.03 * burst),
                z_shift=-0.25 * burst,
            )
            other = "right" if side == "left" else "left"
            _arm_chain(pose, other, elbow=(0.00, -0.01), wrist=(0.02 * -direction, -0.01), z_shift=0.0)
            for idx in [0, 11, 12, 23, 24]:
                pose[idx].x -= 0.02 * direction * burst

        elif label == "kick":
            side = dominant_side
            lift = math.sin(math.pi * min(1.0, phase * 1.15))
            extend = 1.0 / (1.0 + math.exp(-20 * (phase - 0.45)))
            _leg_chain(
                pose,
                side,
                knee=(0.05 * direction, -0.20 * lift),
                ankle=(0.10 * direction + 0.16 * extend * direction, -0.34 * lift),
                z_shift=-0.15 * extend,
            )
            pose[0].x -= 0.04 * direction * lift
            pose[23].x -= 0.02 * direction * lift
            pose[24].x -= 0.02 * direction * lift
            other = "right" if side == "left" else "left"
            _leg_chain(pose, other, knee=(0.0, 0.01), ankle=(0.0, 0.01), z_shift=0.0)

        elif label == "fall":
            drop = 1.0 / (1.0 + math.exp(-15 * (phase - 0.56)))
            tilt = 0.42 * drop * direction
            for idx in range(33):
                pose[idx].y += 0.42 * drop
                pose[idx].x += tilt * (pose[idx].y - 0.54)
            pose[0].y += 0.14 * drop
            pose[15].y += 0.10 * drop
            pose[16].y += 0.10 * drop
            pose[27].y -= 0.05 * drop
            pose[28].y -= 0.05 * drop
            pose[23].visibility = 0.88
            pose[24].visibility = 0.88
        else:
            raise ValueError(label)

        feats.append(build_feature_vector(pose, prev))
        prev = pose
    return np.stack(feats).astype(np.float32)


def make_dataset(n_per_class: int, seq_len: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    xs, ys = [], []
    for class_idx, label in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            xs.append(_make_sequence(label, seq_len))
            ys.append(class_idx)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)
