from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Sequence

import cv2
import numpy as np
import torch

from src.config import CONFIG
from src.features import POSE_CONNECTIONS, build_feature_vector, mp_landmarks_to_struct
from src.io_utils import load_checkpoint


# --------- Drawing ---------
COLOR_MAP = {
    "neutral": (180, 180, 180),
    "punch": (30, 90, 255),
    "kick": (0, 180, 255),
    "push": (255, 220, 0),
    "wave": (0, 220, 120),
    "fall": (0, 0, 255),
}


def draw_pose(frame: np.ndarray, landmarks) -> None:
    h, w = frame.shape[:2]
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
        p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
        cv2.line(frame, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)
    for idx, lm in enumerate(landmarks):
        p = (int(lm.x * w), int(lm.y * h))
        radius = 5 if idx in {15, 16, 27, 28, 0} else 3
        cv2.circle(frame, p, radius, (40, 255, 120), -1, cv2.LINE_AA)


# --------- Temporal helpers ---------
def majority_vote(history: Deque[str]) -> str:
    if not history:
        return "neutral"
    return Counter(history).most_common(1)[0][0]


def _shoulder_width(pose) -> float:
    return max(np.hypot(pose[11].x - pose[12].x, pose[11].y - pose[12].y), 1e-4)


def _hip_center(pose) -> tuple[float, float]:
    return ((pose[23].x + pose[24].x) * 0.5, (pose[23].y + pose[24].y) * 0.5)


def _elbow_angle(pose, side: str) -> float:
    if side == "left":
        s, e, w = 11, 13, 15
    else:
        s, e, w = 12, 14, 16
    a = np.array([pose[s].x - pose[e].x, pose[s].y - pose[e].y], dtype=np.float32)
    b = np.array([pose[w].x - pose[e].x, pose[w].y - pose[e].y], dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    cosang = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.arccos(cosang))


def _knee_angle(pose, side: str) -> float:
    if side == "left":
        h, k, a = 23, 25, 27
    else:
        h, k, a = 24, 26, 28
    u = np.array([pose[h].x - pose[k].x, pose[h].y - pose[k].y], dtype=np.float32)
    v = np.array([pose[a].x - pose[k].x, pose[a].y - pose[k].y], dtype=np.float32)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-6
    cosang = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(np.arccos(cosang))


def _sign_changes(values: Sequence[float], eps: float = 1e-3) -> int:
    if len(values) < 4:
        return 0
    diffs = np.diff(np.asarray(values, dtype=np.float32))
    diffs[np.abs(diffs) < eps] = 0.0
    signs = np.sign(diffs)
    cleaned = [s for s in signs if s != 0]
    return sum(1 for i in range(1, len(cleaned)) if cleaned[i] != cleaned[i - 1])


def compute_motion_stats(seq: Sequence) -> dict[str, float]:
    if len(seq) < 6:
        return {"motion": 0.0}

    widths = np.asarray([_shoulder_width(p) for p in seq], dtype=np.float32)
    scale = float(max(np.median(widths), 1e-4))

    def xs(idx: int) -> np.ndarray:
        return np.asarray([p[idx].x for p in seq], dtype=np.float32)

    def ys(idx: int) -> np.ndarray:
        return np.asarray([p[idx].y for p in seq], dtype=np.float32)

    def speeds(idx: int) -> np.ndarray:
        arr_x = xs(idx)
        arr_y = ys(idx)
        return np.sqrt(np.diff(arr_x) ** 2 + np.diff(arr_y) ** 2) / scale

    nose_y = ys(0)
    lsho_y, rsho_y = ys(11), ys(12)
    lw_x, rw_x = xs(15), xs(16)
    lw_y, rw_y = ys(15), ys(16)
    la_y, ra_y = ys(27), ys(28)
    lhip_y, rhip_y = ys(23), ys(24)
    centers_x = np.asarray([_hip_center(p)[0] for p in seq], dtype=np.float32)
    hip_y = (lhip_y + rhip_y) * 0.5
    shoulder_y = (lsho_y + rsho_y) * 0.5

    lws = speeds(15)
    rws = speeds(16)
    las = speeds(27)
    ras = speeds(28)
    motion = float(np.mean(np.concatenate([lws, rws, las, ras])))

    lw_above = float(np.mean(lw_y < shoulder_y - 0.02))
    rw_above = float(np.mean(rw_y < shoulder_y - 0.02))

    lw_center = np.abs(lw_x - centers_x) / scale
    rw_center = np.abs(rw_x - centers_x) / scale

    cur = seq[-1]
    stats = {
        "motion": motion,
        "lws_peak": float(lws.max(initial=0.0)),
        "rws_peak": float(rws.max(initial=0.0)),
        "las_peak": float(las.max(initial=0.0)),
        "ras_peak": float(ras.max(initial=0.0)),
        "lw_above_ratio": lw_above,
        "rw_above_ratio": rw_above,
        "lw_x_amp": float((lw_x.max() - lw_x.min()) / scale),
        "rw_x_amp": float((rw_x.max() - rw_x.min()) / scale),
        "lw_sign_changes": float(_sign_changes(lw_x / scale, eps=0.01)),
        "rw_sign_changes": float(_sign_changes(rw_x / scale, eps=0.01)),
        "lw_center_now": float(abs(cur[15].x - _hip_center(cur)[0]) / scale),
        "rw_center_now": float(abs(cur[16].x - _hip_center(cur)[0]) / scale),
        "lw_raise_now": float((cur[11].y - cur[15].y) / scale),
        "rw_raise_now": float((cur[12].y - cur[16].y) / scale),
        "la_raise_now": float((cur[23].y - cur[27].y) / scale),
        "ra_raise_now": float((cur[24].y - cur[28].y) / scale),
        "l_elbow_angle": float(_elbow_angle(cur, "left")),
        "r_elbow_angle": float(_elbow_angle(cur, "right")),
        "l_knee_angle": float(_knee_angle(cur, "left")),
        "r_knee_angle": float(_knee_angle(cur, "right")),
        "nose_drop": float((nose_y[-1] - np.min(nose_y[:-1])) / scale),
        "hip_drop": float((hip_y[-1] - np.min(hip_y[:-1])) / scale),
        "wrist_symmetry": float(abs(np.mean(lws[-4:]) - np.mean(rws[-4:])) if len(lws) >= 4 and len(rws) >= 4 else 0.0),
        "both_hands_center": float(np.mean((lw_center < 0.75) & (rw_center < 0.75))),
    }
    return stats


def class_gate(label: str, stats: dict[str, float]) -> tuple[bool, str]:
    motion = stats.get("motion", 0.0)
    if motion < 0.008 and label != "fall":
        return False, "low_motion"

    if label == "wave":
        left_ok = (
            stats.get("lw_above_ratio", 0.0) > 0.50
            and stats.get("lw_x_amp", 0.0) > 0.45
            and stats.get("lw_sign_changes", 0.0) >= 2
            and stats.get("lws_peak", 0.0) > 0.020
        )
        right_ok = (
            stats.get("rw_above_ratio", 0.0) > 0.50
            and stats.get("rw_x_amp", 0.0) > 0.45
            and stats.get("rw_sign_changes", 0.0) >= 2
            and stats.get("rws_peak", 0.0) > 0.020
        )
        return (left_ok or right_ok), "needs_real_oscillation"

    if label == "push":
        both_center = stats.get("both_hands_center", 0.0) > 0.50
        both_raised = stats.get("lw_raise_now", 0.0) > 0.12 and stats.get("rw_raise_now", 0.0) > 0.12
        elbows = stats.get("l_elbow_angle", 0.0) > 2.15 and stats.get("r_elbow_angle", 0.0) > 2.15
        both_move = stats.get("lws_peak", 0.0) > 0.018 and stats.get("rws_peak", 0.0) > 0.018
        symmetric = stats.get("wrist_symmetry", 0.0) < 0.012
        return (both_center and both_raised and elbows and both_move and symmetric), "needs_two_hands"

    if label == "punch":
        left = (
            stats.get("lws_peak", 0.0) > 0.030
            and stats.get("l_elbow_angle", 0.0) > 2.10
            and stats.get("lws_peak", 0.0) > stats.get("rws_peak", 0.0) * 1.20
        )
        right = (
            stats.get("rws_peak", 0.0) > 0.030
            and stats.get("r_elbow_angle", 0.0) > 2.10
            and stats.get("rws_peak", 0.0) > stats.get("lws_peak", 0.0) * 1.20
        )
        return (left or right), "needs_single_arm_burst"

    if label == "kick":
        left = stats.get("las_peak", 0.0) > 0.032 and stats.get("la_raise_now", 0.0) > 0.18 and stats.get("l_knee_angle", 0.0) > 1.90
        right = stats.get("ras_peak", 0.0) > 0.032 and stats.get("ra_raise_now", 0.0) > 0.18 and stats.get("r_knee_angle", 0.0) > 1.90
        return (left or right), "needs_leg_burst"

    if label == "fall":
        return (
            stats.get("nose_drop", 0.0) > 0.45 and stats.get("hip_drop", 0.0) > 0.25
        ), "needs_body_drop"

    return True, "ok"


def calibrate_probs(class_names: list[str], probs: np.ndarray) -> np.ndarray:
    # Penalize the two over-predicted classes and slightly lift punch/kick.
    calibrated = probs.copy()
    weights = {
        "neutral": 1.05,
        "punch": 1.22,
        "kick": 1.22,
        "push": 0.84,
        "wave": 0.80,
        "fall": 1.00,
    }
    for i, name in enumerate(class_names):
        calibrated[i] *= weights.get(name, 1.0)
    total = float(np.sum(calibrated))
    if total > 0:
        calibrated /= total
    return calibrated


def pick_label(class_names: list[str], probs: np.ndarray, stats: dict[str, float], min_confidence: float) -> tuple[str, float, str]:
    thresholds = {
        "neutral": 0.50,
        "punch": max(0.42, min_confidence - 0.08),
        "kick": max(0.42, min_confidence - 0.08),
        "push": max(0.68, min_confidence + 0.05),
        "wave": max(0.72, min_confidence + 0.08),
        "fall": max(0.60, min_confidence),
    }

    order = np.argsort(probs)[::-1]
    reasons: list[str] = []
    for idx in order:
        label = class_names[int(idx)]
        conf = float(probs[int(idx)])
        if label == "neutral":
            continue
        if conf < thresholds.get(label, min_confidence):
            reasons.append(f"{label}:low_prob")
            continue
        ok, reason = class_gate(label, stats)
        if ok:
            return label, conf, "accepted"
        reasons.append(f"{label}:{reason}")

    neutral_idx = class_names.index("neutral") if "neutral" in class_names else int(order[-1])
    return "neutral", float(probs[neutral_idx]), ",".join(reasons[:3]) if reasons else "neutral"


def load_mp_pose():
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return mp.solutions.pose
        from mediapipe.python.solutions import pose as mp_pose
        return mp_pose
    except ImportError as exc:
        raise SystemExit("mediapipe is not installed. Run: pip install mediapipe") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam action recognition using MediaPipe + TemporalActionNet (improved inference).")
    parser.add_argument("--checkpoint", type=str, default=str(CONFIG.checkpoint_path))
    parser.add_argument("--camera", type=int, default=CONFIG.webcam_index)
    parser.add_argument("--min-confidence", type=float, default=0.62)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, bundle = load_checkpoint(Path(args.checkpoint), device=device)
    model.to(device)
    class_names = list(bundle["class_names"])
    seq_len = int(bundle["sequence_length"])

    mp_pose = load_mp_pose()
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
        smooth_landmarks=True,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")

    feature_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)
    pose_buffer: Deque[Sequence] = deque(maxlen=14)
    prev_struct = None
    prob_ema = None
    label_history: Deque[str] = deque(maxlen=5)
    stable_label = "neutral"
    stable_frames = 0
    debug_reason = "boot"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        label = "neutral"
        conf = 0.0
        probs = np.zeros(len(class_names), dtype=np.float32)
        stats = {"motion": 0.0}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            draw_pose(frame, landmarks)
            struct = mp_landmarks_to_struct(landmarks)
            feat = build_feature_vector(struct, prev_struct)
            prev_struct = struct
            feature_buffer.append(feat)
            pose_buffer.append(struct)

            if len(feature_buffer) == seq_len:
                x = np.stack(feature_buffer).astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xt)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                probs = calibrate_probs(class_names, probs)
                if prob_ema is None:
                    prob_ema = probs.copy()
                else:
                    prob_ema = 0.28 * probs + 0.72 * prob_ema
                probs = prob_ema
                stats = compute_motion_stats(list(pose_buffer))
                label, conf, debug_reason = pick_label(class_names, probs, stats, args.min_confidence)

                # Short temporal memory so punch/kick survive a few frames.
                label_history.append(label)
                voted = majority_vote(label_history)
                if voted != "neutral":
                    stable_label = voted
                    stable_frames = 5
                elif stable_frames > 0 and stable_label in {"punch", "kick", "push", "wave", "fall"}:
                    label = stable_label
                    stable_frames -= 1
                else:
                    stable_label = "neutral"
                    stable_frames = 0

                if stable_frames > 0 and label == "neutral" and stable_label != "neutral":
                    label = stable_label
                    conf = max(conf, 0.45)
                elif voted != "neutral":
                    label = voted
        else:
            prev_struct = None
            feature_buffer.clear()
            pose_buffer.clear()
            prob_ema = None
            label_history.clear()
            stable_label = "neutral"
            stable_frames = 0
            debug_reason = "no_pose"

        color = COLOR_MAP.get(label, (220, 220, 220))
        cv2.rectangle(frame, (18, 18), (470, 190), (20, 20, 20), -1)
        cv2.putText(frame, f"Action: {label.upper()}", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.86, color, 2)
        cv2.putText(frame, f"Confidence: {conf:.2f}", (30, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 230, 255), 2)
        cv2.putText(frame, f"Motion: {stats.get('motion', 0.0):.3f}", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
        cv2.putText(frame, f"Gate: {debug_reason[:42]}", (30, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 200, 230), 1)

        start_y = 152
        for i, name in enumerate(class_names):
            y = start_y + i * 22
            pct = float(probs[i]) if i < len(probs) else 0.0
            c = COLOR_MAP.get(name, (0, 210, 255))
            cv2.putText(frame, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
            cv2.rectangle(frame, (118, y - 10), (338, y + 2), (55, 55, 55), -1)
            cv2.rectangle(frame, (118, y - 10), (118 + int(220 * pct), y + 2), c, -1)
            cv2.putText(frame, f"{pct * 100:5.1f}%", (350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

        cv2.putText(frame, "ESC to exit", (30, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.imshow("Temporal Action Recognition - Improved", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
