from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import torch

from src.config import CONFIG
from src.features import POSE_CONNECTIONS, build_feature_vector, mp_landmarks_to_struct
from src.io_utils import load_checkpoint


def resolve_mediapipe_pose():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "mediapipe is not installed. Run: pip install mediapipe"
        ) from exc

    # Legacy style used by a lot of older examples.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose

    # Fallback for newer package layouts where solutions are not re-exported
    # at the top-level module.
    try:
        from mediapipe.python.solutions import pose as mp_pose  # type: ignore
        return mp_pose
    except Exception as exc:
        raise SystemExit(
            "Your installed mediapipe package does not expose the legacy Pose API used by this script.\n"
            "Fix it with one of these options:\n"
            "1) Keep this patched script and install a current mediapipe wheel.\n"
            "2) Or reinstall a legacy-compatible build, for example: pip install mediapipe==0.10.14\n"
            "3) If a local file/folder named 'mediapipe' exists in your project, rename it."
        ) from exc


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


def majority_vote(history: Deque[str]) -> str:
    if not history:
        return "neutral"
    return Counter(history).most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam action recognition using MediaPipe + TemporalActionNet.")
    parser.add_argument("--checkpoint", type=str, default=str(CONFIG.checkpoint_path))
    parser.add_argument("--camera", type=int, default=CONFIG.webcam_index)
    parser.add_argument("--min-confidence", type=float, default=CONFIG.min_confidence)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, bundle = load_checkpoint(Path(args.checkpoint), device=device)
    model.to(device)
    class_names = list(bundle["class_names"])
    seq_len = int(bundle["sequence_length"])

    mp_pose = resolve_mediapipe_pose()
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
        smooth_landmarks=True,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.frame_height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")

    feature_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)
    prev_struct = None
    prob_ema = None
    label_history: Deque[str] = deque(maxlen=CONFIG.vote_window)

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

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            draw_pose(frame, landmarks)
            struct = mp_landmarks_to_struct(landmarks)
            feat = build_feature_vector(struct, prev_struct)
            prev_struct = struct
            feature_buffer.append(feat)

            if len(feature_buffer) == seq_len:
                x = np.stack(feature_buffer).astype(np.float32)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(xt)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                if prob_ema is None:
                    prob_ema = probs.copy()
                else:
                    prob_ema = CONFIG.smoothing_alpha * probs + (1.0 - CONFIG.smoothing_alpha) * prob_ema
                idx = int(np.argmax(prob_ema))
                conf = float(prob_ema[idx])
                raw_label = class_names[idx]
                label_history.append(raw_label if conf >= args.min_confidence else "neutral")
                label = majority_vote(label_history)
                probs = prob_ema
        else:
            prev_struct = None
            feature_buffer.clear()
            prob_ema = None
            label_history.clear()

        cv2.rectangle(frame, (18, 18), (430, 260), (20, 20, 20), -1)
        cv2.putText(frame, f"Action: {label.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {conf:.2f}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2)
        for i, name in enumerate(class_names):
            y = 110 + i * 24
            pct = float(probs[i])
            cv2.putText(frame, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.rectangle(frame, (120, y - 11), (320, y + 2), (55, 55, 55), -1)
            cv2.rectangle(frame, (120, y - 11), (120 + int(200 * pct), y + 2), (0, 210, 255), -1)
            cv2.putText(frame, f"{pct * 100:5.1f}%", (332, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        cv2.putText(frame, "ESC to exit", (30, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.imshow("Temporal Action Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
