from __future__ import annotations

import argparse
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.features import Landmark3D, build_feature_vector


ACTION_MAP = {
    "push": "push",
    "waveHands": "wave",
    "wave_hands": "wave",
    "pushing": "push",
    "kicking": "kick",
    "punching": "punch",
    "fall": "fall",
}


def parse_utkinect_labels(label_path: Path) -> dict[str, list[tuple[str, int, int]]]:
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"(s\d{2}_e\d{2})\s+(.*?)(?=\s+s\d{2}_e\d{2}\s+|$)")
    act_pat = re.compile(r"([A-Za-z]+):\s+(\d+|NaN)\s+(\d+|NaN)")
    results: dict[str, list[tuple[str, int, int]]] = {}
    for sample_id, chunk in pattern.findall(text):
        rows = []
        for action_name, start, end in act_pat.findall(chunk):
            if start == "NaN" or end == "NaN":
                continue
            rows.append((action_name, int(start), int(end)))
        results[sample_id] = rows
    return results


def load_utkinect_joint_txt(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    if df.shape[1] < 61:
        raise ValueError(f"Unexpected shape for {file_path}: {df.shape}")
    cols = ["frame"] + [f"j{j}_{axis}" for j in range(20) for axis in ("x", "y", "z")]
    df.columns = cols[: df.shape[1]]
    return df


def _utk_to_landmarks(row: pd.Series) -> list[Landmark3D]:
    # Map the 20-kinect joints into a 33-slot list with zeros for missing MediaPipe joints.
    lms = [Landmark3D(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    mapping = {
        0: 3,   # head -> nose proxy
        4: 11,  # left shoulder
        5: 13,  # left elbow
        6: 15,  # left wrist
        8: 12,  # right shoulder
        9: 14,  # right elbow
        10: 16, # right wrist
        12: 23, # left hip
        13: 25, # left knee
        14: 27, # left ankle
        16: 24, # right hip
        17: 26, # right knee
        18: 28, # right ankle
    }
    xs = []
    ys = []
    for k_idx, mp_idx in mapping.items():
        x = float(row[f"j{k_idx}_x"])
        y = float(row[f"j{k_idx}_y"])
        z = float(row[f"j{k_idx}_z"])
        lms[mp_idx] = Landmark3D(x, y, z, 1.0)
        xs.append(x)
        ys.append(y)
    # Fill absent joints using nearest available neighbors.
    nose = lms[3]
    lms[0] = Landmark3D(nose.x, nose.y, nose.z, 1.0)
    lms[11].visibility = lms[12].visibility = lms[13].visibility = lms[14].visibility = 1.0
    lms[15].visibility = lms[16].visibility = lms[23].visibility = lms[24].visibility = 1.0
    lms[25].visibility = lms[26].visibility = lms[27].visibility = lms[28].visibility = 1.0
    # Copy wrists to hand tips / feet to toes where necessary.
    for dst in [17, 19, 21]:
        lms[dst] = Landmark3D(lms[15].x, lms[15].y, lms[15].z, lms[15].visibility)
    for dst in [18, 20, 22]:
        lms[dst] = Landmark3D(lms[16].x, lms[16].y, lms[16].z, lms[16].visibility)
    for dst in [29, 31]:
        lms[dst] = Landmark3D(lms[27].x, lms[27].y, lms[27].z, lms[27].visibility)
    for dst in [30, 32]:
        lms[dst] = Landmark3D(lms[28].x, lms[28].y, lms[28].z, lms[28].visibility)
    # Fill untracked face landmarks.
    for idx in [1, 2, 4, 5, 6, 7, 8, 9, 10]:
        lms[idx] = Landmark3D(lms[0].x, lms[0].y, lms[0].z, 0.7)
    return lms


def build_sequence_from_landmarks(landmark_sequence: list[list[Landmark3D]]) -> np.ndarray:
    features = []
    prev = None
    for current in landmark_sequence:
        features.append(build_feature_vector(current, prev))
        prev = current
    return np.stack(features).astype(np.float32)


def prepare_utkinect(joints_zip: Path, labels_txt: Path, out_dir: Path, min_frames: int = 20) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = parse_utkinect_labels(labels_txt)
    produced = 0
    with zipfile.ZipFile(joints_zip, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".txt") and not m.endswith("actionLabel.txt")]
        for member in members:
            sample_id = Path(member).stem
            if sample_id not in labels:
                continue
            with zf.open(member) as fp:
                df = pd.read_csv(fp, sep=r"\s+", header=None)
            if df.shape[1] < 61:
                continue
            df.columns = ["frame"] + [f"j{j}_{axis}" for j in range(20) for axis in ("x", "y", "z")]
            frame_lookup = {int(row.frame): row for row in df.itertuples(index=False)}
            for action_name, start, end in labels[sample_id]:
                label = ACTION_MAP.get(action_name)
                if label is None:
                    continue
                selected_rows = [frame_lookup[f] for f in sorted(frame_lookup) if start <= f <= end]
                if len(selected_rows) < min_frames:
                    continue
                lms_seq = [_utk_to_landmarks(pd.Series(r._asdict())) for r in selected_rows]
                seq = build_sequence_from_landmarks(lms_seq)
                out_path = out_dir / f"{sample_id}_{label}.npz"
                np.savez_compressed(out_path, x=seq, y=label, source="utkinect")
                produced += 1
    return produced


# Placeholder hooks for user-side extraction from segmented UT-Interaction videos and UP-Fall CSVs.
# The downloader grabs the original public archives; this script focuses on UTKinect because it is the
# lowest-friction skeleton source. The other two sources are intentionally left as extension points.


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public datasets into .npz sequence files.")
    parser.add_argument("--utkinect-joints", type=Path, default=Path("data/raw/public_datasets/joints.zip"))
    parser.add_argument("--utkinect-labels", type=Path, default=Path("data/raw/public_datasets/actionLabel.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/public_sequences"))
    args = parser.parse_args()

    count = prepare_utkinect(args.utkinect_joints, args.utkinect_labels, args.output_dir)
    print(f"Prepared {count} UTKinect sequence files into {args.output_dir}")


if __name__ == "__main__":
    main()
