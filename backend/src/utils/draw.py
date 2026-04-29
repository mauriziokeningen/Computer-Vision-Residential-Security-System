"""
Visualization primitives.

Centralizes all bounding box rendering so that the annotator process is the
single source of truth for what an annotated frame looks like. Workers no
longer draw — they only publish detection metadata.

Design notes:
    - All draw functions mutate the input frame in place and return it. The
      caller (annotator) owns the copy semantics. This avoids double allocation
      when iterating over multiple modules' detections on the same frame.
    - Colors are semantic (threat / unknown / known / fall), not per-class, so
      adding a new class to a module's THREAT_CLASSES set automatically inherits
      the right color without changes here.
    - Box thickness and font scale are scaled relative to a 720p reference, so
      1080p cameras don't end up with comically thin lines.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# BGR (OpenCV uses BGR, not RGB)
COLOR_THREAT = (0, 0, 220)       # Red    -> weapons (knife, pistol, ...)
COLOR_UNKNOWN = (0, 140, 255)    # Orange -> unknown person (warning, not yet a threat)
COLOR_KNOWN = (60, 180, 75)      # Green  -> recognized resident
COLOR_AGGRESSION = (0, 0, 220)   # Red    -> aggressive pose actions
COLOR_FALL = (0, 215, 255)       # Yellow -> fall detection
COLOR_DEFAULT = (200, 200, 200)  # Gray   -> fallback

_BASE_THICKNESS = 2
_BASE_FONT_SCALE = 0.7
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scale_for_frame(frame: np.ndarray) -> Tuple[int, float, int]:
    """Returns (box_thickness, font_scale, text_thickness) scaled to frame size."""
    h = frame.shape[0]
    ratio = max(0.6, min(1.6, h / 720.0))
    return (
        max(1, int(round(_BASE_THICKNESS * ratio))),
        _BASE_FONT_SCALE * ratio,
        max(1, int(round(2 * ratio))),
    )


def _clamp_box(x1: int, y1: int, x2: int, y2: int, frame: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    confidence: float,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """
    Draws a single bounding box plus a "label conf" tag (e.g. "pistol 0.87")
    in the YOLO/Roboflow visual style. Mutates `frame` in place.
    """
    box_thick, font_scale, text_thick = _scale_for_frame(frame)
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)

    text = f"{label} {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, text_thick)

    pad = 4
    label_top = y1 - th - baseline - pad
    if label_top < 0:
        label_top = y1 + pad
        text_y = label_top + th
    else:
        text_y = y1 - pad - baseline // 2

    label_left = x1
    label_right = min(frame.shape[1] - 1, x1 + tw + 2 * pad)
    label_bottom = label_top + th + baseline + pad

    cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), color, thickness=-1)
    cv2.putText(
        frame, text,
        (label_left + pad, text_y),
        _FONT, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA,
    )
    return frame


def draw_module_detections(
    frame: np.ndarray,
    module: str,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Dispatches to the right per-module drawing logic.

    Mutates `frame` in place and returns it, so the annotator can chain
    multiple modules' detections on the same buffer without intermediate copies.

    Module contracts:
        weapons: { "class": str, "confidence": float, "bbox": [x1, y1, x2, y2] }
        face:    { "name": str,  "confidence": float, "bbox": {"x":, "y":, "w":, "h":} }
        pose:    { "action": str, "confidence": float, "bbox": [x1, y1, x2, y2] }   (future)
    """
    if module == "weapons":
        for d in detections:
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw_detection(
                frame, x1, y1, x2, y2,
                label=d.get("class", "object"),
                confidence=float(d.get("confidence", 0.0)),
                color=COLOR_THREAT,
            )

    elif module == "face":
        for d in detections:
            bbox = d.get("bbox") or {}
            try:
                x = int(bbox["x"]); y = int(bbox["y"])
                w = int(bbox["w"]); h = int(bbox["h"])
            except (KeyError, TypeError, ValueError):
                continue
            name = d.get("name", "unknown_person")
            confidence = float(d.get("confidence", 0.0))
            color = COLOR_UNKNOWN if name == "unknown_person" else COLOR_KNOWN
            label = "unknown" if name == "unknown_person" else name
            draw_detection(frame, x, y, x + w, y + h, label, confidence, color)

    elif module == "pose":
        # Reserved for the pose worker once it's wired in.
        for d in detections:
            bbox = d.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            action = d.get("action", "action")
            aggressive = {"punch", "kick", "push", "fight", "struggle",
                          "golpe", "patada", "empujon", "pelea", "forcejeo"}
            fall = {"fall", "caida"}
            color = (
                COLOR_AGGRESSION if action.lower() in aggressive
                else COLOR_FALL if action.lower() in fall
                else COLOR_DEFAULT
            )
            draw_detection(
                frame, x1, y1, x2, y2,
                label=action,
                confidence=float(d.get("confidence", 0.0)),
                color=color,
            )

    return frame

